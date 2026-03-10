"""
LangMemory — Memory that behaves like a system, not a log.

AGE Memory turns recall into a tiered runtime built from queues, indexes,
graphs, and promotion logic. Fast. Inspectable. Durable.

Usage:
    from langmemory import LangMemory

    mc = LangMemory()
    mc.insert("User prefers concise responses")
    results = mc.retrieve("What does the user prefer?")
    for r in results:
        print(r.node.content, r.score)
"""
from __future__ import annotations

import asyncio
import time
from typing import List, Optional, Tuple

from langmemory.algorithms.decay import DecayWorker
from langmemory.algorithms.insert import InsertPipeline
from langmemory.algorithms.retrieve import RetrievalResult, RetrievePipeline
from langmemory.core.config import ChainConfig
from langmemory.core.node import MemoryNode, MemoryType
from langmemory.integrity.verifier import ChainVerifier, IntegrityReport
from langmemory.scoring.embedder import Embedder, make_embedder
from langmemory.scoring.importance import ImportanceScorer
from langmemory.structures.bloom_filter import BloomFilter
from langmemory.structures.concept_trie import ConceptTrie
from langmemory.structures.importance_heap import ImportanceHeap
from langmemory.structures.lsm_tiers import LSMTierManager
from langmemory.structures.merkle_tree import MerkleTree
from langmemory.structures.skip_list import TemporalSkipList


class LangMemory:
    """
    Groundbreaking LLM memory system built on core DSA fundamentals.

    Data structures used:
      - Bloom Filter:   O(1) deduplication check before insert
      - Skip List:      O(log n) temporal index for range queries
      - Merkle Tree:    Cryptographic integrity chain for all memories
      - LSM Tiers:      Hot/Warm/Cold storage inspired by LevelDB
      - Min-Heap:       O(log n) importance-weighted retrieval
      - Trie:           O(k) hierarchical concept indexing

    All operations have documented worst-case complexity.
    Provably correct via Merkle chain integrity.
    """

    def __init__(self, config: Optional[ChainConfig] = None) -> None:
        self.config = config or ChainConfig()
        self._build_structures()
        self._build_pipelines()
        self._decay_task: Optional[asyncio.Task] = None

    # ─── Public API ───────────────────────────────────────────────────────────

    def insert(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        concepts: Optional[List[str]] = None,
        base_importance: Optional[float] = None,
        decay_factor: Optional[float] = None,
        source_session_id: str = "",
        tags: Optional[dict] = None,
        parent_id: Optional[str] = None,
    ) -> MemoryNode:
        """
        Insert a memory into the chain.

        Deduplication, embedding, Merkle chaining, and all index updates
        happen automatically.

        Returns the MemoryNode (may be an existing node if duplicate detected).

        Complexity: O(log n) amortized
        """
        return self._insert.execute(
            content=content,
            memory_type=memory_type,
            concepts=concepts,
            base_importance=base_importance,
            decay_factor=decay_factor,
            source_session_id=source_session_id,
            tags=tags,
            parent_id=parent_id,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        time_range: Optional[Tuple[float, float]] = None,
        concept_filter: Optional[str] = None,
        verify_integrity: bool = False,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most relevant memories for a query.

        Uses three-path fan-out (semantic + temporal + concept) with
        importance-weighted score fusion.

        Args:
            query:            Natural language query or context
            top_k:            Number of results (default: config.default_top_k)
            time_range:       (start_unix_ts, end_unix_ts) for temporal filter
            concept_filter:   Concept path prefix, e.g. "programming/python"
            verify_integrity: Verify Merkle chain for each result

        Returns list of RetrievalResult sorted by score descending.

        Complexity: O(m log m) where m = candidate set size (m << n)
        """
        k = top_k or self.config.default_top_k
        return self._retrieve.execute(
            query=query,
            top_k=k,
            time_range=time_range,
            concept_filter=concept_filter,
            verify_integrity=verify_integrity,
        )

    def get_context_window(
        self,
        query: str,
        token_budget: int = 2000,
        top_k: Optional[int] = None,
        chars_per_token: float = 4.0,
    ) -> str:
        """
        Retrieve memories and format them into a context string
        fitting within a token budget.

        Suitable for direct injection into an LLM system prompt.
        """
        char_budget = int(token_budget * chars_per_token)
        results = self.retrieve(query, top_k=top_k or self.config.default_top_k)

        parts: List[str] = ["[MEMORIES]"]
        used = len(parts[0])

        for r in results:
            line = f"- [{r.node.memory_type.value}] {r.node.content} (importance={r.node.current_importance:.2f})"
            if used + len(line) > char_budget:
                break
            parts.append(line)
            used += len(line)

        parts.append("[/MEMORIES]")
        return "\n".join(parts)

    def verify_integrity(self) -> IntegrityReport:
        """
        Verify the cryptographic integrity of the entire memory chain.
        Returns an IntegrityReport detailing any corrupted nodes.

        Complexity: O(n)
        """
        return self._verifier.verify_full()

    def run_decay(self) -> None:
        """
        Run one decay cycle synchronously.
        Updates importance scores and demotes stale nodes to lower tiers.
        Normally runs automatically as a background task; call this for testing.
        """
        self._decay_worker.run_once()

    async def start_background_decay(self) -> None:
        """Start the async decay worker. Call from an async context."""
        if self._decay_task is None or self._decay_task.done():
            self._decay_task = asyncio.create_task(self._decay_worker.run())

    def stop_background_decay(self) -> None:
        """Stop the background decay worker."""
        self._decay_worker.stop()
        if self._decay_task:
            self._decay_task.cancel()

    # ─── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a snapshot of current memory system state."""
        return {
            "hot_count": self._tiers.hot_size(),
            "warm_count": self._tiers.warm_size(),
            "cold_count": self._tiers.cold_size(),
            "total_indexed": len(self._heap),
            "merkle_nodes": len(self._merkle),
            "bloom_count": len(self._bloom),
            "bloom_fp_rate": round(self._bloom.false_positive_rate(), 4),
            "root_hash": self._merkle.root_hash().hex(),
            "skip_list_size": len(self._skip),
        }

    # ─── Direct structure access (advanced use) ───────────────────────────────

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def merkle(self) -> MerkleTree:
        return self._merkle

    @property
    def verifier(self) -> ChainVerifier:
        return self._verifier

    # ─── Internal wiring ──────────────────────────────────────────────────────

    def _build_structures(self) -> None:
        cfg = self.config

        self._bloom = BloomFilter(
            capacity=cfg.bloom_capacity,
            error_rate=cfg.bloom_false_positive_rate,
        )
        self._skip = TemporalSkipList(
            max_level=cfg.skip_list_max_level,
            probability=cfg.skip_list_probability,
        )
        self._merkle = MerkleTree()
        self._heap = ImportanceHeap()
        self._trie = ConceptTrie()
        self._tiers = LSMTierManager(hot_limit=cfg.hot_tier_limit)
        self._embedder = make_embedder(cfg.embedder_backend, dim=cfg.embedding_dim)
        self._scorer = ImportanceScorer()

    def _build_pipelines(self) -> None:
        cfg = self.config

        self._insert = InsertPipeline(
            bloom=self._bloom,
            merkle=self._merkle,
            skip_list=self._skip,
            trie=self._trie,
            heap=self._heap,
            tiers=self._tiers,
            embedder=self._embedder,
            importance_scorer=self._scorer,
            hot_tier_limit=cfg.hot_tier_limit,
            dedup_similarity_threshold=cfg.dedup_similarity_threshold,
        )
        self._retrieve = RetrievePipeline(
            tiers=self._tiers,
            merkle=self._merkle,
            skip_list=self._skip,
            trie=self._trie,
            heap=self._heap,
            embedder=self._embedder,
            semantic_weight=cfg.semantic_weight,
            importance_weight=cfg.importance_weight,
            recency_weight=cfg.recency_weight,
            candidate_multiplier=cfg.candidate_multiplier,
        )
        self._verifier = ChainVerifier(self._merkle)
        self._decay_worker = DecayWorker(
            tiers=self._tiers,
            heap=self._heap,
            scorer=self._scorer,
            interval_seconds=cfg.decay_cycle_interval_seconds,
            hot_threshold=cfg.hot_threshold,
            warm_threshold=cfg.warm_threshold,
            prune_threshold=cfg.prune_threshold,
        )
