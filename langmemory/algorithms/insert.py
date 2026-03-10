"""
Insert Pipeline - the write path for LangMemory.

Stages:
  1. Deduplication (Bloom Filter)     O(1)
  2. Embedding                        O(model latency)
  3. Node creation                    O(1)
  4. Merkle integration               O(log n)
  5. Index registration               O(log n) x3
  6. HOT tier storage                 O(1)
  7. Tier compaction (if needed)      O(k log k) where k = evict count
"""
from __future__ import annotations

from typing import List, Optional

from langmemory.core.node import MemoryNode, MemoryTier, MemoryType
from langmemory.scoring.importance import ImportanceScorer
from langmemory.structures.bloom_filter import BloomFilter
from langmemory.structures.concept_trie import ConceptTrie
from langmemory.structures.importance_heap import ImportanceHeap
from langmemory.structures.lsm_tiers import LSMTierManager
from langmemory.structures.merkle_tree import MerkleTree
from langmemory.structures.skip_list import TemporalSkipList


def _extract_concepts(content: str, memory_type: MemoryType) -> List[str]:
    """
    Simple heuristic concept extractor.
    In production, replace with an LLM-based or NLP extractor.

    Returns paths like ["general", "general/memory_type_value"].
    """
    concepts = [f"general/{memory_type.value}"]
    # Keyword-based concept hints (extend with spacy/NLP in production)
    keywords = {
        "python": "programming/python",
        "javascript": "programming/javascript",
        "typescript": "programming/typescript",
        "rust": "programming/rust",
        "database": "data/database",
        "api": "engineering/api",
        "security": "engineering/security",
        "ml": "ai/ml",
        "llm": "ai/llm",
        "memory": "ai/memory",
        "user": "context/user",
        "project": "context/project",
    }
    lower = content.lower()
    for kw, concept in keywords.items():
        if kw in lower:
            concepts.append(concept)
    return list(set(concepts))


class InsertPipeline:
    """
    Orchestrates the seven-stage insert flow.

    All stage references are injected so the pipeline is testable
    with stub implementations.
    """

    def __init__(
        self,
        bloom: BloomFilter,
        merkle: MerkleTree,
        skip_list: TemporalSkipList,
        trie: ConceptTrie,
        heap: ImportanceHeap,
        tiers: LSMTierManager,
        embedder,
        importance_scorer: ImportanceScorer,
        hot_tier_limit: int = 1_000,
        dedup_similarity_threshold: float = 0.95,
    ) -> None:
        self._bloom = bloom
        self._merkle = merkle
        self._skip = skip_list
        self._trie = trie
        self._heap = heap
        self._tiers = tiers
        self._embedder = embedder
        self._scorer = importance_scorer
        self._hot_limit = hot_tier_limit
        self._dedup_threshold = dedup_similarity_threshold

    def execute(
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
        Execute the full insert pipeline.
        Returns existing node if duplicate detected, new node otherwise.
        """
        # --- Stage 1: Deduplication ---
        candidate_key = MemoryNode._normalize_content(content).encode()
        import hashlib
        bloom_key = hashlib.sha256(candidate_key).digest()

        if self._bloom.might_contain(bloom_key):
            existing = self._find_duplicate(content, bloom_key)
            if existing is not None:
                existing.touch()
                self._scorer.update(existing)
                self._heap.update_priority(existing)
                return existing

        # --- Stage 2: Embedding ---
        embedding = self._embedder.encode(content)

        # --- Stage 3: Node creation ---
        _base_importance = base_importance or self._scorer.initial_base_importance(
            content, memory_type
        )
        decay = decay_factor or self._default_decay(memory_type)
        node_concepts = concepts or _extract_concepts(content, memory_type)

        node = MemoryNode(
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            concepts=node_concepts,
            base_importance=_base_importance,
            decay_factor=decay,
            source_session_id=source_session_id,
            tags=tags or {},
        )
        node.current_importance = _base_importance

        # --- Stage 4: Merkle integration ---
        self._merkle.insert(node, parent_id=parent_id)

        # --- Stage 5: Index registration ---
        self._skip.insert(node.created_at, node.node_id)
        self._trie.insert(node.concepts, node.node_id)
        self._heap.push(node)
        self._bloom.add(bloom_key)

        # --- Stage 6: HOT tier storage ---
        self._tiers.put_hot(node)

        # --- Stage 7: Compaction if needed ---
        if self._tiers.hot_size() > self._hot_limit:
            self._tiers.compact_to_warm()

        return node

    def _find_duplicate(self, content: str, bloom_key: bytes) -> Optional[MemoryNode]:
        """
        Check if an existing memory is semantically equivalent.
        Called only when bloom filter returns a positive hit.
        """
        from langmemory.scoring.embedder import cosine_similarity
        query_vec = self._embedder.encode(content)

        # Check recent HOT nodes for similarity (bounded search)
        for node in self._tiers.iter_hot():
            if node.embedding is None:
                continue
            sim = cosine_similarity(query_vec, node.embedding)
            if sim >= self._dedup_threshold:
                return node
        return None

    @staticmethod
    def _default_decay(memory_type: MemoryType) -> float:
        defaults = {
            MemoryType.SEMANTIC: 0.001,
            MemoryType.EPISODIC: 0.01,
            MemoryType.PROCEDURAL: 0.005,
            MemoryType.WORKING: 0.5,
        }
        return defaults.get(memory_type, 0.01)
