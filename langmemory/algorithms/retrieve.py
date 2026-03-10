"""
Retrieval Pipeline - the read path for LangMemory.

Three-path fan-out with score fusion:
  Path A - Semantic:  embedding cosine similarity
  Path B - Temporal:  skip list range query
  Path C - Concept:   trie prefix lookup

All paths produce candidates, fusion ranks them, top-k returned.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langmemory.core.node import MemoryNode
from langmemory.scoring.embedder import Embedder, Vector, cosine_similarity
from langmemory.structures.concept_trie import ConceptTrie
from langmemory.structures.importance_heap import ImportanceHeap
from langmemory.structures.lsm_tiers import LSMTierManager
from langmemory.structures.merkle_tree import MerkleTree
from langmemory.structures.skip_list import TemporalSkipList


@dataclass
class RetrievalResult:
    node: MemoryNode
    score: float
    semantic_score: float
    importance_score: float
    recency_score: float


class RetrievePipeline:
    """
    Multi-index retrieval with configurable score fusion.
    """

    def __init__(
        self,
        tiers: LSMTierManager,
        merkle: MerkleTree,
        skip_list: TemporalSkipList,
        trie: ConceptTrie,
        heap: ImportanceHeap,
        embedder,
        semantic_weight: float = 0.5,
        importance_weight: float = 0.3,
        recency_weight: float = 0.2,
        candidate_multiplier: int = 3,
    ) -> None:
        self._tiers = tiers
        self._merkle = merkle
        self._skip = skip_list
        self._trie = trie
        self._heap = heap
        self._embedder = embedder
        self._w_sem = semantic_weight
        self._w_imp = importance_weight
        self._w_rec = recency_weight
        self._multiplier = candidate_multiplier

    def execute(
        self,
        query: str,
        top_k: int = 10,
        time_range: Optional[Tuple[float, float]] = None,
        concept_filter: Optional[str] = None,
        verify_integrity: bool = False,
    ) -> "List[RetrievalResult]":
        """
        Retrieve top-k memories relevant to query.

        Args:
            query:            Natural language query
            top_k:            Number of results to return
            time_range:       Optional (start_ts, end_ts) to restrict temporal candidates
            concept_filter:   Optional concept prefix to restrict concept candidates
            verify_integrity: If True, verify Merkle chain for each result (slower)
        """
        query_embedding = self._embedder.encode(query)
        candidate_limit = top_k * self._multiplier
        now = time.time()

        # --- Path A: Semantic (importance heap gives us "hot" important nodes) ---
        semantic_candidates = self._semantic_candidates(
            query_embedding, candidate_limit
        )

        # --- Path B: Temporal (skip list range or recent tail) ---
        temporal_candidates = self._temporal_candidates(time_range, candidate_limit)

        # --- Path C: Concept (trie prefix lookup) ---
        concept_candidates = self._concept_candidates(
            query, concept_filter, candidate_limit
        )

        # --- Merge and deduplicate candidate set ---
        all_ids = {
            *[r.node_id for r in semantic_candidates],
            *temporal_candidates,
            *concept_candidates,
        }

        # Resolve ids to nodes
        nodes: Dict[str, MemoryNode] = {}
        for nid in all_ids:
            node = self._resolve(nid)
            if node is not None:
                nodes[nid] = node

        # --- Score fusion ---
        results: List[RetrievalResult] = []
        for node in nodes.values():
            sem_score = self._compute_semantic_score(query_embedding, node)
            imp_score = node.current_importance
            rec_score = self._compute_recency_score(node, now)

            fused = (
                self._w_sem * sem_score
                + self._w_imp * imp_score
                + self._w_rec * rec_score
            )

            results.append(
                RetrievalResult(
                    node=node,
                    score=fused,
                    semantic_score=sem_score,
                    importance_score=imp_score,
                    recency_score=rec_score,
                )
            )

        # --- Sort by fused score descending ---
        results.sort(key=lambda r: r.score, reverse=True)
        top_results = results[:top_k]

        # --- Update access metadata ---
        for result in top_results:
            result.node.touch()
            # Re-register updated priority in heap
            self._heap.update_priority(result.node)

        # --- Optional integrity check ---
        if verify_integrity:
            for result in top_results:
                if not self._merkle.verify_chain(result.node.node_id):
                    result.node.tags["integrity_warning"] = "merkle_chain_broken"

        return top_results

    def _semantic_candidates(
        self, query_embedding: Vector, limit: int
    ) -> List[MemoryNode]:
        """Get importance-ordered candidates from heap, score by embedding sim."""
        candidates = self._heap.extract_top_k(limit)
        # Filter to those with embeddings, sort by cosine similarity
        scored = []
        for node in candidates:
            if node.embedding is not None:
                sim = cosine_similarity(query_embedding, node.embedding)
                scored.append((sim, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:limit]]

    def _temporal_candidates(
        self,
        time_range: Optional[Tuple[float, float]],
        limit: int,
    ) -> List[str]:
        if time_range:
            return self._skip.range_query(time_range[0], time_range[1])
        return self._skip.tail(limit)

    def _concept_candidates(
        self, query: str, concept_filter: Optional[str], limit: int
    ) -> List[str]:
        if concept_filter:
            return self._trie.prefix_search(concept_filter)[:limit]
        # Auto-extract concepts from query
        from langmemory.algorithms.insert import _extract_concepts
        from langmemory.core.node import MemoryType
        concepts = _extract_concepts(query, MemoryType.EPISODIC)
        all_ids: List[str] = []
        for concept in concepts:
            all_ids.extend(self._trie.prefix_search(concept))
        return list(set(all_ids))[:limit]

    def _resolve(self, node_id: str) -> Optional[MemoryNode]:
        return self._tiers.get(node_id)

    @staticmethod
    def _compute_semantic_score(query_emb: Vector, node: MemoryNode) -> float:
        if node.embedding is None:
            return 0.0
        return max(0.0, cosine_similarity(query_emb, node.embedding))

    @staticmethod
    def _compute_recency_score(node: MemoryNode, now: float) -> float:
        """Exponential recency: 1.0 now, 0.5 after 24 hours, 0.25 after 48 hours."""
        import math
        age_hours = (now - node.created_at) / 3600.0
        return math.exp(-math.log(2) * age_hours / 24.0)
