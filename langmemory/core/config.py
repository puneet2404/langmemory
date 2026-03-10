"""
ChainConfig - all tuning parameters for LangMemory in one place.

Every numeric threshold, weight, and limit lives here.
Pass a custom ChainConfig to LangMemory() to override defaults.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChainConfig:
    # --- LSM Tier thresholds ---
    hot_tier_limit: int = 1_000          # Max nodes in RAM (HOT tier)
    hot_threshold: float = 0.3           # importance below this -> demote to WARM
    warm_threshold: float = 0.05         # importance below this -> demote to COLD
    prune_threshold: float = 0.001       # importance below this -> consolidate/archive

    # --- Bloom filter ---
    bloom_capacity: int = 100_000        # Expected number of unique memories
    bloom_false_positive_rate: float = 0.01

    # --- Retrieval score fusion weights (must sum to 1.0) ---
    semantic_weight: float = 0.5         # alpha: embedding similarity
    importance_weight: float = 0.3       # beta: node importance score
    recency_weight: float = 0.2          # gamma: time-based recency bonus

    # --- Skip list ---
    skip_list_max_level: int = 16
    skip_list_probability: float = 0.5

    # --- Decay ---
    default_decay_factor: float = 0.01       # Episodic memories
    semantic_decay_factor: float = 0.001     # Semantic memories decay slowly
    working_decay_factor: float = 0.5        # Working memory decays fast
    decay_cycle_interval_seconds: int = 3600  # Run decay every hour

    # --- Consolidation ---
    min_consolidation_cluster_size: int = 3
    consolidation_concept_overlap_threshold: float = 0.7
    consolidation_interval_hours: float = 6.0

    # --- Similarity deduplication ---
    dedup_similarity_threshold: float = 0.95  # cosine similarity for dedup

    # --- Embedder ---
    embedder_backend: str = "stub"  # "openai" | "sentence-transformers" | "stub"
    embedding_dim: int = 1536
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Retrieval ---
    default_top_k: int = 10
    candidate_multiplier: int = 3  # Fetch top_k * multiplier before fusion

    def __post_init__(self) -> None:
        total = self.semantic_weight + self.importance_weight + self.recency_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Retrieval weights must sum to 1.0, got {total:.4f}. "
                f"Adjust semantic_weight + importance_weight + recency_weight."
            )
