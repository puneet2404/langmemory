"""
Importance scoring - composite score for memory priority.

importance = recency_weight(age)
           * frequency_weight(access_count)
           * base_importance
           * (1 - decay_factor)^idle_days
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from langmemory.core.node import MemoryNode, MemoryType


@dataclass
class ImportanceWeights:
    recency: float = 0.4
    frequency: float = 0.3
    base: float = 0.3

    def __post_init__(self) -> None:
        total = self.recency + self.frequency + self.base
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ImportanceWeights must sum to 1.0, got {total}")


class ImportanceScorer:
    """
    Compute and update current_importance for a MemoryNode.

    The score is a weighted combination of:
      - Recency:   How recently was this created? (exponential decay)
      - Frequency: How often accessed? (logarithmic growth)
      - Base:      The original importance set at creation time

    All components are in [0, 1]. Final score is in [0, 1].
    """

    def __init__(self, weights: ImportanceWeights | None = None) -> None:
        self.weights = weights or ImportanceWeights()

    def score(self, node: MemoryNode, now: float | None = None) -> float:
        """Compute importance score for a node at the given timestamp."""
        now = now or time.time()
        recency = self._recency_score(node, now)
        frequency = self._frequency_score(node)
        base = node.base_importance

        raw = (
            self.weights.recency * recency
            + self.weights.frequency * frequency
            + self.weights.base * base
        )

        # Apply time-based decay on idle time
        idle_days = (now - node.last_accessed_at) / 86400.0
        decayed = raw * (1.0 - node.decay_factor) ** idle_days

        return max(0.0, min(1.0, decayed))

    def update(self, node: MemoryNode, now: float | None = None) -> float:
        """Recompute and update node.current_importance in place."""
        node.current_importance = self.score(node, now)
        return node.current_importance

    def initial_base_importance(self, content: str, memory_type: MemoryType) -> float:
        """
        Heuristic base importance at creation time.
        Semantic memories start more important than episodic.
        Working memories start low (they'll decay fast anyway).
        """
        defaults = {
            MemoryType.SEMANTIC: 0.8,
            MemoryType.EPISODIC: 0.5,
            MemoryType.PROCEDURAL: 0.7,
            MemoryType.WORKING: 0.3,
        }
        return defaults.get(memory_type, 0.5)

    @staticmethod
    def _recency_score(node: MemoryNode, now: float) -> float:
        """Exponential decay by age. Half-life = 7 days."""
        age_days = (now - node.created_at) / 86400.0
        half_life = 7.0
        return math.exp(-math.log(2) * age_days / half_life)

    @staticmethod
    def _frequency_score(node: MemoryNode) -> float:
        """Log-normalized access frequency. 10 accesses = ~0.5 score."""
        if node.access_count == 0:
            return 0.0
        return min(1.0, math.log1p(node.access_count) / math.log1p(20))
