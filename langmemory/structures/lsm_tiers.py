"""
LSM Tier Manager - Hot/Warm/Cold storage inspired by Log-Structured Merge Trees.

Memory access follows a power law: recent memories accessed constantly, old ones rarely.
LSM tiering formalizes this with automatic promotion/demotion based on importance.

HOT  (L0): In-memory dict.       Fast R/W. Limited size.
WARM (L1): SQLite store.         Persistent. Moderate speed.
COLD (L2): File-based archive.   Cheap storage. Slow access.

Compaction runs asynchronously: hot -> warm when hot_limit exceeded.
"""
from __future__ import annotations

import threading
from typing import Dict, Iterator, List, Optional

from langmemory.core.node import MemoryNode, MemoryTier


class LSMTierManager:
    """
    Manages three-tier memory storage.

    Nodes live in exactly one tier at a time.
    Demotion is explicit (called by decay worker).
    Promotion happens on access (hot-on-read).
    """

    def __init__(
        self,
        hot_limit: int = 1_000,
        on_warm_store: Optional[callable] = None,
        on_warm_load: Optional[callable] = None,
        on_cold_store: Optional[callable] = None,
        on_cold_load: Optional[callable] = None,
    ) -> None:
        self.hot_limit = hot_limit
        self._hot: Dict[str, MemoryNode] = {}
        self._warm_ids: set[str] = set()
        self._cold_ids: set[str] = set()
        self._lock = threading.RLock()

        # Storage backend callbacks (injected by LangMemory)
        self._warm_store = on_warm_store or (lambda n: None)
        self._warm_load = on_warm_load or (lambda nid: None)
        self._cold_store = on_cold_store or (lambda n: None)
        self._cold_load = on_cold_load or (lambda nid: None)

    def put_hot(self, node: MemoryNode) -> None:
        """Place a node in the HOT tier."""
        with self._lock:
            node.tier = MemoryTier.HOT
            self._hot[node.node_id] = node
            self._warm_ids.discard(node.node_id)
            self._cold_ids.discard(node.node_id)

    def get(self, node_id: str) -> Optional[MemoryNode]:
        """
        Retrieve a node from any tier.
        Promotes WARM/COLD to HOT on access (hot-on-read).
        """
        with self._lock:
            if node_id in self._hot:
                return self._hot[node_id]

            if node_id in self._warm_ids:
                node = self._warm_load(node_id)
                if node:
                    self._promote_to_hot(node)
                return node

            if node_id in self._cold_ids:
                node = self._cold_load(node_id)
                if node:
                    self._promote_to_hot(node)
                return node

            return None

    def demote(self, node: MemoryNode, target: MemoryTier) -> None:
        """Move a node to a lower tier."""
        with self._lock:
            if target == MemoryTier.WARM:
                self._demote_to_warm(node)
            elif target == MemoryTier.COLD:
                self._demote_to_cold(node)

    def compact_to_warm(self, importance_heap=None) -> int:
        """
        Move lowest-importance HOT nodes to WARM until hot_size <= hot_limit.
        Returns number of nodes demoted.
        """
        with self._lock:
            if len(self._hot) <= self.hot_limit:
                return 0

            # Sort hot nodes by importance ascending (lowest first = demote first)
            evict_count = len(self._hot) - self.hot_limit
            candidates = sorted(
                self._hot.values(), key=lambda n: n.current_importance
            )[:evict_count]

            for node in candidates:
                self._demote_to_warm(node)

            return len(candidates)

    def hot_size(self) -> int:
        return len(self._hot)

    def warm_size(self) -> int:
        return len(self._warm_ids)

    def cold_size(self) -> int:
        return len(self._cold_ids)

    def iter_hot(self) -> Iterator[MemoryNode]:
        with self._lock:
            yield from list(self._hot.values())

    def tier_of(self, node_id: str) -> Optional[MemoryTier]:
        with self._lock:
            if node_id in self._hot:
                return MemoryTier.HOT
            if node_id in self._warm_ids:
                return MemoryTier.WARM
            if node_id in self._cold_ids:
                return MemoryTier.COLD
            return None

    def remove(self, node_id: str) -> bool:
        with self._lock:
            if node_id in self._hot:
                del self._hot[node_id]
                return True
            if node_id in self._warm_ids:
                self._warm_ids.discard(node_id)
                return True
            if node_id in self._cold_ids:
                self._cold_ids.discard(node_id)
                return True
            return False

    def _demote_to_warm(self, node: MemoryNode) -> None:
        node.tier = MemoryTier.WARM
        self._hot.pop(node.node_id, None)
        self._cold_ids.discard(node.node_id)
        self._warm_ids.add(node.node_id)
        self._warm_store(node)

    def _demote_to_cold(self, node: MemoryNode) -> None:
        node.tier = MemoryTier.COLD
        self._hot.pop(node.node_id, None)
        self._warm_ids.discard(node.node_id)
        self._cold_ids.add(node.node_id)
        self._cold_store(node)

    def _promote_to_hot(self, node: MemoryNode) -> None:
        node.tier = MemoryTier.HOT
        self._warm_ids.discard(node.node_id)
        self._cold_ids.discard(node.node_id)
        self._hot[node.node_id] = node
        # If over limit after promotion, compact
        if len(self._hot) > self.hot_limit:
            self.compact_to_warm()
