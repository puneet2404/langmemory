"""
Importance Heap - O(log n) priority retrieval by importance score.

Uses Python's heapq (min-heap) with the lazy deletion pattern:
- Updating a node's priority = push new entry, mark old as stale
- Extract-min skips stale entries

This gives O(log n) push, O(log n) amortized pop, O(log n) priority update.
"""
from __future__ import annotations

import heapq
import threading
from typing import List, Optional, Tuple

from langmemory.core.node import MemoryNode


class ImportanceHeap:
    """
    Min-heap ordered by MemoryNode.current_importance.

    Lower importance = higher priority for eviction (this is a min-heap).
    Use extract_top_k(k) to get the k MOST important nodes (highest scores).
    Use extract_min() to get the LEAST important node (for eviction).
    """

    def __init__(self) -> None:
        # Heap entries: (importance_score, insertion_order, node_id)
        # insertion_order breaks ties deterministically
        self._heap: List[Tuple[float, int, str]] = []
        self._node_map: dict[str, MemoryNode] = {}
        self._stale: set[str] = set()  # node_ids with stale heap entries
        self._counter = 0
        self._lock = threading.RLock()

    def push(self, node: MemoryNode) -> None:
        """Add a node to the heap."""
        with self._lock:
            self._node_map[node.node_id] = node
            entry = (node.current_importance, self._counter, node.node_id)
            heapq.heappush(self._heap, entry)
            self._counter += 1

    def update_priority(self, node: MemoryNode) -> None:
        """
        Update a node's importance score.
        Marks old entry as stale and pushes a new one (lazy deletion).
        O(log n) amortized.
        """
        with self._lock:
            if node.node_id in self._node_map:
                self._stale.add(node.node_id)
            self._node_map[node.node_id] = node
            entry = (node.current_importance, self._counter, node.node_id)
            heapq.heappush(self._heap, entry)
            self._counter += 1

    def extract_min(self) -> Optional[MemoryNode]:
        """
        Extract the LEAST important node (lowest score).
        Used for eviction decisions.
        """
        with self._lock:
            return self._pop_valid()

    def extract_top_k(self, k: int) -> List[MemoryNode]:
        """
        Return the k MOST important nodes (highest current_importance).
        Does NOT remove them from the heap.
        """
        with self._lock:
            if not self._node_map:
                return []
            # Sort all valid nodes by importance descending
            valid = sorted(
                self._node_map.values(),
                key=lambda n: n.current_importance,
                reverse=True,
            )
            return valid[:k]

    def peek_min(self) -> Optional[MemoryNode]:
        """Peek at the least important node without removing it."""
        with self._lock:
            self._clean_stale()
            if not self._heap:
                return None
            _, _, node_id = self._heap[0]
            return self._node_map.get(node_id)

    def remove(self, node_id: str) -> bool:
        """Remove a node by id. O(1) amortized via lazy deletion."""
        with self._lock:
            if node_id not in self._node_map:
                return False
            self._stale.add(node_id)
            del self._node_map[node_id]
            return True

    def _pop_valid(self) -> Optional[MemoryNode]:
        while self._heap:
            importance, _, node_id = heapq.heappop(self._heap)
            if node_id in self._stale:
                self._stale.discard(node_id)
                continue
            return self._node_map.get(node_id)
        return None

    def _clean_stale(self) -> None:
        """Remove all stale heap entries. Called before peek operations."""
        cleaned: List[Tuple[float, int, str]] = []
        while self._heap:
            entry = self._heap[0]
            if entry[2] in self._stale:
                heapq.heappop(self._heap)
                self._stale.discard(entry[2])
            else:
                break

    def __len__(self) -> int:
        return len(self._node_map)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._node_map
