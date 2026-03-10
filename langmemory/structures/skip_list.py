"""
Temporal Skip List - O(log n) ordered traversal of memory timestamps.

A skip list is a probabilistic data structure that maintains sorted order
with O(log n) average complexity for insert, delete, and range queries.
Redis uses skip lists for sorted sets; we use them for temporal memory indexing.

Key operations:
  insert(timestamp, node_id)          O(log n)
  range_query(start, end) -> ids      O(log n + k) where k = results
  tail(n) -> ids                      O(log n + n)
  delete(timestamp, node_id)          O(log n)
"""
from __future__ import annotations

import random
import threading
from typing import List, Optional, Tuple


class _SkipNode:
    __slots__ = ("key", "value", "forward")

    def __init__(self, key: Optional[float], value: Optional[str], level: int) -> None:
        self.key = key          # timestamp (float), None for head/tail sentinels
        self.value = value      # node_id (str)
        self.forward: List[Optional["_SkipNode"]] = [None] * (level + 1)


class TemporalSkipList:
    """
    Skip list keyed by float timestamps, storing node_id strings.

    Multiple node_ids can share the same timestamp (concurrent inserts).
    All operations are thread-safe via a reentrant lock.
    """

    NEG_INF = float("-inf")
    POS_INF = float("inf")

    def __init__(self, max_level: int = 16, probability: float = 0.5) -> None:
        self.max_level = max_level
        self.probability = probability
        self._level = 0
        self._head = _SkipNode(self.NEG_INF, None, max_level)
        self._tail = _SkipNode(self.POS_INF, None, max_level)
        for i in range(max_level + 1):
            self._head.forward[i] = self._tail
        self._count = 0
        self._lock = threading.RLock()

    def _random_level(self) -> int:
        lvl = 0
        while random.random() < self.probability and lvl < self.max_level:
            lvl += 1
        return lvl

    def insert(self, timestamp: float, node_id: str) -> None:
        """Insert a (timestamp, node_id) pair. Duplicate timestamps allowed."""
        with self._lock:
            update: List[Optional[_SkipNode]] = [None] * (self.max_level + 1)
            current = self._head

            for i in range(self._level, -1, -1):
                while (
                    current.forward[i] is not None
                    and current.forward[i].key is not None
                    and current.forward[i].key != self.POS_INF
                    and (
                        current.forward[i].key < timestamp
                        or (
                            current.forward[i].key == timestamp
                            and current.forward[i].value is not None
                            and current.forward[i].value < node_id
                        )
                    )
                ):
                    current = current.forward[i]
                update[i] = current

            lvl = self._random_level()
            if lvl > self._level:
                for i in range(self._level + 1, lvl + 1):
                    update[i] = self._head
                self._level = lvl

            new_node = _SkipNode(timestamp, node_id, lvl)
            for i in range(lvl + 1):
                new_node.forward[i] = update[i].forward[i]  # type: ignore[union-attr]
                update[i].forward[i] = new_node  # type: ignore[union-attr]

            self._count += 1

    def delete(self, timestamp: float, node_id: str) -> bool:
        """Remove a (timestamp, node_id) pair. Returns True if found."""
        with self._lock:
            update: List[Optional[_SkipNode]] = [None] * (self.max_level + 1)
            current = self._head

            for i in range(self._level, -1, -1):
                while (
                    current.forward[i] is not None
                    and current.forward[i].key is not None
                    and current.forward[i].key != self.POS_INF
                    and (
                        current.forward[i].key < timestamp
                        or (
                            current.forward[i].key == timestamp
                            and current.forward[i].value is not None
                            and current.forward[i].value < node_id
                        )
                    )
                ):
                    current = current.forward[i]
                update[i] = current

            target = current.forward[0]
            if (
                target is None
                or target.key != timestamp
                or target.value != node_id
                or target.key == self.POS_INF
            ):
                return False

            for i in range(self._level + 1):
                if update[i] is not None and update[i].forward[i] == target:  # type: ignore[union-attr]
                    update[i].forward[i] = target.forward[i]  # type: ignore[union-attr]

            while self._level > 0 and self._head.forward[self._level] == self._tail:
                self._level -= 1

            self._count -= 1
            return True

    def range_query(self, start: float, end: float) -> List[str]:
        """
        Return all node_ids with timestamps in [start, end].
        O(log n + k) where k is the number of results.
        """
        with self._lock:
            results: List[str] = []
            current = self._head

            # Descend to level 0, finding first node >= start
            for i in range(self._level, -1, -1):
                while (
                    current.forward[i] is not None
                    and current.forward[i].key is not None
                    and current.forward[i].key != self.POS_INF
                    and current.forward[i].key < start
                ):
                    current = current.forward[i]

            # Walk level-0 forward pointers collecting results
            current = current.forward[0]
            while (
                current is not None
                and current.key is not None
                and current.key != self.POS_INF
                and current.key <= end
            ):
                if current.value is not None:
                    results.append(current.value)
                current = current.forward[0]

            return results

    def tail(self, n: int) -> List[str]:
        """Return the n most-recent (highest timestamp) node_ids."""
        with self._lock:
            # Collect all from level 0 in O(count), take last n
            # For large n this is acceptable; could optimize with reverse pointer
            all_ids: List[Tuple[float, str]] = []
            current = self._head.forward[0]
            while (
                current is not None
                and current.key is not None
                and current.key != self.POS_INF
            ):
                if current.value is not None and current.key is not None:
                    all_ids.append((current.key, current.value))
                current = current.forward[0]
            return [nid for _, nid in all_ids[-n:]]

    def __len__(self) -> int:
        return self._count
