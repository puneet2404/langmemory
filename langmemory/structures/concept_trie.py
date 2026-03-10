"""
Concept Trie - O(key_length) hierarchical concept indexing.

Concepts are hierarchical paths: "Python/types/generics"
The trie lets you find all memories under a concept subtree in O(prefix_length + k).

This preserves concept hierarchy that flat vector search collapses.
Querying "Python" returns all memories under Python/*, Python/types/*, etc.
"""
from __future__ import annotations

import threading
from typing import Dict, Iterator, List, Optional, Set


class _TrieNode:
    __slots__ = ("children", "node_ids", "concept")

    def __init__(self, concept: str = "") -> None:
        self.concept = concept
        self.children: Dict[str, "_TrieNode"] = {}
        self.node_ids: Set[str] = set()  # memory node_ids at this concept level


class ConceptTrie:
    """
    Prefix trie for hierarchical concept indexing.

    Concept paths use "/" as separator: "ML/optimization/Adam"
    creates trie levels: root -> "ML" -> "optimization" -> "Adam"

    All node_ids inserted under "ML" are also retrievable via prefix_search("ML").
    """

    SEPARATOR = "/"

    def __init__(self) -> None:
        self._root = _TrieNode()
        self._lock = threading.RLock()
        self._total_entries = 0

    def insert(self, concepts: List[str], node_id: str) -> None:
        """
        Register node_id under each concept in the list.
        Each concept is a path like "Python/types/generics".
        """
        with self._lock:
            for concept in concepts:
                self._insert_path(concept, node_id)
            self._total_entries += 1

    def remove(self, concepts: List[str], node_id: str) -> None:
        """Remove a node_id from all its concept paths."""
        with self._lock:
            for concept in concepts:
                self._remove_path(concept, node_id)
            self._total_entries = max(0, self._total_entries - 1)

    def prefix_search(self, prefix: str) -> List[str]:
        """
        Return all node_ids under the given concept prefix (inclusive).
        "Python" returns ids from "Python", "Python/types", "Python/types/generics", etc.
        O(prefix_length + k) where k = result count.
        """
        with self._lock:
            node = self._find_node(prefix)
            if node is None:
                return []
            result: List[str] = []
            self._collect_all(node, result)
            return result

    def exact_search(self, concept: str) -> List[str]:
        """Return node_ids at exactly this concept level (no children)."""
        with self._lock:
            node = self._find_node(concept)
            if node is None:
                return []
            return list(node.node_ids)

    def subtree_size(self, prefix: str) -> int:
        """Count total node_ids under a concept prefix."""
        with self._lock:
            node = self._find_node(prefix)
            if node is None:
                return 0
            return self._count_all(node)

    def all_concepts(self) -> List[str]:
        """Return all concept paths in the trie (DFS traversal)."""
        with self._lock:
            result: List[str] = []
            self._collect_concepts(self._root, "", result)
            return result

    def _insert_path(self, concept: str, node_id: str) -> None:
        parts = concept.split(self.SEPARATOR)
        current = self._root
        for part in parts:
            if not part:
                continue
            if part not in current.children:
                current.children[part] = _TrieNode(part)
            current = current.children[part]
        current.node_ids.add(node_id)

    def _remove_path(self, concept: str, node_id: str) -> None:
        parts = concept.split(self.SEPARATOR)
        current = self._root
        for part in parts:
            if not part:
                continue
            if part not in current.children:
                return
            current = current.children[part]
        current.node_ids.discard(node_id)

    def _find_node(self, concept: str) -> Optional[_TrieNode]:
        parts = concept.split(self.SEPARATOR)
        current = self._root
        for part in parts:
            if not part:
                continue
            if part not in current.children:
                return None
            current = current.children[part]
        return current

    def _collect_all(self, node: _TrieNode, result: List[str]) -> None:
        result.extend(node.node_ids)
        for child in node.children.values():
            self._collect_all(child, result)

    def _count_all(self, node: _TrieNode) -> int:
        count = len(node.node_ids)
        for child in node.children.values():
            count += self._count_all(child)
        return count

    def _collect_concepts(
        self, node: _TrieNode, path: str, result: List[str]
    ) -> None:
        if node.node_ids:
            result.append(path.lstrip(self.SEPARATOR))
        for key, child in node.children.items():
            new_path = f"{path}{self.SEPARATOR}{key}"
            self._collect_concepts(child, new_path, result)

    def __len__(self) -> int:
        return self._total_entries
