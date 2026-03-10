"""
Memory Merkle Tree - cryptographic integrity for memory chains.

Every MemoryNode gets a Merkle hash: SHA-256(content_hash + children_hashes).
Any modification to any node invalidates its hash and all ancestor hashes.

This enables:
  - Tamper detection: has this memory been modified?
  - Audit trails: provable history of what was known when
  - Consolidation lineage: semantic nodes chain to their episodic sources

O(log n) proof generation, O(log n) verification per node.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Dict, List, Optional

from langmemory.core.node import MemoryNode


class MerkleTree:
    """
    Integrity tree over MemoryNodes.

    The tree is stored as a flat dictionary of node_id -> MemoryNode.
    Parent-child relationships are maintained in the MemoryNode itself
    (parent_id, children_ids fields).

    The root hash changes whenever any node in the tree changes.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, MemoryNode] = {}
        self._roots: List[str] = []  # node_ids with no parent
        self._lock = threading.RLock()

    def insert(self, node: MemoryNode, parent_id: Optional[str] = None) -> bytes:
        """
        Insert a node into the tree.
        Computes its Merkle hash and propagates hash updates to ancestors.
        Returns the node's new merkle_hash.
        """
        with self._lock:
            if parent_id and parent_id in self._nodes:
                parent = self._nodes[parent_id]
                node.parent_id = parent_id
                parent.children_ids.append(node.node_id)
                self._recompute_hash(parent)

            node.recompute_merkle_hash()
            self._nodes[node.node_id] = node

            if node.parent_id is None:
                if node.node_id not in self._roots:
                    self._roots.append(node.node_id)

            # Propagate hash change upward
            self._propagate_up(node)
            return node.merkle_hash

    def verify(self, node_id: str) -> bool:
        """
        Verify that a node's Merkle hash matches its current content.
        Returns False if the node has been tampered with.
        """
        with self._lock:
            if node_id not in self._nodes:
                return False
            node = self._nodes[node_id]
            expected = self._compute_hash(node)
            return node.merkle_hash == expected

    def verify_chain(self, node_id: str) -> bool:
        """Verify a node and all its ancestors."""
        with self._lock:
            nid: Optional[str] = node_id
            while nid is not None:
                if not self.verify(nid):
                    return False
                node = self._nodes.get(nid)
                if node is None:
                    break
                nid = node.parent_id
            return True

    def verify_full(self) -> Dict[str, bool]:
        """Verify every node. Returns {node_id: is_valid}."""
        with self._lock:
            return {nid: self.verify(nid) for nid in self._nodes}

    def root_hash(self) -> bytes:
        """
        Aggregate hash of all root nodes.
        Changes if ANY memory in the system changes.
        """
        with self._lock:
            if not self._roots:
                return b""
            h = hashlib.sha256()
            for root_id in sorted(self._roots):
                node = self._nodes.get(root_id)
                if node:
                    h.update(node.merkle_hash)
            return h.digest()

    def get_proof(self, node_id: str) -> List[bytes]:
        """
        Return the chain of Merkle hashes from this node to a root.
        Allows external verification without the full tree.
        """
        with self._lock:
            proof: List[bytes] = []
            nid: Optional[str] = node_id
            while nid is not None:
                node = self._nodes.get(nid)
                if node is None:
                    break
                proof.append(node.merkle_hash)
                nid = node.parent_id
            return proof

    def find_nearest_ancestor(self, node: MemoryNode) -> Optional[MemoryNode]:
        """
        Find the most recent root node (for attaching new memories).
        Simple heuristic: pick the root with the highest created_at.
        """
        with self._lock:
            if not self._roots:
                return None
            roots = [self._nodes[rid] for rid in self._roots if rid in self._nodes]
            if not roots:
                return None
            return max(roots, key=lambda n: n.created_at)

    def _compute_hash(self, node: MemoryNode) -> bytes:
        h = hashlib.sha256()
        h.update(node.content_hash)
        for child_id in sorted(node.children_ids):
            h.update(child_id.encode())
        return h.digest()

    def _recompute_hash(self, node: MemoryNode) -> None:
        node.merkle_hash = self._compute_hash(node)

    def _propagate_up(self, node: MemoryNode) -> None:
        """Walk up the tree recomputing hashes after an insert/update."""
        nid = node.parent_id
        while nid is not None:
            parent = self._nodes.get(nid)
            if parent is None:
                break
            old_hash = parent.merkle_hash
            self._recompute_hash(parent)
            if parent.merkle_hash == old_hash:
                break  # No change propagated further up
            nid = parent.parent_id

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes
