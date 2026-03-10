"""
Chain Verifier - Merkle integrity verification for LangMemory.

Provides auditable proof that memories have not been tampered with.
This is the key differentiator from every existing LLM memory system.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langmemory.structures.merkle_tree import MerkleTree


@dataclass
class IntegrityReport:
    timestamp: float = field(default_factory=time.time)
    total_nodes: int = 0
    valid_nodes: int = 0
    corrupted_nodes: List[str] = field(default_factory=list)
    root_hash: bytes = b""
    is_valid: bool = True
    verification_time_ms: float = 0.0

    @property
    def corruption_rate(self) -> float:
        if self.total_nodes == 0:
            return 0.0
        return len(self.corrupted_nodes) / self.total_nodes


class ChainVerifier:
    """
    Verifies the integrity of the LangMemory Merkle tree.

    Use verify_node() for spot-checks.
    Use verify_full() for complete audit.
    Use root_hash() to detect any change in the entire memory system.
    """

    def __init__(self, merkle: MerkleTree) -> None:
        self._merkle = merkle

    def verify_node(self, node_id: str) -> bool:
        """Verify a single node's Merkle hash. O(1)."""
        return self._merkle.verify(node_id)

    def verify_chain(self, node_id: str) -> bool:
        """Verify a node and all its ancestors. O(depth)."""
        return self._merkle.verify_chain(node_id)

    def verify_full(self) -> IntegrityReport:
        """
        Verify every node in the tree.
        Returns an IntegrityReport with details on any corruption found.
        """
        start = time.time()
        results: Dict[str, bool] = self._merkle.verify_full()

        corrupted = [nid for nid, valid in results.items() if not valid]
        valid_count = len(results) - len(corrupted)

        report = IntegrityReport(
            total_nodes=len(results),
            valid_nodes=valid_count,
            corrupted_nodes=corrupted,
            root_hash=self._merkle.root_hash(),
            is_valid=len(corrupted) == 0,
            verification_time_ms=(time.time() - start) * 1000,
        )
        return report

    def get_proof(self, node_id: str) -> List[bytes]:
        """
        Return the Merkle proof chain for a node.
        Can be used to verify memory integrity without the full tree.
        """
        return self._merkle.get_proof(node_id)

    def root_hash(self) -> bytes:
        """
        Current root hash of the entire memory system.
        Store this externally to detect any future tampering.
        """
        return self._merkle.root_hash()
