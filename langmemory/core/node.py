"""
MemoryNode - the atomic unit of LangMemory.

Every piece of information stored in LangMemory is a MemoryNode.
The schema is intentionally stable: changing it breaks Merkle hashes.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class MemoryTier(str, Enum):
    """LSM-inspired storage tier for a memory node."""
    HOT = "hot"    # RAM - recent, frequently accessed
    WARM = "warm"  # Fast disk - moderately accessed
    COLD = "cold"  # Slow disk/object store - archived


class MemoryType(str, Enum):
    """Classification of what kind of memory this node represents."""
    EPISODIC = "episodic"       # Specific events ("user mentioned X in session Y")
    SEMANTIC = "semantic"       # General facts ("user prefers Python over JS")
    PROCEDURAL = "procedural"   # How-to knowledge ("steps to deploy the app")
    WORKING = "working"         # Transient context, high decay rate


@dataclass
class MemoryNode:
    """
    Atomic unit of LangMemory.

    Fields are ordered from most-stable (identity) to most-mutable (importance).
    Do not reorder: Merkle hash computation depends on field ordering.
    """

    # --- Identity (immutable after creation) ---
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""

    # --- Merkle chain (set by MerkleTree on insert) ---
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    content_hash: bytes = field(init=False)
    merkle_hash: bytes = field(init=False, default=b"")

    # --- Embedding (set by embedder pipeline) ---
    embedding: Optional[List[float]] = field(default=None, repr=False)

    # --- Temporal ---
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0

    # --- Importance (mutable, recomputed by decay) ---
    base_importance: float = 0.5
    decay_factor: float = 0.01
    current_importance: float = field(init=False)

    # --- Classification ---
    tier: MemoryTier = MemoryTier.HOT
    memory_type: MemoryType = MemoryType.EPISODIC
    concepts: List[str] = field(default_factory=list)

    # --- Provenance ---
    source_session_id: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    bloom_key: bytes = field(init=False)

    def __post_init__(self) -> None:
        self.content_hash = hashlib.sha256(self.content.encode()).digest()
        self.bloom_key = hashlib.sha256(
            self._normalize_content(self.content).encode()
        ).digest()
        self.current_importance = self.base_importance
        # merkle_hash is set by MerkleTree.insert(), not here

    @staticmethod
    def _normalize_content(content: str) -> str:
        """Canonical form used for deduplication bloom key."""
        return " ".join(content.lower().split())

    def recompute_merkle_hash(self) -> bytes:
        """
        Recompute this node's Merkle hash from content + children.
        Called by MerkleTree after children change.
        """
        h = hashlib.sha256()
        h.update(self.content_hash)
        for child_id in sorted(self.children_ids):  # sorted for determinism
            h.update(child_id.encode())
        self.merkle_hash = h.digest()
        return self.merkle_hash

    def touch(self) -> None:
        """Update access metadata. Called on every retrieval."""
        self.last_accessed_at = time.time()
        self.access_count += 1

    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400.0

    def idle_days(self) -> float:
        return (time.time() - self.last_accessed_at) / 86400.0

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryNode):
            return NotImplemented
        return self.node_id == other.node_id

    def __lt__(self, other: "MemoryNode") -> bool:
        # Used by ImportanceHeap (min-heap by current_importance)
        return self.current_importance < other.current_importance

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "content": self.content,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "content_hash": self.content_hash.hex(),
            "merkle_hash": self.merkle_hash.hex(),
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "base_importance": self.base_importance,
            "decay_factor": self.decay_factor,
            "current_importance": self.current_importance,
            "tier": self.tier.value,
            "memory_type": self.memory_type.value,
            "concepts": self.concepts,
            "source_session_id": self.source_session_id,
            "tags": self.tags,
            "bloom_key": self.bloom_key.hex(),
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryNode":
        node = cls(
            node_id=data["node_id"],
            content=data["content"],
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            created_at=data["created_at"],
            last_accessed_at=data["last_accessed_at"],
            access_count=data.get("access_count", 0),
            base_importance=data.get("base_importance", 0.5),
            decay_factor=data.get("decay_factor", 0.01),
            tier=MemoryTier(data.get("tier", "hot")),
            memory_type=MemoryType(data.get("memory_type", "episodic")),
            concepts=data.get("concepts", []),
            source_session_id=data.get("source_session_id", ""),
            tags=data.get("tags", {}),
        )
        node.current_importance = data.get("current_importance", node.base_importance)
        node.content_hash = bytes.fromhex(data["content_hash"])
        node.merkle_hash = bytes.fromhex(data.get("merkle_hash", ""))
        node.bloom_key = bytes.fromhex(data["bloom_key"])
        if data.get("embedding") is not None:
            node.embedding = [float(x) for x in data["embedding"]]
        return node
