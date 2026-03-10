"""
Bloom Filter - O(1) probabilistic membership check.

Used as a deduplication guard before expensive embedding similarity checks.
"Definitely not seen" in O(1). False positives possible (tunable rate).

Implementation uses MurmurHash3 via mmh3 if available, falls back to SHA-256
double hashing (Kirsch-Mitzenmacher optimization) otherwise.
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional


def _double_hash(key: bytes, i: int, m: int) -> int:
    """Kirsch-Mitzenmacher double hashing: h(i, k) = h1(k) + i * h2(k) mod m."""
    h1 = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    h2 = int.from_bytes(hashlib.md5(key).digest()[:8], "big")  # noqa: S324
    return (h1 + i * h2) % m


try:
    import mmh3 as _mmh3

    def _hash_fn(key: bytes, i: int, m: int) -> int:
        return _mmh3.hash(key, seed=i, signed=False) % m

except ImportError:
    _hash_fn = _double_hash


class BloomFilter:
    """
    Space-efficient probabilistic set.

    Args:
        capacity:   Expected number of elements to insert.
        error_rate: Acceptable false positive probability (0 < error_rate < 1).
    """

    def __init__(self, capacity: int = 100_000, error_rate: float = 0.01) -> None:
        if not (0 < error_rate < 1):
            raise ValueError("error_rate must be between 0 and 1")
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.error_rate = error_rate
        self.size = self._optimal_size(capacity, error_rate)
        self.hash_count = self._optimal_hash_count(self.size, capacity)
        self._bits = bytearray(math.ceil(self.size / 8))
        self._count = 0

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Optimal bit array size: m = -n*ln(p) / (ln(2))^2"""
        return math.ceil(-n * math.log(p) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Optimal number of hash functions: k = (m/n) * ln(2)"""
        return max(1, round((m / n) * math.log(2)))

    def _bit_positions(self, key: bytes):
        for i in range(self.hash_count):
            yield _hash_fn(key, i, self.size)

    def add(self, key: bytes) -> None:
        for pos in self._bit_positions(key):
            byte_idx, bit_idx = divmod(pos, 8)
            self._bits[byte_idx] |= 1 << bit_idx
        self._count += 1

    def might_contain(self, key: bytes) -> bool:
        """
        Returns False if key is DEFINITELY not in the set.
        Returns True if key MIGHT be in the set (possible false positive).
        """
        for pos in self._bit_positions(key):
            byte_idx, bit_idx = divmod(pos, 8)
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def false_positive_rate(self) -> float:
        """Current estimated false positive rate based on actual insertions."""
        if self._count == 0:
            return 0.0
        # p = (1 - e^(-k*n/m))^k  where k=hash_count, n=inserted, m=bit_size
        return (1 - math.exp(-self.hash_count * self._count / self.size)) ** self.hash_count

    def __len__(self) -> int:
        return self._count

    def to_bytes(self) -> bytes:
        """Serialize to bytes for persistence."""
        import struct
        header = struct.pack(">IId", self.capacity, self.hash_count, self.error_rate)
        return header + bytes(self._bits)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BloomFilter":
        """Deserialize from bytes."""
        import struct
        header_size = struct.calcsize(">IId")
        capacity, hash_count, error_rate = struct.unpack(">IId", data[:header_size])
        bf = cls(capacity=capacity, error_rate=error_rate)
        bf._bits = bytearray(data[header_size:])
        bf.hash_count = hash_count
        return bf
