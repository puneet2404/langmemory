"""
Embedder abstraction - pluggable text -> vector encoding.

LangMemory is embedder-agnostic. Zero required dependencies.
Swap backends without changing any other code.

Vector math uses only Python stdlib (math, random, array).
"""
from __future__ import annotations

import hashlib
import math
import random
import struct
from abc import ABC, abstractmethod
from typing import List


# Type alias: a float vector is just List[float]
Vector = List[float]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def normalize(v: Vector) -> Vector:
    n = norm(v) + 1e-9
    return [x / n for x in v]


def cosine_similarity(a: Vector, b: Vector) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


class Embedder(ABC):
    """Abstract base class for text embedders."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""

    @abstractmethod
    def encode(self, text: str) -> Vector:
        """Encode text to a unit-norm float vector of length dim."""

    def encode_batch(self, texts: List[str]) -> List[Vector]:
        """Encode multiple texts. Override for batched efficiency."""
        return [self.encode(t) for t in texts]

    @staticmethod
    def cosine_similarity(a: Vector, b: Vector) -> float:
        return cosine_similarity(a, b)


class StubEmbedder(Embedder):
    """
    Deterministic stub embedder for testing and zero-dep usage.

    Produces a consistent vector for each text using a hash-based seed.
    Not semantically meaningful, but reproducible and requires nothing installed.
    """

    def __init__(self, dim: int = 1536) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> Vector:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(self._dim)]
        return normalize(vec)


class OpenAIEmbedder(Embedder):
    """
    OpenAI text-embedding-3-small (or any OpenAI embedding model).

    Optional dependency: pip install openai
    Set OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required: pip install openai"
            ) from e
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dim = 1536 if "small" in model else 3072

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> Vector:
        response = self._client.embeddings.create(input=text, model=self._model)
        vec = [float(x) for x in response.data[0].embedding]
        return normalize(vec)

    def encode_batch(self, texts: List[str]) -> List[Vector]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [normalize([float(x) for x in d.embedding]) for d in response.data]


class SentenceTransformerEmbedder(Embedder):
    """
    Local embedding via sentence-transformers.

    Optional dependency: pip install sentence-transformers
    No API key needed. Runs locally.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> Vector:
        vec = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vec]

    def encode_batch(self, texts: List[str]) -> List[Vector]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]


def make_embedder(backend: str, **kwargs) -> Embedder:
    """Factory for embedder backends."""
    backends = {
        "stub": StubEmbedder,
        "openai": OpenAIEmbedder,
        "sentence-transformers": SentenceTransformerEmbedder,
    }
    if backend not in backends:
        raise ValueError(
            f"Unknown embedder backend: {backend!r}. Choose from: {list(backends)}"
        )
    return backends[backend](**kwargs)
