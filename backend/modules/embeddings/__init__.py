"""Embedding providers for the tiny Phase 3C experiment.

This module keeps the application modular so the real embedding implementation can be
replaced later without changing the retrieval layer contract.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class EmbeddingsProvider(EmbeddingProvider):
    """Backward-compatible alias kept for older module references."""


class InMemoryEmbeddingProvider(EmbeddingProvider):
    """Deterministic local implementation used for lightweight experiment tests."""

    def __init__(self, dimension: int = 3) -> None:
        self._dimension = max(1, dimension)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for text in texts:
            if text is None or text == "":
                vectors.append([0.0 for _ in range(self._dimension)])
                continue
            chars = [ord(ch) for ch in text]
            base = float(len(text))
            second = float(sum(chars) % 1000)
            third = float(len(set(text)))
            vectors.append([base, second, third][: self._dimension] + [0.0] * max(0, self._dimension - 3))
        return vectors

    def dimension(self) -> int:
        return self._dimension


class BGE3EmbeddingProvider(EmbeddingProvider):
    """Wrapper around the BAAI/bge-m3 model when it is installed locally.

    This is intentionally modular: the live implementation may be swapped later without
    changing the rest of the embedding or retrieval flow.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = 1024

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only when model is used explicitly
            raise RuntimeError(
                "sentence-transformers is required to use BGE3EmbeddingProvider. "
                "Install it for the tiny local BGE-M3 prototype."
            ) from exc

        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)

        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)

    def dimension(self) -> int:
        return self._dimension


class HashingEmbeddingProvider(EmbeddingProvider):
    """Dependency-free deterministic embedding via the hashing trick.

    Produces fixed-dimension, L2-normalized vectors from word unigrams and character
    n-grams using signed feature hashing (à la ``sklearn.HashingVectorizer``). This gives
    genuine lexical/sub-word similarity between a query and documents *without* requiring
    torch/sentence-transformers, so dense retrieval is meaningful even in a minimal install.

    Crucially it uses ``hashlib`` (not the salted built-in ``hash``) so vectors are stable
    across processes and runs — the seeder and the query path therefore share one space.
    """

    _TOKEN_RE = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)

    def __init__(
        self,
        dimension: int = 256,
        char_ngram: int = 3,
        model_name: str = "hashing-ngram-v1",
    ) -> None:
        self._dimension = max(1, dimension)
        self._char_ngram = max(2, char_ngram)
        self.model_name = model_name

    def _features(self, text: str) -> list[str]:
        """Word unigrams plus intra-token character n-grams (boundary-padded)."""
        tokens = self._TOKEN_RE.findall(text.lower())
        features: list[str] = []
        for token in tokens:
            features.append(f"w:{token}")
            padded = f"^{token}$"
            if len(padded) >= self._char_ngram:
                for i in range(len(padded) - self._char_ngram + 1):
                    features.append(f"c:{padded[i : i + self._char_ngram]}")
        return features

    @staticmethod
    def _hash(feature: str) -> int:
        return int(hashlib.md5(feature.encode("utf-8")).hexdigest(), 16)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        if not text:
            return vector
        for feature in self._features(text):
            h = self._hash(feature)
            index = h % self._dimension
            sign = 1.0 if (h // self._dimension) % 2 == 0 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude > 0.0:
            vector = [value / magnitude for value in vector]
        return vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._embed_one(text or "") for text in texts]

    def dimension(self) -> int:
        return self._dimension


__all__ = [
    "EmbeddingProvider",
    "EmbeddingsProvider",
    "InMemoryEmbeddingProvider",
    "BGE3EmbeddingProvider",
    "HashingEmbeddingProvider",
]
