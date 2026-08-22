"""Retrieval interfaces and lightweight local implementations for Phase 3C.

This keeps dense, sparse, and hybrid search modular while allowing a later Qdrant-backed
implementation to replace the in-memory retrieval engine without changing the rest of the
pipeline contracts.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A retrieved chunk or document with provenance and retrieval metadata."""

    chunk_id: str
    text: str
    score: float
    method: str = "dense"
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(ABC):
    """Abstract interface for retrieval strategies."""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> list[RetrievedDocument]:
        """Retrieve relevant documents for a query."""
        ...


class VectorStore(ABC):
    """Abstract interface for local or remote vector stores."""

    @abstractmethod
    async def upsert(self, documents: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        """Insert or update documents with associated vectors."""
        ...

    @abstractmethod
    async def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Search the vector store and return ranked hits."""
        ...


class InMemoryVectorStore(VectorStore):
    """Local vector store used for tiny, deterministic retrieval experiments."""

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    async def upsert(self, documents: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        if not documents:
            return
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")

        for document, embedding in zip(documents, embeddings, strict=True):
            self._items = [
                item for item in self._items if item["id"] != document["id"]
            ]
            self._items.append(
                {
                    "id": document["id"],
                    "text": document["text"],
                    "metadata": document.get("metadata", {}),
                    "embedding": embedding,
                }
            )

    async def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        if not query_vector or not self._items:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self._items:
            vector = item["embedding"]
            score = dot_product(query_vector, vector) / (norm(query_vector) * norm(vector)) if norm(query_vector) and norm(vector) else 0.0
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, item in scored[:max(1, top_k)]:
            results.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "score": score,
                }
            )
        return results


class QdrantVectorStore(VectorStore):
    """Thin local adapter for a tiny Qdrant prototype. This is intentionally modular and
    not wired into the rest of the app by default.
    """

    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "thinkzen_tiny") -> None:
        self.url = url
        self.collection_name = collection_name
        self._client = None

    async def upsert(self, documents: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - intentional local-only runtime check
            raise RuntimeError("qdrant-client is required for the tiny Qdrant prototype.") from exc

        if self._client is None:
            self._client = QdrantClient(url=self.url)

        payload = [{"id": document["id"], **document.get("metadata", {}), "text": document["text"]} for document in documents]
        ids = [document["id"] for document in documents]
        self._client.upsert(
            collection_name=self.collection_name,
            points=[{"id": idx, "vector": vector, "payload": payload_item} for idx, (vector, payload_item) in enumerate(zip(embeddings, payload, strict=True))],
        )
        _ = ids

    async def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        if self._client is None:
            return []
        hits = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        return [{"id": hit.id, "text": hit.payload.get("text", ""), "metadata": hit.payload, "score": float(hit.score)} for hit in hits]


class SparseRetriever(Retriever):
    """Simple lexical retrieval used for the tiny prototype and future hybrid fusion."""

    def __init__(self, method: str = "simple") -> None:
        self.method = method

    async def retrieve(self, query: str, documents: list[RetrievedDocument] | None = None, top_k: int = 5, **kwargs: Any) -> list[RetrievedDocument]:
        if not query or not documents:
            return []

        query_terms = {token.lower() for token in query.split() if token.strip()}
        if not query_terms:
            return []

        scored = []
        for document in documents:
            text_terms = {token.lower() for token in document.text.split() if token.strip()}
            overlap = len(query_terms & text_terms)
            score = overlap / max(1, len(query_terms | text_terms))
            scored.append(
                RetrievedDocument(
                    chunk_id=document.chunk_id,
                    text=document.text,
                    score=score,
                    method="sparse",
                    metadata={**(document.metadata or {}), "retrieval_method": "sparse"},
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(1, top_k)]


async def dense_retriever(query: str, vector_store: VectorStore, top_k: int = 5) -> list[RetrievedDocument]:
    """Dense retrieval over a vector store using a lightweight deterministic embedding function."""
    if not query or vector_store is None:
        return []

    query_vector = _text_to_vector(query)
    hits = await vector_store.search(query_vector, top_k=top_k)
    return [
        RetrievedDocument(
            chunk_id=str(hit["id"]),
            text=str(hit.get("text", "")),
            score=float(hit.get("score", 0.0)),
            method="dense",
            metadata={**(hit.get("metadata", {}) or {}), "retrieval_method": "dense"},
        )
        for hit in hits
    ]


class HybridRetriever(Retriever):
    """Merge dense and sparse candidate scores while preserving provenance metadata."""

    def __init__(self, dense_weight: float = 0.6, sparse_weight: float = 0.4) -> None:
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    async def retrieve(
        self,
        query: str,
        dense_results: list[RetrievedDocument] | None = None,
        sparse_results: list[RetrievedDocument] | None = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[RetrievedDocument]:
        if not query:
            return []

        merged: dict[str, RetrievedDocument] = {}
        dense_results = dense_results or []
        sparse_results = sparse_results or []

        for result in dense_results:
            merged[result.chunk_id] = RetrievedDocument(
                chunk_id=result.chunk_id,
                text=result.text,
                score=self.dense_weight * result.score,
                method="hybrid",
                metadata={**(result.metadata or {}), "retrieval_method": "hybrid"},
            )

        for result in sparse_results:
            if result.chunk_id in merged:
                merged[result.chunk_id].score += self.sparse_weight * result.score
                merged[result.chunk_id].metadata.update({**(result.metadata or {}), "retrieval_method": "hybrid"})
            else:
                merged[result.chunk_id] = RetrievedDocument(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=self.sparse_weight * result.score,
                    method="hybrid",
                    metadata={**(result.metadata or {}), "retrieval_method": "hybrid"},
                )

        ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ordered[: max(1, top_k)]


class OrchestratedHybridRetriever(Retriever):
    """Orchestrated hybrid retriever that combines dense and sparse retrieval in one call.
    
    This retriever:
    1. Retrieves from both dense (embedding-based) and sparse (token-based) stores
    2. Normalizes scores to [0, 1] range
    3. Combines using configurable alpha parameter
    4. Preserves all metadata from both retrieval methods
    """

    def __init__(
        self,
        dense_store: VectorStore | None = None,
        sparse_store: Any | None = None,
        alpha: float = 0.5,
        embedding_provider: Any | None = None,
    ) -> None:
        """Initialize orchestrated hybrid retriever.
        
        Args:
            dense_store: Vector store for dense (embedding-based) retrieval
            sparse_store: Sparse store for token-based retrieval (e.g., BM25VectorStore)
            alpha: Weight for dense retrieval (0-1). 0=sparse only, 1=dense only
            embedding_provider: Provider for embedding queries (if dense_store is None)
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in range [0, 1]")
        
        self.dense_store = dense_store
        self.sparse_store = sparse_store
        self.alpha = alpha
        self.embedding_provider = embedding_provider

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[RetrievedDocument]:
        """Retrieve documents using hybrid dense + sparse scoring.
        
        Args:
            query: Query text
            top_k: Maximum number of results to return
            **kwargs: Additional arguments (ignored)
            
        Returns:
            List of RetrievedDocument objects sorted by hybrid score
        """
        if not query:
            return []

        dense_results: dict[str, tuple[float, str, dict[str, Any]]] = {}
        sparse_results: dict[str, tuple[float, str, dict[str, Any]]] = {}

        # Dense retrieval
        if self.dense_store and self.alpha > 0:
            try:
                if self.embedding_provider:
                    query_embedding = (await self.embedding_provider.embed([query]))[0]
                else:
                    query_embedding = _text_to_vector(query)
                
                dense_hits = await self.dense_store.search(query_embedding, top_k=top_k)
                for hit in dense_hits:
                    doc_id = str(hit.get("id", ""))
                    text = str(hit.get("text", ""))
                    score = float(hit.get("score", 0.0))
                    metadata = {**(hit.get("metadata", {}) or {}), "retrieval_method": "dense"}
                    dense_results[doc_id] = (score, text, metadata)
            except Exception as e:
                logger.warning("Dense retrieval failed: %s", e)

        # Sparse retrieval
        if self.sparse_store and self.alpha < 1:
            try:
                sparse_hits = await self.sparse_store.search(query, top_k=top_k)
                for hit in sparse_hits:
                    doc_id = str(hit.get("id", ""))
                    text = str(hit.get("text", ""))
                    score = float(hit.get("score", 0.0))
                    metadata = {**(hit.get("metadata", {}) or {}), "retrieval_method": "sparse"}
                    sparse_results[doc_id] = (score, text, metadata)
            except Exception as e:
                logger.warning("Sparse retrieval failed: %s", e)

        # Normalize and combine scores
        merged: dict[str, tuple[float, str, dict[str, Any]]] = {}

        # Normalize dense scores
        dense_scores = [score for score, _, _ in dense_results.values()]
        dense_max = max(dense_scores) if dense_scores else 1.0
        dense_min = min(dense_scores) if dense_scores else 0.0
        dense_range = dense_max - dense_min if dense_max > dense_min else 1.0

        for doc_id, (score, text, metadata) in dense_results.items():
            normalized_score = (score - dense_min) / dense_range if dense_range > 0 else 0.0
            merged[doc_id] = (self.alpha * normalized_score, text, metadata)

        # Normalize sparse scores
        sparse_scores = [score for score, _, _ in sparse_results.values()]
        sparse_max = max(sparse_scores) if sparse_scores else 1.0
        sparse_min = min(sparse_scores) if sparse_scores else 0.0
        sparse_range = sparse_max - sparse_min if sparse_max > sparse_min else 1.0

        for doc_id, (score, text, metadata) in sparse_results.items():
            normalized_score = (score - sparse_min) / sparse_range if sparse_range > 0 else 0.0
            sparse_contribution = (1.0 - self.alpha) * normalized_score
            
            if doc_id in merged:
                # Combine both scores
                existing_score, existing_text, existing_metadata = merged[doc_id]
                merged[doc_id] = (existing_score + sparse_contribution, existing_text or text, {**existing_metadata, **metadata})
            else:
                merged[doc_id] = (sparse_contribution, text, metadata)

        # Convert to RetrievedDocument and sort
        results: list[RetrievedDocument] = []
        for doc_id, (score, text, metadata) in sorted(merged.items(), key=lambda x: x[1][0], reverse=True):
            results.append(
                RetrievedDocument(
                    chunk_id=doc_id,
                    text=text,
                    score=score,
                    method="hybrid",
                    metadata={**metadata, "alpha": self.alpha},
                )
            )

        return results[:max(1, top_k)]


def _text_to_vector(text: str) -> list[float]:
    if not text:
        return [0.0, 0.0, 0.0]
    token_values = [ord(ch) for ch in text.lower() if ch.isalnum()]
    if not token_values:
        return [0.0, 0.0, 0.0]
    return [
        float(len(text)),
        float(sum(token_values) % 1000),
        float(len(set(text.lower()))),
    ]


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


__all__ = [
    "RetrievedDocument",
    "Retriever",
    "VectorStore",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "SparseRetriever",
    "HybridRetriever",
    "OrchestratedHybridRetriever",
    "dense_retriever",
]
