"""BM25-based sparse retrieval for the hybrid Phase 5 experiment.

This module provides token-based sparse retrieval using BM25 scoring,
complementing the dense embedding-based retrieval in a hybrid pipeline.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


class BM25Indexer:
    """Build and maintain a BM25 sparse index over text documents.
    
    BM25 (Best Matching 25) is a probabilistic ranking function that scores
    documents based on term frequency, inverse document frequency, and
    document length normalization.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 indexer with tuning parameters.
        
        Args:
            k1: Controls term frequency saturation (typically 1.2-2.0)
            b: Controls length normalization (0-1, typically 0.75)
        """
        self.k1 = k1
        self.b = b
        self._documents: dict[str, str] = {}
        self._doc_terms: dict[str, set[str]] = {}
        self._idf: dict[str, float] = {}
        self._avg_doc_length = 0.0
        self._total_docs = 0

    def add_document(self, doc_id: str, text: str) -> None:
        """Add or update a document in the index.
        
        Args:
            doc_id: Unique document identifier
            text: Document text to tokenize and index
        """
        self._documents[doc_id] = text
        tokens = self._tokenize(text)
        self._doc_terms[doc_id] = set(tokens)

    def build_index(self) -> None:
        """Build the index after all documents are added.
        
        Computes IDF values and average document length.
        """
        if not self._documents:
            return

        self._total_docs = len(self._documents)
        total_length = 0

        # Single O(n) pass keyed by doc_id: accumulate document frequencies (each unique
        # term counts once per document) and total token length. The previous version did
        # an O(n^2) reverse lookup by set-equality, which silently double-counted length
        # for any two documents that happened to share the same unique-term set.
        df: dict[str, int] = Counter()
        for doc_id, tokens in self._doc_terms.items():
            for token in tokens:
                df[token] += 1
            total_length += len(self._tokenize(self._documents[doc_id]))

        # Calculate IDF values (log of inverse document frequency)
        for token in df:
            self._idf[token] = math.log((self._total_docs - df[token] + 0.5) / (df[token] + 0.5) + 1.0)

        self._avg_doc_length = total_length / self._total_docs if self._total_docs > 0 else 0

    def score_query(self, doc_id: str, query: str) -> float:
        """Score a document against a query using BM25 formula.
        
        Args:
            doc_id: Document to score
            query: Query text
            
        Returns:
            BM25 score (higher is better)
        """
        if doc_id not in self._documents or not query:
            return 0.0

        query_tokens = self._tokenize(query)
        doc_text = self._documents[doc_id]
        doc_tokens = self._tokenize(doc_text)
        doc_length = len(doc_tokens)

        score = 0.0
        for token in query_tokens:
            if token not in self._idf:
                continue

            tf = doc_tokens.count(token)
            idf = self._idf[token]

            # BM25 formula
            numerator = idf * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / max(1, self._avg_doc_length)))
            score += numerator / denominator

        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search for documents matching a query.
        
        Args:
            query: Query text
            top_k: Maximum number of results to return
            
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        if not query or not self._documents:
            return []

        scores: list[tuple[str, float]] = []
        for doc_id in self._documents:
            score = self.score_query(doc_id, query)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:max(1, top_k)]

    def get_document(self, doc_id: str) -> str | None:
        """Retrieve original document text by ID."""
        return self._documents.get(doc_id)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace-based tokenization with lowercase normalization."""
        if not text:
            return []
        return [token.lower() for token in text.split() if token.strip()]


class SparseVectorStore:
    """Abstract interface for sparse token-based retrieval."""

    async def index_documents(self, documents: list[dict[str, Any]]) -> None:
        """Index documents for sparse retrieval.
        
        Args:
            documents: List of dicts with 'id' and 'text' keys
        """
        raise NotImplementedError

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for documents using sparse (token-based) retrieval.
        
        Args:
            query: Query text
            top_k: Maximum results to return
            
        Returns:
            List of dicts with 'id', 'text', 'score', and 'metadata' keys
        """
        raise NotImplementedError


class BM25VectorStore(SparseVectorStore):
    """BM25-based sparse vector store for token-based retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25 sparse store.
        
        Args:
            k1: BM25 term frequency saturation (default 1.5)
            b: BM25 length normalization (default 0.75)
        """
        self._indexer = BM25Indexer(k1=k1, b=b)
        self._metadata: dict[str, dict[str, Any]] = {}
        self._indexed = False

    async def index_documents(self, documents: list[dict[str, Any]]) -> None:
        """Index documents for sparse retrieval.
        
        Args:
            documents: List of dicts with 'id', 'text', and optional 'metadata' keys
        """
        if not documents:
            return

        for doc in documents:
            doc_id = str(doc.get("id", ""))
            text = str(doc.get("text", ""))
            metadata = doc.get("metadata", {})

            if doc_id and text:
                self._indexer.add_document(doc_id, text)
                self._metadata[doc_id] = metadata

        self._indexer.build_index()
        self._indexed = True
        logger.info("Indexed %d documents in BM25 sparse store", len(documents))

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for documents using BM25.
        
        Args:
            query: Query text
            top_k: Maximum results to return
            
        Returns:
            List of result dicts with id, text, score, and metadata
        """
        if not self._indexed or not query:
            return []

        results: list[dict[str, Any]] = []
        for doc_id, score in self._indexer.search(query, top_k=top_k):
            text = self._indexer.get_document(doc_id)
            metadata = self._metadata.get(doc_id, {})
            results.append({
                "id": doc_id,
                "text": text,
                "score": score,
                "metadata": metadata,
            })

        return results


__all__ = [
    "BM25Indexer",
    "SparseVectorStore",
    "BM25VectorStore",
]
