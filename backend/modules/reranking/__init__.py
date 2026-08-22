"""
Reranking module.

Future implementations will reorder retrieved candidates by relevance.
Swap models by implementing Reranker.
"""

from abc import ABC, abstractmethod

from modules.retrieval import RetrievedDocument


class Reranker(ABC):
    """Abstract interface for reranking strategies."""

    @abstractmethod
    async def rerank(
        self, query: str, documents: list[RetrievedDocument], top_k: int = 5
    ) -> list[RetrievedDocument]:
        """Rerank retrieved documents by relevance to the query."""
        ...
