"""FlashRank / Lightweight candidate reranker implementation.

Re-ranks retrieved candidates from hybrid retrieval to prioritize exact semantic
relevance before grounded answer generation.
"""

from __future__ import annotations

import logging
from typing import Any

from modules.reranking import Reranker
from modules.retrieval import RetrievedDocument

logger = logging.getLogger(__name__)


class FlashRankReranker(Reranker):
    """Lightweight candidate reranker with fallback logic."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2") -> None:
        self.model_name = model_name
        self._ranker = None

    async def rerank(
        self, query: str, documents: list[RetrievedDocument], top_k: int = 5
    ) -> list[RetrievedDocument]:
        """Rerank candidate retrieved documents.
        
        Args:
            query: User search text
            documents: List of RetrievedDocument candidates from hybrid retrieval
            top_k: Number of reranked candidates to return
            
        Returns:
            Reranked list of RetrievedDocument objects
        """
        if not query or not documents:
            return []

        try:
            # Try importing flashrank if installed
            from flashrank import Ranker, RerankRequest
            if self._ranker is None:
                self._ranker = Ranker(model_name=self.model_name)

            passages = [
                {"id": doc.chunk_id, "text": doc.text, "metadata": doc.metadata}
                for doc in documents
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(rerank_request)

            reranked_docs: list[RetrievedDocument] = []
            doc_map = {doc.chunk_id: doc for doc in documents}

            for res in results[:top_k]:
                doc_id = res["id"]
                orig_doc = doc_map.get(doc_id)
                if orig_doc:
                    reranked_docs.append(
                        RetrievedDocument(
                            chunk_id=orig_doc.chunk_id,
                            text=orig_doc.text,
                            score=float(res.get("score", orig_doc.score)),
                            method="reranked",
                            metadata={**orig_doc.metadata, "rerank_score": float(res.get("score", 0.0))},
                        )
                    )
            return reranked_docs
        except Exception as exc:
            logger.debug("FlashRank not available or failed (%s), using native score ordering fallback.", exc)
            sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
            return sorted_docs[:max(1, top_k)]
