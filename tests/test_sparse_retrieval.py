"""Tests for Phase 5: Dense + Sparse Hybrid Retrieval.

This test suite validates:
- BM25 sparse indexing and retrieval
- Dense and sparse score normalization
- Hybrid retrieval with configurable alpha
- Metadata preservation across retrieval methods
- Edge cases and error handling
"""

from __future__ import annotations

import pytest

from backend.modules.embeddings import InMemoryEmbeddingProvider
from backend.modules.retrieval import (
    InMemoryVectorStore,
    OrchestratedHybridRetriever,
    RetrievedDocument,
)
from backend.modules.sparse_retrieval import (
    BM25Indexer,
    BM25VectorStore,
)


class TestBM25Indexer:
    """Test BM25 sparse indexing and scoring."""

    def test_bm25_indexer_initialization(self):
        """Test BM25 indexer creation with default parameters."""
        indexer = BM25Indexer(k1=1.5, b=0.75)
        assert indexer.k1 == 1.5
        assert indexer.b == 0.75

    def test_bm25_add_and_index_documents(self):
        """Test adding documents and building index."""
        indexer = BM25Indexer()
        indexer.add_document("doc1", "नई दिल्ली भारत की राजधानी है।")
        indexer.add_document("doc2", "भारत एक बड़ा देश है।")
        indexer.add_document("doc3", "दिल्ली भारत का एक प्रमुख शहर है।")
        
        assert len(indexer._documents) == 3
        indexer.build_index()
        assert indexer._total_docs == 3
        assert len(indexer._idf) > 0

    def test_bm25_scoring(self):
        """Test BM25 scoring for a query."""
        indexer = BM25Indexer()
        indexer.add_document("doc1", "India is a country")
        indexer.add_document("doc2", "India capital is New Delhi")
        indexer.build_index()
        
        score1 = indexer.score_query("doc1", "India")
        score2 = indexer.score_query("doc2", "India capital")
        
        assert score1 > 0
        assert score2 > score1  # doc2 matches better (both terms present)

    def test_bm25_search(self):
        """Test BM25 search retrieval."""
        indexer = BM25Indexer()
        indexer.add_document("doc1", "The capital of India is New Delhi")
        indexer.add_document("doc2", "India is a large country")
        indexer.add_document("doc3", "New York is a city in the USA")
        indexer.build_index()
        
        results = indexer.search("India capital", top_k=2)
        
        assert len(results) <= 2
        assert results[0][0] == "doc1"  # Best match
        assert results[0][1] > 0

    def test_bm25_empty_query(self):
        """Test BM25 with empty query."""
        indexer = BM25Indexer()
        indexer.add_document("doc1", "Some text")
        indexer.build_index()
        
        results = indexer.search("", top_k=5)
        assert results == []

    def test_bm25_empty_index(self):
        """Test BM25 search on empty index."""
        indexer = BM25Indexer()
        results = indexer.search("query", top_k=5)
        assert results == []

    def test_bm25_multilingual_indexing(self):
        """Test BM25 indexing with multilingual text."""
        indexer = BM25Indexer()
        indexer.add_document("hi1", "नई दिल्ली भारत की राजधानी है")
        indexer.add_document("en1", "New Delhi is the capital of India")
        indexer.add_document("hi2", "भारत एशिया का एक देश है")
        indexer.build_index()
        
        results_hi = indexer.search("दिल्ली", top_k=2)
        results_en = indexer.search("Delhi", top_k=2)
        
        assert len(results_hi) > 0
        assert len(results_en) > 0


class TestBM25VectorStore:
    """Test BM25 sparse vector store implementation."""

    @pytest.mark.asyncio
    async def test_bm25_vectorstore_index_documents(self):
        """Test indexing documents in BM25 vector store."""
        store = BM25VectorStore()
        docs = [
            {"id": "1", "text": "India has many states", "metadata": {"lang": "en"}},
            {"id": "2", "text": "New Delhi is the capital", "metadata": {"lang": "en"}},
        ]
        
        await store.index_documents(docs)
        assert store._indexed is True
        assert len(store._metadata) == 2

    @pytest.mark.asyncio
    async def test_bm25_vectorstore_search(self):
        """Test searching in BM25 vector store."""
        store = BM25VectorStore()
        docs = [
            {"id": "1", "text": "India is a country", "metadata": {"source": "wiki"}},
            {"id": "2", "text": "New Delhi is the capital of India", "metadata": {"source": "wiki"}},
            {"id": "3", "text": "India has many languages", "metadata": {"source": "wiki"}},
        ]
        await store.index_documents(docs)
        
        results = await store.search("India capital", top_k=2)
        
        assert len(results) == 2
        assert results[0]["id"] == "2"  # Best match
        assert results[0]["score"] > results[1]["score"]
        assert "metadata" in results[0]

    @pytest.mark.asyncio
    async def test_bm25_vectorstore_metadata_preservation(self):
        """Test that metadata is preserved through BM25 retrieval."""
        store = BM25VectorStore()
        docs = [
            {
                "id": "chunk-1",
                "text": "Sample text here",
                "metadata": {
                    "source_id": "doc-1",
                    "language": "en",
                    "query_type": "factual",
                },
            }
        ]
        await store.index_documents(docs)
        
        results = await store.search("sample", top_k=1)
        
        assert len(results) == 1
        assert results[0]["metadata"]["source_id"] == "doc-1"
        assert results[0]["metadata"]["language"] == "en"

    @pytest.mark.asyncio
    async def test_bm25_vectorstore_empty_search(self):
        """Test BM25 search on empty store."""
        store = BM25VectorStore()
        results = await store.search("query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_bm25_vectorstore_before_indexing(self):
        """Test search before indexing raises no error."""
        store = BM25VectorStore()
        results = await store.search("query", top_k=5)
        assert results == []


class TestOrchestratedHybridRetriever:
    """Test orchestrated hybrid retrieval combining dense and sparse."""

    @pytest.mark.asyncio
    async def test_hybrid_retriever_initialization(self):
        """Test hybrid retriever creation with alpha parameter."""
        retriever = OrchestratedHybridRetriever(alpha=0.5)
        assert retriever.alpha == 0.5

    @pytest.mark.asyncio
    async def test_hybrid_retriever_alpha_validation(self):
        """Test alpha parameter validation."""
        # Valid alphas
        OrchestratedHybridRetriever(alpha=0.0)
        OrchestratedHybridRetriever(alpha=0.5)
        OrchestratedHybridRetriever(alpha=1.0)
        
        # Invalid alphas
        with pytest.raises(ValueError, match="alpha must be in range"):
            OrchestratedHybridRetriever(alpha=-0.1)
        
        with pytest.raises(ValueError, match="alpha must be in range"):
            OrchestratedHybridRetriever(alpha=1.1)

    @pytest.mark.asyncio
    async def test_hybrid_retriever_dense_only(self):
        """Test hybrid retriever with alpha=1.0 (dense only)."""
        # Setup dense store
        dense_store = InMemoryVectorStore()
        docs = [
            {"id": "1", "text": "India capital", "metadata": {"source": "s1"}},
            {"id": "2", "text": "India country", "metadata": {"source": "s2"}},
        ]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        await dense_store.upsert(docs, embeddings)
        
        retriever = OrchestratedHybridRetriever(dense_store=dense_store, alpha=1.0)
        results = await retriever.retrieve("capital", top_k=2)
        
        assert len(results) > 0
        assert all(r.method == "hybrid" for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_retriever_sparse_only(self):
        """Test hybrid retriever with alpha=0.0 (sparse only)."""
        # Setup sparse store
        sparse_store = BM25VectorStore()
        docs = [
            {"id": "1", "text": "India is a capital country", "metadata": {"source": "s1"}},
            {"id": "2", "text": "India is a large nation", "metadata": {"source": "s2"}},
        ]
        await sparse_store.index_documents(docs)
        
        retriever = OrchestratedHybridRetriever(sparse_store=sparse_store, alpha=0.0)
        results = await retriever.retrieve("India capital", top_k=2)
        
        assert len(results) > 0
        assert all(r.method == "hybrid" for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_retriever_balanced(self):
        """Test hybrid retriever with alpha=0.5 (balanced)."""
        # Setup both stores
        dense_store = InMemoryVectorStore()
        docs = [
            {"id": "1", "text": "India capital", "metadata": {"lang": "en"}},
            {"id": "2", "text": "India country", "metadata": {"lang": "en"}},
        ]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        await dense_store.upsert(docs, embeddings)
        
        sparse_store = BM25VectorStore()
        await sparse_store.index_documents(docs)
        
        retriever = OrchestratedHybridRetriever(
            dense_store=dense_store,
            sparse_store=sparse_store,
            alpha=0.5,
        )
        results = await retriever.retrieve("India capital", top_k=2)
        
        assert len(results) > 0
        assert all(r.metadata.get("alpha") == 0.5 for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_retriever_with_embedding_provider(self):
        """Test hybrid retriever using embedding provider for queries."""
        # Setup stores
        dense_store = InMemoryVectorStore()
        docs = [
            {"id": "1", "text": "India capital Delhi", "metadata": {"source": "s1"}},
            {"id": "2", "text": "India is large", "metadata": {"source": "s2"}},
        ]
        embeddings = [[1.0, 0.5, 0.2], [0.2, 0.3, 1.0]]
        await dense_store.upsert(docs, embeddings)
        
        sparse_store = BM25VectorStore()
        await sparse_store.index_documents(docs)
        
        embedding_provider = InMemoryEmbeddingProvider(dimension=3)
        
        retriever = OrchestratedHybridRetriever(
            dense_store=dense_store,
            sparse_store=sparse_store,
            alpha=0.6,
            embedding_provider=embedding_provider,
        )
        results = await retriever.retrieve("India capital", top_k=2)
        
        assert len(results) > 0
        assert all(r.method == "hybrid" for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_retriever_metadata_preservation(self):
        """Test that metadata is preserved in hybrid retrieval."""
        dense_store = InMemoryVectorStore()
        docs = [
            {
                "id": "chunk-1",
                "text": "India capital",
                "metadata": {"query_id": "q1", "language": "en", "source_id": "s1"},
            }
        ]
        embeddings = [[1.0, 0.0, 0.0]]
        await dense_store.upsert(docs, embeddings)
        
        sparse_store = BM25VectorStore()
        await sparse_store.index_documents(docs)
        
        retriever = OrchestratedHybridRetriever(
            dense_store=dense_store,
            sparse_store=sparse_store,
            alpha=0.5,
        )
        results = await retriever.retrieve("India", top_k=1)
        
        assert len(results) == 1
        assert results[0].metadata.get("query_id") == "q1"
        assert results[0].metadata.get("language") == "en"
        assert results[0].metadata.get("source_id") == "s1"

    @pytest.mark.asyncio
    async def test_hybrid_retriever_empty_query(self):
        """Test hybrid retriever with empty query."""
        dense_store = InMemoryVectorStore()
        sparse_store = BM25VectorStore()
        
        retriever = OrchestratedHybridRetriever(
            dense_store=dense_store,
            sparse_store=sparse_store,
            alpha=0.5,
        )
        results = await retriever.retrieve("", top_k=5)
        
        assert results == []

    @pytest.mark.asyncio
    async def test_hybrid_retriever_no_stores(self):
        """Test hybrid retriever with no stores."""
        retriever = OrchestratedHybridRetriever(alpha=0.5)
        results = await retriever.retrieve("query", top_k=5)
        
        assert results == []

    @pytest.mark.asyncio
    async def test_hybrid_retriever_ranking(self):
        """Test that hybrid retriever properly ranks results."""
        dense_store = InMemoryVectorStore()
        docs = [
            {"id": "1", "text": "India capital Delhi", "metadata": {}},
            {"id": "2", "text": "India country", "metadata": {}},
            {"id": "3", "text": "New Delhi city", "metadata": {}},
        ]
        embeddings = [[1.0, 0.8, 0.0], [0.0, 1.0, 0.2], [0.2, 0.0, 1.0]]
        await dense_store.upsert(docs, embeddings)
        
        sparse_store = BM25VectorStore()
        await sparse_store.index_documents(docs)
        
        retriever = OrchestratedHybridRetriever(
            dense_store=dense_store,
            sparse_store=sparse_store,
            alpha=0.5,
        )
        results = await retriever.retrieve("India capital", top_k=3)
        
        # Verify results are ranked by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    @pytest.mark.asyncio
    async def test_hybrid_retriever_top_k_limit(self):
        """Test that top_k parameter limits results."""
        dense_store = InMemoryVectorStore()
        docs = [
            {"id": str(i), "text": f"Document {i}", "metadata": {}}
            for i in range(10)
        ]
        embeddings = [[float(i), 0.0, 0.0] for i in range(10)]
        await dense_store.upsert(docs, embeddings)
        
        sparse_store = BM25VectorStore()
        await sparse_store.index_documents(docs)
        
        retriever = OrchestratedHybridRetriever(
            dense_store=dense_store,
            sparse_store=sparse_store,
            alpha=0.5,
        )
        results = await retriever.retrieve("Document", top_k=3)
        
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_hybrid_retriever_multilingual(self):
        """Test hybrid retriever with multilingual content."""
        dense_store = InMemoryVectorStore()
        docs = [
            {
                "id": "hi1",
                "text": "नई दिल्ली भारत की राजधानी है",
                "metadata": {"language": "hi"},
            },
            {
                "id": "en1",
                "text": "New Delhi is the capital of India",
                "metadata": {"language": "en"},
            },
        ]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        await dense_store.upsert(docs, embeddings)
        
        sparse_store = BM25VectorStore()
        await sparse_store.index_documents(docs)
        
        retriever = OrchestratedHybridRetriever(
            dense_store=dense_store,
            sparse_store=sparse_store,
            alpha=0.5,
        )
        results_hi = await retriever.retrieve("दिल्ली", top_k=2)
        results_en = await retriever.retrieve("Delhi", top_k=2)
        
        assert len(results_hi) > 0
        assert len(results_en) > 0

    @pytest.mark.asyncio
    async def test_hybrid_retriever_alpha_variations(self):
        """Test hybrid retriever with different alpha values."""
        dense_store = InMemoryVectorStore()
        docs = [
            {"id": "1", "text": "Text one", "metadata": {}},
            {"id": "2", "text": "Text two", "metadata": {}},
        ]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        await dense_store.upsert(docs, embeddings)
        
        sparse_store = BM25VectorStore()
        await sparse_store.index_documents(docs)
        
        # Test different alpha values
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            retriever = OrchestratedHybridRetriever(
                dense_store=dense_store,
                sparse_store=sparse_store,
                alpha=alpha,
            )
            results = await retriever.retrieve("Text", top_k=2)
            
            assert len(results) <= 2
            assert all(r.metadata.get("alpha") == alpha for r in results)


class TestHybridRetrievalIntegration:
    """Integration tests for complete hybrid retrieval workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_hybrid_retrieval(self):
        """Test end-to-end workflow: index → retrieve → rank."""
        # Create dense store
        dense_store = InMemoryVectorStore()
        documents = [
            {
                "id": "chunk-1",
                "text": "India is a large country with many states",
                "metadata": {"source": "wiki", "lang": "en"},
            },
            {
                "id": "chunk-2",
                "text": "The capital of India is New Delhi",
                "metadata": {"source": "wiki", "lang": "en"},
            },
            {
                "id": "chunk-3",
                "text": "Delhi has many historical monuments",
                "metadata": {"source": "wiki", "lang": "en"},
            },
        ]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        await dense_store.upsert(documents, embeddings)
        
        # Create sparse store
        sparse_store = BM25VectorStore()
        await sparse_store.index_documents(documents)
        
        # Retrieve with hybrid retriever
        retriever = OrchestratedHybridRetriever(
            dense_store=dense_store,
            sparse_store=sparse_store,
            alpha=0.6,
        )
        results = await retriever.retrieve("India capital Delhi", top_k=3)
        
        assert len(results) > 0
        assert results[0].method == "hybrid"
        assert all(r.chunk_id in ["chunk-1", "chunk-2", "chunk-3"] for r in results)
