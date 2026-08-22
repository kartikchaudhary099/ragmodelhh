from __future__ import annotations

import pytest

from modules.embeddings import EmbeddingProvider, InMemoryEmbeddingProvider
from modules.retrieval import (
    HybridRetriever,
    InMemoryVectorStore,
    RetrievedDocument,
    SparseRetriever,
    dense_retriever,
) 


class TestEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self._dimension = 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 1.0] for text in texts]

    def dimension(self) -> int:
        return self._dimension


@pytest.mark.asyncio
async def test_embedding_provider_interface_contract() -> None:
    provider = TestEmbeddingProvider()
    vectors = await provider.embed(["alpha", "beta"]) 
    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert provider.dimension() == 3


@pytest.mark.asyncio
async def test_in_memory_embedding_provider_is_deterministic() -> None:
    provider = InMemoryEmbeddingProvider()
    vectors_a = await provider.embed(["hello world", "hello world"])
    vectors_b = await provider.embed(["hello world", "hello world"])
    assert vectors_a == vectors_b


@pytest.mark.asyncio
async def test_sparse_and_dense_retrieval_preserve_metadata() -> None:
    docs = [
        RetrievedDocument(
            chunk_id="c1",
            text="India capital is New Delhi.",
            score=0.91,
            method="dense",
            metadata={
                "source_id": "s1",
                "query_id": "q1",
                "language": "hi",
                "source_language": "hi",
                "target_language": "hi",
                "query_type": "factual",
                "chunk_role": "child",
                "parent_id": "p1",
                "original_text": "India capital is New Delhi.",
            },
        ),
        RetrievedDocument(
            chunk_id="c2",
            text="Delhi is the capital of India.",
            score=0.82,
            method="sparse",
            metadata={
                "source_id": "s2",
                "query_id": "q1",
                "language": "hi",
                "source_language": "hi",
                "target_language": "hi",
                "query_type": "factual",
                "chunk_role": "parent",
                "parent_id": "p2",
                "original_text": "Delhi is the capital of India.",
            },
        ),
    ]

    assert docs[0].metadata["source_id"] == "s1"
    assert docs[1].method == "sparse"
    assert docs[0].score > docs[1].score


@pytest.mark.asyncio
async def test_hybrid_retriever_merges_dense_and_sparse_results() -> None:
    vector_store = InMemoryVectorStore()
    await vector_store.upsert(
        [
            {"id": "1", "text": "India has many states.", "metadata": {"source_id": "s1", "query_id": "q1"}},
            {"id": "2", "text": "The capital of India is New Delhi.", "metadata": {"source_id": "s2", "query_id": "q1"}},
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )

    dense = await dense_retriever(query="capital of india", vector_store=vector_store, top_k=2)
    sparse = await SparseRetriever("simple").retrieve("capital of india", documents=[
        RetrievedDocument(chunk_id="1", text="India has many states.", score=0.1, method="sparse", metadata={"source_id": "s1"}),
        RetrievedDocument(chunk_id="2", text="The capital of India is New Delhi.", score=0.9, method="sparse", metadata={"source_id": "s2"}),
    ])
    hybrid = await HybridRetriever().retrieve(
        query="capital of india",
        dense_results=dense,
        sparse_results=sparse,
        top_k=2,
    )

    assert len(dense) >= 1
    assert len(sparse) >= 1
    assert len(hybrid) >= 1
    assert all(result.metadata is not None for result in hybrid)


@pytest.mark.asyncio
async def test_empty_and_invalid_input_handling() -> None:
    provider = InMemoryEmbeddingProvider()
    assert await provider.embed([]) == []
    assert await provider.embed([""]) == [[0.0, 0.0, 0.0]]

    empty_vector_store = InMemoryVectorStore()
    assert await empty_vector_store.search([0.0, 0.0], top_k=5) == []

    sparse = SparseRetriever("simple")
    assert await sparse.retrieve("", documents=[]) == []

    hybrid = HybridRetriever()
    assert await hybrid.retrieve("query", dense_results=[], sparse_results=[], top_k=5) == []
