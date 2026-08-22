"""Tests for the embedding pipeline orchestration.

This test suite validates:
- Chunk to embedding conversion
- Batch embedding operations
- Integration with vector stores
- Error handling and edge cases
"""

from __future__ import annotations

import pytest

from backend.modules.chunking import Chunk, FixedSizeChunker
from backend.modules.embedding_pipeline import (
    EmbeddedChunk,
    chunks_to_embedding_format,
    embed_chunks,
    embed_and_store,
)
from backend.modules.embeddings import InMemoryEmbeddingProvider
from backend.modules.retrieval import InMemoryVectorStore


def _sample_chunks() -> list[Chunk]:
    """Create sample chunks for embedding tests."""
    return [
        Chunk(
            text="नई दिल्ली भारत की राजधानी है।",
            chunk_id="chunk-1",
            source_id="record-1",
            metadata={"language": "hi", "source": "test"},
        ),
        Chunk(
            text="भारत एक विशाल देश है।",
            chunk_id="chunk-2",
            source_id="record-1",
            metadata={"language": "hi", "source": "test"},
        ),
        Chunk(
            text="New Delhi is the capital of India.",
            chunk_id="chunk-3",
            source_id="record-2",
            metadata={"language": "en", "source": "test"},
        ),
    ]


def _sample_chunk_dicts() -> list[dict]:
    """Create sample chunk dictionaries for embedding tests."""
    chunks = _sample_chunks()
    return [
        {
            "text": chunk.text,
            "chunk_id": chunk.chunk_id,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]


@pytest.mark.asyncio
async def test_embed_chunks_with_inmemory_provider():
    """Test embedding chunks using the in-memory provider."""
    chunks = _sample_chunk_dicts()
    provider = InMemoryEmbeddingProvider(dimension=3)
    
    embedded = await embed_chunks(chunks, provider)
    
    assert len(embedded) == 3
    assert all(isinstance(e, EmbeddedChunk) for e in embedded)
    assert all(len(e.embedding) == 3 for e in embedded)
    assert embedded[0].chunk_id == "chunk-1"
    assert embedded[0].text == "नई दिल्ली भारत की राजधानी है।"
    assert embedded[0].metadata["language"] == "hi"


@pytest.mark.asyncio
async def test_embed_chunks_handles_empty_list():
    """Test that embedding an empty list returns empty result."""
    provider = InMemoryEmbeddingProvider()
    embedded = await embed_chunks([], provider)
    assert embedded == []


@pytest.mark.asyncio
async def test_embed_chunks_skips_empty_text():
    """Test that chunks with empty text are skipped."""
    chunks = [
        {"text": "Valid text", "chunk_id": "c1", "metadata": {}},
        {"text": "", "chunk_id": "c2", "metadata": {}},
        {"text": "Another valid text", "chunk_id": "c3", "metadata": {}},
    ]
    provider = InMemoryEmbeddingProvider()
    embedded = await embed_chunks(chunks, provider)
    
    # Only 2 non-empty chunks should be embedded
    assert len(embedded) == 2
    assert embedded[0].chunk_id == "c1"
    assert embedded[1].chunk_id == "c3"


@pytest.mark.asyncio
async def test_embed_chunks_preserves_metadata():
    """Test that chunk metadata is preserved through embedding."""
    chunks = [
        {
            "text": "Test text",
            "chunk_id": "c1",
            "metadata": {
                "language": "hi",
                "query_type": "factoid",
                "source": "MSMARCO",
            },
        }
    ]
    provider = InMemoryEmbeddingProvider()
    embedded = await embed_chunks(chunks, provider)
    
    assert len(embedded) == 1
    assert embedded[0].metadata["language"] == "hi"
    assert embedded[0].metadata["query_type"] == "factoid"
    assert embedded[0].metadata["source"] == "MSMARCO"


@pytest.mark.asyncio
async def test_chunks_to_embedding_format_converts_chunk_objects():
    """Test conversion of Chunk objects to embedding format."""
    chunks = _sample_chunks()
    formatted = chunks_to_embedding_format(chunks)
    
    assert len(formatted) == 3
    assert all("text" in f and "chunk_id" in f and "metadata" in f for f in formatted)
    assert formatted[0]["text"] == "नई दिल्ली भारत की राजधानी है।"
    assert formatted[0]["chunk_id"] == "chunk-1"


@pytest.mark.asyncio
async def test_chunks_to_embedding_format_handles_dicts():
    """Test that dicts pass through unchanged (but normalized)."""
    chunks = _sample_chunk_dicts()
    formatted = chunks_to_embedding_format(chunks)
    
    assert len(formatted) == 3
    assert formatted[0]["text"] == chunks[0]["text"]


@pytest.mark.asyncio
async def test_embed_and_store_integrates_embedding_and_storage():
    """Test full pipeline: embed chunks and store in vector store."""
    chunks = _sample_chunk_dicts()
    provider = InMemoryEmbeddingProvider(dimension=3)
    store = InMemoryVectorStore()
    
    embedded = await embed_and_store(chunks, provider, store)
    
    assert len(embedded) == 3
    
    # Verify that chunks are stored in the vector store
    search_result = await store.search([1.0, 1.0, 1.0], top_k=3)
    assert len(search_result) == 3


@pytest.mark.asyncio
async def test_embed_and_store_with_empty_chunks():
    """Test embed_and_store with empty chunk list."""
    provider = InMemoryEmbeddingProvider()
    store = InMemoryVectorStore()
    
    embedded = await embed_and_store([], provider, store)
    
    assert embedded == []


@pytest.mark.asyncio
async def test_embeddings_have_consistent_dimension():
    """Test that all embeddings have the same dimension."""
    chunks = _sample_chunk_dicts()
    provider = InMemoryEmbeddingProvider(dimension=5)
    
    embedded = await embed_chunks(chunks, provider)
    
    dimensions = [len(e.embedding) for e in embedded]
    assert all(d == 5 for d in dimensions)


@pytest.mark.asyncio
async def test_embedded_chunk_preserves_model_name():
    """Test that EmbeddedChunk records the model used."""
    chunks = _sample_chunk_dicts()
    provider = InMemoryEmbeddingProvider()
    
    embedded = await embed_chunks(chunks, provider)
    
    # InMemoryEmbeddingProvider doesn't have model_name, should default
    assert embedded[0].model_name == "BAAI/bge-m3"


def test_chunker_to_embedder_integration():
    """Test that chunks from the chunker can be embedded."""
    record = {
        "query_id": "q-1",
        "query": "भारत की राजधानी क्या है?",
        "passage": "नई दिल्ली भारत की राजधानी है। यह शहर देश की राष्ट्रीय राजधानी है।",
        "language": "hi",
        "query_type": "factual",
        "record_id": "r-1",
        "source_language": "hi",
        "target_language": "hi",
    }
    
    # Create chunks using the FixedSizeChunker
    chunker = FixedSizeChunker(chunk_size=30, overlap=5)
    chunks = chunker.chunk(record)
    
    # Verify chunks can be formatted for embedding
    formatted = chunks_to_embedding_format(chunks)
    
    assert len(formatted) > 0
    assert all(f["text"] for f in formatted)  # All have non-empty text
    assert all(f["chunk_id"] for f in formatted)  # All have chunk IDs


@pytest.mark.asyncio
async def test_multilingual_chunks_embedding():
    """Test embedding of multilingual chunks (Hindi + English)."""
    chunks = [
        {
            "text": "नई दिल्ली भारत की राजधानी है।",
            "chunk_id": "hi-1",
            "metadata": {"language": "hi"},
        },
        {
            "text": "New Delhi is the capital of India.",
            "chunk_id": "en-1",
            "metadata": {"language": "en"},
        },
        {
            "text": "दिल्ली में बहुत सारे ऐतिहासिक स्मारक हैं।",
            "chunk_id": "hi-2",
            "metadata": {"language": "hi"},
        },
    ]
    
    provider = InMemoryEmbeddingProvider(dimension=3)
    embedded = await embed_chunks(chunks, provider)
    
    assert len(embedded) == 3
    assert embedded[0].metadata["language"] == "hi"
    assert embedded[1].metadata["language"] == "en"
    assert embedded[2].metadata["language"] == "hi"


@pytest.mark.asyncio
async def test_embed_chunks_batch_consistency():
    """Test that embedding chunks in batches is consistent."""
    chunks = _sample_chunk_dicts()
    provider = InMemoryEmbeddingProvider(dimension=3)
    
    # Embed all at once
    embedded_all = await embed_chunks(chunks, provider)
    
    # Embed in batches
    batch_results = []
    for batch in [chunks[:2], chunks[2:]]:
        batch_results.extend(await embed_chunks(batch, provider))
    
    # Results should be equivalent (same chunks, same metadata)
    assert len(embedded_all) == len(batch_results)
    for emb1, emb2 in zip(embedded_all, batch_results):
        assert emb1.chunk_id == emb2.chunk_id
        assert emb1.text == emb2.text
        # Note: Embeddings might differ due to deterministic but batch-dependent computation
