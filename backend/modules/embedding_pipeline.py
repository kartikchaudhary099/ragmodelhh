"""Embedding pipeline orchestration for chunked documents.

This module provides the integration layer between chunking and retrieval:
- Convert chunk objects to embeddable documents
- Apply embeddings using the configured provider (BGE-M3 by default)
- Store embeddings in a vector store for retrieval

The pipeline is modular: embedding provider and vector store can be swapped without
changing this orchestration layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """A chunk that has been embedded and indexed for retrieval."""

    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]
    model_name: str = "BAAI/bge-m3"


async def embed_chunks(
    chunks: list[dict[str, Any]],
    embedding_provider: Any,
) -> list[EmbeddedChunk]:
    """Embed a list of chunk objects using the configured provider.
    
    Args:
        chunks: List of chunk dictionaries (or Chunk objects converted to dict)
        embedding_provider: EmbeddingProvider instance (e.g., BGE3EmbeddingProvider)
        
    Returns:
        List of EmbeddedChunk objects with embeddings
    """
    if not chunks:
        return []

    texts_to_embed = []
    chunk_metadata = []
    
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            # Accept both "chunk_id" (pipeline/test convention) and "id" (seeder convention)
            # so seeded corpus chunks keep distinct ids instead of collapsing to "".
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or ""
            metadata = chunk.get("metadata", {})
        else:
            text = getattr(chunk, "text", "")
            chunk_id = getattr(chunk, "chunk_id", "")
            metadata = getattr(chunk, "metadata", {})
        
        if text:
            texts_to_embed.append(text)
            chunk_metadata.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
            })
    
    if not texts_to_embed:
        logger.warning("No non-empty chunk texts to embed")
        return []
    
    embeddings = await embedding_provider.embed(texts_to_embed)
    
    if len(embeddings) != len(chunk_metadata):
        raise ValueError(
            f"Embedding provider returned {len(embeddings)} embeddings "
            f"but {len(chunk_metadata)} chunks were provided."
        )
    
    embedded_chunks: list[EmbeddedChunk] = []
    for metadata, embedding in zip(chunk_metadata, embeddings, strict=True):
        embedded_chunks.append(
            EmbeddedChunk(
                chunk_id=metadata["chunk_id"],
                text=metadata["text"],
                embedding=embedding,
                metadata=metadata["metadata"],
                model_name=getattr(embedding_provider, "model_name", "BAAI/bge-m3"),
            )
        )
    
    logger.info("Embedded %d chunks using %s", len(embedded_chunks), 
                getattr(embedding_provider, "model_name", "unknown"))
    return embedded_chunks


async def embed_and_store(
    chunks: list[dict[str, Any]],
    embedding_provider: Any,
    vector_store: Any,
) -> list[EmbeddedChunk]:
    """Embed chunks and immediately store them in the vector store.
    
    Args:
        chunks: List of chunk objects
        embedding_provider: EmbeddingProvider instance
        vector_store: VectorStore instance (e.g., InMemoryVectorStore)
        
    Returns:
        List of EmbeddedChunk objects
    """
    embedded = await embed_chunks(chunks, embedding_provider)
    
    if embedded:
        documents = [
            {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in embedded
        ]
        embeddings = [chunk.embedding for chunk in embedded]
        await vector_store.upsert(documents, embeddings)
        logger.info("Stored %d embedded chunks in vector store", len(embedded))
    
    return embedded


def chunks_to_embedding_format(chunks: list[Any]) -> list[dict[str, Any]]:
    """Convert Chunk objects or native records to dict format for embedding pipeline.
    
    This is a helper to normalize various chunk representations into a standard
    dictionary format suitable for the embedding pipeline.
    
    Args:
        chunks: List of Chunk objects or dictionaries
        
    Returns:
        List of dictionaries with 'text', 'chunk_id', and 'metadata' keys
    """
    formatted: list[dict[str, Any]] = []
    
    for chunk in chunks:
        if isinstance(chunk, dict):
            formatted.append({
                "text": chunk.get("text", ""),
                "chunk_id": chunk.get("chunk_id") or chunk.get("id") or "",
                "metadata": chunk.get("metadata", {}),
            })
        else:
            formatted.append({
                "text": getattr(chunk, "text", ""),
                "chunk_id": getattr(chunk, "chunk_id", ""),
                "metadata": getattr(chunk, "metadata", {}),
            })
    
    return formatted


__all__ = [
    "EmbeddedChunk",
    "embed_chunks",
    "embed_and_store",
    "chunks_to_embedding_format",
]
