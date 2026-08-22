"""Sample data seeder for ThinkZen demo.

Loads demo documents from data/samples/demo_docs.json, chunks them, and indexes
them into sparse (BM25) and dense (InMemory / Qdrant) vector stores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from modules.chunking import SentenceChunker
from modules.embedding_pipeline import embed_and_store, embed_chunks
from modules.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from modules.retrieval import InMemoryVectorStore
from modules.sparse_retrieval import BM25VectorStore

logger = logging.getLogger(__name__)

# Base directory for sample files
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"


async def load_and_seed_demo_data(
    dense_store: InMemoryVectorStore | None = None,
    sparse_store: BM25VectorStore | None = None,
    sample_file: Path | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> tuple[InMemoryVectorStore, BM25VectorStore, int]:
    """Load sample documents, chunk them, embed them, and index into dense and sparse stores.

    Args:
        dense_store: Pre-existing dense store instance or None to create one
        sparse_store: Pre-existing sparse store instance or None to create one
        sample_file: Path to JSON demo docs file (defaults to data/samples/demo_docs.json)
        embedding_provider: Embedding provider for the dense store. Defaults to a
            dependency-free HashingEmbeddingProvider so dense retrieval is meaningful
            without torch/sentence-transformers. Pass the SAME provider to the retriever
            so query and document vectors live in one space.

    Returns:
        Tuple of (dense_store, sparse_store, total_chunks_indexed)
    """
    dense_store = dense_store or InMemoryVectorStore()
    sparse_store = sparse_store or BM25VectorStore()
    sample_file = sample_file or (SAMPLE_DIR / "demo_docs.json")

    if not sample_file.exists():
        logger.warning("Sample demo file not found at %s. Creating empty stores.", sample_file)
        return dense_store, sparse_store, 0

    try:
        with open(sample_file, "r", encoding="utf-8") as f:
            documents: list[dict[str, Any]] = json.load(f)
    except Exception as exc:
        logger.error("Failed to read sample docs file %s: %s", sample_file, exc)
        return dense_store, sparse_store, 0

    chunker = SentenceChunker(target_size=180)
    raw_chunks: list[dict[str, Any]] = []

    for doc in documents:
        doc_id = doc.get("id", "doc_unknown")
        doc_text = doc.get("text", "")
        doc_title = doc.get("title", doc_id)
        doc_meta = doc.get("metadata", {})

        record = {
            "record_id": doc_id,
            "passage": doc_text,
            "language": doc.get("language", "hi"),
        }
        chunks = chunker.chunk(record)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_c{idx+1}"
            raw_chunks.append({
                "id": chunk_id,
                "text": chunk.text,
                "metadata": {
                    "doc_id": doc_id,
                    "title": doc_title,
                    "language": doc.get("language", "en"),
                    # Explicit provenance so demo chunks are never mistaken for the
                    # official MSMARCO-XI corpus in API sources / evidence / Judge Mode.
                    "corpus": "demo",
                    "is_official": False,
                    "provenance_note": (
                        "DEMO/SAMPLE fallback corpus (data/samples/demo_docs.json). "
                        "Not the official ai4bharat/MSMARCO-XI dataset."
                    ),
                    **doc_meta,
                }
            })

    if not raw_chunks:
        return dense_store, sparse_store, 0

    # 1. Index into Sparse BM25 Store
    await sparse_store.index_documents(raw_chunks)

    # 2. Embed & Index into Dense Vector Store
    provider = embedding_provider or HashingEmbeddingProvider()
    await embed_and_store(raw_chunks, provider, dense_store)

    logger.info(
        "Successfully seeded demo corpus: %d raw docs -> %d chunks indexed into dense and sparse stores.",
        len(documents),
        len(raw_chunks),
    )

    return dense_store, sparse_store, len(raw_chunks)
