"""Official MSMARCO-XI corpus ingestion for ThinkZen.

This module ingests a *validated, traceable* subset of the official Hugging Face
dataset ``ai4bharat/MSMARCO-XI`` into the EXISTING retrieval pipeline. It deliberately
mirrors :func:`modules.sample_seeder.load_and_seed_demo_data` (same chunker, same
sparse+dense indexing calls) so nothing about retrieval, grounding, or Judge Mode
changes — only the *source* of the corpus and its provenance metadata.

Design guarantees (aligned with the project's non-negotiable rules):

* **No silent fallback.** If the official artifact is missing or fails validation,
  ingestion raises :class:`OfficialCorpusUnavailable`. It NEVER quietly substitutes the
  demo corpus while claiming MSMARCO-XI.
* **Provenance gate.** Records pass through :func:`modules.sample_ingestion.validate_external_sample`,
  the same strict validator used by the ingestion contract tests. Only a real, bounded
  (≤100-record) subset extracted from ``train/hintrain.parquet`` is accepted.
* **Advanced chunking.** Official passages are chunked with :class:`SentenceChunker`
  (strategy ``"sentence-boundary"``), NOT the naive :class:`FixedSizeChunker` baseline.
* **Traceable metadata.** Every produced chunk carries ``corpus="official"``,
  ``is_official=True``, ``dataset``, ``source_file``, ``source_revision`` and
  ``extraction_timestamp`` so the API ``sources[].metadata`` (and the frontend evidence
  cards / Judge Mode) can prove where each chunk came from and distinguish it from demo data.

The tiny artifact itself is produced OUT-OF-BAND by ``scripts/build_msmarco_xi_sample.py``
(which needs network access and the ``datasets`` package). The dataset content is never
committed to the repository.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from modules.chunking import BaseChunker, SentenceChunker
from modules.data_pipeline import normalize_record
from modules.embedding_pipeline import embed_and_store
from modules.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from modules.retrieval import InMemoryVectorStore
from modules.sample_ingestion import (
    EXPECTED_DATASET,
    EXPECTED_SOURCE_FILE,
    validate_external_sample,
)
from modules.sparse_retrieval import BM25VectorStore

logger = logging.getLogger(__name__)

# Corpus mode identifiers (also used by config + query.py branch).
CORPUS_OFFICIAL = "official"
CORPUS_DEMO = "demo"

# Default on-disk locations for the validated official artifact. These live under
# data/official/ which is git-ignored (except the README + .gitkeep) so real dataset
# content is never committed.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = PROJECT_ROOT / "data" / "official"
DEFAULT_SAMPLE_PATH = OFFICIAL_DIR / "msmarco_xi_sample.json"
DEFAULT_PROVENANCE_PATH = OFFICIAL_DIR / "provenance.json"

# The exact command a user must run to produce the artifact (network + `datasets` required).
BUILD_COMMAND = "python scripts/build_msmarco_xi_sample.py --limit 100"


class OfficialCorpusUnavailable(RuntimeError):
    """Raised when official ingestion is requested but no validated artifact is available.

    We raise loudly instead of falling back to the demo corpus: the project must never
    claim to be serving MSMARCO-XI while actually serving demo data.
    """


def load_official_records(
    sample_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and strictly validate the official MSMARCO-XI sample artifact.

    Args:
        sample_path: Path to the JSON array of raw MSMARCO-XI records.
        provenance_path: Path to the JSON provenance descriptor.

    Returns:
        ``(validated_records, provenance)`` where ``validated_records`` is the output of
        :func:`validate_external_sample` (raw MSMARCO-XI row shape) and ``provenance`` is
        the parsed provenance mapping.

    Raises:
        OfficialCorpusUnavailable: If either file is missing, unreadable, or the sample
            fails the strict provenance/record validation.
    """
    sample_path = Path(sample_path) if sample_path else DEFAULT_SAMPLE_PATH
    provenance_path = Path(provenance_path) if provenance_path else DEFAULT_PROVENANCE_PATH

    if not sample_path.exists() or not provenance_path.exists():
        raise OfficialCorpusUnavailable(
            "Official MSMARCO-XI artifact not found "
            f"(expected '{sample_path.name}' and '{provenance_path.name}' in {sample_path.parent}). "
            "Build it first (network access + `pip install datasets` required):\n"
            f"    {BUILD_COMMAND}\n"
            "Refusing to fall back to demo data while claiming ai4bharat/MSMARCO-XI."
        )

    try:
        with open(sample_path, "r", encoding="utf-8") as fh:
            sample = json.load(fh)
        with open(provenance_path, "r", encoding="utf-8") as fh:
            provenance = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialCorpusUnavailable(
            f"Failed to read the official MSMARCO-XI artifact: {exc}"
        ) from exc

    try:
        # Strict, shared provenance gate (same validator as the ingestion contract tests).
        validated = validate_external_sample(sample, provenance)
    except ValueError as exc:
        raise OfficialCorpusUnavailable(
            f"Official MSMARCO-XI artifact failed provenance validation: {exc}"
        ) from exc

    logger.info(
        "Loaded %d validated OFFICIAL records (dataset=%s, source_file=%s)",
        len(validated),
        provenance.get("dataset"),
        provenance.get("source_file"),
    )
    return validated, provenance


def official_records_to_chunks(
    records: list[dict[str, Any]],
    provenance: dict[str, Any],
    chunker: BaseChunker | None = None,
) -> list[dict[str, Any]]:
    """Chunk validated official records into indexable raw-chunk dicts with provenance.

    Uses the project's advanced :class:`SentenceChunker` (strategy ``"sentence-boundary"``)
    by default — never the naive fixed-size baseline. Each returned dict has the exact
    ``{"id", "text", "metadata"}`` shape the sparse + dense indexers expect.

    Args:
        records: Validated MSMARCO-XI records (raw row shape) from :func:`load_official_records`.
        provenance: Parsed provenance mapping (dataset/source_file/revision/timestamp).
        chunker: Optional chunker override; defaults to ``SentenceChunker(target_size=180)``
            to match the demo seeder. A naive ``FixedSizeChunker`` should not be used here.

    Returns:
        List of ``{"id", "text", "metadata"}`` raw-chunk dicts, provenance-tagged.
    """
    chunker = chunker or SentenceChunker(target_size=180)

    dataset = provenance.get("dataset", EXPECTED_DATASET)
    source_file = provenance.get("source_file", EXPECTED_SOURCE_FILE)
    source_revision = provenance.get("source_revision")
    extraction_timestamp = provenance.get("extraction_timestamp")

    raw_chunks: list[dict[str, Any]] = []
    for raw in records:
        # normalize_record maps the raw MSMARCO row -> chunking schema (passage/record_id/etc.).
        normalized = normalize_record(raw)
        record_id = normalized.get("record_id") or normalized.get("query_id") or "unknown"
        title = normalized.get("Eng_Query") or normalized.get("query") or record_id

        # What we actually index & ground on. The official Hindi split has no `passages`
        # column, so the retrievable knowledge content is the dataset's real answer text:
        #   * Hindi answer  (`Answer`, Devanagari)  -> supports Hindi / Hinglish queries
        #   * English answer (`Eng_Answer`)         -> supports English queries
        # Both are genuine dataset fields, so indexing both makes this bilingual Q/A corpus
        # retrievable in either language without inventing content. If a future artifact
        # DOES carry `passages`, `normalize_record` surfaces it as `passage` and we prefer it
        # for the primary (source-language) variant.
        hi_text = (normalized.get("passage") or normalized.get("Answer") or "").strip()
        en_text = (normalized.get("Eng_Answer") or "").strip()
        source_language = normalized.get("source_language") or normalized.get("language") or "hi"
        target_language = normalized.get("target_language") or "hi"

        # (variant_lang_tag, chunk_lang, text) — the primary variant uses the record's
        # target language (Devanagari answer for the Hindi split); the English answer is a
        # second variant only when it differs (avoids duplicate indexing when they coincide).
        variants: list[tuple[str, str]] = []
        if hi_text:
            variants.append((target_language, hi_text))
        if en_text and en_text != hi_text:
            variants.append(("eng_Latn", en_text))

        running_index = 0
        for chunk_lang, variant_text in variants:
            # Feed the chunker a minimal sub-record so it resolves THIS variant's text and
            # tags the chunk with the correct per-variant language.
            sub_record = {
                "record_id": record_id,
                "query_id": normalized.get("query_id", ""),
                "passage": variant_text,
                "language": chunk_lang,
                "query_type": normalized.get("query_type", "unknown"),
                "source_language": source_language,
                "target_language": target_language,
            }
            for chunk in chunker.chunk(sub_record):
                # Start from the chunker's own metadata (chunk_role, strategy, language,
                # query_id, query_type, source/target language) then overlay the provenance
                # that proves this chunk is OFFICIAL MSMARCO-XI, not demo data.
                metadata: dict[str, Any] = dict(chunk.metadata)
                metadata.update(
                    {
                        "doc_id": record_id,
                        "title": title,
                        "chunk_language": chunk_lang,
                        "corpus": CORPUS_OFFICIAL,
                        "is_official": True,
                        "dataset": dataset,
                        "source_file": source_file,
                        "source_revision": source_revision,
                        "extraction_timestamp": extraction_timestamp,
                        "provenance_note": (
                            "Validated subset of the official ai4bharat/MSMARCO-XI dataset. "
                            "Not demo/sample fallback data."
                        ),
                    }
                )
                running_index += 1
                raw_chunks.append(
                    {
                        "id": f"{record_id}_c{running_index}",
                        "text": chunk.text,
                        "metadata": metadata,
                    }
                )

    return raw_chunks


async def load_and_seed_official_data(
    dense_store: InMemoryVectorStore | None = None,
    sparse_store: BM25VectorStore | None = None,
    sample_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    chunker: BaseChunker | None = None,
) -> tuple[InMemoryVectorStore, BM25VectorStore, int, dict[str, Any]]:
    """Load, validate, chunk, and index the official MSMARCO-XI subset.

    This mirrors :func:`modules.sample_seeder.load_and_seed_demo_data` stage-for-stage
    (SentenceChunker -> BM25 index_documents -> embed_and_store) so the retrieval,
    grounding, and telemetry behaviour is identical; only the corpus source and its
    provenance metadata differ.

    Args:
        dense_store: Pre-existing dense store or None to create one.
        sparse_store: Pre-existing sparse store or None to create one.
        sample_path: Override path to the validated sample JSON.
        provenance_path: Override path to the provenance JSON.
        embedding_provider: Shared embedding provider (pass the SAME instance to the
            retriever so query and document vectors share one space).
        chunker: Optional chunker override (defaults to SentenceChunker).

    Returns:
        ``(dense_store, sparse_store, total_chunks_indexed, provenance)``.

    Raises:
        OfficialCorpusUnavailable: If the artifact is missing/invalid or yields no chunks.
            Never falls back to demo data.
    """
    dense_store = dense_store or InMemoryVectorStore()
    sparse_store = sparse_store or BM25VectorStore()

    # Strict gate: raises OfficialCorpusUnavailable if missing/invalid (no silent fallback).
    records, provenance = load_official_records(sample_path, provenance_path)

    raw_chunks = official_records_to_chunks(records, provenance, chunker=chunker)
    if not raw_chunks:
        raise OfficialCorpusUnavailable(
            "Official MSMARCO-XI artifact validated but produced no usable passages "
            "after chunking. Refusing to serve an empty corpus while claiming MSMARCO-XI."
        )

    # 1. Sparse BM25 index (same call the demo seeder uses).
    await sparse_store.index_documents(raw_chunks)

    # 2. Dense embed + upsert (same call the demo seeder uses; metadata preserved end-to-end).
    provider = embedding_provider or HashingEmbeddingProvider()
    await embed_and_store(raw_chunks, provider, dense_store)

    logger.info(
        "Seeded OFFICIAL MSMARCO-XI corpus: %d records -> %d chunks "
        "(dataset=%s, source_file=%s, revision=%s)",
        len(records),
        len(raw_chunks),
        provenance.get("dataset"),
        provenance.get("source_file"),
        provenance.get("source_revision"),
    )
    return dense_store, sparse_store, len(raw_chunks), provenance


__all__ = [
    "CORPUS_OFFICIAL",
    "CORPUS_DEMO",
    "OfficialCorpusUnavailable",
    "load_official_records",
    "official_records_to_chunks",
    "load_and_seed_official_data",
    "DEFAULT_SAMPLE_PATH",
    "DEFAULT_PROVENANCE_PATH",
    "BUILD_COMMAND",
]
