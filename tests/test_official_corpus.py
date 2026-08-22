"""Offline tests for official MSMARCO-XI corpus ingestion (modules.official_corpus).

IMPORTANT: The records in this file are SYNTHETIC fixtures that merely MIMIC the shape of
ai4bharat/MSMARCO-XI rows. They are NOT real dataset content and are used only to exercise
the ingestion / chunking / provenance / retrieval plumbing without network access. The
provenance `dataset` field must literally be "ai4bharat/MSMARCO-XI" because that is what
the strict validator (modules.sample_ingestion) requires — this is a test of the gate, not
a claim that the fixture text came from the real dataset.

These tests never assert any dataset metric, latency, or benchmark result — only structural
and behavioural invariants of the ingestion code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.embeddings import HashingEmbeddingProvider
from modules.official_corpus import (
    BUILD_COMMAND,
    CORPUS_OFFICIAL,
    OfficialCorpusUnavailable,
    load_and_seed_official_data,
    load_official_records,
    official_records_to_chunks,
)
from modules.retrieval import OrchestratedHybridRetriever
from modules.sample_ingestion import EXPECTED_DATASET, EXPECTED_SOURCE_FILE, validate_external_sample
from modules.sample_seeder import load_and_seed_demo_data

# --- synthetic (non-real) fixtures shaped like MSMARCO-XI rows -----------------------

PASSAGE_1 = (
    "Retrieval augmented generation combines a retriever with a generator. "
    "The retriever finds relevant passages from a corpus. "
    "The generator conditions its answer on those retrieved passages. "
    "This reduces hallucination and improves grounding. "
    "Hybrid search mixes dense and sparse retrieval signals."
)
PASSAGE_2 = (
    "Hybrid search fuses dense vector similarity with sparse BM25 term matching. "
    "An alpha weight balances the two retrieval signals. "
    "Reranking can reorder the fused candidates. "
    "The final passages are handed to the grounded generator."
)


def _raw_record(
    qid: str,
    query: str,
    answer: str,
    passage: str,
    query_type: str = "description",
    src: str = "hi",
    tgt: str = "en",
) -> dict:
    """Build one SYNTHETIC raw MSMARCO-XI-shaped row (all nine required fields)."""
    return {
        "query": query,
        "Answer": answer,
        "query_id": qid,
        "query_type": query_type,
        "passages": passage,  # a plain str is a valid `passages` payload
        "source_lang": src,
        "target_lang": tgt,
        "Eng_Query": query,
        "Eng_Answer": answer,
    }


def _provenance(rows: list[dict], revision: str | None = None) -> dict:
    return {
        "dataset": EXPECTED_DATASET,
        "source_file": EXPECTED_SOURCE_FILE,
        "source_revision": revision,
        "extraction_timestamp": "2026-08-21T00:00:00+00:00",
        "sample_size": len(rows),
        "selected_query_ids": [str(r["query_id"]) for r in rows],
    }


def _write_artifact(dir_path: Path, rows: list[dict], provenance: dict) -> tuple[Path, Path]:
    sample_path = dir_path / "msmarco_xi_sample.json"
    prov_path = dir_path / "provenance.json"
    sample_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    prov_path.write_text(json.dumps(provenance, ensure_ascii=False), encoding="utf-8")
    return sample_path, prov_path


# --- chunking + provenance metadata --------------------------------------------------


def test_official_chunks_carry_official_provenance():
    rows = [_raw_record("q1", "What is RAG?", "RAG combines retrieval and generation.", PASSAGE_1)]
    prov = _provenance(rows, revision="abc123")

    chunks = official_records_to_chunks(rows, prov)

    assert chunks, "expected at least one chunk from a non-empty passage"
    for ch in chunks:
        md = ch["metadata"]
        assert md["corpus"] == CORPUS_OFFICIAL
        assert md["is_official"] is True
        assert md["dataset"] == EXPECTED_DATASET
        assert md["source_file"] == EXPECTED_SOURCE_FILE
        assert md["source_revision"] == "abc123"
        assert md["extraction_timestamp"] == prov["extraction_timestamp"]
        assert md["doc_id"] == "q1"
        assert ch["id"].startswith("q1_c")
        assert ch["text"].strip()


def test_official_ingestion_uses_advanced_sentence_chunker_not_fixed():
    """Official ingestion must use the advanced sentence-boundary strategy, never fixed-size."""
    rows = [_raw_record("q1", "q", "a", PASSAGE_1)]
    chunks = official_records_to_chunks(rows, _provenance(rows))
    assert chunks
    for ch in chunks:
        assert ch["metadata"]["strategy"] == "sentence-boundary"
        assert ch["metadata"]["strategy"] != "fixed-size"


def test_official_long_passage_splits_into_multiple_sequential_chunks():
    rows = [_raw_record("q1", "q", "a", PASSAGE_1)]
    chunks = official_records_to_chunks(rows, _provenance(rows))
    assert len(chunks) >= 2, "a multi-sentence passage should yield multiple sentence chunks"
    ids = [c["id"] for c in chunks]
    assert ids == [f"q1_c{i + 1}" for i in range(len(chunks))]


# --- strict load + validation --------------------------------------------------------


def test_load_official_records_reads_and_validates(tmp_path):
    rows = [
        _raw_record("q1", "q1?", "a1", PASSAGE_1),
        _raw_record("q2", "q2?", "a2", PASSAGE_2),
    ]
    sp, pp = _write_artifact(tmp_path, rows, _provenance(rows))

    records, provenance = load_official_records(sp, pp)

    assert len(records) == 2
    assert provenance["dataset"] == EXPECTED_DATASET
    assert {r["query_id"] for r in records} == {"q1", "q2"}


def test_load_official_records_missing_raises_with_build_command(tmp_path):
    with pytest.raises(OfficialCorpusUnavailable) as excinfo:
        load_official_records(tmp_path / "nope.json", tmp_path / "none.json")
    # Error must be actionable — it points at the reproducible build script.
    assert "build_msmarco_xi_sample.py" in str(excinfo.value)
    assert "build_msmarco_xi_sample.py" in BUILD_COMMAND


def test_load_official_records_wrong_dataset_rejected(tmp_path):
    rows = [_raw_record("q1", "q", "a", PASSAGE_1)]
    prov = _provenance(rows)
    prov["dataset"] = "some/other-dataset"  # not the official MSMARCO-XI
    sp, pp = _write_artifact(tmp_path, rows, prov)

    with pytest.raises(OfficialCorpusUnavailable):
        load_official_records(sp, pp)


def test_load_official_records_provenance_id_mismatch_rejected(tmp_path):
    rows = [_raw_record("q1", "q", "a", PASSAGE_1)]
    prov = _provenance(rows)
    prov["selected_query_ids"] = ["different-id"]  # does not match the actual record
    sp, pp = _write_artifact(tmp_path, rows, prov)

    with pytest.raises(OfficialCorpusUnavailable):
        load_official_records(sp, pp)


# --- no silent fallback --------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_official_missing_artifact_raises_never_falls_back(tmp_path):
    """Requesting official ingestion with no artifact must RAISE, not seed demo data."""
    with pytest.raises(OfficialCorpusUnavailable):
        await load_and_seed_official_data(
            sample_path=tmp_path / "missing.json",
            provenance_path=tmp_path / "missing_prov.json",
        )


# --- end-to-end retrievability with provenance intact --------------------------------


@pytest.mark.asyncio
async def test_official_corpus_retrievable_with_provenance_intact(tmp_path):
    rows = [
        _raw_record("q1", "What is retrieval augmented generation?", "It combines retrieval and generation.", PASSAGE_1),
        _raw_record("q2", "How does hybrid search work?", "It fuses dense and sparse signals.", PASSAGE_2),
    ]
    sp, pp = _write_artifact(tmp_path, rows, _provenance(rows))
    provider = HashingEmbeddingProvider()

    dense, sparse, count, provenance = await load_and_seed_official_data(
        sample_path=sp, provenance_path=pp, embedding_provider=provider,
    )
    assert count >= 2
    assert provenance["dataset"] == EXPECTED_DATASET

    retriever = OrchestratedHybridRetriever(
        dense_store=dense, sparse_store=sparse, alpha=0.5, embedding_provider=provider,
    )
    results = await retriever.retrieve("retrieval augmented generation", top_k=5)

    assert results, "expected retrieval hits from the seeded official corpus"
    for doc in results:
        assert doc.metadata.get("corpus") == CORPUS_OFFICIAL
        assert doc.metadata.get("is_official") is True
        assert doc.metadata.get("dataset") == EXPECTED_DATASET
        assert doc.metadata.get("source_file") == EXPECTED_SOURCE_FILE


@pytest.mark.asyncio
async def test_demo_corpus_chunks_are_labelled_demo_not_official():
    """The demo corpus must be explicitly labelled demo so it is never mistaken for official."""
    provider = HashingEmbeddingProvider()
    dense, sparse, count = await load_and_seed_demo_data(embedding_provider=provider)
    assert count > 0

    retriever = OrchestratedHybridRetriever(
        dense_store=dense, sparse_store=sparse, alpha=0.5, embedding_provider=provider,
    )
    results = await retriever.retrieve("ThinkZen retrieval augmented generation", top_k=5)

    assert results, "expected retrieval hits from the demo corpus"
    for doc in results:
        assert doc.metadata.get("corpus") == "demo"
        assert doc.metadata.get("is_official") is False


# --- REAL schema shape: integer query_id, NO `passages` column, bilingual answers -----
#
# The official ai4bharat/MSMARCO-XI Hindi split (train/hintrain.parquet) is a
# scalar-column table: integer `query_id`, Devanagari `query`/`Answer`, English
# `Eng_Query`/`Eng_Answer`, and NO `passages` column. These fixtures mimic that exact
# shape (still SYNTHETIC text, not real dataset content) to lock in the schema handling.

def _real_shape_record(qid: int) -> dict:
    """A synthetic row matching the REAL hintrain.parquet schema (int id, no passages)."""
    return {
        "query_id": qid,  # INTEGER, as in the real dataset
        "query": "मैनहट्टन परियोजना का प्रभाव क्या था?",  # Devanagari (synthetic)
        "Answer": "मैनहट्टन परियोजना ने परमाणु युग की शुरुआत की। इसने वैज्ञानिक अनुसंधान को बदल दिया।",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "query_type": "DESCRIPTION",
        "Eng_Query": "what was the impact of the manhattan project?",
        "Eng_Answer": "The manhattan project began the atomic age. It transformed scientific research.",
        # NOTE: no `passages` key at all — matches the real scalar-column split.
    }


def _real_shape_provenance(rows: list[dict]) -> dict:
    return {
        "dataset": EXPECTED_DATASET,
        "source_file": EXPECTED_SOURCE_FILE,
        "source_revision": None,  # not invented
        "extraction_timestamp": "2026-08-21T09:58:14+00:00",
        "sample_size": len(rows),
        "selected_query_ids": [str(r["query_id"]) for r in rows],
    }


def test_validator_accepts_int_query_id_and_missing_passages():
    """Real rows have an INTEGER query_id and no `passages` — the validator must accept them."""
    rows = [_real_shape_record(1185869), _real_shape_record(620830)]
    validated = validate_external_sample(rows, _real_shape_provenance(rows))
    assert len(validated) == 2
    # query_id must be coerced to a string so downstream id comparisons are type-stable.
    assert validated[0]["query_id"] == "1185869"
    assert all(isinstance(r["query_id"], str) for r in validated)
    # `passages` was absent and must NOT have been fabricated.
    assert "passages" not in validated[0]


def test_validator_rejects_record_with_no_answer_content():
    """`passages` is optional, but a record with no Answer/Eng_Answer content is still rejected."""
    row = _real_shape_record(1)
    row["Answer"] = ""
    row["Eng_Answer"] = ""
    prov = _real_shape_provenance([row])
    with pytest.raises(ValueError):
        validate_external_sample([row], prov)


def test_real_shape_indexes_bilingual_answer_content_with_provenance():
    """Real-shape rows (no passages) must chunk the REAL Answer text in both languages."""
    rows = [_real_shape_record(1185869)]
    prov = _real_shape_provenance(rows)
    validated = validate_external_sample(rows, prov)

    chunks = official_records_to_chunks(validated, prov)
    assert chunks, "expected chunks from the real Answer/Eng_Answer content"

    # Bilingual: both the Devanagari answer and the English answer are indexed.
    languages = {c["metadata"].get("chunk_language") for c in chunks}
    assert "hin_Deva" in languages
    assert "eng_Latn" in languages

    # The indexed text is the ANSWER content (grounding knowledge), not the question.
    all_text = " ".join(c["text"] for c in chunks)
    assert "परमाणु युग" in all_text  # from the Hindi Answer
    assert "atomic age" in all_text  # from the English Eng_Answer

    # Provenance intact on every chunk; ids sequential per record.
    ids = [c["id"] for c in chunks]
    assert ids == [f"1185869_c{i + 1}" for i in range(len(chunks))]
    for c in chunks:
        assert c["metadata"]["corpus"] == CORPUS_OFFICIAL
        assert c["metadata"]["is_official"] is True
        assert c["metadata"]["dataset"] == EXPECTED_DATASET
        assert c["metadata"]["strategy"] == "sentence-boundary"
        assert c["metadata"]["source_revision"] is None


@pytest.mark.asyncio
async def test_real_shape_bilingual_retrieval_en_and_hi(tmp_path):
    """A real-shape (no-passages) corpus must be retrievable by BOTH English and Hindi queries."""
    rows = [_real_shape_record(1185869)]
    prov = _real_shape_provenance(rows)
    sample_path = tmp_path / "msmarco_xi_sample.json"
    prov_path = tmp_path / "provenance.json"
    sample_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    prov_path.write_text(json.dumps(prov, ensure_ascii=False), encoding="utf-8")

    provider = HashingEmbeddingProvider()
    dense, sparse, count, _ = await load_and_seed_official_data(
        sample_path=sample_path, provenance_path=prov_path, embedding_provider=provider,
    )
    assert count >= 2  # at least one Hindi + one English chunk

    retriever = OrchestratedHybridRetriever(
        dense_store=dense, sparse_store=sparse, alpha=0.5, embedding_provider=provider,
    )
    en_hits = await retriever.retrieve("impact of the manhattan project", top_k=5)
    hi_hits = await retriever.retrieve("मैनहट्टन परियोजना का प्रभाव", top_k=5)

    assert en_hits, "English query must retrieve from the official corpus"
    assert hi_hits, "Hindi query must retrieve from the official corpus"
    for doc in en_hits + hi_hits:
        assert doc.metadata.get("corpus") == CORPUS_OFFICIAL
        assert doc.metadata.get("dataset") == EXPECTED_DATASET
