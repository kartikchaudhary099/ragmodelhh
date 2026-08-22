from __future__ import annotations

from modules.chunking import (
    FixedSizeChunker,
    ParentChildChunker,
    SentenceChunker,
    normalize_record,
)
from modules.data_pipeline import load_hindi_sample_records


def test_dataset_record_extraction_uses_hindi_stream() -> None:
    import pytest
    pytest.importorskip("datasets")
    records = load_hindi_sample_records(limit=3, streaming=True)

    assert len(records) == 3
    for record in records:
        assert "query_id" in record
        assert "query" in record
        assert "passage" in record
        assert "language" in record
        assert record["language"] == "hi"


def test_msmarco_xi_runtime_fields_are_normalized() -> None:
    record = {
        "query": "भारत की राजधानी क्या है?",
        "Answer": "नई दिल्ली",
        "query_id": "123",
        "query_type": "factual",
        "passages": ["नई दिल्ली भारत की राजधानी है।", "यह एक बड़ा शहर है।"],
        "source_lang": "hi",
        "target_lang": "hi",
        "Eng_Query": "What is the capital of India?",
        "Eng_Answer": "New Delhi",
        "meta": {"source": "MSMARCO-XI"},
    }

    normalized = normalize_record(record)
    assert normalized["query"] == "भारत की राजधानी क्या है?"
    assert normalized["Answer"] == "नई दिल्ली"
    assert normalized["Eng_Query"] == "What is the capital of India?"
    assert normalized["source_language"] == "hi"
    assert normalized["target_language"] == "hi"
    assert "नई दिल्ली भारत की राजधानी है।" in normalized["passage"]


def test_fixed_size_chunker_creates_chunks_with_metadata() -> None:
    record = {
        "query_id": "q-1",
        "query": "भारत की राजधानी क्या है?",
        "passage": "नई दिल्ली भारत की राजधानी है। यह शहर देश की राष्ट्रीय राजधानी है।",
        "language": "hi",
        "query_type": "factual",
        "source_language": "hi",
        "target_language": "hi",
    }
    normalized = normalize_record(record)
    chunks = FixedSizeChunker(chunk_size=25, overlap=5).chunk(normalized)

    assert chunks
    for chunk in chunks:
        assert chunk.text
        assert chunk.metadata["query_id"] == "q-1"
        assert chunk.metadata["language"] == "hi"


def test_sentence_chunker_keeps_sentence_boundaries() -> None:
    record = {
        "query_id": "q-2",
        "query": "भारत में सबसे लंबा नदी कौन सी है?",
        "passage": "गंगा भारत की सबसे लंबी नदी है। यह हिमालय से निकलती है। यह देश की पवित्र नदियों में से एक है.",
        "language": "hi",
    }
    normalized = normalize_record(record)
    chunks = SentenceChunker(target_size=45).chunk(normalized)

    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)
    assert any("गंगा" in chunk.text for chunk in chunks)


def test_parent_child_relationships_are_preserved() -> None:
    record = {
        "query_id": "q-3",
        "query": "भारत की राजधानी क्या है?",
        "passage": "नई दिल्ली भारत की राजधानी है। यह शहर देश की राजधानी है। यह एक प्रमुख शहर है।",
        "language": "hi",
    }
    normalized = normalize_record(record)
    chunks = ParentChildChunker(parent_size=60, child_size=25, overlap=5).chunk(normalized)

    parents = [chunk for chunk in chunks if chunk.metadata.get("chunk_role") == "parent"]
    children = [chunk for chunk in chunks if chunk.metadata.get("chunk_role") == "child"]

    assert parents
    assert children
    assert all(child.metadata.get("parent_id") for child in children)
    assert all(child.metadata.get("source_id") == normalized["record_id"] for child in children)


def test_empty_and_malformed_text_are_handled_gracefully() -> None:
    assert FixedSizeChunker(chunk_size=20, overlap=5).chunk({}) == []
    assert SentenceChunker(target_size=25).chunk({}) == []
    assert ParentChildChunker(parent_size=40, child_size=20, overlap=5).chunk({}) == []

    malformed = {"query_id": "q-4", "query": "bad", "passage": None, "language": "hi"}
    normalized = normalize_record(malformed)
    assert normalized["passage"] == ""


def test_fixed_size_chunker_is_deterministic() -> None:
    record = {
        "query_id": "q-5",
        "query": "भारत के प्रमुख त्योहार कौन से हैं?",
        "passage": "भारत के प्रमुख त्योहार दिवाली, होली, दशहरा, eid और रक्षा बंधन हैं।",
        "language": "hi",
    }
    normalized = normalize_record(record)
    first = FixedSizeChunker(chunk_size=20, overlap=5).chunk(normalized)
    second = FixedSizeChunker(chunk_size=20, overlap=5).chunk(normalized)

    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]
