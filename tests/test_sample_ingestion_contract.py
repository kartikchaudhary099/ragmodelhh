import pytest

from backend.modules.sample_ingestion import (
    EXPECTED_DATASET,
    EXPECTED_SOURCE_FILE,
    MAX_SAMPLE_SIZE,
    validate_external_sample,
)


def _sample_records():
    return [
        {
            "query": "what is the capital of India",
            "Answer": "New Delhi is the capital of India.",
            "query_id": "q1",
            "query_type": "factoid",
            "passages": [
                {"text": "India's capital is New Delhi."},
                {"text": "New Delhi is the capital city of India."},
            ],
            "source_lang": "hi",
            "target_lang": "en",
            "Eng_Query": "what is the capital of India",
            "Eng_Answer": "New Delhi is the capital of India.",
        },
        {
            "query": "who wrote hamlet",
            "Answer": "William Shakespeare wrote Hamlet.",
            "query_id": "q2",
            "query_type": "factoid",
            "passages": [{"text": "Hamlet was written by William Shakespeare."}],
            "source_lang": "hi",
            "target_lang": "en",
            "Eng_Query": "who wrote hamlet",
            "Eng_Answer": "William Shakespeare wrote Hamlet.",
        },
    ]


def _provenance(sample_size=2):
    return {
        "dataset": EXPECTED_DATASET,
        "source_file": EXPECTED_SOURCE_FILE,
        "source_revision": "main",
        "extraction_timestamp": "2025-01-01T00:00:00Z",
        "sample_size": sample_size,
        "selected_query_ids": ["q1", "q2"],
    }


def test_validate_external_sample_accepts_authoritative_small_real_sample():
    records = _sample_records()
    validated = validate_external_sample(records, _provenance())
    assert len(validated) == 2
    assert validated[0]["query_id"] == "q1"
    assert validated[1]["query_id"] == "q2"


def test_validate_external_sample_rejects_wrong_dataset():
    provenance = _provenance()
    provenance["dataset"] = "different/dataset"
    with pytest.raises(ValueError, match="dataset"):
        validate_external_sample(_sample_records(), provenance)


def test_validate_external_sample_rejects_size_overflow():
    oversized = _sample_records() + [
        {
            "query": "third sample",
            "Answer": "answer",
            "query_id": "q3",
            "query_type": "factoid",
            "passages": [{"text": "text"}],
            "source_lang": "hi",
            "target_lang": "en",
            "Eng_Query": "third sample",
            "Eng_Answer": "answer",
        }
    ]
    provenance = _provenance(sample_size=MAX_SAMPLE_SIZE + 1)
    provenance["selected_query_ids"] = [f"q{i}" for i in range(1, MAX_SAMPLE_SIZE + 2)]
    with pytest.raises(ValueError, match="at most"):
        validate_external_sample(oversized, provenance)


def test_validate_external_sample_rejects_mismatched_provenance_ids():
    provenance = _provenance()
    provenance["selected_query_ids"] = ["q1", "q99"]
    with pytest.raises(ValueError, match="selected_query_ids"):
        validate_external_sample(_sample_records(), provenance)
