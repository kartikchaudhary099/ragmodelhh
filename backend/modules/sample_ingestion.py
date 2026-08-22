from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

EXPECTED_DATASET = "ai4bharat/MSMARCO-XI"
EXPECTED_SOURCE_FILE = "train/hintrain.parquet"
MAX_SAMPLE_SIZE = 100

# Fields that MUST be present on every record. The official ai4bharat/MSMARCO-XI
# `train/hintrain.parquet` split is a scalar-column table (query_id / query / Answer /
# source_lang / target_lang / query_type / Eng_Query / Eng_Answer); it has **no**
# `passages` column, so `passages` is intentionally NOT required here (see OPTIONAL_FIELDS).
REQUIRED_FIELDS = (
    "query",
    "Answer",
    "query_id",
    "query_type",
    "source_lang",
    "target_lang",
    "Eng_Query",
    "Eng_Answer",
)

# Present in some MSMARCO variants but absent from the Hindi scalar-column split. Validated
# for type when present; never required.
OPTIONAL_FIELDS = ("passages",)

# At least one of these must carry real, non-empty answer text — this is what the retriever
# actually indexes and grounds on, so an empty record is rejected even though `passages` is
# optional.
CONTENT_FIELDS = ("Answer", "Eng_Answer")


def _as_str(value: Any, field_name: str) -> str:
    if isinstance(value, str):
        return value.strip()
    raise ValueError(f"Field '{field_name}' must be a non-empty string.")


def _as_id(value: Any, field_name: str) -> str:
    """Coerce a record/query id to a non-empty string.

    The official MSMARCO-XI rows store ``query_id`` as an integer, while synthetic
    fixtures and provenance lists use strings. Accept both (rejecting bool, which is an
    ``int`` subclass) and normalise to ``str`` so downstream comparisons are type-stable.
    """
    if isinstance(value, bool):
        raise ValueError(f"Field '{field_name}' must be an int or string id, not a bool.")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"Field '{field_name}' must be a non-empty id.")
        return stripped
    raise ValueError(f"Field '{field_name}' must be an int or string id.")


def _validate_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError("Provenance metadata is required for the external sample artifact.")

    dataset = provenance.get("dataset")
    source_file = provenance.get("source_file")
    sample_size = provenance.get("sample_size")
    extraction_timestamp = provenance.get("extraction_timestamp")
    selected_query_ids = provenance.get("selected_query_ids")

    if dataset != EXPECTED_DATASET:
        raise ValueError(f"Provenance dataset must be '{EXPECTED_DATASET}'.")
    if source_file != EXPECTED_SOURCE_FILE:
        raise ValueError(f"Provenance source_file must be '{EXPECTED_SOURCE_FILE}'.")
    if not extraction_timestamp:
        raise ValueError("Provenance must include 'extraction_timestamp'.")
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("Provenance 'sample_size' must be a positive integer.")
    if sample_size > MAX_SAMPLE_SIZE:
        raise ValueError(f"External sample must contain at most {MAX_SAMPLE_SIZE} records.")
    if not isinstance(selected_query_ids, list) or not selected_query_ids:
        raise ValueError("Provenance must include a non-empty 'selected_query_ids' list.")

    normalized_selected = []
    seen = set()
    for item in selected_query_ids:
        value = str(item)
        if value in seen:
            raise ValueError("Duplicate query IDs are not allowed in provenance.")
        seen.add(value)
        normalized_selected.append(value)

    if len(normalized_selected) != sample_size:
        raise ValueError("Provenance sample_size must match the number of selected query IDs.")

    source_revision = provenance.get("source_revision")
    if source_revision is not None and not isinstance(source_revision, str):
        raise ValueError("Provenance 'source_revision' must be a string when provided.")

    normalized = {
        "dataset": dataset,
        "source_file": source_file,
        "source_revision": source_revision,
        "extraction_timestamp": _as_str(extraction_timestamp, "extraction_timestamp"),
        "sample_size": sample_size,
        "selected_query_ids": normalized_selected,
    }
    return normalized


def _validate_record(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"Record at index {index} must be a mapping.")

    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Record at index {index} is missing fields: {missing}")

    validated: dict[str, Any] = {}
    validated["query"] = _as_str(record["query"], f"query[{index}]")
    validated["Answer"] = _as_str(record["Answer"], f"Answer[{index}]")
    validated["query_id"] = _as_id(record["query_id"], f"query_id[{index}]")
    validated["query_type"] = _as_str(record["query_type"], f"query_type[{index}]")
    validated["source_lang"] = _as_str(record["source_lang"], f"source_lang[{index}]")
    validated["target_lang"] = _as_str(record["target_lang"], f"target_lang[{index}]")
    validated["Eng_Query"] = _as_str(record["Eng_Query"], f"Eng_Query[{index}]")
    validated["Eng_Answer"] = _as_str(record["Eng_Answer"], f"Eng_Answer[{index}]")

    # `passages` is optional: the official Hindi scalar-column split has no such column.
    # When present, validate its type; when absent, that is acceptable.
    if "passages" in record:
        passages = record["passages"]
        if passages is None:
            raise ValueError(f"Record at index {index} has null passages.")
        if isinstance(passages, (list, tuple, dict, str)):
            validated["passages"] = passages
        else:
            raise ValueError(f"Record at index {index} has invalid passages type.")

    # Require real answer content to ground on (guards against empty/garbage records even
    # though `passages` is optional). `_as_str` already stripped these values.
    if not any(validated.get(field) for field in CONTENT_FIELDS):
        raise ValueError(
            f"Record at index {index} has no answer content "
            f"(all of {list(CONTENT_FIELDS)} are empty)."
        )

    return validated


def validate_external_sample(
    sample: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate an externally prepared MSMARCO-XI sample artifact.

    The sample must already be derived from the official dataset repo and should contain
    exactly 100 real records at most. This is intentionally strict so the project can
    accept only a tiny, traceable subset and reject malformed or ambiguous uploads.
    """
    if isinstance(sample, Mapping):
        data = [sample]
    else:
        data = list(sample)

    if not data:
        raise ValueError("External sample must contain at least one record.")
    if len(data) > MAX_SAMPLE_SIZE:
        raise ValueError(f"External sample must not exceed {MAX_SAMPLE_SIZE} records.")

    normalized_provenance = _validate_provenance(provenance)
    validated_records = []
    seen_query_ids: set[str] = set()

    for idx, record in enumerate(data):
        validated = _validate_record(record, idx)
        query_id = validated["query_id"]
        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate query_id '{query_id}' found in external sample.")
        seen_query_ids.add(query_id)
        validated_records.append(validated)

    if len(validated_records) != normalized_provenance["sample_size"]:
        raise ValueError("Record count must match provenance sample_size.")

    selected_ids = {str(item) for item in normalized_provenance["selected_query_ids"]}
    actual_ids = {record["query_id"] for record in validated_records}
    if selected_ids != actual_ids:
        raise ValueError("Provenance selected_query_ids must exactly match the sample query_ids.")

    return validated_records


def validate_external_sample_file(sample: Any, provenance: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Helper for JSON/YAML-like artifact structures; keeps validation explicit."""
    return validate_external_sample(sample, provenance)


__all__ = [
    "EXPECTED_DATASET",
    "EXPECTED_SOURCE_FILE",
    "MAX_SAMPLE_SIZE",
    "validate_external_sample",
    "validate_external_sample_file",
]
