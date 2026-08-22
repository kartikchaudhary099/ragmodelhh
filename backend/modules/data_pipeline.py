from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a tiny sample from the public Hugging Face dataset."""

    dataset_name: str = "ai4bharat/MSMARCO-XI"
    config: str = "hi"
    split: str = "train"
    streaming: bool = True
    sample_size: int = 100
    limit: int | None = None
    trust_remote_code: bool = False


def _safe_get(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if record is None:
            return ""
        value = record.get(key)
        if value is not None and value != "":
            return value
    return ""


def _normalize_passages(value: Any) -> str:
    """Extract one meaningful passage candidate without flattening the entire nested row.

    The real MSMARCO-XI `passages` field is commonly a list or nested mapping, not a
    single flat string. The tiny experiment only needs the first useful passage text and
    should avoid concatenating the entire row payload into a giant string, which is what
    triggers the streamed-memory blow-up seen in a large parquet batch.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            candidate = _normalize_passages(item)
            if candidate:
                return candidate
        return ""
    if isinstance(value, dict):
        for key in ("passage_text", "passage", "text", "content", "body", "snippet"):
            text = value.get(key)
            if text:
                return str(text).strip()
        for key in ("passages", "texts"):
            nested = value.get(key)
            candidate = _normalize_passages(nested)
            if candidate:
                return candidate
        for key, nested_value in value.items():
            if key in {"score", "is_selected", "passage_id"}:
                continue
            candidate = _normalize_passages(nested_value)
            if candidate:
                return candidate
        return ""
    return str(value).strip()


def normalize_record(record: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize the real MSMARCO-XI row shape into the chunking experiment schema."""
    if not isinstance(record, dict):
        return {
            "record_id": "",
            "query_id": "",
            "query": "",
            "Answer": "",
            "Eng_Query": "",
            "Eng_Answer": "",
            "passage": "",
            "language": "hi",
            "query_type": "",
            "source_language": "",
            "target_language": "",
            "passage_id": "",
            "metadata": {},
        }

    query = _safe_get(record, "query")
    answer = _safe_get(record, "Answer")
    english_query = _safe_get(record, "Eng_Query")
    english_answer = _safe_get(record, "Eng_Answer")
    passages = _normalize_passages(_safe_get(record, "passages"))
    query_id = _safe_get(record, "query_id")
    record_id = _safe_get(record, "record_id", "query_id", "id")
    source_lang = _safe_get(record, "source_lang")
    target_lang = _safe_get(record, "target_lang")
    query_type = _safe_get(record, "query_type")

    normalized = {
        "record_id": str(record_id) if record_id not in (None, "") else f"record-{query_id or 'unknown'}",
        "query_id": str(query_id) if query_id not in (None, "") else "",
        "query": str(query if query not in (None, "") else english_query),
        "Answer": str(answer if answer not in (None, "") else english_answer),
        "Eng_Query": str(english_query if english_query not in (None, "") else query),
        "Eng_Answer": str(english_answer if english_answer not in (None, "") else answer),
        "passage": passages,
        "language": str(source_lang if source_lang not in (None, "") else target_lang if target_lang not in (None, "") else "hi"),
        "query_type": str(query_type if query_type not in (None, "") else "unknown"),
        "source_language": str(source_lang if source_lang not in (None, "") else "hi"),
        "target_language": str(target_lang if target_lang not in (None, "") else "hi"),
        "passage_id": str(_safe_get(record, "passage_id", "pid")),
        "metadata": {
            k: v
            for k, v in record.items()
            if k not in {"query", "Answer", "Eng_Query", "Eng_Answer", "passages", "source_lang", "target_lang", "query_id"}
        },
    }

    if not normalized["source_language"]:
        normalized["source_language"] = normalized["language"]
    if not normalized["target_language"]:
        normalized["target_language"] = normalized["language"]
    if not normalized["query_type"]:
        normalized["query_type"] = "unknown"
    return normalized


def _iter_hindi_records(limit: int, streaming: bool = True) -> Iterator[dict[str, Any]]:
    """Load a tiny Hindi sample from the authoritative MSMARCO-XI dataset without materializing the full corpus."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("The 'datasets' package is required for the data-sampling experiment.") from exc

    bounded_limit = max(0, int(limit))
    official_hindi_file = "train/hintrain.parquet"

    try:
        dataset = load_dataset(
            "ai4bharat/MSMARCO-XI",
            data_files={"train": official_hindi_file},
            split="train",
            streaming=streaming,
        )
        if streaming and bounded_limit:
            dataset = dataset.take(bounded_limit)
    except (ValueError, FileNotFoundError):
        logger.warning("Official Hindi parquet file was not exposed at runtime; falling back to default dataset config with a hard cap. [VERIFY]")
        dataset = load_dataset(
            "ai4bharat/MSMARCO-XI",
            split="train",
            streaming=streaming,
        )
        if streaming and bounded_limit:
            dataset = dataset.take(bounded_limit)

    yielded = 0
    for record in dataset:
        normalized = normalize_record(record)
        if not normalized["query"] and not normalized["passage"]:
            logger.warning("Skipping empty dataset record at sample index %s", yielded)
            continue

        yield normalized
        yielded += 1

        if bounded_limit and yielded >= bounded_limit:
            break

    if yielded == 0:
        logger.warning("No records were yielded from the MSMARCO-XI Hindi stream.")


def load_hindi_sample_records(limit: int = 100, streaming: bool = True) -> list[dict[str, Any]]:
    """Return a reproducible, tiny real sample from the authoritative MSMARCO-XI dataset."""
    limit = max(0, int(limit))
    records: list[dict[str, Any]] = []
    for record in _iter_hindi_records(limit=limit, streaming=streaming):
        records.append(record)
        if len(records) >= limit:
            break
    logger.info("Loaded %s Hindi sample records from ai4bharat/MSMARCO-XI (streaming=%s)", len(records), streaming)
    return records


def load_official_raw_sample(
    limit: int = 100,
    streaming: bool = True,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch a tiny REAL raw sample of ai4bharat/MSMARCO-XI (Hindi split) as *raw* rows.

    Unlike :func:`load_hindi_sample_records` (which returns the normalized chunking schema),
    this returns rows projected to exactly the fields required by
    :func:`modules.sample_ingestion.validate_external_sample`, preserving the original
    nested ``passages`` structure. It is the fetch step used by
    ``scripts/build_msmarco_xi_sample.py`` to produce the validated on-disk artifact.

    Requires network access and the ``datasets`` package (neither is available in the
    isolated build sandbox — this runs user-side on Windows).

    Args:
        limit: Maximum number of records to fetch (the caller/build script additionally
            caps this at ``MAX_SAMPLE_SIZE``).
        streaming: Stream the parquet instead of materializing the full corpus.
        revision: Optional dataset git revision/tag for reproducibility.

    Returns:
        A list of raw row dicts, each containing the nine fields required by the
        provenance validator (``passages`` kept as-is).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard (user-side only)
        raise RuntimeError(
            "The 'datasets' package is required to build the official sample. "
            "Install it with: pip install datasets"
        ) from exc

    from modules.sample_ingestion import EXPECTED_SOURCE_FILE, OPTIONAL_FIELDS, REQUIRED_FIELDS

    bounded_limit = max(1, int(limit))

    def _open_stream() -> Any:
        base_kwargs: dict[str, Any] = {"split": "train", "streaming": streaming}
        if revision:
            base_kwargs["revision"] = revision
        try:
            dataset = load_dataset(
                "ai4bharat/MSMARCO-XI",
                data_files={"train": EXPECTED_SOURCE_FILE},
                **base_kwargs,
            )
        except (ValueError, FileNotFoundError):
            logger.warning(
                "Official Hindi parquet file was not exposed; falling back to default "
                "config with a hard cap. [VERIFY]"
            )
            dataset = load_dataset("ai4bharat/MSMARCO-XI", **base_kwargs)
        if streaming and bounded_limit:
            dataset = dataset.take(bounded_limit)
        return dataset

    dataset = _open_stream()

    # Every REQUIRED_FIELDS value in this scalar-column split is a string except query_id
    # (an int). Coerce the string-typed ones to real strings; keep query_id numeric-or-str.
    string_fields = tuple(f for f in REQUIRED_FIELDS if f != "query_id")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in dataset:
        # Project onto exactly the required scalar fields; carry any OPTIONAL_FIELDS
        # (e.g. `passages`) through untouched ONLY when the source actually provides them.
        row = {key: record.get(key) for key in REQUIRED_FIELDS}
        for opt in OPTIONAL_FIELDS:
            if opt in record and record.get(opt) not in (None, "", [], {}):
                row[opt] = record[opt]

        query_id = str(row.get("query_id") or "").strip()
        # Quality filter (not fabrication): require a stable id, a query, and real answer
        # content to ground on. The Hindi split has no `passages`, so we do NOT require it.
        if not query_id or query_id in seen_ids:
            continue
        if not (row.get("query") or row.get("Eng_Query")):
            continue
        if not (str(row.get("Answer") or "").strip() or str(row.get("Eng_Answer") or "").strip()):
            continue

        # Coerce string-typed required fields to real strings for the strict validator;
        # leave query_id as the source type (int) — the validator coerces it.
        for key in string_fields:
            value = row.get(key)
            row[key] = "" if value is None else str(value)

        seen_ids.add(query_id)
        rows.append(row)
        if len(rows) >= bounded_limit:
            break

    logger.info(
        "Fetched %s raw MSMARCO-XI rows for artifact build (streaming=%s, revision=%s)",
        len(rows),
        streaming,
        revision,
    )
    return rows
