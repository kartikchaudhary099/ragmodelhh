#!/usr/bin/env python3
"""Build a tiny, reproducible, REAL subset of the official ai4bharat/MSMARCO-XI dataset.

This is the USER-SIDE step (run on Windows) that fetches a bounded sample directly from
Hugging Face and writes a validated on-disk artifact the app can ingest. It requires
network access and the `datasets` package; the isolated build sandbox has neither, which
is why this is a separate, explicit command rather than something that runs automatically.

What it does:
  1. Streams up to --limit records (capped at 100) from `train/hintrain.parquet`
     (the official Hindi split) via `modules.data_pipeline.load_official_raw_sample`.
  2. Builds a provenance descriptor (dataset, source_file, revision, UTC timestamp,
     sample_size, selected_query_ids).
  3. Validates the sample + provenance with the SAME strict gate the app uses
     (`modules.sample_ingestion.validate_external_sample`) — fail-fast, before writing.
  4. Writes data/official/msmarco_xi_sample.json + data/official/provenance.json.

It never fabricates records: if the fetch yields nothing, it exits non-zero and writes
nothing. The dataset content it writes is git-ignored and must not be committed.

RUN (from the repo root, using the project venv):
    .venv\\Scripts\\python.exe scripts\\build_msmarco_xi_sample.py --limit 100
    # optional reproducibility pin:
    .venv\\Scripts\\python.exe scripts\\build_msmarco_xi_sample.py --limit 100 --revision <git-sha-or-tag>

Then enable the official corpus and run the app / tests:
    set THINKZEN_CORPUS=official
    .venv\\Scripts\\python.exe -m pytest -q
    .venv\\Scripts\\python.exe scripts\\verify_pipeline.py
    .venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir backend
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from modules.data_pipeline import load_official_raw_sample  # noqa: E402
from modules.sample_ingestion import (  # noqa: E402
    EXPECTED_DATASET,
    EXPECTED_SOURCE_FILE,
    MAX_SAMPLE_SIZE,
    validate_external_sample,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and validate a tiny real subset of ai4bharat/MSMARCO-XI.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_SAMPLE_SIZE,
        help=f"Number of records to fetch (capped at {MAX_SAMPLE_SIZE}). Default: {MAX_SAMPLE_SIZE}.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional dataset git revision/tag to pin for reproducibility.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "official"),
        help="Directory to write the sample + provenance JSON (default: data/official).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming (materializes more of the dataset; not recommended).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    limit = max(1, min(int(args.limit), MAX_SAMPLE_SIZE))
    if limit != args.limit:
        print(f"[info] --limit adjusted to {limit} (bounded to 1..{MAX_SAMPLE_SIZE}).")

    print(
        f"[info] Fetching up to {limit} records from {EXPECTED_DATASET} "
        f"({EXPECTED_SOURCE_FILE}), streaming={not args.no_streaming}, revision={args.revision} ..."
    )

    try:
        rows = load_official_raw_sample(
            limit=limit,
            streaming=not args.no_streaming,
            revision=args.revision,
        )
    except RuntimeError as exc:
        # Missing `datasets` package or similar fetch prerequisite.
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if not rows:
        print(
            "[error] No records were fetched from the dataset. Nothing written. "
            "Check network access / dataset availability. (No data was fabricated.)",
            file=sys.stderr,
        )
        return 3

    query_ids = [str(row["query_id"]) for row in rows]
    provenance = {
        "dataset": EXPECTED_DATASET,
        "source_file": EXPECTED_SOURCE_FILE,
        "source_revision": args.revision,
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(rows),
        "selected_query_ids": query_ids,
    }

    # Fail-fast: validate with the SAME strict gate the app uses, BEFORE writing anything.
    try:
        validate_external_sample(rows, provenance)
    except ValueError as exc:
        print(
            f"[error] Fetched sample failed strict validation: {exc}\n"
            "Nothing was written. This usually means the upstream schema shifted; "
            "re-run or pin --revision.",
            file=sys.stderr,
        )
        return 4

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "msmarco_xi_sample.json"
    provenance_path = out_dir / "provenance.json"

    sample_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] Wrote {len(rows)} validated records -> {sample_path}")
    print(f"[ok] Wrote provenance                 -> {provenance_path}")
    print(f"[ok] dataset={EXPECTED_DATASET} source_file={EXPECTED_SOURCE_FILE} "
          f"revision={args.revision} extracted_at={provenance['extraction_timestamp']}")
    print()
    print("Next steps — enable the official corpus and verify:")
    print("    set THINKZEN_CORPUS=official")
    print("    .venv\\Scripts\\python.exe -m pytest -q")
    print("    .venv\\Scripts\\python.exe scripts\\verify_pipeline.py")
    print("    .venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir backend")
    print()
    print("NOTE: data/official/ content is git-ignored and must not be committed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
