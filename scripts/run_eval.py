#!/usr/bin/env python3
"""Runnable entry point for the T6 evaluation framework.

Runs a small, clearly-labelled DEMO evaluation set against a *live* ThinkZen server
(`POST /api/v1/query`) using the real `PipelineEvaluator`, then prints and saves a
summary of **measured** metrics plus a behaviour-accuracy check.

HONESTY / PROVENANCE
--------------------
* The evaluation set below is DEMO / UNIT_TEST_DATA — a handful of queries whose
  outcomes are grounded in the seeded demo corpus (`data/samples/demo_docs.json`).
  It is NOT the official benchmark dataset and is labelled `DataLabel.UNIT_TEST_DATA`,
  so the emitted `data_quality_note` explicitly says the results are NOT
  production-representative and that real-data (MSMARCO-XI) validation is PENDING.
* No metric is fabricated. Every number comes from the real request path. Latency is
  measured by the server, not hard-coded. Behaviour accuracy is computed from the real
  `refused` flag returned by the API.
* `expected_passage_ids` are intentionally left empty (recall is reported as `null`)
  rather than guessing chunk IDs — the framework will report recall only once a
  verified ground-truth mapping exists.

USAGE
-----
    # 1. In one terminal, start the server (from the repo root):
    #    .venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir backend
    # 2. In another terminal:
    #    .venv\\Scripts\\python.exe scripts/run_eval.py
    #
    # Options:
    #    --base-url    default http://localhost:8000
    #    --output      default eval_results/demo_eval.json
    #    --confidence  default 0.10 (matches the API default)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from modules.evaluation.runner import (  # noqa: E402
    DataLabel,
    EvalQuery,
    PipelineEvaluator,
)

# --------------------------------------------------------------------------- #
# DEMO evaluation set. Every query is grounded in data/samples/demo_docs.json.
# `expected_behaviour` is the outcome we expect from the grounding pipeline; it is
# checked against the REAL `refused` flag returned by the API (not fabricated).
# --------------------------------------------------------------------------- #
DEMO_QUERIES: list[tuple[EvalQuery, str]] = [
    # (EvalQuery, expected_behaviour in {"grounded", "refused"})
    (EvalQuery("en_grounded", "What is Retrieval-Augmented Generation?", language="en",
               data_label=DataLabel.UNIT_TEST_DATA), "grounded"),
    (EvalQuery("hi_grounded", "वाणीरैग क्या है?", language="hi",
               data_label=DataLabel.UNIT_TEST_DATA), "grounded"),
    (EvalQuery("hinglish_grounded", "ThinkZen mein hybrid search kaise kaam karta hai?",
               language="hi-en", data_label=DataLabel.UNIT_TEST_DATA), "grounded"),
    (EvalQuery("en_refusal", "What is the capital of France?", language="en",
               data_label=DataLabel.UNIT_TEST_DATA), "refused"),
    (EvalQuery("hi_refusal", "फ्रांस की राजधानी क्या है?", language="hi",
               data_label=DataLabel.UNIT_TEST_DATA), "refused"),
    (EvalQuery("hinglish_refusal", "France ki capital kya hai?", language="hi-en",
               data_label=DataLabel.UNIT_TEST_DATA), "refused"),
]


def _print_header(base_url: str, confidence: float) -> None:
    print("=" * 72)
    print("ThinkZen — T6 Evaluation (DEMO / UNIT_TEST_DATA)")
    print("=" * 72)
    print(f"Target server      : {base_url}")
    print(f"Confidence gate    : {confidence}")
    print(f"Queries            : {len(DEMO_QUERIES)} (demo corpus, NOT MSMARCO-XI)")
    print("-" * 72)


async def _run(base_url: str, confidence: float, output: Path) -> int:
    evaluator = PipelineEvaluator(base_url=base_url)
    queries = [eq for eq, _ in DEMO_QUERIES]
    expected = {eq.query_id: behaviour for eq, behaviour in DEMO_QUERIES}

    summary = await evaluator.evaluate(queries, confidence_threshold=confidence)

    # ---- Measured summary (nothing here is fabricated) -------------------- #
    d = summary.to_dict()
    print("\nMEASURED SUMMARY")
    print("-" * 72)
    print(f"data_label            : {d['data_label']}")
    print(f"total_queries         : {d['total_queries']}")
    print(f"success_rate          : {d['success_rate']}")
    print(f"grounding_rate        : {d['grounding_rate']}")
    print(f"abstention_rate       : {d['abstention_rate']}")
    print(f"mean_total_latency_ms : {d['latency']['mean_total_ms']}")
    print(f"mean_retrieval_ms     : {d['latency']['mean_retrieval_ms']}")
    print(f"mean_generation_ms    : {d['latency']['mean_generation_ms']}")
    print(f"mean_max_score        : {d['retrieval']['mean_max_score']}")
    print(f"mean_recall_at_k      : {d['retrieval']['mean_recall_at_k']}")

    if summary.success_rate == 0.0 and summary.total_queries > 0:
        print("\n[!] Every request failed — is the server running at "
              f"{base_url}?  Start it with:")
        print("    .venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir backend")
        return 2

    # ---- Behaviour accuracy (grounded vs refused), from real `refused` ---- #
    print("\nBEHAVIOUR CHECK (expected vs. actual, from real `refused` flag)")
    print("-" * 72)
    correct = 0
    checked = 0
    for r in summary.results:
        want = expected.get(r.query_id)
        if want is None or not r.success:
            status = "SKIP" if not r.success else "----"
            print(f"  [{status}] {r.query_id:<18} success={r.success}")
            continue
        actual = "refused" if r.refused else "grounded"
        ok = actual == want
        correct += int(ok)
        checked += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {r.query_id:<18} "
              f"expected={want:<9} actual={actual:<9} "
              f"max_score={r.max_retrieval_score:.3f} latency_ms={r.total_latency_ms:.1f}")

    if checked:
        print("-" * 72)
        print(f"Behaviour accuracy    : {correct}/{checked} = {correct / checked:.2%}")

    # ---- Persist with provenance ----------------------------------------- #
    summary.save(output)
    print(f"\nSaved JSON summary → {output}")
    print(f"data_quality_note     : {summary.data_quality_note}")

    # Exit non-zero if any behaviour expectation was violated (useful in CI).
    return 0 if (checked and correct == checked) else (1 if checked else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the T6 DEMO evaluation set.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "eval_results" / "demo_eval.json",
    )
    args = parser.parse_args()

    _print_header(args.base_url, args.confidence)
    exit_code = asyncio.run(_run(args.base_url, args.confidence, args.output))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
