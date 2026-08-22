"""Evaluation runner for ThinkZen.

This module provides a reproducible evaluation framework for the ThinkZen pipeline.
It measures and reports:

- Retrieval recall (fraction of expected answer passages retrieved)
- Grounding rate (fraction of responses that are grounded, not refused)
- Abstention rate (fraction of responses correctly refused when evidence is insufficient)
- Average pipeline latency per stage
- Pipeline success rate (fraction of requests completed without error)
- Evidence quality (mean max retrieval score across queries)

DATA QUALITY LABELS (never mix categories):
    REAL_DATA       — verified evaluation using authoritative dataset records
    UNIT_TEST_DATA  — synthetic data from test suite (not production representative)
    PENDING         — blocked pending real dataset availability

Evaluation results must NEVER fabricate numbers. Only report what was actually measured.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class DataLabel(str, Enum):
    """Data quality label for evaluation results."""
    REAL_DATA = "REAL_DATA"
    UNIT_TEST_DATA = "UNIT_TEST_DATA"
    PENDING = "PENDING"


@dataclass
class EvalQuery:
    """A single evaluation query with optional ground truth."""
    query_id: str
    query: str
    language: str = "en"
    expected_answer: str | None = None
    expected_passage_ids: list[str] = field(default_factory=list)
    data_label: DataLabel = DataLabel.UNIT_TEST_DATA


@dataclass
class EvalResult:
    """Result for a single evaluation query."""
    query_id: str
    query: str
    answer: str
    refused: bool
    evidence_count: int
    max_retrieval_score: float
    retrieved_chunk_ids: list[str]
    grounding_decision: str
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    success: bool                  # Did the pipeline complete without error?
    recall_at_k: float | None      # If expected_passage_ids known
    data_label: DataLabel


@dataclass
class EvalSummary:
    """Aggregated evaluation summary across all queries."""
    total_queries: int
    grounding_rate: float          # fraction answered (not refused)
    abstention_rate: float         # fraction refused
    success_rate: float            # fraction completed without error
    mean_latency_ms: float
    mean_retrieval_latency_ms: float
    mean_generation_latency_ms: float
    mean_max_retrieval_score: float
    mean_recall_at_k: float | None
    data_label: DataLabel
    data_quality_note: str
    results: list[EvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_label": self.data_label.value,
            "data_quality_note": self.data_quality_note,
            "total_queries": self.total_queries,
            "grounding_rate": round(self.grounding_rate, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "success_rate": round(self.success_rate, 4),
            "latency": {
                "mean_total_ms": round(self.mean_latency_ms, 2),
                "mean_retrieval_ms": round(self.mean_retrieval_latency_ms, 2),
                "mean_generation_ms": round(self.mean_generation_latency_ms, 2),
            },
            "retrieval": {
                "mean_max_score": round(self.mean_max_retrieval_score, 4),
                "mean_recall_at_k": (
                    round(self.mean_recall_at_k, 4) if self.mean_recall_at_k is not None else None
                ),
            },
        }

    def save(self, output_path: Path) -> None:
        """Save evaluation summary to a JSON file with provenance metadata."""
        output = {
            "evaluation_metadata": {
                "data_label": self.data_label.value,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data_quality_note": self.data_quality_note,
            },
            "summary": self.to_dict(),
            "per_query_results": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "refused": r.refused,
                    "evidence_count": r.evidence_count,
                    "max_retrieval_score": round(r.max_retrieval_score, 4),
                    "total_latency_ms": round(r.total_latency_ms, 2),
                    "success": r.success,
                    "recall_at_k": r.recall_at_k,
                    "data_label": r.data_label.value,
                }
                for r in self.results
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


class PipelineEvaluator:
    """Run evaluation queries against the live pipeline and collect results.

    Usage (unit test data)::

        evaluator = PipelineEvaluator(base_url="http://localhost:8000")
        queries = [
            EvalQuery("q1", "What is ThinkZen?", data_label=DataLabel.UNIT_TEST_DATA),
        ]
        summary = await evaluator.evaluate(queries)
        print(summary.to_dict())

    Real-data evaluation (pending dataset availability)::

        # Load from authoritative MSMARCO-XI sample
        queries = load_msmarco_sample(path="data/samples/msmarco_hi_100.json")
        summary = await evaluator.evaluate(queries, data_label=DataLabel.REAL_DATA)
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    async def evaluate(
        self,
        queries: list[EvalQuery],
        confidence_threshold: float = 0.10,
    ) -> EvalSummary:
        """Evaluate a list of queries against the running pipeline.

        Args:
            queries: List of EvalQuery objects to evaluate.
            confidence_threshold: Confidence threshold to use for all queries.

        Returns:
            EvalSummary with aggregated metrics and per-query results.
        """
        import httpx

        results: list[EvalResult] = []

        async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
            for eq in queries:
                result = await self._run_single_query(client, eq, confidence_threshold)
                results.append(result)

        return self._aggregate(results, queries[0].data_label if queries else DataLabel.UNIT_TEST_DATA)

    async def _run_single_query(
        self,
        client: Any,
        eq: EvalQuery,
        confidence_threshold: float,
    ) -> EvalResult:
        try:
            response = await client.post(
                "/api/v1/query",
                json={
                    "query": eq.query,
                    "confidence_threshold": confidence_threshold,
                    "use_adaptive_retrieval": True,
                },
            )
            response.raise_for_status()
            data = response.json()

            retrieved_ids = [s["chunk_id"] for s in data.get("sources", [])]
            tel = data.get("telemetry", {})
            eb = tel.get("evidence_bundle", {})

            recall: float | None = None
            if eq.expected_passage_ids:
                hits = len(set(retrieved_ids) & set(eq.expected_passage_ids))
                recall = hits / max(1, len(eq.expected_passage_ids))

            return EvalResult(
                query_id=eq.query_id,
                query=eq.query,
                answer=data.get("answer", ""),
                refused=data.get("refused", False),
                evidence_count=eb.get("evidence_count", 0),
                max_retrieval_score=eb.get("max_retrieval_score", 0.0),
                retrieved_chunk_ids=retrieved_ids,
                grounding_decision=eb.get("grounding_decision", "unknown"),
                total_latency_ms=tel.get("total_latency_ms", 0.0),
                retrieval_latency_ms=tel.get("retrieval_latency_ms", 0.0),
                generation_latency_ms=tel.get("generation_latency_ms", 0.0),
                success=True,
                recall_at_k=recall,
                data_label=eq.data_label,
            )

        except Exception as exc:
            return EvalResult(
                query_id=eq.query_id,
                query=eq.query,
                answer="",
                refused=True,
                evidence_count=0,
                max_retrieval_score=0.0,
                retrieved_chunk_ids=[],
                grounding_decision="error",
                total_latency_ms=0.0,
                retrieval_latency_ms=0.0,
                generation_latency_ms=0.0,
                success=False,
                recall_at_k=None,
                data_label=eq.data_label,
            )

    def _aggregate(self, results: list[EvalResult], data_label: DataLabel) -> EvalSummary:
        if not results:
            return EvalSummary(
                total_queries=0,
                grounding_rate=0.0,
                abstention_rate=0.0,
                success_rate=0.0,
                mean_latency_ms=0.0,
                mean_retrieval_latency_ms=0.0,
                mean_generation_latency_ms=0.0,
                mean_max_retrieval_score=0.0,
                mean_recall_at_k=None,
                data_label=data_label,
                data_quality_note=_quality_note(data_label),
            )

        n = len(results)
        successful = [r for r in results if r.success]
        grounded = [r for r in successful if not r.refused]
        refused = [r for r in successful if r.refused]

        recalls = [r.recall_at_k for r in results if r.recall_at_k is not None]

        return EvalSummary(
            total_queries=n,
            grounding_rate=len(grounded) / n,
            abstention_rate=len(refused) / n,
            success_rate=len(successful) / n,
            mean_latency_ms=_mean([r.total_latency_ms for r in successful]) if successful else 0.0,
            mean_retrieval_latency_ms=_mean([r.retrieval_latency_ms for r in successful]) if successful else 0.0,
            mean_generation_latency_ms=_mean([r.generation_latency_ms for r in successful]) if successful else 0.0,
            mean_max_retrieval_score=_mean([r.max_retrieval_score for r in successful]) if successful else 0.0,
            mean_recall_at_k=_mean(recalls) if recalls else None,
            data_label=data_label,
            data_quality_note=_quality_note(data_label),
            results=results,
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quality_note(label: DataLabel) -> str:
    if label == DataLabel.REAL_DATA:
        return (
            "REAL_DATA: Evaluation performed on verified records from "
            "ai4bharat/MSMARCO-XI (train/hintrain.parquet). "
            "Results are production-representative."
        )
    if label == DataLabel.UNIT_TEST_DATA:
        return (
            "UNIT_TEST_DATA: Evaluation performed on synthetic/demo test data. "
            "Results are NOT production-representative. "
            "Real-data validation is PENDING (blocked on dataset availability)."
        )
    return (
        "PENDING: Real-data evaluation blocked pending ai4bharat/MSMARCO-XI "
        "Hindi shard availability (~3.7GB). Pipeline is ready to evaluate "
        "when a verified 100-record sample is available."
    )


__all__ = [
    "PipelineEvaluator",
    "EvalQuery",
    "EvalResult",
    "EvalSummary",
    "DataLabel",
]
