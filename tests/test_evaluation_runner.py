"""Offline unit tests for the T6 evaluation framework (modules.evaluation.runner).

These tests exercise the *pure* aggregation and provenance logic of the evaluation
framework without a running server: `_aggregate`, `_mean`, `_quality_note`,
`EvalSummary.to_dict`, `EvalSummary.save`, the `EvalQuery` defaults, and the
per-query extraction/recall logic in `PipelineEvaluator._run_single_query` (driven by a
tiny in-memory fake HTTP client, so no network is used).

Every asserted number here is hand-verifiable from the inputs. Nothing is fabricated,
no metric is invented, and no existing test contract is changed.
"""

from __future__ import annotations

import json

import pytest

from modules.evaluation.runner import (
    DataLabel,
    EvalQuery,
    EvalResult,
    EvalSummary,
    PipelineEvaluator,
    _mean,
    _quality_note,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _result(
    query_id: str,
    *,
    refused: bool,
    success: bool = True,
    total_latency_ms: float = 0.0,
    retrieval_latency_ms: float = 0.0,
    generation_latency_ms: float = 0.0,
    max_retrieval_score: float = 0.0,
    recall_at_k: float | None = None,
    data_label: DataLabel = DataLabel.UNIT_TEST_DATA,
) -> EvalResult:
    """Build an EvalResult directly (no server) for aggregation tests."""
    return EvalResult(
        query_id=query_id,
        query=f"query for {query_id}",
        answer="" if refused else "some grounded answer",
        refused=refused,
        evidence_count=0 if refused else 3,
        max_retrieval_score=max_retrieval_score,
        retrieved_chunk_ids=[],
        grounding_decision="refused" if refused else "grounded",
        total_latency_ms=total_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        success=success,
        recall_at_k=recall_at_k,
        data_label=data_label,
    )


class _FakeResponse:
    """Minimal stand-in for an httpx.Response."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # noqa: D401 - matches httpx API
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Minimal async client returning a canned payload for `post`."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls: list[dict] = []

    async def post(self, url: str, json: dict) -> _FakeResponse:  # noqa: A002
        self.calls.append({"url": url, "json": json})
        return _FakeResponse(self._payload)


# --------------------------------------------------------------------------- #
# _mean
# --------------------------------------------------------------------------- #
def test_mean_empty_is_zero() -> None:
    assert _mean([]) == 0.0


def test_mean_basic() -> None:
    assert _mean([2.0, 4.0]) == 3.0
    assert _mean([1.0, 2.0, 3.0, 4.0]) == 2.5


# --------------------------------------------------------------------------- #
# _quality_note — provenance honesty
# --------------------------------------------------------------------------- #
def test_quality_note_unit_test_data_is_not_production_representative() -> None:
    note = _quality_note(DataLabel.UNIT_TEST_DATA)
    assert "NOT production-representative" in note
    assert "PENDING" in note


def test_quality_note_pending_is_blocked_on_dataset() -> None:
    note = _quality_note(DataLabel.PENDING)
    assert "PENDING" in note
    assert "MSMARCO-XI" in note


def test_quality_note_real_data_only_place_that_claims_msmarco() -> None:
    """Only REAL_DATA may assert production representativeness; the others must not."""
    real = _quality_note(DataLabel.REAL_DATA)
    unit = _quality_note(DataLabel.UNIT_TEST_DATA)
    pending = _quality_note(DataLabel.PENDING)
    # REAL_DATA is the only note that positively claims production representativeness.
    assert "production-representative" in real
    assert "NOT production-representative" not in real
    assert "MSMARCO-XI" in real
    # UNIT_TEST_DATA explicitly *negates* production representativeness.
    assert "NOT production-representative" in unit
    # PENDING makes no production-representativeness claim at all.
    assert "production-representative" not in pending


# --------------------------------------------------------------------------- #
# EvalQuery defaults
# --------------------------------------------------------------------------- #
def test_eval_query_defaults_are_safe() -> None:
    eq = EvalQuery(query_id="q1", query="What is ThinkZen?")
    assert eq.language == "en"
    assert eq.expected_passage_ids == []
    assert eq.expected_answer is None
    # Default label must be the conservative UNIT_TEST_DATA, never REAL_DATA.
    assert eq.data_label == DataLabel.UNIT_TEST_DATA


# --------------------------------------------------------------------------- #
# _aggregate
# --------------------------------------------------------------------------- #
def test_aggregate_empty_results_are_zeroed_and_labeled() -> None:
    evaluator = PipelineEvaluator()
    summary = evaluator._aggregate([], DataLabel.UNIT_TEST_DATA)
    assert summary.total_queries == 0
    assert summary.grounding_rate == 0.0
    assert summary.abstention_rate == 0.0
    assert summary.success_rate == 0.0
    assert summary.mean_recall_at_k is None
    assert summary.data_label == DataLabel.UNIT_TEST_DATA
    assert "UNIT_TEST_DATA" in summary.data_quality_note


def test_aggregate_rates_and_means_are_correct() -> None:
    """2 grounded + 1 refused, all successful → verify every rate and mean by hand."""
    results = [
        _result("g1", refused=False, total_latency_ms=10.0, retrieval_latency_ms=4.0,
                generation_latency_ms=2.0, max_retrieval_score=0.8, recall_at_k=1.0),
        _result("g2", refused=False, total_latency_ms=20.0, retrieval_latency_ms=6.0,
                generation_latency_ms=4.0, max_retrieval_score=0.6, recall_at_k=0.5),
        _result("r1", refused=True, total_latency_ms=30.0, retrieval_latency_ms=8.0,
                generation_latency_ms=0.0, max_retrieval_score=0.1),
    ]
    evaluator = PipelineEvaluator()
    summary = evaluator._aggregate(results, DataLabel.UNIT_TEST_DATA)

    assert summary.total_queries == 3
    assert summary.grounding_rate == pytest.approx(2 / 3)
    assert summary.abstention_rate == pytest.approx(1 / 3)
    assert summary.success_rate == 1.0
    # Means are over *successful* results (all 3 here).
    assert summary.mean_latency_ms == pytest.approx((10 + 20 + 30) / 3)
    assert summary.mean_retrieval_latency_ms == pytest.approx((4 + 6 + 8) / 3)
    assert summary.mean_generation_latency_ms == pytest.approx((2 + 4 + 0) / 3)
    assert summary.mean_max_retrieval_score == pytest.approx((0.8 + 0.6 + 0.1) / 3)
    # Recall is averaged only over the two queries that had ground truth.
    assert summary.mean_recall_at_k == pytest.approx((1.0 + 0.5) / 2)


def test_aggregate_failed_query_excluded_from_means_but_counted() -> None:
    """A failed pipeline call must lower success_rate and be excluded from latency means."""
    results = [
        _result("ok", refused=False, success=True, total_latency_ms=10.0),
        _result("boom", refused=True, success=False, total_latency_ms=0.0),
    ]
    evaluator = PipelineEvaluator()
    summary = evaluator._aggregate(results, DataLabel.UNIT_TEST_DATA)

    assert summary.total_queries == 2
    assert summary.success_rate == 0.5
    # Only the one successful result contributes to the latency mean.
    assert summary.mean_latency_ms == pytest.approx(10.0)
    # grounding/abstention are computed among *successful* results:
    # 1 successful, grounded → grounding_rate = 1/2, abstention_rate = 0/2.
    assert summary.grounding_rate == pytest.approx(0.5)
    assert summary.abstention_rate == pytest.approx(0.0)


def test_aggregate_recall_none_when_no_ground_truth() -> None:
    results = [_result("g1", refused=False), _result("g2", refused=False)]
    evaluator = PipelineEvaluator()
    summary = evaluator._aggregate(results, DataLabel.UNIT_TEST_DATA)
    assert summary.mean_recall_at_k is None


# --------------------------------------------------------------------------- #
# EvalSummary.to_dict / save
# --------------------------------------------------------------------------- #
def test_to_dict_rounds_and_preserves_none_recall() -> None:
    summary = EvalSummary(
        total_queries=1,
        grounding_rate=0.66666,
        abstention_rate=0.33333,
        success_rate=1.0,
        mean_latency_ms=12.3456,
        mean_retrieval_latency_ms=4.5678,
        mean_generation_latency_ms=2.1234,
        mean_max_retrieval_score=0.87654,
        mean_recall_at_k=None,
        data_label=DataLabel.UNIT_TEST_DATA,
        data_quality_note=_quality_note(DataLabel.UNIT_TEST_DATA),
    )
    d = summary.to_dict()
    assert d["data_label"] == "UNIT_TEST_DATA"
    assert d["grounding_rate"] == 0.6667
    assert d["latency"]["mean_total_ms"] == 12.35
    assert d["retrieval"]["mean_max_score"] == 0.8765
    assert d["retrieval"]["mean_recall_at_k"] is None


def test_save_writes_json_with_provenance(tmp_path) -> None:
    results = [
        _result("g1", refused=False, total_latency_ms=10.0, max_retrieval_score=0.8, recall_at_k=1.0),
        _result("r1", refused=True, total_latency_ms=5.0, max_retrieval_score=0.1),
    ]
    evaluator = PipelineEvaluator()
    summary = evaluator._aggregate(results, DataLabel.UNIT_TEST_DATA)

    out = tmp_path / "nested" / "eval.json"
    summary.save(out)
    assert out.exists()

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["evaluation_metadata"]["data_label"] == "UNIT_TEST_DATA"
    assert "data_quality_note" in loaded["evaluation_metadata"]
    assert loaded["summary"]["total_queries"] == 2
    assert len(loaded["per_query_results"]) == 2
    assert {r["query_id"] for r in loaded["per_query_results"]} == {"g1", "r1"}


# --------------------------------------------------------------------------- #
# PipelineEvaluator._run_single_query — extraction + recall (offline, fake client)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_run_single_query_extracts_fields_and_computes_recall() -> None:
    payload = {
        "answer": "Grounded answer.",
        "refused": False,
        "sources": [{"chunk_id": "a"}, {"chunk_id": "c"}],
        "telemetry": {
            "total_latency_ms": 12.0,
            "retrieval_latency_ms": 5.0,
            "generation_latency_ms": 3.0,
            "evidence_bundle": {
                "evidence_count": 2,
                "max_retrieval_score": 0.9,
                "grounding_decision": "grounded",
            },
        },
    }
    evaluator = PipelineEvaluator()
    eq = EvalQuery("q1", "What is X?", expected_passage_ids=["a", "b"])
    result = await evaluator._run_single_query(_FakeClient(payload), eq, 0.10)

    assert result.success is True
    assert result.refused is False
    assert result.evidence_count == 2
    assert result.max_retrieval_score == 0.9
    assert result.grounding_decision == "grounded"
    assert result.retrieved_chunk_ids == ["a", "c"]
    # 1 hit ("a") out of 2 expected passages → recall 0.5.
    assert result.recall_at_k == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_run_single_query_recall_none_without_ground_truth() -> None:
    payload = {
        "answer": "x",
        "refused": False,
        "sources": [{"chunk_id": "a"}],
        "telemetry": {"evidence_bundle": {}},
    }
    evaluator = PipelineEvaluator()
    eq = EvalQuery("q1", "What is X?")  # no expected_passage_ids
    result = await evaluator._run_single_query(_FakeClient(payload), eq, 0.10)
    assert result.recall_at_k is None


@pytest.mark.asyncio
async def test_run_single_query_handles_transport_error_honestly() -> None:
    class _BoomClient:
        async def post(self, url: str, json: dict):  # noqa: A002
            raise RuntimeError("connection refused")

    evaluator = PipelineEvaluator()
    eq = EvalQuery("q1", "What is X?")
    result = await evaluator._run_single_query(_BoomClient(), eq, 0.10)
    # Failures are recorded honestly: not successful, no fabricated metrics.
    assert result.success is False
    assert result.refused is True
    assert result.grounding_decision == "error"
    assert result.max_retrieval_score == 0.0
    assert result.recall_at_k is None
