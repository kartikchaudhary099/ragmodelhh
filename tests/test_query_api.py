"""Integration tests for FastAPI /api/v1/query and /api/v1/judge endpoints.

Tests cover: empty query rejection, grounded success, refusal, adaptive retrieval,
alpha override, Hindi language detection, source field completeness, full telemetry
schema validation (including evidence_bundle), and Judge Mode analytics.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_query_endpoint_empty_query() -> None:
    """Whitespace query should be rejected with 400."""
    response = client.post("/api/v1/query", json={"query": "   "})
    assert response.status_code == 400


def test_query_endpoint_grounded_success() -> None:
    """Valid query should return 200 with grounded answer and telemetry."""
    response = client.post(
        "/api/v1/query",
        json={"query": "What is ThinkZen architecture?", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is ThinkZen architecture?"
    assert data["refused"] is False
    assert len(data["sources"]) > 0
    assert "telemetry" in data
    assert data["telemetry"]["total_latency_ms"] >= 0.0
    assert data["telemetry"]["grounding_status"] == "grounded"


def test_query_endpoint_refusal_out_of_domain() -> None:
    """Very high confidence threshold should trigger an intelligent refusal."""
    response = client.post(
        "/api/v1/query",
        json={
            "query": "What is quantum gravity string theory in 2099?",
            "top_k": 3,
            "confidence_threshold": 0.99,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["refused"] is True
    assert data["telemetry"]["grounding_status"] == "refused"


def test_query_endpoint_telemetry_schema_complete() -> None:
    """Telemetry object must expose all pipeline stage fields."""
    response = client.post(
        "/api/v1/query",
        json={"query": "How does hybrid retrieval work?"},
    )
    assert response.status_code == 200
    data = response.json()
    tel = data["telemetry"]

    # run_id
    assert "run_id" in tel
    assert len(tel["run_id"]) > 0

    # All latency fields must be present and non-negative
    for field_name in (
        "total_latency_ms",
        "query_analysis_latency_ms",
        "retrieval_latency_ms",
        "rerank_latency_ms",
        "evidence_latency_ms",
        "generation_latency_ms",
    ):
        assert field_name in tel, f"Missing telemetry field: {field_name}"
        assert tel[field_name] >= 0.0

    # Alpha and top_k tracking
    assert "alpha_used" in tel
    assert "top_k_used" in tel
    assert "alpha_source" in tel
    assert tel["alpha_source"] in ("adaptive", "override")

    # Query analysis sub-object
    assert "query_analysis" in tel
    qa = tel["query_analysis"]
    for f in ("language", "query_type", "complexity", "keywords", "adaptive_alpha", "adaptive_top_k", "strategy_name", "rationale"):
        assert f in qa, f"Missing query_analysis field: {f}"

    # Evidence bundle sub-object
    assert "evidence_bundle" in tel
    eb = tel["evidence_bundle"]
    for f in ("grounding_decision", "source_diversity", "max_retrieval_score", "mean_coherence_score", "evidence_count", "decision_reason"):
        assert f in eb, f"Missing evidence_bundle field: {f}"


def test_query_endpoint_adaptive_alpha_is_applied() -> None:
    """When use_adaptive_retrieval=True and no alpha override, alpha_source should be 'adaptive'."""
    response = client.post(
        "/api/v1/query",
        json={"query": "Compare BM25 versus dense embeddings.", "use_adaptive_retrieval": True},
    )
    assert response.status_code == 200
    assert response.json()["telemetry"]["alpha_source"] == "adaptive"


def test_query_endpoint_alpha_override_is_respected() -> None:
    """When an explicit alpha is provided, alpha_source should be 'override'."""
    response = client.post(
        "/api/v1/query",
        json={"query": "What is ThinkZen?", "alpha": 0.8},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["telemetry"]["alpha_source"] == "override"
    assert abs(data["telemetry"]["alpha_used"] - 0.8) < 0.001


def test_query_endpoint_detected_language_hindi() -> None:
    """Hindi Devanagari query must be detected correctly in telemetry."""
    response = client.post(
        "/api/v1/query",
        json={"query": "वाणीरैग कैसे काम करता है?"},
    )
    assert response.status_code == 200
    assert response.json()["telemetry"]["detected_language"] == "hi"


def test_query_endpoint_sources_have_required_fields() -> None:
    """Source documents must have chunk_id, text, score, method, metadata."""
    response = client.post(
        "/api/v1/query",
        json={"query": "What is ThinkZen architecture?", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    if data["sources"]:
        source = data["sources"][0]
        for f in ("chunk_id", "text", "score", "method", "metadata"):
            assert f in source
        assert isinstance(source["score"], float)


def test_judge_endpoint_returns_stats() -> None:
    """Judge Mode endpoint must return latency stats and data quality note."""
    # Make at least one query to populate the aggregator
    client.post("/api/v1/query", json={"query": "What is ThinkZen?"})

    response = client.get("/api/v1/judge")
    assert response.status_code == 200
    data = response.json()

    assert "total_runs" in data
    assert "latency_stats" in data
    assert "recent_runs" in data
    assert "data_quality_note" in data
    assert data["total_runs"] >= 1
    assert isinstance(data["recent_runs"], list)


def test_judge_endpoint_latency_stats_have_percentiles() -> None:
    """After real runs, REAL_RUN stats should have P50/P70/P90/P100."""
    # Run a few queries to populate stats
    for q in ["What is ThinkZen?", "How does BM25 work?", "Explain grounding."]:
        client.post("/api/v1/query", json={"query": q})

    response = client.get("/api/v1/judge")
    assert response.status_code == 200
    data = response.json()

    if "REAL_RUN" in data["latency_stats"]:
        stats = data["latency_stats"]["REAL_RUN"]
        for field_name in ("p50_ms", "p70_ms", "p90_ms", "p100_ms", "mean_ms", "count"):
            assert field_name in stats


def test_judge_endpoint_recent_runs_structure() -> None:
    """Recent runs must have all per-stage timing fields."""
    client.post("/api/v1/query", json={"query": "What is ThinkZen?"})
    response = client.get("/api/v1/judge")
    assert response.status_code == 200
    data = response.json()

    if data["recent_runs"]:
        run = data["recent_runs"][0]
        for f in ("run_id", "label", "total_ms", "retrieval_ms", "rerank_ms", "generation_ms", "refused"):
            assert f in run
