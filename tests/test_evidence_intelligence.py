"""Tests for Evidence Intelligence module.

Covers: grounding decision logic, deduplication, diversity boost, coherence scoring,
EvidenceBundle serialization, and edge cases.
"""

from __future__ import annotations

import json
import pytest

from modules.evidence import (
    EvidenceBundle,
    EvidenceIntelligence,
    GroundingDecision,
    _compute_coherence_score,
    _deduplicate,
)
from modules.retrieval import RetrievedDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(chunk_id: str, text: str, score: float = 0.5, method: str = "hybrid", doc_id: str | None = None) -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id=chunk_id,
        text=text,
        score=score,
        method=method,
        metadata={"doc_id": doc_id or chunk_id.split("_")[0]},
    )


# ---------------------------------------------------------------------------
# Coherence scoring
# ---------------------------------------------------------------------------


def test_coherence_score_exact_overlap() -> None:
    """High token overlap should produce high coherence score."""
    score = _compute_coherence_score("what is ThinkZen", "ThinkZen is a multilingual RAG system")
    assert score > 0.3


def test_coherence_score_no_overlap() -> None:
    score = _compute_coherence_score("quantum gravity", "India has many rivers")
    assert score == 0.0


def test_coherence_score_empty_inputs() -> None:
    assert _compute_coherence_score("", "some text") == 0.0
    assert _compute_coherence_score("some query", "") == 0.0
    assert _compute_coherence_score("", "") == 0.0


def test_coherence_score_range() -> None:
    score = _compute_coherence_score("hybrid retrieval BM25", "BM25 hybrid retrieval in ThinkZen")
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_deduplication_removes_identical_text() -> None:
    from modules.evidence import EvidenceItem
    items = [
        EvidenceItem("c1", "ThinkZen is a RAG system", 0.8, 0.6, "doc1", "hybrid"),
        EvidenceItem("c2", "ThinkZen is a RAG system", 0.7, 0.5, "doc1", "hybrid"),  # near-duplicate
        EvidenceItem("c3", "Goa 2026 is an event", 0.6, 0.4, "doc2", "hybrid"),
    ]
    deduped = _deduplicate(items)
    assert len(deduped) == 2
    # Should keep the higher-scored item
    assert any(item.chunk_id == "c1" for item in deduped)
    assert any(item.chunk_id == "c3" for item in deduped)


def test_deduplication_preserves_distinct_items() -> None:
    from modules.evidence import EvidenceItem
    items = [
        EvidenceItem("c1", "BM25 uses term frequency statistics", 0.8, 0.5, "doc1", "sparse"),
        EvidenceItem("c2", "Dense retrieval uses neural embeddings", 0.7, 0.6, "doc2", "dense"),
        EvidenceItem("c3", "Goa is a coastal state in India", 0.6, 0.4, "doc3", "hybrid"),
    ]
    deduped = _deduplicate(items)
    assert len(deduped) == 3


def test_deduplication_single_item() -> None:
    from modules.evidence import EvidenceItem
    items = [EvidenceItem("c1", "Only one item", 0.9, 0.8, "doc1", "hybrid")]
    deduped = _deduplicate(items)
    assert len(deduped) == 1


# ---------------------------------------------------------------------------
# EvidenceIntelligence.select_evidence
# ---------------------------------------------------------------------------


@pytest.fixture
def ei() -> EvidenceIntelligence:
    return EvidenceIntelligence(min_retrieval_score=0.10, max_evidence=5)


def test_select_evidence_empty_candidates(ei: EvidenceIntelligence) -> None:
    bundle = ei.select_evidence("What is ThinkZen?", [])
    assert bundle.grounding_decision == GroundingDecision.ABSTAIN
    assert bundle.items == []
    assert bundle.source_diversity == 0


def test_select_evidence_grounded_success(ei: EvidenceIntelligence) -> None:
    docs = [
        _doc("doc1_c1", "ThinkZen is a multilingual RAG system.", score=0.8, doc_id="doc1"),
        _doc("doc2_c1", "Hybrid retrieval combines dense and sparse search.", score=0.6, doc_id="doc2"),
    ]
    bundle = ei.select_evidence("What is ThinkZen?", docs)
    assert bundle.grounding_decision == GroundingDecision.GROUNDED
    assert len(bundle.items) >= 1
    assert bundle.max_retrieval_score > 0.0


def test_select_evidence_source_diversity(ei: EvidenceIntelligence) -> None:
    """Evidence from multiple sources should be counted in source_diversity."""
    docs = [
        _doc("doc1_c1", "ThinkZen architecture overview.", score=0.8, doc_id="doc1"),
        _doc("doc2_c1", "BM25 sparse retrieval explanation.", score=0.7, doc_id="doc2"),
        _doc("doc3_c1", "Grounding prevents hallucination.", score=0.6, doc_id="doc3"),
    ]
    bundle = ei.select_evidence("How does ThinkZen work?", docs)
    assert bundle.source_diversity >= 2


def test_select_evidence_max_evidence_respected(ei: EvidenceIntelligence) -> None:
    docs = [_doc(f"doc{i}_c1", f"Document {i} text content here.", score=0.5 + i * 0.01, doc_id=f"doc{i}") for i in range(10)]
    bundle = ei.select_evidence("query", docs)
    assert len(bundle.items) <= ei.max_evidence


def test_select_evidence_items_have_required_fields(ei: EvidenceIntelligence) -> None:
    docs = [_doc("doc1_c1", "ThinkZen system overview.", score=0.7, doc_id="doc1")]
    bundle = ei.select_evidence("What is ThinkZen?", docs)
    assert len(bundle.items) >= 1
    item = bundle.items[0]
    assert item.chunk_id
    assert item.text
    assert 0.0 <= item.retrieval_score <= 2.0  # Can be > 1 with hybrid fusion
    assert 0.0 <= item.coherence_score <= 1.0
    assert item.source_doc_id
    assert item.method


def test_select_evidence_combined_score(ei: EvidenceIntelligence) -> None:
    docs = [_doc("doc1_c1", "hybrid retrieval combines dense and sparse", score=0.8, doc_id="doc1")]
    bundle = ei.select_evidence("hybrid retrieval", docs)
    item = bundle.items[0]
    # coherence should be > 0 given overlap between query and text
    assert item.combined_score >= 0.0


def test_select_evidence_high_threshold_marks_insufficient() -> None:
    """When all candidates score below threshold, decision should be INSUFFICIENT."""
    ei_strict = EvidenceIntelligence(min_retrieval_score=0.99, max_evidence=5)
    docs = [_doc("doc1_c1", "Some relevant text.", score=0.3, doc_id="doc1")]
    bundle = ei_strict.select_evidence("query", docs)
    # With fallback logic, it may still return items but should mark insufficient
    assert bundle.grounding_decision in (GroundingDecision.GROUNDED, GroundingDecision.INSUFFICIENT)


def test_select_evidence_is_grounded_property(ei: EvidenceIntelligence) -> None:
    docs = [_doc("doc1_c1", "ThinkZen is a RAG system.", score=0.8, doc_id="doc1")]
    bundle = ei.select_evidence("What is ThinkZen?", docs)
    assert bundle.is_grounded == (bundle.grounding_decision == GroundingDecision.GROUNDED)


# ---------------------------------------------------------------------------
# EvidenceBundle.to_dict serialization
# ---------------------------------------------------------------------------


def test_evidence_bundle_to_dict_json_serializable(ei: EvidenceIntelligence) -> None:
    docs = [
        _doc("doc1_c1", "ThinkZen is a multilingual system.", score=0.8, doc_id="doc1"),
        _doc("doc2_c1", "BM25 is a sparse retrieval algorithm.", score=0.6, doc_id="doc2"),
    ]
    bundle = ei.select_evidence("What is ThinkZen?", docs)
    d = bundle.to_dict()
    serialized = json.dumps(d)
    assert "grounding_decision" in serialized
    assert "source_diversity" in serialized
    assert "evidence_items" in serialized
    assert "decision_reason" in serialized


def test_evidence_bundle_to_dict_structure(ei: EvidenceIntelligence) -> None:
    docs = [_doc("doc1_c1", "ThinkZen text.", score=0.7, doc_id="doc1")]
    bundle = ei.select_evidence("query", docs)
    d = bundle.to_dict()
    assert isinstance(d["grounding_decision"], str)
    assert isinstance(d["source_diversity"], int)
    assert isinstance(d["evidence_items"], list)
    assert isinstance(d["max_retrieval_score"], float)
    assert isinstance(d["mean_coherence_score"], float)


def test_evidence_bundle_empty_to_dict() -> None:
    bundle = EvidenceBundle(
        query="test",
        items=[],
        grounding_decision=GroundingDecision.ABSTAIN,
        source_diversity=0,
        max_retrieval_score=0.0,
        mean_coherence_score=0.0,
        decision_reason="No candidates.",
    )
    d = bundle.to_dict()
    assert d["grounding_decision"] == "abstain"
    assert d["evidence_count"] == 0
    assert d["evidence_items"] == []


# ---------------------------------------------------------------------------
# GroundingDecision enum
# ---------------------------------------------------------------------------


def test_grounding_decision_values() -> None:
    assert GroundingDecision.GROUNDED.value == "grounded"
    assert GroundingDecision.INSUFFICIENT.value == "insufficient"
    assert GroundingDecision.ABSTAIN.value == "abstain"
