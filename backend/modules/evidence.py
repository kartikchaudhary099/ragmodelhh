"""Evidence Intelligence — explicit evidence selection and scoring before generation.

This module implements the EvidenceBundle and EvidenceIntelligence classes that sit
between the reranking layer and the generation layer. It makes evidence selection an
explicit, inspectable, measurable step rather than an implicit behaviour inside the
generator.

Key capabilities:
- Deduplication: remove near-duplicate evidence passages
- Source diversity: prefer evidence from multiple source documents
- Coherence scoring: estimate how well the evidence answers the query
- Coverage tracking: how many distinct source docs contributed evidence
- Grounding decision: GROUNDED / INSUFFICIENT / ABSTAIN

This makes Judge Mode output richer and makes the grounding contract testable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from modules.retrieval import RetrievedDocument

logger = logging.getLogger(__name__)


class GroundingDecision(str, Enum):
    """Outcome of the evidence sufficiency check."""
    GROUNDED = "grounded"           # Sufficient evidence found
    INSUFFICIENT = "insufficient"   # Evidence present but below quality threshold
    ABSTAIN = "abstain"             # No evidence retrieved at all


@dataclass
class EvidenceItem:
    """A single piece of selected evidence with scoring metadata."""
    chunk_id: str
    text: str
    retrieval_score: float
    coherence_score: float      # Estimated relevance to query (0.0–1.0)
    source_doc_id: str          # Parent document identifier
    method: str                 # How it was retrieved ("dense", "sparse", "hybrid", "reranked")
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def combined_score(self) -> float:
        """Weighted combination of retrieval and coherence scores."""
        return 0.6 * self.retrieval_score + 0.4 * self.coherence_score


@dataclass
class EvidenceBundle:
    """Full evidence package produced before generation.

    This is the contract between the retrieval/reranking layer and the generation layer.
    The generator must use ONLY the evidence in this bundle.
    """
    query: str
    items: list[EvidenceItem]
    grounding_decision: GroundingDecision
    source_diversity: int           # Number of distinct source documents represented
    max_retrieval_score: float
    mean_coherence_score: float
    decision_reason: str            # Human-readable explanation for Judge Mode
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_grounded(self) -> bool:
        return self.grounding_decision == GroundingDecision.GROUNDED

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation for Judge Mode telemetry."""
        return {
            "grounding_decision": self.grounding_decision.value,
            "source_diversity": self.source_diversity,
            "max_retrieval_score": round(self.max_retrieval_score, 4),
            "mean_coherence_score": round(self.mean_coherence_score, 4),
            "evidence_count": len(self.items),
            "decision_reason": self.decision_reason,
            "evidence_items": [
                {
                    "chunk_id": item.chunk_id,
                    "source_doc_id": item.source_doc_id,
                    "retrieval_score": round(item.retrieval_score, 4),
                    "coherence_score": round(item.coherence_score, 4),
                    "combined_score": round(item.combined_score, 4),
                    "method": item.method,
                    "text_preview": item.text[:120] + "..." if len(item.text) > 120 else item.text,
                }
                for item in self.items
            ],
        }


# ---------------------------------------------------------------------------
# Coherence scoring (deterministic, no external calls)
# ---------------------------------------------------------------------------


def _compute_coherence_score(query: str, text: str) -> float:
    """Estimate how relevant a passage is to a query using token overlap.

    This is a fast, deterministic heuristic. It does NOT replace semantic similarity
    but provides a measurable signal for evidence selection.

    Returns a float in [0.0, 1.0].
    """
    if not query or not text:
        return 0.0

    def tokenize(s: str) -> set[str]:
        return {t.lower() for t in re.findall(r"[\w\u0900-\u097F]+", s) if len(t) >= 2}

    query_tokens = tokenize(query)
    text_tokens = tokenize(text)

    if not query_tokens or not text_tokens:
        return 0.0

    # Jaccard-like overlap: intersection / query length (recall-focused)
    overlap = len(query_tokens & text_tokens)
    return min(1.0, overlap / max(1, len(query_tokens)))


def _deduplicate(items: list[EvidenceItem], similarity_threshold: float = 0.85) -> list[EvidenceItem]:
    """Remove near-duplicate evidence items using token-level Jaccard similarity.

    Keeps the item with the higher combined_score when a near-duplicate pair is found.
    """
    if len(items) <= 1:
        return items

    def tokenize(s: str) -> set[str]:
        return {t.lower() for t in re.findall(r"[\w\u0900-\u097F]+", s)}

    kept: list[EvidenceItem] = []
    tokenized: list[set[str]] = []

    for item in sorted(items, key=lambda x: x.combined_score, reverse=True):
        tokens = tokenize(item.text)
        is_duplicate = False
        for existing_tokens in tokenized:
            union = existing_tokens | tokens
            if not union:
                continue
            jaccard = len(existing_tokens & tokens) / len(union)
            if jaccard >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(item)
            tokenized.append(tokens)

    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class EvidenceIntelligence:
    """Selects, deduplicates, scores, and evaluates evidence before generation.

    Usage::

        ei = EvidenceIntelligence(min_score=0.15, max_evidence=5)
        bundle = ei.select_evidence(query="What is ThinkZen?", candidates=retrieved_docs)
        if bundle.is_grounded:
            # Pass bundle.items to generator
            ...
        print(bundle.to_dict())  # Full Judge Mode trace
    """

    def __init__(
        self,
        min_retrieval_score: float = 0.10,
        min_coherence_score: float = 0.0,
        max_evidence: int = 5,
        deduplicate: bool = True,
        diversity_boost: bool = True,
    ) -> None:
        """Initialize EvidenceIntelligence.

        Args:
            min_retrieval_score: Minimum retrieval score to include a candidate.
            min_coherence_score: Minimum coherence score to include a candidate.
            max_evidence: Maximum evidence items to include in the bundle.
            deduplicate: Whether to remove near-duplicate passages.
            diversity_boost: Prefer evidence from multiple source documents.
        """
        self.min_retrieval_score = max(0.0, min_retrieval_score)
        self.min_coherence_score = max(0.0, min_coherence_score)
        self.max_evidence = max(1, max_evidence)
        self.deduplicate = deduplicate
        self.diversity_boost = diversity_boost

    def select_evidence(
        self,
        query: str,
        candidates: list[RetrievedDocument],
    ) -> EvidenceBundle:
        """Select and evaluate evidence from retrieved candidates.

        Args:
            query: Original user query text.
            candidates: Retrieved and optionally reranked document list.

        Returns:
            EvidenceBundle with grounding decision, scored items, and metadata.
        """
        if not candidates:
            return EvidenceBundle(
                query=query,
                items=[],
                grounding_decision=GroundingDecision.ABSTAIN,
                source_diversity=0,
                max_retrieval_score=0.0,
                mean_coherence_score=0.0,
                decision_reason="No candidate documents were retrieved.",
            )

        # Step 1: Score coherence and build EvidenceItem list
        scored_items: list[EvidenceItem] = []
        for doc in candidates:
            coherence = _compute_coherence_score(query, doc.text)
            source_doc_id = (
                doc.metadata.get("doc_id")
                or doc.metadata.get("source_id")
                or doc.chunk_id.split("_c")[0]
                or doc.chunk_id
            )
            scored_items.append(
                EvidenceItem(
                    chunk_id=doc.chunk_id,
                    text=doc.text,
                    retrieval_score=doc.score,
                    coherence_score=coherence,
                    source_doc_id=str(source_doc_id),
                    method=doc.method,
                    metadata=dict(doc.metadata),
                )
            )

        # Step 2: Filter by minimum scores
        filtered = [
            item for item in scored_items
            if item.retrieval_score >= self.min_retrieval_score
            and item.coherence_score >= self.min_coherence_score
        ]

        if not filtered:
            # Fallback: use best available even if below threshold
            filtered = sorted(scored_items, key=lambda x: x.retrieval_score, reverse=True)[:1]
            logger.debug(
                "All candidates below score thresholds. Using top candidate as fallback."
            )

        # Step 3: Deduplication
        if self.deduplicate:
            filtered = _deduplicate(filtered)

        # Step 4: Diversity boost — prefer items from distinct source documents
        if self.diversity_boost and len(filtered) > 1:
            filtered = self._apply_diversity_boost(filtered)

        # Step 5: Trim to max_evidence
        final_items = sorted(filtered, key=lambda x: x.combined_score, reverse=True)[: self.max_evidence]

        # Step 6: Grounding decision
        max_retrieval = max((item.retrieval_score for item in final_items), default=0.0)
        mean_coherence = (
            sum(item.coherence_score for item in final_items) / len(final_items)
            if final_items else 0.0
        )
        source_diversity = len({item.source_doc_id for item in final_items})

        if max_retrieval >= self.min_retrieval_score and final_items:
            decision = GroundingDecision.GROUNDED
            reason = (
                f"Found {len(final_items)} evidence item(s) from {source_diversity} "
                f"source(s). Max retrieval score: {max_retrieval:.3f}."
            )
        else:
            decision = GroundingDecision.INSUFFICIENT
            reason = (
                f"Evidence quality below threshold (max_score={max_retrieval:.3f}, "
                f"threshold={self.min_retrieval_score:.3f})."
            )

        logger.info(
            "EvidenceIntelligence: decision=%s items=%d diversity=%d max_score=%.3f",
            decision.value,
            len(final_items),
            source_diversity,
            max_retrieval,
        )

        return EvidenceBundle(
            query=query,
            items=final_items,
            grounding_decision=decision,
            source_diversity=source_diversity,
            max_retrieval_score=max_retrieval,
            mean_coherence_score=mean_coherence,
            decision_reason=reason,
        )

    def _apply_diversity_boost(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        """Re-order items to maximize source document diversity in the first positions.

        Keeps total item count unchanged; just reorders so the top items come from
        different source documents where possible.
        """
        reordered: list[EvidenceItem] = []
        seen_sources: set[str] = set()
        remaining: list[EvidenceItem] = list(items)

        # First pass: one from each source
        for item in sorted(remaining, key=lambda x: x.combined_score, reverse=True):
            if item.source_doc_id not in seen_sources:
                reordered.append(item)
                seen_sources.add(item.source_doc_id)

        # Second pass: fill remaining slots with highest-scoring items not yet included
        included_ids = {item.chunk_id for item in reordered}
        for item in sorted(remaining, key=lambda x: x.combined_score, reverse=True):
            if item.chunk_id not in included_ids:
                reordered.append(item)
                included_ids.add(item.chunk_id)

        return reordered


__all__ = [
    "EvidenceIntelligence",
    "EvidenceBundle",
    "EvidenceItem",
    "GroundingDecision",
]
