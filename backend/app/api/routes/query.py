"""Query API endpoint — full ThinkZen pipeline with Query Analysis, Evidence Intelligence,
Hybrid Retrieval, Reranking, Grounded Generation, and Judge Mode telemetry.

Pipeline stages:
  1. Query Analysis   → language + type detection, adaptive parameter selection
  2. Hybrid Retrieval → dense + sparse BM25 with alpha-weighted fusion
  3. Reranking        → FlashRank cross-encoder (with fallback to score sort)
  4. Evidence Intel   → deduplication, diversity, coherence scoring, grounding decision
  5. Generation       → evidence-backed answer or intelligent refusal
  6. Telemetry        → per-request + multi-run P50/P70/P100 latency analytics
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from modules.evidence import EvidenceIntelligence
from modules.embeddings import HashingEmbeddingProvider
from modules.generation.grounded_generator import GroundedGenerator
from modules.official_corpus import (
    CORPUS_OFFICIAL,
    OfficialCorpusUnavailable,
    load_and_seed_official_data,
)
from modules.query_analyzer import QueryAnalyzer
from modules.reranking.flashrank_reranker import FlashRankReranker
from modules.retrieval import OrchestratedHybridRetriever
from modules.sample_seeder import load_and_seed_demo_data
from modules.telemetry import LatencyAggregator, PipelineRun, RunLabel, get_aggregator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["query"])

# Global in-memory pipeline state initialized lazily on first request
_dense_store = None
_sparse_store = None
_hybrid_retriever = None
# Shared, dependency-free embedding provider. Using ONE instance for both seeding and
# querying guarantees document and query vectors live in the same space/dimension, so
# dense retrieval is meaningful (the previous code left the retriever without a provider,
# collapsing queries to a degenerate 3-dim vector against 128-dim document vectors).
_embedding_provider = HashingEmbeddingProvider()
_reranker = FlashRankReranker()
_grounded_generator = GroundedGenerator()
_query_analyzer = QueryAnalyzer()
_evidence_intelligence = EvidenceIntelligence(min_retrieval_score=0.10, max_evidence=5)


async def get_or_init_pipeline() -> OrchestratedHybridRetriever:
    """Lazily initialize stores and hybrid retriever with the configured corpus.

    Corpus selection is driven by ``settings.corpus_mode`` (env ``THINKZEN_CORPUS``):
      * ``"official"`` seeds the validated ai4bharat/MSMARCO-XI subset. If the artifact is
        missing or invalid this RAISES (surfaced as HTTP 503) — it never silently falls
        back to demo data while claiming MSMARCO-XI.
      * anything else (default ``"demo"``) seeds the explicitly-labelled demo corpus,
        preserving the previous default behaviour exactly.
    """
    global _dense_store, _sparse_store, _hybrid_retriever
    if _hybrid_retriever is None:
        settings = get_settings()
        corpus_mode = (settings.corpus_mode or "demo").strip().lower()

        if corpus_mode == CORPUS_OFFICIAL:
            logger.info("Corpus mode = OFFICIAL: seeding validated ai4bharat/MSMARCO-XI subset")
            try:
                _dense_store, _sparse_store, count, provenance = await load_and_seed_official_data(
                    sample_path=settings.official_sample_path,
                    provenance_path=settings.official_provenance_path,
                    embedding_provider=_embedding_provider,
                )
            except OfficialCorpusUnavailable as exc:
                # Fail loud and honest — do NOT fall back to demo while claiming MSMARCO-XI.
                logger.error("Official corpus requested but unavailable: %s", exc)
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            logger.info(
                "OFFICIAL corpus ready: %d chunks (dataset=%s, source_file=%s)",
                count, provenance.get("dataset"), provenance.get("source_file"),
            )
        else:
            _dense_store, _sparse_store, _ = await load_and_seed_demo_data(
                embedding_provider=_embedding_provider,
            )

        _hybrid_retriever = OrchestratedHybridRetriever(
            dense_store=_dense_store,
            sparse_store=_sparse_store,
            alpha=0.5,
            embedding_provider=_embedding_provider,
        )
    return _hybrid_retriever


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Query payload specification."""
    query: str = Field(
        ...,
        description="Text query or STT transcript text",
        min_length=1,
        max_length=4096,
    )
    alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Override hybrid retrieval dense weighting (0.0=sparse, 1.0=dense). "
            "If None, the Query Analyzer selects an adaptive alpha automatically."
        ),
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description=(
            "Override maximum candidate document count. "
            "If None, the Query Analyzer selects top_k based on query complexity."
        ),
    )
    confidence_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Minimum retrieval score required for grounded generation.",
    )
    enable_reranker: bool = Field(
        default=True,
        description="Enable post-retrieval FlashRank candidate reranking.",
    )
    use_adaptive_retrieval: bool = Field(
        default=True,
        description=(
            "If True, the Query Analyzer auto-selects alpha and top_k. "
            "Explicit alpha/top_k fields override this when provided."
        ),
    )


class SourceDocument(BaseModel):
    """Retrieved evidence document representation."""
    chunk_id: str
    text: str
    score: float
    method: str
    metadata: dict[str, Any]


class QueryAnalysisInfo(BaseModel):
    """Query Analyzer output — visible in Judge Mode."""
    language: str
    query_type: str
    complexity: str
    keywords: list[str]
    adaptive_alpha: float
    adaptive_top_k: int
    strategy_name: str
    rationale: str


class EvidenceBundleInfo(BaseModel):
    """Evidence Intelligence output — visible in Judge Mode."""
    grounding_decision: str
    source_diversity: int
    max_retrieval_score: float
    mean_coherence_score: float
    evidence_count: int
    decision_reason: str


class TelemetryData(BaseModel):
    """Judge Mode execution latency and pipeline flow telemetry."""
    run_id: str
    total_latency_ms: float
    query_analysis_latency_ms: float
    retrieval_latency_ms: float
    rerank_latency_ms: float
    evidence_latency_ms: float
    generation_latency_ms: float
    candidate_count: int
    evidence_count: int
    alpha_used: float
    top_k_used: int
    alpha_source: str          # "adaptive" | "override"
    grounding_status: str
    detected_language: str
    query_analysis: QueryAnalysisInfo
    evidence_bundle: EvidenceBundleInfo


class QueryResponse(BaseModel):
    """Unified query response with answer, evidence, and full telemetry."""
    query: str
    answer: str
    refused: bool
    refusal_reason: str | None
    sources: list[SourceDocument]
    telemetry: TelemetryData


# ---------------------------------------------------------------------------
# Main query endpoint
# ---------------------------------------------------------------------------


@router.post("/query", response_model=QueryResponse)
async def process_query(payload: QueryRequest) -> QueryResponse:
    """Process user query through the full ThinkZen pipeline.

    Pipeline:
      1. Query Analysis — language, type, adaptive parameter selection
      2. Hybrid Retrieval — dense + sparse BM25, alpha-weighted fusion
      3. Candidate Reranking — FlashRank (fallback to score sort)
      4. Evidence Intelligence — dedup, diversity, coherence, grounding decision
      5. Grounded Generation — evidence-backed answer or intelligent refusal
      6. Telemetry — per-request latency + emit to multi-run aggregator
    """
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty or whitespace.")

    run_id = str(uuid.uuid4())[:8]
    total_start = time.perf_counter()

    # -------------------------------------------------------------------
    # Stage 1: Query Analysis
    # -------------------------------------------------------------------
    qa_start = time.perf_counter()
    analysis = _query_analyzer.analyze(query_text)
    qa_latency = (time.perf_counter() - qa_start) * 1000.0

    alpha_source = "adaptive"
    if payload.alpha is not None:
        effective_alpha = payload.alpha
        alpha_source = "override"
    elif payload.use_adaptive_retrieval:
        effective_alpha = analysis.retrieval_strategy.alpha
    else:
        effective_alpha = 0.5

    effective_top_k = (
        payload.top_k
        if payload.top_k is not None
        else (analysis.retrieval_strategy.top_k if payload.use_adaptive_retrieval else 5)
    )

    logger.info(
        "[%s] Query Analysis: lang=%s type=%s alpha=%.2f top_k=%d source=%s",
        run_id, analysis.language.value, analysis.query_type.value,
        effective_alpha, effective_top_k, alpha_source,
    )

    # -------------------------------------------------------------------
    # Stage 2: Hybrid Retrieval
    # -------------------------------------------------------------------
    retriever = await get_or_init_pipeline()
    retriever.alpha = effective_alpha

    retrieval_start = time.perf_counter()
    retrieved_docs = await retriever.retrieve(query_text, top_k=effective_top_k)
    retrieval_latency = (time.perf_counter() - retrieval_start) * 1000.0

    logger.info("[%s] Retrieval: %d candidates in %.1fms", run_id, len(retrieved_docs), retrieval_latency)

    # -------------------------------------------------------------------
    # Stage 3: Candidate Reranking
    # -------------------------------------------------------------------
    rerank_start = time.perf_counter()
    reranked_candidates = (
        await _reranker.rerank(query_text, retrieved_docs, top_k=effective_top_k)
        if payload.enable_reranker and retrieved_docs
        else retrieved_docs
    )
    rerank_latency = (time.perf_counter() - rerank_start) * 1000.0

    # -------------------------------------------------------------------
    # Stage 4: Evidence Intelligence
    # -------------------------------------------------------------------
    evidence_start = time.perf_counter()
    _evidence_intelligence.min_retrieval_score = payload.confidence_threshold
    evidence_bundle = _evidence_intelligence.select_evidence(query_text, reranked_candidates)
    evidence_latency = (time.perf_counter() - evidence_start) * 1000.0

    logger.info(
        "[%s] Evidence: %s items=%d diversity=%d",
        run_id, evidence_bundle.grounding_decision.value,
        len(evidence_bundle.items), evidence_bundle.source_diversity,
    )

    # -------------------------------------------------------------------
    # Stage 5: Grounded Generation
    # -------------------------------------------------------------------
    generation_start = time.perf_counter()
    _grounded_generator.confidence_threshold = payload.confidence_threshold
    # Use evidence items (already selected & scored) to guide generation
    generation_context = reranked_candidates  # Generator applies its own threshold
    # Pass the analyzer-detected language so answer-language selection is decoupled from
    # mere script detection (English/Hinglish queries are not answered in Devanagari Hindi).
    answer_obj = await _grounded_generator.generate(
        query_text, generation_context, language=analysis.language.value
    )
    generation_latency = (time.perf_counter() - generation_start) * 1000.0

    total_latency = (time.perf_counter() - total_start) * 1000.0

    logger.info(
        "[%s] Pipeline complete: refused=%s total=%.1fms",
        run_id, answer_obj.refused, total_latency,
    )

    # -------------------------------------------------------------------
    # Stage 6: Record telemetry run
    # -------------------------------------------------------------------
    aggregator = get_aggregator()
    aggregator.record(PipelineRun(
        query_id=run_id,
        label=RunLabel.REAL_RUN,
        total_ms=total_latency,
        query_analysis_ms=qa_latency,
        retrieval_ms=retrieval_latency,
        rerank_ms=rerank_latency,
        generation_ms=generation_latency,
        refused=answer_obj.refused,
        grounding_status="refused" if answer_obj.refused else "grounded",
        alpha_used=effective_alpha,
        query_type=analysis.query_type.value,
        language=analysis.language.value,
        evidence_count=len(evidence_bundle.items),
    ))

    # -------------------------------------------------------------------
    # Build response
    # -------------------------------------------------------------------
    sources_output = [
        SourceDocument(
            chunk_id=doc.chunk_id,
            text=doc.text,
            score=round(doc.score, 4),
            method=doc.method,
            metadata=doc.metadata,
        )
        for doc in answer_obj.sources
    ]

    telemetry = TelemetryData(
        run_id=run_id,
        total_latency_ms=round(total_latency, 2),
        query_analysis_latency_ms=round(qa_latency, 2),
        retrieval_latency_ms=round(retrieval_latency, 2),
        rerank_latency_ms=round(rerank_latency, 2),
        evidence_latency_ms=round(evidence_latency, 2),
        generation_latency_ms=round(generation_latency, 2),
        candidate_count=len(retrieved_docs),
        evidence_count=len(evidence_bundle.items),
        alpha_used=round(effective_alpha, 4),
        top_k_used=effective_top_k,
        alpha_source=alpha_source,
        grounding_status="refused" if answer_obj.refused else "grounded",
        detected_language=analysis.language.value,
        query_analysis=QueryAnalysisInfo(
            language=analysis.language.value,
            query_type=analysis.query_type.value,
            complexity=analysis.complexity.value,
            keywords=analysis.keywords,
            adaptive_alpha=analysis.retrieval_strategy.alpha,
            adaptive_top_k=analysis.retrieval_strategy.top_k,
            strategy_name=analysis.retrieval_strategy.strategy_name,
            rationale=analysis.retrieval_strategy.rationale,
        ),
        evidence_bundle=EvidenceBundleInfo(
            grounding_decision=evidence_bundle.grounding_decision.value,
            source_diversity=evidence_bundle.source_diversity,
            max_retrieval_score=round(evidence_bundle.max_retrieval_score, 4),
            mean_coherence_score=round(evidence_bundle.mean_coherence_score, 4),
            evidence_count=len(evidence_bundle.items),
            decision_reason=evidence_bundle.decision_reason,
        ),
    )

    return QueryResponse(
        query=query_text,
        answer=answer_obj.text,
        refused=answer_obj.refused,
        refusal_reason=answer_obj.refusal_reason,
        sources=sources_output,
        telemetry=telemetry,
    )


# ---------------------------------------------------------------------------
# Judge Mode endpoint
# ---------------------------------------------------------------------------


class JudgeResponse(BaseModel):
    """Full pipeline analytics for Judge Mode."""
    total_runs: int
    latency_stats: dict[str, Any]
    recent_runs: list[dict[str, Any]]
    data_quality_note: str


@router.get("/judge", response_model=JudgeResponse, tags=["judge"])
async def get_judge_stats() -> JudgeResponse:
    """Return full pipeline analytics for Judge Mode.

    Includes:
    - P50/P70/P90/P100 latency statistics across all recorded runs
    - Breakdown by label (REAL_RUN vs UNIT_TEST)
    - Last 10 runs with per-stage timing
    - Data quality labels (never fabricated)
    """
    aggregator = get_aggregator()
    stats = aggregator.get_summary()
    recent = aggregator.get_all_runs()[-10:]

    recent_dicts = [
        {
            "run_id": r.query_id,
            "label": r.label.value,
            "total_ms": round(r.total_ms, 2),
            "query_analysis_ms": round(r.query_analysis_ms, 2),
            "retrieval_ms": round(r.retrieval_ms, 2),
            "rerank_ms": round(r.rerank_ms, 2),
            "generation_ms": round(r.generation_ms, 2),
            "refused": r.refused,
            "grounding_status": r.grounding_status,
            "alpha_used": r.alpha_used,
            "query_type": r.query_type,
            "language": r.language,
            "evidence_count": r.evidence_count,
        }
        for r in reversed(recent)
    ]

    note = (
        "REAL_RUN statistics reflect actual live pipeline executions. "
        "P50/P70/P90/P100 are only statistically meaningful with ≥10 runs per label. "
        "No metrics are fabricated."
    )

    return JudgeResponse(
        total_runs=aggregator.total_run_count,
        latency_stats=stats,
        recent_runs=recent_dicts,
        data_quality_note=note,
    )
