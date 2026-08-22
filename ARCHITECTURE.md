# Architecture — ThinkZen

> Architecture lock for the current project phase. This document describes the approved system design direction and the implementation constraints for Phase 2 and beyond.

> **⚠️ Runtime reality (read this first).** This document captures the *design direction*;
> several items below are still marked `[VERIFY BEFORE IMPLEMENTATION]` and describe intent,
> not what the shipping code does. The **actual implemented runtime** is:
> - **Vector store: in-memory.** Retrieval runs against in-memory dense + sparse (BM25)
>   stores seeded at startup (`backend/modules/sample_seeder.py`). **Qdrant is present as an
>   optional, non-wired path only** (`QdrantVectorStore`, imported lazily) — it is **not** the
>   default runtime.
> - **Embeddings: dependency-free feature-hashing** (`HashingEmbeddingProvider`) by default.
>   The BGE-M3 / `sentence-transformers` provider exists but is optional and only used when
>   explicitly selected. `torch` is not required to run.
> - **Reranking / LLM generation / Sarvam STT are optional.** Each imports its heavy
>   dependency lazily inside `try/except`; absent that dependency or its API key, the system
>   uses a deterministic, honest fallback (score-sort rerank, evidence-quoting generation,
>   `success=false` STT — never a fabricated transcript).
> - **Corpus: a 20-document DEMO/SAMPLE set** (`data/samples/demo_docs.json`), never the
>   official `ai4bharat/MSMARCO-XI` benchmark. MSMARCO-XI runtime evaluation is PENDING
>   dataset availability; the T6 framework is built and labels demo runs `UNIT_TEST_DATA`.
> - **Latency is measured, never hard-coded.** Judge Mode reports P50/P70/P90/P100 from the
>   real request path. No `<200ms` figure is asserted anywhere in code as a guaranteed result.
> - **Streaming/SSE is design direction, not implemented** — the query endpoint returns a
>   single structured JSON response.
>
> See `README.md` and `PROJECT_STATE.md` for the authoritative shipping-state description.

## System overview

```
Voice Input
   ↓
Speech-to-Text (Sarvam) [VERIFY BEFORE IMPLEMENTATION]
   ↓
Query Analyzer
   ↓
Multilingual / Language-aware Query Processing
   ↓
Query-aware Adaptive Retrieval
   ↓
Multiple Chunk Representations + Parent-Child Chunking
   ↓
Dense + Sparse Hybrid Retrieval (Qdrant direction) [VERIFY BEFORE IMPLEMENTATION]
   ↓
Reranking (FlashRank direction) [VERIFY BEFORE IMPLEMENTATION]
   ↓
Evidence Intelligence
   ↓
Grounding / Evidence Sufficiency Check
   ↓
Intelligent Refusal when evidence is insufficient
   ↓
Swappable Generation
   ↓
Streaming Response / SSE
   ↓
Judge Mode
   ↓
Real latency analytics (P50 / P70 / P100)
```

## Scope and design principles

ThinkZen is a voice-first, multilingual RAG system built for a large corpus with evidence-grounded responses and measurable latency. The design deliberately avoids a generic chat UI and instead prioritizes:

- voice-native interaction
- multilingual retrieval quality
- adaptive retrieval behavior by query type
- explicit evidence provenance
- refusal when evidence is insufficient
- judge visibility into the full pipeline
- real metrics rather than assumed performance targets

## Architectural principles

### 1. Voice-first, not text-only

Voice is a first-class input mode, not a decorative microphone on a text chat. The system should preserve transcript quality, language detection, and user intent across the pipeline.

### 2. Evidence-first grounding

Answers should not be generated from model memory alone. Retrieval and reranking evidence must be explicit and inspectable.

### 3. Query-aware retrieval

The system should not use one static retrieval plan for every query. Query classification and strategy selection should adjust retrieval depth, fusion, and evidence selection policies.

### 4. Refusal is a capability

When evidence is insufficient, the system should refuse with an explanation instead of hallucinating. The exact thresholds and criteria must be calibrated experimentally; they are not hard-coded in this document [VERIFY BEFORE IMPLEMENTATION].

### 5. Judge Mode is part of the product

Judges need to inspect the full end-to-end flow, not just final answer text.

### 6. Latency is measured, not assumed

A target like `<200ms` is a performance target to be validated, not a guaranteed benchmark [VERIFY BEFORE IMPLEMENTATION].

---

## Component responsibilities

### 1. Voice / STT layer

**Status:** Architecture direction locked, implementation pending  
**Direction:** Sarvam STT [VERIFY BEFORE IMPLEMENTATION]

Responsibilities:
- accept voice input from frontend
- transcribe audio to text
- preserve language metadata and transcript timing
- provide partial or final transcript depending on UX mode
- route transcript to the query analyzer

Design constraints:
- no hard-coded API keys or vendor assumptions in repo docs
- no unverified endpoint or request/response contract should be treated as final

---

### 2. Query Analyzer

**Status:** Planned, not implemented

Responsibilities:
- normalize transcribed or typed user input
- detect or preserve language information
- identify query type (factual, comparative, procedural, ambiguous, multi-hop, etc.)
- produce retrieval intent for adaptive retrieval
- provide structured metadata for Judge Mode

Notes:
- This is not just a simple string-cleaning step; it is an important routing layer for multilingual and voice-first behavior.

---

### 3. Multilingual / language-aware processing

**Status:** Planned, not implemented

Responsibilities:
- preserve language metadata from STT and query analysis
- handle multilingual queries and mixed-language inputs
- support retrieval routes for cross-lingual or same-language retrieval paths
- detect when language ambiguity requires fallback behavior

This layer is essential for the competition requirement, but exact multilingual policy and fallback logic must be experimentally validated.

---

### 4. Query-aware Adaptive Retrieval

**Status:** Core design requirement

Responsibilities:
- choose retrieval strategy based on query characteristics
- vary retrieval depth, fusion weight, and index path by query type
- support different behaviors for factual vs ambiguous vs procedural queries

Examples of adaptive behavior (not hard-coded yet):
- different top-k values for short factual questions vs long explanatory queries
- hybrid retrieval weights adjusted for query type
- fallback or expanded evidence search when confidence is low

This is one of the core differentiators of the product and must not collapse into one generic retrieval setup.

---

### 5. Chunking and multiple representations

**Status:** Core design requirement

**Direction:** parent-child chunking with multiple representations, not a single naive chunk size

Responsibilities:
- segment source documents into parent and child chunks
- retain larger context for document-level retrieval and summaries
- retain granular child chunks for exact evidence retrieval
- store multiple representations of the same chunk or document, such as:
  - original text
  - normalized text
  - summary/abstraction
  - keyword-rich view
  - optional translated or cross-lingual view if appropriate

Design goals:
- improve retrieval coverage for multilingual documents
- preserve context around evidence
- reduce the failure mode of single fixed-size chunking

Important note:
- exact chunk sizes, overlap policies, and representation count must be set by evaluation rather than invented in this document [VERIFY BEFORE IMPLEMENTATION].

---

### 6. Dense + Sparse Hybrid Retrieval

**Status:** Core design requirement  
**Direction:** Qdrant-based hybrid retrieval [VERIFY BEFORE IMPLEMENTATION]

Responsibilities:
- dense vector retrieval for semantic similarity
- sparse retrieval for lexical and keyword relevance
- fusion strategy for merging results from both search modes
- metadata filtering where applicable

Expected behavior:
- dense retrieval addresses semantic intent and multilingual similarity
- sparse retrieval preserves exact keyword and entity recall
- hybrid fusion balances recall and precision

This layer is central to the project’s performance and retrieval quality, so fusion strategy and weighting need experimental calibration.

---

### 7. Reranking

**Status:** Core design requirement  
**Direction:** FlashRank direction [VERIFY BEFORE IMPLEMENTATION]

Responsibilities:
- reorder the hybrid retrieval candidates by query relevance
- prioritize evidence quality and information density
- reduce noisy or redundant low-value passages

Important constraint:
- ranking quality, bias, and latency must be measured experimentally; no fixed threshold value should be treated as confirmed.

---

### 8. Evidence Intelligence

**Status:** Core design requirement

Responsibilities:
- preserve retrieval metadata with each candidate
- detect redundancy and contradictions across retrieved evidence
- rank evidence by usefulness, not just similarity score
- maintain provenance links to source chunk IDs and document metadata
- support answer grounding and citation flow

This is a major differentiator from generic RAG and should be treated as a first-class component, not a side effect of retrieval.

---

### 9. Grounding / Evidence Sufficiency Check

**Status:** Core design requirement

Responsibilities:
- assess whether the retrieved evidence is sufficient for a confident answer
- detect unsupported claims or weak evidence coverage
- produce a structured evidence status for generation and refusal
- support answer safety and citation quality

This layer is the gate between retrieval and generation. Generation should not run blindly when evidence is thin or inconsistent.

---

### 10. Intelligent Refusal

**Status:** Core design requirement

Responsibilities:
- refuse when evidence is insufficient or contradictory
- explain the limitation clearly to the user
- avoid hallucination when retrieved evidence fails the confidence threshold

Key point:
- threshold values and refusal logic must be calibrated experimentally; they are not fixed in this architecture document [VERIFY BEFORE IMPLEMENTATION].

---

### 11. Swappable Generation Layer

**Status:** Core design requirement

Responsibilities:
- generate grounded answers using retrieved evidence and the query
- support swappable providers without rewriting the rest of the system
- produce structured output with citations and refusal state

This layer remains pluggable so the team can compare providers without forcing a monolithic design.

---

### 12. Streaming Response / SSE

**Status:** Core design requirement  
**Direction:** SSE direction [VERIFY BEFORE IMPLEMENTATION]

Responsibilities:
- stream answer generation when supported
- expose progress and intermediate status to the frontend
- provide judge visibility into answer-building stages
- maintain responsiveness for voice-first use cases

No exact SSE payload schema or event contract is treated as final yet; it must be defined during implementation.

---

### 13. Judge Mode

**Status:** Core design requirement

Responsibilities:
- show the user transcript
- show query analysis details
- show selected retrieval strategy and why it was chosen
- show retrieval hits and reranking decisions
- show grounding status and evidence support
- show per-stage latency and total latency
- expose any refusal decisions and evidence insufficiency explanation

Judge Mode is not optional debugging UI. It is part of the product story and necessarily part of the architecture.

Required fields to capture:
- transcript
- query analysis metadata
- retrieval mode / strategy
- retrieval candidates and scores
- reranking results
- evidence used for final answer
- grounding status
- stage-by-stage latency
- total latency

---

### 14. Latency Analytics

**Status:** Core design requirement

Responsibilities:
- record actual latency for each stage
- emit real P50 / P70 / P100 metrics from production-like runs
- instrument STT, query analysis, retrieval, rerank, generation, and final response assembly
- expose per-stage trace data to Judge Mode

Important rule:
- do not invent latency numbers or assume a performance guarantee [VERIFY BEFORE IMPLEMENTATION]
- latency targets must be measured in the evaluation phase using real runs

---

## Offline vs online pipeline

### Offline / ingestion-time pipeline

This is the planned pipeline for corpus preparation and index setup.

1. ingest raw documents
2. normalize and detect language
3. create parent and child chunk representations
4. generate or assign retrieval metadata
5. embed text representations with the chosen embedding model
6. store to the vector database with metadata and provenance
7. prepare retrieval policies and fusion strategy metadata

### Online / request-time pipeline

This is the runtime path for each question.

1. receive voice or text input
2. transcribe audio if needed
3. analyze query, detect language, identify intent
4. choose retrieval strategy adaptively
5. retrieve dense and sparse candidates
6. rerank candidates
7. evaluate evidence quality and sufficiency
8. refuse if necessary
9. generate grounded answer
10. stream response and expose Judge Mode metadata
11. record latency metrics

---

## Retrieval design

### Retrieval strategy

The system should support adaptive retrieval on a per-query basis, rather than a single static plan.

Candidate design patterns:
- query type determines retrieval depth and fusion weighting
- retrieval may expand or narrow evidence depending on ambiguity or missing support
- both exact lexical signals and dense semantic similarity are considered

### Dense + sparse fusion

Hybrid retrieval is required.

Recommended design principles:
- dense path handles semantic similarity and multilingual retrieval quality
- sparse path preserves exact lexical signal and entity recall
- fusion should be explicit and auditable, not hidden in opaque ranking

### Chunk and evidence selection

Each candidate should retain enough metadata for downstream reasoning:
- chunk ID
- parent document ID
- language
- source metadata
- score from dense retrieval
- score from sparse retrieval
- rerank score
- evidence provenance

This metadata is required for evidence intelligence, grounding, and Judge Mode.

---

## Evidence flow

The evidence flow should be explicit and auditable:

1. retrieval candidates are selected
2. reranker reorders them
3. evidence intelligence identifies supporting versus redundant passages
4. grounding system checks whether the evidence is sufficient
5. generation consumes only supported evidence
6. final answer includes provenance and refusal status when appropriate

The goal is to keep evidence visible to both the model and human judges.

---

## Failure handling

### Input failures
- missing transcript or very weak audio quality
- language detection ambiguity
- empty or malformed query

### Retrieval failures
- no relevant evidence returned
- contradictory evidence clusters
- very low confidence from reranker or retrieval stage

### Generation failures
- unsupported answers
- missing citation support
- hallucination risk due to low evidence quality

### Required behavior
- fail gracefully
- prefer explicit refusal when evidence is insufficient
- preserve the evidence trail and latency trace
- never hide the reason from the judge or user

---

## Latency instrumentation

The system should record and display actual timings at every stage.

Minimum instrumentation points:
- STT start → end
- query analysis start → end
- retrieval start → end
- reranking start → end
- evidence intelligence start → end
- grounding check start → end
- generation start → end
- stream delivery / response completion
- total end-to-end latency

Metrics to compute:
- P50
- P70
- P100
- per-stage latency for visible debugging

Important rule:
- real numbers must be measured during evaluation; never hard-coded or invented [VERIFY BEFORE IMPLEMENTATION].

---

## Judge Mode requirements

Judge Mode should expose a structured trace for each request.

Required fields:
- transcript
- language info
- query analysis output
- retrieval strategy chosen
- retrieval candidates and scores
- reranked order
- evidence selected for answer
- grounding verdict
- refusal status
- total latency
- per-stage latency breakdown

Judge Mode is part of the architecture and should be treated as a product requirement, not a late-stage afterthought.

---

## OUT-OF-THE-BOX DIFFERENTIATION

### 1. Query-aware / adaptive retrieval

Different query types should trigger different retrieval behavior. A one-size-fits-all retrieval policy is not acceptable for this project.

### 2. Multiple chunk representations / parent-child chunking

The system should not rely on one naive fixed-size chunker. The architecture explicitly favors parent-child and multi-representation retrieval strategies.

### 3. Evidence intelligence

Evidence should be selected, clustered, and surfaced based on usefulness and support, not just raw similarity.

### 4. Intelligent refusal

When evidence is insufficient, the system should refuse transparently rather than hallucinate. This is a product feature and an evaluation requirement.

### 5. Voice-native UX

Voice is first-class; the frontend and backend should be designed around voice-first behavior, not just text plus a microphone.

### 6. Judge Mode

The architecture makes the full reasoning process inspectable by design.

### 7. Real telemetry-first measurement

The system must collect real latency metrics and use them as product evidence. No fabricated benchmark values should be used.

---

## Verification required before implementation

The following items remain design-stage references and must be verified against official/current documentation before implementation is considered final:

- Sarvam STT integration requirements
- BGE-M3 embedding direction
- Qdrant retrieval and indexing pattern
- FlashRank integration expectations
- SSE contract and frontend event flow
- any exact latency target, threshold, or benchmark assumption
- any exact model identifiers, endpoints, parameters, or vendor-specific API details

This project is architecture-locked but not implementation-finalized. The repo is ready for engineering decisions, not for unverified production claims.

### 5. Voice-native UX

Design the entire interaction flow around voice: partial transcripts, barge-in, spoken citations, confirmation prompts for ambiguous queries.

**Why it matters:** Most RAG demos are text-first with voice bolted on; voice-native is a differentiator.

### 6. Judge Mode showing real pipeline / evidence / latency

A debug/evaluation UI that shows the full pipeline trace: what was retrieved, how it was reranked, what evidence was used, per-stage latency, and grounding scores.

**Why it matters:** Demonstrates engineering depth and aids evaluation tuning.

### 7. Real P50 / P70 / P100 latency analytics

Instrument every pipeline stage and report actual percentile latencies, not averages. Use this data to make architecture trade-offs visible and measurable.

**Why it matters:** Competition has latency requirements; average latency hides tail problems.

---

## Data flow (planned)

```
Voice/Text Input
    → STT (if voice)
    → Query Processing + Language Detection
    → Multilingual Routing
    → Hybrid Retrieval (dense + sparse)
    → Reranking
    → Guardrails (input already checked)
    → Grounding Verification
    → Generation (with citations)
    → Latency Analytics (recorded throughout)
    → Response to Frontend
```

## Directory structure reference

```
backend/
├── app/                  # FastAPI application (config, routes, logging)
│   ├── api/routes/       # HTTP endpoints
│   ├── config.py         # Environment settings
│   ├── exceptions.py     # Error handling
│   └── main.py           # App factory
└── modules/              # Swappable pipeline stages
    ├── stt/
    ├── chunking/
    ├── embeddings/
    ├── retrieval/
    ├── reranking/
    ├── generation/
    └── evaluation/
```

## Version history

| Date | Change |
|------|--------|
| 2026-08-14 | Initial architecture document created (foundation phase) |
