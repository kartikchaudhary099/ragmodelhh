# Master Context — ThinkZen

> Single source of truth for project intent. Read this before making any architectural or implementation decision.

## Competition task

**Event:** HH Goa 2026  
**Task:** Task 2 — Multilingual RAG over a large corpus  
**Project name:** ThinkZen

The competition requires building a retrieval-augmented generation system that can answer questions accurately and quickly across multiple languages, with emphasis on grounding answers in source documents rather than hallucinating.

## Project goal

Build a **production-grade, voice-native, multilingual RAG system** that:

1. Accepts voice and text queries in multiple languages
2. Retrieves relevant evidence from a large document corpus (10M+ records)
3. Generates grounded, cited answers
4. Measures and reports real latency at P50 / P70 / P100
5. Stands out from generic RAG demos through differentiated architecture choices

## Technical requirements (known)

| Requirement | Notes |
|-------------|-------|
| Multilingual support | Query and corpus may span multiple Indian and global languages |
| Large corpus | 10M+ document scale — not downloaded yet |
| Latency | Real P50/P70/P100 analytics required |
| Grounding | Answers must be evidence-backed |
| Voice input | STT integration expected |
| Evaluation | Retrieval quality, faithfulness, latency metrics |

## Approved architecture direction

The project has reached a documented architecture lock for Phase 2.

Voice
→ Speech-to-text
→ Query Analyzer
→ Multilingual / language-aware processing
→ Query-aware Adaptive Retrieval
→ Multiple Chunk Representations / Parent-Child Chunking
→ Dense + Sparse Hybrid Retrieval
→ Reranking
→ Evidence Intelligence
→ Grounding / Evidence Sufficiency Check
→ Intelligent Refusal when evidence is insufficient
→ Swappable Generation
→ Streaming response / SSE
→ Judge Mode
→ Real latency analytics

### Technology direction from research

- STT: Sarvam [VERIFY BEFORE IMPLEMENTATION]
- Embeddings: BGE-M3 direction [VERIFY BEFORE IMPLEMENTATION]
- Vector database: Qdrant [VERIFY BEFORE IMPLEMENTATION]
- Retrieval: hybrid dense + sparse retrieval [VERIFY BEFORE IMPLEMENTATION]
- Reranking: FlashRank direction [VERIFY BEFORE IMPLEMENTATION]
- Backend: Python / FastAPI
- Frontend: separate frontend application
- Generation: swappable provider
- Streaming: SSE direction [VERIFY BEFORE IMPLEMENTATION]

### Differentiators

1. Query-aware Adaptive Retrieval
2. Multiple Chunk Representations / Parent-Child Chunking
3. Evidence Intelligence
4. Intelligent Refusal when evidence is insufficient
5. Voice-native UX
6. Judge Mode
7. Real P50/P70/P100 latency analytics

## Current strategy

> **Status note (2026-08-21):** The phase plan below is the original charter. The
> implementation phase is now substantially complete — the end-to-end pipeline
> (query analysis → hybrid retrieval → evidence intelligence → grounded generation →
> Judge Mode telemetry), the single-page voice UI, and the test suite are all built.
> See `PROJECT_STATE.md` and `README.md` for the authoritative current state; this
> file is kept for original intent and differentiators.

### Phase 1 — Foundation ✅ Complete

- Clean repo structure with separated frontend/backend
- Modular pipeline interfaces (swappable components)
- Environment variable configuration
- Basic logging, error handling, health endpoint
- Test infrastructure
- Context documentation for AI handoff

### Phase 2 — Architecture Lock ✅ Current

- Finalize architecture direction for STT, query analysis, retrieval, chunking, reranking, generation, and voice UX
- Document offline vs online pipeline
- Document judge mode and latency instrumentation
- Define verification items before implementation

### Phase 3 — Implementation (next)

- Implement each pipeline stage against chosen interfaces
- Build frontend with voice-native UX
- Add Judge Mode and real latency analytics
- Run evaluation on competition metrics

## Important constraints

1. **Do NOT download the full 10M+ dataset** until the ingestion strategy is implemented and approved
2. **Do NOT call paid APIs** until providers are selected and budgeted
3. **Do NOT hard-code API keys** — use `.env` only
4. **Do NOT build a generic RAG demo** — differentiation is a core requirement
5. **Keep dependencies minimal** — add packages only when a stage is implemented
6. **Do NOT invent benchmark numbers** — measure real metrics during evaluation and latency phases
7. **Latency target must be measured, not assumed** — any target like `<200ms` is a measurement target, not a guaranteed benchmark [VERIFY BEFORE IMPLEMENTATION]
8. **Do not treat exact model names, parameters, thresholds, or endpoints as verified unless supported by official/current documentation** [VERIFY BEFORE IMPLEMENTATION]

## Key files

| File | Role |
|------|------|
| `ARCHITECTURE.md` | Final system design and architecture specification |
| `PROJECT_STATE.md` | Live progress tracker |
| `AI_HANDOFF.md` | Instructions for the next AI agent |
| `.env.example` | Environment variable template |
| `backend/modules/` | Swappable pipeline interfaces |

## Team / tooling notes

This project may be worked on across multiple AI assistants (ChatGPT, Cursor, Gemini, Claude). Always update `PROJECT_STATE.md` and `AI_HANDOFF.md` at the end of each session.

## Architecture lock summary

The repository is no longer a generic scaffold. It is now positioned as a voice-native multilingual RAG architecture with explicit differentiation and a measurement-first approach to latency and evidence quality.

The system should be treated as a research-to-production design that requires verification before production-grade implementation specifics are frozen.
