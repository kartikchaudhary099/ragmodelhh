# AI Handoff — ThinkZen

> **Purpose:** Enable seamless transfer between ChatGPT, Cursor, Gemini, and Claude.  
> Read this file first. Then read `MASTER_CONTEXT.md` and `PROJECT_STATE.md`.

---

## Current Status: End-to-end pipeline implemented — finalization/verification

> **Source of truth for current state is `PROJECT_STATE.md`.** This handoff file also
> retains the original phase history and the MSMARCO-XI provenance rules further down.

**Where the project actually is (2026-08-21):**
- The full pipeline is built and wired end to end: Query Analysis → Hybrid Retrieval
  (BM25 + dense, alpha fusion) → optional FlashRank rerank → Evidence Intelligence →
  Grounded Generation (language-aware answers + strict refusal) → Judge Mode telemetry.
- Single unified `POST /api/v1/query`, aggregate `GET /api/v1/judge`, honest optional
  `POST /api/v1/stt`, and a single-page voice UI served at `/`.
- T6 Evaluation Framework is runnable (`scripts/run_eval.py`) and unit-tested
  (`tests/test_evaluation_runner.py`); an in-process nine-case verifier is at
  `scripts/verify_pipeline.py`.
- Heavy dependencies (torch / sentence-transformers / flashrank / qdrant / LLM SDKs) are
  all **optional** with deterministic, honest fallbacks, so the default run is offline.
- **Test inventory: 146 test functions across 13 files (static inventory — see
  `PROJECT_STATE.md`).** The suite was NOT executed this session (the Linux workspace is
  network-isolated — its package proxy returns `403`, so `pytest`/`fastapi` cannot be
  installed there — and the venv is a Windows `.venv`). Run it locally to produce
  measured results.
- **Deployment config is present but not yet deployed.** A `Dockerfile` + `.dockerignore`
  exist at the repo root, and `httpx` was added to `backend/requirements.txt` so the
  containerized `/api/v1/stt` can call Sarvam when a key is set. The image has **not** been
  built or deployed in this environment (no Docker/network here); build/deploy on a machine
  that has them, then verify the public URL before calling it deployed.
- **Git initialized locally (see `PROJECT_STATE.md`); not pushed to GitHub** (no credentials
  available in this environment — exact push commands are documented for the user).

ThinkZen is a multilingual, voice-native RAG system for **HH Goa 2026 Task 2**. It must answer questions grounded in a large corpus (10M+ docs), support voice input, meet latency targets, and differentiate from generic RAG demos.

## Architecture lock (approved direction)

The project is now positioned around the following architecture flow:

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

### Approved technology direction

- STT: Sarvam [VERIFY BEFORE IMPLEMENTATION]
- Embeddings: BGE-M3 direction [VERIFY BEFORE IMPLEMENTATION]
- Vector database: Qdrant [VERIFY BEFORE IMPLEMENTATION]
- Retrieval: hybrid dense + sparse
- Reranking: FlashRank direction [VERIFY BEFORE IMPLEMENTATION]
- Backend: Python / FastAPI
- Frontend: separate frontend application
- Generation: swappable provider
- Streaming: SSE direction [VERIFY BEFORE IMPLEMENTATION]

## Next Immediate Steps (remaining work is runtime verification, not implementation)

The pipeline stages described in the old "Phase 5–9" plan below are **already
implemented**. The remaining work is to execute and confirm, then decide on deployment:

1. **Run the full test suite locally** (Windows venv):
   `.venv\Scripts\python.exe -m pytest -q`
2. **Run the in-process nine-case verifier:**
   `.venv\Scripts\python.exe scripts/verify_pipeline.py`
3. **Start the server and smoke-test the browser UI + endpoints:**
   `.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend`
   then open <http://127.0.0.1:8000/> and try the queries in `DEMO_SCRIPT.md`.
4. **(Optional) Run the T6 DEMO evaluation** against the running server:
   `.venv\Scripts\python.exe scripts/run_eval.py`
5. **Review `git diff`** for accidental changes, then decide on deployment / push
   (neither has been done — do not deploy or push without explicit approval).

The historical phase plan is retained below for context only.

### Historical phase plan (context only — these are done)

- Phase 5: Dense + Sparse hybrid retrieval — **implemented** (`modules/retrieval`, `modules/sparse_retrieval`).
- Phase 6: Vector store — in-memory stores are used by default; Qdrant is an optional, non-wired path.
- Phase 7: Query-aware adaptive retrieval — **implemented** (`modules/query_analyzer`).
- Phase 8: FlashRank reranking — **implemented with honest fallback** (`modules/reranking`).
- Phase 9+: Evidence, refusal, generation, STT, Judge Mode, frontend — **implemented**.

## Current Constraints

- ❌ Do NOT download full 3.7GB Hindi dataset
- ❌ Do NOT switch to IndicMSMARCO
- ❌ Do NOT create synthetic data
- ❌ Do NOT break or weaken existing tests (inventory: **146 across 13 files** — see `PROJECT_STATE.md`)
- ✅ DO test thoroughly before committing
- ✅ DO integrate each phase with existing modules
- ✅ DO run full test suite after changes

## File Structure After Phase 4

```
backend/
├── modules/
│   ├── chunking.py                 # ✅ Multi-strategy chunker (fixed, sentence, parent-child)
│   ├── data_pipeline.py            # ✅ MSMARCO-XI streaming + normalization
│   ├── embedding_pipeline.py       # ✅ NEW: Orchestration for embedding workflow
│   ├── embeddings/
│   │   └── __init__.py            # ✅ BGE3EmbeddingProvider + InMemory provider
│   ├── retrieval/
│   │   └── __init__.py            # ✅ Retriever + VectorStore interfaces
│   ├── sample_ingestion.py        # ✅ MSMARCO-XI validation + provenance
│   └── ...other modules...
├── requirements.txt
├── requirements-dev.txt            # ✅ UPDATED: Added sentence-transformers + datasets
└── app/...

tests/
├── test_sample_ingestion_contract.py      # ✅ 4/4 PASSED
├── test_chunking_experiment.py            # ✅ 3/3 PASSED (non-streaming)
├── test_embedding_pipeline.py             # ✅ 13/13 PASSED (NEW)
└── ...other tests...
```

## How to Continue

1. Read `PROJECT_STATE.md` for complete status
2. Read `ARCHITECTURE.md` for system design
3. Look at the roadmap items 5-13 in the current request
4. Start with Phase 5: implement BM25 sparse indexing
5. Follow the same testing pattern as Phase 4
6. Run `python -m pytest tests/ -v` to verify progress
7. Update `PROJECT_STATE.md` and `AI_HANDOFF.md` before stopping

## Key Implementation Files to Study

- `backend/modules/embedding_pipeline.py` - NEW, study for Phase 5 pattern
- `backend/modules/chunking.py` - Multi-strategy pattern to follow
- `backend/modules/retrieval/__init__.py` - Vector store interface to extend

The project must use the authoritative competition dataset:

- `ai4bharat/MSMARCO-XI`
- Hindi config target: `"hi"`
- Streaming access retained: `True`
- Sample cap: 100 records
- Full dataset download: not allowed

Important: this is not interchangeable with `ai4bharat/IndicMSMARCO`.

Runtime verification in this environment showed that the `"hi"` builder config is not exposed here, so the loader keeps the intended `"hi"` config and falls back to the default config with a `[VERIFY]` marker rather than guessing.

Verified runtime fields for the real dataset are:

- `query`
- `Answer`
- `query_id`
- `query_type`
- `passages`
- `source_lang`
- `target_lang`
- `Eng_Query`
- `Eng_Answer`

The extraction logic preserves both translated and English content where useful for multilingual retrieval design.

## What has been done (Phase 1-4 — foundation + embeddings)

### Phase 1 — Repository scaffolding ✅
- Clean repository structure with separated frontend/backend
- FastAPI backend with application factory pattern
- Environment configuration via pydantic-settings
- Structured logging and exception handling
- `/health` endpoint for smoke tests
- Modular pipeline interfaces
- Comprehensive test infrastructure

### Phase 2 — Architecture Lock ✅
- Documented architecture flow (voice→STT→query analysis→retrieval→grounding→generation)
- Technology decisions: Sarvam STT, BGE-M3 embeddings, Qdrant, FlashRank, SSE streaming
- Identified verification items for production implementation
- Architecture considered locked for Phase 3+ implementation

### Phase 3B — Sample Ingestion + Chunking ✅
- Real dataset validation: `ai4bharat/MSMARCO-XI` with strict provenance tracking
- Data pipeline: streaming Hindi records from official dataset without full download
- Normalization: preserves English and Hindi content for multilingual retrieval
- Chunking strategies implemented:
  - Fixed-size with configurable overlap
  - Sentence-boundary aware chunking
  - Parent-child chunking with provenance links
  - Metadata-aware chunk representation

### Phase 4 — BGE-M3 Embeddings ✅ [JUST COMPLETED]
- **New file:** `backend/modules/embedding_pipeline.py`
  - Orchestration layer for chunk→embedding workflow
  - Batch embedding with async/await support
  - Integration with vector stores
  - Chunk format normalization
- **BGE3EmbeddingProvider** (already defined in embeddings module)
  - Async embedding using sentence-transformers
  - Normalized embeddings (L2 normalization)
  - Configurable device support (CPU/GPU)
- **Test coverage:** 13 new tests
  - Chunk embedding with InMemoryEmbeddingProvider
  - Batch processing and empty input handling
  - Metadata preservation through pipeline
  - Multilingual chunk embedding
  - Integration with vector store
  - Chunker→Embedder workflow verification
- **All 20 tests passing** (4 ingestion + 3 chunking + 13 embedding)
- **Config:** `pydantic-settings` loading from `.env` (see `.env.example`)
- **Logging:** Structured stdout logging via `app/logging_config.py`
- **Errors:** `ThinkZenError` base class + global handlers in `app/exceptions.py`
- **Health endpoint:** `GET /health` returns `{status, service, version, timestamp}`
- **App factory:** `create_app()` in `app/main.py` — use this in tests

### Pipeline modules (interfaces only — no implementations)

| Module | Interface class | File |
|--------|----------------|------|
| STT | `STTProvider` | `backend/modules/stt/__init__.py` |
| Chunking | `Chunker`, `Chunk` | `backend/modules/chunking/__init__.py` |
| Embeddings | `EmbeddingsProvider` | `backend/modules/embeddings/__init__.py` |
| Retrieval | `Retriever`, `RetrievedDocument` | `backend/modules/retrieval/__init__.py` |
| Reranking | `Reranker` | `backend/modules/reranking/__init__.py` |
| Generation | `Generator`, `GeneratedAnswer` | `backend/modules/generation/__init__.py` |
| Evaluation | `Evaluator`, `EvaluationResult` | `backend/modules/evaluation/__init__.py` |

### Deliberate boundaries that remain (by design)

- No full dataset download (the 10M+ corpus). The runtime corpus is the 20-doc
  **DEMO/SAMPLE** set in `data/samples/demo_docs.json`, clearly labeled and never
  presented as the official MSMARCO-XI benchmark.
- No paid API calls in the default run. Generation falls back to a deterministic
  evidence-quoting synthesizer; server STT returns an honest "not configured" response.
- No hard-coded API keys (all optional keys read from `.env`).
- No verified production latency benchmark yet — Judge Mode reports **measured** latency
  for the runs you actually make; it is not a guaranteed target.
- Qdrant vector DB is available as an optional path but is **not wired** by default
  (in-memory stores are used).
- Real-data (MSMARCO-XI) evaluation is **PENDING** dataset availability; the T6 framework
  is ready and labels demo runs as `UNIT_TEST_DATA`.

**Now implemented (previously listed here as missing):** real hybrid retrieval, adaptive
retrieval, reranking (with fallback), evidence intelligence, grounded generation + strict
refusal, the frontend UI, and Judge Mode telemetry. See `PROJECT_STATE.md`.

## What you must do next

The pipeline is built. Do **not** rebuild or re-architect it. The remaining work is
verification and (only when approved) production hardening / deployment:

1. Run the suite, the nine-case verifier, and a live server smoke test (commands in
   "Next Immediate Steps" above and in `PROJECT_STATE.md`).
2. Review `git diff` for accidental or unnecessary changes before any commit.
3. If/when a verified MSMARCO-XI 100-record sample is available, feed it through
   `modules/sample_ingestion.validate_external_sample` and run the T6 framework with
   `DataLabel.REAL_DATA` to produce production-representative metrics.
4. Only with explicit approval: choose a generation provider (set `GEMINI_API_KEY` or
   `OPENAI_API_KEY`), optionally enable Sarvam STT (`SARVAM_API_KEY`), then deploy / push.

Keep every existing test contract intact; measure, never fabricate, latency and metrics.

## Rules for all AI agents

1. **Never hard-code API keys** — use `.env` only
2. **Never download the full dataset** without explicit user approval
3. **Never call paid APIs** without explicit user approval
4. **Keep dependencies minimal** — justify every new package
5. **Use existing interfaces** in `backend/modules/` — extend, don't replace
6. **Update PROJECT_STATE.md and AI_HANDOFF.md** at the end of every session
7. **Do not invent benchmark numbers** — measure real metrics only
8. **Do not build a generic RAG demo** — differentiation is required
9. **Do not treat any exact model, endpoint, threshold, or latency target as verified without official/current documentation** [VERIFY BEFORE IMPLEMENTATION]

---

## Local setup notes

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r backend/requirements-dev.txt
copy .env.example .env

# Run backend
cd backend
uvicorn app.main:app --reload --port 8000

# Test
cd ..
pytest
```

---

## Test status (honest)

**Current inventory:** 146 test functions across 13 files (static inventory — counted by
inspection and traced against the implementation; see `PROJECT_STATE.md` for the table).

**Execution:** the suite was **not executed in the finalization session** because the
Linux workspace is network-isolated (its package proxy returns `403`, so `pytest`/`fastapi`
cannot be installed) and the project venv is a Windows `.venv` that cannot be driven from
that sandbox. No pass/fail counts are
asserted as if they had been run. **Run locally to produce real results.**

**How to run (from the repo root `C:\ThinkZenRag_AI`, Windows):**

```
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/verify_pipeline.py
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend
```

`pytest.ini` sets `testpaths=tests`, `pythonpath=backend`, `-p no:cacheprovider`, and
`asyncio_mode=strict`; run from the repo root so both the root and `backend` import roots
resolve. Expect one `datasets`-gated test to skip when the optional package is absent.

## Architecture summary for the next agent

The repo is now in a design-lock state. The next agent must treat the architecture as the contract and avoid turning this into a general-purpose chatbot demo. The system should remain grounded, multilingual, evidence-first, voice-native, and measurable.
