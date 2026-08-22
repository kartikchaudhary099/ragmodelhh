# Project State — ThinkZen

> Live progress tracker. Updated at the end of each work session.

## Current phase

**Production finalization — end-to-end Voice RAG, grounded answering, and Judge Mode telemetry.**

The full pipeline is implemented and wired end to end: FastAPI backend with an application-factory entry point, a query-analysis → hybrid-retrieval → evidence-intelligence → grounded-generation flow, a single `POST /api/v1/query` contract, an optional server-side STT endpoint, aggregate Judge Mode latency stats, and a single-page dark-mode UI served from `/`.

## Completed work

- [x] Repository scaffolded with a clean backend/frontend/data/tests structure.
- [x] Backend: FastAPI application-factory (`create_app`), CORS, structured logging, global exception handlers, and a `/health` smoke endpoint.
- [x] Backend: environment-driven configuration via `pydantic-settings` (`.env` support, no hard-coded secrets).
- [x] Query Analyzer: deterministic language detection (Hindi / Hinglish / English), query-type and complexity classification, keyword extraction, and adaptive retrieval-strategy selection (alpha + top_k).
- [x] Hybrid retrieval: BM25 sparse retrieval fused with dense embeddings via alpha-weighted min-max normalization.
- [x] Dependency-free dense retrieval: a deterministic feature-hashing embedding provider shared between the sample seeder and the retriever, so query and document vectors live in the same space without `torch`/`sentence-transformers`.
- [x] Evidence Intelligence: coherence scoring, near-duplicate removal, source-diversity boost, and an explicit `GROUNDED` / `INSUFFICIENT` / `ABSTAIN` decision with a human-readable reason.
- [x] Grounded generation: confidence-score threshold **plus** a lexical content-coverage gate, language-aware answers and refusals, optional Gemini/OpenAI with a deterministic evidence-quoting fallback.
- [x] Unified `POST /api/v1/query` returning answer, refusal state/reason, cited sources, and full telemetry (per-stage latency, candidate/evidence counts, alpha value + source, query analysis, evidence decision).
- [x] Judge Mode telemetry aggregation with P50/P70/P90/P100 latency percentiles under `/api/v1/judge`.
- [x] T6 Evaluation Framework (`backend/modules/evaluation/runner.py`) with a runnable entry point (`scripts/run_eval.py`): measures grounding/abstention/success rates, per-stage latency, mean max retrieval score, and optional recall@k against the live API, with strict `DataLabel` provenance (`REAL_DATA` / `UNIT_TEST_DATA` / `PENDING`) so results are never mislabeled as production/MSMARCO-XI. No metric is fabricated.
- [x] In-process end-to-end verification harness (`scripts/verify_pipeline.py`) that drives the real pipeline via FastAPI `TestClient` (no live server needed) across the nine required cases: English/Hindi/Hinglish grounded, English/Hindi/Hinglish refusal, evidence/citations schema, Judge Mode percentiles, and honest server-side STT.
- [x] Sample data seeder: bilingual Hindi + English demo corpus (`data/samples/demo_docs.json`) chunked, embedded, and indexed into both stores at startup.
- [x] Single-page UI (`frontend/static/index.html`, `style.css`, `app.js`): voice input via Web Speech API with per-language selection, adaptive-alpha toggle + hybrid-weight slider, inline (non-blocking) error and loading states, evidence cards, and a live Judge Mode panel; responsive/mobile refinements and accessibility helpers.
- [x] Honest server-side STT endpoint: performs a real Sarvam AI call when `SARVAM_API_KEY` is set, and returns an explicit "not configured" response (never a fabricated transcript) otherwise.

## Test inventory

**How this was counted:** the numbers below are a **static inventory** produced by inspecting the test files and tracing each test against the current implementation. `pytest` was **not executed here**: the Linux workspace available to the assistant is network-isolated (its package proxy returns `403`, so `pytest`/`fastapi`/`httpx` cannot be installed), and the project's virtual environment is a Windows `.venv` that cannot be driven from that sandbox. **Run the suite locally** to confirm (command below). No pass/fail counts are asserted here as if they had been run.

Total: **146 test functions across 13 files.**

| Test file | Test functions | Notes |
|---|---:|---|
| `test_query_analyzer.py` | 36 | Language/type/complexity/keywords/strategy + edge cases; incl. a regression test for short Roman-Hindi queries ending in punctuation (`"… hai?"`). |
| `test_sparse_retrieval.py` | 26 | BM25 indexer, BM25 store, orchestrated hybrid retriever, integration. |
| `test_evidence_intelligence.py` | 19 | Coherence, dedup, diversity, grounding decision, serialization. |
| `test_evaluation_runner.py` | 15 | T6 aggregation, provenance-note honesty, and per-query extraction/recall (offline; uses an in-memory fake client, no server). |
| `test_embedding_pipeline.py` | 13 | Chunk embedding and store-format contracts. |
| `test_query_api.py` | 11 | End-to-end `/api/v1/query` + Judge Mode contract. |
| `test_chunking_experiment.py` | 7 | Chunkers + normalization. 1 test skips without the optional `datasets` package. |
| `test_grounded_generator.py` | 5 | Refusal, Hindi detection, grounded success. |
| `test_embedding_retrieval_experiment.py` | 5 | Embedding-provider contract + experimental retrieval. |
| `test_sample_ingestion_contract.py` | 4 | External-sample provenance validation. |
| `test_health.py` | 2 | `/health` + app factory. |
| `test_config.py` | 2 | Settings defaults + caching. |
| `test_modules.py` | 1 | Module import smoke test. |
| **Total** | **146** | 1 test (`datasets`-gated) skips when the optional package is absent. |

**Run locally (from the repository root `C:\ThinkZenRag_AI`):**

```
.venv\Scripts\python.exe -m pytest -q
```

`pytest.ini` sets `testpaths=tests` and `pythonpath=backend`; run from the repository root so both the root package and `backend` are importable.

## Verification status this session

- **Real pipeline executed offline (dependency-free path).** A stdlib-only harness that mirrors `POST /api/v1/query` stage-for-stage (query analysis → hybrid retrieval → reranker fallback → evidence intelligence → grounded generation → telemetry) was run against the **actual project modules** (not mocks) across the nine required cases (English / Hindi / Hinglish, grounded + refusal), the three brief language examples, the evidence/citation schema, and the `LatencyAggregator` percentiles. All checks passed. The T6 evaluation logic in `backend/modules/evaluation/runner.py` was likewise exercised offline against its real code; all checks passed. This runs the same dependency-free path the app serves when no `torch`/LLM keys are configured.
- **Bug found and fixed by that execution.** The brief's own Hinglish example `"ThinkZen kaise kaam karti hai?"` was misdetected as English: trailing punctuation (`hai?`) survived the naive `.split()` in `_detect_language`, hiding the `hai` function word and dropping the query below the 2-hit Hinglish threshold. Detection now tokenizes on word boundaries (`re.findall(r"[\wऀ-ॿ]+", …)`), matching the existing keyword extractor. A regression test was added (`test_language_detection_hinglish_short_with_trailing_punctuation`). No existing test was changed or weakened.
- **Code-quality audit.** Frontend↔API response schemas are consistent (every field the UI reads exists in the response models; every field it sends is accepted and within validation bounds). No secrets, unsafe calls (`eval`/`exec`/`os.system`/`shell=True`/`pickle`), hard-coded latency, or stray debug prints exist in the backend. The only dead code is `backend/modules/chunking.py`, which is shadowed by the `backend/modules/chunking/` package (unreachable on every platform; safe to remove but left in place).
- **Still Windows-authoritative — NOT run here.** `pytest` and `uvicorn` were **not** executed in this session: the Linux sandbox is network-isolated (its package proxy returns `403`, so `pytest`/`fastapi`/`httpx`/`uvicorn` cannot be installed — re-confirmed this pass) and the project venv is a Windows `.venv` that cannot be driven from it. The full suite (`tests/`), `scripts/verify_pipeline.py` (FastAPI `TestClient`), and a live `uvicorn` server must be run locally to produce authoritative runtime results.
- **Real latency measured (sandbox, dependency-free fallback path — NOT the competition metric).** A benchmark ran the *actual* pipeline modules across 8 representative queries with 6 warm repeats (48 measurements). Warm end-to-end total: **P50 ≈ 2.2 ms, P70 ≈ 2.2 ms, P100 ≈ 2.8 ms** (cold P100 ≈ 2.5 ms); per-stage means — analysis ~0.02 ms, retrieval ~2.0 ms, rerank ~0.07 ms, evidence ~0.10 ms, generation ~0.04 ms. These **exclude STT** and use feature-hashing embeddings + the deterministic generator on sandbox CPU, so they are a lower-bound smoke signal only. Authoritative P50/P70/P100 must come from the Windows configured-pipeline run and `/api/v1/judge`; no latency target is hard-coded anywhere in the code.
- **Production-hardening audit (read-only) — clean except one documented item.** No `.env` is committed or present (only `.env.example`: placeholders, all commented, correct variable names); `.gitignore` excludes `.env*`, `.venv/`, `__pycache__`, `.pytest_cache/`, `none/`, `eval_results/`, `data/raw|processed`, logs, and model/index caches. No hard-coded secrets. No `pdb`/`breakpoint`, no `console.log`; the only `print(` calls sit inside `if __name__ == "__main__"` demo blocks, not the request path. The frontend calls the API **same-origin** (relative `/api/v1/...`), so there is no hard-coded host in the serving path; the three `localhost` literals are configurable defaults in optional/dev tooling (`PipelineEvaluator` base_url; the unwired Qdrant URL). **One item to harden at deploy time:** `backend/app/main.py` sets `CORSMiddleware(allow_origins=["*"], allow_credentials=True, …)` — harmless for the same-origin demo, but it should be restricted for any cross-origin production deployment (exact patch under *Known items* below). Not changed this session: it is not a functional/demo blocker, the correct origin depends on the not-yet-chosen deploy URL, and the change cannot be exercised in this network-isolated sandbox.

## Deployment & compliance pass (2026-08-21)

- **Chunking strategies proven on real demo data.** All three implemented chunkers
  (`FixedSizeChunker`, `SentenceChunker`, `ParentChildChunker`) were run over the 20-doc
  demo corpus and verified by an executable harness (22/22 checks): fixed-size → 44 chunks
  (mean 156 chars), sentence-boundary → 48 chunks (mean 133), parent-child → 125 chunks
  (mean 113). Confirmed: metadata preservation (strategy/chunk_role/source_id/language)
  across en/hi/hi-en, real overlap behaviour, exactly one parent per document with every
  child linked via `parent_id`, parent preserving full source text, determinism across two
  runs, and empty/malformed input returning `[]` without crashing. No "best" strategy is
  claimed — only measured facts. (No product code changed; this is verification only.)
- **Sarvam STT code path re-verified against the known official contract.** `POST /api/v1/stt`
  calls `https://api.sarvam.ai/speech-to-text` with header `api-subscription-key`, model from
  `SARVAM_STT_MODEL` (default `saarika:v2`), multipart form (`file` + `data{model, language_code}`),
  and a 30s timeout, with robust error handling and no fabricated transcript. **Live
  verification against a real key and online re-verification of the API docs are BLOCKED** in
  this environment (no `SARVAM_API_KEY`, and outbound network/egress is restricted). STT is
  therefore code-ready / **live-run PENDING** — not reported as verified.
- **Deployment config created (not built/deployed here).** Added a root `Dockerfile`
  (`python:3.12-slim`, installs `backend/requirements.txt`, copies `backend/ frontend/ data/`,
  `CMD uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}`) and a
  `.dockerignore` (excludes `.venv`/`.git`/caches/tests/`data/raw|processed`/`.env`; keeps
  `frontend/` and `data/samples/`). Verified the startup import chain is safe for a clean
  image: **no top-level heavy imports and no `numpy` on the request path** — all of
  `datasets`/`sentence_transformers`/`google.genai`/`flashrank`/`qdrant_client` are imported
  lazily inside guarded `try/except`. **Added `httpx>=0.27.0,<2.0.0` to
  `backend/requirements.txt`** — it was previously dev-only, but the deployed `/api/v1/stt`
  needs it to call Sarvam (it is still never imported on the core RAG path). The image was
  **not** built or deployed (no Docker/network in this sandbox); build/deploy and public-URL
  verification remain user-side.
- **MSMARCO-XI runtime evaluation remains BLOCKED.** No official `ai4bharat/MSMARCO-XI` shard
  is present in `data/`, and it cannot be downloaded here (no network). The provenance-enforcing
  ingestion (`sample_ingestion.py`, `EXPECTED_DATASET="ai4bharat/MSMARCO-XI"`) and the T6
  framework are ready; demo runs are labeled `UNIT_TEST_DATA` and never presented as MSMARCO-XI.
- **Git initialized in the existing repository** (see push commands under *Known items*). Not
  pushed — no credentials available in this environment.

## Known items / follow-ups

- Runtime execution still needs to be run locally to produce measured results and latency figures:
  - Full suite: `.venv\Scripts\python.exe -m pytest -q`
  - In-process nine-case check: `.venv\Scripts\python.exe scripts/verify_pipeline.py`
  - Live server: `.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend`
  - T6 DEMO evaluation (server must be running): `.venv\Scripts\python.exe scripts/run_eval.py`
- A stray empty `none/` directory at the repository root can be removed manually (`Remove-Item -Recurse -Force none` on Windows); it is not referenced by the application and is now git-ignored so it can never be committed. It will not regenerate (`pytest.ini` uses `-p no:cacheprovider`).
- `.env.example` was aligned with the variable names the code actually reads (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `SARVAM_API_KEY`, `SARVAM_STT_MODEL`); all remain commented/empty so the default run stays fully dependency-free with no secrets.
- Deterministic six-step demo walkthrough is in `DEMO_SCRIPT.md`.
- **CORS hardening for cross-origin production (optional; apply on Windows where it can be tested).** In `backend/app/main.py`, replace the wildcard block with an env-driven allowlist:

  ```python
  import os
  _origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()]
  app.add_middleware(
      CORSMiddleware,
      allow_origins=_origins,
      allow_credentials=(_origins != ["*"]),   # never pair "*" with credentials
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

  Then set `CORS_ALLOW_ORIGINS=https://your-deployed-domain` in the production `.env`. Leaving it unset preserves today's permissive behaviour, so the demo is unaffected. The app uses no cookies/auth, so credentials can safely stay off under a wildcard.
- **Sarvam server-side STT is a HARD competition requirement and is the FINAL SUBMISSION BLOCKER.** The code path (`backend/app/api/routes/stt.py`) is correct and honest — a real Sarvam call (`https://api.sarvam.ai/speech-to-text`, header `api-subscription-key`, model from `SARVAM_STT_MODEL`, default `saarika:v2`) when `SARVAM_API_KEY` is set, and an explicit `success=False` "not configured" response (never a fabricated transcript) otherwise. It has **not** been exercised against a live credential (none available in this environment). To clear the blocker: set `SARVAM_API_KEY` in the Windows `.env`, restart `uvicorn`, and POST real audio to `/api/v1/stt` to confirm a real transcript. Until that is done on Windows, STT status is code-ready / **live-run PENDING** — do not report it as verified.
