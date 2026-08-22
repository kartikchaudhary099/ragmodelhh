# ThinkZen — Deterministic Demo Script

A six-step walkthrough for a live demo or judge evaluation. Every query below is
grounded in the actual seeded demo corpus (`data/samples/demo_docs.json`, 20
documents in English, Hindi, and Hinglish), so the outcomes are reproducible.

> **Honesty notes for judges**
> - The 20-document corpus is clearly **DEMO / SAMPLE** data seeded at startup. It is
>   **not** the official benchmark dataset (e.g. MSMARCO-XI) and is never presented as one.
> - All latency numbers in Judge Mode are **measured live** for your requests. No
>   latency figures are pre-baked or hard-coded.
> - With no API keys set, generation uses the deterministic evidence-quoting
>   synthesizer (it quotes the retrieved passage) and STT reports "not configured"
>   rather than faking a transcript. Both behaviors are intentional and honest.

## Setup

From the repository root (`C:\ThinkZenRag_AI`):

```
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload
```

Open <http://127.0.0.1:8000/>. On startup the demo corpus is chunked, embedded, and
indexed into both the sparse (BM25) and dense stores — the API is queryable immediately.
Leave the **Adaptive retrieval** toggle ON (default) so the Query Analyzer picks alpha.

Optional health check in a second terminal:

```
curl http://127.0.0.1:8000/health
```

Expected: `{"status":"ok","service":"ThinkZen",...}`.

---

## Step 1 — English, grounded

**Type / speak:** `What is Retrieval-Augmented Generation?`

- Detected language: **English (en)**; query type: **factual** (adaptive alpha ≈ 0.70).
- Retrieves the "Retrieval-Augmented Generation — Core Concepts" passage.
- **Expected outcome:** GROUNDED. Answer in **English**, quoting the retrieved passage,
  with the source cited in the evidence cards.
- Point at: the green **Grounded** badge, the evidence card (source title + score), and
  the Judge panel filling in per-stage latencies.

## Step 2 — Hindi, grounded

**Type / speak:** `वाणीरैग क्या है?`

- Detected language: **Hindi (hi)** (Devanagari); answered in **Devanagari Hindi**.
- Retrieves the "वाणीरैग बहुभाषी वॉइस सेवा निर्देश" passage.
- **Expected outcome:** GROUNDED. Answer in **Hindi** (`प्राप्त साक्ष्यों … के अनुसार: …`).
- Point at: the answer is in Hindi — not English — proving language-aware generation.

## Step 3 — Hinglish, grounded (stays Hinglish)

**Type / speak:** `ThinkZen mein hybrid search kaise kaam karta hai?`

- Detected language: **Hinglish (hi-en)** (Romanized-Hindi function words: *mein, kaise, hai*).
- Retrieves the "Voice RAG Ingestion and Hybrid Fusion Policy" (Hinglish) passage.
- **Expected outcome:** GROUNDED. Answer in **natural Hinglish**
  (`Retrieved evidence (…) ke anusaar: …`) — **not** forced into Devanagari-only Hindi.
- Point at: this is the key multilingual differentiator — Hinglish in, Hinglish out.

## Step 4 — Out-of-domain, refused (no hallucination)

**Type / speak:** `What is the capital of France?`

- Neither "capital" nor "France" appears anywhere in the corpus.
- **Expected outcome:** REFUSED. The content-coverage grounding gate fires (zero
  content-term overlap with retrieved evidence), so the system **refuses** with a clear,
  language-appropriate message instead of fabricating an answer.
- Point at: the red **Refused** badge and the honest refusal text — the system would
  rather say "insufficient evidence" than hallucinate.

## Step 5 — Voice input

1. Set the voice-language selector (top-right of the query box) to **English**.
2. Click the microphone button and allow microphone access when prompted.
3. Speak: *"What is Retrieval-Augmented Generation?"*
4. The transcript fills the box and the query runs automatically.

- Primary path is the browser-native **Web Speech API** (Chrome/Edge; no server key).
- **If the browser doesn't support speech recognition or the mic is denied:** the button
  is disabled/errors gracefully — just **type** the query (Steps 1–4 are the text fallback).
  The optional server-side `POST /api/v1/stt` endpoint returns an honest "not configured"
  message unless `SARVAM_API_KEY` is set — it never returns a fake transcript.

## Step 6 — Judge Mode (real telemetry)

After running Steps 1–4 (so several requests are recorded), inspect telemetry:

- **In the UI:** the Judge panel shows, for the latest query, the detected language,
  query type, retrieval strategy, **alpha value and its source** (adaptive vs. override),
  candidate/evidence counts, per-stage latencies, and the grounding decision.
- **Aggregate percentiles:** open <http://127.0.0.1:8000/api/v1/judge> (or `curl` it) to
  see run counts and **P50 / P70 / P90 / P100** latencies computed from your actual runs.

Every value shown is measured or derived from the real request path — nothing is faked.

---

## Optional — showcase adaptive retrieval

To highlight query-aware adaptive alpha, run a **comparison** query:

**Type:** `What is the difference between dense and sparse retrieval?`

- Query type: **comparison** → adaptive alpha ≈ **0.35** (sparse-leaning), vs. ≈ 0.70 for
  the factual Step-1 query. Retrieves "Dense vs Sparse Retrieval: A Comparison". GROUNDED.
- Point at: `alpha_source: adaptive` and the different alpha value in the Judge panel —
  the retrieval weighting genuinely changes with query type.

## One-line recap for the panel

English in → English answer. Hindi in → Hindi answer. Hinglish in → Hinglish answer.
Out-of-domain → honest refusal. Voice in → same grounded pipeline. Judge Mode → real,
measured latency and a full decision trace. No fabricated answers, metrics, or transcripts.
