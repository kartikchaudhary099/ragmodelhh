#!/usr/bin/env python3
"""REAL latency benchmark for ThinkZen — scoped, never conflated.

Measures three DISTINCT latency scopes and reports P50 / P70 / P100 / mean / count for each.
The scopes are kept strictly separate and explicitly labeled so RAG-only latency is NEVER
presented as full voice latency:

  1. RAG_ONLY        — POST /api/v1/query only (text/transcript → grounded answer).
                        STT is EXCLUDED. Always runnable (in-process or live).
  2. SARVAM_STT      — POST /api/v1/stt only (audio → transcript) via the REAL Sarvam API.
                        Requires SARVAM_API_KEY + --audio-dir. SKIPPED honestly otherwise.
  3. FULL_VOICE_E2E  — STT + query end to end per audio file (audio → transcript → answer).
                        Requires SARVAM_API_KEY + --audio-dir. SKIPPED honestly otherwise.

Percentiles are computed with the SAME function Judge Mode uses
(modules.telemetry._compute_percentile), so the numbers are consistent with the app's own
telemetry. Two RAG-only views are reported: client-observed wall-clock latency (what a caller
experiences) and the server-reported internal pipeline latency (telemetry.total_latency_ms).

No number is fabricated. One-time corpus seeding (cold start) is measured and reported
SEPARATELY from steady-state per-query latency. If the Sarvam scopes cannot run, they are
reported as SKIPPED — this harness will not claim a voice-latency figure it did not measure,
and it makes no "<200ms" claim.

RUN (repo root, project venv):
    # RAG-only over the DEMO corpus (default):
    .venv\\Scripts\\python.exe scripts\\benchmark_latency.py --n-queries 36

    # RAG-only over the OFFICIAL ai4bharat/MSMARCO-XI corpus:
    #   .venv\\Scripts\\python.exe scripts\\build_msmarco_xi_sample.py --limit 100
    #   set THINKZEN_CORPUS=official
    .venv\\Scripts\\python.exe scripts\\benchmark_latency.py --n-queries 36

    # include the real Sarvam STT + full-voice scopes (needs key + audio files):
    set SARVAM_API_KEY=...
    .venv\\Scripts\\python.exe scripts\\benchmark_latency.py --audio-dir path\\to\\wavs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from modules.telemetry import _compute_percentile  # noqa: E402  (same method as Judge Mode)

MIN_QUERIES = 30
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}

_HINGLISH_STOP = {
    "what", "was", "were", "is", "are", "the", "a", "an", "of", "to", "for", "in",
    "on", "at", "how", "do", "does", "did", "who", "when", "where", "which", "why",
    "and", "or", "with", "by", "from", "impact",
}

# Representative DEMO-corpus queries (grounded + refusals across EN/HI/Hinglish).
_DEMO_QUERIES = [
    "What is Retrieval-Augmented Generation?",
    "How does hybrid search work in ThinkZen?",
    "What is dense retrieval?",
    "Explain BM25 sparse retrieval.",
    "वाणीरैग क्या है?",
    "हाइब्रिड सर्च कैसे काम करता है?",
    "ThinkZen mein hybrid search kaise kaam karta hai?",
    "reranking kya hota hai?",
    "What is the capital of France?",          # out-of-domain refusal (EN)
    "फ्रांस की राजधानी क्या है?",                    # out-of-domain refusal (HI)
    "France ki capital kya hai?",              # out-of-domain refusal (Hinglish)
    "What is grounded generation?",
]


def _build_hinglish(eng_query: str) -> str:
    toks = [t for t in re.findall(r"[a-zA-Z]+", eng_query.lower())
            if t not in _HINGLISH_STOP and len(t) >= 4]
    salient = " ".join(toks[:3]) if toks else "manhattan project"
    return f"{salient} ke bare mein bataye"


def _active_corpus():
    from app.config import get_settings

    return get_settings()


def build_queries(n: int) -> tuple[list[str], str]:
    """Build >= n representative queries for the ACTIVE corpus.

    Official mode uses REAL dataset queries (English + Hindi per record, plus Hinglish-framed
    and out-of-domain refusals) so the benchmark reflects the real corpus; demo mode cycles a
    representative set. Returns (queries, corpus_mode).
    """
    settings = _active_corpus()
    corpus_mode = (settings.corpus_mode or "demo").strip().lower()

    if corpus_mode == "official":
        from modules.official_corpus import load_official_records

        records, _ = load_official_records(
            settings.official_sample_path, settings.official_provenance_path
        )
        queries: list[str] = []
        for r in records:
            en = (r.get("Eng_Query") or "").lstrip(") ").strip()
            hi = (r.get("query") or "").strip()
            if en:
                queries.append(en)
            if hi:
                queries.append(hi)
            if en:
                queries.append(_build_hinglish(en))
        # A few out-of-domain refusals (tokens verified absent from the MSMARCO-XI sample).
        queries += [
            "What is the recipe for chocolate cake?",
            "चॉकलेट केक कैसे बनाते हैं?",
            "chocolate cake kaise banate hain?",
        ]
    else:
        queries = list(_DEMO_QUERIES)

    if len(queries) < n:
        # Cycle to reach the requested count (steady-state latency is unaffected; noted in output).
        base = list(queries)
        while len(queries) < n:
            queries.extend(base)
    return queries[:n], corpus_mode


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p50_ms": None, "p70_ms": None, "p100_ms": None, "mean_ms": None}
    return {
        "count": len(values),
        "p50_ms": round(_compute_percentile(values, 50), 2),
        "p70_ms": round(_compute_percentile(values, 70), 2),
        "p100_ms": round(_compute_percentile(values, 100), 2),  # == max
        "mean_ms": round(sum(values) / len(values), 2),
        "min_ms": round(min(values), 2),
    }


def _print_scope(title: str, scope_note: str, st: dict) -> None:
    print(f"\n{title}")
    print(f"  scope: {scope_note}")
    if not st.get("count"):
        return
    print(f"  count={st['count']}  P50={st['p50_ms']}ms  P70={st['p70_ms']}ms  "
          f"P100(max)={st['p100_ms']}ms  mean={st['mean_ms']}ms  min={st.get('min_ms')}ms")


# --------------------------------------------------------------------------- clients


class InProcessClient:
    def __init__(self):
        from fastapi.testclient import TestClient

        from app.main import create_app

        self._client = TestClient(create_app())

    def query(self, q: str) -> dict:
        resp = self._client.post("/api/v1/query", json={"query": q})
        resp.raise_for_status()
        return resp.json()

    def stt(self, path: Path, language: str) -> dict:
        import mimetypes

        with open(path, "rb") as fh:
            audio = fh.read()
        ctype = mimetypes.guess_type(str(path))[0] or "audio/wav"
        resp = self._client.post(
            "/api/v1/stt",
            files={"file": (path.name, audio, ctype)},
            data={"language": language},
        )
        resp.raise_for_status()
        return resp.json()


class LiveClient:
    def __init__(self, base_url: str):
        import httpx

        self._httpx = httpx
        self._base = base_url.rstrip("/")

    def query(self, q: str) -> dict:
        resp = self._httpx.post(f"{self._base}/api/v1/query", json={"query": q}, timeout=60.0)
        resp.raise_for_status()
        return resp.json()

    def stt(self, path: Path, language: str) -> dict:
        import mimetypes

        with open(path, "rb") as fh:
            audio = fh.read()
        ctype = mimetypes.guess_type(str(path))[0] or "audio/wav"
        resp = self._httpx.post(
            f"{self._base}/api/v1/stt",
            files={"file": (path.name, audio, ctype)},
            data={"language": language},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description="Scoped real latency benchmark for ThinkZen.")
    parser.add_argument("--n-queries", type=int, default=36,
                        help=f"Number of RAG-only queries to measure (min {MIN_QUERIES}).")
    parser.add_argument("--base-url", default=None, help="Benchmark a live server instead of in-process.")
    parser.add_argument("--audio-dir", default=None,
                        help="Directory of audio files to enable the SARVAM_STT + FULL_VOICE_E2E scopes.")
    parser.add_argument("--language", default="auto", help="Language hint for STT scopes.")
    parser.add_argument("--out", default=None, help="Path to write the JSON result (default eval_results/).")
    args = parser.parse_args()

    n = max(MIN_QUERIES, args.n_queries)

    print("=" * 74)
    print("ThinkZen — REAL latency benchmark (scoped; STT excluded from RAG-only)")
    print("=" * 74)

    try:
        queries, corpus_mode = build_queries(n)
    except Exception as exc:
        print(f"[ERROR] Could not build queries: {exc}")
        if "OfficialCorpusUnavailable" in type(exc).__name__:
            print("Build the official sample first or unset THINKZEN_CORPUS to benchmark the demo corpus.")
        return 1

    distinct = len(set(queries))
    print(f"  active corpus : {corpus_mode.upper()}")
    print(f"  queries       : {len(queries)} total ({distinct} distinct"
          f"{'; cycled to reach count' if distinct < len(queries) else ''})")
    print(f"  transport     : {'live ' + args.base_url if args.base_url else 'in-process TestClient'}")

    try:
        client = LiveClient(args.base_url) if args.base_url else InProcessClient()
    except Exception as exc:
        print(f"\n[ERROR] Could not initialize client ({exc}). Run from repo root with the venv deps.")
        return 1

    # ---- Scope 1: RAG_ONLY --------------------------------------------------- #
    # Warm-up (NOT counted): triggers one-time lazy corpus seeding.
    cold_start_ms = None
    try:
        t0 = time.perf_counter()
        _ = client.query(queries[0])
        cold_start_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    except Exception as exc:
        print(f"\n[ERROR] Warm-up query failed: {exc}")
        if args.base_url is None and corpus_mode == "official":
            print("If official mode: ensure the sample artifact is built (build_msmarco_xi_sample.py).")
        return 1

    client_wall_ms: list[float] = []
    server_internal_ms: list[float] = []
    refused_count = 0
    for q in queries:
        t0 = time.perf_counter()
        data = client.query(q)
        client_wall_ms.append((time.perf_counter() - t0) * 1000.0)
        tel = data.get("telemetry", {})
        srv = tel.get("total_latency_ms")
        if isinstance(srv, (int, float)):
            server_internal_ms.append(float(srv))
        if data.get("refused"):
            refused_count += 1

    wall_stats = _stats(client_wall_ms)
    srv_stats = _stats(server_internal_ms)

    print("\n" + "-" * 74)
    print(f"COLD START (one-time, EXCLUDED from percentiles): {cold_start_ms} ms "
          f"(includes lazy corpus seed on first request)")
    _print_scope("RAG_ONLY — client-observed wall-clock",
                 "POST /api/v1/query round-trip; STT EXCLUDED", wall_stats)
    _print_scope("RAG_ONLY — server internal pipeline (telemetry.total_latency_ms)",
                 "in-pipeline analyze→retrieve→rerank→evidence→generate; STT EXCLUDED", srv_stats)
    print(f"  (grounded={len(queries) - refused_count}, refused={refused_count} of {len(queries)})")

    # ---- Scopes 2 & 3: SARVAM_STT + FULL_VOICE_E2E --------------------------- #
    stt_stats = {"count": 0}
    e2e_stats = {"count": 0}
    stt_skip_reason = None

    key_set = bool(os.getenv("SARVAM_API_KEY", "").strip())
    audio_files: list[Path] = []
    if args.audio_dir:
        d = Path(args.audio_dir).expanduser()
        if d.is_dir():
            audio_files = sorted(p for p in d.iterdir() if p.suffix.lower() in _AUDIO_EXTS)

    if not key_set:
        stt_skip_reason = "SARVAM_API_KEY not set — real STT cannot run (transcript would be fabricated otherwise)."
    elif not args.audio_dir:
        stt_skip_reason = "no --audio-dir provided (no audio to transcribe)."
    elif not audio_files:
        stt_skip_reason = f"no audio files found in {args.audio_dir}."

    if stt_skip_reason:
        print("\nSARVAM_STT       — SKIPPED: " + stt_skip_reason)
        print("FULL_VOICE_E2E   — SKIPPED: cannot measure full voice latency without real STT.")
        print("  NOTE: RAG-only latency above must NOT be cited as full voice latency.")
    else:
        stt_ms: list[float] = []
        e2e_ms: list[float] = []
        for ap in audio_files:
            t0 = time.perf_counter()
            stt = client.stt(ap, args.language)
            stt_dt = (time.perf_counter() - t0) * 1000.0
            if not stt.get("success") or not (stt.get("transcript") or "").strip():
                print(f"  [skip] {ap.name}: STT produced no transcript ({stt.get('message')})")
                continue
            stt_ms.append(stt_dt)
            transcript = stt["transcript"].strip()
            t1 = time.perf_counter()
            client.query(transcript)
            q_dt = (time.perf_counter() - t1) * 1000.0
            e2e_ms.append(stt_dt + q_dt)  # full voice = STT + query
        stt_stats = _stats(stt_ms)
        e2e_stats = _stats(e2e_ms)
        _print_scope("SARVAM_STT — real Sarvam transcription",
                     "POST /api/v1/stt round-trip (audio→transcript)", stt_stats)
        _print_scope("FULL_VOICE_E2E — audio → transcript → grounded answer",
                     "SARVAM_STT + RAG_ONLY combined per audio file", e2e_stats)

    # ---- Persist a truthful JSON artifact ------------------------------------ #
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_mode": corpus_mode,
        "transport": args.base_url or "in-process-testclient",
        "query_count": len(queries),
        "distinct_queries": distinct,
        "cold_start_ms_excluded": cold_start_ms,
        "scopes": {
            "RAG_ONLY_client_wall_ms": wall_stats,
            "RAG_ONLY_server_internal_ms": srv_stats,
            "SARVAM_STT_ms": stt_stats,
            "FULL_VOICE_E2E_ms": e2e_stats,
        },
        "stt_skipped_reason": stt_skip_reason,
        "percentile_method": "modules.telemetry._compute_percentile (linear interpolation; same as Judge Mode)",
        "note": (
            "RAG_ONLY excludes speech-to-text. FULL_VOICE_E2E is the only voice-latency figure and "
            "is present only when real Sarvam STT ran. No <200ms claim is made; all values measured."
        ),
    }
    out_path = Path(args.out).expanduser() if args.out else (REPO_ROOT / "eval_results" / "latency_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"Saved: {out_path}")
    print("Scopes are independent; do not conflate RAG_ONLY with FULL_VOICE_E2E.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
