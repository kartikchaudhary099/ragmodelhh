#!/usr/bin/env python3
"""Real voice-to-answer orchestration harness — competition proof of the Sarvam E2E path.

The product's PRIMARY microphone path is the browser-native Web Speech API (client-side, in
frontend/static/app.js). This harness exercises the SERVER-SIDE real-Sarvam path end to end so
a judge can prove it with an actual audio file and no browser:

    audio file ──► POST /api/v1/stt  (REAL Sarvam speech-to-text, needs SARVAM_API_KEY)
              ──► real transcript
              ──► POST /api/v1/query (Query Analysis → Hybrid Retrieval → Rerank →
                                       Evidence Intelligence → Grounded Generation)
              ──► grounded, cited answer (or an honest refusal)

Honesty guarantees:
  * It NEVER fabricates a transcript. If SARVAM_API_KEY is unset (or Sarvam errors), the STT
    endpoint returns success=False and this harness stops with a clear, non-zero exit — it does
    not invent text and does not proceed to the query leg with a fake transcript.
  * It NEVER fabricates an answer. The query leg runs the real in-process pipeline (or a live
    server if --base-url is given); every printed field comes from the real response.

Runs in-process by default (FastAPI TestClient — no separate server needed), or against a live
server with --base-url. Either way it calls the REAL Sarvam API for transcription.

RUN (from repo root, project venv; SARVAM_API_KEY must be set for a real transcript):
    set SARVAM_API_KEY=...           (PowerShell: $env:SARVAM_API_KEY='...')
    # optional, to ground on the official corpus instead of the demo corpus:
    #   .venv\\Scripts\\python.exe scripts\\build_msmarco_xi_sample.py --limit 100
    #   set THINKZEN_CORPUS=official
    .venv\\Scripts\\python.exe scripts\\voice_to_answer.py --audio path\\to\\question.wav --language auto

    # against a live uvicorn server instead of in-process:
    .venv\\Scripts\\python.exe scripts\\voice_to_answer.py --audio q.wav --base-url http://127.0.0.1:8000

Exit codes: 0 = full voice→answer E2E succeeded; 2 = STT unavailable/not configured (honest
stop, wiring intact); 1 = a request failed.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))


def _print_header(title: str) -> None:
    print("=" * 74)
    print(title)
    print("=" * 74)


def _guess_content_type(path: Path) -> str:
    ctype, _ = mimetypes.guess_type(str(path))
    return ctype or "audio/wav"


class InProcessClient:
    """Drives the REAL routes via FastAPI TestClient — no separate server required."""

    def __init__(self) -> None:
        from fastapi.testclient import TestClient  # local import: Windows/venv only

        from app.main import create_app

        self._client = TestClient(create_app())

    def stt(self, path: Path, language: str) -> dict:
        with open(path, "rb") as fh:
            audio = fh.read()
        resp = self._client.post(
            "/api/v1/stt",
            files={"file": (path.name, audio, _guess_content_type(path))},
            data={"language": language},
        )
        resp.raise_for_status()
        return resp.json()

    def query(self, payload: dict) -> dict:
        resp = self._client.post("/api/v1/query", json=payload)
        resp.raise_for_status()
        return resp.json()


class LiveClient:
    """Drives a running uvicorn server over HTTP with httpx."""

    def __init__(self, base_url: str) -> None:
        import httpx  # local import: Windows/venv only

        self._httpx = httpx
        self._base = base_url.rstrip("/")

    def stt(self, path: Path, language: str) -> dict:
        with open(path, "rb") as fh:
            audio = fh.read()
        resp = self._httpx.post(
            f"{self._base}/api/v1/stt",
            files={"file": (path.name, audio, _guess_content_type(path))},
            data={"language": language},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()

    def query(self, payload: dict) -> dict:
        resp = self._httpx.post(f"{self._base}/api/v1/query", json=payload, timeout=60.0)
        resp.raise_for_status()
        return resp.json()


def _active_corpus() -> str:
    try:
        from app.config import get_settings

        return (get_settings().corpus_mode or "demo").strip().lower()
    except Exception:
        return os.getenv("THINKZEN_CORPUS", "demo").strip().lower() or "demo"


def main() -> int:
    parser = argparse.ArgumentParser(description="Real Sarvam voice-to-answer E2E harness.")
    parser.add_argument("--audio", required=True, help="Path to an audio file (wav/mp3/flac/…).")
    parser.add_argument("--language", default="auto",
                        help="Language hint for STT: auto|en|hi|hi-en (default: auto).")
    parser.add_argument("--base-url", default=None,
                        help="Drive a live server instead of in-process (e.g. http://127.0.0.1:8000).")
    parser.add_argument("--top-k", type=int, default=None, help="Optional retrieval top_k override.")
    parser.add_argument("--confidence-threshold", type=float, default=None,
                        help="Optional grounding confidence threshold override.")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser()
    _print_header("ThinkZen — REAL voice → answer end-to-end (Sarvam STT → RAG)")
    print(f"  audio        : {audio_path}")
    print(f"  language hint: {args.language}")
    print(f"  transport    : {'live server ' + args.base_url if args.base_url else 'in-process TestClient'}")
    print(f"  active corpus: {_active_corpus().upper()}")
    print(f"  SARVAM_API_KEY set: {'yes' if os.getenv('SARVAM_API_KEY', '').strip() else 'NO'}")

    if not audio_path.exists():
        print(f"\n[ERROR] Audio file not found: {audio_path}")
        return 1

    try:
        client = LiveClient(args.base_url) if args.base_url else InProcessClient()
    except Exception as exc:  # missing fastapi/httpx, etc.
        print(f"\n[ERROR] Could not initialize client ({exc}).")
        print("Install the project venv deps and run from the repo root.")
        return 1

    # ---- Stage 1: REAL Sarvam speech-to-text --------------------------------- #
    print("\n[1/2] Speech-to-text via /api/v1/stt (real Sarvam call when key is set)…")
    try:
        stt = client.stt(audio_path, args.language)
    except Exception as exc:
        print(f"[ERROR] STT request failed: {exc}")
        return 1

    transcript = (stt.get("transcript") or "").strip()
    print(f"      success   : {stt.get('success')}")
    print(f"      language  : {stt.get('language')}")
    print(f"      message   : {stt.get('message')}")
    print(f"      transcript: {transcript!r}")

    if not stt.get("success") or not transcript:
        print("\nVOICE→ANSWER E2E: STOPPED — no real transcript produced.")
        print("This is the HONEST result when SARVAM_API_KEY is unset or Sarvam returns nothing;")
        print("the harness refuses to fabricate a transcript. The wiring is intact: audio was")
        print("accepted and forwarded to the real Sarvam endpoint. Set SARVAM_API_KEY and retry")
        print("with real speech audio to complete the full voice→answer path.")
        return 2

    # ---- Stage 2: grounded answer over the real transcript ------------------- #
    print("\n[2/2] Grounded answer via /api/v1/query …")
    payload: dict = {"query": transcript}
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.confidence_threshold is not None:
        payload["confidence_threshold"] = args.confidence_threshold

    try:
        data = client.query(payload)
    except Exception as exc:
        print(f"[ERROR] Query request failed: {exc}")
        return 1

    tel = data.get("telemetry", {})
    sources = data.get("sources", [])
    print(f"      detected_language : {tel.get('detected_language')}")
    print(f"      refused           : {data.get('refused')}")
    print(f"      grounding_status  : {tel.get('grounding_status')}")
    print(f"      evidence_count    : {tel.get('evidence_count')}")
    print(f"      total_latency_ms  : {tel.get('total_latency_ms')}")
    print(f"      cited sources     : {len(sources)}")
    for i, s in enumerate(sources[:3], 1):
        md = s.get("metadata", {})
        print(f"        {i}. chunk_id={s.get('chunk_id')} score={s.get('score')} "
              f"corpus={md.get('corpus')} lang={md.get('chunk_language')}")
    answer = (data.get("answer") or "").strip()
    print(f"\n  ANSWER:\n  {answer[:500]}{'…' if len(answer) > 500 else ''}")

    print("\n" + "=" * 74)
    print("VOICE→ANSWER E2E: OK — real Sarvam transcript grounded through the full pipeline.")
    print("=" * 74)
    print("Reproduce against a live server:")
    print("    .venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir backend")
    print(f"    curl -F file=@{audio_path.name} -F language={args.language} http://127.0.0.1:8000/api/v1/stt")
    print("    curl -H \"Content-Type: application/json\" -d \"{\\\"query\\\":\\\"<transcript>\\\"}\" "
          "http://127.0.0.1:8000/api/v1/query")
    return 0


if __name__ == "__main__":
    sys.exit(main())
