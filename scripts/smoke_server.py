#!/usr/bin/env python3
"""Step 16 automation: start the REAL uvicorn server and verify endpoints over HTTP.

Unlike `verify_pipeline.py` (which drives the app in-process via TestClient), this script
launches an actual `uvicorn app.main:app --app-dir backend` subprocess, waits for
`/health`, then exercises the nine required cases plus `/api/v1/judge` and `/api/v1/stt`
over real HTTP, prints a PASS/FAIL report, and shuts the server down.

Nothing is fabricated — every value printed comes from the live server's real responses.
Exit code is 0 only if all hard checks pass, else 1 (2 if the server never came up).

RUN (from the repo root, using the project venv):
    .venv\\Scripts\\python.exe scripts\\smoke_server.py

Options:
    --host              default 127.0.0.1
    --port              default 8000
    --startup-timeout   seconds to wait for /health (default 40)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import httpx
except ImportError:  # pragma: no cover
    print("This smoke test needs 'httpx' (it is in backend/requirements-dev.txt). "
          "Install dev deps into the venv first:  .venv\\Scripts\\python.exe -m pip "
          "install -r backend/requirements-dev.txt")
    sys.exit(3)

_DEVANAGARI = range(0x0900, 0x0980)

# (label, query, expected_language, expect_refused)
LANGUAGE_CASES = [
    ("1. English grounded", "What is Retrieval-Augmented Generation?", "en", False),
    ("2. Hindi grounded", "वाणीरैग क्या है?", "hi", False),
    ("3. Hinglish grounded", "ThinkZen mein hybrid search kaise kaam karta hai?", "hi-en", False),
    ("4. English refusal", "What is the capital of France?", "en", True),
    ("5. Hindi refusal", "फ्रांस की राजधानी क्या है?", "hi", True),
    ("6. Hinglish refusal", "France ki capital kya hai?", "hi-en", True),
]


def _has_devanagari(text: str) -> bool:
    return any(ord(ch) in _DEVANAGARI for ch in text)


class _Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        tag = "PASS" if condition else "FAIL"
        self.passed += int(bool(condition))
        self.failed += int(not condition)
        print(f"  [{tag}] {label}" + (f"  — {detail}" if detail else ""))


def _wait_for_health(base_url: str, timeout: float, server_log: Path) -> bool:
    deadline = time.time() + timeout
    last_err: object = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                print(f"  server healthy: {r.json()}")
                return True
        except Exception as exc:  # noqa: BLE001 - connection retries are expected
            last_err = exc
        time.sleep(0.5)
    print(f"\n[!] Server did not become healthy within {timeout:.0f}s "
          f"(last error: {last_err}).")
    if server_log.exists():
        tail = server_log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        print("--- last lines of server log ---")
        print("\n".join(tail))
        print("--------------------------------")
    return False


def _run_checks(base_url: str) -> int:
    report = _Report()
    first_grounded: dict | None = None

    # ---- Cases 1-6 ----
    for label, query, expected_lang, expect_refused in LANGUAGE_CASES:
        print(f"\n{label}\n  query: {query}")
        try:
            resp = httpx.post(f"{base_url}/api/v1/query", json={"query": query}, timeout=30.0)
        except Exception as exc:  # noqa: BLE001
            report.check(f"{label}: request", False, f"error: {exc}")
            continue
        report.check(f"{label}: HTTP 200", resp.status_code == 200, f"got {resp.status_code}")
        if resp.status_code != 200:
            continue
        data = resp.json()
        tel = data.get("telemetry", {})
        detected = tel.get("detected_language")
        refused = data.get("refused")
        sources = data.get("sources", [])
        report.check(f"{label}: detected language == {expected_lang}",
                     detected == expected_lang, f"detected={detected}")
        report.check(f"{label}: refused == {expect_refused}",
                     refused == expect_refused, f"refused={refused}")
        if expect_refused:
            report.check(f"{label}: grounding_status == refused",
                         tel.get("grounding_status") == "refused",
                         f"status={tel.get('grounding_status')}")
        else:
            report.check(f"{label}: has cited sources", len(sources) > 0, f"sources={len(sources)}")
            if first_grounded is None:
                first_grounded = data
        answer = (data.get("answer") or "").strip()
        note = ""
        if not expect_refused:
            if expected_lang == "hi":
                note = "Devanagari" if _has_devanagari(answer) else "NO Devanagari (check)"
            elif expected_lang == "hi-en":
                note = "Hinglish" if ("anusaar" in answer.lower() or not _has_devanagari(answer)) else "note"
            else:
                note = "English"
        print(f"    answer  : {answer[:160]}{'…' if len(answer) > 160 else ''}")
        print(f"    lang={detected} refused={refused} "
              f"max_score={tel.get('evidence_bundle', {}).get('max_retrieval_score')} "
              f"latency_ms={tel.get('total_latency_ms')}  {('['+note+']') if note else ''}")

    # ---- Case 7: citations ----
    print("\n7. Evidence / citations")
    if first_grounded and first_grounded.get("sources"):
        src = first_grounded["sources"][0]
        for f in ("chunk_id", "text", "score", "method", "metadata"):
            report.check(f"7. source has '{f}'", f in src)
        report.check("7. score is a float", isinstance(src.get("score"), float))
        print(f"    top source: chunk_id={src.get('chunk_id')} score={src.get('score')} "
              f"method={src.get('method')}")
    else:
        report.check("7. a grounded response produced sources", False, "none captured")

    # ---- Case 8: Judge Mode ----
    print("\n8. Judge Mode (/api/v1/judge)")
    jr = httpx.get(f"{base_url}/api/v1/judge", timeout=10.0)
    report.check("8. judge HTTP 200", jr.status_code == 200)
    if jr.status_code == 200:
        jd = jr.json()
        report.check("8. total_runs >= 6", jd.get("total_runs", 0) >= 6, f"total_runs={jd.get('total_runs')}")
        report.check("8. has latency_stats", "latency_stats" in jd)
        report.check("8. has data_quality_note", "data_quality_note" in jd)
        stats = jd.get("latency_stats", {})
        if "REAL_RUN" in stats:
            real = stats["REAL_RUN"]
            for pct in ("p50_ms", "p70_ms", "p90_ms", "p100_ms", "mean_ms", "count"):
                report.check(f"8. REAL_RUN has {pct}", pct in real)
            print(f"    REAL_RUN: count={real.get('count')} p50={real.get('p50_ms')} "
                  f"p90={real.get('p90_ms')} p100={real.get('p100_ms')}")

    # ---- Case 9: honest STT ----
    print("\n9. Voice flow — server STT honesty (no fabricated transcript)")
    sr = httpx.post(
        f"{base_url}/api/v1/stt",
        files={"file": ("sample.wav", b"\x00\x01\x02\x03", "audio/wav")},
        data={"language": "en"},
        timeout=15.0,
    )
    report.check("9. stt HTTP 200", sr.status_code == 200)
    if sr.status_code == 200:
        sd = sr.json()
        report.check("9. transcript is empty (no fabrication)", sd.get("transcript") == "",
                     f"transcript={sd.get('transcript')!r}")
        report.check("9. success is a bool", isinstance(sd.get("success"), bool))
        print(f"    success={sd.get('success')} message={sd.get('message')}")

    print("\n" + "=" * 74)
    total = report.passed + report.failed
    print(f"LIVE-SERVER RESULT: {report.passed}/{total} checks passed, {report.failed} failed")
    print("=" * 74)
    return 0 if report.failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Start uvicorn and smoke-test endpoints over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=40.0)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    server_log_path = REPO_ROOT / "server_smoke.log"

    print("=" * 74)
    print("ThinkZen — live-server smoke test (real uvicorn over HTTP)")
    print("=" * 74)
    print(f"Launching: uvicorn app.main:app --app-dir backend --host {args.host} --port {args.port}")
    print(f"(server output → {server_log_path.name})")

    with open(server_log_path, "w", encoding="utf-8") as server_log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--app-dir", "backend",
             "--host", args.host, "--port", str(args.port)],
            cwd=str(REPO_ROOT),
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            if not _wait_for_health(base_url, args.startup_timeout, server_log_path):
                return 2
            exit_code = _run_checks(base_url)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()
            print("Server stopped.")
    print("\nNote: this is a real HTTP smoke test. For the full unit suite run:")
    print("    .venv\\Scripts\\python.exe -m pytest -q")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
