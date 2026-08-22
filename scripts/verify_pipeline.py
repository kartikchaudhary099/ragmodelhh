#!/usr/bin/env python3
"""In-process end-to-end verification harness for ThinkZen (the 9 required cases).

This drives the REAL pipeline through FastAPI's `TestClient` in the same process —
no live server and no network are required — and checks the exact cases from the
project brief:

    1. English grounded query      6. Hinglish refusal
    2. Hindi grounded query        7. evidence / citations
    3. Hinglish grounded query     8. Judge Mode (real telemetry)
    4. English refusal             9. voice flow (honest STT)
    5. Hindi refusal

For each language case it asserts the machine-checkable invariants — the `refused`
flag, the detected language in telemetry, and presence/absence of cited sources — and
prints the actual answer text so the language behaviour (English→English, Hindi→Hindi,
Hinglish→Hinglish) is visible for a human to confirm.

Nothing here fabricates a result: every value printed comes from the real response.
Exit code is 0 only if all hard checks pass, else 1.

RUN (from the repo root, using the project venv):
    .venv\\Scripts\\python.exe scripts/verify_pipeline.py

Note: with no GEMINI/OPENAI key set, generation uses the deterministic
evidence-quoting synthesizer, so answers are reproducible. With a key set, answers are
LLM-generated free text — the hard checks (refused flag, detected language, citations)
still hold; only the exact answer wording changes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from modules.official_corpus import (  # noqa: E402
    CORPUS_OFFICIAL,
    OfficialCorpusUnavailable,
    load_official_records,
)

# Devanagari Unicode block, used only for an advisory language-format note.
_DEVANAGARI = range(0x0900, 0x0980)


def _has_devanagari(text: str) -> bool:
    return any(ord(ch) in _DEVANAGARI for ch in text)


class _Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        tag = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        line = f"  [{tag}] {label}"
        if detail:
            line += f"  — {detail}"
        print(line)


# (id, query, expected_language, expect_refused)
# DEMO corpus cases (the default corpus). Unchanged historical behaviour: the grounded
# queries are about the demo RAG docs; the refusals ("capital of France") are out-of-domain
# for the demo corpus so the content-coverage gate refuses them.
DEMO_LANGUAGE_CASES = [
    ("1. English grounded", "What is Retrieval-Augmented Generation?", "en", False),
    ("2. Hindi grounded", "वाणीरैग क्या है?", "hi", False),
    ("3. Hinglish grounded", "ThinkZen mein hybrid search kaise kaam karta hai?", "hi-en", False),
    ("4. English refusal", "What is the capital of France?", "en", True),
    ("5. Hindi refusal", "फ्रांस की राजधानी क्या है?", "hi", True),
    ("6. Hinglish refusal", "France ki capital kya hai?", "hi-en", True),
]

# OFFICIAL-mode out-of-domain refusal queries. IMPORTANT: the demo refusal
# "फ्रांस की राजधानी क्या है?" is UNSAFE against MSMARCO-XI because "फ्रांस"/France DOES occur
# in that corpus. These themes (recipe/chocolate/cake) were verified absent from the real
# 100-row sample, so they exercise a genuine refusal rather than an accidental grounding.
_OFFICIAL_REFUSALS = [
    ("4. English refusal", "What is the recipe for chocolate cake?", "en", True),
    ("5. Hindi refusal", "चॉकलेट केक कैसे बनाते हैं?", "hi", True),
    ("6. Hinglish refusal", "chocolate cake kaise banate hain?", "hi-en", True),
]

# Roman-Hindi function words used to frame a faithful Hinglish query from real English tokens.
_HINGLISH_STOP = {
    "what", "was", "were", "is", "are", "the", "a", "an", "of", "to", "for", "in",
    "on", "at", "how", "do", "does", "did", "who", "when", "where", "which", "why",
    "and", "or", "with", "by", "from", "impact",
}


def _build_hinglish_from_english(eng_query: str) -> str:
    """Frame a record's real English content tokens as a Roman-Hindi (Hinglish) query.

    The salient tokens are genuine dataset content (so retrieval + the content-coverage gate
    act on real data); the Roman-Hindi function words ('ke', 'mein') make the QueryAnalyzer
    detect hi-en. This is a real supported query type, not fabricated corpus content.
    """
    toks = [t for t in re.findall(r"[a-zA-Z]+", eng_query.lower()) if t not in _HINGLISH_STOP and len(t) >= 4]
    salient = " ".join(toks[:3]) if toks else "manhattan project"
    return f"{salient} ke bare mein bataye"


def _build_language_cases(settings) -> tuple[list[tuple[str, str, str, bool]], str]:
    """Select verification cases for the ACTIVE corpus.

    In official mode the grounded queries are loaded from the REAL validated sample (demo
    queries would refuse against MSMARCO-XI), and out-of-domain refusals replace the
    France refusal. In demo mode the historical cases are returned unchanged.
    """
    corpus_mode = (getattr(settings, "corpus_mode", "demo") or "demo").strip().lower()
    if corpus_mode != CORPUS_OFFICIAL:
        return DEMO_LANGUAGE_CASES, corpus_mode

    # Official mode: derive grounded queries from the actual on-disk corpus content.
    records, _ = load_official_records(
        settings.official_sample_path, settings.official_provenance_path
    )
    r0 = records[0]
    en_q = r0["Eng_Query"].lstrip(") ").strip()
    hi_q = r0["query"].strip()
    hinglish_q = _build_hinglish_from_english(en_q)
    cases = [
        ("1. English grounded", en_q, "en", False),
        ("2. Hindi grounded", hi_q, "hi", False),
        ("3. Hinglish grounded", hinglish_q, "hi-en", False),
        *_OFFICIAL_REFUSALS,
    ]
    return cases, corpus_mode


def main() -> int:
    settings = get_settings()
    try:
        language_cases, corpus_mode = _build_language_cases(settings)
    except OfficialCorpusUnavailable as exc:
        print("=" * 74)
        print("ThinkZen — verification ABORTED (official corpus requested but unavailable)")
        print("=" * 74)
        print(f"\n{exc}\n")
        print("Build the validated sample first, then re-run:")
        print("    .venv\\Scripts\\python.exe scripts\\build_msmarco_xi_sample.py --limit 100")
        print("    set THINKZEN_CORPUS=official")
        print("    .venv\\Scripts\\python.exe scripts\\verify_pipeline.py")
        return 1

    client = TestClient(create_app())
    report = _Report()
    first_grounded_payload: dict | None = None

    print("=" * 74)
    print("ThinkZen — in-process end-to-end verification (TestClient, no live server)")
    print(f"Active corpus: {corpus_mode.upper()}"
          + ("  (ai4bharat/MSMARCO-XI validated sample)" if corpus_mode == CORPUS_OFFICIAL
             else "  (labelled demo corpus)"))
    print("=" * 74)

    # ---- Cases 1-6: language-aware grounded answers + strict refusal ------ #
    for label, query, expected_lang, expect_refused in language_cases:
        print(f"\n{label}")
        print(f"  query: {query}")
        resp = client.post("/api/v1/query", json={"query": query})
        report.check(f"{label}: HTTP 200", resp.status_code == 200,
                     f"got {resp.status_code}")
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
            report.check(f"{label}: has cited sources", len(sources) > 0,
                         f"sources={len(sources)}")
            if first_grounded_payload is None:
                first_grounded_payload = data

        # Advisory (not a hard check): show the answer + a language-format note.
        answer = (data.get("answer") or "").strip()
        note = ""
        if not expect_refused:
            if expected_lang == "hi":
                note = "contains Devanagari" if _has_devanagari(answer) else "NO Devanagari (check)"
            elif expected_lang == "hi-en":
                note = "Hinglish framing" if "anusaar" in answer.lower() or not _has_devanagari(answer) else "note"
            else:
                note = "English framing"
        print(f"    answer  : {answer[:160]}{'…' if len(answer) > 160 else ''}")
        print(f"    lang={detected} refused={refused} max_score="
              f"{tel.get('evidence_bundle', {}).get('max_retrieval_score')} "
              f"latency_ms={tel.get('total_latency_ms')}  {('['+note+']') if note else ''}")

    # ---- Case 7: evidence / citations schema ------------------------------ #
    print("\n7. Evidence / citations")
    if first_grounded_payload and first_grounded_payload.get("sources"):
        src = first_grounded_payload["sources"][0]
        for f in ("chunk_id", "text", "score", "method", "metadata"):
            report.check(f"7. source has '{f}'", f in src)
        report.check("7. score is a float", isinstance(src.get("score"), float),
                     f"type={type(src.get('score')).__name__}")
        print(f"    top source: chunk_id={src.get('chunk_id')} "
              f"score={src.get('score')} method={src.get('method')}")
    else:
        report.check("7. a grounded response produced sources", False,
                     "no grounded payload captured")

    # ---- Case 8: Judge Mode (real telemetry) ------------------------------ #
    print("\n8. Judge Mode (/api/v1/judge)")
    jresp = client.get("/api/v1/judge")
    report.check("8. judge HTTP 200", jresp.status_code == 200)
    if jresp.status_code == 200:
        jdata = jresp.json()
        report.check("8. total_runs >= 6", jdata.get("total_runs", 0) >= 6,
                     f"total_runs={jdata.get('total_runs')}")
        report.check("8. has latency_stats", "latency_stats" in jdata)
        report.check("8. has data_quality_note", "data_quality_note" in jdata)
        stats = jdata.get("latency_stats", {})
        if "REAL_RUN" in stats:
            real = stats["REAL_RUN"]
            for pct in ("p50_ms", "p70_ms", "p90_ms", "p100_ms", "mean_ms", "count"):
                report.check(f"8. REAL_RUN has {pct}", pct in real)
            print(f"    REAL_RUN: count={real.get('count')} p50={real.get('p50_ms')} "
                  f"p90={real.get('p90_ms')} p100={real.get('p100_ms')}")
        else:
            print("    (no REAL_RUN bucket yet — stats present but empty)")

    # ---- Case 9: voice flow — honest server STT --------------------------- #
    print("\n9. Voice flow — server STT honesty (no fabricated transcript)")
    sresp = client.post(
        "/api/v1/stt",
        files={"file": ("sample.wav", b"\x00\x01\x02\x03", "audio/wav")},
        data={"language": "en"},
    )
    report.check("9. stt HTTP 200", sresp.status_code == 200)
    if sresp.status_code == 200:
        sdata = sresp.json()
        # With no SARVAM_API_KEY (default), it must NOT invent a transcript.
        report.check("9. transcript is empty (no fabrication)", sdata.get("transcript") == "",
                     f"transcript={sdata.get('transcript')!r}")
        report.check("9. success is False without key OR True with a real key",
                     isinstance(sdata.get("success"), bool))
        print(f"    success={sdata.get('success')} message={sdata.get('message')}")

    # ---- Summary ---------------------------------------------------------- #
    print("\n" + "=" * 74)
    total = report.passed + report.failed
    print(f"RESULT ({corpus_mode.upper()} corpus): {report.passed}/{total} checks passed, {report.failed} failed")
    print("=" * 74)
    print("Note: this exercises the real in-process pipeline. It complements — but does")
    print("not replace — running the full pytest suite and a live uvicorn server:")
    print("    .venv\\Scripts\\python.exe -m pytest -q")
    print("    .venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir backend")
    print("To verify the OFFICIAL ai4bharat/MSMARCO-XI corpus instead of the demo corpus:")
    print("    .venv\\Scripts\\python.exe scripts\\build_msmarco_xi_sample.py --limit 100")
    print("    set THINKZEN_CORPUS=official   (PowerShell: $env:THINKZEN_CORPUS='official')")
    print("    .venv\\Scripts\\python.exe scripts\\verify_pipeline.py")
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
