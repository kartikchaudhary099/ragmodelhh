"""Speech-to-Text API endpoint — server-side voice transcription.

Design note (honesty):
  The PRIMARY voice path in ThinkZen is the browser-native Web Speech API, which runs
  entirely client-side in the frontend (see frontend/static/app.js). This server endpoint
  is an OPTIONAL fallback for server-side transcription and performs a REAL call to the
  Sarvam AI speech-to-text service when a SARVAM_API_KEY is configured.

  It never returns a fabricated transcript. If no API key is configured (the default for
  a local/offline run), it responds with success=False and an explanatory message so that
  callers — including judges testing the API directly — are never misled into thinking a
  canned string came from their audio.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["stt"])

# Sarvam AI speech-to-text REST endpoint.
_SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
_SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saarika:v2")

# Map the UI / caller language hints to Sarvam language codes.
# "unknown" enables Sarvam's automatic language detection.
_LANGUAGE_CODE_MAP = {
    "auto": "unknown",
    "unknown": "unknown",
    "en": "en-IN",
    "en-in": "en-IN",
    "hi": "hi-IN",
    "hi-in": "hi-IN",
    "hi-en": "unknown",  # Hinglish → let Sarvam auto-detect
}


class STTResponse(BaseModel):
    """Transcription response model."""
    transcript: str
    language: str
    success: bool
    message: str | None = None


def _resolve_language_code(language: str) -> str:
    """Normalize a caller language hint to a Sarvam language code."""
    return _LANGUAGE_CODE_MAP.get((language or "auto").strip().lower(), "unknown")


@router.post("/stt", response_model=STTResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(default="auto"),
) -> STTResponse:
    """Transcribe an uploaded audio file to text.

    Behavior:
      * SARVAM_API_KEY set  → performs a real Sarvam STT request and returns the
        transcript it produces (or an honest error if the request fails).
      * SARVAM_API_KEY unset → returns success=False with guidance to use the
        browser-native voice input (Web Speech API) instead. No fake transcript.
    """
    try:
        audio_bytes = await file.read()
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        logger.error("Failed to read uploaded audio: %s", exc)
        return STTResponse(
            transcript="", language=language, success=False,
            message=f"Could not read uploaded audio: {exc}",
        )

    if not audio_bytes:
        return STTResponse(
            transcript="", language=language, success=False,
            message="Uploaded audio file is empty.",
        )

    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        logger.info(
            "STT called with %d bytes but SARVAM_API_KEY is not configured; "
            "returning honest 'not configured' response.", len(audio_bytes),
        )
        return STTResponse(
            transcript="",
            language=language,
            success=False,
            message=(
                "Server-side speech-to-text is not configured (no SARVAM_API_KEY). "
                "Use the in-browser microphone (Web Speech API), which is the primary "
                "voice path and needs no server key. To enable this endpoint, set "
                "SARVAM_API_KEY in the backend environment."
            ),
        )

    language_code = _resolve_language_code(language)

    # Real Sarvam STT request. httpx is a project dependency; import locally so a missing
    # optional dependency degrades to an honest error rather than a hard import failure.
    try:
        import httpx
    except ImportError:
        return STTResponse(
            transcript="", language=language, success=False,
            message="Server-side STT requires the 'httpx' package, which is not installed.",
        )

    files = {"file": (file.filename or "audio.wav", audio_bytes, file.content_type or "audio/wav")}
    data = {"model": _SARVAM_MODEL, "language_code": language_code}
    headers = {"api-subscription-key": api_key}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(_SARVAM_STT_URL, data=data, files=files, headers=headers)

        if response.status_code != 200:
            logger.warning("Sarvam STT returned HTTP %d: %s", response.status_code, response.text[:200])
            return STTResponse(
                transcript="", language=language, success=False,
                message=f"Sarvam STT request failed (HTTP {response.status_code}).",
            )

        payload = response.json()
        transcript = (payload.get("transcript") or "").strip()
        detected = payload.get("language_code") or language_code

        if not transcript:
            return STTResponse(
                transcript="", language=detected, success=False,
                message="Sarvam STT returned no transcript for the provided audio.",
            )

        logger.info("Sarvam STT succeeded: %d chars, language=%s", len(transcript), detected)
        return STTResponse(
            transcript=transcript,
            language=detected,
            success=True,
            message="Transcribed via Sarvam speech-to-text.",
        )
    except Exception as exc:
        logger.error("Sarvam STT request error: %s", exc)
        return STTResponse(
            transcript="", language=language, success=False,
            message=f"Speech-to-text request error: {exc}",
        )
