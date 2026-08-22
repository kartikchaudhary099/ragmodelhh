"""
Speech-to-text module.

Future implementations will convert voice input to text.
Swap providers by implementing STTProvider.
"""

from abc import ABC, abstractmethod


class STTProvider(ABC):
    """Abstract interface for speech-to-text providers."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        """Transcribe audio bytes to text."""
        ...
