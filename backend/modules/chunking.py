from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Chunk representation designed to support future retrieval strategy selection."""

    text: str
    chunk_id: str
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, record: dict[str, Any]) -> list[Chunk]:
        """Build chunk objects from a normalized dataset record."""


class FixedSizeChunker(BaseChunker):
    """Baseline chunking with configurable chunk size and overlap."""

    def __init__(self, chunk_size: int = 200, overlap: int = 20) -> None:
        self.chunk_size = max(1, chunk_size)
        self.overlap = max(0, min(overlap, self.chunk_size - 1))

    def chunk(self, record: dict[str, Any]) -> list[Chunk]:
        text = self._resolve_text(record)
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            window = text[start:end]
            if not window.strip():
                break
            chunk_id = f"{record.get('record_id', 'unknown')}-fixed-{idx}"
            chunks.append(
                Chunk(
                    text=window.strip(),
                    chunk_id=chunk_id,
                    source_id=str(record.get("record_id", "unknown")),
                    metadata={
                        "chunk_role": "fixed",
                        "query_id": record.get("query_id", ""),
                        "language": record.get("language", "hi"),
                        "query_type": record.get("query_type", "unknown"),
                        "source_language": record.get("source_language", record.get("language", "hi")),
                        "target_language": record.get("target_language", record.get("language", "hi")),
                        "strategy": "fixed-size",
                    },
                )
            )
            idx += 1
            if end == len(text):
                break
            start = max(0, end - self.overlap)
        return chunks

    @staticmethod
    def _resolve_text(record: dict[str, Any]) -> str:
        if not record:
            return ""
        text = record.get("passage", "") or record.get("query", "") or ""
        return text.strip()


class SentenceChunker(BaseChunker):
    """Chunk by natural sentence boundaries, with a target-size ceiling."""

    def __init__(self, target_size: int = 180, min_sentence_length: int = 30) -> None:
        self.target_size = max(1, target_size)
        self.min_sentence_length = max(1, min_sentence_length)

    def chunk(self, record: dict[str, Any]) -> list[Chunk]:
        text = self._resolve_text(record)
        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_length = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and current_length + len(sentence) > self.target_size:
                chunks.append(self._build_chunk(record, " ".join(current), len(chunks)))
                current = [sentence]
                current_length = len(sentence)
            else:
                current.append(sentence)
                current_length += len(sentence)

        if current:
            chunks.append(self._build_chunk(record, " ".join(current), len(chunks)))
        return chunks

    @staticmethod
    def _build_chunk(record: dict[str, Any], text: str, index: int) -> Chunk:
        return Chunk(
            text=text.strip(),
            chunk_id=f"{record.get('record_id', 'unknown')}-sentence-{index}",
            source_id=str(record.get("record_id", "unknown")),
            metadata={
                "chunk_role": "sentence",
                "query_id": record.get("query_id", ""),
                "language": record.get("language", "hi"),
                "query_type": record.get("query_type", "unknown"),
                "source_language": record.get("source_language", record.get("language", "hi")),
                "target_language": record.get("target_language", record.get("language", "hi")),
                "strategy": "sentence-boundary",
            },
        )

    @staticmethod
    def _resolve_text(record: dict[str, Any]) -> str:
        if not record:
            return ""
        return str(record.get("passage", "") or record.get("query", "") or "").strip()


class ParentChildChunker(BaseChunker):
    """Create larger parent chunks and smaller child chunks with provenance links."""

    def __init__(self, parent_size: int = 250, child_size: int = 80, overlap: int = 15) -> None:
        self.parent_size = max(1, parent_size)
        self.child_size = max(1, child_size)
        self.overlap = max(0, min(overlap, self.child_size - 1))

    def chunk(self, record: dict[str, Any]) -> list[Chunk]:
        text = self._resolve_text(record)
        if not text:
            return []

        parent = Chunk(
            text=text.strip(),
            chunk_id=f"{record.get('record_id', 'unknown')}-parent",
            source_id=str(record.get("record_id", "unknown")),
            metadata={
                "chunk_role": "parent",
                "query_id": record.get("query_id", ""),
                "language": record.get("language", "hi"),
                "query_type": record.get("query_type", "unknown"),
                "source_language": record.get("source_language", record.get("language", "hi")),
                "target_language": record.get("target_language", record.get("language", "hi")),
                "strategy": "parent-child",
            },
        )

        child_chunks: list[Chunk] = []
        start = 0
        child_index = 0
        while start < len(text):
            end = min(start + self.child_size, len(text))
            window = text[start:end].strip()
            if not window:
                break
            child_chunks.append(
                Chunk(
                    text=window,
                    chunk_id=f"{record.get('record_id', 'unknown')}-child-{child_index}",
                    source_id=str(record.get("record_id", "unknown")),
                    metadata={
                        "chunk_role": "child",
                        "parent_id": parent.chunk_id,
                        "query_id": record.get("query_id", ""),
                        "language": record.get("language", "hi"),
                        "query_type": record.get("query_type", "unknown"),
                        "source_language": record.get("source_language", record.get("language", "hi")),
                        "target_language": record.get("target_language", record.get("language", "hi")),
                        "strategy": "parent-child",
                    },
                )
            )
            child_index += 1
            if end == len(text):
                break
            start = max(0, end - self.overlap)

        return [parent, *child_chunks]

    @staticmethod
    def _resolve_text(record: dict[str, Any]) -> str:
        if not record:
            return ""
        return str(record.get("passage", "") or record.get("query", "") or "").strip()


def normalize_record(record: dict[str, Any] | None) -> dict[str, Any]:
    """Compatibility wrapper for normalized raw dataset records."""
    from modules.data_pipeline import normalize_record as pipeline_normalize

    return pipeline_normalize(record)
