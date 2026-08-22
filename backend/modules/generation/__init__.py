"""
Generation module.

Future implementations will produce grounded answers from retrieved context.
Swap LLMs by implementing Generator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from modules.retrieval import RetrievedDocument


@dataclass
class GeneratedAnswer:
    """A generated answer with optional citations."""

    text: str
    sources: list[RetrievedDocument]
    refused: bool = False
    refusal_reason: str | None = None


class Generator(ABC):
    """Abstract interface for answer generation."""

    @abstractmethod
    async def generate(
        self, query: str, context: list[RetrievedDocument]
    ) -> GeneratedAnswer:
        """Generate an answer grounded in retrieved context."""
        ...
