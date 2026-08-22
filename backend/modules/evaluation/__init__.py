"""
Evaluation module.

Future implementations will measure retrieval quality, answer faithfulness,
latency, and competition-specific metrics.
Swap evaluators by implementing Evaluator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Result of an evaluation run."""

    metric_name: str
    value: float
    details: dict | None = None


class Evaluator(ABC):
    """Abstract interface for evaluation strategies."""

    @abstractmethod
    async def evaluate(self, dataset_path: str) -> list[EvaluationResult]:
        """Run evaluation on a dataset and return metrics."""
        ...
