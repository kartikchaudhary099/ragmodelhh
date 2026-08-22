"""Pipeline latency aggregator — multi-run P50/P70/P100 analytics.

This module records real pipeline run timing across sessions and computes
latency percentiles. All data is in-memory only (resets on server restart).

Label semantics:
    REAL_RUN    — actual user/query request processed by the live pipeline
    UNIT_TEST   — synthetic request from test suite (not counted in real analytics)
    PENDING     — placeholder; real data validation pending dataset availability
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Maximum number of runs to retain in memory
_MAX_RUNS = 1000


class RunLabel(str, Enum):
    REAL_RUN = "REAL_RUN"
    UNIT_TEST = "UNIT_TEST"
    PENDING = "PENDING"


@dataclass
class PipelineRun:
    """A single recorded pipeline execution."""
    query_id: str
    label: RunLabel
    total_ms: float
    query_analysis_ms: float
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    refused: bool
    grounding_status: str
    alpha_used: float
    query_type: str
    language: str
    evidence_count: int


@dataclass
class LatencyStats:
    """Computed latency statistics for a set of pipeline runs."""
    count: int
    p50_ms: float
    p70_ms: float
    p90_ms: float
    p100_ms: float
    mean_ms: float
    min_ms: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "p50_ms": round(self.p50_ms, 2),
            "p70_ms": round(self.p70_ms, 2),
            "p90_ms": round(self.p90_ms, 2),
            "p100_ms": round(self.p100_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "label": self.label,
            "data_quality_note": (
                "REAL_RUN: measured from live pipeline requests." if self.label == RunLabel.REAL_RUN
                else "UNIT_TEST: synthetic test data, not production representative."
            ),
        }


def _compute_percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a list of floats. p in [0, 100]."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p / 100.0
    lo = int(k)
    hi = lo + 1
    if hi >= len(sorted_vals):
        return sorted_vals[-1]
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class LatencyAggregator:
    """Thread-safe (GIL) in-memory aggregator for pipeline run latencies.

    Maintains a rolling window of the last N runs and computes P50/P70/P90/P100
    on demand. All stats are labeled with data quality markers.

    Usage::

        aggregator = LatencyAggregator()
        aggregator.record(run)
        stats = aggregator.compute_stats(label=RunLabel.REAL_RUN)
    """

    def __init__(self, max_runs: int = _MAX_RUNS) -> None:
        self._runs: deque[PipelineRun] = deque(maxlen=max_runs)

    def record(self, run: PipelineRun) -> None:
        """Record a pipeline run."""
        self._runs.append(run)

    def compute_stats(self, label: RunLabel = RunLabel.REAL_RUN) -> LatencyStats | None:
        """Compute latency statistics for runs matching the given label.

        Returns None if no runs exist for the requested label.
        Note: P50/P70/P100 are only meaningful with ≥10 runs. With fewer runs,
        the values are reported but labeled 'low_sample_warning'.
        """
        matching = [r for r in self._runs if r.label == label]
        if not matching:
            return None

        total_times = [r.total_ms for r in matching]
        return LatencyStats(
            count=len(matching),
            p50_ms=_compute_percentile(total_times, 50),
            p70_ms=_compute_percentile(total_times, 70),
            p90_ms=_compute_percentile(total_times, 90),
            p100_ms=max(total_times),
            mean_ms=statistics.mean(total_times),
            min_ms=min(total_times),
            label=label.value,
        )

    def get_all_runs(self, label: RunLabel | None = None) -> list[PipelineRun]:
        """Return all recorded runs, optionally filtered by label."""
        if label is None:
            return list(self._runs)
        return [r for r in self._runs if r.label == label]

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of all recorded runs by label."""
        result: dict[str, Any] = {}
        for label in RunLabel:
            stats = self.compute_stats(label)
            if stats:
                d = stats.to_dict()
                if stats.count < 10:
                    d["low_sample_warning"] = (
                        f"Only {stats.count} run(s) recorded. "
                        "P50/P70/P90/P100 are not statistically reliable below 10 runs."
                    )
                result[label.value] = d
        return result

    @property
    def total_run_count(self) -> int:
        return len(self._runs)


# Global singleton — shared across the FastAPI application
_global_aggregator = LatencyAggregator()


def get_aggregator() -> LatencyAggregator:
    """Return the global LatencyAggregator instance."""
    return _global_aggregator


__all__ = [
    "LatencyAggregator",
    "LatencyStats",
    "PipelineRun",
    "RunLabel",
    "get_aggregator",
]
