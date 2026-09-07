"""Result dataclasses shared by the runner and the reporter.

These live in a dedicated module to avoid the circular import that would otherwise arise
between :mod:`fylharness.runner` (which produces results) and :mod:`fylharness.reporting`
(which serialises them).  Both modules import from here, keeping the dependency graph
one-directional: ``runner -> results <- reporting``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SampleResult:
    """One per-sample outcome, serialised into the JSONL checkpoint and JSON report."""

    index: int
    task: str
    model: str
    prediction: Any = None
    reference: Any = None
    raw: str = ""
    error: Optional[str] = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    sample: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Aggregate scores for one ``(model, task)`` pair."""

    model: str
    task: str
    metrics: Optional[Dict[str, Any]] = None
    num_samples: int = 0
    num_errors: int = 0
    total_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    samples: List[SampleResult] = field(default_factory=list)

    def as_dict(self, include_samples: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "model": self.model,
            "task": self.task,
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "num_errors": self.num_errors,
            "total_latency_ms": self.total_latency_ms,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
        if include_samples:
            out["samples"] = [
                {k: v for k, v in s.__dict__.items()} for s in self.samples
            ]
        return out
