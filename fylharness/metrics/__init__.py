"""Metric registry and built-in metrics.

A metric is a small, stateless callable wrapped by the :class:`Metric` dataclass.  Each
metric receives the list of per-sample ``(prediction, reference)`` records produced by a
task and returns either a single float or a dict of named floats.  Built-in metrics
cover the common cases (accuracy, F1, BLEU, ROUGE) with pure-Python implementations so
that no heavy native dependency is required on Windows.
"""

from __future__ import annotations

from .base import (
    Metric,
    MetricResult,
    register_metric,
    get_metric,
    list_metrics,
    metric_registry,
)
from .accuracy import accuracy
from .f1 import f1
from .bleu import bleu
from .rouge import rouge

__all__ = [
    "Metric",
    "MetricResult",
    "register_metric",
    "get_metric",
    "list_metrics",
    "metric_registry",
    "accuracy",
    "f1",
    "bleu",
    "rouge",
]
