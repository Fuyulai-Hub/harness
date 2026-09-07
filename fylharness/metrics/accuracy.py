"""Accuracy metric for classification / multiple-choice tasks.

Reads ``prediction`` (the model's chosen label or index) and ``reference`` (the gold
label) from each sample record and returns the fraction of exact matches.  When the task
records ``prediction`` as a 0-based choice index (multiple-choice), the comparison is
still a direct equality, so no special handling is required here.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import register_metric, SampleRecord


def _normalise(value: Any) -> str:
    return str(value).strip().lower()


@register_metric(
    "accuracy",
    needs=("prediction", "reference"),
    description="Fraction of samples whose prediction exactly matches the reference.",
)
def accuracy(samples: List[SampleRecord]) -> float:
    if not samples:
        return 0.0
    correct = 0
    for s in samples:
        if _normalise(s.get("prediction")) == _normalise(s.get("reference")):
            correct += 1
    return correct / len(samples)
