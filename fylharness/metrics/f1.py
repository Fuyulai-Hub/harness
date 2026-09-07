"""Macro-averaged F1 metric.

Treats ``reference`` as the set of class labels.  For each unique class the F1 score is
computed from token-overlap precision/recall (tokenisation is a simple whitespace +
punctuation split, sufficient for short QA answers) and the per-class scores are macro
averaged.  For pure classification tasks where ``prediction``/``reference`` are single
labels, this collapses to the standard classification F1.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Set

from .base import register_metric, SampleRecord


_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> Set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _f1_for_pair(pred_tokens: Set[str], ref_tokens: Set[str]) -> float:
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


@register_metric(
    "f1",
    needs=("prediction", "reference"),
    description="Macro-averaged token-overlap F1 across all samples.",
)
def f1(samples: List[SampleRecord]) -> float:
    if not samples:
        return 0.0
    scores: List[float] = []
    for s in samples:
        pred = _tokens(str(s.get("prediction", "")))
        ref = _tokens(str(s.get("reference", "")))
        scores.append(_f1_for_pair(pred, ref))
    return sum(scores) / len(scores)


@register_metric(
    "f1_macro_by_label",
    needs=("prediction", "reference"),
    description="Macro F1 averaged by unique reference label (classification use case).",
)
def f1_macro_by_label(samples: List[SampleRecord]) -> float:
    """Macro F1 averaged by unique reference label (classification use case)."""

    if not samples:
        return 0.0
    groups: dict = defaultdict(list)
    for s in samples:
        label = str(s.get("reference", "")).strip().lower()
        groups[label].append(s)
    per_label_f1: List[float] = []
    for label, group in groups.items():
        tp = sum(
            1 for s in group if str(s.get("prediction", "")).strip().lower() == label
        )
        fp = 0
        fn = 0
        # Count false positives across other groups
        for other_label, other_group in groups.items():
            if other_label == label:
                continue
            fp += sum(
                1
                for s in other_group
                if str(s.get("prediction", "")).strip().lower() == label
            )
        # False negatives = total for this label - correctly predicted
        fn = len(group) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_label_f1.append(
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
    return sum(per_label_f1) / len(per_label_f1) if per_label_f1 else 0.0
