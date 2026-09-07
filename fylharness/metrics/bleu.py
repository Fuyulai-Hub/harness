"""BLEU score (corpus-level, up to 4-gram) implemented in pure Python.

This is a compact, dependency-free re-implementation of the classic BLEU metric with
brevity penalty and uniform 4-gram weighting.  It is intentionally simpler than the
``sacrebleu`` reference implementation but produces comparable numbers for the common
short-answer QA case where ``reference`` and ``prediction`` are single sentences.

For research-grade reporting we recommend plugging in ``sacrebleu`` via the custom
metric hook (see :func:`register_metric`).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence

from .base import register_metric, SampleRecord


_TOKEN_RE = re.compile(r"\w+")
_MAX_N = 4


def _tokenise(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _modified_precision(pred: Sequence[str], refs: Sequence[Sequence[str]], n: int) -> float:
    pred_grams = Counter(tuple(pred[i : i + n]) for i in range(len(pred) - n + 1))
    if not pred_grams:
        return 0.0
    max_ref_grams: Counter = Counter()
    for ref in refs:
        ref_grams = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        for gram, count in ref_grams.items():
            max_ref_grams[gram] = max(max_ref_grams[gram], count)
    clipped = sum(min(c, max_ref_grams.get(g, 0)) for g, c in pred_grams.items())
    total = sum(pred_grams.values())
    return clipped / total if total else 0.0


@register_metric(
    "bleu",
    needs=("prediction", "reference"),
    description="Corpus-level BLEU-4 with brevity penalty (pure Python).",
)
def bleu(samples: List[SampleRecord]) -> float:
    if not samples:
        return 0.0
    log_sum = 0.0
    any_zero = False
    pred_len_total = 0
    ref_len_total = 0
    for s in samples:
        pred = _tokenise(str(s.get("prediction", "")))
        ref = _tokenise(str(s.get("reference", "")))
        refs = [ref]
        pred_len_total += len(pred)
        ref_len_total += len(ref)
        ps: List[float] = []
        for n in range(1, _MAX_N + 1):
            p = _modified_precision(pred, refs, n)
            if p == 0.0:
                any_zero = True
                break
            ps.append(p)
        if not any_zero:
            # Geometric mean of n-gram precisions (uniform weights).
            log_sum += math.log(ps[-1]) if len(ps) == 1 else sum(
                math.log(p) for p in ps
            ) / len(ps)
    if any_zero and pred_len_total:
        # Standard BLEU convention: any zero n-gram precision => 0.
        return 0.0
    if pred_len_total == 0:
        return 0.0
    bp = (
        1.0
        if pred_len_total > ref_len_total
        else math.exp(1 - ref_len_total / pred_len_total)
    )
    geo_mean = math.exp(log_sum / len(samples)) if samples else 0.0
    return bp * geo_mean
