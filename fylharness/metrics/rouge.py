"""ROUGE-1 / ROUGE-2 / ROUGE-L (F-measure) implemented in pure Python.

Returns a dict with three keys so a single metric registration covers the three
commonly reported ROUGE variants.  The implementation uses LCS for ROUGE-L and
overlapping n-gram counts for ROUGE-N, with F-measure weighted equally between
precision and recall.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Sequence

from .base import register_metric, SampleRecord


_TOKEN_RE = re.compile(r"\w+")


def _tokenise(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f_score(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision + recall == 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def _rouge_n(pred: Sequence[str], ref: Sequence[str], n: int) -> float:
    pred_grams = _ngrams(pred, n)
    ref_grams = _ngrams(ref, n)
    overlap = sum((pred_grams & ref_grams).values())
    if not pred_grams or not ref_grams:
        return 0.0
    precision = overlap / sum(pred_grams.values())
    recall = overlap / sum(ref_grams.values())
    return _f_score(precision, recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    # Standard DP over two token sequences.
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _rouge_l(pred: Sequence[str], ref: Sequence[str]) -> float:
    if not pred or not ref:
        return 0.0
    lcs = _lcs_length(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return _f_score(precision, recall)


@register_metric(
    "rouge",
    needs=("prediction", "reference"),
    description="ROUGE-1/2/L F-measure averaged over samples (pure Python).",
)
def rouge(samples: List[SampleRecord]) -> Dict[str, float]:
    if not samples:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    r1, r2, rl = 0.0, 0.0, 0.0
    for s in samples:
        pred = _tokenise(str(s.get("prediction", "")))
        ref = _tokenise(str(s.get("reference", "")))
        r1 += _rouge_n(pred, ref, 1)
        r2 += _rouge_n(pred, ref, 2)
        rl += _rouge_l(pred, ref)
    n = len(samples)
    return {"rouge_1": r1 / n, "rouge_2": r2 / n, "rouge_l": rl / n}
