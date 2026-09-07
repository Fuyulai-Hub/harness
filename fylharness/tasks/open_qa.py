"""Short-form open-ended QA (TriviaQA / NaturalQuestions-style).

The model produces a free-form answer to a single question.  The prediction is
normalised (lower-cased, articles and punctuation stripped) before being compared with
the reference for exact match, and token-overlap F1 is reported as well.  Reference may
be either a single string or a list of acceptable answers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .base import Task, register_task


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]")


def _normalise(text: str) -> str:
    text = _ARTICLES.sub(" ", text.lower())
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


@register_task
class OpenQATask(Task):
    """Open-ended question answering with exact-match + F1 scoring."""

    name = "open_qa"

    def __init__(self, config, harness_config=None) -> None:
        super().__init__(config, harness_config)
        self.question_field: str = self.params.get("question_field", "question")
        self.reference_field: str = self.params.get("reference_field", "answer")

    def build_prompt(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = str(sample.get(self.question_field, "")).strip()
        prompt = f"Question: {question}\nAnswer:"
        return {"prompt": prompt, "messages": [{"role": "user", "content": prompt}]}

    def postprocess(self, raw_output: str, sample: Dict[str, Any]) -> Any:
        if not raw_output:
            return ""
        # Take the first line and trim trailing periods -- common model behaviour.
        first_line = raw_output.strip().splitlines()[0].strip()
        return first_line.rstrip(".").strip()

    def reference(self, sample: Dict[str, Any]) -> Any:
        ref = sample.get(self.reference_field, "")
        if isinstance(ref, list):
            return [_normalise(r) for r in ref]
        return _normalise(str(ref))
