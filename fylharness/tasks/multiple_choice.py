"""Multiple-choice task (MMLU / ARC / HellaSwag-style).

Each sample is expected to expose a ``question`` (the stem), a ``choices`` list and the
index of the correct choice under ``answer`` (0-based) or ``answer_letter`` (``"A"``,
``"B"`` ...).  The prompt is a few-shot template ending with ``Answer:``; the model's
response is parsed for the first ``A``-``Z`` letter and compared to the gold letter.

This mirrors the canonical MMLU prompt shape used by DeepSeek-LLM and
lm-evaluation-harness while staying self-contained.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import Task, register_task


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ANSWER_RE = re.compile(r"\b([A-Z])\b")


def _format_few_shot(sample: Dict[str, Any]) -> str:
    """Render one MCQ item *without* revealing the answer (used for both context and target)."""

    choices: List[str] = sample.get("choices", [])
    question = sample.get("question", "")
    lines = [question.strip()] if question else []
    for idx, choice in enumerate(choices):
        lines.append(f"{_LETTERS[idx]}. {choice}")
    return "\n".join(lines)


@register_task
class MultipleChoiceTask(Task):
    """Few-shot multiple-choice question answering."""

    name = "multiple_choice"

    def __init__(self, config, harness_config=None) -> None:
        super().__init__(config, harness_config)
        # Number of few-shot exemplars drawn from the dataset's ``few_shot`` field, or
        # ``params.num_few_shot``.  Exemplars come from the same dataset by default.
        self.num_few_shot: int = int(self.params.get("num_few_shot", 0))
        # Custom instruction header prepended to the prompt.
        self.instruction: str = self.params.get(
            "instruction",
            "The following are multiple choice questions. Answer with the letter "
            "(A, B, C, D) of the correct option.",
        )
        # Allow callers to pin specific few-shot indices instead of the first N.
        self.few_shot_indices: Optional[List[int]] = self.params.get(
            "few_shot_indices"
        )

    # ------------------------------------------------------------------ dataset
    def load_dataset(self) -> List[Dict[str, Any]]:
        samples = super().load_dataset()
        # Separate few-shot exemplars (if provided) from evaluation samples.
        few_shot = []
        if self.few_shot_indices is not None:
            few_shot = [samples[i] for i in self.few_shot_indices if 0 <= i < len(samples)]
            eval_idx = [i for i in range(len(samples)) if i not in set(self.few_shot_indices)]
            samples = [samples[i] for i in eval_idx]
        elif self.num_few_shot and self.num_few_shot < len(samples):
            few_shot = samples[: self.num_few_shot]
            samples = samples[self.num_few_shot :]
        # Stash exemplars on the instance for prompt building.
        self._few_shot = few_shot
        return samples

    # ------------------------------------------------------------------- prompt
    def build_prompt(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        blocks: List[str] = [self.instruction, ""]
        for ex in getattr(self, "_few_shot", []):
            blocks.append(_format_few_shot(ex))
            blocks.append(f"Answer: {self._gold_letter(ex)}\n")
        blocks.append(_format_few_shot(sample))
        blocks.append("Answer:")
        prompt = "\n".join(blocks)
        return {"prompt": prompt, "messages": [{"role": "user", "content": prompt}]}

    # --------------------------------------------------------------- postprocess
    def postprocess(self, raw_output: str, sample: Dict[str, Any]) -> Any:
        if not raw_output:
            return ""
        # Prefer the very first standalone letter encountered.
        match = _ANSWER_RE.search(raw_output.strip())
        if match:
            return match.group(1)
        # Fallback: take the first alphabetic character if it is a valid choice index.
        for ch in raw_output:
            up = ch.upper()
            if up in _LETTERS[: len(sample.get("choices", []))]:
                return up
        return ""

    # ---------------------------------------------------------------- reference
    def reference(self, sample: Dict[str, Any]) -> Any:
        return self._gold_letter(sample)

    @staticmethod
    def _gold_letter(sample: Dict[str, Any]) -> str:
        if "answer_letter" in sample:
            return str(sample["answer_letter"]).strip().upper()
        if "answer" in sample:
            ans = sample["answer"]
            if isinstance(ans, str) and ans.strip().upper() in _LETTERS:
                return ans.strip().upper()
            try:
                idx = int(ans)
                return _LETTERS[idx]
            except (TypeError, ValueError):
                return str(ans).strip().upper()
        return ""
