"""Open-ended text generation task.

The model is asked to complete an instruction-style prompt and the raw completion is
scored verbatim against a gold reference with BLEU/ROUGE (or any other token-overlap
metric the user configures).  This is the simplest task shape and a good template for
custom generation tasks.

Sample fields
-------------
- ``instruction`` : the prompt text (required).
- ``input``       : optional supporting context appended to the instruction.
- ``reference``   : the gold output (required for scoring).
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import Task, register_task


@register_task
class TextGenerationTask(Task):
    """Instruction-following text generation scored by overlap metrics."""

    name = "text_generation"

    def __init__(self, config, harness_config=None) -> None:
        super().__init__(config, harness_config)
        self.instruction_field: str = self.params.get("instruction_field", "instruction")
        self.input_field: str = self.params.get("input_field", "input")
        self.reference_field: str = self.params.get("reference_field", "reference")
        self.template: str = self.params.get(
            "template",
            "{instruction}\n\n{input}",
        )

    def build_prompt(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        instruction = str(sample.get(self.instruction_field, "")).strip()
        ctx = str(sample.get(self.input_field, "")).strip()
        try:
            prompt = self.template.format(instruction=instruction, input=ctx)
        except KeyError:
            # Fall back to a simple concatenation if the template uses unknown keys.
            prompt = f"{instruction}\n\n{ctx}"
        prompt = prompt.strip()
        return {"prompt": prompt, "messages": [{"role": "user", "content": prompt}]}

    def postprocess(self, raw_output: str, sample: Dict[str, Any]) -> Any:
        return (raw_output or "").strip()

    def reference(self, sample: Dict[str, Any]) -> Any:
        return str(sample.get(self.reference_field, "")).strip()
