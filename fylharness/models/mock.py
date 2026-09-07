"""A deterministic mock model for offline demos and tests.

The :class:`MockModel` never touches the network.  It inspects the request payload and
returns a canned, rule-based completion so that the whole harness (concurrency,
checkpointing, metrics, reporting) can be exercised without a running LLM server.

Behaviour is deliberately simple and predictable:

* For multiple-choice prompts (ending with ``Answer:``) it returns the letter of the
  first choice listed in the prompt, i.e. it always picks ``A``.  Override with the
  ``answer_letter`` param to simulate a specific accuracy.
* For open-ended prompts it echoes a short snippet of the instruction so that BLEU/ROUGE
  are non-trivial.

This makes it ideal for CI smoke tests on Windows where no GPU/server is available.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, Iterator, List, Optional

from ..config import ModelConfig
from .base import Model, ModelResponse


class MockModel(Model):
    """A network-free, deterministic model used for demos and tests."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        # When set, MCQ prompts return this fixed letter (useful to simulate accuracy).
        self.fixed_letter: Optional[str] = config.extra.get("answer_letter")
        # Simulated per-request latency in ms (drawn from a small range).
        self.latency_ms = float(config.extra.get("latency_ms", 5.0))
        self.rng = random.Random(config.extra.get("seed", 0))

    def generate(self, payload: Dict[str, Any]) -> ModelResponse:
        start = time.perf_counter()
        content = self._respond(payload)
        latency = (time.perf_counter() - start) * 1000 + self.latency_ms
        # Rough token estimate so token accounting paths are exercised.
        ptoks = max(1, len(str(payload.get("prompt", ""))) // 4)
        ctoks = max(1, len(content) // 4)
        return ModelResponse(
            text=content,
            prompt_tokens=ptoks,
            completion_tokens=ctoks,
            finish_reason="stop",
            latency_ms=latency,
            raw={"model": self.config.name, "mock": True},
        )

    async def agenerate(self, payload: Dict[str, Any]) -> ModelResponse:
        return self.generate(payload)

    # ------------------------------------------------------------- chat
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> ModelResponse:
        """Conversational mock: echoes the user's text with a canned reply."""

        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        content = (
            f"[MockModel:{self.config.name}] You said: \"{last_user[:200]}\". "
            "I'm a deterministic offline model for testing the harness UI. "
            "Point the config at a real OpenAI-compatible endpoint to get actual completions."
        )
        return ModelResponse(
            text=content,
            prompt_tokens=max(1, len(last_user) // 4),
            completion_tokens=max(1, len(content) // 4),
            finish_reason="stop",
            latency_ms=self.latency_ms + 2.0,
            raw={"mock": True, "model": self.config.name},
        )

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs: Any) -> Iterator[str]:
        """Yield the canned reply word-by-word so the UI streaming path is exercised."""

        resp = self.chat(messages, **kwargs)
        if resp.error:
            yield f"[ERROR] {resp.error}"
            return
        for word in resp.text.split():
            yield word + " "

    # ------------------------------------------------------------- internals
    def _respond(self, payload: Dict[str, Any]) -> str:
        prompt = payload.get("prompt", "")
        # Detect the MCQ shape: a trailing "Answer:" after labelled options.
        if prompt.rstrip().endswith("Answer:") and ". " in prompt:
            if self.fixed_letter:
                return self.fixed_letter
            # Default mock "model" picks A (option 0) -- yields ~25% on 4-choice MCQ.
            return "A"
        # For open-ended prompts, echo the first sentence-ish chunk of the input so
        # overlap metrics are non-zero against references that share wording.
        body = prompt.strip()
        # Trim a leading "Question:" / "Instruction:" marker if present.
        for marker in ("Question:", "Answer:", "Instruction:"):
            if body.startswith(marker):
                body = body[len(marker):].strip()
        snippet = body.split("\n")[0][:32].strip()
        return snippet or "unknown"
