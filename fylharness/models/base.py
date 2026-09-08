"""Abstract model interface.

Models expose three complementary methods:

* :meth:`generate`  -- synchronous one-shot completion used by the evaluation runner.
* :meth:`chat`      -- multi-turn conversational completion returning a full response,
  used by the playground chat UI.
* :meth:`chat_stream` -- a generator yielding incremental text chunks, enabling
  real-time streaming in the browser (SSE).  Default falls back to the non-streaming
  ``chat`` so every backend works in the playground; streaming backends override this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class ModelResponse:
    """Container for one model completion plus bookkeeping metadata."""

    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Provider-reported cache/reasoning sub-counters when available
    # (e.g. ``usage.prompt_tokens_details.cached_tokens`` on OpenAI-compatible APIs).
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str = ""
    latency_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class Model(ABC):
    """Abstract LLM interface consumed by both the runner and the playground."""

    @abstractmethod
    def generate(self, payload: Dict[str, Any]) -> ModelResponse:
        """Synchronously complete ``payload`` and return a :class:`ModelResponse`."""

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> ModelResponse:
        """Complete a multi-turn conversation.

        ``messages`` follows the OpenAI schema: ``[{"role": "user", "content": "..."}]``.
        Default implementation wraps the last user message into a ``generate`` payload.
        """

        last = messages[-1] if messages else {"role": "user", "content": ""}
        return self.generate({"messages": messages, "prompt": last.get("content", "")})

    def chat_stream(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Iterator[str]:
        """Yield incremental text chunks for streaming UIs.

        Default implementation falls back to the non-streaming :meth:`chat` and
        emits the full text as a single chunk.  Backends that support true streaming
        (e.g. OpenAI SSE) override this for token-by-token output.
        """

        resp = self.chat(messages, **kwargs)
        if resp.error:
            yield f"[ERROR] {resp.error}"
        else:
            yield resp.text

    async def agenerate(self, payload: Dict[str, Any]) -> ModelResponse:
        """Asynchronous variant.  Default delegates to :meth:`generate`."""

        return self.generate(payload)
