"""Model interfaces.

A :class:`Model` exposes a single :meth:`generate` coroutine that turns a request
payload (produced by a task's ``build_prompt``) into a raw string completion.  The
built-in :class:`OpenAICompatibleModel` talks to any OpenAI Chat Completions /
Completions endpoint and is the default for local servers (vLLM, Ollama, TGI, ...) as
well as hosted APIs.  :class:`MockModel` provides a network-free backend for demos and
tests.
"""

from .base import Model, ModelResponse
from .openai_api import OpenAICompatibleModel
from .mock import MockModel
from .registry import build_model, register_model_type

__all__ = [
    "Model",
    "ModelResponse",
    "OpenAICompatibleModel",
    "MockModel",
    "build_model",
    "register_model_type",
]
