"""Dataset loading and adapter registry.

This package exposes the adapter registry and re-exports the built-in adapters so that
simply importing :mod:`fylharness.data` registers ``local``, ``huggingface`` and
``inline``.
"""

from .adapters import (
    DataAdapter,
    LocalAdapter,
    HuggingFaceAdapter,
    InlineAdapter,
    register_adapter,
    get_adapter,
    list_adapters,
    adapter_registry,
)

__all__ = [
    "DataAdapter",
    "LocalAdapter",
    "HuggingFaceAdapter",
    "InlineAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "adapter_registry",
]
