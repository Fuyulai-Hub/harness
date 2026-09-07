"""Model factory.

Builds the concrete :class:`Model` implementation for a :class:`~fylharness.config.ModelConfig`
based on its ``type`` field.  This is the single place the runner looks up the model
class, so adding a new backend (e.g. a local transformers pipeline) is a matter of
registering it here.
"""

from __future__ import annotations

from typing import Dict, Type

from ..config import ModelConfig
from .base import Model
from .openai_api import OpenAICompatibleModel
from .mock import MockModel


_MODEL_REGISTRY: Dict[str, Type[Model]] = {
    "openai": OpenAICompatibleModel,
    "mock": MockModel,
}


def register_model_type(name: str) -> "object":
    """Class decorator to register a new model backend under ``name``."""

    def _wrap(cls: Type[Model]) -> Type[Model]:
        _MODEL_REGISTRY[name] = cls
        return cls

    return _wrap


def build_model(config: ModelConfig) -> Model:
    cls = _MODEL_REGISTRY.get(config.type)
    if cls is None:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown model type '{config.type}'. Registered types: {available}"
        )
    return cls(config)


__all__ = ["build_model", "register_model_type", "OpenAICompatibleModel", "MockModel"]
