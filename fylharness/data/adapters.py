"""Dataset adapters.

An adapter turns a dataset *spec* (the ``dataset`` block of a task config) into a list of
plain sample dicts.  Built-in adapters cover the common cases:

* ``local``  -- JSON / JSONL / CSV files on disk (default, no extra dependencies).
* ``huggingface`` -- the ``datasets`` library, used lazily only when referenced.

Custom formats are supported by implementing :class:`DataAdapter` and registering it
with :func:`register_adapter`.  Adapters receive the config source path so that relative
``path`` fields resolve against the config file rather than the CWD.
"""

from __future__ import annotations

import csv
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from ..config import resolve_path


class DataAdapter(ABC):
    """Base class for dataset adapters."""

    #: Registry key matched against ``dataset.adapter`` in the config.
    name: str = ""

    @abstractmethod
    def load(
        self, spec: Dict[str, Any], reference: Optional[str] = None
    ) -> Union[List[Dict[str, Any]], Iterator[Dict[str, Any]]]:
        """Load the dataset described by ``spec`` and yield/return sample dicts."""


# --------------------------------------------------------------------- registry
class _AdapterRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, type] = {}

    def register(self, cls: type) -> type:
        if not getattr(cls, "name", ""):
            raise ValueError(f"Adapter {cls.__name__} must set a non-empty 'name'.")
        if cls.name in self._items:
            raise ValueError(f"Adapter '{cls.name}' is already registered.")
        self._items[cls.name] = cls
        return cls

    def get(self, name: str) -> "DataAdapter":
        try:
            return self._items[name]()
        except KeyError:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"Unknown dataset adapter '{name}'. Registered adapters: {available}"
            ) from None

    def list(self) -> List[str]:
        return sorted(self._items)


adapter_registry = _AdapterRegistry()


def register_adapter(cls: type) -> type:
    if not isinstance(cls, type) or not issubclass(cls, DataAdapter):
        raise TypeError("register_adapter must decorate a DataAdapter subclass.")
    return adapter_registry.register(cls)


def get_adapter(name: str) -> DataAdapter:
    return adapter_registry.get(name)


def list_adapters() -> List[str]:
    return adapter_registry.list()


# ----------------------------------------------------------------- built-ins
@register_adapter
class LocalAdapter(DataAdapter):
    """Load local JSON / JSONL / CSV files.

    Spec fields
    -----------
    path : str
        File path (relative to the config file if not absolute).
    format : str
        One of ``json`` (a single list), ``jsonl`` (one JSON object per line) or ``csv``.
        When omitted, the file extension is used.
    field_map : dict, optional
        Rename dataset columns to the task's expected keys, e.g. ``{"answer": "reference"}``.
    """

    name = "local"

    def load(self, spec: Dict[str, Any], reference: Optional[str] = None):
        if "path" not in spec:
            raise ValueError("Local dataset spec requires a 'path' field.")
        path = resolve_path(spec["path"], reference)
        fmt = (spec.get("format") or path.suffix.lstrip(".")).lower()
        samples: List[Dict[str, Any]] = []
        if fmt in ("json",):
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # Allow ``{"data": [...]}`` or ``{"examples": [...]}`` wrappers.
                data = next(
                    (v for v in data.values() if isinstance(v, list)), []
                )
            samples = list(data)
        elif fmt in ("jsonl", "ndjson"):
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        elif fmt == "csv":
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                samples = [dict(row) for row in reader]
        else:
            raise ValueError(
                f"Unsupported local dataset format '{fmt}' for {path}. "
                "Use json, jsonl or csv."
            )
        return self._apply_field_map(samples, spec.get("field_map"))

    @staticmethod
    def _apply_field_map(samples: List[Dict[str, Any]], field_map):
        if not field_map:
            return samples
        remapped: List[Dict[str, Any]] = []
        for s in samples:
            new = dict(s)
            for src, dst in field_map.items():
                if src in new:
                    new[dst] = new.pop(src)
            remapped.append(new)
        return remapped


@register_adapter
class HuggingFaceAdapter(DataAdapter):
    """Load a dataset from the Hugging Face Hub via the optional ``datasets`` package.

    Spec fields
    -----------
    name : str
        Hub dataset id (e.g. ``cais/mmlu``).
    subset : str, optional
        Configuration name.
    split : str
        Split to load (``validation``, ``test`` ...).
    limit : int, optional
        Cap the number of samples.
    field_map : dict, optional
        Same semantics as :class:`LocalAdapter`.
    """

    name = "huggingface"

    def load(self, spec: Dict[str, Any], reference: Optional[str] = None):
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'datasets' package is required for the 'huggingface' adapter. "
                "Install it with `pip install datasets`."
            ) from exc

        ds = load_dataset(
            spec["name"],
            spec.get("subset"),
            split=spec.get("split", "test"),
        )
        samples: List[Dict[str, Any]] = [dict(row) for row in ds]
        if spec.get("limit"):
            samples = samples[: int(spec["limit"])]
        return self._apply_field_map(samples, spec.get("field_map"))

    @staticmethod
    def _apply_field_map(samples, field_map):
        if not field_map:
            return samples
        return [
            {field_map.get(k, k): v for k, v in s.items()} for s in samples
        ]


@register_adapter
class InlineAdapter(DataAdapter):
    """Embed a handful of samples directly in the config (useful for demos/tests).

    Spec field
    ----------
    samples : list[dict]
        The samples themselves.
    """

    name = "inline"

    def load(self, spec: Dict[str, Any], reference: Optional[str] = None):
        return list(spec.get("samples", []))
