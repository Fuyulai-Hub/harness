"""Task abstraction and global registry.

A :class:`Task` is the unit of evaluation: it knows how to turn one dataset row into a
model prompt, how to turn a raw model response into a *prediction*, and which metrics
should be computed over the resulting ``(prediction, reference)`` pairs.  Tasks are kept
deliberately thin -- dataset loading, model I/O and metric computation all live in
dedicated modules -- so that a new task is usually ~40 lines of code.

The registry mirrors :mod:`fylharness.metrics.base`: tasks are referenced by name from
the config file and instantiated by the runner with the task-specific ``params``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import TaskConfig, resolve_path
from ..data.adapters import DataAdapter, get_adapter
from ..metrics.base import Metric, get_metric, MetricValue, SampleRecord


class Task(ABC):
    """Abstract base class for all evaluation tasks.

    Subclasses implement :meth:`build_prompt`, :meth:`postprocess` and
    :meth:`reference`.  :meth:`load_dataset` delegates to a :class:`DataAdapter` so that
    a single task can consume data from Hugging Face, a local JSON file or a custom
    adapter without changing its own code.
    """

    #: Registry key, set by :func:`register_task`.  Subclasses MUST set this.
    name: str = ""

    def __init__(self, config: TaskConfig, harness_config=None) -> None:
        self.config = config
        self.harness_config = harness_config
        # Resolve the dataset adapter once; the adapter returns a list of plain dicts.
        ds_spec = config.dataset or {}
        adapter_name = ds_spec.get("adapter", "local")
        self.adapter: DataAdapter = get_adapter(adapter_name)
        self.dataset_spec = ds_spec
        self.params = config.params or {}

    # ------------------------------------------------------------------ dataset
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and pre-process the dataset, returning a list of sample dicts.

        The adapter handles the actual I/O; this hook lets a task perform extra
        normalisation (e.g. shuffling with the global seed) if needed.
        """

        reference = getattr(self.harness_config, "source", None)
        samples = self.adapter.load(self.dataset_spec, reference=reference)
        if isinstance(samples, list):
            return samples
        # Adapters may yield generators; materialise once for index-based resume.
        return list(samples)

    # ------------------------------------------------------------------- prompt
    @abstractmethod
    def build_prompt(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Turn one sample into a model request payload.

        Returns a dict with keys understood by the :class:`~fylharness.models.base.Model`
        interface: ``prompt`` (str, for completions) and/or ``messages`` (list, for
        chat).  This lets a single task target both API styles.
        """

    # --------------------------------------------------------------- postprocess
    @abstractmethod
    def postprocess(self, raw_output: str, sample: Dict[str, Any]) -> Any:
        """Convert a raw model string into a *prediction* comparable to the reference."""

    @abstractmethod
    def reference(self, sample: Dict[str, Any]) -> Any:
        """Return the gold answer for ``sample`` used by metrics."""

    # ----------------------------------------------------------------- evaluate
    def make_record(
        self,
        sample: Dict[str, Any],
        prediction: Any,
        raw_output: str,
    ) -> SampleRecord:
        """Build the metric input record.  Tasks may override to attach extra context."""

        return {
            "prediction": prediction,
            "reference": self.reference(sample),
            "raw": raw_output,
            **{k: v for k, v in sample.items() if k != "reference"},
        }

    def evaluate(self, records: List[SampleRecord]) -> Dict[str, MetricValue]:
        """Run every configured metric over ``records`` and return ``{name: value}``."""

        results: Dict[str, MetricValue] = {}
        for metric_name in self.config.metrics:
            metric: Metric = get_metric(metric_name)
            self._check_metric_needs(metric, records)
            results[metric_name] = metric(records)
        return results

    @staticmethod
    def _check_metric_needs(metric: Metric, records: List[SampleRecord]) -> None:
        if not records:
            return
        sample = records[0]
        missing = [k for k in metric.needs if k not in sample]
        if missing:
            raise ValueError(
                f"Metric '{metric.name}' requires sample keys {missing} but the "
                f"task produced records with keys {list(sample)}."
            )


# --------------------------------------------------------------------- registry
class _TaskRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, type] = {}

    def register(self, cls: type) -> type:
        name = getattr(cls, "name", "")
        if not name:
            raise ValueError(
                f"Task {cls.__name__} must set a non-empty 'name' class attribute."
            )
        if name in self._items:
            raise ValueError(f"Task '{name}' is already registered.")
        self._items[name] = cls
        return cls

    def get(self, name: str) -> type:
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"Unknown task type '{name}'. Registered tasks: {available}"
            ) from None

    def list(self) -> List[str]:
        return sorted(self._items)


task_registry = _TaskRegistry()


def register_task(cls: type) -> type:
    """Class decorator that registers a :class:`Task` subclass under ``cls.name``."""

    if not isinstance(cls, type) or not issubclass(cls, Task):
        raise TypeError("register_task must decorate a Task subclass.")
    return task_registry.register(cls)


def get_task(name: str) -> type:
    return task_registry.get(name)


def list_tasks() -> List[str]:
    return task_registry.list()
