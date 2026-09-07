"""Metric abstraction and global registry.

Metrics are deliberately decoupled from tasks: a task declares *which* metrics it wants
by name and the registry resolves the implementation at evaluation time.  This makes it
trivial to share a metric (e.g. ``accuracy``) across very different task shapes -- the
task is responsible for producing comparable ``(prediction, reference)`` pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# A metric receives the full list of per-sample records and returns either a single
# score or a mapping of sub-metric name -> score.  Each record carries the task's
# post-processed prediction, the gold reference, and any extra context the task chose
# to attach (e.g. multiple-choice logprobs).
SampleRecord = Dict[str, Any]
MetricValue = Union[float, Dict[str, float]]
MetricFunc = Callable[[List[SampleRecord]], MetricValue]


@dataclass
class Metric:
    """Declarative description of a registered metric.

    ``func`` is the concrete implementation.  ``needs`` advertises which optional keys
    a metric reads from each sample record so that a task can fail fast when a required
    piece of information is missing.
    """

    name: str
    func: MetricFunc
    needs: Tuple[str, ...] = ()
    description: str = ""

    def __call__(self, samples: List[SampleRecord]) -> MetricValue:
        return self.func(samples)


MetricResult = MetricValue


# --------------------------------------------------------------------- registry
class _MetricRegistry:
    """A tiny dict-backed registry with friendly error messages."""

    def __init__(self) -> None:
        self._items: Dict[str, Metric] = {}

    def register(self, metric: Metric) -> Metric:
        if metric.name in self._items:
            raise ValueError(
                f"Metric '{metric.name}' is already registered. "
                "Use a different name or unregister first."
            )
        self._items[metric.name] = metric
        return metric

    def get(self, name: str) -> Metric:
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"Unknown metric '{name}'. Registered metrics: {available}"
            ) from None

    def list(self) -> List[str]:
        return sorted(self._items)

    def contains(self, name: str) -> bool:
        return name in self._items


metric_registry = _MetricRegistry()


def register_metric(
    name: Optional[str] = None,
    needs: Tuple[str, ...] = (),
    description: str = "",
) -> Callable[[MetricFunc], Metric]:
    """Decorator that turns a plain function into a registered :class:`Metric`.

    Example
    -------
    >>> @register_metric("exact_match")
    ... def exact_match(samples):
    ...     correct = sum(1 for s in samples if s["prediction"] == s["reference"])
    ...     return correct / max(1, len(samples))
    """

    def _wrap(func: MetricFunc) -> Metric:
        metric = Metric(
            name=name or func.__name__,
            func=func,
            needs=tuple(needs),
            description=description or (func.__doc__ or "").strip().splitlines()[0]
            if func.__doc__
            else "",
        )
        return metric_registry.register(metric)

    return _wrap


def get_metric(name: str) -> Metric:
    return metric_registry.get(name)


def list_metrics() -> List[str]:
    return metric_registry.list()
