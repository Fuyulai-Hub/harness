"""FylHarness: a lightweight, Windows-friendly evaluation harness for OpenAI-compatible LLMs.

FylHarness provides a modular, out-of-the-box toolkit for benchmarking large language
models that expose an OpenAI-style Chat Completions / Completions API. It is inspired by
the design philosophy of DeepSeek-LLM's evaluation harness and lm-evaluation-harness,
but intentionally re-implemented from scratch to stay small, dependency-light and fully
portable across Windows 10/11 and POSIX systems.

Public surface
--------------
``run``        -- one-shot Python API to execute a configuration and write reports.
``register_*``-- decorators used to plug in custom tasks, metrics and data adapters.
``__version__`` -- package version string.
"""

from __future__ import annotations

from .config import HarnessConfig, load_config
from .runner import Runner
from .runner import run as run
from .tasks.base import Task, register_task, get_task, list_tasks
from .metrics.base import Metric, register_metric, get_metric, list_metrics
from .data.adapters import DataAdapter, register_adapter, get_adapter, list_adapters
from .web import serve_dashboard, open_results

__version__ = "0.1.0"


def __getattr__(name: str):  # noqa: N807 - lazy import for optional Flask dep
    """Lazily resolve ``HarnessServer`` so core commands never require Flask."""

    if name == "HarnessServer":
        from .server import HarnessServer
        return HarnessServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "HarnessConfig",
    "load_config",
    "Runner",
    "run",
    "Task",
    "register_task",
    "get_task",
    "list_tasks",
    "Metric",
    "register_metric",
    "get_metric",
    "list_metrics",
    "DataAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "serve_dashboard",
    "open_results",
    "HarnessServer",
]
