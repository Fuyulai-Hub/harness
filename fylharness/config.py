"""Configuration management for FylHarness.

Configuration is centralised in a single YAML/JSON document that describes the models
to evaluate, the tasks to run, the datasets they consume, concurrency options and where
to write results.  Sensitive values (API keys) are *never* hard-coded: they are read from
environment variables, either explicitly via the ``env`` field or implicitly by naming
the variable ``<UPPER_PREFIX>_<KEY>``.

The :class:`HarnessConfig` dataclass is the canonical in-memory representation consumed
by every other module; it is constructed through :func:`load_config` which performs
validation, environment-variable interpolation and default propagation.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # PyYAML is the only hard third-party dependency for config.
    import yaml
except ImportError as exc:  # pragma: no cover - guarded by requirements.txt
    raise ImportError(
        "PyYAML is required to load YAML configuration. Install it with "
        "`pip install pyyaml`."
    ) from exc


# A regex matching ``${VAR}`` and ``${VAR:-default}`` style placeholders, mirroring shell
# semantics so that secrets never need to be committed to the config file.
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


@dataclass
class ModelConfig:
    """Connection parameters for a single OpenAI-compatible endpoint.

    ``api_key_env`` takes priority over ``api_key`` so that the key can be supplied via
    the environment without touching the config file.  At least one of the two must
    resolve to a non-empty value before the runner actually issues a request.
    """

    name: str
    #: Implementation selector. ``openai`` (default) uses the HTTP client; ``mock``
    #: returns deterministic canned responses for offline demos and tests.
    type: str = "openai"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    model: str = ""
    # Empty ``model`` falls back to ``name`` so a barebones config still works.
    chat: bool = True
    max_tokens: int = 256
    temperature: float = 0.0
    timeout: float = 120.0
    max_retries: int = 5
    retry_backoff: float = 2.0
    # Concurrency budget reserved for this model.  Honoured by :class:`Runner`.
    concurrency: int = 4
    # Free-form extra kwargs forwarded verbatim to the API call body.
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolved_api_key(self) -> str:
        """Resolve the API key preferring the environment variable over the literal."""

        env_val = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        return env_val or self.api_key


@dataclass
class TaskConfig:
    """Declarative description of one evaluation task instance.

    ``type`` is the registry key of a :class:`~fylharness.tasks.base.Task` subclass.
    The remaining fields are forwarded as ``**params`` to the task constructor, giving
    tasks full freedom to declare whatever knobs they need without bloating the
    dataclass.
    """

    name: str
    type: str
    dataset: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    # When set, the runner only considers samples whose index falls in this range.
    sample_range: Optional[str] = None


@dataclass
class RunConfig:
    """Global execution options shared by every task/model pair."""

    output_dir: str = "outputs"
    log_level: str = "INFO"
    # ``thread`` (default) or ``async``.  Both are Windows-safe.
    executor: str = "thread"
    # Hard upper bound on total in-flight requests across the whole run.
    max_workers: int = 8
    # When true the runner records per-sample predictions in the JSON dump.
    save_predictions: bool = True
    # Resume support: completed sample indices are persisted in ``output_dir``.
    resume: bool = True
    seed: int = 42
    tags: List[str] = field(default_factory=list)


@dataclass
class HarnessConfig:
    """Top-level configuration object consumed by :class:`Runner`."""

    models: List[ModelConfig] = field(default_factory=list)
    tasks: List[TaskConfig] = field(default_factory=list)
    run: RunConfig = field(default_factory=RunConfig)
    # Path of the file this config was loaded from; used for resolving relative paths.
    source: Optional[str] = None

    # ------------------------------------------------------------------ helpers
    def output_path(self) -> Path:
        return Path(self.run.output_dir)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "HarnessConfig":
        """Build a validated config from a plain dictionary (already interpolated)."""

        models = [
            ModelConfig(
                name=_req(m, "name", "models"),
                type=m.get("type", "openai"),
                base_url=m.get("base_url", "http://127.0.0.1:8000/v1"),
                api_key=m.get("api_key", ""),
                api_key_env=m.get("api_key_env", "OPENAI_API_KEY"),
                model=m.get("model") or m.get("name", ""),
                chat=m.get("chat", True),
                max_tokens=m.get("max_tokens", 256),
                temperature=m.get("temperature", 0.0),
                timeout=m.get("timeout", 120.0),
                max_retries=m.get("max_retries", 5),
                retry_backoff=m.get("retry_backoff", 2.0),
                concurrency=m.get("concurrency", 4),
                extra=m.get("extra", {}),
            )
            for m in raw.get("models", [])
        ]
        tasks = [
            TaskConfig(
                name=_req(t, "name", "tasks"),
                type=_req(t, "type", "tasks"),
                dataset=t.get("dataset", {}),
                params=t.get("params", {}),
                metrics=t.get("metrics", []),
                limit=t.get("limit"),
                sample_range=t.get("sample_range"),
            )
            for t in raw.get("tasks", [])
        ]
        run_raw = raw.get("run", {})
        run = RunConfig(
            output_dir=run_raw.get("output_dir", "outputs"),
            log_level=run_raw.get("log_level", "INFO"),
            executor=run_raw.get("executor", "thread"),
            max_workers=run_raw.get("max_workers", 8),
            save_predictions=run_raw.get("save_predictions", True),
            resume=run_raw.get("resume", True),
            seed=run_raw.get("seed", 42),
            tags=run_raw.get("tags", []),
        )
        return cls(models=models, tasks=tasks, run=run)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _req(d: Dict[str, Any], key: str, section: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required field '{key}' in '{section}' entry: {d}")
    return d[key]


# ---------------------------------------------------------------- file loading
def _interpolate_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` / ``${VAR:-default}`` placeholders with env values.

    Missing variables without a default resolve to an empty string so that downstream
    validation (e.g. an empty ``api_key``) surfaces a clear error rather than a stray
    ``${...}`` token.
    """

    if isinstance(value, str):
        def _sub(match: "re.Match[str]") -> str:
            var, default = match.group(1), match.group(2)
            return os.environ.get(var, default if default is not None else "")

        return _PLACEHOLDER_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def load_config(path: str | os.PathLike) -> HarnessConfig:
    """Load a YAML or JSON config file and return a validated :class:`HarnessConfig`.

    Environment-variable interpolation runs *before* construction, so ``${...}``
    placeholders may appear anywhere in the document.  Relative paths in the config are
    resolved against the config file's directory later (each consumer receives the raw
    string and is responsible for resolution via :func:`resolve_path`).
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        raw = yaml.safe_load(text) or {}
    elif p.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        # Best-effort: try YAML (a superset of JSON) then fall back to JSON.
        try:
            raw = yaml.safe_load(text) or {}
        except yaml.YAMLError:
            raw = json.loads(text)
    raw = _interpolate_env(raw)
    cfg = HarnessConfig.from_dict(raw)
    cfg.source = str(p.resolve())
    return cfg


def resolve_path(config_path: Optional[str], reference: Optional[str] = None) -> Path:
    """Resolve a possibly-relative path against the config file directory.

    ``reference`` is typically the config file location; when omitted the path is
    treated as already absolute or CWD-relative.
    """

    p = Path(config_path or "")
    if not p.is_absolute() and reference:
        base = Path(reference).resolve().parent
        return (base / p).resolve()
    return p
