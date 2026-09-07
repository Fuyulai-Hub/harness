"""Evaluation orchestration: concurrency, checkpointing and result collection.

The :class:`Runner` is the only object a caller needs to drive an evaluation.  Given a
:class:`~fylharness.config.HarnessConfig` it:

1. enumerates every ``(model, task)`` pair,
2. loads the task dataset and applies ``limit`` / ``sample_range`` / resume state,
3. runs inference concurrently (``ThreadPoolExecutor`` by default, ``asyncio`` optional),
4. streams per-sample results to a JSONL checkpoint so an interrupted run can resume,
5. computes metrics, and
6. hands the aggregated results to :mod:`fylharness.reporting`.

Windows compatibility
---------------------
``ThreadPoolExecutor`` is the default and the safest choice on Windows: it avoids the
``spawn`` cost of multiprocessing and the global interpreter lock is irrelevant for
HTTP-bound work.  The ``async`` executor uses ``asyncio`` with the default
``ProactorEventLoop`` on Windows 3.9+, which is fully supported.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import HarnessConfig, ModelConfig, TaskConfig
from .models.base import Model, ModelResponse
from .models.registry import build_model
from .reporting import ReportWriter
from .results import SampleResult, TaskResult
from .tasks.base import Task, get_task

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- runner
class Runner:
    """Drive an evaluation described by a :class:`HarnessConfig`."""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.output_dir = config.output_path()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        random.seed(config.run.seed)
        # ReportWriter handles JSON/CSV/Markdown emission.
        self.reporter = ReportWriter(self.output_dir)

    # ------------------------------------------------------------- public API
    def run(self) -> Dict[str, Any]:
        """Execute every model x task pair and write reports. Returns a summary dict."""

        results: List[TaskResult] = []
        for model_cfg in self.config.models:
            model = self._build_model(model_cfg)
            for task_cfg in self.config.tasks:
                result = self._run_pair(model, model_cfg, task_cfg)
                results.append(result)
        meta = self._metadata()
        self.reporter.write(results, meta, save_predictions=self.config.run.save_predictions)
        return self._summary(results, meta)

    async def arun(self) -> Dict[str, Any]:
        """Async variant of :meth:`run`; used when ``executor: async`` is selected."""

        results: List[TaskResult] = []
        for model_cfg in self.config.models:
            model = self._build_model(model_cfg)
            for task_cfg in self.config.tasks:
                result = await self._arun_pair(model, model_cfg, task_cfg)
                results.append(result)
        meta = self._metadata()
        self.reporter.write(results, meta, save_predictions=self.config.run.save_predictions)
        return self._summary(results, meta)

    # ----------------------------------------------------------------- helpers
    def _build_model(self, cfg: ModelConfig) -> Model:
        return build_model(cfg)

    def _metadata(self) -> Dict[str, Any]:
        return {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "config_source": self.config.source,
            "tags": self.config.run.tags,
            "seed": self.config.run.seed,
        }

    def _summary(self, results: List[TaskResult], meta) -> Dict[str, Any]:
        return {
            "metadata": meta,
            "results": [r.as_dict(include_samples=False) for r in results],
            "output_dir": str(self.output_dir),
        }

    # --------------------------------------------------- single (model, task)
    def _run_pair(
        self, model: Model, model_cfg: ModelConfig, task_cfg: TaskConfig
    ) -> TaskResult:
        task = self._build_task(task_cfg)
        samples = self._prepare_samples(task, task_cfg)
        ckpt = self._checkpoint_path(model_cfg.name, task_cfg.name)
        completed = self._load_checkpoint(ckpt) if self.config.run.resume else {}

        records: List[SampleResult] = []
        # Reuse already-completed samples; only run the missing indices.
        todo: List[Tuple[int, Dict[str, Any]]] = []
        for idx, sample in samples:
            if str(idx) in completed:
                records.append(self._sample_from_cache(completed[str(idx)]))
            else:
                todo.append((idx, sample))

        if todo:
            log.info(
                "Model '%s' / task '%s': %d samples (%d cached, %d to run)",
                model_cfg.name, task_cfg.name, len(samples), len(records), len(todo),
            )
            new_results = self._execute(model, task, model_cfg, task_cfg, todo, ckpt)
            records.extend(new_results)
        else:
            log.info("Model '%s' / task '%s': fully cached (%d samples)",
                     model_cfg.name, task_cfg.name, len(records))

        return self._aggregate(model, model_cfg, task_cfg, records)

    async def _arun_pair(
        self, model: Model, model_cfg: ModelConfig, task_cfg: TaskConfig
    ) -> TaskResult:
        task = self._build_task(task_cfg)
        samples = self._prepare_samples(task, task_cfg)
        ckpt = self._checkpoint_path(model_cfg.name, task_cfg.name)
        completed = self._load_checkpoint(ckpt) if self.config.run.resume else {}

        records: List[SampleResult] = []
        todo: List[Tuple[int, Dict[str, Any]]] = []
        for idx, sample in samples:
            if str(idx) in completed:
                records.append(self._sample_from_cache(completed[str(idx)]))
            else:
                todo.append((idx, sample))

        if todo:
            log.info(
                "Model '%s' / task '%s' (async): %d samples (%d cached, %d to run)",
                model_cfg.name, task_cfg.name, len(samples), len(records), len(todo),
            )
            new_results = await self._aexecute(
                model, task, model_cfg, task_cfg, todo, ckpt
            )
            records.extend(new_results)

        return self._aggregate(model, model_cfg, task_cfg, records)

    def _build_task(self, cfg: TaskConfig) -> Task:
        cls = get_task(cfg.type)
        return cls(cfg, harness_config=self.config)

    def _prepare_samples(
        self, task: Task, cfg: TaskConfig
    ) -> List[Tuple[int, Dict[str, Any]]]:
        samples = task.load_dataset()
        if cfg.sample_range:
            samples = _slice_samples(samples, cfg.sample_range)
        if cfg.limit is not None:
            samples = samples[: int(cfg.limit)]
        return list(enumerate(samples))

    def _checkpoint_path(self, model_name: str, task_name: str) -> Path:
        safe_model = model_name.replace("/", "_").replace(":", "_")
        safe_task = task_name.replace("/", "_").replace(":", "_")
        return self.checkpoint_dir / f"{safe_model}__{safe_task}.jsonl"

    def _load_checkpoint(self, path: Path) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            return {}
        cache: Dict[str, Dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[str(rec.get("index"))] = rec
        log.info("Loaded %d cached samples from %s", len(cache), path.name)
        return cache

    def _append_checkpoint(self, path: Path, record: SampleResult) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.__dict__, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _sample_from_cache(rec: Dict[str, Any]) -> SampleResult:
        return SampleResult(
            index=int(rec.get("index", -1)),
            task=rec.get("task", ""),
            model=rec.get("model", ""),
            prediction=rec.get("prediction"),
            reference=rec.get("reference"),
            raw=rec.get("raw", ""),
            error=rec.get("error"),
            latency_ms=rec.get("latency_ms", 0.0),
            prompt_tokens=rec.get("prompt_tokens", 0),
            completion_tokens=rec.get("completion_tokens", 0),
            sample=rec.get("sample", {}),
        )

    # ------------------------------------------------------- execution cores
    def _execute(
        self,
        model: Model,
        task: Task,
        model_cfg: ModelConfig,
        task_cfg: TaskConfig,
        todo: List[Tuple[int, Dict[str, Any]]],
        ckpt: Path,
    ) -> List[SampleResult]:
        results: List[SampleResult] = []
        max_workers = max(1, min(self.config.run.max_workers, model_cfg.concurrency))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self._infer_one, model, task, model_cfg, task_cfg, idx, sample): idx
                for idx, sample in todo
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    rec = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    log.exception("Sample %d raised unexpectedly", idx)
                    rec = SampleResult(
                        index=idx, task=task_cfg.name, model=model_cfg.name, error=f"crash: {exc!s}"
                    )
                self._append_checkpoint(ckpt, rec)
                results.append(rec)
        # Sort to preserve dataset order in the final report.
        results.sort(key=lambda r: r.index)
        return results

    async def _aexecute(
        self,
        model: Model,
        task: Task,
        model_cfg: ModelConfig,
        task_cfg: TaskConfig,
        todo: List[Tuple[int, Dict[str, Any]]],
        ckpt: Path,
    ) -> List[SampleResult]:
        results: List[SampleResult] = []

        async def _run_one(idx, sample):
            rec = await self._ainfer_one(model, task, model_cfg, task_cfg, idx, sample)
            self._append_checkpoint(ckpt, rec)
            return rec

        tasks = [asyncio.create_task(_run_one(idx, sample)) for idx, sample in todo]
        for coro in asyncio.as_completed(tasks):
            rec = await coro
            results.append(rec)
        results.sort(key=lambda r: r.index)
        return results

    def _infer_one(
        self,
        model: Model,
        task: Task,
        model_cfg: ModelConfig,
        task_cfg: TaskConfig,
        idx: int,
        sample: Dict[str, Any],
    ) -> SampleResult:
        payload = task.build_prompt(sample)
        resp: ModelResponse = model.generate(payload)
        if resp.error:
            prediction = None
        else:
            prediction = task.postprocess(resp.text, sample)
        return SampleResult(
            index=idx,
            task=task_cfg.name,
            model=model_cfg.name,
            prediction=prediction,
            reference=task.reference(sample),
            raw=resp.text,
            error=resp.error,
            latency_ms=resp.latency_ms,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            sample=sample,
        )

    async def _ainfer_one(
        self,
        model: Model,
        task: Task,
        model_cfg: ModelConfig,
        task_cfg: TaskConfig,
        idx: int,
        sample: Dict[str, Any],
    ) -> SampleResult:
        payload = task.build_prompt(sample)
        resp: ModelResponse = await model.agenerate(payload)
        if resp.error:
            prediction = None
        else:
            prediction = task.postprocess(resp.text, sample)
        return SampleResult(
            index=idx,
            task=task_cfg.name,
            model=model_cfg.name,
            prediction=prediction,
            reference=task.reference(sample),
            raw=resp.text,
            error=resp.error,
            latency_ms=resp.latency_ms,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            sample=sample,
        )

    # ----------------------------------------------------------- aggregation
    def _aggregate(
        self,
        model: Model,
        model_cfg: ModelConfig,
        task_cfg: TaskConfig,
        records: List[SampleResult],
    ) -> TaskResult:
        records.sort(key=lambda r: r.index)
        # Build metric input records from sample results.
        metric_records = [
            {
                "prediction": r.prediction,
                "reference": r.reference,
                "raw": r.raw,
                **{
                    k: v for k, v in r.sample.items() if k != "reference"
                },
            }
            for r in records
        ]
        # Re-create a task purely to call evaluate(); it needs metrics from cfg.
        task = self._build_task(task_cfg)
        metrics = task.evaluate(metric_records)
        num_errors = sum(1 for r in records if r.error)
        return TaskResult(
            model=model_cfg.name,
            task=task_cfg.name,
            metrics=metrics,
            num_samples=len(records),
            num_errors=num_errors,
            total_latency_ms=sum(r.latency_ms for r in records),
            total_prompt_tokens=sum(r.prompt_tokens for r in records),
            total_completion_tokens=sum(r.completion_tokens for r in records),
            samples=records,
        )


# ----------------------------------------------------------------- utilities
def _slice_samples(samples: List[Dict[str, Any]], spec: str) -> List[Dict[str, Any]]:
    """Apply a ``start:stop`` or ``start:stop:step`` slice string to the sample list."""

    parts = spec.split(":")
    if len(parts) == 1:
        return samples[int(parts[0]):]
    if len(parts) == 2:
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else None
        return samples[start:stop]
    if len(parts) == 3:
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else None
        step = int(parts[2]) if parts[2] else 1
        return samples[start:stop:step]
    raise ValueError(f"Invalid sample_range spec '{spec}'.")


def _git_commit() -> Optional[str]:
    # Resolve the current git commit (best-effort, used purely as metadata).
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, shell=False
        )
        return out.decode().strip()
    except Exception:  # pragma: no cover
        return None


# --------------------------------------------------------------- module-level
def run(config: str | os.PathLike, *, executor: Optional[str] = None) -> Dict[str, Any]:
    """Convenience entry point: load a config file and run the evaluation.

    Parameters
    ----------
    config : path
        YAML or JSON config file.
    executor : str, optional
        Override the config's ``run.executor`` (``thread`` or ``async``).
    """

    from .config import load_config

    cfg = load_config(config)
    _configure_logging(cfg.run.log_level)
    # Importing the sub-packages registers built-in tasks/metrics/adapters.
    import fylharness.tasks  # noqa: F401
    import fylharness.data  # noqa: F401
    import fylharness.metrics  # noqa: F401

    exe = executor or cfg.run.executor
    runner = Runner(cfg)
    if exe == "async":
        return asyncio.run(runner.arun())
    return runner.run()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
