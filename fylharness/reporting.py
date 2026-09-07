"""Result serialisation and human-readable report generation.

Three artefacts are produced for every run:

* ``results.json``  -- the complete, machine-readable record (metadata + per-task
  metrics + optional per-sample predictions).  This is the source of truth.
* ``summary.csv``   -- a flat one-row-per-(model,task) table for spreadsheet viewing.
* ``report.md``      -- a Markdown summary with one table per model, suitable for
  pasting into a PR description or README.

The writer is intentionally side-effect free apart from file I/O so it can be unit
tested in isolation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .results import TaskResult


class ReportWriter:
    """Persist evaluation results to JSON/CSV/Markdown."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        results: List[TaskResult],
        metadata: Dict[str, Any],
        save_predictions: bool = True,
    ) -> Dict[str, Path]:
        """Write all artefacts; returns the paths of the files produced."""

        json_path = self.output_dir / "results.json"
        csv_path = self.output_dir / "summary.csv"
        md_path = self.output_dir / "report.md"

        payload = {
            "metadata": metadata,
            "results": [r.as_dict(include_samples=save_predictions) for r in results],
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self._write_csv(results, csv_path)
        self._write_markdown(results, metadata, md_path)
        return {"json": json_path, "csv": csv_path, "markdown": md_path}

    # ------------------------------------------------------------------- JSON
    @staticmethod
    def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
        """Flatten nested metric dicts (e.g. ROUGE) into ``metric.sub`` keys."""

        flat: Dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for sub, val in v.items():
                    flat[f"{k}.{sub}"] = _coerce_float(val)
            else:
                flat[k] = _coerce_float(v)
        return flat

    # -------------------------------------------------------------------- CSV
    def _write_csv(self, results: List[TaskResult], path: Path) -> None:
        # Collect the union of all metric keys across results so the table is square.
        all_metric_keys: List[str] = []
        seen = set()
        for r in results:
            for k in self._flatten_metrics(r.metrics or {}):
                if k not in seen:
                    seen.add(k)
                    all_metric_keys.append(k)

        base_cols = ["model", "task", "num_samples", "num_errors",
                     "total_latency_ms", "total_prompt_tokens",
                     "total_completion_tokens"]
        header = base_cols + all_metric_keys
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for r in results:
                metrics = self._flatten_metrics(r.metrics or {})
                row = [
                    r.model, r.task, r.num_samples, r.num_errors,
                    f"{r.total_latency_ms:.2f}", r.total_prompt_tokens,
                    r.total_completion_tokens,
                ]
                for k in all_metric_keys:
                    val = metrics.get(k, "")
                    row.append(f"{val:.4f}" if isinstance(val, float) else val)
                writer.writerow(row)

    # --------------------------------------------------------------- Markdown
    def _write_markdown(
        self,
        results: List[TaskResult],
        metadata: Dict[str, Any],
        path: Path,
    ) -> None:
        lines: List[str] = []
        lines.append("# FylHarness Evaluation Report\n")
        lines.append(f"- **Started:** {metadata.get('started_at', 'n/a')}")
        lines.append(f"- **Platform:** {metadata.get('platform', 'n/a')}")
        lines.append(f"- **Python:** {metadata.get('python', 'n/a')}")
        commit = metadata.get("git_commit")
        if commit:
            lines.append(f"- **Git commit:** `{commit[:12]}`")
        tags = metadata.get("tags") or []
        if tags:
            lines.append(f"- **Tags:** {', '.join(tags)}")
        lines.append("")

        if not results:
            lines.append("_No tasks were run._\n")
            path.write_text("\n".join(lines), encoding="utf-8")
            return

        # Group by model for a per-model table.
        by_model: Dict[str, List[TaskResult]] = {}
        for r in results:
            by_model.setdefault(r.model, []).append(r)

        for model, group in by_model.items():
            lines.append(f"## Model: `{model}`\n")
            all_keys: List[str] = []
            seen = set()
            for r in group:
                for k in self._flatten_metrics(r.metrics or {}):
                    if k not in seen:
                        seen.add(k)
                        all_keys.append(k)
            header = ["Task", "Samples", "Errors"] + all_keys
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
            for r in group:
                metrics = self._flatten_metrics(r.metrics or {})
                cells = [r.task, str(r.num_samples), str(r.num_errors)]
                for k in all_keys:
                    v = metrics.get(k, "")
                    cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")


def _coerce_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
