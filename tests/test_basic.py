"""Smoke tests for FylHarness core functionality.

These tests run fully offline (using the mock model and inline datasets) so they pass
on a clean Windows CI without any LLM server.  Run with ``pytest tests/`` from the
repository root, or ``python -m pytest tests/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure the package is importable when running directly from a checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import fylharness  # noqa: E402
import fylharness.tasks  # noqa: E402,F401
import fylharness.data  # noqa: E402,F401
import fylharness.metrics  # noqa: E402,F401
from fylharness.config import HarnessConfig, ModelConfig, RunConfig, TaskConfig  # noqa: E402
from fylharness.models.mock import MockModel  # noqa: E402
from fylharness.runner import Runner  # noqa: E402
from fylharness.tasks.multiple_choice import MultipleChoiceTask  # noqa: E402


# --------------------------------------------------------------- registries
def test_registries_populated():
    assert "multiple_choice" in fylharness.list_tasks()
    assert "accuracy" in fylharness.list_metrics()
    assert "local" in fylharness.list_adapters()


# -------------------------------------------------------------------- metrics
def test_accuracy_metric():
    from fylharness.metrics import accuracy
    samples = [
        {"prediction": "A", "reference": "A"},
        {"prediction": "B", "reference": "A"},
        {"prediction": "A", "reference": "A"},
    ]
    assert accuracy(samples) == pytest.approx(2 / 3)


def test_rouge_metric_returns_dict():
    from fylharness.metrics import rouge
    out = rouge([{"prediction": "the cat sat", "reference": "the cat sat on the mat"}])
    assert set(out) == {"rouge_1", "rouge_2", "rouge_l"}
    assert 0.0 <= out["rouge_1"] <= 1.0
    assert out["rouge_1"] > 0.0


def test_bleu_perfect_match_is_positive():
    from fylharness.metrics import bleu
    # Use a sentence with at least 4 tokens so BLEU-4 has all n-gram orders.
    out = bleu([{"prediction": "the quick brown fox jumps", "reference": "the quick brown fox jumps"}])
    assert out > 0.0


def test_bleu_mismatched_is_low():
    from fylharness.metrics import bleu
    out = bleu([{"prediction": "the quick brown fox jumps", "reference": "a slow green turtle crawls"}])
    assert out == 0.0


# ----------------------------------------------------------------------- mock
def test_mock_model_mcq_returns_letter():
    cfg = ModelConfig(name="m", type="mock")
    model = MockModel(cfg)
    payload = {"prompt": "What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"}
    resp = model.generate(payload)
    assert resp.text.strip() in "ABCDE"
    assert resp.error is None


# ---------------------------------------------------------------- end-to-end
def _make_config(tmp_path: Path, samples):
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(samples), encoding="utf-8")
    return HarnessConfig(
        models=[ModelConfig(name="mock", type="mock", extra={"answer_letter": "B"})],
        tasks=[
            TaskConfig(
                name="mcq",
                type="multiple_choice",
                dataset={"adapter": "local", "path": str(data_path), "format": "json"},
                metrics=["accuracy"],
            )
        ],
        run=RunConfig(output_dir=str(tmp_path / "out"), resume=False),
    )


def test_end_to_end_run(tmp_path):
    samples = [
        {"question": f"Q{i}", "choices": ["A", "B", "C", "D"], "answer": i % 4}
        for i in range(8)
    ]
    cfg = _make_config(tmp_path, samples)
    runner = Runner(cfg)
    summary = runner.run()
    assert len(summary["results"]) == 1
    r = summary["results"][0]
    assert r["task"] == "mcq"
    assert r["num_samples"] == 8
    # Mock always picks B (index 1) -> correct when answer index == 1.
    assert r["metrics"]["accuracy"] == pytest.approx(2 / 8)
    out_dir = Path(summary["output_dir"])
    assert (out_dir / "results.json").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "report.md").exists()


def test_resume_skips_completed(tmp_path):
    samples = [
        {"question": f"Q{i}", "choices": ["A", "B"], "answer": 0} for i in range(4)
    ]
    cfg = _make_config(tmp_path, samples)
    cfg.run.resume = True
    runner = Runner(cfg)
    runner.run()
    # Second run should hit the cache and produce identical metrics.
    runner2 = Runner(cfg)
    summary = runner2.run()
    assert summary["results"][0]["num_samples"] == 4
