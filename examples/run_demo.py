#!/usr/bin/env python3
"""Quick demo: run FylHarness offline against the built-in mock model.

This script exercises the full pipeline -- task loading, concurrency, checkpointing,
metric computation and report generation -- without any network access, which makes it
ideal for a first smoke test on a fresh Windows machine.

Run from the repository root:
    python examples/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    # Make the fylharness package importable when running this file directly.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import fylharness  # noqa: F401  (registers built-ins via sub-imports)
    import fylharness.tasks  # noqa: F401
    import fylharness.data  # noqa: F401
    import fylharness.metrics  # noqa: F401

    config_path = Path(__file__).parent / "config_mock.yaml"
    print(f"Running FylHarness demo with config: {config_path}")
    summary = fylharness.run(str(config_path))

    print("\n===== Summary =====")
    for r in summary.get("results", []):
        metrics = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in (r.get("metrics") or {}).items()
        )
        print(f"[{r['model']}] {r['task']}: samples={r['num_samples']} "
              f"errors={r['num_errors']} {metrics}")
    print(f"\nReports written to: {summary.get('output_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
