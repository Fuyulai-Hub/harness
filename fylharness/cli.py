"""Command-line interface for FylHarness.

Examples
--------
Run a full evaluation from a config file::

    fylharness run config.yaml --executor thread

List registered tasks/metrics/adapters::

    fylharness list

Run a single quick smoke test against a local server with inline data::

    fylharness run examples/config_example.yaml --limit 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from . import __version__
from .config import load_config
from .runner import Runner, _configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fylharness",
        description="A lightweight, Windows-friendly evaluation harness for "
                    "OpenAI-compatible LLMs.",
    )
    parser.add_argument("--version", action="version", version=f"fylharness {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- run -------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run an evaluation from a config file.")
    p_run.add_argument("config", help="Path to the YAML/JSON config file.")
    p_run.add_argument(
        "--executor",
        choices=["thread", "async"],
        default=None,
        help="Override the executor (default: from config).",
    )
    p_run.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of samples per task (overrides config).",
    )
    p_run.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing checkpoints and re-run all samples.",
    )
    p_run.add_argument(
        "--log-level", default=None,
        help="Logging level (DEBUG/INFO/WARNING/ERROR).",
    )
    p_run.add_argument(
        "--model", default=None,
        help="Only run models whose name matches this filter.",
    )
    p_run.add_argument(
        "--task", default=None,
        help="Only run tasks whose name matches this filter.",
    )
    p_run.add_argument(
        "--open", action="store_true",
        help="Open the results dashboard in a browser after the run finishes.",
    )
    p_run.add_argument(
        "--port", type=int, default=0,
        help="Port for the dashboard server (default: random free port).",
    )

    # ---- list -------------------------------------------------------------
    sub.add_parser("list", help="List registered tasks, metrics and adapters.")

    # ---- show -------------------------------------------------------------
    p_show = sub.add_parser("show", help="Print the resolved config (after env interp).")
    p_show.add_argument("config", help="Path to the YAML/JSON config file.")

    # ---- web --------------------------------------------------------------
    p_web = sub.add_parser(
        "web", help="Serve the results dashboard for a previous run's output dir."
    )
    p_web.add_argument(
        "results_dir", nargs="?", default="outputs",
        help="Directory containing results.json (default: outputs).",
    )
    p_web.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)."
    )
    p_web.add_argument(
        "--port", type=int, default=0, help="Bind port (default: random free port)."
    )
    p_web.add_argument(
        "--no-browser", action="store_true",
        help="Do not auto-open the browser.",
    )

    # ---- serve (interactive harness) ------------------------------------
    p_serve = sub.add_parser(
        "serve",
        help="Launch the interactive harness: chat playground + evaluation UI + results.",
    )
    p_serve.add_argument(
        "config", help="Path to the YAML/JSON config file (defines available models & tasks)."
    )
    p_serve.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)."
    )
    p_serve.add_argument(
        "--port", type=int, default=5000, help="Bind port (default: 5000)."
    )
    p_serve.add_argument(
        "--no-browser", action="store_true",
        help="Do not auto-open the browser.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Importing the sub-packages registers built-in tasks/metrics/adapters.
    import fylharness.tasks  # noqa: F401
    import fylharness.data  # noqa: F401
    import fylharness.metrics  # noqa: F401

    if args.command == "list":
        from . import list_tasks, list_metrics, list_adapters
        print("Tasks:    " + ", ".join(list_tasks()) or "(none)")
        print("Metrics:  " + ", ".join(list_metrics()))
        print("Adapters: " + ", ".join(list_adapters()))
        return 0

    if args.command == "show":
        cfg = load_config(args.config)
        print(json.dumps(cfg.to_dict(), indent=2, default=str, ensure_ascii=False))
        return 0

    if args.command == "run":
        return _cmd_run(args)

    if args.command == "web":
        from .web import serve_dashboard
        serve_dashboard(
            args.results_dir,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
        return 0

    if args.command == "serve":
        cfg = load_config(args.config)
        _configure_logging(cfg.run.log_level)
        from .server import HarnessServer
        server = HarnessServer(cfg)
        server.run(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
        return 0

    parser.print_help()
    return 1


def _cmd_run(args) -> int:
    import asyncio

    cfg = load_config(args.config)
    if args.log_level:
        cfg.run.log_level = args.log_level
    _configure_logging(cfg.run.log_level)

    # Apply CLI overrides.
    if args.executor:
        cfg.run.executor = args.executor
    if args.no_resume:
        cfg.run.resume = False
    if args.model:
        cfg.models = [m for m in cfg.models if args.model in m.name]
        if not cfg.models:
            print(f"No models matched filter '{args.model}'", file=sys.stderr)
            return 2
    if args.task:
        cfg.tasks = [t for t in cfg.tasks if args.task in t.name]
        if not cfg.tasks:
            print(f"No tasks matched filter '{args.task}'", file=sys.stderr)
            return 2
    if args.limit is not None:
        for t in cfg.tasks:
            t.limit = args.limit

    runner = Runner(cfg)
    if cfg.run.executor == "async":
        summary = asyncio.run(runner.arun())
    else:
        summary = runner.run()

    print("\n===== Evaluation Summary =====")
    for r in summary.get("results", []):
        metrics_str = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in (r.get("metrics") or {}).items()
        )
        print(f"[{r['model']}] {r['task']}: samples={r['num_samples']} "
              f"errors={r['num_errors']} {metrics_str}")
    print(f"\nReports written to: {summary.get('output_dir')}")

    if args.open:
        # Launch the web dashboard for the freshly written results.  The server
        # blocks until the user stops it with Ctrl+C, mirroring the DeepSeek
        # harness workflow of "run -> view results in browser".
        from .web import serve_dashboard
        serve_dashboard(
            summary["output_dir"],
            port=args.port,
            open_browser=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
