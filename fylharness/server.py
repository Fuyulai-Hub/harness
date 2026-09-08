"""Interactive harness server: chat playground + evaluation runner + results viewer.

This is the real application layer of FylHarness.  Built on Flask, it exposes a
single-page app with three working surfaces:

1. **Playground** -- a chat UI (like DeepSeek's) where you pick a configured model,
   type messages and watch streamed responses token-by-token.
2. **Evaluate** -- pick a model + task + dataset, set a sample cap and run a real
   evaluation inline; progress and per-sample predictions appear live.
3. **Results** -- browse the JSON results of any past or just-finished run, with
   summary matrix, per-sample drill-down and filtering.

Run it with::

    fylharness serve examples/config_mock.yaml        # full UI + chat
    fylharness serve examples/config_mock.yaml --port 5000 --no-browser

The server builds :class:`~fylharness.config.HarnessConfig` once at startup; the model
and task registries are populated by importing the built-in sub-packages.  Every model
listed in the config becomes available in the playground, so pointing the config at a
real endpoint immediately makes the model chat-able in the browser.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import HarnessConfig, load_config
from .models.base import Model
from .models.registry import build_model

log = logging.getLogger(__name__)

try:
    from flask import Flask, Response, jsonify, request, stream_with_context
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Flask is required for the interactive harness server. "
        "Install it with `pip install flask`."
    ) from exc


class HarnessServer:
    """A Flask application bundling chat, evaluation and results endpoints.

    The server holds one :class:`HarnessConfig` and lazily-constructed model
    instances keyed by name.  Evaluation runs are dispatched to a background thread
    so the request returns immediately and the UI polls for progress.
    """

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        # Importing the sub-packages registers built-in tasks/metrics/adapters.
        import fylharness.tasks  # noqa: F401
        import fylharness.data  # noqa: F401
        import fylharness.metrics  # noqa: F401
        self._models: Dict[str, Model] = {}
        self._lock = threading.Lock()
        # Per-run progress store: run_id -> {status, done, total, results, error}
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._run_counter = 0
        # Per-agent stop flags so the user can abort a running agent.
        self._agent_stop: Dict[str, bool] = {}
        self.app = Flask(__name__)
        self._register_routes()

    # ---------------------------------------------------------------- models
    def get_model(self, name: str) -> Model:
        """Return a cached :class:`Model` instance for config model ``name``."""

        with self._lock:
            if name not in self._models:
                cfg = next((m for m in self.config.models if m.name == name), None)
                if cfg is None:
                    raise KeyError(f"Unknown model '{name}'")
                self._models[name] = build_model(cfg)
            return self._models[name]

    def add_model(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically register a new model from the Workspace UI.

        Accepts: name, type, base_url, api_key, model, temperature,
        max_tokens, reasoning_effort, and any other ModelConfig fields.
        """

        from .config import ModelConfig

        name = spec.get("name", "").strip()
        if not name:
            raise ValueError("Model name is required.")
        with self._lock:
            if any(m.name == name for m in self.config.models):
                raise ValueError(f"Model '{name}' already exists.")
            cfg = ModelConfig(
                name=name,
                type=spec.get("type", "openai"),
                base_url=spec.get("base_url", "http://127.0.0.1:8000/v1"),
                api_key=spec.get("api_key", ""),
                api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
                model=spec.get("model") or name,
                chat=spec.get("chat", True),
                max_tokens=int(spec.get("max_tokens", 1024)),
                temperature=float(spec.get("temperature", 0.7)),
                timeout=float(spec.get("timeout", 120)),
                max_retries=int(spec.get("max_retries", 5)),
                retry_backoff=float(spec.get("retry_backoff", 2.0)),
                concurrency=int(spec.get("concurrency", 4)),
                extra=dict(spec.get("extra", {})),
            )
            # Store reasoning_effort in extra so the model can pick it up.
            if spec.get("reasoning_effort"):
                cfg.extra["reasoning_effort"] = spec["reasoning_effort"]
            self.config.models.append(cfg)
            return {
                "name": cfg.name,
                "type": cfg.type,
                "base_url": cfg.base_url,
                "model": cfg.model,
                "has_key": bool(cfg.resolved_api_key()),
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "reasoning_effort": cfg.extra.get("reasoning_effort", ""),
            }

    def remove_model(self, name: str) -> bool:
        """Remove a dynamically-added model by name."""

        with self._lock:
            before = len(self.config.models)
            self.config.models = [m for m in self.config.models if m.name != name]
            self._models.pop(name, None)
            return len(self.config.models) < before

    def model_names(self) -> List[str]:
        return [m.name for m in self.config.models]

    # ------------------------------------------------------------------ runs
    def _new_run_id(self) -> str:
        with self._lock:
            self._run_counter += 1
            run_id = f"run-{int(time.time())}-{self._run_counter}"
            self._runs[run_id] = {
                "status": "pending",
                "done": 0,
                "total": 0,
                "model": "",
                "task": "",
                "results": [],
                "error": None,
            }
            return run_id

    def _update_run(self, run_id: str, **patch) -> None:
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id].update(patch)

    def _get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._runs.get(run_id)

    # ------------------------------------------------------------- routes
    def _register_routes(self) -> None:
        app = self.app

        @app.route("/")
        def index() -> str:
            return _INDEX_HTML

        # ---- metadata ------------------------------------------------------
        @app.route("/api/config")
        def api_config():
            models = [
                {
                    "name": m.name,
                    "type": m.type,
                    "base_url": m.base_url,
                    "model": m.model or m.name,
                    "has_key": bool(m.resolved_api_key()),
                    "temperature": m.temperature,
                    "max_tokens": m.max_tokens,
                    "reasoning_effort": m.extra.get("reasoning_effort", ""),
                }
                for m in self.config.models
            ]
            tasks = [
                {
                    "name": t.name,
                    "type": t.type,
                    "metrics": t.metrics,
                    "limit": t.limit,
                    "dataset": t.dataset,
                }
                for t in self.config.tasks
            ]
            return jsonify({
                "models": models,
                "tasks": tasks,
                "output_dir": self.config.run.output_dir,
            })

        # ---- model management (Workspace) ---------------------------------
        @app.route("/api/models", methods=["GET", "POST"])
        def api_models():
            if request.method == "POST":
                body = request.get_json(force=True, silent=True) or {}
                try:
                    m = self.add_model(body)
                except ValueError as exc:
                    return jsonify({"error": str(exc)}), 400
                return jsonify(m), 201
            # GET: list all models with full detail
            return jsonify({
                "models": [
                    {
                        "name": m.name,
                        "type": m.type,
                        "base_url": m.base_url,
                        "model": m.model or m.name,
                        "has_key": bool(m.resolved_api_key()),
                        "temperature": m.temperature,
                        "max_tokens": m.max_tokens,
                        "reasoning_effort": m.extra.get("reasoning_effort", ""),
                    }
                    for m in self.config.models
                ]
            })

        @app.route("/api/models/<model_name>", methods=["DELETE"])
        def api_delete_model(model_name: str):
            if self.remove_model(model_name):
                return jsonify({"ok": True})
            return jsonify({"error": "not found"}), 404

        # ---- chat (streaming SSE) -----------------------------------------
        @app.route("/api/chat", methods=["POST"])
        def api_chat():
            body = request.get_json(force=True, silent=True) or {}
            model_name = body.get("model", "")
            messages = body.get("messages", [])
            if not messages:
                return jsonify({"error": "messages required"}), 400
            try:
                model = self.get_model(model_name)
            except KeyError as exc:
                return jsonify({"error": str(exc)}), 404

            # Per-request inference overrides from the UI (reasoning strength).
            overrides = {}
            for k in ("temperature", "max_tokens", "reasoning_effort"):
                v = body.get(k)
                if v is not None and v != "":
                    overrides[k] = v

            def generate():
                try:
                    for chunk in model.chat_stream(messages, **overrides):
                        # SSE format: one event per chunk.
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:  # pragma: no cover
                    yield f"data: {json.dumps('[ERROR] ' + str(exc), ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

            return Response(
                stream_with_context(generate()),
                content_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # ---- non-streaming chat (fallback) --------------------------------
        @app.route("/api/chat/sync", methods=["POST"])
        def api_chat_sync():
            body = request.get_json(force=True, silent=True) or {}
            model_name = body.get("model", "")
            messages = body.get("messages", [])
            try:
                model = self.get_model(model_name)
            except KeyError as exc:
                return jsonify({"error": str(exc)}), 404
            resp = model.chat(messages)
            if resp.error:
                return jsonify({"error": resp.error}), 502
            return jsonify({
                "text": resp.text,
                "latency_ms": resp.latency_ms,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
            })

        # ---- evaluation: start -------------------------------------------
        @app.route("/api/evaluate", methods=["POST"])
        def api_evaluate():
            body = request.get_json(force=True, silent=True) or {}
            model_name = body.get("model")
            task_name = body.get("task")
            limit = body.get("limit")  # optional cap

            task_cfg = next(
                (t for t in self.config.tasks if t.name == task_name), None
            )
            if task_cfg is None:
                return jsonify({"error": f"Unknown task '{task_name}'"}), 404
            if not any(m.name == model_name for m in self.config.models):
                return jsonify({"error": f"Unknown model '{model_name}'"}), 404

            run_id = self._new_run_id()
            self._update_run(
                run_id, model=model_name, task=task_name, status="running"
            )

            # Heavy work runs in a background thread; the UI polls /api/runs/<id>.
            thread = threading.Thread(
                target=self._run_evaluation,
                args=(run_id, model_name, task_cfg, limit),
                daemon=True,
            )
            thread.start()
            return jsonify({"run_id": run_id})

        # ---- evaluation: poll status -------------------------------------
        @app.route("/api/runs/<run_id>")
        def api_run_status(run_id: str):
            run = self._get_run(run_id)
            if run is None:
                return jsonify({"error": "unknown run"}), 404
            return jsonify(run)

        # ---- results: list + fetch --------------------------------------
        @app.route("/api/results")
        def api_results_list():
            out_dir = Path(self.config.run.output_dir)
            runs = []
            if out_dir.exists():
                for p in sorted(out_dir.rglob("results.json"), reverse=True):
                    runs.append(str(p.parent.relative_to(out_dir).as_posix()))
            return jsonify({"runs": runs, "output_dir": str(out_dir)})

        @app.route("/api/results/<path:run_path>")
        def api_results_get(run_path: str):
            out_dir = Path(self.config.run.output_dir)
            target = (out_dir / run_path / "results.json").resolve()
            # Guard against path traversal.
            try:
                target.relative_to(out_dir.resolve())
            except ValueError:
                return jsonify({"error": "invalid path"}), 400
            if not target.exists():
                return jsonify({"error": "not found"}), 404
            return jsonify(json.loads(target.read_text(encoding="utf-8")))

        # ---- agent: run autonomous task -----------------------------------
        @app.route("/api/agent", methods=["POST"])
        def api_agent():
            body = request.get_json(force=True, silent=True) or {}
            model_name = body.get("model", "")
            workspace = body.get("workspace", "")
            task = body.get("task", "")
            max_steps = int(body.get("max_steps", 25))
            agent_cfg = self.config.agent
            planning = bool(body.get("planning", agent_cfg.planning))
            reflection = bool(body.get("reflection", agent_cfg.reflection))
            reflect_every = int(body.get("reflect_every", agent_cfg.reflect_every))
            overrides = {}
            for k in ("temperature", "max_tokens", "reasoning_effort"):
                v = body.get(k)
                if v is not None and v != "":
                    overrides[k] = v
            if not model_name:
                return jsonify({"error": "model is required"}), 400
            if not workspace:
                return jsonify({"error": "workspace is required"}), 400
            if not task:
                return jsonify({"error": "task is required"}), 400
            try:
                model = self.get_model(model_name)
            except KeyError as exc:
                return jsonify({"error": str(exc)}), 404
            ws = Path(workspace)
            if not ws.exists() or not ws.is_dir():
                return jsonify({"error": f"Workspace directory not found: {workspace}"}), 400

            run_id = self._new_run_id()
            self._update_run(run_id, model=model_name, task="agent",
                             task_desc=task, workspace=str(ws), total=max_steps,
                             status="running", done=0, events=[],
                             created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            self._agent_stop[run_id] = False

            def generate():
                from .agent import run_agent
                try:
                    for event in run_agent(model, task, ws,
                                            max_steps=max_steps,
                                            overrides=overrides,
                                            planning=planning,
                                            reflection=reflection,
                                            reflect_every=reflect_every,
                                            max_history_messages=agent_cfg.max_history_messages):
                        if self._agent_stop.get(run_id):
                            yield f"data: {json.dumps({'type': 'end', 'reason': 'stopped by user'}, ensure_ascii=False)}\n\n"
                            break
                        # Inject the run_id so the UI can call the stop endpoint.
                        if "run_id" not in event:
                            event["run_id"] = run_id
                        # Persist events for the /api/runs/<id> polling fallback.
                        with self._lock:
                            if run_id in self._runs:
                                self._runs[run_id].setdefault("events", []).append(event)
                                if event.get("type") == "tool_call":
                                    self._runs[run_id]["done"] = event.get("step", 0)
                                if event.get("type") == "finish":
                                    self._runs[run_id]["status"] = "done"
                        yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                except Exception as exc:  # pragma: no cover
                    yield f"data: {json.dumps({'type': 'error', 'error': str(exc)}, ensure_ascii=False)}\n\n"
                finally:
                    with self._lock:
                        if run_id in self._runs and self._runs[run_id]["status"] == "running":
                            self._runs[run_id]["status"] = "done"
                    self._agent_stop.pop(run_id, None)
                yield "data: [DONE]\n\n"

            return Response(
                stream_with_context(generate()),
                content_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @app.route("/api/agent/<run_id>/stop", methods=["POST"])
        def api_agent_stop(run_id: str):
            self._agent_stop[run_id] = True
            return jsonify({"ok": True})

        @app.route("/api/agent/settings")
        def api_agent_settings():
            """Expose the agent loop defaults so the UI checkboxes can mirror them."""
            a = self.config.agent
            return jsonify({
                "planning": a.planning,
                "reflection": a.reflection,
                "reflect_every": a.reflect_every,
                "max_history_messages": a.max_history_messages,
            })

        @app.route("/api/agent/runs")
        def api_agent_runs():
            """List past agent runs (newest first) with their full event traces."""
            with self._lock:
                runs = []
                for rid, r in sorted(self._runs.items(), reverse=True):
                    if r.get("task") != "agent":
                        continue
                    runs.append({
                        "id": rid,
                        "task_desc": r.get("task_desc", ""),
                        "workspace": r.get("workspace", ""),
                        "model": r.get("model", ""),
                        "status": r.get("status", ""),
                        "done": r.get("done", 0),
                        "total": r.get("total", 0),
                        "created_at": r.get("created_at", ""),
                        "events": r.get("events", []),
                    })
            return jsonify({"runs": runs[:50]})

    # --------------------------------------------------------- evaluation
    def _run_evaluation(
        self,
        run_id: str,
        model_name: str,
        task_cfg,
        limit: Optional[int],
    ) -> None:
        """Background worker: run one (model, task) pair and stream updates."""

        try:
            from .runner import Runner, SampleResult
            from .tasks.base import get_task

            # Build a single-pair config clone so the runner only does this task.
            sub_config = HarnessConfig(
                models=[m for m in self.config.models if m.name == model_name],
                tasks=[task_cfg],
                run=self.config.run,
                source=self.config.source,
            )
            # Apply limit override.
            if limit is not None:
                sub_config.tasks[0].limit = int(limit)
            sub_config.run.resume = False  # fresh run from the UI

            runner = Runner(sub_config)
            model = self.get_model(model_name)

            # Manually drive the task so we can push per-sample progress to the UI.
            task = get_task(task_cfg.type)(task_cfg, harness_config=sub_config)
            samples = runner._prepare_samples(task, task_cfg)
            total = len(samples)
            self._update_run(run_id, total=total, status="running")

            results: List[Dict[str, Any]] = []
            for idx, sample in samples:
                rec = runner._infer_one(model, task, model.config, task_cfg, idx, sample)
                # Materialise to dict for JSON serialisation.
                rec_dict = {
                    "index": rec.index,
                    "prediction": rec.prediction,
                    "reference": rec.reference,
                    "raw": rec.raw,
                    "error": rec.error,
                    "latency_ms": rec.latency_ms,
                    "prompt_tokens": rec.prompt_tokens,
                    "completion_tokens": rec.completion_tokens,
                }
                results.append(rec_dict)
                self._update_run(run_id, done=len(results), results=results)

            # Compute metrics over the collected records.
            metric_records = [
                {"prediction": r["prediction"], "reference": r["reference"],
                 "raw": r["raw"]}
                for r in results
            ]
            metrics = task.evaluate(metric_records)
            self._update_run(
                run_id,
                status="done",
                metrics=metrics,
                results=results,
                num_samples=len(results),
                num_errors=sum(1 for r in results if r["error"]),
            )
            log.info("Run %s complete: %d samples, metrics=%s",
                     run_id, len(results), metrics)
        except Exception as exc:  # pragma: no cover
            log.exception("Evaluation run %s failed", run_id)
            self._update_run(run_id, status="error", error=str(exc))

    # ----------------------------------------------------------- launch
    def run(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        open_browser: bool = True,
    ) -> None:
        """Launch the Flask server and optionally open the browser."""

        import webbrowser

        url = f"http://{host}:{port}/"
        print(f"\n  FylHarness interactive server:  {url}\n"
              f"  Models:  {', '.join(self.model_names()) or '(none)'}\n"
              f"  Tasks:   {', '.join(t.name for t in self.config.tasks) or '(none)'}\n")
        if open_browser:
            threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
        # debug=False, threaded=True so SSE + polling don't deadlock.
        self.app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


# ------------------------------------------------------------------- HTML
# Single-page app: three tabs (Playground / Evaluate / Results) with vanilla JS.
# All CSS/JS inline; the page talks to the JSON/SSE APIs above.
_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FylHarness</title>
<style>
  :root {
    --bg:#0f1115;--panel:#181b22;--panel2:#1f232c;--border:#2a2f3a;
    --text:#e6e8eb;--muted:#8b93a1;--accent:#4f9cf9;--good:#3fb950;
    --warn:#d29922;--bad:#f85149;--user:#2563eb;--asst:#1f232c;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    background:var(--bg);color:var(--text);line-height:1.6;height:100vh;display:flex;flex-direction:column}
  header{background:var(--panel);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;align-items:center;gap:20px;flex-shrink:0}
  header h1{font-size:18px;font-weight:600;white-space:nowrap}
  .tabs{display:flex;gap:2px}
  .tab{padding:8px 16px;cursor:pointer;color:var(--muted);font-size:14px;border-radius:6px;transition:.15s}
  .tab:hover{background:var(--panel2)}
  .tab.active{background:var(--panel2);color:var(--text)}
  .spacer{flex:1}
  .status-dot{width:8px;height:8px;border-radius:50%;background:var(--good);display:inline-block;margin-right:6px}
  main{flex:1;overflow:hidden;position:relative}
  .view{position:absolute;inset:0;display:none;flex-direction:column}
  .view.active{display:flex}

  /* ---------- shared controls ---------- */
  select,input,button{font-family:inherit;font-size:14px;color:var(--text)}
  select,input{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:7px 10px;outline:none}
  select:focus,input:focus{border-color:var(--accent)}
  button{background:var(--accent);color:#fff;border:none;border-radius:6px;padding:8px 16px;cursor:pointer;font-weight:500}
  button:hover{opacity:.9}
  button:disabled{opacity:.5;cursor:not-allowed}
  button.secondary{background:var(--panel2);border:1px solid var(--border)}

  /* ---------- chat ---------- */
  .chat-layout{flex:1;display:flex;flex-direction:row;min-height:0}
  .chat-sidebar{width:240px;background:var(--panel);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0}
  .chat-sidebar .new-chat{margin:12px;padding:10px 14px;text-align:center;font-weight:500}
  .conv-list{flex:1;overflow-y:auto;padding:0 8px 8px}
  .conv-item{padding:9px 12px;border-radius:6px;cursor:pointer;font-size:13px;color:var(--text);margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;transition:background .15s}
  .conv-item:hover{background:var(--panel2)}
  .conv-item.active{background:var(--panel2)}
  .conv-item .conv-del{float:right;color:var(--muted);opacity:0;transition:opacity .15s}
  .conv-item:hover .conv-del{opacity:1}
  .conv-item .conv-del:hover{color:var(--bad)}
  .conv-empty{padding:20px 12px;text-align:center;color:var(--muted);font-size:12px}
  .chat-main{flex:1;display:flex;flex-direction:column;min-width:0}
  .chat-toolbar{padding:12px 20px;border-bottom:1px solid var(--border);display:flex;gap:10px;align-items:center;background:var(--panel)}
  .chat-toolbar label{color:var(--muted);font-size:13px}
  .chat-messages{flex:1;overflow-y:auto;padding:20px;scroll-behavior:smooth}
  .msg{margin-bottom:16px;max-width:80%}
  .msg.user{margin-left:auto}
  .msg-bubble{padding:10px 14px;border-radius:12px;white-space:pre-wrap;word-break:break-word}
  .msg.user .msg-bubble{background:var(--user);color:#fff}
  .msg.assistant .msg-bubble{background:var(--asst);border:1px solid var(--border)}
  .msg-role{font-size:11px;color:var(--muted);margin-bottom:3px}
  .chat-input{padding:12px 20px;border-top:1px solid var(--border);display:flex;gap:10px;background:var(--panel)}
  .chat-input textarea{flex:1;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 12px;resize:none;height:48px;font-family:inherit;font-size:14px;color:var(--text);outline:none}
  .chat-input textarea:focus{border-color:var(--accent)}

  /* ---------- sidebar model panel ---------- */
  .sidebar-model-panel{border-top:1px solid var(--border);padding:12px;flex-shrink:0}
  .sidebar-label{font-size:11px;color:var(--muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em}
  .sidebar-select{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:6px 8px;color:var(--text);font-size:13px;font-family:inherit;outline:none}
  .sidebar-select:focus{border-color:var(--accent)}
  .sidebar-slider{width:100%;accent-color:var(--accent);cursor:pointer}

  /* ---------- workspace ---------- */
  .ws-form{display:flex;flex-direction:column;gap:8px}
  .ws-row{display:flex;align-items:center;gap:12px}
  .ws-row label{color:var(--muted);font-size:13px;min-width:130px;flex-shrink:0}
  .ws-row input,.ws-row select{flex:1;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:7px 10px;color:var(--text);font-size:14px;font-family:inherit;outline:none}
  .ws-row input:focus,.ws-row select:focus{border-color:var(--accent)}
  .ws-model-item{display:flex;align-items:center;gap:10px;padding:10px 12px;border:1px solid var(--border);border-radius:8px;margin-bottom:8px}
  .ws-model-item .name{font-weight:600;font-size:14px}
  .ws-model-item .detail{font-size:12px;color:var(--muted);flex:1}
  .ws-model-item .del{background:rgba(248,81,73,.15);color:var(--bad);border:none;border-radius:6px;padding:5px 10px;cursor:pointer;font-size:12px}
  .ws-model-item .del:hover{background:rgba(248,81,73,.25)}

  /* ---------- agent ---------- */
  .agent-layout{display:flex;gap:16px;height:100%;padding:16px;overflow:hidden}
  .agent-side{padding:0}
  .agent-side-label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em;padding:10px 14px 4px}
  .ws-switch{display:flex;gap:6px;padding:0 12px 4px}
  .ws-switch select{flex:1;min-width:0;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:7px 8px;color:var(--text);font-size:12px;outline:none}
  .ws-switch select:focus{border-color:var(--accent)}
  .ws-switch button{padding:6px 10px;font-size:13px}
  #agent-task-list{flex:1;overflow-y:auto;padding:0 8px}
  .atk-item{padding:8px 10px;border-radius:6px;cursor:pointer;font-size:12px;margin-bottom:2px;transition:background .15s}
  .atk-item:hover,.atk-item.active{background:var(--panel2)}
  .atk-item .atk-title{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--text)}
  .atk-item .atk-meta{color:var(--muted);font-size:11px;margin-top:2px;display:flex;justify-content:space-between;gap:6px}
  .atk-item .live-badge{color:var(--warn)}
  .agent-main{flex:1;display:flex;flex-direction:column;gap:12px;min-width:0}
  .agent-taskbar{display:flex;align-items:center;gap:14px;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:10px 16px;flex-shrink:0}
  .agent-taskbar .tb-main{flex:1;min-width:0}
  .agent-taskbar .tb-title{font-size:14px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .agent-taskbar .tb-sub{color:var(--muted);font-size:12px;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .agent-taskbar .tb-status{display:flex;align-items:center;gap:8px;font-size:13px;flex-shrink:0}
  .agent-taskbar .tb-status .dot{width:8px;height:8px;border-radius:50%;background:var(--muted);flex-shrink:0}
  .agent-taskbar .tb-status .dot.running{background:var(--warn);animation:agent-pulse 1.2s infinite}
  .agent-taskbar .tb-status .dot.done{background:var(--good)}
  .agent-taskbar .tb-status .dot.error{background:var(--bad)}
  @keyframes agent-pulse{0%,100%{opacity:1}50%{opacity:.3}}
  .agent-taskbar .tb-status .meta{color:var(--muted);font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .agent-taskbar .tb-actions{display:flex;gap:6px;flex-shrink:0}
  .agent-taskbar .mini{padding:5px 12px;font-size:12px}
  .agent-trace{flex:1;overflow-y:auto;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px;font-size:13px}
  .run-summary{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px;padding:6px 10px;margin-bottom:10px;background:var(--panel2);border:1px solid var(--border);border-radius:8px;cursor:pointer;user-select:none}
  .run-summary:hover{color:var(--text)}
  .run-summary .chev{font-size:9px;transition:transform .15s;display:inline-block}
  .run-summary.collapsed .chev{transform:rotate(-90deg)}
  .agent-composer{flex-shrink:0;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:10px 12px}
  .agent-composer textarea{width:100%;box-sizing:border-box;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:9px 12px;color:var(--text);font-size:13px;font-family:inherit;outline:none;min-height:56px;resize:vertical}
  .agent-composer textarea:focus{border-color:var(--accent)}
  .composer-row{display:flex;gap:8px;margin-top:8px;align-items:center;flex-wrap:wrap}
  .composer-row select,.composer-row input{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:7px 9px;color:var(--text);font-size:12px;outline:none}
  .composer-row select:focus,.composer-row input:focus{border-color:var(--accent)}
  .composer-row .ck{display:flex;align-items:center;gap:4px;font-size:12px;color:var(--muted);cursor:pointer;user-select:none}
  .composer-row .ck input{accent-color:var(--accent);cursor:pointer}
  .composer-row #agent-max-steps{width:64px}
  .composer-row #agent-run{margin-left:auto}
  .agent-placeholder{color:var(--muted);text-align:center;padding:60px 24px;line-height:2}
  .step-card{background:var(--panel2);border:1px solid var(--border);border-radius:10px;margin-bottom:12px;overflow:hidden}
  .step-card.collapsed .sec{display:none}
  .step-head{display:flex;align-items:center;gap:9px;padding:9px 14px;background:rgba(255,255,255,.03);border-bottom:1px solid var(--border);font-size:12px;cursor:pointer;user-select:none}
  .step-badge{background:var(--accent);color:#fff;border-radius:10px;padding:1px 9px;font-size:11px;font-weight:600;flex-shrink:0}
  .step-head .tool-name{color:var(--accent);font-weight:600;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .step-head .step-state{margin-left:auto;color:var(--good);font-size:11px}
  .sec{padding:0 14px}
  .sec + .sec{border-top:1px solid var(--border)}
  .sec-toggle{display:flex;align-items:center;gap:8px;padding:8px 0;cursor:pointer;user-select:none;color:var(--muted);font-size:12px}
  .sec-toggle:hover{color:var(--text)}
  .sec-toggle .chev{font-size:9px;transition:transform .15s}
  .sec.collapsed .chev{transform:rotate(-90deg)}
  .sec.collapsed .sec-body{display:none}
  .sec-body{padding:0 0 10px;line-height:1.65;white-space:pre-wrap;word-break:break-word}
  .sec-body.thought{font-style:italic}
  .sec-body.args,.sec-body.obs{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:8px 10px;margin-top:2px}
  .sec-body.obs{max-height:260px;overflow-y:auto}
  .sec-body.errtext{color:var(--bad)}
  .finish-card{background:rgba(63,185,80,.08);border:1px solid var(--good);border-radius:10px;padding:14px 16px;margin-bottom:12px}
  .finish-card .fh{color:var(--good);font-weight:600;font-size:13px;margin-bottom:6px}
  .finish-card .fb{white-space:pre-wrap;word-break:break-word;line-height:1.7}
  .agent-err-card{background:rgba(248,81,73,.07);border:1px solid var(--bad);border-radius:10px;padding:12px 16px;margin-bottom:12px;color:var(--bad);white-space:pre-wrap;word-break:break-word}

  /* ---------- evaluate ---------- */
  .eval-view{padding:20px;overflow-y:auto}
  .eval-card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:16px}
  .eval-card h2{font-size:15px;margin-bottom:14px}
  .eval-row{display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
  .eval-row label{color:var(--muted);font-size:13px;min-width:80px}
  .eval-row select,.eval-row input{min-width:200px}
  .progress-bar{height:6px;background:var(--panel2);border-radius:3px;overflow:hidden;margin:10px 0}
  .progress-fill{height:100%;background:var(--accent);transition:width .3s;width:0}
  .eval-results{margin-top:10px}
  .eval-results table{width:100%;border-collapse:collapse;font-size:13px}
  .eval-results th,.eval-results td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border)}
  .eval-results th{color:var(--muted);font-weight:500}
  .eval-results .err{background:rgba(248,81,73,.08)}
  .metric-box{display:inline-block;background:var(--panel2);border:1px solid var(--border);border-radius:6px;padding:6px 14px;margin:4px 8px 4px 0}
  .metric-box b{color:var(--accent)}

  /* ---------- results ---------- */
  .results-view{padding:20px;overflow-y:auto}
  .results-view select{margin-bottom:12px}
  .results-empty{color:var(--muted);text-align:center;padding:40px}
  .results-view table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}
  .results-view th,.results-view td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--border)}
  .results-view th{color:var(--muted);font-weight:500;font-size:12px;text-transform:uppercase}
  .results-view td.num{text-align:right;font-variant-numeric:tabular-nums}
  .pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:500}
  .pill.good{background:rgba(63,185,80,.15);color:var(--good)}
  .pill.mid{background:rgba(210,153,34,.15);color:var(--warn)}
  .pill.bad{background:rgba(248,81,73,.15);color:var(--bad)}

  .scrollbox{max-height:300px;overflow-y:auto;border:1px solid var(--border);border-radius:8px;margin-top:8px}
  .scrollbox table{font-size:12px}
  .scrollbox td{font-family:ui-monospace,monospace;font-size:11px}
</style>
</head>
<body>

<header>
  <h1>FylHarness</h1>
  <div class="tabs">
    <div class="tab active" data-view="playground">Playground</div>
    <div class="tab" data-view="workspace">Workspace</div>
    <div class="tab" data-view="agent">Agent</div>
    <div class="tab" data-view="evaluate">Evaluate</div>
    <div class="tab" data-view="results">Results</div>
  </div>
  <div class="spacer"></div>
  <div style="font-size:13px;color:var(--muted)"><span class="status-dot"></span><span id="models-count">0 models</span></div>
</header>

<main>
  <!-- =================== PLAYGROUND =================== -->
  <div class="view active" id="view-playground">
    <div class="chat-layout">
      <div class="chat-sidebar">
        <button class="new-chat" id="new-conversation">+ New Conversation</button>
        <div class="conv-list" id="conv-list">
          <div class="conv-empty">No conversations yet.<br>Type below to start.</div>
        </div>
        <div class="sidebar-model-panel">
          <div class="sidebar-label">Model</div>
          <select id="chat-model" class="sidebar-select"></select>
          <div class="sidebar-label" style="margin-top:8px">Reasoning Effort</div>
          <select id="chat-reasoning" class="sidebar-select">
            <option value="">Default</option>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
          <div class="sidebar-label" style="margin-top:8px">Temperature: <span id="temp-val">0.7</span></div>
          <input id="chat-temperature" type="range" min="0" max="2" step="0.1" value="0.7" class="sidebar-slider">
          <div class="sidebar-label" style="margin-top:8px">Max Tokens</div>
          <input id="chat-max-tokens" type="number" min="1" max="32768" value="1024" class="sidebar-select" style="width:100%">
        </div>
      </div>
      <div class="chat-main">
        <div class="chat-messages" id="chat-messages">
          <div style="color:var(--muted);text-align:center;padding:40px 20px" id="chat-placeholder">
            Type a message below to start chatting. Responses stream in real time.
          </div>
        </div>
        <div class="chat-input">
          <textarea id="chat-input" placeholder="Type a message... (Enter to send, Shift+Enter for newline)" rows="1"></textarea>
          <button id="chat-send">Send</button>
        </div>
      </div>
    </div>
  </div>

  <!-- =================== WORKSPACE =================== -->
  <div class="view" id="view-workspace">
    <div class="eval-view">
      <div class="eval-card">
        <h2>Add Model</h2>
        <div class="ws-form">
          <div class="ws-row"><label>Name *</label><input id="ws-name" placeholder="my-qwen"></div>
          <div class="ws-row"><label>Type</label>
            <select id="ws-type"><option value="openai">openai</option><option value="mock">mock</option></select>
          </div>
          <div class="ws-row"><label>Base URL</label><input id="ws-base-url" value="http://127.0.0.1:8000/v1" style="min-width:300px"></div>
          <div class="ws-row"><label>Model ID</label><input id="ws-model" placeholder="Qwen/Qwen2.5-7B-Instruct"></div>
          <div class="ws-row"><label>API Key</label><input id="ws-api-key" type="password" placeholder="sk-... or EMPTY for local"></div>
          <div class="ws-row"><label>Temperature</label><input id="ws-temp" type="number" min="0" max="2" step="0.1" value="0.7" style="width:100px"></div>
          <div class="ws-row"><label>Max Tokens</label><input id="ws-max-tokens" type="number" min="1" value="1024" style="width:100px"></div>
          <div class="ws-row"><label>Reasoning Effort</label>
            <select id="ws-reasoning"><option value="">Default</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select>
          </div>
          <div class="ws-row" style="margin-top:12px"><button id="ws-add">Add Model</button></div>
        </div>
      </div>
      <div class="eval-card">
        <h2>Configured Models</h2>
        <div id="ws-model-list"></div>
      </div>
    </div>
  </div>

  <!-- =================== EVALUATE =================== -->
  <div class="view" id="view-evaluate">
    <div class="eval-view">
      <div class="eval-card">
        <h2>Run Evaluation</h2>
        <div class="eval-row">
          <label>Model:</label>
          <select id="eval-model"></select>
        </div>
        <div class="eval-row">
          <label>Task:</label>
          <select id="eval-task"></select>
        </div>
        <div class="eval-row">
          <label>Sample limit:</label>
          <input id="eval-limit" type="number" min="1" placeholder="all" style="width:120px">
          <button id="eval-run" style="margin-left:auto">Run</button>
        </div>
      </div>
      <div class="eval-card" id="eval-status-card" style="display:none">
        <h2 id="eval-status-title">Running...</h2>
        <div class="progress-bar"><div class="progress-fill" id="eval-progress"></div></div>
        <div id="eval-metrics" style="margin-top:8px"></div>
        <div class="eval-results" id="eval-results"></div>
      </div>
    </div>
  </div>

  <!-- =================== AGENT =================== -->
  <div class="view" id="view-agent">
    <div class="agent-layout">
      <div class="chat-sidebar agent-side">
        <button class="new-chat" id="agent-new">＋ 新建任务</button>
        <div class="agent-side-label">工作区</div>
        <div class="ws-switch">
          <select id="agent-workspace-select"></select>
          <button id="agent-ws-add" title="添加工作区目录">＋</button>
        </div>
        <div class="agent-side-label">任务历史</div>
        <div id="agent-task-list">
          <div class="conv-empty">暂无任务<br>点击上方新建</div>
        </div>
      </div>
      <div class="agent-main">
        <div class="agent-taskbar">
          <div class="tb-main">
            <div class="tb-title" id="agent-tb-title">新任务</div>
            <div class="tb-sub" id="agent-tb-sub">描述任务后点击运行，每一步的思考、执行与观察实时可见</div>
          </div>
          <div class="tb-status" id="agent-tb-status" style="display:none">
            <span class="dot" id="agent-status-dot"></span>
            <span id="agent-status-text"></span>
            <span class="meta" id="agent-status-meta"></span>
          </div>
          <div class="tb-actions">
            <button id="agent-collapse-all" class="mini" title="收起执行过程">收起过程</button>
            <button id="agent-expand-all" class="mini" title="展开执行过程">展开过程</button>
          </div>
        </div>
        <div class="agent-trace" id="agent-trace">
          <div id="agent-run-summary" class="run-summary" style="display:none">
            <span class="chev">▼</span><span id="agent-run-summary-text"></span>
          </div>
          <div id="agent-steps">
            <div class="agent-placeholder">
              在下方输入任务描述，点击 <b>▶ 运行</b> 开始。<br>
              Agent 会自主探索文件、编写代码、执行命令来完成任务，<br>
              每一步的思考过程、执行步骤与观察结果都会实时展示，可随时展开或收起。
            </div>
          </div>
        </div>
        <div class="agent-composer">
          <textarea id="agent-task" placeholder="描述任务… 例如：阅读 note.txt 并完成其中的要求"></textarea>
          <div class="composer-row">
            <select id="agent-model" title="模型"></select>
            <select id="agent-reasoning" title="推理强度">
              <option value="">推理强度：默认</option>
              <option value="low">推理强度：低</option>
              <option value="medium">推理强度：中</option>
              <option value="high">推理强度：高</option>
            </select>
            <label class="ck" title="运行前先生成执行计划（Plan-and-Execute）"><input type="checkbox" id="agent-planning"> 规划</label>
            <label class="ck" title="定期自我复盘并将结论注入后续步骤（Reflection）"><input type="checkbox" id="agent-reflection"> 反思</label>
            <input id="agent-max-steps" type="number" min="1" max="50" value="25" title="最大步数">
            <button id="agent-run">▶ 运行</button>
            <button id="agent-stop" class="secondary" disabled>停止</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- =================== RESULTS =================== -->
  <div class="view" id="view-results">
    <div class="results-view">
      <h2 style="font-size:15px;margin-bottom:12px">Past Results</h2>
      <select id="results-select"><option value="">-- select a run --</option></select>
      <div id="results-content"></div>
    </div>
  </div>
</main>

<script>
let MODELS = [], TASKS = [];
let chatHistory = [];
let evalPollTimer = null;

// ---------- tab switching ----------
document.querySelectorAll('.tab').forEach(t => {
  t.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.view').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('view-' + t.dataset.view).classList.add('active');
    if (t.dataset.view === 'results') loadResultsList();
    if (t.dataset.view === 'workspace') renderWorkspaceModels();
  };
});

// ---------- model selector population (reusable) ----------
function populateModelSelectors() {
  const cm = document.getElementById('chat-model');
  const em = document.getElementById('eval-model');
  const am = document.getElementById('agent-model');
  const prevChat = cm.value;
  const prevEval = em.value;
  const prevAgent = am ? am.value : '';
  [cm, em, am].filter(Boolean).forEach(sel => {
    sel.innerHTML = '';
    MODELS.forEach(m => {
      const o = document.createElement('option');
      o.value = m.name; o.textContent = `${m.name} (${m.model})`;
      sel.appendChild(o);
    });
  });
  if (prevChat && MODELS.find(m => m.name === prevChat)) cm.value = prevChat;
  else if (MODELS.length) cm.value = MODELS[0].name;
  if (prevEval && MODELS.find(m => m.name === prevEval)) em.value = prevEval;
  else if (MODELS.length) em.value = MODELS[0].name;
  if (am) {
    if (prevAgent && MODELS.find(m => m.name === prevAgent)) am.value = prevAgent;
    else if (MODELS.length) am.value = MODELS[0].name;
  }
  document.getElementById('models-count').textContent = MODELS.length + ' models';
}

// ---------- init: load config ----------
fetch('/api/config').then(r => r.json()).then(cfg => {
  MODELS = cfg.models || [];
  TASKS = cfg.tasks || [];
  populateModelSelectors();
  const et = document.getElementById('eval-task');
  TASKS.forEach(t => {
    const o = document.createElement('option');
    o.value = t.name; o.textContent = `${t.name} [${t.type}]`;
    et.appendChild(o);
  });
  if (TASKS.length) et.value = TASKS[0].name;
});

// ---------- sidebar reasoning controls ----------
document.getElementById('chat-temperature').oninput = function() {
  document.getElementById('temp-val').textContent = parseFloat(this.value).toFixed(1);
};

// ---------- workspace: model management ----------
function renderWorkspaceModels() {
  fetch('/api/models').then(r => r.json()).then(data => {
    const list = document.getElementById('ws-model-list');
    const models = data.models || [];
    if (!models.length) {
      list.innerHTML = '<div style="color:var(--muted);padding:10px">No models configured.</div>';
      return;
    }
    list.innerHTML = models.map(m => {
      const re = m.reasoning_effort ? ` | effort=${m.reasoning_effort}` : '';
      return `<div class="ws-model-item">
        <div style="flex:1">
          <div class="name">${esc(m.name)}</div>
          <div class="detail">${esc(m.type)} · ${esc(m.base_url)} · ${esc(m.model)}${re} · temp=${m.temperature} · maxtok=${m.max_tokens}${m.has_key ? '' : ' · no key'}</div>
        </div>
        <button class="del" data-name="${esc(m.name)}">Delete</button>
      </div>`;
    }).join('');
    list.querySelectorAll('.del').forEach(btn => {
      btn.onclick = () => deleteModel(btn.dataset.name);
    });
  });
}

document.getElementById('ws-add').onclick = function() {
  const spec = {
    name: document.getElementById('ws-name').value.trim(),
    type: document.getElementById('ws-type').value,
    base_url: document.getElementById('ws-base-url').value.trim(),
    model: document.getElementById('ws-model').value.trim(),
    api_key: document.getElementById('ws-api-key').value,
    temperature: parseFloat(document.getElementById('ws-temp').value),
    max_tokens: parseInt(document.getElementById('ws-max-tokens').value),
    reasoning_effort: document.getElementById('ws-reasoning').value,
  };
  if (!spec.name) { alert('Name is required'); return; }
  fetch('/api/models', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(spec),
  }).then(r => r.json()).then(data => {
    if (data.error) { alert(data.error); return; }
    // Update local MODELS and refresh all selectors.
    MODELS.push(data);
    populateModelSelectors();
    renderWorkspaceModels();
    // Clear form.
    document.getElementById('ws-name').value = '';
    document.getElementById('ws-api-key').value = '';
  }).catch(err => alert('Failed: ' + err));
};

function deleteModel(name) {
  if (!confirm(`Delete model '${name}'?`)) return;
  fetch('/api/models/' + encodeURIComponent(name), { method: 'DELETE' })
    .then(r => r.json())
    .then(data => {
      if (data.error) { alert(data.error); return; }
      MODELS = MODELS.filter(m => m.name !== name);
      populateModelSelectors();
      renderWorkspaceModels();
    });
}

// ---------- agent ----------
let agentRunId = null;          // run id of the streaming task
let agentStreamDone = true;
let agentActiveKey = null;      // '__new__' | '__live__' (stream, id pending) | runId
const agentTraceEl = document.getElementById('agent-trace');
const agentStepsEl = document.getElementById('agent-steps');
const agentRunSummaryEl = document.getElementById('agent-run-summary');
const AGENT_PLACEHOLDER = '<div class="agent-placeholder">在下方输入任务描述，点击 <b>▶ 运行</b> 开始。<br>' +
  'Agent 会自主探索文件、编写代码、执行命令来完成任务，<br>' +
  '每一步的思考过程、执行步骤与观察结果都会实时展示，可随时展开或收起。</div>';
let agentCurCard = null;      // step card currently receiving sections
let agentCurStep = null;      // step number of agentCurCard
let agentStepCount = 0;       // tool calls performed
let agentThoughtCount = 0;    // thought sections rendered
let agentTotalSteps = 0;
let agentRunning = false;     // the streaming task is still running
let agentTimer = null;
let agentStartTime = null;
let agentTasks = {};          // runId -> {id,title,workspace,model,status,created,events}
let agentWorkspaces = [];     // saved workspace directories
let agentPendingTitle = '';   // task text captured at submit (textarea is cleared)

function agentScroll() {
  requestAnimationFrame(() => { agentTraceEl.scrollTop = agentTraceEl.scrollHeight; });
}

function agentSetStatus(state, text) {
  document.getElementById('agent-tb-status').style.display = 'flex';
  document.getElementById('agent-status-dot').className = 'dot ' + state;
  document.getElementById('agent-status-text').textContent = text;
}

function agentSetStatusHidden() {
  document.getElementById('agent-tb-status').style.display = 'none';
}

function agentSetMeta(text) {
  document.getElementById('agent-status-meta').textContent = text || '';
}

function agentSetComposerEnabled(enabled) {
  ['agent-model', 'agent-reasoning', 'agent-max-steps', 'agent-planning', 'agent-reflection']
    .forEach(id => {
      const el = document.getElementById(id);
      if (el) el.disabled = !enabled;
    });
}

function agentClearRender() {
  agentStepsEl.innerHTML = '';
  agentCurCard = null;
  agentCurStep = null;
  agentStepCount = 0;
  agentThoughtCount = 0;
  agentTotalSteps = 0;
  agentRunSummaryEl.style.display = 'none';
  agentRunSummaryEl.classList.remove('collapsed');
  agentStepsEl.style.display = '';
}

function agentShowPlaceholder() {
  agentStepsEl.innerHTML = AGENT_PLACEHOLDER;
}

function agentViewNew() {
  agentActiveKey = '__new__';
  agentClearRender();
  agentShowPlaceholder();
  agentSetStatusHidden();
  agentSetMeta('');
  document.getElementById('agent-tb-title').textContent = '新任务';
  document.getElementById('agent-tb-sub').textContent = '描述任务后点击运行，每一步的思考、执行与观察实时可见';
  agentRenderTaskList();
}

function agentViewTask(runId) {
  const t = agentTasks[runId];
  if (!t) return;
  agentActiveKey = runId;
  agentClearRender();
  t.events.forEach(evt => renderAgentEvent(evt));
  if (t.status === 'running') {
    agentSetStatus('running', 'Agent 运行中…');
    agentSetMeta(`step ${agentStepCount}/${agentTotalSteps || '?'}`);
  } else {
    const map = {done: ['done', '✅ 任务完成'], stopped: ['error', '⏹ 已停止'], error: ['error', '✕ 出错']};
    const m = map[t.status] || ['done', '已结束'];
    agentSetStatus(m[0], m[1]);
    agentSetMeta(`${agentStepCount} 次工具调用`);
  }
  document.getElementById('agent-tb-title').textContent = t.title;
  document.getElementById('agent-tb-sub').textContent = (t.workspace || '') + (t.model ? ' · ' + t.model : '');
  agentRenderTaskList();
  agentScroll();
}

function agentUpdateSummary() {
  if (agentActiveKey === '__new__' || (!agentStepCount && !agentThoughtCount)) {
    agentRunSummaryEl.style.display = 'none';
    return;
  }
  agentRunSummaryEl.style.display = 'flex';
  document.getElementById('agent-run-summary-text').textContent =
    `${agentStepCount} 次工具调用 · ${agentThoughtCount} 步思考 · 点击收起/展开过程`;
}

function agentNewStepCard(step) {
  const card = document.createElement('div');
  card.className = 'step-card';
  card.innerHTML = `<div class="step-head"><span class="step-badge">Step ${esc(String(step))}</span>` +
    `<span class="tool-name"></span><span class="step-state"></span></div>`;
  card.querySelector('.step-head').onclick = () => card.classList.toggle('collapsed');
  agentStepsEl.appendChild(card);
  agentScroll();
  return card;
}

function agentEnsureCard(step) {
  if (!agentCurCard || agentCurStep !== step) {
    agentCurCard = agentNewStepCard(step);
    agentCurStep = step;
  }
  return agentCurCard;
}

function agentAddSection(card, icon, label, bodyClass, text, collapsed) {
  const sec = document.createElement('div');
  sec.className = 'sec' + (collapsed ? ' collapsed' : '');
  const toggle = document.createElement('div');
  toggle.className = 'sec-toggle';
  toggle.innerHTML = `<span class="chev">▼</span><span>${icon} ${esc(label)}</span>`;
  const body = document.createElement('div');
  body.className = 'sec-body ' + bodyClass;
  body.textContent = text;
  toggle.onclick = () => sec.classList.toggle('collapsed');
  sec.appendChild(toggle);
  sec.appendChild(body);
  card.appendChild(sec);
  agentScroll();
  return sec;
}

function agentRelTime(ms) {
  const d = Date.now() - ms;
  if (d < 60000) return '刚刚';
  if (d < 3600000) return Math.floor(d / 60000) + ' 分钟前';
  if (d < 86400000) return Math.floor(d / 3600000) + ' 小时前';
  return Math.floor(d / 86400000) + ' 天前';
}

function agentStatusBadge(t) {
  if (t.status === 'running') return '<span class="live-badge">● 运行中</span>';
  if (t.status === 'error') return '<span style="color:var(--bad)">✕ 出错</span>';
  if (t.status === 'stopped') return '<span>⏹ 已停止</span>';
  return '<span style="color:var(--good)">✓ 完成</span>';
}

function agentRenderTaskList() {
  const list = document.getElementById('agent-task-list');
  const ids = Object.keys(agentTasks).sort((a, b) => agentTasks[b].created - agentTasks[a].created);
  if (!ids.length) {
    list.innerHTML = '<div class="conv-empty">暂无任务<br>点击上方新建</div>';
    return;
  }
  list.innerHTML = '';
  ids.forEach(id => {
    const t = agentTasks[id];
    const item = document.createElement('div');
    item.className = 'atk-item' + (agentActiveKey === id ? ' active' : '');
    item.innerHTML = `<div class="atk-title">${esc(t.title)}</div>` +
      `<div class="atk-meta"><span>${esc(agentRelTime(t.created))}</span>${agentStatusBadge(t)}</div>`;
    item.onclick = () => agentViewTask(id);
    list.appendChild(item);
  });
}

// ------------------- workspaces -------------------
const AGENT_WS_KEY = 'fylharness_agent_workspaces';

function agentLoadWorkspaces() {
  try {
    agentWorkspaces = JSON.parse(localStorage.getItem(AGENT_WS_KEY) || '[]');
  } catch (e) { agentWorkspaces = []; }
  if (!Array.isArray(agentWorkspaces)) agentWorkspaces = [];
  agentRenderWorkspaceSelect();
}

function agentSaveWorkspaces() {
  localStorage.setItem(AGENT_WS_KEY, JSON.stringify(agentWorkspaces));
}

function agentRenderWorkspaceSelect() {
  const sel = document.getElementById('agent-workspace-select');
  const cur = sel.value;
  sel.innerHTML = '';
  if (!agentWorkspaces.length) {
    const o = document.createElement('option');
    o.value = '';
    o.textContent = '选择工作区…';
    sel.appendChild(o);
  }
  agentWorkspaces.forEach(w => {
    const o = document.createElement('option');
    o.value = w;
    o.textContent = w;
    sel.appendChild(o);
  });
  if (cur && agentWorkspaces.includes(cur)) sel.value = cur;
}

function agentAddWorkspace(path) {
  path = (path || '').trim();
  if (!path) return;
  if (!agentWorkspaces.includes(path)) {
    agentWorkspaces.unshift(path);
    agentSaveWorkspaces();
  }
  agentRenderWorkspaceSelect();
  document.getElementById('agent-workspace-select').value = path;
}

// ------------------- history -------------------
function agentLoadHistory() {
  fetch('/api/agent/runs').then(r => r.json()).then(data => {
    (data.runs || []).forEach(r => {
      if (agentTasks[r.id]) return;  // keep live/cached versions
      agentTasks[r.id] = {
        id: r.id,
        title: (r.task_desc || 'Agent 任务').slice(0, 50),
        workspace: r.workspace || '',
        model: r.model || '',
        status: r.status === 'error' ? 'error' : 'done',
        created: Date.parse(r.created_at) || Date.now(),
        events: r.events || [],
      };
      const w = (r.workspace || '').trim();
      if (w && !agentWorkspaces.includes(w)) agentWorkspaces.push(w);
    });
    agentRenderWorkspaceSelect();
    agentRenderTaskList();
  }).catch(() => {});
}

document.getElementById('agent-run').onclick = function() {
  if (!agentStreamDone) return;
  const model = document.getElementById('agent-model').value;
  const workspace = document.getElementById('agent-workspace-select').value;
  const task = document.getElementById('agent-task').value.trim();
  const maxSteps = parseInt(document.getElementById('agent-max-steps').value);
  const reasoning = document.getElementById('agent-reasoning').value;
  if (!model) { alert('请选择模型'); return; }
  if (!workspace) { alert('请先在左侧选择或添加工作区目录'); return; }
  if (!task) { alert('请输入任务描述'); return; }
  agentAddWorkspace(workspace);
  // fresh view for the incoming live task
  agentActiveKey = '__live__';
  agentClearRender();
  agentRunning = true;
  agentStreamDone = false;
  agentRunId = null;
  agentPendingTitle = task.slice(0, 50);
  this.disabled = true;
  document.getElementById('agent-stop').disabled = false;
  agentSetComposerEnabled(false);
  // The task moves to the top of the window; the composer is cleared.
  document.getElementById('agent-task').value = '';
  document.getElementById('agent-tb-title').textContent = agentPendingTitle;
  document.getElementById('agent-tb-sub').textContent = workspace + ' · ' + model;
  agentSetStatus('running', 'Agent 运行中…');
  agentSetMeta('');
  agentStartTime = Date.now();
  if (agentTimer) clearInterval(agentTimer);
  agentTimer = setInterval(() => {
    if (!agentRunning || agentActiveKey !== agentRunId) return;
    const s = Math.floor((Date.now() - agentStartTime) / 1000);
    const mm = String(Math.floor(s / 60)).padStart(2, '0');
    const ss = String(s % 60).padStart(2, '0');
    agentSetMeta(`step ${agentStepCount}/${agentTotalSteps || '?'} · ${mm}:${ss}`);
  }, 1000);

  const body = { model, workspace, task, max_steps: maxSteps };
  if (reasoning) body.reasoning_effort = reasoning;
  body.planning = document.getElementById('agent-planning').checked;
  body.reflection = document.getElementById('agent-reflection').checked;
  fetch('/api/agent', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  }).then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    function read() {
      reader.read().then(({done, value}) => {
        if (done) {
          agentStreamDone = true;
          if (agentRunning) {
            agentRunning = false;
            agentStopTimer();
            agentSetStatus('done', '已结束');
          }
          document.getElementById('agent-run').disabled = false;
          document.getElementById('agent-stop').disabled = true;
          agentSetComposerEnabled(true);
          agentRenderTaskList();
          return;
        }
        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6);
          if (payload === '[DONE]') {
            agentStreamDone = true;
            if (agentRunning) {
              agentRunning = false;
              agentStopTimer();
              agentSetStatus('done', '已结束');
            }
            document.getElementById('agent-run').disabled = false;
            document.getElementById('agent-stop').disabled = true;
            agentSetComposerEnabled(true);
            agentRenderTaskList();
            return;
          }
          try {
            const evt = JSON.parse(payload);
            agentHandleStreamEvent(evt);
          } catch(e) {}
        }
        read();
      });
    }
    read();
  }).catch(err => {
    appendAgentError('请求失败: ' + err);
    agentStreamDone = true;
    if (agentRunning) {
      agentRunning = false;
      agentStopTimer();
      agentSetStatus('error', '✕ 连接失败');
    }
    if (agentRunId && agentTasks[agentRunId]) agentTasks[agentRunId].status = 'error';
    document.getElementById('agent-run').disabled = false;
    document.getElementById('agent-stop').disabled = true;
    agentSetComposerEnabled(true);
    agentRenderTaskList();
  });
};

// Register, persist and route one streamed event. Rendering only happens
// when the affected task is the one being viewed.
function agentHandleStreamEvent(evt) {
  if (evt.run_id && !agentRunId) {
    agentRunId = evt.run_id;
    agentTasks[agentRunId] = {
      id: agentRunId,
      title: agentPendingTitle || 'Agent 任务',
      workspace: document.getElementById('agent-workspace-select').value,
      model: document.getElementById('agent-model').value,
      status: 'running',
      created: Date.now(),
      events: [],
    };
    if (agentActiveKey === '__live__') agentActiveKey = agentRunId;
    document.getElementById('agent-tb-title').textContent = agentTasks[agentRunId].title;
    document.getElementById('agent-tb-sub').textContent =
      agentTasks[agentRunId].workspace + ' · ' + agentTasks[agentRunId].model;
    agentRenderTaskList();
  }
  const t = agentRunId ? agentTasks[agentRunId] : null;
  if (t) {
    t.events.push(evt);
    if (evt.type === 'finish') t.status = 'done';
    if (evt.type === 'end') {
      t.status = String(evt.reason || '').includes('stopped') ? 'stopped' : 'done';
    }
  }
  const viewing = agentActiveKey === agentRunId || agentActiveKey === '__live__';
  if (viewing) renderAgentEvent(evt);
  if (viewing && evt.type === 'finish' && agentRunning) {
    agentRunning = false;
    agentStopTimer();
    agentSetStatus('done', '✅ 任务完成');
    agentSetMeta(`共 ${agentStepCount} 次工具调用`);
    agentRenderTaskList();
  }
  if (viewing && evt.type === 'end' && agentRunning) {
    const stopped = String(evt.reason || '').includes('stopped');
    agentRunning = false;
    agentStopTimer();
    agentSetStatus(stopped ? 'error' : 'done', stopped ? '⏹ 已停止' : '已结束');
    agentSetMeta(evt.reason || '');
    agentRenderTaskList();
  }
}

document.getElementById('agent-stop').onclick = function() {
  if (agentRunId) fetch('/api/agent/' + agentRunId + '/stop', {method: 'POST'});
  this.disabled = true;
};

function renderAgentEvent(evt) {
  switch (evt.type) {
    case 'start': {
      agentTotalSteps = evt.max_steps || 0;
      const d = document.createElement('div');
      d.className = 'step-card';
      d.innerHTML = `<div class="step-head"><span class="step-badge">任务</span>` +
        `<span class="step-state">max ${esc(String(evt.max_steps))} steps</span></div>` +
        `<div class="sec"><div class="sec-toggle" style="cursor:default"><span>📋 任务描述</span></div><div class="sec-body"></div></div>` +
        `<div class="sec"><div class="sec-toggle" style="cursor:default"><span>📁 工作区</span></div><div class="sec-body"></div></div>`;
      const bodies = d.querySelectorAll('.sec-body');
      bodies[0].textContent = evt.task || '';
      bodies[1].textContent = evt.workspace || '';
      d.querySelector('.step-head').onclick = () => d.classList.toggle('collapsed');
      agentStepsEl.appendChild(d);
      agentScroll();
      break;
    }
    case 'plan': {
      const d = document.createElement('div');
      d.className = 'step-card';
      d.innerHTML = `<div class="step-head"><span class="step-badge">计划</span>` +
        `<span class="step-state">${(evt.plan || []).length} 步</span></div>` +
        `<div class="sec"><div class="sec-toggle" style="cursor:default"><span>🗺️ 执行计划</span></div><div class="sec-body"></div></div>`;
      d.querySelector('.sec-body').textContent =
        (evt.plan && evt.plan.length) ? evt.plan.join('\n') : '（模型未返回有效计划，直接按 ReAct 执行）';
      d.querySelector('.step-head').onclick = () => d.classList.toggle('collapsed');
      agentStepsEl.appendChild(d);
      agentScroll();
      break;
    }
    case 'reflection': {
      const card = agentEnsureCard(evt.step);
      agentAddSection(card, '🪞', '反思', 'thought', evt.text || '');
      break;
    }
    case 'compress':
      break;  // silent bookkeeping; history stays bounded
    case 'thought': {
      agentThoughtCount++;
      const card = agentEnsureCard(evt.step);
      agentAddSection(card, '🧠', '思考过程', 'thought', evt.text || '(空)');
      break;
    }
    case 'tool_call': {
      agentStepCount++;
      const card = agentEnsureCard(evt.step);
      card.querySelector('.tool-name').textContent = evt.tool || '';
      agentAddSection(card, '⚡', '执行步骤 · ' + (evt.tool || ''), 'args',
                      JSON.stringify(evt.args || {}, null, 2));
      break;
    }
    case 'tool_result': {
      const card = agentEnsureCard(evt.step);
      const result = evt.result || '';
      const long = result.length > 600;
      agentAddSection(card, '👁', '观察结果' + (long ? `（${result.length} 字符，点击展开）` : ''),
                      'obs', result, long);
      card.querySelector('.step-state').textContent = '✓';
      break;
    }
    case 'error': {
      const card = agentEnsureCard(evt.step);
      agentAddSection(card, '❌', '出错', 'errtext', evt.error || '');
      break;
    }
    case 'finish': {
      const d = document.createElement('div');
      d.className = 'finish-card';
      d.innerHTML = `<div class="fh">✅ 任务完成 · Step ${esc(String(evt.step || ''))}</div><div class="fb"></div>`;
      d.querySelector('.fb').textContent = evt.summary || '';
      agentStepsEl.appendChild(d);
      agentScroll();
      break;
    }
    case 'end':
      break;  // reflected in the taskbar status by the stream handler
    default: {
      const card = agentCurCard || agentNewStepCard(evt.step || '?');
      agentAddSection(card, '•', evt.type || 'event', 'obs', JSON.stringify(evt));
    }
  }
  agentUpdateSummary();
}

// ------------------- agent UI wiring -------------------
document.getElementById('agent-new').onclick = agentViewNew;
document.getElementById('agent-ws-add').onclick = function() {
  const p = prompt('输入工作区目录路径，例如 F:\\test');
  if (p) agentAddWorkspace(p);
};
agentRunSummaryEl.onclick = () => {
  const collapsed = agentRunSummaryEl.classList.toggle('collapsed');
  agentStepsEl.style.display = collapsed ? 'none' : '';
};
document.getElementById('agent-collapse-all').onclick = () => {
  if (!agentStepCount && !agentThoughtCount) return;
  agentRunSummaryEl.style.display = 'flex';
  agentRunSummaryEl.classList.add('collapsed');
  agentStepsEl.style.display = 'none';
};
document.getElementById('agent-expand-all').onclick = () => {
  agentRunSummaryEl.classList.remove('collapsed');
  agentStepsEl.style.display = '';
  agentStepsEl.querySelectorAll('.step-card').forEach(c => c.classList.remove('collapsed'));
};
document.getElementById('agent-task').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (agentStreamDone) document.getElementById('agent-run').click();
  }
});
function agentStopTimer() {
  if (agentTimer) { clearInterval(agentTimer); agentTimer = null; }
}

// init
agentLoadWorkspaces();
agentViewNew();
agentLoadHistory();
fetch('/api/agent/settings').then(r => r.json()).then(s => {
  document.getElementById('agent-planning').checked = !!s.planning;
  document.getElementById('agent-reflection').checked = !!s.reflection;
}).catch(() => {});

function appendAgentError(msg) {
  const d = document.createElement('div');
  d.className = 'agent-err-card';
  d.textContent = '❌ ' + msg;
  agentStepsEl.appendChild(d);
  agentScroll();
}

// ---------- playground: conversation management ----------
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const convList = document.getElementById('conv-list');

// Conversations are stored in memory: [{id, title, messages: [...], active}]
let conversations = [];
let activeConvId = null;
let chatStreaming = false;

// Scroll the chat container to the bottom after the browser paints new content.
// requestAnimationFrame guarantees the DOM has reflowed so scrollHeight is current.
function scrollChatToBottom() {
  requestAnimationFrame(() => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  });
}

function renderConvList() {
  if (!conversations.length) {
    convList.innerHTML = '<div class="conv-empty">No conversations yet.<br>Type below to start.</div>';
    return;
  }
  convList.innerHTML = '';
  conversations.forEach(c => {
    const item = document.createElement('div');
    item.className = 'conv-item' + (c.id === activeConvId ? ' active' : '');
    const title = c.title || 'New chat';
    item.innerHTML = `<span>${esc(title)}</span><span class="conv-del" data-id="${c.id}">&#x2715;</span>`;
    item.onclick = (e) => {
      if (e.target.classList.contains('conv-del')) {
        e.stopPropagation();
        deleteConversation(c.id);
      } else {
        switchConversation(c.id);
      }
    };
    convList.appendChild(item);
  });
}

function renderMessages(conv) {
  chatMessages.innerHTML = '';
  if (!conv || !conv.messages.length) {
    chatMessages.innerHTML = '<div style="color:var(--muted);text-align:center;padding:40px 20px">Send a message to start the conversation.</div>';
    return;
  }
  conv.messages.forEach(m => appendMessage(m.role, m.content, false));
  scrollChatToBottom();
}

function appendMessage(role, text, scroll = true) {
  // Remove placeholder if present
  const ph = document.getElementById('chat-placeholder');
  if (ph) ph.remove();
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = `<div class="msg-role">${role}</div><div class="msg-bubble"></div>`;
  chatMessages.appendChild(div);
  const bubble = div.querySelector('.msg-bubble');
  bubble.textContent = text;
  if (scroll) scrollChatToBottom();
  return bubble;
}

function newConversation() {
  const id = 'conv-' + Date.now();
  const conv = { id, title: 'New chat', messages: [], model: document.getElementById('chat-model').value };
  conversations.unshift(conv);
  activeConvId = id;
  renderConvList();
  renderMessages(conv);
  enableInput(true);
  chatInput.focus();
}

function switchConversation(id) {
  const conv = conversations.find(c => c.id === id);
  if (!conv) return;
  activeConvId = id;
  if (conv.model) document.getElementById('chat-model').value = conv.model;
  renderConvList();
  renderMessages(conv);
  enableInput(!chatStreaming);
}

function deleteConversation(id) {
  conversations = conversations.filter(c => c.id !== id);
  if (activeConvId === id) {
    activeConvId = conversations.length ? conversations[0].id : null;
    if (activeConvId) {
      renderMessages(conversations.find(c => c.id === activeConvId));
    } else {
      chatMessages.innerHTML = '<div style="color:var(--muted);text-align:center;padding:40px 20px" id="chat-placeholder">Type a message below to start chatting. Responses stream in real time.</div>';
      enableInput(true);
      chatInput.focus();
    }
  }
  renderConvList();
}

function enableInput(enabled) {
  chatInput.disabled = !enabled;
  chatSend.disabled = !enabled;
}

function getActiveConv() {
  return conversations.find(c => c.id === activeConvId);
}

document.getElementById('new-conversation').onclick = newConversation;
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
chatSend.onclick = sendChat;

function sendChat() {
  if (chatStreaming) return;
  const text = chatInput.value.trim();
  if (!text) return;
  const conv = getActiveConv();
  if (!conv) { newConversation(); return sendChat(); }
  const model = document.getElementById('chat-model').value;
  if (!model) { alert('No model selected'); return; }
  conv.model = model;
  chatInput.value = '';
  conv.messages.push({ role: 'user', content: text });
  if (conv.title === 'New chat') conv.title = text.slice(0, 40);
  renderConvList();
  appendMessage('user', text);
  // assistant bubble (empty, will fill during streaming)
  const bubble = appendMessage('assistant', '');
  chatStreaming = true;
  enableInput(false);
  let firstChunk = true;
  let fullText = '';

  // Gather per-request inference params from the sidebar.
  const reasoningEffort = document.getElementById('chat-reasoning').value;
  const temperature = parseFloat(document.getElementById('chat-temperature').value);
  const maxTokens = parseInt(document.getElementById('chat-max-tokens').value);

  const messagesPayload = conv.messages.map(m => ({ role: m.role, content: m.content }));
  const chatBody = { model, messages: messagesPayload };
  if (reasoningEffort) chatBody.reasoning_effort = reasoningEffort;
  if (!isNaN(temperature)) chatBody.temperature = temperature;
  if (!isNaN(maxTokens)) chatBody.max_tokens = maxTokens;
  fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(chatBody),
  }).then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    function read() {
      reader.read().then(({ done, value }) => {
        if (done) {
          chatStreaming = false;
          enableInput(true);
          chatInput.focus();
          // Persist the full streamed response into conversation history so
          // subsequent turns include it (fixes multi-turn context loss).
          conv.messages.push({ role: 'assistant', content: fullText });
          return;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6);
          if (payload === '[DONE]') {
            chatStreaming = false;
            enableInput(true);
            chatInput.focus();
            conv.messages.push({ role: 'assistant', content: fullText });
            return;
          }
          try {
            const chunk = JSON.parse(payload);
            if (chunk.startsWith && chunk.startsWith('[ERROR]') && firstChunk) {
              bubble.textContent = chunk;
              fullText = chunk;
            } else {
              if (firstChunk) { bubble.textContent = ''; firstChunk = false; }
              bubble.textContent += chunk;
              fullText += chunk;
            }
            // Scroll on every chunk so the user sees text appear live.
            scrollChatToBottom();
          } catch (e) {}
        }
        read();
      });
    }
    read();
  }).catch(err => {
    bubble.textContent = '[ERROR] ' + err;
    fullText = '[ERROR] ' + err;
    conv.messages.push({ role: 'assistant', content: fullText });
    chatStreaming = false;
    enableInput(true);
  });
}

// ---------- evaluate ----------
document.getElementById('eval-run').onclick = startEval;

function startEval() {
  const model = document.getElementById('eval-model').value;
  const task = document.getElementById('eval-task').value;
  const limit = document.getElementById('eval-limit').value;
  if (!model || !task) { alert('Select model and task'); return; }
  document.getElementById('eval-run').disabled = true;
  document.getElementById('eval-status-card').style.display = 'block';
  document.getElementById('eval-status-title').textContent = 'Starting...';
  document.getElementById('eval-progress').style.width = '0';
  document.getElementById('eval-metrics').innerHTML = '';
  document.getElementById('eval-results').innerHTML = '';

  fetch('/api/evaluate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({model, task, limit: limit ? parseInt(limit) : null}),
  }).then(r => r.json()).then(data => {
    if (data.error) { alert(data.error); document.getElementById('eval-run').disabled = false; return; }
    pollEval(data.run_id);
  }).catch(err => {
    alert('Request failed: ' + err);
    document.getElementById('eval-run').disabled = false;
  });
}

function pollEval(runId) {
  if (evalPollTimer) clearInterval(evalPollTimer);
  evalPollTimer = setInterval(() => {
    fetch('/api/runs/' + runId).then(r => r.json()).then(run => {
      const title = document.getElementById('eval-status-title');
      const bar = document.getElementById('eval-progress');
      const total = run.total || 1;
      const pct = (run.done / total) * 100;
      bar.style.width = pct + '%';
      title.textContent = `${run.status} — ${run.done}/${run.total} samples`;
      if (run.metrics) {
        const mDiv = document.getElementById('eval-metrics');
        mDiv.innerHTML = Object.entries(flatten(run.metrics)).map(([k, v]) =>
          `<div class="metric-box"><b>${k}</b>: ${typeof v === 'number' ? v.toFixed(4) : v}</div>`
        ).join('');
      }
      if (run.results && run.results.length) {
        const tbl = document.getElementById('eval-results');
        const shown = run.results.slice(-5); // last 5
        tbl.innerHTML = `<table><thead><tr><th>#</th><th>Prediction</th><th>Reference</th><th>Err</th></tr></thead><tbody>` +
          shown.map(r => `<tr class="${r.error ? 'err' : ''}"><td>${r.index}</td><td>${esc(r.prediction)}</td><td>${esc(r.reference)}</td><td>${r.error ? '✗' : ''}</td></tr>`).join('') +
          `</tbody></table>`;
      }
      if (run.status === 'done' || run.status === 'error') {
        clearInterval(evalPollTimer);
        evalPollTimer = null;
        document.getElementById('eval-run').disabled = false;
        if (run.status === 'error') {
          title.textContent = 'Error: ' + (run.error || 'unknown');
        }
      }
    }).catch(() => {});
  }, 500);
}

// ---------- results ----------
function loadResultsList() {
  fetch('/api/results').then(r => r.json()).then(data => {
    const sel = document.getElementById('results-select');
    sel.innerHTML = '<option value="">-- select a run --</option>';
    (data.runs || []).forEach(r => {
      const o = document.createElement('option');
      o.value = r; o.textContent = r;
      sel.appendChild(o);
    });
  });
}

document.getElementById('results-select').onchange = function() {
  const path = this.value;
  const div = document.getElementById('results-content');
  if (!path) { div.innerHTML = ''; return; }
  div.innerHTML = '<div class="results-empty">Loading...</div>';
  fetch('/api/results/' + path).then(r => r.json()).then(data => {
    renderResults(data, div);
  }).catch(err => {
    div.innerHTML = '<div class="results-empty">Error: ' + err + '</div>';
  });
};

function renderResults(data, container) {
  const md = data.metadata || {};
  const results = data.results || [];
  let metricKeys = [];
  results.forEach(r => Object.keys(flatten(r.metrics || {})).forEach(k => {
    if (!metricKeys.includes(k)) metricKeys.push(k);
  }));
  let html = `<div style="font-size:13px;color:var(--muted);margin-bottom:10px">
    <b>Started:</b> ${esc(md.started_at)} | <b>Platform:</b> ${esc(md.platform)} | <b>Git:</b> ${esc(md.git_commit ? md.git_commit.slice(0,8) : '')}
  </div>`;
  html += `<table><thead><tr><th>Model</th><th>Task</th><th class="num">N</th><th class="num">Err</th>` +
    metricKeys.map(k => `<th class="num">${esc(k)}</th>`).join('') + `</tr></thead><tbody>`;
  results.forEach(r => {
    const fm = flatten(r.metrics || {});
    html += `<tr><td><b>${esc(r.model)}</b></td><td>${esc(r.task)}</td>` +
      `<td class="num">${r.num_samples}</td><td class="num">${r.num_errors}</td>` +
      metricKeys.map(k => {
        const v = fm[k];
        const cls = typeof v === 'number' ? (v >= 0.75 ? 'good' : v >= 0.4 ? 'mid' : 'bad') : '';
        return `<td class="num"><span class="pill ${cls}">${typeof v === 'number' ? v.toFixed(4) : '—'}</span></td>`;
      }).join('') + `</tr>`;
  });
  html += '</tbody></table>';
  // per-sample drill-down
  results.forEach(r => {
    if (!r.samples || !r.samples.length) return;
    html += `<h3 style="font-size:13px;margin:16px 0 6px;color:var(--muted)">${esc(r.model)} / ${esc(r.task)} — samples</h3>`;
    html += `<div class="scrollbox"><table><thead><tr><th>#</th><th>Prediction</th><th>Reference</th><th>Raw</th></tr></thead><tbody>`;
    r.samples.slice(0, 50).forEach(s => {
      const err = String(s.prediction) !== String(s.reference) || s.error;
      html += `<tr class="${err ? 'err' : ''}"><td>${s.index}</td><td>${esc(short(s.prediction, 120))}</td><td>${esc(short(s.reference, 120))}</td><td style="color:var(--muted)">${esc(short(s.raw, 120))}</td></tr>`;
    });
    html += '</tbody></table></div>';
  });
  container.innerHTML = html;
}

// ---------- utils ----------
function flatten(m) {
  const o = {};
  for (const k in m) {
    if (m[k] && typeof m[k] === 'object') for (const s in m[k]) o[k + '.' + s] = m[k][s];
    else o[k] = m[k];
  }
  return o;
}
function esc(s) { return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function short(s, n) { const t = String(s || ''); return t.length > n ? t.slice(0, n) + '…' : t; }

// ---------- init: input is usable immediately; the first message creates the conversation ----------
enableInput(true);
chatInput.focus();
</script>
</body>
</html>
"""
