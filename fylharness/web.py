"""Web-based results viewer for FylHarness.

After an evaluation finishes (or for any previously-written ``results.json``), this
module builds a self-contained HTML dashboard and serves it through the standard
library's :mod:`http.server` -- no Flask, no extra dependencies.  The browser opens
automatically so the workflow mirrors the DeepSeek evaluation harness experience.

The dashboard is a single page with inline CSS + JS so it works fully offline:

* a metadata header (start time, platform, git commit, tags),
* a model × task summary matrix with colour-coded metric cells,
* pure-CSS bar charts comparing models per metric,
* a per-task drill-down listing every sample's prediction vs. reference with
  error highlighting and a free-text filter box.
"""

from __future__ import annotations

import json
import logging
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- page
def build_dashboard_html(payload: Dict[str, Any]) -> str:
    """Render the results payload into a single self-contained HTML document.

    The results JSON is embedded verbatim inside a ``<script>`` tag and the page
    renders entirely client-side, which keeps the server trivial and lets the user
    save the page as a static ``.html`` file for later sharing.
    """

    results_json = json.dumps(payload, ensure_ascii=False, default=str)
    # Escape ``</script>`` sequences inside the embedded JSON just in case a
    # prediction contains that literal.
    safe_json = results_json.replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__RESULTS_JSON__", safe_json)


# --------------------------------------------------------------------- server
class _DashboardHandler(BaseHTTPRequestHandler):
    """Serves the single dashboard page plus the raw ``results.json`` for fetching."""

    # Injected by :func:`serve_dashboard`.
    html: bytes = b""
    results_json: bytes = b""

    def log_message(self, format: str, *args) -> None:  # silence default logging
        return

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.html)))
            self.end_headers()
            self.wfile.write(self.html)
            return
        if self.path in ("/results.json", "/data"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(self.results_json)))
            self.end_headers()
            self.wfile.write(self.results_json)
            return
        self.send_error(404, "Not found")


def serve_dashboard(
    results_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    once: bool = False,
) -> int:
    """Serve the dashboard for ``results_dir`` and return the bound port.

    Parameters
    ----------
    results_dir : path
        Directory containing ``results.json`` (the runner's output dir).
    host, port : str / int
        Bind address.  ``port=0`` lets the OS pick a free port, which avoids clashes
        with already-running servers.
    open_browser : bool
        When true (default) the default browser is launched at the dashboard URL.
    once : bool
        When true the server shuts down after the first hit so the command returns
        instead of blocking.  Useful for tests.
    """

    results_dir = Path(results_dir)
    results_path = results_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"results.json not found in {results_dir}. Run an evaluation first."
        )
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    html = build_dashboard_html(payload).encode("utf-8")
    raw_json = results_path.read_bytes()

    _DashboardHandler.html = html
    _DashboardHandler.results_json = raw_json

    httpd = ThreadingHTTPServer((host, port), _DashboardHandler)
    bound_port = httpd.server_address[1]
    url = f"http://{host}:{bound_port}/"
    log.info("Serving FylHarness dashboard at %s (Ctrl+C to stop)", url)
    print(f"\n  FylHarness dashboard:  {url}\n  Serving from:          {results_dir}\n")

    if open_browser:
        # Open in a thread so the server can start serving immediately; some browsers
        # block on the initial connection.
        threading.Thread(
            target=webbrowser.open, args=(url,), daemon=True
        ).start()

    try:
        if once:
            httpd.handle_request()
        else:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
    finally:
        httpd.server_close()
    return bound_port


# --------------------------------------------------------------- convenience
def open_results(
    results_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
) -> int:
    """Public alias matching the :func:`serve_dashboard` signature for CLI use."""

    return serve_dashboard(
        results_dir, host=host, port=port, open_browser=open_browser
    )


# ---------------------------------------------------------------- template
# A single-page dashboard.  CSS + JS are inlined so the saved HTML works offline and
# can be emailed around.  No external CDN/JS framework: plain DOM manipulation keeps
# the page small and dependency-free.
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FylHarness Dashboard</title>
<style>
  :root {
    --bg: #0f1115; --panel: #181b22; --panel-2: #1f232c; --border: #2a2f3a;
    --text: #e6e8eb; --muted: #8b93a1; --accent: #4f9cf9; --good: #3fb950;
    --warn: #d29922; --bad: #f85149; --row-alt: #14171d;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      "Helvetica Neue", Arial, sans-serif; background: var(--bg); color: var(--text);
    line-height: 1.5;
  }
  header {
    padding: 18px 28px; background: linear-gradient(180deg, #161a22, var(--panel));
    border-bottom: 1px solid var(--border);
  }
  header h1 { margin: 0 0 6px; font-size: 20px; font-weight: 600; }
  header .meta { color: var(--muted); font-size: 13px; display: flex; flex-wrap: wrap; gap: 14px; }
  header .meta span b { color: var(--text); font-weight: 500; }
  .layout { display: grid; grid-template-columns: 1fr; gap: 18px; padding: 22px 28px; max-width: 1400px; margin: 0 auto; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
  .card h2 { margin: 0; padding: 14px 18px; font-size: 15px; font-weight: 600; border-bottom: 1px solid var(--border); background: var(--panel-2); }
  .card .body { padding: 14px 18px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
  tbody tr:hover { background: var(--panel-2); }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .metric-cell { font-variant-numeric: tabular-nums; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }
  .pill.good { background: rgba(63,185,80,.15); color: var(--good); }
  .pill.mid { background: rgba(210,153,34,.15); color: var(--warn); }
  .pill.bad { background: rgba(248,81,73,.15); color: var(--bad); }
  .bar-wrap { display: flex; align-items: center; gap: 8px; }
  .bar-track { flex: 1; height: 8px; background: var(--panel-2); border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--accent); border-radius: 4px; }
  .tabs { display: flex; gap: 4px; padding: 10px 18px 0; flex-wrap: wrap; }
  .tab { padding: 6px 14px; border-radius: 6px 6px 0 0; cursor: pointer; font-size: 13px; color: var(--muted); border: 1px solid transparent; }
  .tab.active { background: var(--panel-2); color: var(--text); border-color: var(--border); }
  .toolbar { display: flex; gap: 10px; align-items: center; padding: 10px 18px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }
  .toolbar input { flex: 1; min-width: 200px; padding: 7px 10px; background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: 6px; font-size: 13px; }
  .toolbar select { padding: 7px 10px; background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: 6px; font-size: 13px; }
  .sample-row td { vertical-align: top; }
  .sample-row .pred { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
  .sample-row .ref { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--good); }
  .sample-row.err { background: rgba(248,81,73,.06); }
  .sample-row .err-badge { color: var(--bad); font-size: 11px; }
  .empty { padding: 30px; text-align: center; color: var(--muted); }
  .kvs { display: flex; flex-wrap: wrap; gap: 6px; }
  .kv { font-size: 12px; color: var(--muted); }
  .kv b { color: var(--text); }
  details summary { cursor: pointer; padding: 8px 0; color: var(--muted); }
  .flash { color: var(--accent); font-weight: 600; }
  @media (max-width: 720px) { .layout { padding: 14px; } header { padding: 14px; } }
</style>
</head>
<body>
<header>
  <h1>FylHarness Evaluation Dashboard</h1>
  <div class="meta" id="meta"></div>
</header>
<div class="layout">

  <div class="card">
    <h2>Summary</h2>
    <div class="body" id="summary-body"></div>
  </div>

  <div class="card">
    <h2>Model Comparison</h2>
    <div class="body" id="charts-body"></div>
  </div>

  <div class="card">
    <h2>Per-Sample Predictions</h2>
    <div class="tabs" id="task-tabs"></div>
    <div class="toolbar">
      <select id="model-filter"><option value="">All models</option></select>
      <input id="search" type="search" placeholder="Filter samples (prediction / reference / text)..." />
      <label class="kv"><input id="only-errors" type="checkbox"> errors only</label>
    </div>
    <div class="body" id="samples-body" style="padding:0;"></div>
  </div>

</div>

<script>
const DATA = __RESULTS_JSON__;

// -------------------------------------------------------------- helpers
function fmt(v, d=4) {
  if (typeof v === "number") return v.toFixed(d);
  return String(v ?? "");
}
function metricValue(m, key) {
  if (key in m) return m[key];
  // flatten: rouge.rouge_1
  for (const k in m) {
    if (k.startsWith(key + ".")) return m[k];
    if (typeof m[k] === "object" && m[k] && key in m[k]) return m[k][key];
  }
  return null;
}
function flattenMetrics(m) {
  const out = {};
  for (const k in m) {
    if (m[k] && typeof m[k] === "object") {
      for (const sub in m[k]) out[k + "." + sub] = m[k][sub];
    } else out[k] = m[k];
  }
  return out;
}
function classFor(v) {
  if (v == null) return "pill";
  if (typeof v !== "number") return "pill";
  if (v >= 0.75) return "pill good";
  if (v >= 0.4) return "pill mid";
  return "pill bad";
}
function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
function short(s, n=400) {
  const t = String(s ?? "");
  return t.length > n ? t.slice(0, n) + " …" : t;
}

// --------------------------------------------------------------- meta
(function renderMeta() {
  const md = DATA.metadata || {};
  const items = [
    ["Started", md.started_at],
    ["Platform", md.platform],
    ["Python", md.python],
    ["Git", md.git_commit ? md.git_commit.slice(0, 12) : null],
    ["Config", md.config_source],
    ["Seed", md.seed],
    ["Tags", (md.tags || []).join(", ")],
  ].filter(([_, v]) => v);
  document.getElementById("meta").innerHTML = items
    .map(([k, v]) => `<span><b>${k}:</b> ${esc(v)}</span>`).join("");
})();

// ----------------------------------------------------- summary matrix
(function renderSummary() {
  const results = DATA.results || [];
  if (!results.length) {
    document.getElementById("summary-body").innerHTML = '<div class="empty">No results.</div>';
    return;
  }
  const metricKeys = [];
  const seen = new Set();
  results.forEach(r => {
    Object.keys(flattenMetrics(r.metrics || {})).forEach(k => {
      if (!seen.has(k)) { seen.add(k); metricKeys.push(k); }
    });
  });
  let html = '<table><thead><tr>' +
    '<th>Model</th><th>Task</th><th class="num">Samples</th><th class="num">Errors</th>' +
    '<th class="num">Latency (s)</th><th class="num">Tokens (in/out)</th>' +
    metricKeys.map(k => `<th class="num">${esc(k)}</th>`).join('') +
    '</tr></thead><tbody>';
  results.forEach(r => {
    const fm = flattenMetrics(r.metrics || {});
    html += `<tr><td><b>${esc(r.model)}</b></td><td>${esc(r.task)}</td>` +
      `<td class="num">${r.num_samples}</td>` +
      `<td class="num">${r.num_errors}</td>` +
      `<td class="num">${(r.total_latency_ms/1000).toFixed(2)}</td>` +
      `<td class="num">${r.total_prompt_tokens}/${r.total_completion_tokens}</td>` +
      metricKeys.map(k => {
        const v = fm[k];
        const cls = classFor(v);
        return `<td class="num metric-cell"><span class="${cls}">${v == null ? '—' : (typeof v === 'number' ? v.toFixed(4) : esc(v))}</span></td>`;
      }).join('') +
      '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById("summary-body").innerHTML = html;
})();

// ------------------------------------------------------- comparison charts
(function renderCharts() {
  const results = DATA.results || [];
  const models = [...new Set(results.map(r => r.model))];
  if (models.length < 2) {
    document.getElementById("charts-body").innerHTML =
      '<div class="empty">Add a second model to the config to see comparison charts.</div>';
    return;
  }
  // Pick the primary numeric metric present across most results.
  const metricCounts = {};
  results.forEach(r => {
    Object.keys(flattenMetrics(r.metrics || {})).forEach(k => {
      metricCounts[k] = (metricCounts[k] || 0) + 1;
    });
  });
  const metricKeys = Object.keys(metricCounts)
    .filter(k => results.some(r => typeof flattenMetrics(r.metrics)[k] === "number"))
    .sort((a, b) => metricCounts[b] - metricCounts[a])
    .slice(0, 4);
  if (!metricKeys.length) {
    document.getElementById("charts-body").innerHTML =
      '<div class="empty">No numeric metrics to chart.</div>';
    return;
  }
  let html = "";
  metricKeys.forEach(mk => {
    html += `<div style="margin-bottom:14px"><div class="kv" style="margin-bottom:6px"><b>${esc(mk)}</b></div>`;
    models.forEach(m => {
      // average this metric across tasks for model m
      const vals = results.filter(r => r.model === m)
        .map(r => flattenMetrics(r.metrics)[mk])
        .filter(v => typeof v === "number");
      const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      const pct = Math.max(0, Math.min(100, avg * 100));
      html += `<div class="bar-wrap" style="margin:3px 0">
        <div style="width:140px;font-size:12px;color:var(--muted)">${esc(m)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        <div class="num" style="width:60px;text-align:right;font-size:12px">${avg.toFixed(4)}</div>
      </div>`;
    });
    html += '</div>';
  });
  document.getElementById("charts-body").innerHTML = html;
})();

// ----------------------------------------------------- samples drill-down
const results = DATA.results || [];
const tasks = [...new Set(results.map(r => r.task))];
let activeTask = tasks[0] || "";
let models = [...new Set(results.map(r => r.model))];

(function renderModelFilter() {
  const sel = document.getElementById("model-filter");
  models.forEach(m => {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    sel.appendChild(o);
  });
})();

(function renderTabs() {
  const wrap = document.getElementById("task-tabs");
  wrap.innerHTML = "";
  tasks.forEach(t => {
    const el = document.createElement("div");
    el.className = "tab" + (t === activeTask ? " active" : "");
    el.textContent = t;
    el.onclick = () => { activeTask = t; renderTabs(); renderSamples(); };
    wrap.appendChild(el);
  });
})();

function renderSamples() {
  const body = document.getElementById("samples-body");
  const modelFilter = document.getElementById("model-filter").value;
  const q = document.getElementById("search").value.trim().toLowerCase();
  const onlyErrors = document.getElementById("only-errors").checked;
  const matching = results.filter(r =>
    r.task === activeTask &&
    (!modelFilter || r.model === modelFilter)
  );
  if (!matching.length) {
    body.innerHTML = '<div class="empty">No samples for this filter.</div>';
    return;
  }
  // Merge all matching (model,task) pairs into one list, sort by index then model.
  let rows = [];
  matching.forEach(r => {
    (r.samples || []).forEach(s => {
      const hay = JSON.stringify({
        p: s.prediction, r: s.reference, raw: s.raw, s: s.sample, m: r.model
      }).toLowerCase();
      const isError = !!s.error || String(s.prediction) !== String(s.reference);
      if (q && !hay.includes(q)) return;
      if (onlyErrors && !isError) return;
      rows.push({ ...s, model: r.model, isError });
    });
  });
  rows.sort((a, b) => (a.index - b.index) || (a.model < b.model ? -1 : 1));
  let html = '<table><thead><tr>' +
    '<th class="num">#</th><th>Model</th><th>Prediction</th><th>Reference</th>' +
    '<th class="num">Latency (ms)</th><th class="num">Tokens</th><th>Status</th>' +
    '</tr></thead><tbody>';
  if (!rows.length) {
    html += '<tr><td colspan="7" class="empty">No matching samples.</td></tr>';
  }
  rows.slice(0, 500).forEach(s => {
    const cls = "sample-row" + (s.isError ? " err" : "");
    html += `<tr class="${cls}">` +
      `<td class="num">${s.index}</td>` +
      `<td>${esc(s.model)}</td>` +
      `<td class="pred">${esc(short(s.prediction))}</td>` +
      `<td class="ref">${esc(short(s.reference))}</td>` +
      `<td class="num">${(s.latency_ms || 0).toFixed(1)}</td>` +
      `<td class="num">${s.prompt_tokens || 0}/${s.completion_tokens || 0}</td>` +
      `<td>${s.isError ? '<span class="err-badge">✗ ' + (s.error ? esc(s.error) : 'mismatch') + '</span>' : '<span class="pill good">✓</span>'}</td>` +
      `</tr>`;
    if (s.raw && s.raw !== String(s.prediction)) {
      html += `<tr class="${cls}"><td></td><td colspan="6" class="pred" style="color:var(--muted)">raw: ${esc(short(s.raw, 200))}</td></tr>`;
    }
  });
  if (rows.length > 500) {
    html += `<tr><td colspan="7" class="empty">… ${rows.length - 500} more rows (refine the filter to see them)</td></tr>`;
  }
  html += '</tbody></table>';
  body.innerHTML = html;
}

document.getElementById("model-filter").addEventListener("change", renderSamples);
document.getElementById("search").addEventListener("input", renderSamples);
document.getElementById("only-errors").addEventListener("change", renderSamples);
renderSamples();
</script>
</body>
</html>
"""
