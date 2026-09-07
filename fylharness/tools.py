"""Sandboxed filesystem and shell tools for the agent loop.

Each tool is a plain function taking a dict of arguments and returning a string
result suitable for inclusion in an LLM's tool-result context.  All path operations
are clamped to a configurable workspace root to prevent the agent from escaping the
designated project directory.

Tools exposed
------------
- ``list_dir``    -- list entries in a sub-directory
- ``read_file``   -- read a text file (with line numbers, truncated)
- ``write_file``  -- create or overwrite a file
- ``edit_file``   -- replace a contiguous block of lines
- ``run_command`` -- execute a shell command (Windows-safe, no fork)
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List

# Maximum bytes returned by read_file / command stdout to keep prompts bounded.
MAX_OUTPUT = 12000


class ToolError(Exception):
    """Raised when a tool refuses to execute (bad path, missing arg, ...)."""


# ----------------------------------------------------------------- registry
TOOL_REGISTRY: Dict[str, Callable[[Dict[str, Any], Path], str]] = {}
TOOL_ARGS: Dict[str, str] = {}


def register_tool(name: str, args: str = "") -> Callable[[Callable], Callable]:
    """Decorator: register a tool function under ``name``.

    ``args`` is a short hint of the expected argument keys (e.g. ``"path, content"``)
    surfaced in the system prompt so the model emits correct argument names.
    """

    def _wrap(fn: Callable[[Dict[str, Any], Path], str]) -> Callable:
        TOOL_REGISTRY[name] = fn
        TOOL_ARGS[name] = args
        return fn

    return _wrap


def list_tools() -> List[Dict[str, str]]:
    """Return a JSON-serialisable tool catalogue for the system prompt."""

    return [
        {
            "name": name,
            "args": TOOL_ARGS.get(name, ""),
            "description": (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "",
        }
        for name, fn in sorted(TOOL_REGISTRY.items())
    ]


def run_tool(name: str, args: Dict[str, Any], workspace: Path) -> str:
    """Dispatch ``args`` to the registered tool ``name`` inside ``workspace``."""

    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        raise ToolError(f"Unknown tool '{name}'. Available: {sorted(TOOL_REGISTRY)}")
    return fn(args, workspace)


# ------------------------------------------------------------- path safety
def _safe_path(workspace: Path, rel: str) -> Path:
    """Resolve ``rel`` against ``workspace`` and refuse escapes (path traversal)."""

    if not rel:
        rel = "."
    # Block absolute paths and drive letters (Windows D:\...).
    if Path(rel).is_absolute() or (len(rel) >= 2 and rel[1] == ":"):
        raise ToolError(f"Path must be relative to the workspace root, got: {rel}")
    base = workspace.resolve()
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise ToolError(f"Path '{rel}' escapes the workspace root.")
    return target


# ------------------------------------------------------------------ tools
@register_tool("list_dir", args="path")
def _list_dir(args: Dict[str, Any], workspace: Path) -> str:
    """List entries in a directory (default: workspace root)."""

    rel = args.get("path", "")
    target = _safe_path(workspace, rel)
    if not target.exists():
        raise ToolError(f"Not found: {rel}")
    if not target.is_dir():
        raise ToolError(f"Not a directory: {rel}")
    entries = []
    for p in sorted(target.iterdir()):
        kind = "dir" if p.is_dir() else "file"
        size = p.stat().st_size if p.is_file() else ""
        entries.append(f"{kind:4}  {size:>8}  {p.name}")
    return f"Listing {rel or '.'}:\n" + "\n".join(entries) if entries else f"{rel or '.'} is empty."


@register_tool("read_file", args="path")
def _read_file(args: Dict[str, Any], workspace: Path) -> str:
    """Read a text file with line numbers (truncated to keep prompts bounded)."""

    rel = args.get("path", "")
    target = _safe_path(workspace, rel)
    if not target.exists():
        raise ToolError(f"File not found: {rel}")
    if not target.is_file():
        raise ToolError(f"Not a file: {rel}")
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > 200:
        lines = lines[:200]
        truncated = f"\n... ({len(text.splitlines()) - 200} more lines truncated)"
    else:
        truncated = ""
    numbered = "\n".join(f"{i+1:4}  {l}" for i, l in enumerate(lines))
    return numbered + truncated


@register_tool("write_file", args="path, content")
def _write_file(args: Dict[str, Any], workspace: Path) -> str:
    """Create or overwrite a file with the given content."""

    rel = args.get("path", "")
    content = args.get("content", "")
    if not rel:
        raise ToolError("path is required.")
    target = _safe_path(workspace, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {rel}"


@register_tool("edit_file", args="path, old, new")
def _edit_file(args: Dict[str, Any], workspace: Path) -> str:
    """Replace all occurrences of ``old`` with ``new`` in a file."""

    rel = args.get("path", "")
    old = args.get("old", "")
    new = args.get("new", "")
    target = _safe_path(workspace, rel)
    if not target.is_file():
        raise ToolError(f"File not found: {rel}")
    text = target.read_text(encoding="utf-8", errors="replace")
    if old not in text:
        raise ToolError("old text not found in file.")
    count = text.count(old)
    new_text = text.replace(old, new)
    target.write_text(new_text, encoding="utf-8")
    return f"Replaced {count} occurrence(s) in {rel}"


@register_tool("run_command", args="command")
def _run_command(args: Dict[str, Any], workspace: Path) -> str:
    """Run a shell command in the workspace directory (Windows-safe, no fork).

    The command runs via ``subprocess.run`` with a hard 60-second timeout; stdout
    and stderr are captured and returned.  Interactive commands are not supported.
    """

    cmd = args.get("command", "")
    if not cmd:
        raise ToolError("command is required.")
    # Reject obviously dangerous patterns; the sandbox is for coding tasks.
    blocked = ["rm -rf /", "format ", "del /f /s /q c:"]
    if any(b in cmd.lower() for b in blocked):
        raise ToolError("Command blocked by safety filter.")
    try:
        # shell=True so the user can pipe/redirect; cwd pins to workspace.
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(workspace.resolve()),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return f"[command timed out after 60s]\n{cmd}"
    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + f"\n... (output truncated, {len(out) - MAX_OUTPUT} chars cut)"
    return f"[exit {proc.returncode}]\n{out}"


@register_tool("finish", args="summary")
def _finish(args: Dict[str, Any], workspace: Path) -> str:
    """Signal that the task is complete (no more tool calls needed)."""

    return args.get("summary", "Task complete.")
