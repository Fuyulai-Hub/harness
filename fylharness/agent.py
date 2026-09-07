"""ReAct-style agent loop: the harness acts as an autonomous coding agent.

Given a natural-language task, a workspace directory and an LLM, the agent:

1. builds a system prompt describing the available tools and the workspace,
2. asks the model for the next action (a JSON tool call),
3. executes the tool in :mod:`fylharness.tools`,
4. feeds the result back to the model,
5. repeats until the model calls ``finish`` or hits the step budget.

All steps are streamed to the UI as structured events so the user can watch the
agent think, act and observe in real time -- mirroring the agentic coding tools
(Cursor, Aider, OpenDevin) while staying fully Windows-safe.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .models.base import Model
from .tools import ToolError, list_tools, run_tool

log = logging.getLogger(__name__)

#: Maximum agent steps before we stop to avoid runaway loops.
MAX_STEPS = 25

#: The catalogue the model sees so it knows how to call each tool.
_SYSTEM_TEMPLATE = """\
You are an autonomous coding agent operating inside a workspace on the user's \
computer.  You receive a task in natural language and must complete it by calling \
tools, observing their output, and iterating until done.

Workspace root: {workspace}

Available tools (call by emitting a JSON object on a single line):
{tool_catalog}

Response format -- on each turn output ONLY a JSON object with this exact shape:

{{
  "thought": "<brief reasoning about what to do next>",
  "tool": "<one of the tool names above>",
  "args": {{...}}
}}

Rules:
- Always emit exactly one JSON object per turn.  No prose before or after.
- Use `list_dir` and `read_file` to explore before editing.
- Use `write_file` for new files, `edit_file` for targeted changes.
- Use `run_command` to run builds / tests / scripts (60s timeout, Windows shell).
- When the task is complete, call the `finish` tool with a short summary.
- Stay inside the workspace; absolute paths are rejected.
- Keep thoughts to 1-2 sentences.
"""


def _build_system_prompt(workspace: Path) -> str:
    catalog = "\n".join(
        f"- {t['name']}({t.get('args', '')}): {t['description']}" for t in list_tools()
    )
    return _SYSTEM_TEMPLATE.format(workspace=workspace, tool_catalog=catalog)


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse_tool_call(text: str) -> Dict[str, Any]:
    """Extract the JSON tool call from the model's response.

    Models sometimes wrap JSON in prose or fences; we grab the first balanced
    ``{...}`` block and parse it.  Returns ``{}`` (empty) on failure so the loop
    can ask the model to try again.
    """

    # Try the whole text first (common case).
    try:
        return json.loads(text.strip().strip("`"))
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} block.
    match = _JSON_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0).strip().strip("`"))
        except json.JSONDecodeError:
            pass
    return {}


def run_agent(
    model: Model,
    task: str,
    workspace: Path,
    *,
    max_steps: int = MAX_STEPS,
    overrides: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Drive the agent loop, yielding structured events for the UI.

    Event shapes
    ------------
    ``{"type": "start", ...}``          -- loop initialised
    ``{"type": "thought", "step": n, "text": "..."}``
    ``{"type": "tool_call", "step": n, "tool": "...", "args": {...}}``
    ``{"type": "tool_result", "step": n, "result": "..."}``
    ``{"type": "error", "step": n, "error": "..."}``
    ``{"type": "finish", "step": n, "summary": "..."}``
    ``{"type": "end", "step": n, "reason": "..."}``
    """

    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    system_prompt = _build_system_prompt(workspace)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    ov = overrides or {}

    yield {"type": "start", "workspace": str(workspace), "task": task, "max_steps": max_steps}

    for step in range(1, max_steps + 1):
        # Ask the model for the next action.  We use the non-streaming chat path
        # because the agent needs the full JSON before acting; the UI streams
        # the structured events instead.
        resp = model.chat(messages, **ov)
        if resp.error:
            yield {"type": "error", "step": step, "error": resp.error}
            break
        raw = resp.text or ""
        call = _parse_tool_call(raw)
        thought = call.get("thought", "")
        tool = call.get("tool", "")
        args = call.get("args", {}) or {}

        if thought:
            yield {"type": "thought", "step": step, "text": thought}

        if not tool:
            # Model produced no parseable call -- nudge it to retry.
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "Your previous response was not a valid JSON tool call. "
                           "Please respond with ONLY a JSON object containing "
                           "'thought', 'tool' and 'args'.",
            })
            yield {"type": "error", "step": step,
                   "error": "Could not parse tool call; asking model to retry."}
            continue

        yield {"type": "tool_call", "step": step, "tool": tool, "args": args}

        # Execute the tool inside the workspace.
        try:
            result = run_tool(tool, args, workspace)
        except ToolError as exc:
            result = f"[ERROR] {exc}"
        except Exception as exc:  # pragma: no cover -- defensive
            log.exception("Tool %s raised", tool)
            result = f"[ERROR] {exc!s}"

        yield {"type": "tool_result", "step": step, "result": result}

        # Feed the result back so the model can reason about it next turn.
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": f"Tool `{tool}` returned:\n{result}",
        })

        if tool == "finish":
            yield {"type": "finish", "step": step, "summary": result}
            yield {"type": "end", "step": step, "reason": "model called finish; task complete"}
            break
    else:
        yield {"type": "end", "step": max_steps, "reason": "step budget exhausted"}
