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

#: Matches a numbered/bulleted plan line such as ``1. read config`` or ``- run tests``.
_PLAN_LINE_RE = re.compile(r"^(?:\d{1,2}[.、)）]|\*|-)\s*")

_PLANNER_SYSTEM = "You are a precise planning assistant. Output only the numbered plan."

_PLANNER_TEMPLATE = """\
You are planning a coding task inside the workspace: {workspace}

Task:
{task}

Available tools: {tool_names}

Write a short step-by-step plan for this task.
- Output ONLY a numbered list, one step per line, at most {limit} steps.
- Each step is one concrete action (read X, write Y, run Z, verify by ...).
- No prose before or after the list.
"""

_REFLECT_SYSTEM = "You are a rigorous self-review assistant. Output plain text only."

_REFLECT_TEMPLATE = """\
You are reviewing your own progress on a coding task.

Task: {task}
{plan_block}
Recent activity (oldest first):
{recent}

In 2-4 sentences answer:
1. Is the task on track or already complete?
2. What failed or was wasteful?
3. What exactly should the next 1-2 actions be?
Do not call tools; this is a review turn.  Plain text only.
"""


def _parse_plan_lines(text: str, limit: int = 6) -> List[str]:
    """Extract numbered/bulleted plan lines from a planner response."""

    lines: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if line and _PLAN_LINE_RE.match(line):
            lines.append(line)
            if len(lines) >= limit:
                break
    return lines


def _compress_history(messages: List[Dict[str, Any]], keep_head: int = 2,
                      keep_recent: int = 12) -> tuple:
    """Collapse the middle of ``messages`` into a compact summary.

    Returns ``(new_messages, dropped_count)``.  The head (system prompt, task
    and optionally the plan) and the most recent ``keep_recent`` messages are
    kept verbatim; dropped turns are condensed to a one-line-each digest so
    long tasks stay within the context budget.
    """

    if len(messages) <= keep_head + keep_recent:
        return messages, 0
    middle = messages[keep_head:-keep_recent]
    lines = []
    for m in middle:
        content = m.get("content") or ""
        first = content.splitlines()[0] if content else ""
        if first:
            lines.append(first[:160])
    if lines:
        summary = "（历史步骤摘要，工具输出已省略）\n" + "\n".join(lines[-40:])
    else:
        summary = "（早期历史已省略）"
    new_messages = messages[:keep_head] + [{"role": "user", "content": summary}] + messages[-keep_recent:]
    return new_messages, len(middle)


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
    planning: bool = False,
    reflection: bool = False,
    reflect_every: int = 8,
    max_history_messages: int = 40,
) -> Iterator[Dict[str, Any]]:
    """Drive the agent loop, yielding structured events for the UI.

    Event shapes
    ------------
    ``{"type": "start", ...}``          -- loop initialised
    ``{"type": "plan", "step": 0, "plan": [...]}``        -- upfront plan (planning=True)
    ``{"type": "thought", "step": n, "text": "..."}``
    ``{"type": "tool_call", "step": n, "tool": "...", "args": {...}}``
    ``{"type": "tool_result", "step": n, "result": "..."}``
    ``{"type": "reflection", "step": n, "text": "..."}``  -- self-review (reflection=True)
    ``{"type": "compress", "step": n, "dropped": k}``     -- history compression
    ``{"type": "error", "step": n, "error": "..."}``
    ``{"type": "finish", "step": n, "summary": "..."}``
    ``{"type": "end", "step": n, "reason": "..."}``
    """

    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    system_prompt = _build_system_prompt(workspace)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    ov = overrides or {}

    yield {"type": "start", "workspace": str(workspace), "task": task, "max_steps": max_steps}

    # ---- upfront planning (Plan-and-Execute flavour) ----------------------
    plan_lines: List[str] = []
    if planning:
        plan_resp = model.chat(
            [
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user", "content": _PLANNER_TEMPLATE.format(
                    workspace=workspace, task=task, limit=6,
                    tool_names=", ".join(t["name"] for t in list_tools()))},
            ],
            **ov,
        )
        plan_lines = _parse_plan_lines(plan_resp.text or "")
        yield {"type": "plan", "step": 0, "plan": plan_lines}
        if plan_lines:
            messages.append({
                "role": "user",
                "content": "执行计划：\n" + "\n".join(plan_lines) +
                           "\n按计划逐步执行；若计划与实际不符，在 thought 中说明并调整。",
            })

    # The head (system prompt + task + plan) is never compressed away.
    keep_head = len(messages)
    errors_in_a_row = 0

    for step in range(1, max_steps + 1):
        # ---- memory compression --------------------------------------------
        if len(messages) > max_history_messages:
            messages, dropped = _compress_history(messages, keep_head=keep_head)
            if dropped:
                yield {"type": "compress", "step": step, "dropped": dropped}

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
            errors_in_a_row += 1
            # Model produced no parseable call -- nudge it to retry.
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "Your previous response was not a valid JSON tool call. "
                           "Please respond with ONLY a JSON object containing "
                           "'thought', 'tool' and 'args'.",
            })
            hint = ""
            if not raw.strip():
                hint = (" Model returned empty content. Reasoning models (glm-4.7-flash,"
                        " DeepSeek-R1, ...) spend max_tokens on thinking first -- raise"
                        " max_tokens (4096+) and retry.")
            elif resp.finish_reason == "length":
                hint = " Output hit the max_tokens limit before any JSON appeared; raise max_tokens."
            yield {"type": "error", "step": step,
                   "error": f"Could not parse tool call.{hint} Raw response: {raw[:200]!r}"}
            if reflection and errors_in_a_row >= 2:
                yield from _maybe_reflect(model, task, plan_lines, messages, step, ov)
                errors_in_a_row = 0
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

        errors_in_a_row = errors_in_a_row + 1 if result.startswith("[ERROR]") else 0

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

        # ---- periodic self-review (reflection) -----------------------------
        if reflection and (step % max(reflect_every, 1) == 0 or errors_in_a_row >= 3):
            yield from _maybe_reflect(model, task, plan_lines, messages, step, ov)
            errors_in_a_row = 0
    else:
        yield {"type": "end", "step": max_steps, "reason": "step budget exhausted"}


def _maybe_reflect(model: Model, task: str, plan_lines: List[str],
                   messages: List[Dict[str, Any]], step: int,
                   overrides: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Run one self-review turn; yields a ``reflection`` event when produced.

    The conclusions are appended to ``messages`` so the next ReAct step
    continues with the corrected course.  Failures degrade silently -- a
    reflection-less loop is always valid.
    """

    recent = "\n".join(
        f"- {m.get('content', '')[:160]}" for m in messages[-6:] if m.get("role") == "user"
    )
    plan_block = ("Current plan:\n" + "\n".join(plan_lines)) if plan_lines else "(no formal plan)"
    resp = model.chat(
        [
            {"role": "system", "content": _REFLECT_SYSTEM},
            {"role": "user", "content": _REFLECT_TEMPLATE.format(
                task=task, plan_block=plan_block, recent=recent or "(none)")},
        ],
        **overrides,
    )
    text = (resp.text or "").strip()
    if resp.error or not text:
        return
    yield {"type": "reflection", "step": step, "text": text}
    messages.append({
        "role": "user",
        "content": "（自我反思）" + text + "\n根据以上反思继续执行剩余步骤。",
    })
