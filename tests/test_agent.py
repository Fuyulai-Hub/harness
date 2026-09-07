"""Unit tests for the agent loop helpers (plan parsing, history compression)."""

from fylharness.agent import _parse_plan_lines, _compress_history
from fylharness.config import AgentConfig, HarnessConfig


def test_parse_plan_lines_numbered_and_bulleted():
    text = (
        "Here is the plan:\n"
        "1. read note.txt\n"
        "2、write copy.txt\n"
        "3) run dir\n"
        "- verify result\n"
        "not a plan line without marker\n"
    )
    lines = _parse_plan_lines(text)
    assert len(lines) == 4
    assert lines[0] == "1. read note.txt"
    assert lines[3] == "- verify result"


def test_parse_plan_lines_limit():
    text = "\n".join(f"{i}. step {i}" for i in range(1, 11))
    assert len(_parse_plan_lines(text, limit=6)) == 6
    assert _parse_plan_lines("") == []


def test_compress_history_noop_when_short():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    new, dropped = _compress_history(messages)
    assert new is messages and dropped == 0


def test_compress_history_collapses_middle():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(20):
        messages.append({"role": "assistant", "content": f'{{"thought":"t{i}","tool":"read_file"}}'})
        messages.append({"role": "user", "content": f"Tool `read_file` returned:\nline {i}"})
    messages.append({"role": "assistant", "content": "recent"})
    messages.append({"role": "user", "content": "Tool `finish` returned:\nTask complete."})
    new, dropped = _compress_history(messages, keep_head=2, keep_recent=4)
    assert dropped == len(messages) - 2 - 4
    assert len(new) == 2 + 1 + 4
    assert new[0]["content"] == "sys"
    assert "历史步骤摘要" in new[2]["content"]
    assert "read_file" in new[2]["content"]
    assert new[-1]["content"].startswith("Tool `finish`")


def test_agent_config_defaults_and_parsing():
    cfg = HarnessConfig.from_dict({})
    assert cfg.agent.planning is False and cfg.agent.reflect_every == 8
    cfg2 = HarnessConfig.from_dict({"agent": {"planning": True, "reflection": True,
                                              "reflect_every": 4}})
    assert cfg2.agent.planning is True
    assert cfg2.agent.reflection is True
    assert cfg2.agent.reflect_every == 4
    assert isinstance(cfg2.agent, AgentConfig)
