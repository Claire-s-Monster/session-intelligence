"""
Regression tests for issue #201: `commands_executed` was always 0 because the
PostToolUse hook only counted the blocked `Bash` tool.

Covers the stdlib-only `hooks.command_tracking` extractor (shell-runner,
pixi-task, git MCP and Bash) and the engine rendering "n/a" instead of a
fake 0 when no command was ever measured.

https://github.com/Claire-s-Monster/session-intelligence/issues/201
"""

from __future__ import annotations

import json

import pytest

from core.session_engine import SessionIntelligenceEngine
from hooks.command_tracking import TrackedCommand, extract_tracked_command
from persistence.sqlite import SQLiteBackend

SHELL = "mcp__shell-runner__shell_execute"
PIXI = "mcp__pixi-task__execute_tool"
GIT = "mcp__git__execute_tool"
SESSION_ID = "0b1f6c1e-52a1-4a52-9f0e-0a7a1d6f2a01"


def _envelope(result: object, status: str = "success") -> str:
    return json.dumps({"tool": "x", "status": status, "result": result})


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_bash_extracted():
    got = extract_tracked_command("Bash", {"command": "pytest tests/ -x"}, {"exit_code": 0})
    assert got == TrackedCommand("pytest tests/ -x", "pytest", True)


def test_shell_runner_extracted_and_truncated():
    cmd = "git log " + "x" * 200
    got = extract_tracked_command(SHELL, {"command": cmd}, "")
    assert got is not None
    assert got.command == cmd[:100]
    assert got.command_base == "git"


def test_pixi_run_task_extracted_with_environment():
    tool_input = {
        "tool_name": "pixi_run_task",
        "parameters": {"task_name": "test", "environment": "ci"},
    }
    got = extract_tracked_command(PIXI, tool_input, _envelope({"success": None}))
    assert got == TrackedCommand("pixi run -e ci test", "pixi", True)


def test_pixi_run_task_without_environment():
    tool_input = {"tool_name": "pixi_run_task", "parameters": {"task_name": "lint"}}
    got = extract_tracked_command(PIXI, tool_input, "")
    assert got is not None
    assert got.command == "pixi run lint"


def test_git_extracted():
    got = extract_tracked_command(GIT, {"tool_name": "git_status", "parameters": {}}, "")
    assert got == TrackedCommand("git_status", "git", True)


def test_uninteresting_commands_return_none():
    assert extract_tracked_command("Bash", {"command": "ls -la"}, "") is None
    assert extract_tracked_command(SHELL, {"command": "cat file"}, "") is None
    assert extract_tracked_command(SHELL, {"command": "   "}, "") is None


def test_pixi_non_run_tool_returns_none():
    tool_input = {"tool_name": "pixi_list_tasks", "parameters": {}}
    assert extract_tracked_command(PIXI, tool_input, "") is None


def test_git_missing_tool_name_returns_none():
    assert extract_tracked_command(GIT, {"parameters": {}}, "") is None


def test_unknown_tool_returns_none():
    assert extract_tracked_command("Read", {"command": "git status"}, "") is None


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "tool_response"),
    [
        ("Bash", None, None),
        ("Bash", {"command": 123}, None),
        (SHELL, "not a dict", None),
        (PIXI, {"tool_name": "pixi_run_task", "parameters": "oops"}, None),
        (PIXI, {"tool_name": "pixi_run_task", "parameters": None}, None),
        (GIT, {"tool_name": 5}, object()),
        (SHELL, {"command": "git status"}, object()),
        (SHELL, {"command": "git status"}, [None, 3, {"type": "text", "text": None}]),
        (None, {}, None),
    ],
)
def test_malformed_inputs_do_not_raise(tool_name, tool_input, tool_response):
    result = extract_tracked_command(tool_name, tool_input, tool_response)
    assert result is None or isinstance(result, TrackedCommand)


# ---------------------------------------------------------------------------
# Success detection
# ---------------------------------------------------------------------------


def test_envelope_status_error_is_failure():
    tool_input = {"tool_name": "git_status", "parameters": {}}
    got = extract_tracked_command(GIT, tool_input, _envelope("boom", status="error"))
    assert got is not None
    assert got.succeeded is False


def test_shell_runner_denied_is_failure():
    response = json.dumps({"decision": "denied"})
    got = extract_tracked_command(SHELL, {"command": "git push"}, response)
    assert got is not None
    assert got.succeeded is False


def test_shell_runner_prompt_required_is_failure():
    response = json.dumps({"decision": "prompt_required"})
    got = extract_tracked_command(SHELL, {"command": "git push"}, response)
    assert got is not None
    assert got.succeeded is False


def test_shell_runner_nonzero_exit_is_failure():
    got = extract_tracked_command(SHELL, {"command": "pytest"}, json.dumps({"exit_code": 1}))
    assert got is not None
    assert got.succeeded is False


def test_shell_runner_running_is_success():
    response = json.dumps({"status": "running", "exit_code": None})
    got = extract_tracked_command(SHELL, {"command": "pytest"}, response)
    assert got is not None
    assert got.succeeded is True


def test_pixi_result_success_false_is_failure():
    tool_input = {"tool_name": "pixi_run_task", "parameters": {"task_name": "test"}}
    got = extract_tracked_command(PIXI, tool_input, _envelope({"success": False, "exit_code": 1}))
    assert got is not None
    assert got.succeeded is False


def test_pixi_nonzero_exit_is_failure():
    tool_input = {"tool_name": "pixi_run_task", "parameters": {"task_name": "test"}}
    got = extract_tracked_command(PIXI, tool_input, _envelope({"exit_code": 2}))
    assert got is not None
    assert got.succeeded is False


def test_pixi_background_launch_is_success():
    tool_input = {"tool_name": "pixi_run_task", "parameters": {"task_name": "test"}}
    got = extract_tracked_command(PIXI, tool_input, _envelope({"success": None, "pid": 1}))
    assert got is not None
    assert got.succeeded is True


def test_word_error_inside_successful_envelope_is_success():
    tool_input = {"tool_name": "pixi_run_task", "parameters": {"task_name": "test"}}
    output = "tests/test_x.py::test_error_paths PASSED"
    response = _envelope({"success": True, "exit_code": 0, "stdout": output})
    got = extract_tracked_command(PIXI, tool_input, response)
    assert got is not None
    assert got.succeeded is True


def test_list_of_text_blocks_is_parsed():
    tool_input = {"tool_name": "pixi_run_task", "parameters": {"task_name": "test"}}
    blocks = [{"type": "text", "text": _envelope({"success": False})}]
    got = extract_tracked_command(PIXI, tool_input, blocks)
    assert got is not None
    assert got.succeeded is False


def test_content_wrapped_response_is_parsed():
    blocks = {"content": [{"type": "text", "text": json.dumps({"exit_code": 3})}]}
    got = extract_tracked_command(SHELL, {"command": "ruff check"}, blocks)
    assert got is not None
    assert got.succeeded is False


def test_unparseable_response_is_success():
    got = extract_tracked_command(SHELL, {"command": "git status"}, "plain text with error word")
    assert got is not None
    assert got.succeeded is True


def test_bash_failure_dict_forms():
    for response in ({"is_error": True}, {"exit_code": 1}, {"interrupted": True}):
        got = extract_tracked_command("Bash", {"command": "git status"}, response)
        assert got is not None
        assert got.succeeded is False


def test_unparseable_dict_with_is_error_is_failure():
    tool_input = {"tool_name": "pixi_run_task", "parameters": {"task_name": "t"}}
    got = extract_tracked_command(PIXI, tool_input, {"is_error": True})
    assert got is not None
    assert got.succeeded is False


# ---------------------------------------------------------------------------
# Engine: None = never measured
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
    db = SQLiteBackend(db_path=str(tmp_path / "issue201.db"))
    await db.initialize()
    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


@pytest.mark.regression
async def test_no_command_data_is_none_and_renders_na(engine):
    await engine.session_track_execution(
        session_id=SESSION_ID,
        agent_name="agent-a",
        step_data={"phase": "agent_start", "agent_type": "focused"},
    )
    session = engine.session_cache[SESSION_ID]
    assert session.performance_metrics.commands_executed is None
    assert "| Commands Executed | n/a |" in engine._generate_metrics_section(session)


@pytest.mark.regression
async def test_one_command_step_counts_one(engine):
    await engine.session_track_execution(
        session_id=SESSION_ID,
        agent_name="agent-a",
        step_data={"phase": "tool_use", "command": "pytest", "agent_type": "focused"},
    )
    session = engine.session_cache[SESSION_ID]
    assert session.performance_metrics.commands_executed == 1
    assert "| Commands Executed | 1 |" in engine._generate_metrics_section(session)
