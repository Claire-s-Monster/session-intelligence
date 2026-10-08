"""Extract a trackable command from a PostToolUse event.

Issue #201: the PostToolUse hook only counted the ``Bash`` tool, which is
blocked in this ecosystem, so ``commands_executed`` was always 0. Real
commands go through ``mcp__shell-runner__shell_execute``,
``mcp__pixi-task__execute_tool`` and ``mcp__git__execute_tool``.

``extract_tracked_command`` maps any of those tool calls to a
``TrackedCommand`` (or None when the call is not a trackable command).
FAILED commands are still returned (``succeeded=False``): a failed pytest run
still executed, so callers decide the phase rather than dropping it.

STDLIB ONLY. This module is imported by
``~/.claude/hooks/session_intelligence_post_tool.py``, which runs under plain
``python3`` OUTSIDE this project's pixi environment. It must not import any
third-party package or anything from this repo's own ``src`` package tree.
It never raises: malformed input yields None.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

INTERESTING_SHELL_COMMANDS = ("pixi", "pytest", "ruff", "mypy", "git", "rattler-build", "conda")

_SHELL_TOOLS = ("Bash", "mcp__shell-runner__shell_execute")
_PIXI_TOOL = "mcp__pixi-task__execute_tool"
_GIT_TOOL = "mcp__git__execute_tool"
_MAX_COMMAND_LEN = 100


@dataclass(frozen=True)
class TrackedCommand:
    command: str
    command_base: str
    succeeded: bool


def _extract_shell(tool_input: dict) -> tuple[str, str] | None:
    raw = tool_input.get("command")
    if not isinstance(raw, str):
        return None
    tokens = raw.split()
    if not tokens:
        return None
    base = tokens[0]
    # Substring (not equality) test, preserved from the original hook so that
    # e.g. "/usr/bin/git" or "pixi-pack" keep matching as before.
    if not any(cmd in base for cmd in INTERESTING_SHELL_COMMANDS):
        return None
    return raw, base


def _extract_pixi(tool_input: dict) -> tuple[str, str] | None:
    if tool_input.get("tool_name") != "pixi_run_task":
        return None
    params = tool_input.get("parameters") or {}
    if not isinstance(params, dict):
        return None
    task_name = params.get("task_name")
    if not task_name:
        return None
    command = "pixi run"
    environment = params.get("environment")
    if environment:
        command += f" -e {environment}"
    return f"{command} {task_name}", "pixi"


def _extract_git(tool_input: dict) -> tuple[str, str] | None:
    name = tool_input.get("tool_name")
    if not name or not isinstance(name, str):
        return None
    return name, "git"


def _unwrap_text(tool_response: object) -> object:
    """Turn str / list-of-text-blocks into a decoded JSON value when possible."""
    if isinstance(tool_response, list):
        texts = [
            block.get("text", "")
            for block in tool_response
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if not texts:
            return tool_response
        tool_response = "".join(t for t in texts if isinstance(t, str))
    if isinstance(tool_response, str):
        try:
            return json.loads(tool_response)
        except ValueError:
            return tool_response
    return tool_response


def _nonzero_exit(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value != 0


def _mcp_dict(tool_response: object) -> dict | None:
    decoded = _unwrap_text(tool_response)
    # MCP tool responses may wrap the envelope in a {"content": [...]} dict.
    if isinstance(decoded, dict) and isinstance(decoded.get("content"), list):
        inner = _unwrap_text(decoded["content"])
        if isinstance(inner, dict):
            decoded = inner
    return decoded if isinstance(decoded, dict) else None


def _envelope_failed(data: dict) -> bool:
    if data.get("status") == "error":
        return True
    result = data.get("result")
    if isinstance(result, dict):
        if result.get("success") is False:
            return True
        if _nonzero_exit(result.get("exit_code")):
            return True
    return False


def _shell_runner_failed(data: dict) -> bool:
    # Unwrap the {"tool","status","result"} envelope if present.
    if data.get("status") == "error":
        return True
    result = data.get("result")
    payload = result if isinstance(result, dict) else data
    if payload.get("decision") in ("denied", "prompt_required"):
        return True
    # status "running" (background) counts as succeeded.
    return _nonzero_exit(payload.get("exit_code"))


def _bash_failed(data: dict) -> bool:
    if data.get("is_error") or data.get("interrupted"):
        return True
    return _nonzero_exit(data.get("exit_code"))


def _response_succeeded(tool_name: str, tool_response: object) -> bool:
    """Best-effort success verdict; unparseable responses count as success.

    Deliberately NO substring heuristics such as ``"error" in text``: they
    misfire on successful output that merely mentions the word (e.g. pytest
    node ids like ``test_error_paths``).
    """
    try:
        if tool_name == "Bash":
            if isinstance(tool_response, dict):
                return not _bash_failed(tool_response)
            return True
        data = _mcp_dict(tool_response)
        if data is None:
            return True
        if tool_name == "mcp__shell-runner__shell_execute":
            return not _shell_runner_failed(data)
        if data.get("is_error"):
            return False
        return not _envelope_failed(data)
    except Exception:
        return True


def extract_tracked_command(
    tool_name: str, tool_input: dict, tool_response: object
) -> TrackedCommand | None:
    """Return the command a tool call represents, or None if untracked."""
    try:
        if not isinstance(tool_input, dict):
            return None
        if tool_name in _SHELL_TOOLS:
            found = _extract_shell(tool_input)
        elif tool_name == _PIXI_TOOL:
            found = _extract_pixi(tool_input)
        elif tool_name == _GIT_TOOL:
            found = _extract_git(tool_input)
        else:
            return None
        if found is None:
            return None
        command, base = found
        return TrackedCommand(
            command=command[:_MAX_COMMAND_LEN],
            command_base=base,
            succeeded=_response_succeeded(tool_name, tool_response),
        )
    except Exception:
        return None
