"""Parse a Claude Code agent transcript into a compact tool-use summary.

Issue #138: the SubagentStop hook's original parser checked
``entry.get("type") == "tool_use"`` at the TOP LEVEL of each JSONL line, but
real transcripts nest tool calls at
``entry["message"]["content"][i]["type"] == "tool_use"`` (name at
``.name``). Top-level ``type`` is only ever ``"user"``/``"assistant"``, so
that branch was dead code: ``tool_count`` was always 0, and ``success`` --
initialised ``True`` and only ever falsified inside that same dead branch --
always reported ``True``.

``summarize_transcript`` fixes this by walking the real nested shape while
still tolerating the legacy flat shape as a schema-drift fallback, and by
reporting ``success=None`` (indeterminate) whenever the transcript could not
be read or contained zero usable entries, rather than silently defaulting to
``True``.

STDLIB ONLY. This module is imported by
``~/.claude/hooks/session_intelligence_agent_stop.py``, which runs under
plain ``python3`` OUTSIDE this project's pixi environment. It must not
import pydantic, any other third-party package, or anything from this
repo's own ``src`` package tree -- any such import would raise an
``ImportError`` the hook has no way to recover from.
"""

from __future__ import annotations

import json
import os

# Public keys always present in the returned dict, used to build every
# early-return (unreadable/unparseable transcript) result consistently.
_EMPTY_RESULT_BASE = {
    "tool_count": 0,
    "tools_used": [],
    "errors": [],
}


def _indeterminate_result(parse_error: str) -> dict:
    """Build the result for a transcript that could not be read/parsed."""
    return {
        **_EMPTY_RESULT_BASE,
        "success": None,
        "entries_parsed": 0,
        "parse_error": parse_error,
    }


def summarize_transcript(path: str) -> dict:
    """Summarize tool usage and outcome from a JSONL agent transcript.

    Returns a dict with keys: ``tool_count`` (int), ``tools_used``
    (deduplicated list, first-seen order), ``errors`` (list of error
    markers), ``success`` (True/False/None), ``entries_parsed`` (int),
    ``parse_error`` (str or None). Never raises.
    """
    if not path:
        return _indeterminate_result("no transcript path provided")

    if not os.path.exists(path):
        return _indeterminate_result(f"transcript path does not exist: {path}")

    try:
        with open(path, encoding="utf-8") as handle:
            raw_lines = handle.readlines()
    except OSError as exc:
        return _indeterminate_result(f"could not read transcript: {exc}")

    tool_count = 0
    tools_used: list[str] = []
    errors: list[str] = []
    entries_parsed = 0

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (ValueError, TypeError):
            continue

        entries_parsed += 1
        if not isinstance(entry, dict):
            continue

        message = entry.get("message")
        content = message.get("content") if isinstance(message, dict) else None

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "tool_use":
                    tool_count += 1
                    name = block.get("name")
                    if name and name not in tools_used:
                        tools_used.append(name)
                elif block_type == "tool_result" and block.get("is_error"):
                    errors.append(str(block.get("tool_use_id", "unknown")))
        else:
            # Legacy flat shape: tool_use/tool_result at the top level.
            entry_type = entry.get("type")
            if entry_type == "tool_use":
                tool_count += 1
                name = entry.get("name")
                if name and name not in tools_used:
                    tools_used.append(name)
            elif entry_type == "tool_result" and entry.get("is_error"):
                errors.append(str(entry.get("tool_use_id", "unknown")))

    if entries_parsed == 0:
        return _indeterminate_result("no usable entries found in transcript")

    success = False if errors else True

    return {
        "tool_count": tool_count,
        "tools_used": tools_used,
        "errors": errors,
        "success": success,
        "entries_parsed": entries_parsed,
        "parse_error": None,
    }
