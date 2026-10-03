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

Issue #175: after #167, ``success`` was set to ``False`` whenever ANY
tool_result in the whole transcript carried ``is_error``. PreToolUse hooks
routinely reject individual tool calls that the agent then recovers from
(retries, alternate tool, etc.), so an agent that finished its work cleanly
was still recorded as ERROR. ``success`` now reflects the transcript's
*final state* rather than its full history:

- ``False`` only if (a) the LAST tool_result in the transcript is an error,
  or (b) the transcript contains at least one tool call/result but no
  assistant text block follows the last tool activity (the agent never
  produced a final answer -- an incomplete/truncated run). Only text
  blocks in a message whose role is ``"assistant"`` count for (b); a
  user-role message carrying a ``type: "text"`` block (e.g. a
  hook-injected/harness nudge) does not.
- ``True`` otherwise, even if earlier tool_results in the same transcript
  errored and were recovered from.
- ``errors``/``error_count`` still report every errored tool_result seen,
  regardless of whether it affected the final ``success`` verdict, so
  recovered errors remain visible for debugging/learning extraction.
- The pre-existing ``None`` (indeterminate) semantics for unreadable,
  missing, or empty transcripts are unchanged by this issue.

Issue #186: SubagentStop fires while the transcript still ends with the
assistant's ``SubagentHandback`` tool_use -- its tool_result has not been
written yet. Rule (b) therefore scored every normally-finishing agent False,
because #181 only recognised the handback's tool_result. Invoking
``SubagentHandback`` is itself the agent's final answer, so the tool_use
(nested and legacy flat shapes) now counts as one; an errored result that
follows still wins via rule (a).

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

# Issue #181: the tool whose successful result is the agent's final answer.
_HANDBACK_TOOL_NAME = "SubagentHandback"

# Public keys always present in the returned dict, used to build every
# early-return (unreadable/unparseable transcript) result consistently.
_EMPTY_RESULT_BASE = {
    "tool_count": 0,
    "tools_used": [],
    "errors": [],
    "error_count": 0,
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
    markers), ``error_count`` (int, total errored tool_results -- not
    truncated), ``success`` (True/False/None, see issue #175 for the
    final-state rule), ``entries_parsed`` (int), ``parse_error`` (str or
    None). Never raises.
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

    # Issue #175 final-state tracking. `last_tool_result_is_error` reflects
    # only the most recently seen tool_result block (rule (a)); it is left
    # untouched by tool_use blocks, so it always answers "was the last
    # *resolved* tool_result an error", even across an unresolved trailing
    # tool_use. `saw_text_since_last_tool_event` is reset False on every
    # tool_use/tool_result and set True on every text block, so it answers
    # "did the agent produce a final answer after its last tool activity"
    # (rule (b)). Issue #181: a successful SubagentHandback result is the
    # agent's final answer (a tool_use/tool_result pair, not a text block),
    # so it sets that flag right after the reset. Issue #186: SubagentStop
    # fires before that result is written, so the SubagentHandback tool_use
    # itself also sets the flag right after the reset.
    last_tool_result_is_error: bool | None = None
    saw_text_since_last_tool_event = False
    any_tool_event = False
    handback_ids: set[str] = set()

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

        # Issue #175 follow-up: rule (b) is "assistant text after last tool
        # activity" specifically. Hook-injected/harness nudges and other
        # user-role messages can carry a `type: "text"` content block too
        # (e.g. a synthetic reminder), and those must NOT count as the
        # agent's final answer. Prefer the message's own `role`; fall back
        # to the entry's top-level `type` only when `role` is absent.
        message_role = message.get("role") if isinstance(message, dict) else None
        if message_role is None:
            message_role = entry.get("type")
        is_assistant_message = message_role == "assistant"

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "tool_use":
                    tool_count += 1
                    any_tool_event = True
                    saw_text_since_last_tool_event = False
                    name = block.get("name")
                    if name and name not in tools_used:
                        tools_used.append(name)
                    if name == _HANDBACK_TOOL_NAME:
                        saw_text_since_last_tool_event = True
                        if isinstance(block.get("id"), str):
                            handback_ids.add(block["id"])
                elif block_type == "tool_result":
                    any_tool_event = True
                    saw_text_since_last_tool_event = False
                    is_error = bool(block.get("is_error"))
                    last_tool_result_is_error = is_error
                    if is_error:
                        errors.append(str(block.get("tool_use_id", "unknown")))
                    elif (
                        isinstance(block.get("tool_use_id"), str)
                        and block["tool_use_id"] in handback_ids
                    ):
                        saw_text_since_last_tool_event = True
                elif block_type == "text" and is_assistant_message:
                    saw_text_since_last_tool_event = True
        else:
            # Legacy flat shape: tool_use/tool_result at the top level.
            entry_type = entry.get("type")
            if entry_type == "tool_use":
                tool_count += 1
                any_tool_event = True
                saw_text_since_last_tool_event = False
                name = entry.get("name")
                if name and name not in tools_used:
                    tools_used.append(name)
                if name == _HANDBACK_TOOL_NAME:
                    saw_text_since_last_tool_event = True
                    if isinstance(entry.get("id"), str):
                        handback_ids.add(entry["id"])
            elif entry_type == "tool_result":
                any_tool_event = True
                saw_text_since_last_tool_event = False
                is_error = bool(entry.get("is_error"))
                last_tool_result_is_error = is_error
                if is_error:
                    errors.append(str(entry.get("tool_use_id", "unknown")))
                elif (
                    isinstance(entry.get("tool_use_id"), str)
                    and entry["tool_use_id"] in handback_ids
                ):
                    saw_text_since_last_tool_event = True
            elif entry_type == "text":
                saw_text_since_last_tool_event = True

    if entries_parsed == 0:
        return _indeterminate_result("no usable entries found in transcript")

    # Issue #175: final-state judgment, not "any error anywhere => False".
    if last_tool_result_is_error:
        success = False
    elif any_tool_event and not saw_text_since_last_tool_event:
        success = False
    else:
        success = True

    return {
        "tool_count": tool_count,
        "tools_used": tools_used,
        "errors": errors,
        "error_count": len(errors),
        "success": success,
        "entries_parsed": entries_parsed,
        "parse_error": None,
    }
