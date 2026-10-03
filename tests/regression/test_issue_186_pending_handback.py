"""
Regression tests for issue #186: SubagentStop fires while the transcript still
ends with the assistant's `SubagentHandback` tool_use -- its tool_result is not
yet written. That pending handback IS the agent's final answer, so rule (b) of
#175 must not score the agent False.

https://github.com/Claire-s-Monster/session-intelligence/issues/186
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hooks.transcript_summary import summarize_transcript


def _tool_use(tool_id: str, name: str, parent: str | None) -> dict:
    return {
        "parentUuid": parent,
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": {}}],
        },
        "uuid": f"a-{tool_id}",
    }


def _tool_result(tool_id: str, parent: str, is_error: bool = False) -> dict:
    block: dict = {
        "tool_use_id": tool_id,
        "type": "tool_result",
        "content": [{"type": "text", "text": "ok"}],
    }
    if is_error:
        block["is_error"] = True
    return {
        "parentUuid": parent,
        "type": "user",
        "message": {"role": "user", "content": [block]},
        "uuid": f"u-{tool_id}",
    }


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


@pytest.mark.regression
def test_pending_handback_tool_use_is_final_answer_nested(tmp_path):
    lines = [
        _tool_use("toolu_r", "Read", None),
        _tool_result("toolu_r", "a-toolu_r"),
        _tool_use("h1", "SubagentHandback", "u-toolu_r"),
    ]
    transcript_path = tmp_path / "pending_nested.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is True
    assert result["error_count"] == 0


@pytest.mark.regression
def test_pending_handback_tool_use_is_final_answer_flat(tmp_path):
    lines = [
        {"type": "tool_use", "id": "toolu_r", "name": "Read"},
        {"type": "tool_result", "tool_use_id": "toolu_r"},
        {"type": "tool_use", "id": "h1", "name": "SubagentHandback"},
    ]
    transcript_path = tmp_path / "pending_flat.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is True


@pytest.mark.regression
def test_recovered_error_then_pending_handback_is_success(tmp_path):
    lines = [
        _tool_use("toolu_1", "Bash", None),
        _tool_result("toolu_1", "a-toolu_1", is_error=True),
        _tool_use("toolu_2", "Bash", "u-toolu_1"),
        _tool_result("toolu_2", "a-toolu_2"),
        _tool_use("h1", "SubagentHandback", "u-toolu_2"),
    ]
    transcript_path = tmp_path / "recovered_pending.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is True
    assert result["error_count"] == 1
    assert result["errors"] == ["toolu_1"]


@pytest.mark.regression
def test_pending_handback_then_errored_result_is_failure(tmp_path):
    lines = [
        _tool_use("h1", "SubagentHandback", None),
        _tool_result("h1", "a-h1", is_error=True),
    ]
    transcript_path = tmp_path / "pending_then_error.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is False
    assert result["error_count"] == 1


@pytest.mark.regression
def test_pending_non_handback_tool_use_is_still_failure(tmp_path):
    lines = [
        _tool_use("toolu_r", "Read", None),
    ]
    transcript_path = tmp_path / "pending_read.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is False
