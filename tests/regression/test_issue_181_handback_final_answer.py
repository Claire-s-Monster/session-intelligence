"""
Regression tests for issue #181: a successful `SubagentHandback` tool_result
is the agent's final answer, so `summarize_transcript` must not report
`success=False` (rule (b) of #175) for a transcript that ends with it.

`SubagentHandback` is a tool_use/tool_result pair, not an assistant text
block, so every handback-ending agent was being stored as ERROR.

https://github.com/Claire-s-Monster/session-intelligence/issues/181

Rules verified here:
  - A non-error tool_result answering a SubagentHandback tool_use counts as
    the final answer (success True) in both nested and legacy flat shapes.
  - An errored handback result is still False via rule (a).
  - The handback is not sticky: later tool activity without text -> False.
  - Results answering any other tool (e.g. Read) are unchanged.
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


def _tool_result(tool_id: str, parent: str, is_error: bool = False, text: str = "ok") -> dict:
    block: dict = {
        "tool_use_id": tool_id,
        "type": "tool_result",
        "content": [{"type": "text", "text": text}],
    }
    if is_error:
        block["is_error"] = True
    return {
        "parentUuid": parent,
        "type": "user",
        "message": {"role": "user", "content": [block]},
        "uuid": f"u-{tool_id}",
    }


def _trailing_non_message_lines() -> list[dict]:
    """Entries with no message content (attachment/hook records) that follow
    the handback in real transcripts."""
    return [
        {"type": "attachment", "uuid": "att-1", "attachment": {"type": "hook_success"}},
        {"type": "system", "uuid": "sys-1", "subtype": "stop_hook_summary"},
    ]


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


@pytest.mark.regression
def test_successful_handback_is_final_answer(tmp_path):
    lines = [
        _tool_use("toolu_r", "Read", None),
        _tool_result("toolu_r", "a-toolu_r"),
        _tool_use("h1", "SubagentHandback", "u-toolu_r"),
        _tool_result("h1", "a-h1"),
        *_trailing_non_message_lines(),
    ]
    transcript_path = tmp_path / "handback_ok.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is True
    assert result["error_count"] == 0


@pytest.mark.regression
def test_errored_handback_is_failure(tmp_path):
    lines = [
        _tool_use("toolu_r", "Read", None),
        _tool_result("toolu_r", "a-toolu_r"),
        _tool_use("h1", "SubagentHandback", "u-toolu_r"),
        _tool_result("h1", "a-h1", is_error=True, text="handback rejected"),
        *_trailing_non_message_lines(),
    ]
    transcript_path = tmp_path / "handback_error.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is False
    assert result["error_count"] == 1


@pytest.mark.regression
def test_handback_followed_by_more_tool_work_without_text_is_failure(tmp_path):
    lines = [
        _tool_use("h1", "SubagentHandback", None),
        _tool_result("h1", "a-h1"),
        _tool_use("toolu_r", "Read", "u-h1"),
        _tool_result("toolu_r", "a-toolu_r"),
    ]
    transcript_path = tmp_path / "handback_then_tools.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is False


@pytest.mark.regression
def test_recovered_error_then_successful_handback_is_success(tmp_path):
    lines = [
        _tool_use("toolu_1", "Bash", None),
        _tool_result("toolu_1", "a-toolu_1", is_error=True, text="boom"),
        _tool_use("toolu_2", "Bash", "u-toolu_1"),
        _tool_result("toolu_2", "a-toolu_2"),
        _tool_use("h1", "SubagentHandback", "u-toolu_2"),
        _tool_result("h1", "a-h1"),
    ]
    transcript_path = tmp_path / "recovered_then_handback.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is True
    assert result["error_count"] == 1


@pytest.mark.regression
def test_legacy_flat_shape_successful_handback_is_success(tmp_path):
    lines = [
        {"type": "tool_use", "id": "toolu_r", "name": "Read"},
        {"type": "tool_result", "tool_use_id": "toolu_r"},
        {"type": "tool_use", "id": "h1", "name": "SubagentHandback"},
        {"type": "tool_result", "tool_use_id": "h1"},
    ]
    transcript_path = tmp_path / "flat_handback.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is True
    assert result["error_count"] == 0


@pytest.mark.regression
def test_non_string_ids_do_not_raise_nested_shape(tmp_path):
    lines = [
        _tool_use("toolu_r", "Read", None),
        _tool_result("toolu_r", "a-toolu_r"),
        _tool_use(["h1"], "SubagentHandback", "u-toolu_r"),
        _tool_result(["h1"], "a-h1"),
    ]
    transcript_path = tmp_path / "nonstring_nested.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is False


@pytest.mark.regression
def test_non_string_ids_do_not_raise_flat_shape(tmp_path):
    lines = [
        {"type": "tool_use", "id": ["h1"], "name": "SubagentHandback"},
        {"type": "tool_result", "tool_use_id": ["h1"]},
    ]
    transcript_path = tmp_path / "nonstring_flat.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is False


@pytest.mark.regression
def test_non_handback_tool_result_at_end_is_still_failure(tmp_path):
    lines = [
        _tool_use("h1", "SubagentHandback", None),
        _tool_result("h1", "a-h1"),
        _tool_use("toolu_r", "Read", "u-h1"),
        _tool_result("toolu_r", "a-toolu_r"),
        *_trailing_non_message_lines(),
    ]
    transcript_path = tmp_path / "non_handback_end.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is False
