"""
Regression tests for issue #175: `summarize_transcript` must judge
`success` from the transcript's FINAL STATE, not from "any tool_result in
the whole transcript errored".

After #167, `success` was set to `False` whenever ANY tool_result carried
`is_error`, even if the agent recovered (retried, used an alternate tool,
etc.) and went on to finish successfully. PreToolUse hooks here routinely
reject individual tool calls, so this made agents that completed real work
(e.g. a merged PR) get stored as ERROR.

https://github.com/Claire-s-Monster/session-intelligence/issues/175

New rule verified here:
  - `success = False` only if (a) the LAST tool_result in the transcript is
    an error, OR (b) the transcript contains tool calls but no assistant
    text block follows the last tool activity (no final answer).
  - Otherwise `success = True`, even if earlier tool_results errored.
  - `errors`/`error_count` still report every errored tool_result seen,
    regardless of the final `success` verdict.
  - The pre-existing indeterminate (`success=None`) semantics for
    unreadable/missing/empty transcripts are unchanged.
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


def _final_text(parent: str, text: str = "Done.") -> dict:
    return {
        "parentUuid": parent,
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "uuid": f"final-{parent}",
    }


def _user_text(parent: str, text: str = "Reminder: stay on task.") -> dict:
    """A user-role message carrying a `type: "text"` content block, e.g. a
    hook-injected/harness nudge. Must NOT count as the agent's final answer."""
    return {
        "parentUuid": parent,
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        "uuid": f"usertext-{parent}",
    }


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


# ---------------------------------------------------------------------------
# 1. Recovered error then final text -> success True, error preserved
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_recovered_error_then_final_text_is_success(tmp_path):
    """An early tool_result error followed by a successful retry and a
    final assistant text message must report success=True. The error must
    still be visible in `errors`/`error_count` for debugging."""
    lines = [
        _tool_use("toolu_1", "Bash", None),
        _tool_result("toolu_1", "a-toolu_1", is_error=True, text="boom"),
        _tool_use("toolu_2", "Bash", "u-toolu_1"),
        _tool_result("toolu_2", "a-toolu_2", is_error=False),
        _final_text("u-toolu_2"),
    ]
    transcript_path = tmp_path / "recovered.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is True
    assert result["error_count"] == 1
    assert len(result["errors"]) == 1


# ---------------------------------------------------------------------------
# 2. Last tool_result errored -> success False
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_last_tool_result_errored_is_failure(tmp_path):
    lines = [
        _tool_use("toolu_1", "Bash", None),
        _tool_result("toolu_1", "a-toolu_1", is_error=False),
        _tool_use("toolu_2", "Bash", "u-toolu_1"),
        _tool_result("toolu_2", "a-toolu_2", is_error=True, text="fatal"),
    ]
    transcript_path = tmp_path / "last_errored.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is False
    assert result["error_count"] == 1


# ---------------------------------------------------------------------------
# 3. Errored then tool succeeds but no final text -> success False
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_recovered_error_but_no_final_text_is_failure(tmp_path):
    """The last tool_result is NOT an error, but the transcript ends right
    after it with no assistant text: the agent never produced a final
    answer, so this must still report success=False."""
    lines = [
        _tool_use("toolu_1", "Bash", None),
        _tool_result("toolu_1", "a-toolu_1", is_error=True, text="boom"),
        _tool_use("toolu_2", "Bash", "u-toolu_1"),
        _tool_result("toolu_2", "a-toolu_2", is_error=False),
    ]
    transcript_path = tmp_path / "no_final_text.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is False
    assert result["error_count"] == 1


# ---------------------------------------------------------------------------
# 4. No errors + final text -> success True
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_clean_run_with_final_text_is_success(tmp_path):
    lines = [
        _tool_use("toolu_1", "Bash", None),
        _tool_result("toolu_1", "a-toolu_1", is_error=False),
        _final_text("u-toolu_1"),
    ]
    transcript_path = tmp_path / "clean.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is True
    assert result["error_count"] == 0
    assert result["errors"] == []


# ---------------------------------------------------------------------------
# 5. Existing indeterminate cases unchanged
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_nonexistent_path_still_indeterminate():
    result = summarize_transcript("/nonexistent/path/does-not-exist.jsonl")

    assert result["success"] is None
    assert result["entries_parsed"] == 0
    assert result["parse_error"] is not None
    assert result["error_count"] == 0


@pytest.mark.regression
def test_empty_string_path_still_indeterminate():
    result = summarize_transcript("")

    assert result["success"] is None
    assert result["error_count"] == 0


@pytest.mark.regression
def test_blank_and_invalid_json_only_still_indeterminate(tmp_path):
    transcript_path = tmp_path / "garbage.jsonl"
    transcript_path.write_text("\n\n   \nnot json at all\n{not: valid, json\n\n")

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is None
    assert result["entries_parsed"] == 0
    assert result["error_count"] == 0


# ---------------------------------------------------------------------------
# 6a. User-role text block (hook-injected/harness nudge) does not count as
# the agent's final answer -> success False
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_user_role_text_after_tool_activity_does_not_count_as_final_answer(tmp_path):
    """tool_use -> tool_result (ok) -> a USER message containing a
    `type: "text"` block, with no later assistant text. This must still
    report success=False: only assistant-role text satisfies rule (b)."""
    lines = [
        _tool_use("toolu_1", "Bash", None),
        _tool_result("toolu_1", "a-toolu_1", is_error=False),
        _user_text("u-toolu_1"),
    ]
    transcript_path = tmp_path / "user_text_no_final_answer.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is False
    assert result["error_count"] == 0


# ---------------------------------------------------------------------------
# 6. No tool activity at all (plain conversation) -> success True
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_no_tool_activity_with_text_is_success(tmp_path):
    """A transcript with zero tool calls (e.g. a direct text-only answer)
    must not be penalized by rule (b), which only applies when there was
    tool activity."""
    lines = [
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi."}]},
            "uuid": "a1",
        }
    ]
    transcript_path = tmp_path / "no_tools.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is True
    assert result["tool_count"] == 0
