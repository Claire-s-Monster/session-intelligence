"""
Regression tests for issue #138: SubagentStop hook always reports 0 tools
and success=True because its transcript parser checks `entry.get('type')
== 'tool_use'` at the TOP LEVEL, but real transcripts nest tool calls at
`entry['message']['content'][i]['type'] == 'tool_use'` (name at `.name`).
Top-level `type` is only ever "user"/"assistant", so the old parser's
branch is dead code: tool_count is always 0, and `success` -- initialised
True and only ever falsified inside that same dead branch -- always
reports True.

https://github.com/Claire-s-Monster/session-intelligence/issues/138

Verifies the fix: a new stdlib-only module `src/hooks/transcript_summary.py`
exposing `summarize_transcript(path: str) -> dict` that:
  - walks the PRIMARY nested shape (`message.content[].type == 'tool_use'`,
    name at `.name`; `message.content[].type == 'tool_result'`, error at
    `.is_error`),
  - also tolerates the LEGACY FLAT shape (top-level `type == 'tool_use'`
    with top-level `name`), for schema-drift tolerance,
  - tolerates `message.content` being a plain string (no tool activity)
    without raising,
  - reports `success=None` (INDETERMINATE) whenever the transcript could
    not be read/parsed at all or contained zero usable entries -- never
    defaulting to True the way the old parser did,
  - is importable with plain stdlib `python3` outside the pixi env (the
    SubagentStop hook runs there), so it must not import anything
    third-party or from this repo's own `src` package tree.

These tests target `src/hooks/transcript_summary.py`, which does not exist
yet. Collection therefore fails for every test in this module until the
fix lands -- that failure is expected and is the point of this file.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from hooks.transcript_summary import summarize_transcript

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
NESTED_FIXTURE = FIXTURES_DIR / "agent_transcript_nested.jsonl"

# Real, observed nested-fixture tool_use count (matches the 56 nested
# tool_use blocks the old top-level parser returned 0 for on the real
# agent-a45fb09c7fe9b051b.jsonl transcript this fixture is trimmed from).
SCALE_TOOL_USE_COUNT = 56


# ---------------------------------------------------------------------------
# Helpers to build synthetic transcripts
# ---------------------------------------------------------------------------


def _nested_tool_use_line(tool_id: str, name: str, parent: str | None) -> dict:
    return {
        "parentUuid": parent,
        "isSidechain": True,
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": {}}],
        },
        "uuid": f"a-{tool_id}",
    }


def _nested_tool_result_line(
    tool_id: str, parent: str, is_error: bool = False, text: str = "ok"
) -> dict:
    result_block: dict = {
        "tool_use_id": tool_id,
        "type": "tool_result",
        "content": [{"type": "text", "text": text}],
    }
    if is_error:
        result_block["is_error"] = True
    return {
        "parentUuid": parent,
        "isSidechain": True,
        "type": "user",
        "message": {"role": "user", "content": [result_block]},
        "uuid": f"u-{tool_id}",
    }


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


# ---------------------------------------------------------------------------
# 1. Nested fixture: real tool_use count and names
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_nested_fixture_reports_real_tool_count_and_names():
    """tool_count must equal the 3 nested tool_use blocks actually present
    in the fixture; tools_used must contain the real tool names used."""
    result = summarize_transcript(str(NESTED_FIXTURE))

    assert result["tool_count"] == 3
    assert "mcp__session-intelligence__execute_tool" in result["tools_used"]
    assert "Read" in result["tools_used"]


# ---------------------------------------------------------------------------
# 2. De-duplication of tools_used, tool_count keeps every occurrence
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_nested_fixture_dedups_tools_used_but_not_tool_count():
    """The fixture uses 'mcp__session-intelligence__execute_tool' twice
    (first and third tool_use blocks) and 'Read' once: tool_count counts
    all 3 occurrences, but tools_used lists each name exactly once, in
    first-seen order."""
    result = summarize_transcript(str(NESTED_FIXTURE))

    assert result["tool_count"] == 3
    assert result["tools_used"] == ["mcp__session-intelligence__execute_tool", "Read"]


# ---------------------------------------------------------------------------
# 3. Scale case: 56 nested tool_use entries (the real observed count)
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_scale_56_nested_tool_use_entries_all_counted(tmp_path):
    """The old top-level parser returned 0 tools for this agent's real
    transcript, which contained 56 nested tool_use blocks. A synthetic
    transcript with exactly 56 nested tool_use/tool_result pairs must
    yield tool_count == 56."""
    lines: list[dict] = [
        {
            "parentUuid": None,
            "isSidechain": True,
            "type": "user",
            "message": {"role": "user", "content": "start"},
            "uuid": "root",
        }
    ]
    parent = "root"
    for i in range(SCALE_TOOL_USE_COUNT):
        tool_id = f"toolu_{i}"
        lines.append(_nested_tool_use_line(tool_id, "Bash", parent))
        lines.append(_nested_tool_result_line(tool_id, f"a-{tool_id}"))
        parent = f"u-{tool_id}"

    transcript_path = tmp_path / "scale_transcript.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["tool_count"] == SCALE_TOOL_USE_COUNT


# ---------------------------------------------------------------------------
# 4. Nested tool_result with is_error truthy -> success False
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_nested_tool_result_is_error_flips_success_false(tmp_path):
    lines = [
        _nested_tool_use_line("toolu_e1", "Bash", None),
        _nested_tool_result_line("toolu_e1", "a-toolu_e1", is_error=True, text="boom"),
    ]
    transcript_path = tmp_path / "error_transcript.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is False
    assert len(result["errors"]) > 0


# ---------------------------------------------------------------------------
# 5. Clean nested transcript, no errors -> success True
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_clean_nested_fixture_reports_success_true():
    result = summarize_transcript(str(NESTED_FIXTURE))

    assert result["success"] is True
    assert result["errors"] == []


# ---------------------------------------------------------------------------
# 6/7. Missing / empty path -> indeterminate, not a silent True
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_nonexistent_path_is_indeterminate_not_success():
    result = summarize_transcript("/nonexistent/path/does-not-exist.jsonl")

    assert result["success"] is None
    assert result["entries_parsed"] == 0
    assert result["parse_error"] is not None


@pytest.mark.regression
def test_empty_string_path_is_indeterminate_not_success():
    result = summarize_transcript("")

    assert result["success"] is None
    assert result["entries_parsed"] == 0
    assert result["parse_error"] is not None


# ---------------------------------------------------------------------------
# 8. Blank lines / invalid JSON only -> indeterminate, NOT True
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_blank_and_invalid_json_only_is_indeterminate(tmp_path):
    """A readable file with zero usable entries must report success=None,
    never the old parser's silent success=True default."""
    transcript_path = tmp_path / "garbage.jsonl"
    transcript_path.write_text("\n\n   \nnot json at all\n{not: valid, json\n\n")

    result = summarize_transcript(str(transcript_path))

    assert result["success"] is None
    assert result["entries_parsed"] == 0


# ---------------------------------------------------------------------------
# 9. Legacy FLAT shape must also still parse
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_legacy_flat_shape_still_parses(tmp_path):
    """Some historical/alternate transcripts may carry tool_use at the top
    level directly (`type` == 'tool_use', `name` at top level too). This
    must still be tolerated as a schema-drift fallback."""
    lines = [
        {"type": "tool_use", "id": "toolu_flat_1", "name": "Grep", "input": {}},
        {"type": "tool_use", "id": "toolu_flat_2", "name": "Read", "input": {}},
    ]
    transcript_path = tmp_path / "legacy_flat.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))

    assert result["tool_count"] > 0
    assert "Grep" in result["tools_used"]


# ---------------------------------------------------------------------------
# 10. message.content as a plain string must not raise
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_message_content_plain_string_does_not_raise(tmp_path):
    lines = [
        {
            "type": "user",
            "message": {"role": "user", "content": "just a plain string, no list here"},
            "uuid": "u1",
        }
    ]
    transcript_path = tmp_path / "plain_string.jsonl"
    _write_jsonl(transcript_path, lines)

    result = summarize_transcript(str(transcript_path))  # must not raise

    assert result["tool_count"] == 0
    assert result["entries_parsed"] == 1


# ---------------------------------------------------------------------------
# 11. Structural guard: the hook module must be stdlib-only, importable
# outside the pixi env by plain python3.
# ---------------------------------------------------------------------------

_STDLIB_ALLOWLIST = {
    "__future__",
    "abc",
    "argparse",
    "collections",
    "dataclasses",
    "datetime",
    "functools",
    "io",
    "itertools",
    "json",
    "logging",
    "os",
    "pathlib",
    "re",
    "sys",
    "typing",
}


@pytest.mark.regression
def test_transcript_summary_module_is_stdlib_only():
    """Parse src/hooks/transcript_summary.py with `ast` and assert every
    import targets stdlib only -- no third-party packages, no `src.`
    imports, no relative imports back into this package tree. The
    SubagentStop hook runs under plain python3 outside the pixi env, so a
    third-party (e.g. pydantic) or intra-repo import would break it at
    runtime with an ImportError the hook can't recover from."""
    module_path = Path(__file__).parent.parent.parent / "src" / "hooks" / "transcript_summary.py"
    source = module_path.read_text()
    tree = ast.parse(source, filename=str(module_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                assert top_level in _STDLIB_ALLOWLIST, f"non-stdlib import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import not allowed: level={node.level}"
            assert node.module is not None, "bare relative import not allowed"
            top_level = node.module.split(".")[0]
            assert top_level in _STDLIB_ALLOWLIST, f"non-stdlib import: {node.module}"
