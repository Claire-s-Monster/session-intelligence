"""Regression tests for issue #141: outline / section / window retrieval of notebook bodies."""

from __future__ import annotations

import pytest

from core.notebook_windowing import (
    apply_window,
    extract_section,
    find_matches,
    parse_outline,
    validate_window_params,
)
from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.sqlite import SQLiteBackend

BODY = (
    "# Title\n"
    "intro\n"
    "## Decisions\n"
    "chose A\n"
    "### Detail\n"
    "deep dive\n"
    "## Risks\n"
    "```\n"
    "# not a heading\n"
    "```\n"
    "risk text\n"
    "## Risk Mitigation\n"
    "mitigate\n"
)


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "test_issue_141.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def engine(tmp_path, db):
    return SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )


def _stub_rows(db, monkeypatch, rows):
    async def fake(**kwargs):
        return [dict(r) for r in rows]

    monkeypatch.setattr(db, "query_session_summaries", fake)


def _row(**extra):
    return {"session_id": "s1", "title": "T", "tags": [], "project_name": "p", **extra}


class TestHelpers:
    def test_outline_levels_offsets_and_fence_ignored(self):
        outline = parse_outline(BODY)
        assert [(h["heading"], h["level"]) for h in outline] == [
            ("Title", 1),
            ("Decisions", 2),
            ("Detail", 3),
            ("Risks", 2),
            ("Risk Mitigation", 2),
        ]
        for h in outline:
            assert BODY[h["offset"] :].startswith("#" * h["level"] + " " + h["heading"])
        assert outline[0]["chars"] == len(BODY)
        assert outline[1]["chars"] == len("## Decisions\nchose A\n### Detail\ndeep dive\n")

    def test_section_exact_case_insensitive(self):
        text, err = extract_section(BODY, "  decisions ")
        assert err is None
        assert text == "## Decisions\nchose A\n### Detail\ndeep dive\n"

    def test_section_includes_nested_until_same_level(self):
        text, _ = extract_section(BODY, "Decisions")
        assert "### Detail" in text
        assert "## Risks" not in text

    def test_section_exact_beats_substring(self):
        text, err = extract_section(BODY, "Risks")
        assert err is None
        assert text.startswith("## Risks\n")
        assert "# not a heading" in text

    def test_section_unique_substring(self):
        text, err = extract_section(BODY, "mitig")
        assert err is None
        assert text == "## Risk Mitigation\nmitigate\n"

    def test_section_ambiguous(self):
        text, err = extract_section(BODY, "risk")
        assert text == ""
        assert err is not None
        assert "ambiguous" in err
        assert "Decisions" in err

    def test_section_missing(self):
        text, err = extract_section(BODY, "nonexistent")
        assert text == ""
        assert err is not None
        assert "Risks" in err

    def test_error_lists_capped_at_50(self):
        body = "".join(f"## H{i}\nx\n" for i in range(80))
        _, err = extract_section(body, "zzz")
        assert err is not None
        assert "H49" in err
        assert "H50" not in err

    def test_window_metadata(self):
        window, meta = apply_window("abcdefghij", 2, 3)
        assert window == "cde"
        assert meta == {
            "total_chars": 10,
            "offset": 2,
            "returned_chars": 3,
            "elided_chars": 5,
            "next_offset": 5,
        }

    def test_window_end_has_null_next_offset(self):
        window, meta = apply_window("abcdefghij", 8, 5)
        assert window == "ij"
        assert meta["elided_chars"] == 0
        assert meta["next_offset"] is None

    def test_window_offset_past_end(self):
        window, meta = apply_window("abc", 10, 5)
        assert window == ""
        assert meta["returned_chars"] == 0
        assert meta["elided_chars"] == 0
        assert meta["next_offset"] is None

    def test_validate(self):
        validate_window_params(0, None)
        validate_window_params(5, 1)
        with pytest.raises(ValueError):
            validate_window_params(-1, None)
        with pytest.raises(ValueError):
            validate_window_params(0, 0)


class TestFindMatches:
    def test_case_insensitive_literal_absolute_offsets(self):
        matches, total = find_matches(BODY, "CHOSE a")
        assert total == 1
        assert matches[0]["offset"] == BODY.index("chose A")
        assert BODY[matches[0]["offset"] :].startswith("chose A")

    def test_literal_not_regex(self):
        matches, total = find_matches("a.c abc a.c", ".")
        assert total == 2
        assert [m["offset"] for m in matches] == [1, 9]

    def test_non_overlapping(self):
        _, total = find_matches("aaaa", "aa")
        assert total == 2

    def test_heading_attribution(self):
        matches, _ = find_matches(BODY, "deep dive")
        assert matches[0]["heading"] == "Detail"
        matches, _ = find_matches(BODY, "mitigate")
        assert matches[0]["heading"] == "Risk Mitigation"

    def test_heading_none_before_first_heading(self):
        matches, _ = find_matches("preamble text\n# H\nbody\n", "preamble")
        assert matches[0]["heading"] is None

    def test_snippet_bounds_start_and_end(self):
        body = "needle in the middle of text needle"
        matches, _ = find_matches(body, "needle", context_chars=5)
        assert matches[0]["snippet"] == "needle in t"
        assert matches[1]["snippet"] == "text needle"

    def test_max_matches_truncation_and_total(self):
        body = "x " * 30
        matches, total = find_matches(body, "x", max_matches=5)
        assert len(matches) == 5
        assert total == 30

    def test_no_match(self):
        assert find_matches(BODY, "zzz") == ([], 0)


class TestEngine:
    async def test_default_call_unchanged(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body=BODY, summary_markdown=BODY)])
        rows = await engine.session_query_notebooks()
        assert set(rows[0]) <= {"session_id", "title", "tags", "created_at", "project_name"}

    async def test_full_mode_without_new_params_unchanged(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body=BODY, summary_markdown="sum")])
        rows = await engine.session_query_notebooks(summary_only=False)
        assert rows[0]["authored_body"] == BODY
        assert rows[0]["summary_markdown"] == "sum"
        assert "_body_window" not in rows[0]
        assert "_section_error" not in rows[0]

    async def test_outline_omits_bodies(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body=BODY, summary_markdown="x")])
        rows = await engine.session_query_notebooks(outline=True)
        assert "authored_body" not in rows[0]
        assert "summary_markdown" not in rows[0]
        assert len(rows[0]["outline"]) == 5
        assert rows[0]["outline"][1]["heading"] == "Decisions"

    async def test_section_applies_to_both_fields_independently(self, engine, db, monkeypatch):
        _stub_rows(
            db,
            monkeypatch,
            [_row(authored_body=BODY, summary_markdown="## Decisions\nonly summary\n")],
        )
        rows = await engine.session_query_notebooks(section="decisions")
        assert rows[0]["authored_body"].startswith("## Decisions\nchose A")
        assert rows[0]["summary_markdown"] == "## Decisions\nonly summary\n"
        assert rows[0]["_body_window"]["summary_markdown"]["total_chars"] == len(
            "## Decisions\nonly summary\n"
        )

    async def test_section_error_on_row(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body=BODY, summary_markdown=None)])
        rows = await engine.session_query_notebooks(section="nope")
        assert rows[0]["authored_body"] == ""
        assert "Decisions" in rows[0]["_section_error"]
        assert "summary_markdown" not in rows[0]["_body_window"]

    async def test_window_metadata_and_newlines_preserved(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body=BODY)])
        rows = await engine.session_query_notebooks(offset=2, max_chars=10)
        assert rows[0]["authored_body"] == BODY[2:12]
        assert "\n" in rows[0]["authored_body"]
        win = rows[0]["_body_window"]["authored_body"]
        assert win["total_chars"] == len(BODY)
        assert win["next_offset"] == 12
        assert win["elided_chars"] == len(BODY) - 12

    async def test_offset_only_implies_full_mode(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body="abcdef")])
        rows = await engine.session_query_notebooks(offset=3)
        assert rows[0]["authored_body"] == "def"

    async def test_invalid_params_raise(self, engine):
        with pytest.raises(ValueError):
            await engine.session_query_notebooks(offset=-1)
        with pytest.raises(ValueError):
            await engine.session_query_notebooks(max_chars=0)


class TestSearchEngine:
    async def test_search_drops_bodies_and_reports_matches(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body=BODY, summary_markdown="no hit")])
        rows = await engine.session_query_notebooks(search="chose")
        row = rows[0]
        assert "authored_body" not in row
        assert "summary_markdown" not in row
        assert "_body_window" not in row
        found = row["search_matches"]["authored_body"]
        assert found["total"] == 1
        assert found["returned"] == 1
        assert found["matches"][0]["offset"] == BODY.index("chose")
        assert found["matches"][0]["heading"] == "Decisions"
        assert row["search_matches"]["summary_markdown"]["total"] == 0

    async def test_empty_bodies_skipped(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body=BODY, summary_markdown=None)])
        rows = await engine.session_query_notebooks(search="chose")
        assert set(rows[0]["search_matches"]) == {"authored_body"}

    async def test_search_and_outline_combined(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body=BODY)])
        rows = await engine.session_query_notebooks(search="risk", outline=True)
        assert len(rows[0]["outline"]) == 5
        assert rows[0]["search_matches"]["authored_body"]["total"] >= 1
        assert "authored_body" not in rows[0]

    async def test_search_ignores_section_and_max_chars(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body=BODY)])
        rows = await engine.session_query_notebooks(
            search="mitigate", section="Decisions", offset=3, max_chars=5
        )
        assert "_body_window" not in rows[0]
        assert "_section_error" not in rows[0]
        found = rows[0]["search_matches"]["authored_body"]
        assert found["total"] == 1
        assert found["matches"][0]["offset"] == BODY.index("mitigate")

    @pytest.mark.parametrize("bad", ["", "   "])
    async def test_empty_search_raises(self, engine, bad):
        with pytest.raises(ValueError):
            await engine.session_query_notebooks(search=bad)


class TestKeyChanges:
    KC = ["/a/b/c.py", "/a/b/d.py"]

    async def test_body_mode_replaces_with_count(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body="abc", key_changes=self.KC)])
        rows = await engine.session_query_notebooks(max_chars=2)
        assert "key_changes" not in rows[0]
        assert rows[0]["key_changes_count"] == 2

    async def test_count_zero_when_missing_or_none(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body="abc"), _row(key_changes=None)])
        rows = await engine.session_query_notebooks(outline=True)
        assert [r["key_changes_count"] for r in rows] == [0, 0]
        assert all("key_changes" not in r for r in rows)

    async def test_search_mode_replaces_with_count(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body="abc", key_changes=self.KC)])
        rows = await engine.session_query_notebooks(search="a")
        assert rows[0]["key_changes_count"] == 2

    async def test_include_key_changes_keeps_list(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body="abc", key_changes=self.KC)])
        rows = await engine.session_query_notebooks(max_chars=2, include_key_changes=True)
        assert rows[0]["key_changes"] == self.KC
        assert "key_changes_count" not in rows[0]

    async def test_include_alone_does_not_trigger_body_mode(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body="abc", key_changes=self.KC)])
        rows = await engine.session_query_notebooks(include_key_changes=True)
        assert set(rows[0]) <= {"session_id", "title", "tags", "created_at", "project_name"}

    async def test_summary_only_false_untouched(self, engine, db, monkeypatch):
        _stub_rows(db, monkeypatch, [_row(authored_body="abc", key_changes=self.KC)])
        rows = await engine.session_query_notebooks(summary_only=False)
        assert rows[0]["key_changes"] == self.KC
        assert "key_changes_count" not in rows[0]


class TestSchema:
    def test_schema_accepts_new_params(self, engine):
        interface = LeanMCPInterface(engine)
        props = interface.tool_registry["session_query_notebooks"]["schema"]["properties"]
        for name in ("outline", "section", "offset", "max_chars", "search", "include_key_changes"):
            assert name in props
        params = {"outline": True, "section": "x", "offset": 1, "max_chars": 5}
        assert interface.validate_tool_parameters("session_query_notebooks", params) is None
        params = {"search": "needle", "include_key_changes": True}
        assert interface.validate_tool_parameters("session_query_notebooks", params) is None
