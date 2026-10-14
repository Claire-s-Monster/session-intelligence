"""Regression tests for issue #86: optional `summary` on decisions and learnings.

summary is optional, max SUMMARY_MAX_CHARS (100, = ENTRY_CLIP in the SessionStart
hook). Longer values are rejected before any write; blank means None.

https://github.com/Claire-s-Monster/session-intelligence/issues/86
"""

from __future__ import annotations

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import SUMMARY_MAX_CHARS, normalize_summary
from persistence.sqlite import SQLiteBackend

LONG = "x" * (SUMMARY_MAX_CHARS + 1)
EXACT = "y" * SUMMARY_MAX_CHARS


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "issue86.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def engine(tmp_path, db, monkeypatch):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
    return SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )


@pytest.mark.regression
class TestNormalizeSummary:
    def test_limit_is_100(self):
        assert SUMMARY_MAX_CHARS == 100

    def test_none_and_blank_become_none(self):
        assert normalize_summary(None) is None
        assert normalize_summary("   \n") is None

    def test_strips_and_accepts_exact_limit(self):
        assert normalize_summary(f"  {EXACT}  ") == EXACT

    def test_over_limit_rejected_not_truncated(self):
        with pytest.raises(ValueError, match="100"):
            normalize_summary(LONG)


@pytest.mark.regression
class TestDecisionSummary:
    async def test_recall_returns_summary(self, engine, db):
        await engine.session_log_decision(
            "body of decision", project_name="p86", summary="Headline"
        )
        recalled = await db.recall_project("p86", include=["decisions"])
        assert recalled["decisions"][0]["summary"] == "Headline"

    async def test_no_summary_is_none(self, engine, db):
        await engine.session_log_decision("body of decision", project_name="p86")
        recalled = await db.recall_project("p86", include=["decisions"])
        assert recalled["decisions"][0]["summary"] is None

    async def test_too_long_rejected_before_write(self, engine, db):
        with pytest.raises(ValueError, match="summary"):
            await engine.session_log_decision("body", project_name="p86", summary=LONG)
        recalled = await db.recall_project("p86", include=["decisions"])
        assert recalled["decisions"] == []

    async def test_supersedes_and_near_dup_still_work(self, engine):
        first = await engine.session_log_decision(
            "switch the session cache invalidation to a write-through strategy for hydrate",
            project_name="p86",
            summary="first",
        )
        res = await engine.session_log_decision(
            "Switch session cache invalidation to write-through strategy for hydrate paths",
            project_name="p86",
            supersedes=first.decision_id,
            summary="second",
        )
        assert res.decision_id != "error"
        assert res.warnings == []


@pytest.mark.regression
class TestLearningSummary:
    async def test_recall_returns_summary(self, engine, db):
        res = await engine.session_log_learning(
            "pattern", "learning body text", project_name="p86", summary="Learn headline"
        )
        assert res.learning.summary == "Learn headline"
        recalled = await db.recall_project("p86", include=["learnings"])
        assert recalled["learnings"][0]["summary"] == "Learn headline"

    async def test_blank_summary_is_none(self, engine, db):
        await engine.session_log_learning(
            "pattern", "learning body text", project_name="p86", summary="  "
        )
        recalled = await db.recall_project("p86", include=["learnings"])
        assert recalled["learnings"][0]["summary"] is None

    async def test_too_long_rejected_before_write(self, engine, db):
        with pytest.raises(ValueError, match="summary"):
            await engine.session_log_learning(
                "pattern", "learning body text", project_name="p86", summary=LONG
            )
        recalled = await db.recall_project("p86", include=["learnings"])
        assert recalled["learnings"] == []


@pytest.mark.regression
class TestSchemas:
    def test_tool_schemas_declare_summary(self, engine):
        from lean_mcp_interface import LeanMCPInterface

        registry = LeanMCPInterface(engine).tool_registry
        for name in ("session_log_decision", "session_log_learning"):
            prop = registry[name]["schema"]["properties"]["summary"]
            assert prop["maxLength"] == SUMMARY_MAX_CHARS
            assert "summary" in registry[name]["description"]

    def test_store_division_documented(self, engine):
        from lean_mcp_interface import LeanMCPInterface

        registry = LeanMCPInterface(engine).tool_registry
        for name in ("session_log_decision", "session_log_learning", "session_create_notebook"):
            assert "MEMORY.md" in registry[name]["description"]
