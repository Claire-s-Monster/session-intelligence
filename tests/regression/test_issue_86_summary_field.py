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
class TestSummaryInOutputs:
    async def test_search_decisions_returns_summary(self, engine):
        await engine.session_log_decision(
            "adopt zebrafish indexing", project_name="p86", summary="Zebra headline"
        )
        res = await engine.session_search("zebrafish", search_type="decisions")
        assert [r.summary for r in res.results] == ["Zebra headline"]

    async def test_search_learnings_returns_summary(self, engine):
        await engine.session_log_learning(
            "pattern", "narwhal caching trick", project_name="p86", summary="Narwhal headline"
        )
        res = await engine.session_search("narwhal", search_type="learnings")
        assert [r.summary for r in res.results] == ["Narwhal headline"]

    async def test_search_without_summary_is_none(self, engine):
        await engine.session_log_decision("adopt quokka indexing", project_name="p86")
        res = await engine.session_search("quokka", search_type="decisions")
        assert res.results[0].summary is None

    async def test_query_project_learnings_and_get_learning(self, engine, db):
        res = await engine.session_log_learning(
            "pattern", "ocelot body text", project_name="p86", summary="Ocelot"
        )
        rows = await db.query_project_learnings(res.learning.project_path)
        assert rows[0]["summary"] == "Ocelot"
        got = await engine.session_get_learning(res.id)
        assert got["learning"]["summary"] == "Ocelot"

    async def test_query_decisions_by_session_returns_summary(self, engine, db):
        res = await engine.session_log_decision("body", project_name="p86", summary="Sess headline")
        cursor = await db._ensure_connected().execute(
            "SELECT session_id FROM decisions WHERE id = ?", (res.decision_id,)
        )
        session_id = (await cursor.fetchone())["session_id"]
        rows = await db.query_decisions_by_session(session_id)
        assert rows[0]["summary"] == "Sess headline"

    def test_postgres_search_selects_summary(self):
        import inspect

        from persistence.postgresql import PostgreSQLBackend

        src = inspect.getsource(PostgreSQLBackend.search_sessions)
        assert "d.summary" in src
        assert src.count("summary") >= 2


@pytest.mark.regression
class TestUpdateSummary:
    async def _decision(self, engine, summary="Original"):
        res = await engine.session_log_decision("body", project_name="p86", summary=summary)
        return res.decision_id

    async def _learning(self, engine, summary="Original"):
        res = await engine.session_log_learning(
            "pattern", "learning body text", project_name="p86", summary=summary
        )
        return res.id

    async def _recalled(self, db, kind):
        return (await db.recall_project("p86", include=[kind]))[kind][0]["summary"]

    async def test_update_decision_sets_summary(self, engine, db):
        did = await self._decision(engine)
        res = await engine.session_update_decision(did, summary="  New headline ")
        assert res["status"] == "success"
        assert res["updated_fields"] == ["summary"]
        assert await self._recalled(db, "decisions") == "New headline"

    async def test_update_decision_omitted_leaves_summary(self, engine, db):
        did = await self._decision(engine)
        await engine.session_update_decision(did, rationale="why")
        assert await self._recalled(db, "decisions") == "Original"

    async def test_update_decision_empty_clears_summary(self, engine, db):
        did = await self._decision(engine)
        res = await engine.session_update_decision(did, summary="")
        assert res["status"] == "success"
        assert await self._recalled(db, "decisions") is None

    async def test_update_decision_too_long_rejected_before_write(self, engine, db):
        did = await self._decision(engine)
        res = await engine.session_update_decision(did, rationale="why", summary=LONG)
        assert res["status"] == "error"
        assert "summary" in res["message"]
        assert await self._recalled(db, "decisions") == "Original"

    async def test_update_decision_syncs_cache(self, engine, db):
        did = await self._decision(engine)
        await engine.session_update_decision(did, summary="Edited")
        cached = [d for s in engine.session_cache.values() for d in s.decisions]
        assert [d.summary for d in cached if d.decision_id == did] == ["Edited"]

    async def test_update_learning_sets_summary(self, engine, db):
        lid = await self._learning(engine)
        res = await engine.session_update_learning(lid, summary="New learn headline")
        assert res["status"] == "success"
        assert await self._recalled(db, "learnings") == "New learn headline"

    async def test_update_learning_omitted_leaves_summary(self, engine, db):
        lid = await self._learning(engine)
        await engine.session_update_learning(lid, trigger_context="ctx")
        assert await self._recalled(db, "learnings") == "Original"

    async def test_update_learning_empty_clears_summary(self, engine, db):
        lid = await self._learning(engine)
        res = await engine.session_update_learning(lid, summary="   ")
        assert res["status"] == "success"
        assert await self._recalled(db, "learnings") is None

    async def test_update_learning_too_long_rejected_before_write(self, engine, db):
        lid = await self._learning(engine)
        res = await engine.session_update_learning(lid, trigger_context="ctx", summary=LONG)
        assert res["status"] == "error"
        assert await self._recalled(db, "learnings") == "Original"

    def test_update_schemas_declare_summary(self, engine):
        from lean_mcp_interface import LeanMCPInterface

        registry = LeanMCPInterface(engine).tool_registry
        for name in ("session_update_decision", "session_update_learning"):
            prop = registry[name]["schema"]["properties"]["summary"]
            assert prop["maxLength"] == SUMMARY_MAX_CHARS


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
