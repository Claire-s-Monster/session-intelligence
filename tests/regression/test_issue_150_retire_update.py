"""
Regression tests for issue #150: retire / un-retire / edit-in-place for learnings
and decisions, plus the supersede/retire filter on session_find_solution and
session_search (which previously leaked superseded rows).

SQLite is exercised for real; PostgreSQL through a fake asyncpg pool (no live DB).
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.postgresql import PostgreSQLBackend
from persistence.sqlite import SQLiteBackend

PROJECT = "proj-150"


@pytest.fixture
async def engine(tmp_path):
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


def _project_path(tmp_path) -> str:
    return str(tmp_path / "proj")


async def _log_learning(engine, tmp_path, content, **kw):
    return await engine.session_log_learning(
        category="error_fix",
        learning_content=content,
        trigger_context="quokkaerror",
        project_name=PROJECT,
        project_path=_project_path(tmp_path),
        **kw,
    )


async def _recall_ids(engine, key):
    recalled = await engine.session_recall(project_name=PROJECT)
    return str(recalled[key])


@pytest.mark.regression
class TestRetireLearning:
    async def test_retire_hides_everywhere_and_unretire_restores(self, engine, tmp_path):
        logged = await _log_learning(engine, tmp_path, "quokkafix alpha")
        path = _project_path(tmp_path)

        result = await engine.session_retire_learning(logged.id, reason="stale")
        assert result["status"] == "success"

        assert logged.id not in await _recall_ids(engine, "learnings")
        found = await engine.session_find_solution("quokkaerror", project_path=path)
        assert found.total_found == 0
        searched = await engine.session_search("quokkafix", search_type="learnings")
        assert searched.total_results == 0

        got = (await engine.session_get_learning(logged.id))["learning"]
        assert got["retired_at"] is not None
        assert got["retired_reason"] == "stale"

        restored = await engine.session_retire_learning(logged.id, unretire=True)
        assert restored["status"] == "success"
        assert logged.id in await _recall_ids(engine, "learnings")
        found = await engine.session_find_solution("quokkaerror", project_path=path)
        assert found.total_found == 1
        searched = await engine.session_search("quokkafix", search_type="learnings")
        assert searched.total_results == 1
        got = (await engine.session_get_learning(logged.id))["learning"]
        assert got["retired_at"] is None
        assert got["retired_reason"] is None

    async def test_unknown_id_is_error(self, engine):
        result = await engine.session_retire_learning("learn_nope")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    async def test_blank_id_is_error(self, engine):
        assert (await engine.session_retire_learning("  "))["status"] == "error"


@pytest.mark.regression
class TestSupersededLeaksClosed:
    async def test_superseded_learning_hidden_from_find_solution_and_search(self, engine, tmp_path):
        old = await _log_learning(engine, tmp_path, "quokkafix old")
        new = await _log_learning(engine, tmp_path, "quokkafix new", supersedes=old.id)

        found = await engine.session_find_solution(
            "quokkaerror", project_path=_project_path(tmp_path)
        )
        assert found.total_found == 1

        searched = await engine.session_search("quokkafix", search_type="learnings")
        assert [r.learning_id for r in searched.results] == [new.id]

    async def test_superseded_decision_hidden_from_search(self, engine):
        old = await engine.session_log_decision("quokkadecision old", project_name=PROJECT)
        new = await engine.session_log_decision(
            "quokkadecision new", project_name=PROJECT, supersedes=old.decision_id
        )

        searched = await engine.session_search("quokkadecision", search_type="decisions")
        assert [r.session_id for r in searched.results] == [new.decision_id]


@pytest.mark.regression
class TestRetireDecision:
    async def test_retire_hides_from_recall_and_search_then_restores(self, engine):
        logged = await engine.session_log_decision("quokkadecision keep", project_name=PROJECT)
        did = logged.decision_id

        assert (await engine.session_retire_decision(did, reason="obsolete"))["status"] == (
            "success"
        )
        # recall rows carry description, not id, so match on the text.
        assert "quokkadecision keep" not in await _recall_ids(engine, "decisions")
        searched = await engine.session_search("quokkadecision", search_type="decisions")
        assert searched.total_results == 0

        assert (await engine.session_retire_decision(did, unretire=True))["status"] == "success"
        assert "quokkadecision keep" in await _recall_ids(engine, "decisions")
        searched = await engine.session_search("quokkadecision", search_type="decisions")
        assert searched.total_results == 1

    async def test_unknown_id_is_error(self, engine):
        result = await engine.session_retire_decision("dec_nope")
        assert result["status"] == "error"


@pytest.mark.regression
class TestUpdateLearning:
    async def test_update_changes_content_and_get_reflects_it(self, engine, tmp_path):
        logged = await _log_learning(engine, tmp_path, "quokkafix before")

        result = await engine.session_update_learning(
            logged.id, learning_content="quokkafix after", category="pattern"
        )

        assert result["status"] == "success"
        assert result["updated_fields"] == ["category", "learning_content"]
        got = (await engine.session_get_learning(logged.id))["learning"]
        assert got["learning_content"] == "quokkafix after"
        assert got["category"] == "pattern"
        assert got["trigger_context"] == "quokkaerror"

    async def test_unknown_id_is_error(self, engine):
        result = await engine.session_update_learning("learn_nope", learning_content="x")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    async def test_no_fields_is_error(self, engine, tmp_path):
        logged = await _log_learning(engine, tmp_path, "quokkafix")
        assert (await engine.session_update_learning(logged.id))["status"] == "error"

    async def test_invalid_category_is_error(self, engine, tmp_path):
        logged = await _log_learning(engine, tmp_path, "quokkafix")
        result = await engine.session_update_learning(logged.id, category="bogus")
        assert result["status"] == "error"

    async def test_disallowed_field_rejected_at_persistence_layer(self, engine, tmp_path):
        logged = await _log_learning(engine, tmp_path, "quokkafix")
        for field in ("supersedes", "retired_at", "id", "project_path", "created_at"):
            with pytest.raises(ValueError, match="not updatable"):
                await engine.database.update_project_learning(logged.id, **{field: "x"})
        for field in ("supersedes", "retired_reason", "timestamp", "session_id"):
            with pytest.raises(ValueError, match="not updatable"):
                await engine.database.update_decision("dec_x", **{field: "x"})


@pytest.mark.regression
class TestUpdateDecision:
    async def test_update_changes_text_and_search_sees_it(self, engine):
        logged = await engine.session_log_decision("quokkadecision before", project_name=PROJECT)

        result = await engine.session_update_decision(
            logged.decision_id, decision="quokkadecision after", impact_level="high"
        )

        assert result["status"] == "success"
        searched = await engine.session_search("quokkadecision", search_type="decisions")
        assert searched.total_results == 1
        assert "after" in searched.results[0].snippet

    async def test_unknown_id_and_empty_and_bad_impact(self, engine):
        missing = await engine.session_update_decision("dec_nope", decision="x")
        assert missing["status"] == "error"
        assert "not found" in missing["message"]
        assert (await engine.session_update_decision("dec_nope"))["status"] == "error"
        bad = await engine.session_update_decision("dec_nope", impact_level="enormous")
        assert bad["status"] == "error"


@pytest.mark.regression
class TestUpdateDecisionSurvivesPersistSweep:
    async def test_edit_of_cached_decision_not_reverted_by_sweep(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from transport.http_server import HTTPSessionIntelligenceServer
        from transport.persist_tracker import PersistDigestTracker

        monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path), use_filesystem=False)
        eng.database = SQLiteBackend(str(tmp_path / "sweep.db"))
        await eng.database.initialize()
        try:
            created = await eng.session_manage_lifecycle(
                operation="create", mode="local", project_name=PROJECT
            )
            sid = created.session_id
            logged = await eng.session_log_decision("quokka before", session_id=sid)
            server = HTTPSessionIntelligenceServer.__new__(HTTPSessionIntelligenceServer)
            server.persist_tracker = PersistDigestTracker()
            state = SimpleNamespace(database=eng.database, session_engine=eng)
            request = SimpleNamespace(app=SimpleNamespace(state=state))
            # No sweep before the edit: a prior sweep would commit a digest of the
            # stale cached Decision, letting every later sweep skip it and hiding
            # the revert this test guards against.

            result = await eng.session_update_decision(
                logged.decision_id, decision="quokka after", rationale="because"
            )
            assert result["status"] == "success"
            await server._persist_sessions_to_database(request)
            await server._persist_sessions_to_database(request)

            rows = await eng.database.query_decisions_by_session(sid)
            row = next(r for r in rows if r["id"] == logged.decision_id)
            assert row["description"] == "quokka after"
            assert row["rationale"] == "because"
        finally:
            await eng.database.close()


@pytest.mark.regression
async def test_tools_registered_and_dispatch(engine, tmp_path):
    interface = LeanMCPInterface(engine)
    for name in (
        "session_retire_learning",
        "session_retire_decision",
        "session_update_learning",
        "session_update_decision",
    ):
        assert name in interface.tool_registry
        assert interface.tool_registry[name]["schema"]["required"]

    logged = await _log_learning(engine, tmp_path, "quokkafix via tool")
    execute = interface.app._tool_manager._tools["execute_tool"].fn
    ok = await execute("session_retire_learning", {"learning_id": logged.id})
    assert ok["status"] == "success"
    assert ok["result"]["status"] == "success"
    missing = await execute("session_retire_learning", {"learning_id": "learn_nope"})
    assert missing["result"]["status"] == "error"


class _FakeConn:
    def __init__(self, status):
        self.status = status
        self.calls = []

    async def execute(self, query, *args):
        self.calls.append((query, args))
        return self.status


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


@pytest.mark.regression
class TestPostgresPersistence:
    def _backend(self, status):
        backend = PostgreSQLBackend(dsn="postgresql://localhost/fake_issue_150")
        conn = _FakeConn(status)
        backend._pool = _FakePool(conn)  # type: ignore[assignment]
        return backend, conn

    async def test_retire_and_unretire(self):
        backend, conn = self._backend("UPDATE 1")

        assert await backend.set_learning_retired("l1", True, "why") is True
        query, args = conn.calls[-1]
        assert "UPDATE project_learnings SET retired_at" in query
        assert args[0] is not None
        assert args[1:] == ("why", "l1")

        assert await backend.set_decision_retired("d1", False, "ignored") is True
        query, args = conn.calls[-1]
        assert "UPDATE decisions" in query
        assert args == (None, None, "d1")

    async def test_not_found_returns_false(self):
        backend, _ = self._backend("UPDATE 0")
        assert await backend.set_learning_retired("nope", True, None) is False
        assert await backend.update_project_learning("nope", learning_content="x") is False

    async def test_update_builds_numbered_params_and_rejects_disallowed(self):
        backend, conn = self._backend("UPDATE 1")

        assert await backend.update_decision("d1", description="x", rationale="y") is True
        query, args = conn.calls[-1]
        assert "description = $1, rationale = $2 WHERE id = $3" in query
        assert args == ("x", "y", "d1")

        with pytest.raises(ValueError, match="not updatable"):
            await backend.update_project_learning("l1", supersedes="z")
