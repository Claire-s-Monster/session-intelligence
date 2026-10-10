"""
Regression tests for issue #207: a read tool `session_get_learning(learning_id)`
returning one project learning (knowledge-bridge's promote_learning needs it),
plus a `learning_id` key on session_search rows of search_type="learnings"
(they previously exposed the learning id only under the misleading `session_id`).

PostgreSQL is exercised through a fake asyncpg pool (no live DB; never the
production database), as in test_issue_179_agent_stats_duration.py.
"""

from datetime import UTC, datetime

import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.postgresql import PostgreSQLBackend
from persistence.sqlite import SQLiteBackend

LEARNING_KEYS = {
    "id",
    "category",
    "trigger_context",
    "learning_content",
    "project_name",
    "project_path",
    "source_session_id",
    "success_count",
    "failure_count",
    "created_at",
    "last_used",
    "promoted_to_universal",
    "supersedes",
    "superseded_by",
    "retired_at",
    "retired_reason",
}


@pytest.fixture
async def engine(tmp_path):
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


@pytest.fixture
async def lean_interface(engine):
    return LeanMCPInterface(engine)


def _get_meta_tool(interface: LeanMCPInterface, name: str):
    return interface.app._tool_manager._tools[name].fn


@pytest.mark.regression
class TestSessionGetLearning:
    async def test_get_returns_full_record(self, engine):
        logged = await engine.session_log_learning(
            category="pattern",
            learning_content="Use bound parameters",
            trigger_context="writing SQL",
            project_name="proj-a",
        )

        result = await engine.session_get_learning(logged.id)

        assert result["status"] == "success"
        learning = result["learning"]
        assert set(learning) == LEARNING_KEYS
        assert learning["id"] == logged.id
        assert learning["category"] == "pattern"
        assert learning["learning_content"] == "Use bound parameters"
        assert learning["project_name"] == "proj-a"
        assert isinstance(learning["created_at"], str)
        assert learning["superseded_by"] == []
        assert learning["supersedes"] is None

    async def test_supersession_links_both_directions(self, engine):
        first = await engine.session_log_learning(
            category="error_fix", learning_content="Old fix", project_name="proj-a"
        )
        second = await engine.session_log_learning(
            category="error_fix",
            learning_content="New fix",
            project_name="proj-a",
            supersedes=first.id,
        )

        old = (await engine.session_get_learning(first.id))["learning"]
        new = (await engine.session_get_learning(second.id))["learning"]

        assert old["superseded_by"] == [second.id]
        assert new["supersedes"] == first.id
        assert new["superseded_by"] == []

    async def test_unknown_id_is_error(self, engine):
        result = await engine.session_get_learning("learn_doesnotexist")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    @pytest.mark.parametrize("blank", ["", "   "])
    async def test_blank_id_is_error(self, engine, blank):
        result = await engine.session_get_learning(blank)
        assert result["status"] == "error"

    async def test_lean_execute_tool_success_and_error(self, engine, lean_interface):
        logged = await engine.session_log_learning(
            category="pattern", learning_content="Via meta-tool", project_name="proj-a"
        )
        execute = _get_meta_tool(lean_interface, "execute_tool")

        ok = await execute("session_get_learning", {"learning_id": logged.id})
        assert ok["status"] == "success"
        assert ok["result"]["learning"]["id"] == logged.id

        missing = await execute("session_get_learning", {"learning_id": "learn_nope"})
        assert missing["status"] == "error"

    async def test_search_learnings_rows_carry_learning_id(self, engine):
        logged = await engine.session_log_learning(
            category="pattern", learning_content="zebrafish quirk", project_name="proj-a"
        )

        found = await engine.session_search("zebrafish", search_type="learnings")

        assert found.total_results == 1
        row = found.results[0]
        assert row.learning_id == logged.id
        assert row.session_id == logged.id


class _FakeConn:
    def __init__(self, row, successors):
        self._row = row
        self._successors = successors

    async def fetchrow(self, query, *args):
        return self._row if args == (self._row["id"],) else None

    async def fetch(self, query, *args):
        return self._successors


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
async def test_postgres_datetimes_come_back_as_iso_strings(tmp_path):
    stamp = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
    row = {
        "id": "learn_pgpgpgpgpgpg",
        "project_path": "/p",
        "project_name": None,
        "category": "pattern",
        "trigger_context": None,
        "learning_content": "pg content",
        "source_session_id": None,
        "success_count": 0,
        "failure_count": 0,
        "last_used": stamp,
        "promoted_to_universal": False,
        "created_at": stamp,
        "supersedes": None,
    }
    backend = PostgreSQLBackend(dsn="postgresql://localhost/fake_issue_207")
    backend._pool = _FakePool(_FakeConn(row, [{"id": "learn_successor0001"}]))  # type: ignore[assignment]
    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=backend
    )

    result = await eng.session_get_learning("learn_pgpgpgpgpgpg")

    learning = result["learning"]
    assert set(learning) == LEARNING_KEYS
    assert learning["created_at"] == stamp.isoformat()
    assert learning["last_used"] == stamp.isoformat()
    assert learning["project_name"] is None
    assert learning["superseded_by"] == ["learn_successor0001"]
