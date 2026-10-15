"""
Regression tests: warn when a project_name-only call guesses among several
active sessions (two concurrent Claude Code sessions in one project used to
bind a learning to the newer session silently).

SQLite is exercised for real; no PostgreSQL needed.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.base import get_session_max_age_hours
from persistence.sqlite import SQLiteBackend
from tests.persistence.conftest import POSTGRES_AVAILABLE
from tests.persistence.contract_tests import _session
from tests.persistence.test_postgresql_migration import pg_backend  # noqa: F401

PROJECT = "proj-ambiguous"


@pytest.fixture
async def engine(tmp_path):
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


async def _new_session(engine, name, path=None):
    created = await engine.session_manage_lifecycle(
        "create", project_name=PROJECT, session_name=name, project_path=path
    )
    return created.session_id


def _kinds(warnings):
    return [w["kind"] for w in warnings]


@pytest.mark.regression
class TestAmbiguousSessionBinding:
    async def test_count_matches_active_sessions(self, engine):
        assert await engine.database.count_recent_sessions_by_project(PROJECT) == 0
        await _new_session(engine, "one")
        await _new_session(engine, "two")
        assert await engine.database.count_recent_sessions_by_project(PROJECT) == 2

    async def test_decision_warns_when_several_active(self, engine):
        await _new_session(engine, "one")
        await _new_session(engine, "two")

        result = await engine.session_log_decision("pick one", project_name=PROJECT)

        ambiguous = [w for w in result.warnings if w["kind"] == "ambiguous_session_binding"]
        assert len(ambiguous) == 1
        assert f"bound to most recent active session {result.session_id}" in ambiguous[0]["message"]
        assert "2 active sessions matched" in ambiguous[0]["message"]
        assert "Pass session_id to bind explicitly." in ambiguous[0]["message"]

    async def test_learning_warns_when_several_active(self, engine):
        await _new_session(engine, "one")
        await _new_session(engine, "two")

        result = await engine.session_log_learning(
            category="pattern",
            learning_content="guessed binding",
            trigger_context="two sessions",
            project_name=PROJECT,
        )

        assert "ambiguous_session_binding" in _kinds(result.warnings)

    async def test_single_candidate_has_no_warning(self, engine):
        await _new_session(engine, "only")

        decision = await engine.session_log_decision("solo", project_name=PROJECT)
        learning = await engine.session_log_learning(
            category="pattern",
            learning_content="solo learning",
            trigger_context="one session",
            project_name=PROJECT,
        )

        assert "ambiguous_session_binding" not in _kinds(decision.warnings)
        assert "ambiguous_session_binding" not in _kinds(learning.warnings)

    async def test_explicit_session_id_never_warns(self, engine):
        sid = await _new_session(engine, "one")
        await _new_session(engine, "two")

        result = await engine.session_log_decision("explicit", session_id=sid)

        assert "ambiguous_session_binding" not in _kinds(result.warnings)

    async def test_project_path_scope_narrows_candidates(self, engine, tmp_path):
        path_a, path_b = tmp_path / "a", tmp_path / "b"
        path_a.mkdir()
        path_b.mkdir()
        await _new_session(engine, "a", str(path_a))
        await _new_session(engine, "b", str(path_b))

        result = await engine.session_log_decision(
            "scoped", project_name=PROJECT, project_path=str(path_a)
        )

        assert "ambiguous_session_binding" not in _kinds(result.warnings)


async def _seed_pg(
    db,
    project,
    *,
    path="/tmp/pg-a",
    status="active",
    started_offset_min=0,
    last_seen_age_hours=None,
):
    now = datetime.now(UTC)
    started = now - timedelta(minutes=started_offset_min)
    data = _session(
        session_id=f"s-{uuid.uuid4().hex[:12]}",
        project_name=project,
        project_path=path,
        status=status,
    )
    data["started_at"] = started.isoformat()
    if last_seen_age_hours is not None:
        data["last_seen_at"] = (now - timedelta(hours=last_seen_age_hours)).isoformat()
    await db.save_session(data)
    return data["id"]


@pytest.mark.regression
@pytest.mark.postgresql
@pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not available")
class TestPostgresCountRecentSessions:
    async def test_counts_active_sessions(self, pg_backend):  # noqa: F811
        project = f"proj-{uuid.uuid4().hex[:8]}"
        await _seed_pg(pg_backend, project, started_offset_min=2)
        await _seed_pg(pg_backend, project, started_offset_min=1)
        assert await pg_backend.count_recent_sessions_by_project(project) == 2

    async def test_completed_session_not_counted(self, pg_backend):  # noqa: F811
        project = f"proj-{uuid.uuid4().hex[:8]}"
        await _seed_pg(pg_backend, project, started_offset_min=2)
        await _seed_pg(pg_backend, project, started_offset_min=1)
        await _seed_pg(pg_backend, project, status="completed")
        assert await pg_backend.count_recent_sessions_by_project(project) == 2
        assert await pg_backend.count_recent_sessions_by_project(project, status="completed") == 1

    async def test_stale_session_not_counted(self, pg_backend):  # noqa: F811
        project = f"proj-{uuid.uuid4().hex[:8]}"
        await _seed_pg(pg_backend, project)
        stale_age = get_session_max_age_hours() + 1
        await _seed_pg(pg_backend, project, last_seen_age_hours=stale_age)
        assert await pg_backend.count_recent_sessions_by_project(project) == 1

    async def test_project_path_filter_narrows_count(self, pg_backend):  # noqa: F811
        project = f"proj-{uuid.uuid4().hex[:8]}"
        await _seed_pg(pg_backend, project, path="/tmp/pg-a")
        await _seed_pg(pg_backend, project, path="/tmp/pg-a")
        await _seed_pg(pg_backend, project, path="/tmp/pg-b")
        assert await pg_backend.count_recent_sessions_by_project(project) == 3
        assert (
            await pg_backend.count_recent_sessions_by_project(project, project_path="/tmp/pg-a")
            == 2
        )
        assert (
            await pg_backend.count_recent_sessions_by_project(project, project_path="/tmp/pg-b")
            == 1
        )

    async def test_parity_with_find_recent_session(self, pg_backend):  # noqa: F811
        project = f"proj-{uuid.uuid4().hex[:8]}"
        older = await _seed_pg(pg_backend, project, started_offset_min=10)
        newest = await _seed_pg(pg_backend, project, started_offset_min=1)
        await _seed_pg(pg_backend, project, status="completed", started_offset_min=0)
        await _seed_pg(
            pg_backend,
            project,
            started_offset_min=0,
            last_seen_age_hours=get_session_max_age_hours() + 1,
        )

        counted = await pg_backend.count_recent_sessions_by_project(project)
        found = await pg_backend.find_recent_session_by_project(project)

        assert counted == 2
        assert found["id"] in {older, newest}
        assert found["id"] == newest
