"""Regression test for issue #103 -- notebook creation on a cold (uncached) session.

`session_create_notebook` used to fail with "No session found" for any
session that was not resident in the process's in-memory `session_cache`.
Because the cache starts empty on every process start, the defect was a
cache-membership gate on a read path: a session that was fully persisted in
the database, and correctly resolved to the right session_id, was still
treated as nonexistent solely because this process never happened to create
or touch it in memory. Every restart therefore made every prior session
permanently un-notebookable.

The fix replaced the `session_id in self.session_cache` membership test with
`await self._hydrate_session(session_id)`, which returns the cached Session
if present, otherwise loads the row from the database, reconstructs the
Session (and its decisions), populates the cache, and returns it -- only
returning None when the session genuinely does not exist in the database.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import Session, SessionMetadata
from persistence.sqlite import SQLiteBackend


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "test_issue_103.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def engine(tmp_path, db):
    return SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )


def _session_row(session_id: str, status: str = "active") -> dict:
    started = datetime.now(UTC).isoformat()
    return {
        "id": session_id,
        "started": started,
        "project_path": "/tmp/issue-103-project",
        "project_name": "issue-103-project",
        "mode": "local",
        "status": status,
        "metadata": {},
        "performance_metrics": {},
        "health_status": {},
    }


def _decision_row(decision_id: str, session_id: str, description: str) -> dict:
    return {
        "id": decision_id,
        "session_id": session_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "category": "implementation",
        "description": description,
        "rationale": "because it was needed",
        "context": {},
        "impact_level": "medium",
        "artifacts": [],
    }


async def test_notebook_succeeds_for_uncached_active_session(engine, db):
    sid = "issue-103-active"
    await db.save_session(_session_row(sid, status="active"))
    engine.session_cache.clear()
    assert sid not in engine.session_cache

    result = await engine.session_create_notebook(session_id=sid)

    assert result.status == "success"


async def test_notebook_succeeds_for_uncached_completed_session(engine, db):
    sid = "issue-103-completed"
    await db.save_session(_session_row(sid, status="completed"))
    engine.session_cache.clear()
    assert sid not in engine.session_cache

    result = await engine.session_create_notebook(session_id=sid)

    assert result.status == "success"


async def test_hydration_populates_the_cache(engine, db):
    sid = "issue-103-populates-cache"
    await db.save_session(_session_row(sid))
    engine.session_cache.clear()

    await engine.session_create_notebook(session_id=sid)

    assert sid in engine.session_cache


async def test_missing_session_still_reports_error(engine, db):
    sid = "issue-103-never-persisted"

    result = await engine.session_create_notebook(session_id=sid)

    assert result.status == "error"
    assert "no session found" in result.message.lower()


async def test_cold_loaded_decisions_reach_the_notebook(engine, db):
    sid = "issue-103-decisions"
    await db.save_session(_session_row(sid))
    await db.save_decision(_decision_row("dec-1", sid, "First cold decision"))
    await db.save_decision(_decision_row("dec-2", sid, "Second cold decision"))
    engine.session_cache.clear()

    result = await engine.session_create_notebook(session_id=sid)

    assert result.status == "success"
    assert "First cold decision" in result.markdown_output
    assert "Second cold decision" in result.markdown_output
    cached = engine.session_cache[sid]
    decision_ids = {d.decision_id for d in cached.decisions}
    assert decision_ids == {"dec-1", "dec-2"}


def test_artifacts_stored_as_json_string_are_decoded(engine):
    sid = "issue-103-artifacts"
    row = {
        "id": "dec-artifacts",
        "description": "decision with stringified artifacts",
        "timestamp": datetime.now(UTC).isoformat(),
        "impact_level": "medium",
        "artifacts": '["a.py", "b.py"]',
        "context": "{}",
    }

    decision = engine._decision_from_row(row, sid)

    assert decision.artifacts == ["a.py", "b.py"]

    malformed_row = {**row, "artifacts": "not json"}
    malformed_decision = engine._decision_from_row(malformed_row, sid)

    assert malformed_decision.artifacts == []


async def test_async_variant_also_cold_loads(engine, db):
    sid = "issue-103-async-variant"
    await db.save_session(_session_row(sid))
    engine.session_cache.clear()
    assert sid not in engine.session_cache

    result = await engine.session_create_notebook_async(session_id=sid)

    assert result.status == "success"


async def test_cached_session_is_not_refetched(engine, db):
    sid = "issue-103-cached-short-circuit"
    # Deliberately NOT persisted to the database: if _hydrate_session ever
    # fell through to a database lookup instead of honoring the cache hit,
    # it would find nothing and return None instead of the cached object.
    cached_session = Session(
        id=sid,
        started=datetime.now(UTC),
        project_name="cached-project",
        project_path="/tmp/cached-project",
        metadata=SessionMetadata(
            session_type="development", environment="local", user="user"
        ),
    )
    engine.session_cache[sid] = cached_session

    result = await engine._hydrate_session(sid)

    assert result is cached_session
