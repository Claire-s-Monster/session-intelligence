"""
Regression tests for issue #218: reactivate completed/abandoned sessions on adopt.

`session_manage_lifecycle(operation="create", session_id=X)` adopts an existing
session (issue #115) but used to leave its status 'completed'/'abandoned' and
the old `completed` timestamp, so the session kept collecting rows while
status='active' filters (find_recent_session_by_project, the reaper) missed it.

https://github.com/Claire-s-Monster/session-intelligence/issues/218
"""

from datetime import UTC, datetime

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import SessionStatus
from persistence.sqlite import SQLiteBackend

SESSION_ID = "0b1f6c9e-2d4a-4c1e-9a77-5f3e8d2b1a64"
PROJECT = "demo-project"


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    """Engine wired to a fresh SQLite backend, filesystem persistence off."""
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issue218.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


@pytest.fixture
def memory_engine(tmp_path):
    """Engine with NO database (memory-only mode)."""
    return SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=None,
    )


async def _create(eng):
    return await eng.session_manage_lifecycle(
        operation="create", project_name=PROJECT, session_id=SESSION_ID
    )


@pytest.mark.regression
async def test_recreate_after_finalize_reactivates_and_persists(engine):
    assert (await _create(engine)).status == "success"
    fin = await engine.session_manage_lifecycle(operation="finalize", session_id=SESSION_ID)
    assert fin.status == "success"

    result = await _create(engine)

    assert result.status == "success"
    assert "reactivated" in result.message
    assert result.session_data["status"] == SessionStatus.ACTIVE.value
    assert result.session_data["completed_at"] is None
    assert result.session_data["last_seen_at"] is not None

    engine.session_cache.pop(SESSION_ID)
    row = await engine.database.get_session(SESSION_ID)
    assert row is not None
    assert row["status"] == "active"
    assert not row.get("ended_at")
    assert not row.get("completed")


@pytest.mark.regression
async def test_db_only_abandoned_row_is_reactivated(engine):
    await engine.database.save_session(
        {
            "id": SESSION_ID,
            "started": datetime.now(UTC).isoformat(),
            "project_name": PROJECT,
            "project_path": "",
            "mode": "local",
            "status": "abandoned",
            "metadata": {},
            "performance_metrics": {},
            "health_status": {},
        }
    )
    assert SESSION_ID not in engine.session_cache

    result = await _create(engine)

    assert result.status == "success"
    assert "reactivated" in result.message
    assert result.session_data["status"] == SessionStatus.ACTIVE.value


@pytest.mark.regression
async def test_adopting_active_session_is_not_reactivated(engine):
    await _create(engine)

    result = await _create(engine)

    assert result.status == "success"
    assert "reactivated" not in result.message
    assert result.session_data["status"] == SessionStatus.ACTIVE.value


@pytest.mark.regression
async def test_failed_session_is_not_reactivated(memory_engine):
    await _create(memory_engine)
    memory_engine.session_cache[SESSION_ID].status = SessionStatus.FAILED

    result = await _create(memory_engine)

    assert result.status == "success"
    assert "reactivated" not in result.message
    assert result.session_data["status"] == SessionStatus.FAILED.value


@pytest.mark.regression
async def test_memory_only_completed_session_is_reactivated(memory_engine):
    # Memory-only finalize evicts the cache entry, so mark the cached session
    # completed directly to drive the reactivation path without a database.
    await _create(memory_engine)
    session = memory_engine.session_cache[SESSION_ID]
    session.status = SessionStatus.COMPLETED
    session.completed = datetime.now(UTC)

    result = await _create(memory_engine)

    assert result.status == "success"
    assert "reactivated" in result.message
    assert memory_engine.session_cache[SESSION_ID] is session
    assert result.session_data["status"] == SessionStatus.ACTIVE.value
    assert result.session_data["completed_at"] is None


@pytest.mark.regression
async def test_reactivated_session_found_by_active_project_lookup(engine):
    await _create(engine)
    await engine.session_manage_lifecycle(operation="finalize", session_id=SESSION_ID)
    await _create(engine)

    found = await engine.database.find_recent_session_by_project(PROJECT, status="active")

    assert found is not None
    assert found["id"] == SESSION_ID
