"""
Regression tests for issue #146 (split binding detection).

`session_monitor_health` scored an orphaned session 100.0 with no warnings when
the session had zero recorded executions while another active session of the
same project recorded executions after it started.
"""

from datetime import UTC, datetime, timedelta

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend

A_ID = "aaaaaaaa-0000-4000-8000-000000000001"
B_ID = "bbbbbbbb-0000-4000-8000-000000000002"
PROJECT = "split-project"
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    """SessionIntelligenceEngine wired to a fresh in-process SQLite backend."""
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issue146.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


async def _save_session(
    engine: SessionIntelligenceEngine,
    session_id: str,
    started: datetime,
    project: str = PROJECT,
    status: str = "active",
) -> None:
    await engine.database.save_session(
        {
            "id": session_id,
            "started": started.isoformat(),
            "project_name": project,
            "project_path": "",
            "mode": "local",
            "status": status,
            "metadata": {},
            "performance_metrics": {},
            "health_status": {},
        }
    )


async def _save_execution(
    engine: SessionIntelligenceEngine, session_id: str, started: datetime
) -> None:
    await engine.database.save_agent_execution(
        {
            "id": f"exec-{session_id}-{started.timestamp()}",
            "session_id": session_id,
            "agent_name": "some-agent",
            "started_at": started.isoformat(),
            "status": "success",
        }
    )


async def _health(engine, checks=None):
    return await engine._monitor_health_sync(
        session_id=A_ID,
        health_checks=checks or ["agents"],
        auto_recover=False,
        alert_thresholds=None,
        include_diagnostics=True,
    )


@pytest.mark.regression
async def test_split_binding_detected(engine):
    await _save_session(engine, A_ID, T0)
    await _save_session(engine, B_ID, T0 + timedelta(seconds=30))
    await _save_execution(engine, B_ID, T0 + timedelta(minutes=1))

    result = await _health(engine)

    assert result.health_score == 90.0
    assert result.issues == []
    assert len(result.warnings) == 1
    assert "split binding" in result.warnings[0]
    assert B_ID in result.warnings[0]
    assert result.diagnostics["split_binding_peers"] == [B_ID]
    assert any("native Claude Code session UUID" in a for a in result.recovery_actions)


@pytest.mark.regression
async def test_split_binding_detected_beyond_first_page_of_active_sessions(engine):
    await _save_session(engine, A_ID, T0)
    await _save_session(engine, B_ID, T0 + timedelta(seconds=30))
    await _save_execution(engine, B_ID, T0 + timedelta(minutes=1))
    # 60 unrelated active sessions that sort ahead of B (newer started_at).
    for i in range(60):
        await _save_session(
            engine,
            f"cccccccc-0000-4000-8000-{i:012d}",
            T0 + timedelta(hours=1, seconds=i),
            project="unrelated-project",
        )

    result = await _health(engine)

    assert result.diagnostics["split_binding_peers"] == [B_ID]
    assert result.health_score == 90.0


@pytest.mark.regression
async def test_own_executions_suppress_warning(engine):
    await _save_session(engine, A_ID, T0)
    await _save_session(engine, B_ID, T0)
    await _save_execution(engine, A_ID, T0 + timedelta(seconds=10))
    await _save_execution(engine, B_ID, T0 + timedelta(minutes=1))

    result = await _health(engine)

    assert result.warnings == []
    assert result.health_score == 100.0


@pytest.mark.regression
async def test_peer_execution_before_start_ignored(engine):
    await _save_session(engine, A_ID, T0)
    await _save_session(engine, B_ID, T0 - timedelta(hours=1))
    await _save_execution(engine, B_ID, T0 - timedelta(minutes=5))

    result = await _health(engine)

    assert result.warnings == []
    assert result.health_score == 100.0


@pytest.mark.regression
async def test_peer_in_other_project_ignored(engine):
    await _save_session(engine, A_ID, T0)
    await _save_session(engine, B_ID, T0, project="other-project")
    await _save_execution(engine, B_ID, T0 + timedelta(minutes=1))

    result = await _health(engine)

    assert result.warnings == []
    assert result.health_score == 100.0


@pytest.mark.regression
@pytest.mark.parametrize("status", ["completed", "abandoned"])
async def test_inactive_peer_ignored(engine, status):
    await _save_session(engine, A_ID, T0)
    await _save_session(engine, B_ID, T0, status=status)
    await _save_execution(engine, B_ID, T0 + timedelta(minutes=1))

    result = await _health(engine)

    assert result.warnings == []
    assert result.health_score == 100.0


@pytest.mark.regression
async def test_agents_check_not_requested(engine):
    await _save_session(engine, A_ID, T0)
    await _save_session(engine, B_ID, T0)
    await _save_execution(engine, B_ID, T0 + timedelta(minutes=1))

    result = await _health(engine, checks=["state"])

    assert result.warnings == []
    assert result.health_score == 100.0
