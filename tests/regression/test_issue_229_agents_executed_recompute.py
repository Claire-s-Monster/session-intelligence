"""Regression tests for issue #229: performance_metrics.agents_executed was
never recomputed on hydrate.

`_recompute_derived_metrics` derived every PerformanceMetrics counter except
agents_executed, so a stale stored value survived `_hydrate_session` even
though the agent_executions rows were loaded. It now derives it from
`len(session.agents_executed)` like the others.

https://github.com/Claire-s-Monster/session-intelligence/issues/229
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import (
    AgentContext,
    AgentExecution,
    ExecutionStatus,
    Session,
    SessionMetadata,
)
from persistence.sqlite import SQLiteBackend

SESSION_ID = "9b1d5c3e-2a4f-4e7b-8c6d-1f3a5b7c9d2e"


def _make_session(count: int) -> Session:
    executions = [
        AgentExecution(
            agent_name=f"agent-{i}",
            agent_type="focused",
            execution_id=f"exec-{i}",
            started=datetime.now(UTC),
            status=ExecutionStatus.SUCCESS,
            execution_steps=[],
            context=AgentContext(session_id=SESSION_ID, project_path="", working_directory=""),
        )
        for i in range(count)
    ]
    return Session(
        id=SESSION_ID,
        started=datetime.now(UTC),
        project_name="demo-project",
        project_path="",
        metadata=SessionMetadata(session_type="test", environment="test", user="test"),
        agents_executed=executions,
    )


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "issue229-agents-executed.db"))
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
async def test_recompute_overrides_stale_agents_executed(engine):
    """Issue #229: a stored agents_executed=2 over a 5-item list is corrected."""
    session = _make_session(5)
    session.performance_metrics.agents_executed = 2

    engine._recompute_derived_metrics(session)

    assert session.performance_metrics.agents_executed == 5


@pytest.mark.regression
async def test_hydrate_corrects_stale_stored_agents_executed(engine, db):
    """Issue #229: hydrate with stored blob agents_executed=2 and 5 rows -> 5."""
    await db.save_session(
        {
            "id": SESSION_ID,
            "started": datetime.now(UTC).isoformat(),
            "project_path": "/tmp/issue-229-project",
            "project_name": "issue-229-project",
            "mode": "local",
            "status": "active",
            "metadata": {},
            "performance_metrics": {"agents_executed": 2},
            "health_status": {},
        }
    )
    for i in range(5):
        await db.save_agent_execution(
            {
                "id": f"exec-{i}",
                "session_id": SESSION_ID,
                "agent_name": f"agent-{i}",
                "agent_type": "focused",
                "started_at": datetime.now(UTC).isoformat(),
                "completed_at": datetime.now(UTC).isoformat(),
                "status": "success",
                "execution_steps": [],
                "performance": {},
                "errors": [],
            }
        )
    engine.session_cache.clear()

    session = await engine._hydrate_session(SESSION_ID)

    assert len(session.agents_executed) == 5
    assert session.performance_metrics.agents_executed == 5


@pytest.mark.regression
async def test_empty_session_agents_executed_is_zero(engine):
    """Issue #229: an empty session recomputes to agents_executed == 0."""
    session = _make_session(0)
    session.performance_metrics.agents_executed = 3

    engine._recompute_derived_metrics(session)

    assert session.performance_metrics.agents_executed == 0
