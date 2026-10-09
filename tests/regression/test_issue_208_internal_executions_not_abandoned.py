"""
Regression tests for issue #208: internal pseudo-agent executions wrongly
closed as ABANDONED.

INTERNAL_AGENT_NAMES ("task-manager", "bash-executor") are hook-side
record-keeping containers. They never receive agent_stop, so the #70
finalize reconciliation and startup sweep closed them as ABANDONED, inflating
abandoned_executions. Fix: they close as INDETERMINATE instead, and their
sweep/finalize-time `completed` is kept out of average_execution_time_ms.

https://github.com/Claire-s-Monster/session-intelligence/issues/208
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from core.session_engine import INTERNAL_AGENT_NAMES, SessionIntelligenceEngine
from models.session_models import (
    AgentContext,
    AgentExecution,
    ExecutionStatus,
    Session,
    SessionMetadata,
)
from persistence.base import DEFAULT_EXECUTION_MAX_AGE_HOURS
from persistence.sqlite import SQLiteBackend

REAL_AGENT = "focused-code-modifier"
NATIVE_SESSION_ID = "3f7a2c1e-5b8d-4a6f-9c2e-7d1b4e6a8c3f"


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
    db = SQLiteBackend(str(tmp_path / "test.db"))
    await db.initialize()
    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )
    yield eng
    await db.close()


async def _start(engine, session_id: str, agent_name: str) -> None:
    result = await engine.session_track_execution(
        session_id=session_id,
        agent_name=agent_name,
        step_data={"operation": "start", "description": "hook start"},
    )
    assert result.status == "success"


async def _new_session(engine) -> str:
    create_result = await engine.session_manage_lifecycle(
        operation="create", mode="local", project_name="proj-208"
    )
    return create_result.session_id


def _exec_by_name(engine, session_id: str, agent_name: str) -> AgentExecution:
    session = engine.session_cache[session_id]
    return next(a for a in session.agents_executed if a.agent_name == agent_name)


@pytest.mark.regression
class TestFinalizeClosesInternalExecutionsAsIndeterminate:
    async def _finalize_with_internal(self, engine, internal_name: str):
        session_id = await _new_session(engine)
        await _start(engine, session_id, internal_name)
        await _start(engine, session_id, REAL_AGENT)
        await engine.session_track_execution(
            session_id=session_id,
            agent_name=REAL_AGENT,
            step_data={"phase": "agent_stop", "agent_type": "focused", "success": True},
        )
        internal_exec = _exec_by_name(engine, session_id, internal_name)
        real_exec = _exec_by_name(engine, session_id, REAL_AGENT)
        assert internal_exec.status == ExecutionStatus.RUNNING
        assert real_exec.status == ExecutionStatus.SUCCESS

        result = await engine.session_manage_lifecycle(operation="finalize", session_id=session_id)
        assert result.status == "success"
        # finalize evicts the session from session_cache; the in-memory
        # execution object is mutated in place, and the session is re-read
        # from the DB for the persisted metrics.
        return session_id, internal_exec, real_exec

    @pytest.mark.parametrize("internal_name", ["bash-executor", "task-manager"])
    async def test_internal_execution_becomes_indeterminate(self, engine, internal_name):
        session_id, internal_exec, real_exec = await self._finalize_with_internal(
            engine, internal_name
        )

        session = await engine._hydrate_session(session_id)
        assert session.performance_metrics.abandoned_executions == 0
        assert session.performance_metrics.indeterminate_executions == 1
        assert internal_exec.status == ExecutionStatus.INDETERMINATE

        rows = await engine.database.query_agent_executions(session_id=session_id)
        internal_rows = [row for row in rows if row["agent_name"] == internal_name]
        assert [row["status"] for row in internal_rows] == ["indeterminate"]
        assert internal_rows[0]["completed_at"] is not None
        assert real_exec.status == ExecutionStatus.SUCCESS

    async def test_non_internal_running_agent_still_abandoned(self, engine):
        session_id = await _new_session(engine)
        await _start(engine, session_id, REAL_AGENT)
        real_exec = _exec_by_name(engine, session_id, REAL_AGENT)

        result = await engine.session_manage_lifecycle(operation="finalize", session_id=session_id)
        assert result.status == "success"

        session = await engine._hydrate_session(session_id)
        assert session.performance_metrics.abandoned_executions == 1
        assert session.performance_metrics.indeterminate_executions == 0
        assert real_exec.status == ExecutionStatus.ABANDONED


def _make_agent_execution(
    agent_name: str, status: ExecutionStatus, started: datetime, completed: datetime
) -> AgentExecution:
    return AgentExecution(
        agent_name=agent_name,
        agent_type="focused",
        execution_id=f"{agent_name}-exec",
        started=started,
        completed=completed,
        status=status,
        execution_steps=[],
        context=AgentContext(session_id=NATIVE_SESSION_ID, project_path="", working_directory=""),
    )


@pytest.mark.regression
async def test_average_execution_time_ignores_internal_execution(engine):
    started = datetime.now(UTC)
    session = Session(
        id=NATIVE_SESSION_ID,
        started=started,
        project_name="demo-project",
        project_path="",
        metadata=SessionMetadata(session_type="test", environment="test", user="test"),
        agents_executed=[
            _make_agent_execution(
                "agent-success", ExecutionStatus.SUCCESS, started, started + timedelta(seconds=2)
            ),
            _make_agent_execution(
                "bash-executor",
                ExecutionStatus.INDETERMINATE,
                started,
                started + timedelta(hours=6),
            ),
        ],
    )

    engine._recompute_derived_metrics(session)

    assert session.performance_metrics.average_execution_time_ms == pytest.approx(2000.0)


@pytest.mark.regression
class TestReapStaleInternalExecutions:
    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "reap.db"))
        await db.initialize()
        yield db
        await db.close()

    async def test_reap_internal_indeterminate_normal_abandoned(self, backend):
        sid = f"session-{uuid.uuid4().hex[:8]}"
        await backend.save_session(
            {
                "id": sid,
                "started_at": datetime.now(UTC).isoformat(),
                "ended_at": None,
                "project_path": "/tmp/proj",
                "project_name": "proj",
                "mode": "local",
                "status": "active",
                "metadata": {},
                "performance_metrics": {},
                "health_status": {},
            }
        )
        started = datetime.now(UTC) - timedelta(hours=DEFAULT_EXECUTION_MAX_AGE_HOURS + 1)
        ids = {}
        for name in (*sorted(INTERNAL_AGENT_NAMES), REAL_AGENT):
            ids[name] = f"exec-{uuid.uuid4().hex[:8]}"
            await backend.save_agent_execution(
                {
                    "id": ids[name],
                    "session_id": sid,
                    "agent_name": name,
                    "agent_type": "focused",
                    "started_at": started.isoformat(),
                    "completed_at": None,
                    "status": "running",
                    "execution_steps": [],
                    "performance": {},
                    "errors": [],
                }
            )

        assert await backend.reap_stale_executions() == 3

        rows = {row["id"]: row for row in await backend.query_agent_executions(session_id=sid)}
        for name in INTERNAL_AGENT_NAMES:
            assert rows[ids[name]]["status"] == "indeterminate"
        assert rows[ids[REAL_AGENT]]["status"] == "abandoned"
