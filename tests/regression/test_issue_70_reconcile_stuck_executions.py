"""
Regression tests for issue #70: AgentExecution status permanently stuck at
RUNNING when the SubagentStop hook never reports a stop event.

https://github.com/Claire-s-Monster/session-intelligence/issues/70

Background:
    #40 (issue #39) transitions an AgentExecution out of RUNNING only when
    the SubagentStop hook reports phase == "agent_stop" (see
    session_engine.session_track_execution). If that event never arrives --
    agent killed, session ends mid-flight, hook fails or times out, server
    restart between start and stop -- the row keeps the Pydantic default
    RUNNING forever. As of 2026-08-15, 847 executions started AFTER the #40
    fix were still 'running'. Because get_agent_stats counted every row
    toward the success_rate denominator regardless of status, these rows
    silently understated success_rate for every agent -- the exact metric
    #39 existed to fix.

Fix:
- New ExecutionStatus.ABANDONED, distinct from ERROR: "never reported" is
  not "failed".
- session_engine._finalize_session now reconciles any still-RUNNING
  AgentExecution (and its still-running ExecutionSteps) to ABANDONED before
  persisting the session, and persists each reconciled execution
  individually (agent_executions is a separate table from sessions).
- New reap_stale_executions() backend method (both SQLite and PostgreSQL)
  flips stale 'running' rows to 'abandoned' at HTTP transport startup,
  mirroring issue #69's reap_abandoned_sessions -- catches executions whose
  session was never explicitly finalized either.
- get_agent_stats (both backends) now excludes 'abandoned' rows from the
  success_rate denominator entirely, rather than counting them as
  non-successes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import ExecutionStatus
from persistence.base import DEFAULT_EXECUTION_MAX_AGE_HOURS
from persistence.sqlite import SQLiteBackend

AGENT_NAME = "focused-code-modifier"


def _session_dict(
    *, status: str = "active", project_path: str = "/tmp/proj", project_name: str = "proj"
) -> dict:
    """Build a raw session dict ready for SQLiteBackend.save_session.

    agent_executions has a FOREIGN KEY on session_id, enforced (PRAGMA
    foreign_keys=ON), so tests inserting raw execution rows need a real
    session row to reference first.
    """
    sid = f"session-{uuid.uuid4().hex[:8]}"
    return {
        "id": sid,
        "started_at": datetime.now(UTC).isoformat(),
        "ended_at": None,
        "project_path": project_path,
        "project_name": project_name,
        "mode": "local",
        "status": status,
        "metadata": {},
        "performance_metrics": {},
        "health_status": {},
    }


def _execution_dict(
    *,
    session_id: str,
    status: str = "running",
    age_hours: float = 0.0,
    agent_name: str = AGENT_NAME,
    agent_type: str = "focused",
) -> dict:
    """Build a raw agent_executions dict ready for SQLiteBackend.save_agent_execution."""
    started_at = datetime.now(UTC) - timedelta(hours=age_hours)
    return {
        "id": f"exec-{uuid.uuid4().hex[:8]}",
        "session_id": session_id,
        "agent_name": agent_name,
        "agent_type": agent_type,
        "started_at": started_at.isoformat(),
        "completed_at": None,
        "status": status,
        "execution_steps": [],
        "performance": {},
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Reconcile-on-finalize
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestFinalizeReconcilesRunningExecutions:
    """Finalizing a session must flip its still-RUNNING executions to
    ABANDONED, not leave them RUNNING and not misreport them as ERROR/SUCCESS."""

    @pytest.fixture
    async def engine(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        eng = SessionIntelligenceEngine(
            repository_path=str(tmp_path), use_filesystem=False, database=db
        )
        yield eng
        await db.close()

    async def _start_agent(self, engine):
        # Create via session_manage_lifecycle (not session_track_execution's
        # auto-create path) so the session row is persisted to the DB --
        # finalize-by-session_id requires database.get_session() to resolve
        # it (see _resolve_session_context).
        create_result = await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-70"
        )
        session_id = create_result.session_id

        start_result = await engine.session_track_execution(
            session_id=session_id,
            agent_name=AGENT_NAME,
            step_data={"operation": "start", "description": "SubagentStart hook"},
        )
        assert start_result.status == "success"
        return session_id

    async def test_running_execution_becomes_abandoned_on_finalize(self, engine):
        session_id = await self._start_agent(engine)

        session = engine.session_cache[session_id]
        agent_execution = next(
            a for a in session.agents_executed if a.agent_name == AGENT_NAME
        )
        assert agent_execution.status == ExecutionStatus.RUNNING

        finalize_result = await engine.session_manage_lifecycle(
            operation="finalize", session_id=session_id
        )
        assert finalize_result.status == "success"

        rows = await engine.database.query_agent_executions(session_id=session_id)
        assert len(rows) == 1
        assert rows[0]["status"] == "abandoned", (
            "Issue #70: an execution that never received agent_stop must "
            "become 'abandoned' on finalize -- not 'error' (it never failed) "
            "and not stuck at 'running'."
        )
        assert rows[0]["completed_at"] is not None

    async def test_success_execution_untouched_by_finalize(self, engine):
        """An execution that DID receive agent_stop with success=True must
        not be disturbed by finalize-time reconciliation (no regression of
        #40). Only still-RUNNING executions are reconciled (and thereby
        (re-)persisted); a terminal execution is left exactly as it was, so
        assert against the in-memory object itself rather than the DB (which
        finalize has no reason to touch for an already-terminal row)."""
        session_id = await self._start_agent(engine)

        await engine.session_track_execution(
            session_id=session_id,
            agent_name=AGENT_NAME,
            step_data={"phase": "agent_stop", "agent_type": "focused", "success": True},
        )
        session = engine.session_cache[session_id]
        agent_execution = next(
            a for a in session.agents_executed if a.agent_name == AGENT_NAME
        )
        assert agent_execution.status == ExecutionStatus.SUCCESS
        completed_before = agent_execution.completed

        finalize_result = await engine.session_manage_lifecycle(
            operation="finalize", session_id=session_id
        )
        assert finalize_result.status == "success"

        assert agent_execution.status == ExecutionStatus.SUCCESS, (
            "Finalize must not touch an already-terminal execution."
        )
        assert agent_execution.completed == completed_before

    async def test_error_execution_untouched_by_finalize(self, engine):
        """An execution that DID receive agent_stop with success=False must
        stay ERROR, not get reclassified as ABANDONED by finalize."""
        session_id = await self._start_agent(engine)

        await engine.session_track_execution(
            session_id=session_id,
            agent_name=AGENT_NAME,
            step_data={"phase": "agent_stop", "agent_type": "focused", "success": False},
        )
        session = engine.session_cache[session_id]
        agent_execution = next(
            a for a in session.agents_executed if a.agent_name == AGENT_NAME
        )
        assert agent_execution.status == ExecutionStatus.ERROR
        completed_before = agent_execution.completed

        finalize_result = await engine.session_manage_lifecycle(
            operation="finalize", session_id=session_id
        )
        assert finalize_result.status == "success"

        assert agent_execution.status == ExecutionStatus.ERROR, (
            "Finalize must not touch an already-terminal execution."
        )
        assert agent_execution.completed == completed_before


# ---------------------------------------------------------------------------
# Staleness sweep
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestReapStaleExecutions:
    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db
        await db.close()

    async def test_reap_flips_only_stale_running_rows(self, backend):
        session = _session_dict()
        await backend.save_session(session)

        stale = _execution_dict(
            session_id=session["id"], age_hours=DEFAULT_EXECUTION_MAX_AGE_HOURS + 1
        )
        fresh = _execution_dict(session_id=session["id"], age_hours=1)
        stale_but_success = _execution_dict(
            session_id=session["id"],
            status="success",
            age_hours=DEFAULT_EXECUTION_MAX_AGE_HOURS + 100,
        )
        await backend.save_agent_execution(stale)
        await backend.save_agent_execution(fresh)
        await backend.save_agent_execution(stale_but_success)

        reaped = await backend.reap_stale_executions()

        assert reaped == 1, "Only the stale 'running' row should be reaped."

        rows = {
            row["id"]: row
            for row in await backend.query_agent_executions(session_id=session["id"])
        }
        assert rows[stale["id"]]["status"] == "abandoned"
        assert rows[fresh["id"]]["status"] == "running"
        assert rows[stale_but_success["id"]]["status"] == "success"

    async def test_reap_respects_custom_older_than_hours(self, backend):
        session = _session_dict()
        await backend.save_session(session)
        borderline = _execution_dict(session_id=session["id"], age_hours=5)
        await backend.save_agent_execution(borderline)

        reaped = await backend.reap_stale_executions(older_than_hours=1)

        assert reaped == 1
        rows = await backend.query_agent_executions(session_id=session["id"])
        assert rows[0]["status"] == "abandoned"

    async def test_reap_returns_zero_when_nothing_stale(self, backend):
        session = _session_dict()
        await backend.save_session(session)
        fresh = _execution_dict(session_id=session["id"], age_hours=1)
        await backend.save_agent_execution(fresh)

        reaped = await backend.reap_stale_executions()

        assert reaped == 0
        rows = await backend.query_agent_executions(session_id=session["id"])
        assert rows[0]["status"] == "running"


# ---------------------------------------------------------------------------
# THE METRIC FIX: success_rate must exclude abandoned from the denominator
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestSuccessRateExcludesAbandoned:
    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db
        await db.close()

    async def test_abandoned_rows_excluded_from_denominator(self, backend):
        session = _session_dict()
        await backend.save_session(session)

        for _ in range(8):
            await backend.save_agent_execution(
                _execution_dict(session_id=session["id"], status="success")
            )
        for _ in range(2):
            await backend.save_agent_execution(
                _execution_dict(session_id=session["id"], status="abandoned")
            )

        stats = await backend.get_agent_stats(time_window_hours=168)
        entry = next(a for a in stats["agents"] if a["agent_type"] == "focused")

        assert entry["invocations"] == 8, (
            "Abandoned executions must be excluded entirely from the "
            "invocations count, not counted as a non-success."
        )
        assert entry["successes"] == 8
        assert entry["successes"] / entry["invocations"] == 1.0, (
            "Issue #70: 8 success + 2 abandoned must compute as 100% "
            "success_rate, NOT 80%. Counting abandoned executions in the "
            "denominator silently understates success_rate for every agent "
            "-- the exact metric #39 existed to fix."
        )
