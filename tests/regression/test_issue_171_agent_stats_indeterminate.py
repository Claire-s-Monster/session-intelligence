"""
Regression tests for issue #171: INDETERMINATE agent executions silently
deflate session_agent_stats' success_rate.

https://github.com/Claire-s-Monster/session-intelligence/issues/171

Background:
    Issue #138 introduced ExecutionStatus.INDETERMINATE for agent_stop
    events that reported no parseable outcome (success key absent/None).
    get_agent_stats (both backends) already excludes 'abandoned' rows from
    its denominator (issue #70 / #39), but it only filtered
    `status != 'abandoned'` -- INDETERMINATE rows still landed in
    "invocations" with no success/failure credit, deflating success_rate
    exactly the way ABANDONED rows used to.

Fix (mirrors #168's treatment of abandoned/indeterminate in
PerformanceMetrics.efficiency_score):
- get_agent_stats (both backends) excludes 'indeterminate' rows from
  invocations/successes/failures/avg_duration_ms, and reports a separate
  per-agent "indeterminate" count.
- session_engine.session_agent_stats passes the "indeterminate" count
  through; success_rate is computed over the reduced "invocations" (same
  zero-guard as before: 0.0 when invocations is 0).

PostgreSQL coverage: not added here. get_agent_stats's success_rate
behavior has, by existing convention (see test_issue_70_reconcile_stuck_
executions.py), only ever been covered as a SQLite-backend regression test,
not added to tests/persistence/contract_tests.py's backend-agnostic suite.
This file follows that same precedent; a PostgreSQL run would require a
live POSTGRES_DSN (tests/persistence/test_postgresql_contract.py, gated on
POSTGRES_AVAILABLE).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend

AGENT_NAME = "focused-code-modifier"
AGENT_TYPE = "focused"


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
    agent_name: str = AGENT_NAME,
    agent_type: str = AGENT_TYPE,
) -> dict:
    """Build a raw agent_executions dict ready for SQLiteBackend.save_agent_execution."""
    return {
        "id": f"exec-{uuid.uuid4().hex[:8]}",
        "session_id": session_id,
        "agent_name": agent_name,
        "agent_type": agent_type,
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "status": status,
        "execution_steps": [],
        "performance": {},
        "errors": [],
    }


@pytest.mark.regression
class TestGetAgentStatsExcludesIndeterminate:
    """Backend-level (SQLiteBackend.get_agent_stats) assertions."""

    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db
        await db.close()

    async def test_indeterminate_excluded_from_invocations_and_counted_separately(self, backend):
        session = _session_dict()
        await backend.save_session(session)

        await backend.save_agent_execution(
            _execution_dict(session_id=session["id"], status="success")
        )
        for _ in range(9):
            await backend.save_agent_execution(
                _execution_dict(session_id=session["id"], status="indeterminate")
            )

        stats = await backend.get_agent_stats(time_window_hours=168)
        entry = next(a for a in stats["agents"] if a["agent_type"] == AGENT_TYPE)

        assert entry["invocations"] == 1, (
            "Issue #171: indeterminate executions must not count toward "
            "invocations, only the real success should."
        )
        assert entry["successes"] == 1
        assert entry["failures"] == 0
        assert entry["indeterminate"] == 9, (
            "Indeterminate executions must be reported via a dedicated "
            "per-agent counter, not silently dropped."
        )

    async def test_abandoned_still_excluded_and_not_counted_as_indeterminate(self, backend):
        session = _session_dict()
        await backend.save_session(session)

        await backend.save_agent_execution(
            _execution_dict(session_id=session["id"], status="success")
        )
        await backend.save_agent_execution(
            _execution_dict(session_id=session["id"], status="abandoned")
        )
        await backend.save_agent_execution(
            _execution_dict(session_id=session["id"], status="indeterminate")
        )

        stats = await backend.get_agent_stats(time_window_hours=168)
        entry = next(a for a in stats["agents"] if a["agent_type"] == AGENT_TYPE)

        assert entry["invocations"] == 1, "abandoned must stay excluded entirely (issue #70)."
        assert entry["successes"] == 1
        assert entry["indeterminate"] == 1, (
            "abandoned rows must not be folded into the indeterminate counter."
        )

    async def test_all_indeterminate_zero_invocations(self, backend):
        session = _session_dict()
        await backend.save_session(session)

        for _ in range(5):
            await backend.save_agent_execution(
                _execution_dict(session_id=session["id"], status="indeterminate")
            )

        stats = await backend.get_agent_stats(time_window_hours=168)
        entry = next(a for a in stats["agents"] if a["agent_type"] == AGENT_TYPE)

        assert entry["invocations"] == 0
        assert entry["successes"] == 0
        assert entry["failures"] == 0
        assert entry["indeterminate"] == 5


@pytest.mark.regression
class TestSessionAgentStatsSuccessRate:
    """Engine-level (session_engine.session_agent_stats) success_rate assertions."""

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

    async def test_success_rate_ignores_indeterminate_denominator(self, engine):
        session = _session_dict()
        await engine.database.save_session(session)

        await engine.database.save_agent_execution(
            _execution_dict(session_id=session["id"], status="success")
        )
        for _ in range(9):
            await engine.database.save_agent_execution(
                _execution_dict(session_id=session["id"], status="indeterminate")
            )

        result = await engine.session_agent_stats(time_window_hours=168)
        entry = next(a for a in result["agent_stats"] if a["agent_type"] == AGENT_TYPE)

        assert entry["invocations"] == 1
        assert entry["indeterminate"] == 9
        assert entry["success_rate"] == 1.0, (
            "Issue #171: 1 success + 9 indeterminate must compute as 100% "
            "success_rate, NOT 10%. Counting indeterminate executions in "
            "the denominator silently understates success_rate."
        )

    async def test_all_indeterminate_success_rate_zero_guard(self, engine):
        session = _session_dict()
        await engine.database.save_session(session)

        for _ in range(4):
            await engine.database.save_agent_execution(
                _execution_dict(session_id=session["id"], status="indeterminate")
            )

        result = await engine.session_agent_stats(time_window_hours=168)
        entry = next(a for a in result["agent_stats"] if a["agent_type"] == AGENT_TYPE)

        assert entry["invocations"] == 0
        assert entry["indeterminate"] == 4
        assert entry["success_rate"] == 0.0, (
            "Pre-existing zero-guard: when invocations is 0 (every "
            "execution in the window was indeterminate), success_rate "
            "falls back to 0.0 rather than raising ZeroDivisionError."
        )
