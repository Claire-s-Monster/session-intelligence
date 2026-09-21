"""
Regression tests for issue #115, derived-metrics class.

`Session.performance_metrics` has almost no write sites: only
`total_execution_time_ms` (set in `_finalize_session`) and `agents_executed`
(set in `_track_execution_sync`) were ever assigned. `successful_executions`,
`failed_executions`, and `decisions_made` stayed at their pydantic defaults
forever -- a session with 7 successful executions and 2 decisions still
reported `successful_executions: 0` and `decisions_made: 0`.

Verifies the fix: `_recompute_derived_metrics` derives all three fields from
`session.agents_executed` / `session.decisions` wherever a Session is built,
hydrated, or read back -- including the case where a session exists only in
the database (no `session_cache` entry) and is adopted via `_hydrate_session`
after a process restart, where an increment-style counter would still read
stale or zero.

https://github.com/Claire-s-Monster/session-intelligence/issues/115
"""

from datetime import UTC, datetime

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend

NATIVE_SESSION_ID = "cd2b76d0-37cc-4768-a466-2b61d3fd8947"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    """SessionIntelligenceEngine wired to a fresh in-process SQLite backend."""
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issue115-counters.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


# ---------------------------------------------------------------------------
# successful_executions / failed_executions
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_mixed_success_and_failure_executions_counted_correctly(engine):
    """A session with both successful and failed agent executions must
    report both counts correctly, not just agents_executed.

    agent_type must be supplied here: a typeless agent_stop with no prior
    start is issue #108's progress-summarizer filter and is ignored outright
    (no session/execution is created at all), which is not what this test
    is exercising.
    """
    await engine.session_track_execution(
        session_id=NATIVE_SESSION_ID,
        agent_name="agent-success-1",
        step_data={"phase": "agent_stop", "success": True, "agent_type": "focused"},
    )
    await engine.session_track_execution(
        session_id=NATIVE_SESSION_ID,
        agent_name="agent-success-2",
        step_data={"phase": "agent_stop", "success": True, "agent_type": "focused"},
    )
    await engine.session_track_execution(
        session_id=NATIVE_SESSION_ID,
        agent_name="agent-failure-1",
        step_data={"phase": "agent_stop", "success": False, "agent_type": "focused"},
    )

    session = engine.session_cache[NATIVE_SESSION_ID]
    assert len(session.agents_executed) == 3
    assert session.performance_metrics.successful_executions == 2
    assert session.performance_metrics.failed_executions == 1


# ---------------------------------------------------------------------------
# decisions_made
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_decisions_made_tracks_decisions_list(engine):
    """decisions_made must reflect len(session.decisions), surfaced through
    the validate read-back path."""
    create_result = await engine.session_manage_lifecycle(
        operation="create",
        project_name="demo-project",
        session_id=NATIVE_SESSION_ID,
    )
    assert create_result.status == "success"

    for i in range(3):
        decision_result = await engine.session_log_decision(
            decision=f"decision-{i}",
            session_id=NATIVE_SESSION_ID,
        )
        assert decision_result.decision_id

    validate_result = await engine.session_manage_lifecycle(
        operation="validate",
        session_id=NATIVE_SESSION_ID,
    )

    assert validate_result.session_data is not None
    assert len(validate_result.session_data.decisions) == 3
    assert validate_result.session_data.performance_metrics.decisions_made == 3


# ---------------------------------------------------------------------------
# Hydration: the case that pins the design constraint
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_hydrated_db_only_session_reports_correct_derived_counters(engine):
    """A session that exists ONLY in the database (no session_cache entry --
    e.g. after a `systemctl --user restart`) must report correct derived
    counters once adopted via `_hydrate_session`. An increment-based counter
    would read stale/zero here because nothing incremented it in THIS
    process; deriving from the freshly-loaded lists is self-healing."""
    started = datetime.now(UTC)
    await engine.database.save_session(
        {
            "id": NATIVE_SESSION_ID,
            "started": started.isoformat(),
            "project_name": "demo-project",
            "project_path": "",
            "mode": "local",
            "status": "active",
            "metadata": {},
            "performance_metrics": {},
            "health_status": {},
        }
    )
    await engine.database.save_agent_execution(
        {
            "id": "exec-success-1",
            "session_id": NATIVE_SESSION_ID,
            "agent_name": "agent-success-1",
            "agent_type": "focused",
            "started": started.isoformat(),
            "status": "success",
        }
    )
    await engine.database.save_agent_execution(
        {
            "id": "exec-success-2",
            "session_id": NATIVE_SESSION_ID,
            "agent_name": "agent-success-2",
            "agent_type": "focused",
            "started": started.isoformat(),
            "status": "success",
        }
    )
    await engine.database.save_agent_execution(
        {
            "id": "exec-error-1",
            "session_id": NATIVE_SESSION_ID,
            "agent_name": "agent-error-1",
            "agent_type": "focused",
            "started": started.isoformat(),
            "status": "error",
        }
    )
    await engine.database.save_decision(
        {
            "id": "decision-db-1",
            "session_id": NATIVE_SESSION_ID,
            "timestamp": started.isoformat(),
            "description": "a decision made before the restart",
        }
    )

    # Precondition: nothing in the cache -- this is the DB-only path.
    assert NATIVE_SESSION_ID not in engine.session_cache

    result = await engine.session_manage_lifecycle(
        operation="create",
        project_name="demo-project",
        session_id=NATIVE_SESSION_ID,
    )

    assert result.status == "success"
    assert result.session_data is not None
    assert len(result.session_data.agents_executed) == 3
    metrics = result.session_data.performance_metrics
    assert metrics.successful_executions == 2, (
        "Hydration did not derive successful_executions from the freshly "
        "loaded agent_executions -- an increment-based counter would fail "
        "exactly this way after a process restart."
    )
    assert metrics.failed_executions == 1
    assert metrics.decisions_made == 1


# ---------------------------------------------------------------------------
# Zero case
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_no_executions_or_decisions_reports_zero(engine):
    """A session with no executions and no decisions must report zeros for
    all three derived counters -- no crash, no None."""
    create_result = await engine.session_manage_lifecycle(
        operation="create",
        project_name="demo-project",
        session_id=NATIVE_SESSION_ID,
    )
    assert create_result.status == "success"

    validate_result = await engine.session_manage_lifecycle(
        operation="validate",
        session_id=NATIVE_SESSION_ID,
    )

    assert validate_result.session_data is not None
    metrics = validate_result.session_data.performance_metrics
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 0
    assert metrics.decisions_made == 0
