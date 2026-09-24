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

from datetime import UTC, datetime, timedelta

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import (
    AgentContext,
    AgentExecution,
    ExecutionStatus,
    ExecutionStep,
    Session,
    SessionMetadata,
)
from persistence.sqlite import SQLiteBackend

NATIVE_SESSION_ID = "cd2b76d0-37cc-4768-a466-2b61d3fd8947"


def _make_agent_execution(
    agent_name: str,
    status: ExecutionStatus,
    started: datetime,
    completed: datetime | None = None,
    execution_steps: list[ExecutionStep] | None = None,
) -> AgentExecution:
    """Build a minimal AgentExecution for direct metrics-derivation tests,
    bypassing the session_track_execution hook-event flow so started/
    completed timestamps are exactly controllable."""
    return AgentExecution(
        agent_name=agent_name,
        agent_type="focused",
        execution_id=f"{agent_name}-exec",
        started=started,
        completed=completed,
        status=status,
        execution_steps=execution_steps or [],
        context=AgentContext(
            session_id=NATIVE_SESSION_ID,
            project_path="",
            working_directory="",
        ),
    )


def _make_session(agents_executed: list[AgentExecution]) -> Session:
    """Build a minimal Session wrapping the given agent executions."""
    return Session(
        id=NATIVE_SESSION_ID,
        started=datetime.now(UTC),
        project_name="demo-project",
        project_path="",
        metadata=SessionMetadata(session_type="test", environment="test", user="test"),
        agents_executed=agents_executed,
    )


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


# ---------------------------------------------------------------------------
# efficiency_score (issue #115 cause 2)
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_efficiency_score_none_with_zero_terminal_executions(engine):
    """efficiency_score must be None, not 0.0, when no execution has
    reached a terminal (success/failed) state -- an unmeasured session
    must not read as 0% efficient."""
    started = datetime.now(UTC)
    session = _make_session(
        [_make_agent_execution("agent-running", ExecutionStatus.RUNNING, started)]
    )
    engine._recompute_derived_metrics(session)
    assert session.performance_metrics.efficiency_score is None


@pytest.mark.regression
async def test_efficiency_score_all_success_is_100(engine):
    """All-SUCCESS executions must report a 100.0 efficiency score."""
    started = datetime.now(UTC)
    session = _make_session(
        [
            _make_agent_execution("agent-1", ExecutionStatus.SUCCESS, started, started),
            _make_agent_execution("agent-2", ExecutionStatus.SUCCESS, started, started),
        ]
    )
    engine._recompute_derived_metrics(session)
    assert session.performance_metrics.efficiency_score == 100.0


@pytest.mark.regression
async def test_efficiency_score_mixed_success_and_failure(engine):
    """3 SUCCESS + 1 ERROR -> 75.0."""
    started = datetime.now(UTC)
    agents = [
        _make_agent_execution(f"agent-success-{i}", ExecutionStatus.SUCCESS, started, started)
        for i in range(3)
    ] + [_make_agent_execution("agent-error-1", ExecutionStatus.ERROR, started, started)]
    session = _make_session(agents)
    engine._recompute_derived_metrics(session)
    assert session.performance_metrics.efficiency_score == 75.0


@pytest.mark.regression
async def test_efficiency_score_excludes_running_from_denominator(engine):
    """1 SUCCESS + 1 RUNNING -> 100.0, not 50.0: RUNNING is not a terminal
    outcome and must not widen the denominator (see #70 -- the same
    reasoning that keeps ABANDONED out of failed_executions)."""
    started = datetime.now(UTC)
    session = _make_session(
        [
            _make_agent_execution("agent-success", ExecutionStatus.SUCCESS, started, started),
            _make_agent_execution("agent-running", ExecutionStatus.RUNNING, started),
        ]
    )
    engine._recompute_derived_metrics(session)
    assert session.performance_metrics.efficiency_score == 100.0


# ---------------------------------------------------------------------------
# commands_executed
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_commands_executed_sums_across_steps_and_executions(engine):
    """commands_executed must sum CommandExecution entries across every
    step of every agent execution, not just the most recently added step."""
    await engine.session_track_execution(
        session_id=NATIVE_SESSION_ID,
        agent_name="agent-a",
        step_data={"phase": "tool_use", "command": "cmd1", "agent_type": "focused"},
    )
    await engine.session_track_execution(
        session_id=NATIVE_SESSION_ID,
        agent_name="agent-a",
        step_data={"phase": "tool_use", "command": "cmd2", "agent_type": "focused"},
    )
    await engine.session_track_execution(
        session_id=NATIVE_SESSION_ID,
        agent_name="agent-b",
        step_data={
            "phase": "agent_stop",
            "success": True,
            "agent_type": "focused",
            "commands_executed": ["cmd3", "cmd4"],
        },
    )

    session = engine.session_cache[NATIVE_SESSION_ID]
    assert session.performance_metrics.commands_executed == 4


# ---------------------------------------------------------------------------
# average_execution_time_ms
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_average_execution_time_none_when_none_completed(engine):
    """average_execution_time_ms must be None, not 0.0, when no agent
    execution has completed -- an unmeasured session must not read as a
    confident 0s average."""
    started = datetime.now(UTC)
    session = _make_session(
        [_make_agent_execution("agent-running", ExecutionStatus.RUNNING, started)]
    )
    engine._recompute_derived_metrics(session)
    assert session.performance_metrics.average_execution_time_ms is None


@pytest.mark.regression
async def test_average_execution_time_is_correct_mean(engine):
    """Mean must come from AgentExecution.started/.completed wall-clock
    duration, NOT from summing ExecutionStep.duration_ms -- duration_ms is
    sourced from hook step_data via `.get("duration_ms", 0)` and is 0 for
    most real steps, so summing it would silently reproduce the exact
    confident-zero bug this issue reports, just moved to a different
    field. The step below is deliberately given duration_ms=0 to prove it
    is ignored."""
    started = datetime.now(UTC)
    agents = [
        _make_agent_execution(
            "agent-1",
            ExecutionStatus.SUCCESS,
            started,
            started + timedelta(milliseconds=1000),
            execution_steps=[
                ExecutionStep(
                    step_id="agent-1-step-1",
                    step_number=1,
                    agent="agent-1",
                    operation="agent_stop",
                    description="",
                    started=started,
                    completed=started + timedelta(milliseconds=1000),
                    duration_ms=0,
                )
            ],
        ),
        _make_agent_execution(
            "agent-2",
            ExecutionStatus.SUCCESS,
            started,
            started + timedelta(milliseconds=3000),
        ),
    ]
    session = _make_session(agents)
    engine._recompute_derived_metrics(session)
    assert session.performance_metrics.average_execution_time_ms == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# Notebook renderer: n/a rendering (TypeError regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_metrics_section_renders_na_for_unmeasured_session(engine):
    """The notebook metrics table must render 'n/a' for an unmeasured
    session's efficiency score and total execution time, not raise
    TypeError from formatting None with ':.1f'."""
    session = _make_session([])
    rendered = engine._generate_metrics_section(session)
    assert "| Efficiency Score | n/a |" in rendered
    assert "| Total Execution Time | n/a |" in rendered


@pytest.mark.regression
async def test_metrics_section_renders_real_numbers_for_measured_session(engine):
    """Once terminal executions and a finalized wall-clock time exist, the
    renderer must show real numbers, not 'n/a'."""
    started = datetime.now(UTC)
    session = _make_session(
        [
            _make_agent_execution("agent-1", ExecutionStatus.SUCCESS, started, started),
            _make_agent_execution("agent-2", ExecutionStatus.ERROR, started, started),
        ]
    )
    session.performance_metrics.total_execution_time_ms = 5000
    engine._recompute_derived_metrics(session)

    rendered = engine._generate_metrics_section(session)
    assert "| Efficiency Score | 50.0% |" in rendered
    assert "| Total Execution Time | 5.0s |" in rendered
