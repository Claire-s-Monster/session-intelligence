"""
Regression tests for issue #173: average_execution_time_ms inflated by
ABANDONED executions.

`_recompute_derived_metrics` (src/core/session_engine.py) built
average_execution_time_ms from every AgentExecution with `completed is not
None`, regardless of status. The startup sweep (issue #70,
http_server.py:199-208) sets an ABANDONED execution's `completed` to the
sweep time -- not an observed end -- so a single stale-then-reaped execution
can dominate the mean. One abandoned agent inflated a live session's average
to 22.4M ms (~6h).

Decided (#173): exclude executions whose status is ABANDONED from the
average_execution_time_ms computation. INDETERMINATE stays included --
unlike ABANDONED, its `completed` comes from a real agent_stop event, so its
duration is an observed measurement, not a sweep artifact. If no non-
abandoned completed execution exists, the average stays None (unmeasured,
not zero -- same convention as issue #115).

https://github.com/Claire-s-Monster/session-intelligence/issues/173
"""

from datetime import UTC, datetime, timedelta

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

NATIVE_SESSION_ID = "3f7a2c1e-5b8d-4a6f-9c2e-7d1b4e6a8c3f"


def _make_agent_execution(
    agent_name: str,
    status: ExecutionStatus | str,
    started: datetime,
    completed: datetime | None = None,
) -> AgentExecution:
    """Build a minimal AgentExecution for direct metrics-derivation tests."""
    return AgentExecution(
        agent_name=agent_name,
        agent_type="focused",
        execution_id=f"{agent_name}-exec",
        started=started,
        completed=completed,
        status=status,
        execution_steps=[],
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

    db = SQLiteBackend(db_path=str(tmp_path / "issue173-avg-time.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


# ---------------------------------------------------------------------------
# ABANDONED excluded from the average
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_abandoned_execution_excluded_from_average(engine):
    """An ABANDONED execution whose sweep-set `completed` implies a huge
    duration must not be averaged in -- its `completed` is the reaper's
    sweep time, not an observed end."""
    started = datetime.now(UTC)
    session = _make_session(
        [
            _make_agent_execution(
                "agent-success", ExecutionStatus.SUCCESS, started, started + timedelta(seconds=2)
            ),
            _make_agent_execution(
                "agent-abandoned",
                ExecutionStatus.ABANDONED,
                started,
                started + timedelta(hours=6),
            ),
        ]
    )

    engine._recompute_derived_metrics(session)

    assert session.performance_metrics.average_execution_time_ms == pytest.approx(2000.0)


@pytest.mark.regression
async def test_average_over_remaining_completed_executions_is_correct(engine):
    """With multiple non-abandoned completed executions, the mean must be
    computed only over those, excluding the abandoned outlier entirely."""
    started = datetime.now(UTC)
    session = _make_session(
        [
            _make_agent_execution(
                "agent-1", ExecutionStatus.SUCCESS, started, started + timedelta(seconds=1)
            ),
            _make_agent_execution(
                "agent-2", ExecutionStatus.SUCCESS, started, started + timedelta(seconds=3)
            ),
            _make_agent_execution(
                "agent-abandoned",
                ExecutionStatus.ABANDONED,
                started,
                started + timedelta(hours=6),
            ),
        ]
    )

    engine._recompute_derived_metrics(session)

    assert session.performance_metrics.average_execution_time_ms == pytest.approx(2000.0)


@pytest.mark.regression
async def test_only_abandoned_completed_reports_none(engine):
    """When every completed execution is ABANDONED, average_execution_time_ms
    must be None, not 0.0 or the sweep-inflated mean -- unmeasured, not
    zero."""
    started = datetime.now(UTC)
    session = _make_session(
        [
            _make_agent_execution(
                "agent-abandoned-1",
                ExecutionStatus.ABANDONED,
                started,
                started + timedelta(hours=6),
            ),
            _make_agent_execution(
                "agent-abandoned-2",
                ExecutionStatus.ABANDONED,
                started,
                started + timedelta(hours=3),
            ),
        ]
    )

    engine._recompute_derived_metrics(session)

    assert session.performance_metrics.average_execution_time_ms is None


# ---------------------------------------------------------------------------
# INDETERMINATE stays included
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_indeterminate_with_completed_is_included(engine):
    """INDETERMINATE executions must remain in the average -- their
    `completed` comes from a real agent_stop, not the sweep, so their
    duration is a genuine measurement (unlike ABANDONED)."""
    started = datetime.now(UTC)
    session = _make_session(
        [
            _make_agent_execution(
                "agent-success", ExecutionStatus.SUCCESS, started, started + timedelta(seconds=1)
            ),
            _make_agent_execution(
                "agent-indeterminate",
                ExecutionStatus.INDETERMINATE,
                started,
                started + timedelta(seconds=3),
            ),
        ]
    )

    engine._recompute_derived_metrics(session)

    assert session.performance_metrics.average_execution_time_ms == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# Plain string status "abandoned" is excluded identically (StrEnum equality)
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_plain_string_abandoned_status_also_excluded(engine):
    """AgentExecution.status given as the plain string "abandoned" (equal to
    ExecutionStatus.ABANDONED's value) must be excluded identically to the
    enum member."""
    started = datetime.now(UTC)
    session = _make_session(
        [
            _make_agent_execution(
                "agent-success", ExecutionStatus.SUCCESS, started, started + timedelta(seconds=2)
            ),
            _make_agent_execution(
                "agent-abandoned-string",
                "abandoned",
                started,
                started + timedelta(hours=6),
            ),
        ]
    )

    engine._recompute_derived_metrics(session)

    assert session.performance_metrics.average_execution_time_ms == pytest.approx(2000.0)
