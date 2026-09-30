"""
Regression tests for issue #168: surface the executions efficiency_score's
denominator silently excludes.

`_recompute_derived_metrics` (src/core/session_engine.py) computes
successful_executions (SUCCESS) and failed_executions (ERROR), then derives
efficiency_score = successful / (successful + failed) * 100, or None when
that denominator is zero. ABANDONED and INDETERMINATE are terminal states
(see ExecutionStatus in src/models/session_models.py) that fall out of both
the numerator and the denominator silently: 1 SUCCESS + 9 INDETERMINATE
reports efficiency_score 100.0, and an all-INDETERMINATE session reports
None -- identical to a session where nothing ran at all.

Decided (#168): surface the excluded counts as two new PerformanceMetrics
fields, abandoned_executions and indeterminate_executions. Do NOT change the
efficiency_score formula or any existing field's meaning -- these are purely
additive, read-only counters a reader can use to tell "could not measure"
apart from "nothing happened".

https://github.com/Claire-s-Monster/session-intelligence/issues/168
"""

from datetime import UTC, datetime

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import (
    AgentContext,
    AgentExecution,
    ExecutionStatus,
    PerformanceMetrics,
    Session,
    SessionMetadata,
)
from persistence.sqlite import SQLiteBackend

NATIVE_SESSION_ID = "8a4f1c2e-9b3d-4e7a-8c6f-1d2e3f4a5b6c"


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

    db = SQLiteBackend(db_path=str(tmp_path / "issue168-excluded.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


# ---------------------------------------------------------------------------
# PerformanceMetrics defaults
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_performance_metrics_new_fields_default_to_zero():
    """abandoned_executions and indeterminate_executions must default to 0,
    not None, matching the existing successful_executions/failed_executions
    defaults."""
    metrics = PerformanceMetrics()
    assert metrics.abandoned_executions == 0
    assert metrics.indeterminate_executions == 0


# ---------------------------------------------------------------------------
# 1 SUCCESS + 9 INDETERMINATE -- the issue #168 headline case
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_one_success_nine_indeterminate_score_100_but_excluded_visible(engine):
    """efficiency_score must remain 100.0 (formula unchanged, #168 decided
    against touching it), but indeterminate_executions must report 9 so a
    reader can see the score's denominator only covers 1 of 10 executions."""
    started = datetime.now(UTC)
    agents = [_make_agent_execution("agent-success", ExecutionStatus.SUCCESS, started, started)]
    agents += [
        _make_agent_execution(f"agent-indeterminate-{i}", ExecutionStatus.INDETERMINATE, started)
        for i in range(9)
    ]
    session = _make_session(agents)

    engine._recompute_derived_metrics(session)

    metrics = session.performance_metrics
    assert metrics.efficiency_score == 100.0
    assert metrics.successful_executions == 1
    assert metrics.failed_executions == 0
    assert metrics.indeterminate_executions == 9
    assert metrics.abandoned_executions == 0


# ---------------------------------------------------------------------------
# All-INDETERMINATE -- distinguishable from an empty session
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_all_indeterminate_score_none_but_distinguishable_from_empty(engine):
    """An all-INDETERMINATE session must still report efficiency_score None
    (unchanged formula), but indeterminate_executions must equal the agent
    count -- unlike an empty session where both fields report 0, making the
    two "None" cases distinguishable."""
    started = datetime.now(UTC)
    agents = [
        _make_agent_execution(f"agent-indeterminate-{i}", ExecutionStatus.INDETERMINATE, started)
        for i in range(4)
    ]
    session = _make_session(agents)

    engine._recompute_derived_metrics(session)

    metrics = session.performance_metrics
    assert metrics.efficiency_score is None
    assert metrics.indeterminate_executions == 4
    assert metrics.abandoned_executions == 0

    empty_session = _make_session([])
    engine._recompute_derived_metrics(empty_session)
    empty_metrics = empty_session.performance_metrics
    assert empty_metrics.efficiency_score is None
    assert empty_metrics.indeterminate_executions == 0


# ---------------------------------------------------------------------------
# ABANDONED counted in abandoned_executions, not failed_executions
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_abandoned_counted_separately_not_in_failed(engine):
    """ABANDONED executions must be counted in abandoned_executions and must
    NOT be counted in failed_executions (issue #70's distinction is
    preserved -- this is purely additive)."""
    started = datetime.now(UTC)
    session = _make_session(
        [
            _make_agent_execution("agent-success", ExecutionStatus.SUCCESS, started, started),
            _make_agent_execution("agent-abandoned-1", ExecutionStatus.ABANDONED, started),
            _make_agent_execution("agent-abandoned-2", ExecutionStatus.ABANDONED, started),
        ]
    )

    engine._recompute_derived_metrics(session)

    metrics = session.performance_metrics
    assert metrics.abandoned_executions == 2
    assert metrics.failed_executions == 0
    assert metrics.successful_executions == 1
    assert metrics.efficiency_score == 100.0


# ---------------------------------------------------------------------------
# PENDING / RUNNING / SKIPPED land in neither new field
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_pending_running_skipped_counted_in_neither_new_field(engine):
    """Non-terminal states must not be swept into either new counter."""
    started = datetime.now(UTC)
    session = _make_session(
        [
            _make_agent_execution("agent-pending", ExecutionStatus.PENDING, started),
            _make_agent_execution("agent-running", ExecutionStatus.RUNNING, started),
            _make_agent_execution("agent-skipped", ExecutionStatus.SKIPPED, started),
        ]
    )

    engine._recompute_derived_metrics(session)

    metrics = session.performance_metrics
    assert metrics.abandoned_executions == 0
    assert metrics.indeterminate_executions == 0
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 0
    assert metrics.efficiency_score is None


# ---------------------------------------------------------------------------
# Plain string status values are counted identically (StrEnum equality)
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_plain_string_status_values_counted_identically(engine):
    """AgentExecution.status given as a plain string equal to the enum
    value must be counted the same way as the enum member itself."""
    started = datetime.now(UTC)
    session = _make_session(
        [
            _make_agent_execution("agent-abandoned", "abandoned", started),
            _make_agent_execution("agent-indeterminate", "indeterminate", started),
        ]
    )

    engine._recompute_derived_metrics(session)

    metrics = session.performance_metrics
    assert metrics.abandoned_executions == 1
    assert metrics.indeterminate_executions == 1


# ---------------------------------------------------------------------------
# Notebook metrics table: Abandoned Executions row
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_metrics_section_renders_abandoned_executions_row(engine):
    """The notebook's metrics table must render an "Abandoned Executions"
    row, following the same pattern as the existing "Indeterminate
    Executions" row (issue #138)."""
    started = datetime.now(UTC)
    session = _make_session(
        [
            _make_agent_execution("agent-success", ExecutionStatus.SUCCESS, started, started),
            _make_agent_execution("agent-abandoned", ExecutionStatus.ABANDONED, started),
        ]
    )
    engine._recompute_derived_metrics(session)

    rendered = engine._generate_metrics_section(session)

    assert "| Abandoned Executions | 1 |" in rendered
