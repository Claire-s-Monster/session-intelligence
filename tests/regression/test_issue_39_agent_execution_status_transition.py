"""
Regression tests for issue #39: AgentExecution status never transitions
from RUNNING; successes/failures always 0 in session_agent_stats.

https://github.com/Claire-s-Monster/session-intelligence/issues/39

Verifies the fix:
  `_track_execution_sync` now inspects `step_data["phase"]`. When the
  SubagentStop hook reports `phase="agent_stop"`, the ExecutionStep and
  the matching AgentExecution transition out of ExecutionStatus.RUNNING
  into ExecutionStatus.SUCCESS (step_data["success"] truthy) or
  ExecutionStatus.ERROR (falsy), with `completed` timestamps set.
  Non-agent_stop phases (e.g. agent_start) continue to create/append
  RUNNING steps unchanged.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import ExecutionStatus
from persistence.sqlite import SQLiteBackend

AGENT_NAME = "focused-code-modifier"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    """SessionIntelligenceEngine wired to a fresh in-process SQLite backend.

    Filesystem persistence is OFF - in-memory AgentExecution/ExecutionStep
    status assertions are the focus of the issue #39 fix verification.
    """
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issue39.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


# ---------------------------------------------------------------------------
# Issue #39: agent_stop phase transitions RUNNING -> SUCCESS/ERROR
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_agent_stop_success_transitions_to_success_status(engine):
    """A SubagentStart-like call followed by an agent_stop call with
    success=True must transition the AgentExecution out of RUNNING into
    SUCCESS, not leave it stuck at RUNNING."""
    start_result = await engine.session_track_execution(
        session_id=None,
        agent_name=AGENT_NAME,
        step_data={"operation": "start", "description": "SubagentStart hook"},
        allow_unbound=True,
    )
    assert start_result.status == "success"
    session_id = start_result.session_id

    session = engine.session_cache[session_id]
    agent_execution = next(
        a for a in session.agents_executed if a.agent_name == AGENT_NAME
    )
    assert agent_execution.status == ExecutionStatus.RUNNING

    stop_result = await engine.session_track_execution(
        session_id=session_id,
        agent_name=AGENT_NAME,
        step_data={
            "phase": "agent_stop",
            "agent_type": "focused",
            "success": True,
            "tools_used": ["Read", "Edit"],
        },
    )
    assert stop_result.status == "success"

    agent_execution = next(
        a for a in session.agents_executed if a.agent_name == AGENT_NAME
    )
    assert agent_execution.status == ExecutionStatus.SUCCESS, (
        "AgentExecution.status did not transition from RUNNING to SUCCESS "
        "on agent_stop with success=True. This is issue #39."
    )
    assert agent_execution.completed is not None

    last_step = agent_execution.execution_steps[-1]
    assert last_step.status == ExecutionStatus.SUCCESS
    assert last_step.completed is not None


@pytest.mark.regression
async def test_agent_stop_failure_transitions_to_error_status(engine):
    """An agent_stop call with success=False must transition the
    AgentExecution into ERROR, not leave it stuck at RUNNING."""
    start_result = await engine.session_track_execution(
        session_id=None,
        agent_name=AGENT_NAME,
        step_data={"operation": "start", "description": "SubagentStart hook"},
        allow_unbound=True,
    )
    session_id = start_result.session_id

    stop_result = await engine.session_track_execution(
        session_id=session_id,
        agent_name=AGENT_NAME,
        step_data={
            "phase": "agent_stop",
            "agent_type": "focused",
            "success": False,
            "error_count": 2,
        },
    )
    assert stop_result.status == "success"

    session = engine.session_cache[session_id]
    agent_execution = next(
        a for a in session.agents_executed if a.agent_name == AGENT_NAME
    )
    assert agent_execution.status == ExecutionStatus.ERROR, (
        "AgentExecution.status did not transition from RUNNING to ERROR "
        "on agent_stop with success=False. This is issue #39."
    )
    assert agent_execution.completed is not None

    last_step = agent_execution.execution_steps[-1]
    assert last_step.status == ExecutionStatus.ERROR


@pytest.mark.regression
async def test_non_agent_stop_phase_keeps_running_status(engine):
    """Phases other than agent_stop (e.g. agent_start) must continue to
    create/append RUNNING steps, unaffected by the agent_stop fix."""
    start_result = await engine.session_track_execution(
        session_id=None,
        agent_name=AGENT_NAME,
        step_data={
            "phase": "agent_start",
            "operation": "start",
            "description": "SubagentStart hook",
        },
        allow_unbound=True,
    )
    assert start_result.status == "success"
    session_id = start_result.session_id

    session = engine.session_cache[session_id]
    agent_execution = next(
        a for a in session.agents_executed if a.agent_name == AGENT_NAME
    )
    assert agent_execution.status == ExecutionStatus.RUNNING
    assert agent_execution.completed is None

    last_step = agent_execution.execution_steps[-1]
    assert last_step.status == ExecutionStatus.RUNNING
    assert last_step.completed is None
