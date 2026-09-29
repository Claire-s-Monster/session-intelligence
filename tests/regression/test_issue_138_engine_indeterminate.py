"""
Regression tests for issue #138 (engine side): agent_stop step_data with
success=None (or no 'success' key at all -- both indistinguishable via
`.get("success")`) must resolve to a tri-state INDETERMINATE status, not
be coerced into ExecutionStatus.ERROR the way a falsy `success` currently
is.

https://github.com/Claire-s-Monster/session-intelligence/issues/138

Current code (src/core/session_engine.py, ~line 1449-1452):

    terminal_status = (
        ExecutionStatus.SUCCESS if step_data.get("success") else ExecutionStatus.ERROR
    )

`step_data.get("success")` is falsy for both `success=False` (a real
failure) AND `success=None`/missing (the transcript parser could not
determine an outcome, per issue #138's `summarize_transcript`). These
must NOT be conflated: an indeterminate transcript is not evidence of
failure.

Verifies the fix:
  - `ExecutionStatus` gains a new INDETERMINATE member
    (src/models/session_models.py).
  - agent_stop with `success=None` -> ExecutionStatus.INDETERMINATE.
  - agent_stop with no 'success' key at all -> ExecutionStatus.INDETERMINATE.
  - agent_stop with `success=True` -> ExecutionStatus.SUCCESS (regression
    guard: existing behavior must not change).
  - agent_stop with `success=False` -> ExecutionStatus.ERROR (regression
    guard: existing behavior must not change).
  - end-to-end: once tools_used/tool_count actually arrive in step_data
    (as they will once the issue #138 transcript-parsing fix is wired
    into the SubagentStop hook), `_describe_step` renders them into the
    stored step description exactly as issue #108 designed it to.

`ExecutionStatus.INDETERMINATE` does not exist yet, so tests 1-2 below
fail (AttributeError) until the fix lands. That failure is expected.
"""

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

AGENT_TYPE = "focused-code-modifier"
NOTEBOOK_SESSION_ID = "3f1c9e2a-6b4d-4a7e-9c0d-1e2f3a4b5c6d"


def _make_agent_execution(
    agent_name: str,
    status: ExecutionStatus,
    started: datetime,
    completed: datetime | None = None,
) -> AgentExecution:
    """Build a minimal AgentExecution for direct notebook-rendering tests."""
    return AgentExecution(
        agent_name=agent_name,
        agent_type=AGENT_TYPE,
        execution_id=f"{agent_name}-exec",
        started=started,
        completed=completed,
        status=status,
        execution_steps=[],
        context=AgentContext(
            session_id=NOTEBOOK_SESSION_ID,
            project_path="",
            working_directory="",
        ),
    )


def _make_session(agents_executed: list[AgentExecution]) -> Session:
    """Build a minimal Session wrapping the given agent executions."""
    return Session(
        id=NOTEBOOK_SESSION_ID,
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
    """SessionIntelligenceEngine wired to a fresh in-process SQLite backend.

    Filesystem persistence is OFF - in-memory AgentExecution/ExecutionStep
    status assertions are the focus of the issue #138 fix verification.
    """
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issue138.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


# ---------------------------------------------------------------------------
# 12. success=None -> INDETERMINATE
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_agent_stop_success_none_is_indeterminate(engine):
    """An agent_stop reporting success=None (the transcript parser could
    not determine an outcome) must transition into INDETERMINATE, not the
    ERROR that `.get("success")` falsy-coercion currently produces."""
    result = await engine.session_track_execution(
        session_id=None,
        agent_name="indeterminate-none-agent",
        step_data={
            "phase": "agent_stop",
            "agent_type": AGENT_TYPE,
            "success": None,
        },
        allow_unbound=True,
    )
    assert result.status == "success"

    session = engine.session_cache[result.session_id]
    agent_execution = next(
        a for a in session.agents_executed if a.agent_name == "indeterminate-none-agent"
    )
    last_step = agent_execution.execution_steps[-1]

    assert last_step.status == ExecutionStatus.INDETERMINATE
    assert agent_execution.status == ExecutionStatus.INDETERMINATE


# ---------------------------------------------------------------------------
# 13. No 'success' key at all -> INDETERMINATE
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_agent_stop_missing_success_key_is_indeterminate(engine):
    """An agent_stop with no 'success' key at all is indistinguishable
    from success=None via .get() and must resolve the same way:
    INDETERMINATE, not ERROR."""
    result = await engine.session_track_execution(
        session_id=None,
        agent_name="indeterminate-missing-agent",
        step_data={
            "phase": "agent_stop",
            "agent_type": AGENT_TYPE,
        },
        allow_unbound=True,
    )
    assert result.status == "success"

    session = engine.session_cache[result.session_id]
    agent_execution = next(
        a for a in session.agents_executed if a.agent_name == "indeterminate-missing-agent"
    )
    last_step = agent_execution.execution_steps[-1]

    assert last_step.status == ExecutionStatus.INDETERMINATE
    assert agent_execution.status == ExecutionStatus.INDETERMINATE


# ---------------------------------------------------------------------------
# 14/15. Regression guards: True/False behavior must not change
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_agent_stop_success_true_still_success(engine):
    """Regression guard: success=True must still resolve to SUCCESS,
    unaffected by the new tri-state handling."""
    result = await engine.session_track_execution(
        session_id=None,
        agent_name="true-agent",
        step_data={
            "phase": "agent_stop",
            "agent_type": AGENT_TYPE,
            "success": True,
        },
        allow_unbound=True,
    )
    assert result.status == "success"

    session = engine.session_cache[result.session_id]
    agent_execution = next(a for a in session.agents_executed if a.agent_name == "true-agent")

    assert agent_execution.status == ExecutionStatus.SUCCESS
    assert agent_execution.execution_steps[-1].status == ExecutionStatus.SUCCESS


@pytest.mark.regression
async def test_agent_stop_success_false_still_error(engine):
    """Regression guard: success=False (an actual observed failure, not
    an indeterminate parse) must still resolve to ERROR."""
    result = await engine.session_track_execution(
        session_id=None,
        agent_name="false-agent",
        step_data={
            "phase": "agent_stop",
            "agent_type": AGENT_TYPE,
            "success": False,
        },
        allow_unbound=True,
    )
    assert result.status == "success"

    session = engine.session_cache[result.session_id]
    agent_execution = next(a for a in session.agents_executed if a.agent_name == "false-agent")

    assert agent_execution.status == ExecutionStatus.ERROR
    assert agent_execution.execution_steps[-1].status == ExecutionStatus.ERROR


# ---------------------------------------------------------------------------
# 16. End-to-end: tools_used/tool_count render into the step description
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_tools_used_renders_into_step_description(engine):
    """Once tools_used/tool_count actually arrive in step_data (as they
    will once the #138 transcript-parsing fix feeds real data into the
    SubagentStop hook payload), `_describe_step` (session_engine.py:137-166,
    from issue #108) must render them into the stored step description."""
    result = await engine.session_track_execution(
        session_id=None,
        agent_name="tools-used-agent",
        step_data={
            "phase": "agent_stop",
            "agent_type": AGENT_TYPE,
            "success": True,
            "tools_used": ["Read", "Grep"],
            "tool_count": 2,
        },
        allow_unbound=True,
    )
    assert result.status == "success"

    session = engine.session_cache[result.session_id]
    agent_execution = next(a for a in session.agents_executed if a.agent_name == "tools-used-agent")
    last_step = agent_execution.execution_steps[-1]

    assert last_step.description == "2 tools: Read, Grep"


# ---------------------------------------------------------------------------
# 17. Notebook agents section: INDETERMINATE renders as U+2754, not U+274C
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_agents_section_renders_indeterminate_emoji_not_failure(engine):
    """An INDETERMINATE agent execution must render with the explicit "❔"
    marker in the notebook's agents section, and must NOT render "❌" --
    the final-else fallback would otherwise make an indeterminate outcome
    visually identical to a real failure."""
    started = datetime.now(UTC)
    session = _make_session(
        [_make_agent_execution("agent-indeterminate", ExecutionStatus.INDETERMINATE, started)]
    )

    rendered, agents_used = engine._generate_agents_section(session)

    assert agents_used == ["agent-indeterminate"]
    assert "❔" in rendered
    assert "❌" not in rendered


# ---------------------------------------------------------------------------
# 18. Metrics table: Successful + Failed + Indeterminate == Agents Executed
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_metrics_section_three_buckets_sum_to_agents_executed(engine):
    """For a session mixing SUCCESS + ERROR + INDETERMINATE executions, the
    rendered notebook's "Successful" + "Failed" + "Indeterminate" row values
    must sum to the "Agents Executed" row value."""
    started = datetime.now(UTC)
    session = _make_session(
        [
            _make_agent_execution("agent-success-1", ExecutionStatus.SUCCESS, started, started),
            _make_agent_execution("agent-success-2", ExecutionStatus.SUCCESS, started, started),
            _make_agent_execution("agent-error-1", ExecutionStatus.ERROR, started, started),
            _make_agent_execution(
                "agent-indeterminate-1", ExecutionStatus.INDETERMINATE, started, started
            ),
        ]
    )
    engine._recompute_derived_metrics(session)

    rendered = engine._generate_metrics_section(session)

    assert "| Agents Executed | 4 |" in rendered
    assert "| Successful Executions | 2 |" in rendered
    assert "| Failed Executions | 1 |" in rendered
    assert "| Indeterminate Executions | 1 |" in rendered
