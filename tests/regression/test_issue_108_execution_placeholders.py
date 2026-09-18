"""
Regression tests for issue #108: execution tracking records placeholder
operation/description ("unknown"/"") and spurious "unknown" AgentExecution
rows from Claude Code's background-task progress summarizer.

https://github.com/Claire-s-Monster/session-intelligence/issues/108

Root cause:
  1. Hooks never send `operation`/`description` in `step_data`; they send
     `phase` ("agent_start", "agent_stop", "task_created", "task_updated",
     "command_success") plus raw fields (`command`/`command_base`,
     `task_subject`/`task_description`, `task_id`/`new_status`,
     `tools_used`/`tool_count`). `_track_execution_sync` derived
     `operation`/`description` solely via `.get(key, default)` against
     keys that were never present, always falling back to the default.
  2. Claude Code's background-task progress summarizer fires a
     SubagentStop hook (`phase="agent_stop"`) roughly every 32s with
     `agent_type=""` and NO matching SubagentStart. These synthesized
     one-step "unknown" AgentExecution rows accounted for 68% of all
     execution rows.

Verifies the fix:
  - `operation` prefers explicit `step_data["operation"]`, then
    `step_data["phase"]`, then "unknown".
  - `description` prefers explicit `step_data["description"]`, else is
    derived from hook fields via `_describe_step()`.
  - `INTERNAL_AGENT_NAMES` ("task-manager", "bash-executor") resolve to
    agent_type "internal" instead of "unknown" when no real agent_type is
    supplied.
  - A typeless `agent_stop` (empty/"unknown" agent_type) with no cached
    agent_type AND no RUNNING execution for that agent_name in the
    session is ignored outright (`status="ignored-no-start"`), with no
    session mutation and no session auto-creation, *before* any other
    side effect. A stop with a real agent_type, or one whose start WAS
    observed, is still recorded exactly as before.
"""

import uuid

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import ExecutionStatus
from persistence.sqlite import SQLiteBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    """SessionIntelligenceEngine wired to a fresh in-process SQLite backend.

    Filesystem persistence is OFF - in-memory AgentExecution/ExecutionStep
    assertions are the focus of the issue #108 fix verification.
    """
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issue108.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


# ---------------------------------------------------------------------------
# Defect A: operation/description derivation
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_agent_start_with_only_phase_sets_operation_to_phase(engine):
    """A hook call reporting only `phase` (no `operation`) must use the
    phase as the operation, not the "unknown" placeholder."""
    result = await engine.session_track_execution(
        session_id=None,
        agent_name="phase-only-agent",
        step_data={"phase": "agent_start"},
        allow_unbound=True,
    )
    assert result.status == "success"

    session = engine.session_cache[result.session_id]
    agent_execution = next(a for a in session.agents_executed if a.agent_name == "phase-only-agent")
    last_step = agent_execution.execution_steps[-1]
    assert last_step.operation == "agent_start"


@pytest.mark.regression
async def test_explicit_operation_and_description_win_over_derived(engine):
    """When `operation`/`description` are present in step_data, they must
    be used verbatim, not overridden by phase/derived values."""
    result = await engine.session_track_execution(
        session_id=None,
        agent_name="explicit-agent",
        step_data={
            "phase": "agent_start",
            "operation": "custom-op",
            "description": "custom description",
            "command": "should-be-ignored",
        },
        allow_unbound=True,
    )
    assert result.status == "success"

    session = engine.session_cache[result.session_id]
    agent_execution = next(a for a in session.agents_executed if a.agent_name == "explicit-agent")
    last_step = agent_execution.execution_steps[-1]
    assert last_step.operation == "custom-op"
    assert last_step.description == "custom description"


@pytest.mark.regression
async def test_bash_executor_command_success_derives_description_and_internal_type(engine):
    """bash-executor's `command_success` phase must derive its description
    from `command`, and resolve agent_type to "internal"."""
    result = await engine.session_track_execution(
        session_id=None,
        agent_name="bash-executor",
        step_data={"phase": "command_success", "command": "pytest tests/regression"},
        allow_unbound=True,
    )
    assert result.status == "success"

    session = engine.session_cache[result.session_id]
    agent_execution = next(a for a in session.agents_executed if a.agent_name == "bash-executor")
    assert agent_execution.agent_type == "internal"

    last_step = agent_execution.execution_steps[-1]
    assert last_step.operation == "command_success"
    assert "pytest tests/regression" in last_step.description


@pytest.mark.regression
async def test_task_manager_task_created_derives_description_and_internal_type(engine):
    """task-manager's `task_created` phase must derive its description
    from `task_subject`, and resolve agent_type to "internal"."""
    result = await engine.session_track_execution(
        session_id=None,
        agent_name="task-manager",
        step_data={"phase": "task_created", "task_subject": "Fix issue #108"},
        allow_unbound=True,
    )
    assert result.status == "success"

    session = engine.session_cache[result.session_id]
    agent_execution = next(a for a in session.agents_executed if a.agent_name == "task-manager")
    assert agent_execution.agent_type == "internal"

    last_step = agent_execution.execution_steps[-1]
    assert "Fix issue #108" in last_step.description


# ---------------------------------------------------------------------------
# Defect B: ignore typeless agent_stop with no matching start
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_typeless_agent_stop_with_no_start_is_ignored(engine):
    """A typeless agent_stop for an agent_name never seen before must be
    ignored: no session mutation for a cached session, and no session
    auto-creation for an uncached session_id."""
    # Cached session with an unrelated, already-running agent: the ignored
    # stop for a *different*, never-seen agent_name must not touch
    # agents_executed at all.
    start_result = await engine.session_track_execution(
        session_id=None,
        agent_name="unrelated-agent",
        step_data={"phase": "agent_start", "agent_type": "real-type"},
        allow_unbound=True,
    )
    session_id = start_result.session_id
    session = engine.session_cache[session_id]
    agents_before = len(session.agents_executed)

    stop_result = await engine.session_track_execution(
        session_id=session_id,
        agent_name="ghost-agent",
        step_data={"phase": "agent_stop", "agent_type": "", "success": True},
    )
    assert stop_result.status == "ignored-no-start"
    assert stop_result.step_id == "ignored"
    assert len(session.agents_executed) == agents_before

    # Uncached session_id: must not auto-create a session.
    fake_session_id = str(uuid.uuid4())
    assert fake_session_id not in engine.session_cache

    result2 = await engine.session_track_execution(
        session_id=fake_session_id,
        agent_name="ghost-agent-2",
        step_data={"phase": "agent_stop", "agent_type": "unknown", "success": True},
    )
    assert result2.status == "ignored-no-start"
    assert fake_session_id not in engine.session_cache


@pytest.mark.regression
async def test_agent_stop_with_real_type_and_no_start_is_recorded(engine):
    """A stop reporting a real agent_type (not empty/"unknown") must
    always be recorded, even with no prior start observed."""
    result = await engine.session_track_execution(
        session_id=None,
        agent_name="typed-ghost",
        step_data={
            "phase": "agent_stop",
            "agent_type": "focused-code-modifier",
            "success": True,
        },
        allow_unbound=True,
    )
    assert result.status == "success"

    session = engine.session_cache[result.session_id]
    executions = [a for a in session.agents_executed if a.agent_name == "typed-ghost"]
    assert len(executions) == 1
    assert executions[0].agent_type == "focused-code-modifier"
    assert executions[0].status == ExecutionStatus.SUCCESS


@pytest.mark.regression
async def test_typed_start_then_typeless_stop_terminates_single_execution(engine):
    """A typed start followed by a typeless stop (cache hit) must still
    resolve and terminate that single execution, unaffected by the
    new ignore guard."""
    start_result = await engine.session_track_execution(
        session_id=None,
        agent_name="typed-agent",
        step_data={"phase": "agent_start", "agent_type": "focused-quality-resolver"},
        allow_unbound=True,
    )
    session_id = start_result.session_id

    stop_result = await engine.session_track_execution(
        session_id=session_id,
        agent_name="typed-agent",
        step_data={"phase": "agent_stop", "agent_type": "", "success": True},
    )
    assert stop_result.status == "success"

    session = engine.session_cache[session_id]
    executions = [a for a in session.agents_executed if a.agent_name == "typed-agent"]
    assert len(executions) == 1
    assert executions[0].agent_type == "focused-quality-resolver"
    assert executions[0].status == ExecutionStatus.SUCCESS


@pytest.mark.regression
async def test_typeless_stop_after_cache_clear_still_terminates_running_execution(engine):
    """Simulated restart: `_agent_type_cache` is wiped but a RUNNING
    execution for the agent_name still exists in the session. The
    typeless stop must NOT be ignored -- it must terminate that
    execution, since the RUNNING-execution fallback check covers exactly
    this case."""
    start_result = await engine.session_track_execution(
        session_id=None,
        agent_name="restart-agent",
        step_data={"phase": "agent_start", "agent_type": "focused-code-modifier"},
        allow_unbound=True,
    )
    session_id = start_result.session_id

    # Simulate a server restart: in-memory type cache wiped, RUNNING
    # execution persists (it lives on the session, not the cache).
    engine._agent_type_cache.clear()

    stop_result = await engine.session_track_execution(
        session_id=session_id,
        agent_name="restart-agent",
        step_data={"phase": "agent_stop", "agent_type": "", "success": True},
    )
    assert stop_result.status == "success"

    session = engine.session_cache[session_id]
    execution = next(a for a in session.agents_executed if a.agent_name == "restart-agent")
    assert execution.status == ExecutionStatus.SUCCESS
    assert execution.completed is not None


# ---------------------------------------------------------------------------
# Issue #115 (cause 3): commands_executed/duration_ms never populated
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_commands_executed_and_duration_ms_are_populated_from_step_data(engine):
    """`commands_executed`/`duration_ms` in step_data must populate the
    ExecutionStep fields of the same name, not be silently dropped."""
    result = await engine.session_track_execution(
        session_id=None,
        agent_name="metrics-agent",
        step_data={
            "phase": "command_success",
            "commands_executed": ["pytest tests/", "ruff check ."],
            "duration_ms": 4200,
        },
        allow_unbound=True,
    )
    assert result.status == "success"

    session = engine.session_cache[result.session_id]
    agent_execution = next(a for a in session.agents_executed if a.agent_name == "metrics-agent")
    last_step = agent_execution.execution_steps[-1]

    assert [c.command for c in last_step.commands_executed] == [
        "pytest tests/",
        "ruff check .",
    ]
    assert last_step.duration_ms == 4200


@pytest.mark.regression
async def test_singular_command_becomes_one_element_commands_executed_list(engine):
    """The hook's documented singular `command` field must be normalized
    into a one-element `commands_executed` list, absent the plural key."""
    result = await engine.session_track_execution(
        session_id=None,
        agent_name="single-command-agent",
        step_data={"phase": "start", "command": "pytest"},
        allow_unbound=True,
    )
    assert result.status == "success"

    session = engine.session_cache[result.session_id]
    agent_execution = next(
        a for a in session.agents_executed if a.agent_name == "single-command-agent"
    )
    last_step = agent_execution.execution_steps[-1]

    assert len(last_step.commands_executed) == 1
    assert last_step.commands_executed[0].command == "pytest"
