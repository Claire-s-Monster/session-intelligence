"""
Tests for session-intelligence engine agent operation methods.

Tests all 11 agent-related methods on SessionIntelligenceEngine:
  agent_register, agent_get_info, agent_log_decision, agent_query_decisions,
  agent_update_decision_outcome, agent_log_learning, agent_query_learnings,
  agent_update_learning_outcome, agent_create_notebook, agent_query_notebooks,
  agent_search_all.

Uses the shared ``engine`` fixture from conftest.py which provides a real
SQLite-backed engine (one fresh DB per test).

asyncio_mode = "auto" is set project-wide — do NOT add @pytest.mark.asyncio.
"""

import pytest


# ============================================================================
# Helper: register an agent and return its registration result
# ============================================================================


async def _register(engine, name="test-agent", agent_type="domain"):
    return await engine.agent_register(
        agent_name=name,
        agent_type=agent_type,
        display_name="Test Agent",
        description="Agent used in unit tests",
        capabilities=["testing"],
    )


# ============================================================================
# Registration tests
# ============================================================================


async def test_register_new_agent(engine):
    """Registering a fresh agent returns status='created' and a non-empty id."""
    result = await _register(engine)

    assert result.status == "created"
    assert result.agent_id != ""
    assert result.name == "test-agent"


async def test_register_existing_agent_updates(engine):
    """Registering the same agent twice returns status='updated' on second call."""
    first = await _register(engine)
    second = await _register(engine)

    assert first.status == "created"
    assert second.status == "updated"
    # The UUID must be stable across updates
    assert first.agent_id == second.agent_id


# ============================================================================
# Get info tests
# ============================================================================


async def test_get_agent_info_by_name(engine):
    """agent_get_info returns an Agent model with expected fields when queried by name."""
    reg = await _register(engine)

    agent = await engine.agent_get_info("test-agent")

    assert agent is not None
    assert agent.id == reg.agent_id
    assert agent.name == "test-agent"
    assert agent.agent_type == "domain"
    assert agent.display_name == "Test Agent"
    assert agent.description == "Agent used in unit tests"
    assert "testing" in agent.capabilities
    assert agent.is_active is True


async def test_get_agent_info_by_id(engine):
    """agent_get_info resolves a UUID and returns the correct agent."""
    reg = await _register(engine)

    agent = await engine.agent_get_info(reg.agent_id)

    assert agent is not None
    assert agent.name == "test-agent"
    assert agent.id == reg.agent_id


async def test_get_nonexistent_agent(engine):
    """agent_get_info returns None when the agent does not exist."""
    result = await engine.agent_get_info("no-such-agent")
    assert result is None


# ============================================================================
# Decision tests
# ============================================================================


async def test_agent_log_decision(engine):
    """Logging a decision returns status='success' and a non-empty decision_id."""
    await _register(engine)

    result = await engine.agent_log_decision(
        agent_name="test-agent",
        decision_type="implementation",
        context="Choosing between option A and option B",
        decision="Chose option A",
        reasoning="Option A is simpler",
        alternatives=["option B", "option C"],
        confidence=0.9,
        tags=["simplicity"],
    )

    assert result.status == "success"
    assert result.decision_id != ""
    assert result.agent_id != ""


async def test_agent_log_decision_unknown_agent(engine):
    """Logging a decision for an unregistered agent returns status='error'."""
    result = await engine.agent_log_decision(
        agent_name="ghost-agent",
        decision_type="architecture",
        context="ctx",
        decision="do something",
    )
    assert result.status == "error"


async def test_agent_query_decisions(engine):
    """Querying decisions returns the logged decision for the correct agent."""
    await _register(engine)
    log_result = await engine.agent_log_decision(
        agent_name="test-agent",
        decision_type="pattern",
        context="Deciding on design pattern",
        decision="Use factory pattern",
    )

    decisions = await engine.agent_query_decisions("test-agent")

    assert len(decisions) >= 1
    ids = [d.id for d in decisions]
    assert log_result.decision_id in ids


async def test_agent_query_decisions_filtered_by_type(engine):
    """Decision query respects decision_type filter."""
    await _register(engine)
    await engine.agent_log_decision(
        agent_name="test-agent",
        decision_type="architecture",
        context="ctx",
        decision="use hexagonal",
    )
    await engine.agent_log_decision(
        agent_name="test-agent",
        decision_type="implementation",
        context="ctx",
        decision="use SQLite",
    )

    arch_decisions = await engine.agent_query_decisions(
        "test-agent", decision_type="architecture"
    )

    assert len(arch_decisions) >= 1
    for d in arch_decisions:
        assert d.decision_type == "architecture"


async def test_agent_update_decision_outcome(engine):
    """Updating a decision outcome returns status='success' and reflects the decision_id."""
    await _register(engine)
    log_result = await engine.agent_log_decision(
        agent_name="test-agent",
        decision_type="implementation",
        context="ctx",
        decision="pick library X",
    )

    update_result = await engine.agent_update_decision_outcome(
        decision_id=log_result.decision_id,
        outcome="Library X worked perfectly",
        success=True,
    )

    assert update_result["status"] == "success"
    assert update_result["decision_id"] == log_result.decision_id
    assert update_result["success"] is True


# ============================================================================
# Learning tests
# ============================================================================


async def test_agent_log_learning(engine):
    """Logging a learning returns status='success' and a non-empty learning_id."""
    await _register(engine)

    result = await engine.agent_log_learning(
        agent_name="test-agent",
        learning_type="pattern",
        title="Use context managers",
        content="Always use context managers for resource cleanup",
        source_context="Observed resource leaks without them",
        applicability=["file handling", "db connections"],
        confidence=0.95,
        tags=["python", "best-practice"],
    )

    assert result.status == "success"
    assert result.learning_id != ""
    assert result.agent_id != ""


async def test_agent_log_learning_unknown_agent(engine):
    """Logging a learning for an unregistered agent returns status='error'."""
    result = await engine.agent_log_learning(
        agent_name="ghost-agent",
        learning_type="pattern",
        title="title",
        content="content",
    )
    assert result.status == "error"


async def test_agent_query_learnings(engine):
    """Querying learnings returns the logged learning for the agent."""
    await _register(engine)
    log_result = await engine.agent_log_learning(
        agent_name="test-agent",
        learning_type="technique",
        title="Mock async calls",
        content="Use AsyncMock for awaitable methods in tests",
    )

    learnings = await engine.agent_query_learnings("test-agent")

    assert len(learnings) >= 1
    ids = [l.id for l in learnings]
    assert log_result.learning_id in ids


async def test_agent_update_learning_outcome(engine):
    """Updating learning outcome returns status='success'."""
    await _register(engine)
    log_result = await engine.agent_log_learning(
        agent_name="test-agent",
        learning_type="anti-pattern",
        title="Avoid bare except",
        content="Never use bare except; always specify exception types",
    )

    update_result = await engine.agent_update_learning_outcome(
        learning_id=log_result.learning_id,
        times_applied_increment=2,
        new_success_rate=0.9,
    )

    assert update_result["status"] == "success"
    assert update_result["learning_id"] == log_result.learning_id
    assert update_result["times_applied_increment"] == 2
    assert update_result["success"] is True


# ============================================================================
# Notebook tests
# ============================================================================


async def test_agent_create_notebook(engine):
    """Creating a notebook returns status='success' and a non-empty notebook_id."""
    await _register(engine)

    result = await engine.agent_create_notebook(
        agent_name="test-agent",
        title="Execution Summary: task-42",
        content="## Summary\n\nCompleted task-42 successfully.",
        summary="Task 42 done",
        notebook_type="execution",
        tags=["task-42"],
    )

    assert result.status == "success"
    assert result.notebook_id != ""
    assert result.title == "Execution Summary: task-42"
    assert result.agent_id != ""


async def test_agent_create_notebook_unknown_agent(engine):
    """Creating a notebook for an unregistered agent returns status='error'."""
    result = await engine.agent_create_notebook(
        agent_name="ghost-agent",
        title="Some notebook",
        content="content",
    )
    assert result.status == "error"


async def test_agent_query_notebooks(engine):
    """Querying notebooks returns the created notebook for the agent."""
    await _register(engine)
    create_result = await engine.agent_create_notebook(
        agent_name="test-agent",
        title="Research: DB indexes",
        content="## Research\n\nIndexes speed up queries.",
        notebook_type="research",
    )

    notebooks = await engine.agent_query_notebooks("test-agent")

    assert len(notebooks) >= 1
    ids = [n.id for n in notebooks]
    assert create_result.notebook_id in ids


# ============================================================================
# Search all test
# ============================================================================


async def test_agent_search_all(engine):
    """search_all returns matching decisions, learnings, and notebooks."""
    await _register(engine)

    # Log content with a distinctive term
    await engine.agent_log_decision(
        agent_name="test-agent",
        decision_type="implementation",
        context="searching for xyzzy patterns",
        decision="use xyzzy library",
    )
    await engine.agent_log_learning(
        agent_name="test-agent",
        learning_type="pattern",
        title="xyzzy technique",
        content="The xyzzy technique reduces boilerplate",
    )
    await engine.agent_create_notebook(
        agent_name="test-agent",
        title="xyzzy notebook",
        content="## xyzzy\n\nNotes about xyzzy.",
    )

    result = await engine.agent_search_all("test-agent", query="xyzzy")

    assert "decisions" in result
    assert "learnings" in result
    assert "notebooks" in result
    assert result["total_matches"] >= 1
