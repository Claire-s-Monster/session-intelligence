"""
Integration tests for concurrent access to the session engine.

Uses asyncio.gather to run multiple simultaneous operations against a shared
SessionIntelligenceEngine backed by SQLite.

asyncio_mode = "auto" — no @pytest.mark.asyncio decorators needed.
"""

from __future__ import annotations

import asyncio

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    """Initialised SQLite backend."""
    backend = SQLiteBackend(str(tmp_path / "test_concurrency.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def engine(tmp_path, db):
    """Engine backed by the test SQLite database."""
    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_session(engine: SessionIntelligenceEngine, name: str):
    """Create a session and return its SessionResult."""
    return await engine.session_manage_lifecycle(
        operation="create",
        project_name=name,
    )


async def _register_agent(engine: SessionIntelligenceEngine, name: str):
    """Register an agent and return its AgentRegistrationResult."""
    return await engine.agent_register(
        agent_name=name,
        agent_type="domain",
        display_name=f"Agent {name}",
    )


# ===========================================================================
# Concurrent session creates
# ===========================================================================


async def test_concurrent_session_creates_all_succeed(engine):
    """Five concurrent session_manage_lifecycle(create) calls all return status=success."""
    results = await asyncio.gather(
        *[_create_session(engine, f"project-{i}") for i in range(5)]
    )
    for result in results:
        assert result.status == "success", f"Unexpected status: {result.status}"


async def test_concurrent_session_creates_all_have_session_ids(engine):
    """Concurrent creates all return a non-empty session_id."""
    results = await asyncio.gather(
        *[_create_session(engine, f"project-{i}") for i in range(5)]
    )
    for result in results:
        assert result.session_id, "session_id must be non-empty"


# ===========================================================================
# Concurrent agent registration
# ===========================================================================


async def test_concurrent_agent_registrations_all_succeed(engine):
    """Five simultaneous agent_register calls all produce a result."""
    results = await asyncio.gather(
        *[_register_agent(engine, f"agent-concurrent-{i}") for i in range(5)]
    )
    for result in results:
        assert result is not None
        assert result.agent_id


async def test_concurrent_agent_registrations_same_name_is_idempotent(engine):
    """Registering the same agent name concurrently does not crash."""
    results = await asyncio.gather(
        *[_register_agent(engine, "shared-agent-name") for _ in range(4)]
    )
    # All calls should return a result (created or updated, not an exception)
    for result in results:
        assert result is not None
        assert result.agent_id


# ===========================================================================
# Concurrent decision logging
# ===========================================================================


async def test_concurrent_decision_logging(engine):
    """Multiple simultaneous session_log_decision calls all return results."""
    # Create a session first
    session_result = await _create_session(engine, "decision-test-project")
    session_id = session_result.session_id

    async def _log(i: int):
        return await engine.session_log_decision(
            decision=f"Decision number {i}",
            session_id=session_id,
            context={"index": i},
        )

    results = await asyncio.gather(*[_log(i) for i in range(6)])
    for result in results:
        assert result is not None


# ===========================================================================
# Reads during writes
# ===========================================================================


async def test_reads_concurrent_with_writes(engine):
    """Session creates and agent_query_learnings can run simultaneously."""
    async def _create():
        return await _create_session(engine, "concurrent-rw-project")

    async def _read():
        # agent_query_learnings returns a list (empty is fine)
        result = await engine.agent_query_learnings(agent_name="nobody", limit=5)
        return result

    results = await asyncio.gather(
        _create(), _read(), _create(), _read(),
    )
    # All four coroutines should complete without raising
    assert len(results) == 4
