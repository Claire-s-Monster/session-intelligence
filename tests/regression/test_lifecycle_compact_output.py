"""
Regression tests: session_manage_lifecycle create(adopt)/finalize/validate
return a compact summary by default (full Session only with
include_details=True), and session_get_steps pages through execution steps.

SQLite is exercised for real; no PostgreSQL needed.
"""

import json
from datetime import UTC, datetime

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import AgentContext, AgentExecution, ExecutionStep
from persistence.sqlite import SQLiteBackend

SESSION_ID = "compact-session-1"
PROJECT = "proj-compact"
AGENTS = 12
STEPS_PER_AGENT = 3


@pytest.fixture
async def engine(tmp_path):
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path), use_filesystem=False)
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


async def _seed(engine):
    created = await engine.session_manage_lifecycle(
        "create", project_name=PROJECT, session_id=SESSION_ID
    )
    assert created.status == "success"
    session = engine.session_cache[SESSION_ID]
    now = datetime.now(UTC)
    ctx = AgentContext(session_id=SESSION_ID, project_path="/p", working_directory="/p")
    for a in range(AGENTS):
        steps = [
            ExecutionStep(
                step_id=f"s-{a}-{i}",
                step_number=i,
                agent=f"agent-{a}",
                operation="op",
                description="d" * 1000,
                started=now,
            )
            for i in range(STEPS_PER_AGENT)
        ]
        session.agents_executed.append(
            AgentExecution(
                agent_name=f"agent-{a}",
                agent_type="focused",
                execution_id=f"exec-{a}",
                started=now,
                execution_steps=steps,
                context=ctx,
            )
        )
    return session


def _assert_no_key(obj, key):
    text = json.dumps(obj, default=str)
    assert f'"{key}"' not in text


@pytest.mark.regression
class TestCompactLifecycle:
    async def test_validate_is_compact_by_default(self, engine):
        await _seed(engine)

        result = await engine.session_manage_lifecycle("validate", session_id=SESSION_ID)

        data = result.session_data
        assert isinstance(data, dict)
        _assert_no_key(data, "execution_steps")
        _assert_no_key(data, "commands_executed")
        assert data["id"] == SESSION_ID
        assert data["project_name"] == PROJECT
        assert data["counts"]["agents_executed"] == AGENTS
        assert data["counts"]["total_steps"] == AGENTS * STEPS_PER_AGENT
        assert data["counts"]["errors"] == 0
        assert len(data["recent_agents"]) == 10
        assert data["recent_agents"][-1]["agent_name"] == f"agent-{AGENTS - 1}"
        assert data["recent_agents"][0]["step_count"] == STEPS_PER_AGENT
        assert result.status == "success"

    async def test_finalize_is_compact_by_default(self, engine):
        await _seed(engine)

        result = await engine.session_manage_lifecycle("finalize", session_id=SESSION_ID)

        data = result.session_data
        assert isinstance(data, dict)
        _assert_no_key(data, "execution_steps")
        assert data["status"] == "completed"
        assert data["completed_at"] is not None
        assert data["counts"]["agents_executed"] == AGENTS

    async def test_adopt_is_compact_by_default(self, engine):
        await _seed(engine)

        result = await engine.session_manage_lifecycle(
            "create", project_name=PROJECT, session_id=SESSION_ID
        )

        assert isinstance(result.session_data, dict)
        _assert_no_key(result.session_data, "execution_steps")
        assert result.session_data["counts"]["agents_executed"] == AGENTS

    async def test_include_details_returns_full_session(self, engine):
        await _seed(engine)

        validated = await engine.session_manage_lifecycle(
            "validate", session_id=SESSION_ID, include_details=True
        )
        adopted = await engine.session_manage_lifecycle(
            "create", project_name=PROJECT, session_id=SESSION_ID, include_details=True
        )
        finalized = await engine.session_manage_lifecycle(
            "finalize", session_id=SESSION_ID, include_details=True
        )

        for result in (validated, adopted, finalized):
            assert not isinstance(result.session_data, dict)
            assert len(result.session_data.agents_executed) == AGENTS
            first_agent = result.session_data.agents_executed[0]
            assert len(first_agent.execution_steps) == STEPS_PER_AGENT


@pytest.mark.regression
class TestSessionGetSteps:
    async def test_pagination(self, engine):
        await _seed(engine)
        total = AGENTS * STEPS_PER_AGENT

        first = await engine.session_get_steps(SESSION_ID, limit=10)
        assert first["total"] == total
        assert first["offset"] == 0
        assert first["limit"] == 10
        assert first["has_more"] is True
        assert len(first["steps"]) == 10
        assert first["steps"][0]["agent_name"] == "agent-0"
        assert first["steps"][0]["agent_execution_id"] == "exec-0"
        assert len(first["steps"][0]["description"]) == 500

        last = await engine.session_get_steps(SESSION_ID, limit=10, offset=30)
        assert len(last["steps"]) == total - 30
        assert last["has_more"] is False

    async def test_agent_name_filter(self, engine):
        await _seed(engine)

        result = await engine.session_get_steps(SESSION_ID, agent_name="agent-3")

        assert result["total"] == STEPS_PER_AGENT
        assert {s["agent_name"] for s in result["steps"]} == {"agent-3"}
        assert result["has_more"] is False

    async def test_limit_clamped(self, engine):
        await _seed(engine)

        high = await engine.session_get_steps(SESSION_ID, limit=10_000)
        low = await engine.session_get_steps(SESSION_ID, limit=0, offset=-5)

        assert high["limit"] == 200
        assert low["limit"] == 1
        assert low["offset"] == 0

    async def test_unknown_session_raises(self, engine):
        with pytest.raises(ValueError, match="not found"):
            await engine.session_get_steps("nope")
