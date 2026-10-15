"""
Regression: step_number must strictly increment per step within ONE agent
execution (it used to be len(session.agents_executed) + 1, so every step of a
long-lived execution shared the same number).
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import AgentExecution
from persistence.sqlite import SQLiteBackend

SESSION_ID = "step-number-session"
AGENT = "bash-executor"


@pytest.fixture
async def engine(tmp_path):
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path), use_filesystem=False)
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


async def _track(engine, n):
    for i in range(n):
        result = await engine.session_track_execution(
            SESSION_ID,
            AGENT,
            {"phase": "post_tool_use", "command": f"git cmd{i}", "agent_type": "internal"},
        )
        assert not result.status.startswith("error")


def _numbers(engine):
    session = engine.session_cache[SESSION_ID]
    execs = [e for e in session.agents_executed if e.agent_name == AGENT]
    assert len(execs) == 1
    return [s.step_number for s in execs[0].execution_steps]


@pytest.mark.regression
class TestStepNumberIncrement:
    async def test_steps_increment_within_execution(self, engine):
        await engine.session_manage_lifecycle(
            "create", project_name="proj-steps", session_id=SESSION_ID
        )
        # Other executions in the session must not influence numbering.
        await engine.session_track_execution(SESSION_ID, "other-agent", {"phase": "x"})

        await _track(engine, 5)

        assert _numbers(engine) == [1, 2, 3, 4, 5]

    async def test_increment_survives_rehydration(self, engine):
        await engine.session_manage_lifecycle(
            "create", project_name="proj-steps", session_id=SESSION_ID
        )
        await _track(engine, 3)

        # Simulate cache eviction: execution rebuilt from its persisted form.
        session = engine.session_cache[SESSION_ID]
        session.agents_executed = [
            AgentExecution.model_validate(e.model_dump()) for e in session.agents_executed
        ]

        await _track(engine, 3)

        numbers = _numbers(engine)
        assert numbers == [1, 2, 3, 4, 5, 6]
        assert len(set(numbers)) == len(numbers)
