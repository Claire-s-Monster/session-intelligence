"""
Regression: step_id must be unique even for steps of one agent created within
the same second (it used to be agent + second-resolution local timestamp).
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend

SESSION_ID = "step-id-session"
AGENT = "bash-executor"


@pytest.fixture
async def engine(tmp_path):
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path), use_filesystem=False)
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


@pytest.mark.regression
class TestStepIdUnique:
    async def test_same_second_steps_get_distinct_ids(self, engine):
        await engine.session_manage_lifecycle(
            "create", project_name="proj-stepid", session_id=SESSION_ID
        )
        results = [
            await engine.session_track_execution(
                SESSION_ID,
                AGENT,
                {"phase": "post_tool_use", "command": f"git cmd{i}", "agent_type": "internal"},
            )
            for i in range(5)
        ]
        assert all(not r.status.startswith("error") for r in results)

        session = engine.session_cache[SESSION_ID]
        execs = [e for e in session.agents_executed if e.agent_name == AGENT]
        assert len(execs) == 1
        ids = [s.step_id for s in execs[0].execution_steps]
        assert len(ids) == 5
        assert len(set(ids)) == 5
        assert all(i.startswith(f"{AGENT}-") for i in ids)
