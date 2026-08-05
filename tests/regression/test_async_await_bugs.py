"""
Regression tests for PR #12: fire-and-forget async calls.

Bug: async DB calls were called without await, so data was silently lost.
Fix: All async engine methods now properly await database calls.
"""

import inspect

import pytest

from core.session_engine import SessionIntelligenceEngine, UNKNOWN_PROJECT_PATH
from persistence.sqlite import SQLiteBackend


@pytest.fixture
async def engine(tmp_path):
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


@pytest.mark.regression
class TestAsyncAwaitBugs:

    async def test_session_log_decision_persists_data(self, engine):
        """Verify decision data actually reaches the database (not fire-and-forget)."""
        result = await engine.session_manage_lifecycle(operation="create", mode="local", project_name="test")
        session_id = result.session_id

        await engine.session_log_decision(
            decision="Test decision",
            context={"rationale": "Test rationale", "category": "test"},
        )

        decisions = await engine.database.query_decisions_by_session(session_id)
        assert len(decisions) >= 1
        assert decisions[0]["description"] == "Test decision"

    async def test_session_log_learning_persists_data(self, engine):
        """Verify learning data actually reaches the database."""
        await engine.session_manage_lifecycle(operation="create", mode="local", project_name="test")

        await engine.session_log_learning(
            category="pattern",
            learning_content="Test learning",
            trigger_context="Test trigger",
            allow_unbound=True,
        )

        learnings = await engine.database.query_project_learnings(
            project_path=UNKNOWN_PROJECT_PATH
        )
        assert len(learnings) >= 1

    async def test_no_coroutine_objects_from_mcp_tools(self, engine):
        """Verify all async engine tools are wrapped with _wrap_async_tool."""
        from lean_mcp_interface import LeanMCPInterface

        interface = LeanMCPInterface(engine)

        for tool_name, tool_info in interface.tool_registry.items():
            func = tool_info["implementation"]
            underlying = getattr(func, "__wrapped__", None)
            if underlying is not None and inspect.iscoroutinefunction(underlying):
                assert inspect.iscoroutinefunction(func), (
                    f"Tool '{tool_name}' wraps an async function but is not async. "
                    "This is the PR #12 bug — use _wrap_async_tool instead of _wrap_tool."
                )
