"""
Regression tests for PR #13: isoformat strings vs datetime objects.

Bug: asyncpg rejects ISO format strings for TIMESTAMPTZ columns.
     SQLite silently accepts both, masking the bug.
Fix: Always pass datetime objects, never .isoformat() strings.
"""

from datetime import UTC, datetime

import pytest

from persistence.sqlite import SQLiteBackend

from tests.persistence.builders import (
    make_mcp_session_data,
    make_session_data,
    make_agent_data,
    make_agent_decision_data,
    make_agent_learning_data,
    make_agent_notebook_data,
)


@pytest.mark.regression
class TestDatetimeTypeBugs:

    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db
        await db.close()

    async def test_session_datetime_roundtrip(self, backend):
        """Verify datetime objects survive save/retrieve without isoformat conversion."""
        now = datetime.now(UTC)
        raw = make_session_data()
        session = {**raw, "id": raw["session_id"], "started_at": now}
        await backend.save_session(session)
        result = await backend.get_session(session["id"])
        assert result is not None

    async def test_mcp_session_datetime_fields(self, backend):
        """MCP session save uses datetime objects for created_at and last_activity."""
        mcp_data = make_mcp_session_data()
        await backend.save_mcp_session(mcp_data)
        result = await backend.get_mcp_session(mcp_data["mcp_session_id"])
        assert result is not None

    async def test_agent_datetime_fields(self, backend):
        """Agent save uses datetime objects for first_seen_at and last_active_at."""
        agent = make_agent_data()
        await backend.save_agent(agent)
        result = await backend.get_agent(agent["id"])
        assert result is not None

    async def test_agent_decision_datetime(self, backend):
        """Agent decision timestamps are datetime objects."""
        agent = make_agent_data()
        await backend.save_agent(agent)
        decision = make_agent_decision_data(agent_id=agent["id"])
        await backend.save_agent_decision(decision)
        results = await backend.query_agent_decisions(agent["id"])
        assert len(results) >= 1

    async def test_agent_learning_datetime(self, backend):
        """Agent learning timestamps are datetime objects."""
        agent = make_agent_data()
        await backend.save_agent(agent)
        learning = make_agent_learning_data(agent_id=agent["id"])
        await backend.save_agent_learning(learning)
        results = await backend.query_agent_learnings(agent["id"])
        assert len(results) >= 1

    async def test_agent_notebook_datetime(self, backend):
        """Agent notebook timestamps are datetime objects."""
        agent = make_agent_data()
        await backend.save_agent(agent)
        notebook = make_agent_notebook_data(agent_id=agent["id"])
        await backend.save_agent_notebook(notebook)
        results = await backend.query_agent_notebooks(agent["id"])
        assert len(results) >= 1
