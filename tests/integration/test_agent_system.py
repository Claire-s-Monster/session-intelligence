"""Test script for agent system tables and CRUD methods."""

import asyncio
import pytest

# Check if asyncpg is available
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False


@pytest.mark.asyncio
@pytest.mark.postgresql
@pytest.mark.skipif(not HAS_ASYNCPG, reason="asyncpg not installed")
async def test_schema():
    """Test agent system using PostgreSQL backend."""
    from src.persistence.postgresql import PostgreSQLBackend

    # Use test database
    db = PostgreSQLBackend(dsn="postgresql://localhost/session_intelligence")
    await db.initialize()

    try:
        # Test that all agent tables exist by querying them
        # The tables should exist from previous migrations

        # Test agent CRUD
        import uuid
        test_id = f"test-agent-{uuid.uuid4().hex[:8]}"

        agent_data = {
            "id": test_id,
            "name": f"test-agent-{uuid.uuid4().hex[:8]}",  # Unique name
            "agent_type": "focused",
            "display_name": "Test Agent",
            "description": "A test agent",
            "capabilities": ["testing", "validation"],
        }
        await db.save_agent(agent_data)

        # Retrieve by ID
        agent = await db.get_agent(test_id)
        assert agent is not None
        assert agent["name"] == agent_data["name"]
        assert agent["capabilities"] == ["testing", "validation"]

        # Retrieve by name
        agent = await db.get_agent_by_name(agent_data["name"])
        assert agent is not None
        assert agent["id"] == test_id

        # Test stats update
        await db.update_agent_stats(test_id, "executions")
        agent = await db.get_agent(test_id)
        assert agent["total_executions"] >= 1

        # Test decision CRUD
        decision_id = f"decision-{uuid.uuid4().hex[:8]}"
        decision_data = {
            "id": decision_id,
            "agent_id": test_id,
            "description": "Test decision description",
            "rationale": "Test reasoning",
            "category": "testing",
        }
        await db.save_agent_decision(decision_data)

        decisions = await db.query_agent_decisions(test_id)
        assert len(decisions) >= 1
        found = [d for d in decisions if d["id"] == decision_id]
        assert len(found) == 1
        assert found[0]["description"] == "Test decision description"

        # Test outcome update
        await db.update_agent_decision_outcome(decision_id, "success", "Worked well")
        decisions = await db.query_agent_decisions(test_id)
        found = [d for d in decisions if d["id"] == decision_id]
        assert found[0]["outcome"] == "success"

        # Test learning CRUD
        learning_id = f"learning-{uuid.uuid4().hex[:8]}"
        learning_data = {
            "id": learning_id,
            "agent_id": test_id,
            "category": "pattern",
            "trigger_context": "Test trigger",
            "learning_content": "Always validate input",
            "applies_to": ["testing", "validation"],
        }
        await db.save_agent_learning(learning_data)

        learnings = await db.query_agent_learnings(test_id)
        assert len(learnings) >= 1
        found = [l for l in learnings if l["id"] == learning_id]
        assert len(found) == 1
        assert found[0]["learning_content"] == "Always validate input"

        # Test learning outcome update
        await db.update_agent_learning_outcome(learning_id, True)
        learnings = await db.query_agent_learnings(test_id)
        found = [l for l in learnings if l["id"] == learning_id]
        assert found[0]["success_count"] >= 1

        # Test notebook CRUD
        notebook_id = f"notebook-{uuid.uuid4().hex[:8]}"
        notebook_data = {
            "id": notebook_id,
            "agent_id": test_id,
            "title": "Weekly Summary",
            "summary_markdown": "# Summary\n\nThis week...",
            "notebook_type": "summary",
            "tags": ["weekly", "summary"],
        }
        await db.save_agent_notebook(notebook_data)

        notebooks = await db.query_agent_notebooks(test_id)
        assert len(notebooks) >= 1
        found = [n for n in notebooks if n["id"] == notebook_id]
        assert len(found) == 1
        assert found[0]["title"] == "Weekly Summary"

        # Cleanup: We don't delete test data to avoid affecting other tests
        # In a real test suite, you'd use a test database or transactions

        print("All tests passed!")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(test_schema())
