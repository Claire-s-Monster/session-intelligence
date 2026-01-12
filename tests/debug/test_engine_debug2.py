"""Debug test for engine query functions with verbose logging."""

import asyncio
import sys

sys.path.insert(0, "src")

from persistence.postgresql import PostgreSQLBackend
from core.session_engine import SessionIntelligenceEngine


async def test():
    # Initialize database
    db = PostgreSQLBackend("postgresql://localhost/session_intelligence")
    await db.initialize()

    # Create engine with database
    engine = SessionIntelligenceEngine(repository_path=".", use_filesystem=False, database=db)

    print(f"Engine has database: {engine.database is not None}")
    print(f"Database type: {type(engine.database).__name__}")

    # Test get_agent_by_name directly
    agent_data = await db.get_agent_by_name("focused-quality-resolver")
    print(f"Direct DB get_agent_by_name: {agent_data}")

    # Check if engine.database.get_agent_by_name works
    agent_from_engine_db = await engine.database.get_agent_by_name("focused-quality-resolver")
    print(f"Engine DB get_agent_by_name: {agent_from_engine_db}")

    if agent_from_engine_db:
        agent_id = agent_from_engine_db["id"]
        print(f"Agent ID: {agent_id}")

        # Query learnings through engine.database
        learning_rows = await engine.database.query_agent_learnings(agent_id)
        print(f"Engine DB query_agent_learnings: {learning_rows}")

        # Query decisions through engine.database
        decision_rows = await engine.database.query_agent_decisions(agent_id)
        print(f"Engine DB query_agent_decisions: {decision_rows}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(test())
