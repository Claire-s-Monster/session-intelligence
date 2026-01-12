"""Debug test with verbose error output."""

import asyncio
import sys
import traceback

sys.path.insert(0, "src")

from persistence.postgresql import PostgreSQLBackend
from core.session_engine import SessionIntelligenceEngine, debug_logger
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)


async def test():
    # Initialize database
    db = PostgreSQLBackend("postgresql://localhost/session_intelligence")
    await db.initialize()

    # Create engine with database
    engine = SessionIntelligenceEngine(repository_path=".", use_filesystem=False, database=db)

    print(f"Engine has database: {engine.database is not None}")

    try:
        # Query through engine with verbose output
        print("Calling agent_query_learnings...")
        learnings = await engine.agent_query_learnings("focused-quality-resolver")
        print(f"Engine learnings count: {len(learnings)}")
        for l in learnings:
            print(f"  - {l.id}: {l.learning_type} - {l.title}")
    except Exception as e:
        print(f"Error in learnings query: {e}")
        traceback.print_exc()

    try:
        print("Calling agent_query_decisions...")
        decisions = await engine.agent_query_decisions("focused-quality-resolver")
        print(f"Engine decisions count: {len(decisions)}")
        for d in decisions:
            print(f"  - {d.id}: {d.decision_type} - {d.decision[:50]}...")
    except Exception as e:
        print(f"Error in decisions query: {e}")
        traceback.print_exc()

    await db.close()


if __name__ == "__main__":
    asyncio.run(test())
