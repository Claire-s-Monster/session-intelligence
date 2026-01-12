"""Debug test for engine query functions."""

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

    # Query through engine
    learnings = await engine.agent_query_learnings("focused-quality-resolver")
    print(f"Engine learnings count: {len(learnings)}")
    for l in learnings:
        print(f"  - {l.id}: {l.learning_type} - {l.title}")

    decisions = await engine.agent_query_decisions("focused-quality-resolver")
    print(f"Engine decisions count: {len(decisions)}")
    for d in decisions:
        print(f"  - {d.id}: {d.decision_type} - {d.decision[:50]}...")

    await db.close()


if __name__ == "__main__":
    asyncio.run(test())
