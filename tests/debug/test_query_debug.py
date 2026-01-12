"""Debug test for agent query functions."""

import asyncio
import sys

sys.path.insert(0, "src")

from persistence.postgresql import PostgreSQLBackend


async def test():
    db = PostgreSQLBackend("postgresql://localhost/session_intelligence")
    await db.initialize()

    # Get agent by name
    agent = await db.get_agent_by_name("focused-quality-resolver")
    print(f"Agent: {agent}")
    print(f'Agent ID: {agent["id"] if agent else None}')

    if agent:
        agent_id = agent["id"]

        # Query learnings directly
        learnings = await db.query_agent_learnings(agent_id)
        print(f"Learnings count: {len(learnings)}")
        for l in learnings:
            print(f'  - {l["id"]}: {l.get("category")}')

        # Query decisions directly
        decisions = await db.query_agent_decisions(agent_id)
        print(f"Decisions count: {len(decisions)}")
        for d in decisions:
            print(f'  - {d["id"]}: {d.get("category")}')

    await db.close()


if __name__ == "__main__":
    asyncio.run(test())
