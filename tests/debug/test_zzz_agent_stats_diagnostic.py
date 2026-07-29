"""One-off diagnostic for the session_agent_stats 0-usage bug.

Not part of the regular suite intent - safe to delete after use.

Part 1 inspects the live postgres DB directly (before/after evidence).
Part 2 exercises the FIXED save_agent_execution/get_agent_stats code path
end-to-end with a throwaway record (http_server.py-shaped dict, i.e. what
AgentExecution.model_dump() actually produces: execution_id/started, not
id/started_at), then cleans the throwaway row up immediately after.
"""

import asyncio
import sys
import uuid
from datetime import UTC, datetime

sys.path.insert(0, "src")

from persistence.postgresql import PostgreSQLBackend  # noqa: E402


async def _run_inspect():
    db = PostgreSQLBackend("postgresql://localhost/session_intelligence")
    await db.initialize()

    pool = db._ensure_connected()
    async with pool.acquire() as conn:
        n_sessions = await conn.fetchval("SELECT count(*) FROM sessions")
        n_execs = await conn.fetchval("SELECT count(*) FROM agent_executions")
        now = await conn.fetchval("SELECT now()")

    print(f"DB now(): {now}")
    print(f"sessions count: {n_sessions}")
    print(f"agent_executions count: {n_execs}")

    await db.close()


async def _run_end_to_end():
    db = PostgreSQLBackend("postgresql://localhost/session_intelligence")
    await db.initialize()

    pool = db._ensure_connected()

    # Grab a real, recent session id to attach the throwaway execution to.
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1"
        )
    session_id = row["id"]

    diagnostic_agent_type = "META-function-manager-diagnostic-DELETE-ME"
    exec_id = f"diag-{uuid.uuid4().hex[:8]}"

    # This is the exact shape http_server.py's _persist_sessions_to_database
    # produces from AgentExecution.model_dump(): execution_id/started, NOT
    # id/started_at. Before the fix this raised KeyError on execution_data["id"]
    # (silently swallowed by http_server's broad except Exception).
    exec_data = {
        "agent_name": "diagnostic-agent",
        "agent_type": diagnostic_agent_type,
        "execution_id": exec_id,
        "started": datetime.now(UTC),
        "completed": datetime.now(UTC),
        "status": "success",
        "execution_steps": [],
        "performance": {},
        "errors": [],
        "session_id": session_id,
    }

    # Should NOT raise after the fix.
    await db.save_agent_execution(exec_data)
    print(f"save_agent_execution succeeded for {exec_id}")

    stats = await db.get_agent_stats(time_window_hours=1)
    print(f"get_agent_stats(1h) -> total_sessions_scanned={stats['total_sessions_scanned']}")
    matches = [a for a in stats["agents"] if a["agent_type"] == diagnostic_agent_type]
    print(f"matching agent_type entries: {matches}")

    assert matches, "diagnostic agent_type not found in get_agent_stats output"
    assert matches[0]["invocations"] == 1

    # Cleanup: remove the throwaway row so it doesn't pollute real telemetry.
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM agent_executions WHERE id = $1", exec_id)
    print(f"cleaned up diagnostic row {exec_id}")

    await db.close()


def test_diagnostic_inspect():
    asyncio.run(_run_inspect())


def test_diagnostic_end_to_end():
    asyncio.run(_run_end_to_end())
