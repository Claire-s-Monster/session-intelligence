"""Regression tests for hydrating agent executions on a cold-loaded session.

PR #104 added `_hydrate_session` so a session absent from the in-memory cache
is loaded from the database. It loaded the session row and its decisions but
NOT its agent executions, so a cold-loaded notebook silently omitted its
"Agents Executed" section (that section reads `session.agents_executed`, an
in-memory list) while the metrics table still reported a non-zero agent
count. `_load_agent_executions` / `_agent_execution_from_row` close that gap.

Match the house style: in-memory SQLite backend, asyncio_mode = "auto" (from
pyproject.toml) -- no @pytest.mark.asyncio decorators needed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from core.session_engine import AGENT_EXECUTION_PAGE_SIZE, SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "test_hydrate_agent_executions.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def engine(tmp_path, db):
    return SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )


def _session_row(session_id: str, performance_metrics: dict | None = None) -> dict:
    return {
        "id": session_id,
        "started": datetime.now(UTC).isoformat(),
        "project_path": "/tmp/hydrate-agent-executions-project",
        "project_name": "hydrate-agent-executions-project",
        "mode": "local",
        "status": "active",
        "metadata": {},
        "performance_metrics": performance_metrics or {},
        "health_status": {},
    }


def _execution_row(
    execution_id: str,
    session_id: str,
    agent_name: str = "focused-code-modifier",
    status: str = "success",
    execution_steps: list | str | None = None,
) -> dict:
    return {
        "id": execution_id,
        "session_id": session_id,
        "agent_name": agent_name,
        "agent_type": "focused",
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "status": status,
        "execution_steps": [] if execution_steps is None else execution_steps,
        "performance": {},
        "errors": [],
    }


async def test_cold_session_loads_its_agent_executions(engine, db):
    sid = "hydrate-two-executions"
    await db.save_session(_session_row(sid))
    await db.save_agent_execution(_execution_row("exec-1", sid, agent_name="agent-alpha"))
    await db.save_agent_execution(_execution_row("exec-2", sid, agent_name="agent-beta"))
    engine.session_cache.clear()

    session = await engine._hydrate_session(sid)

    assert len(session.agents_executed) == 2
    agent_names = {e.agent_name for e in session.agents_executed}
    assert agent_names == {"agent-alpha", "agent-beta"}


async def test_notebook_for_cold_session_includes_agents_section(engine, db):
    sid = "hydrate-notebook-agents-section"
    await db.save_session(_session_row(sid))
    await db.save_agent_execution(
        _execution_row("exec-1", sid, agent_name="agent-notebook-visible")
    )
    engine.session_cache.clear()

    result = await engine.session_create_notebook_async(session_id=sid, save_to_file=False)

    assert result.status == "success"
    assert "Agents Executed" in result.markdown_output
    assert "agent-notebook-visible" in result.markdown_output


async def test_all_executions_load_when_count_exceeds_page_size(engine, db):
    sid = "hydrate-exceeds-page-size"
    await db.save_session(_session_row(sid))
    total = AGENT_EXECUTION_PAGE_SIZE + 5
    for i in range(total):
        await db.save_agent_execution(_execution_row(f"exec-{i}", sid, agent_name=f"agent-{i}"))
    engine.session_cache.clear()

    session = await engine._hydrate_session(sid)

    assert len(session.agents_executed) == total


async def test_partial_load_cannot_shrink_the_stored_agent_count(engine, db):
    sid = "hydrate-stored-count-no-shrink"
    total = AGENT_EXECUTION_PAGE_SIZE + 5
    await db.save_session(_session_row(sid, performance_metrics={"agents_executed": total}))
    for i in range(total):
        await db.save_agent_execution(_execution_row(f"exec-{i}", sid, agent_name=f"agent-{i}"))
    engine.session_cache.clear()

    session = await engine._hydrate_session(sid)

    assert len(session.agents_executed) == session.performance_metrics.agents_executed


async def test_unreadable_execution_row_is_skipped_not_fatal(engine, db):
    sid = "hydrate-unreadable-row-skipped"
    await db.save_session(_session_row(sid))
    await db.save_agent_execution(_execution_row("exec-good-1", sid, agent_name="agent-good-1"))
    await db.save_agent_execution(
        _execution_row("exec-bad", sid, agent_name="agent-bad", status="not-a-real-status")
    )
    await db.save_agent_execution(_execution_row("exec-good-2", sid, agent_name="agent-good-2"))
    engine.session_cache.clear()

    session = await engine._hydrate_session(sid)

    # An unrecognised status falls back to RUNNING rather than raising, so
    # all three rows -- including the "bad" one -- load successfully. The
    # scenario still guards against an exception escaping hydration.
    agent_names = {e.agent_name for e in session.agents_executed}
    assert {"agent-good-1", "agent-good-2"}.issubset(agent_names)
    assert len(session.agents_executed) == 3


async def test_execution_steps_stored_as_json_string_are_decoded(engine, db):
    sid = "hydrate-steps-json-string"
    await db.save_session(_session_row(sid))
    step = {
        "step_id": "step-1",
        "step_number": 1,
        "agent": "focused-code-modifier",
        "operation": "edit_file",
        "description": "Applied a targeted edit",
        "started": datetime.now(UTC).isoformat(),
    }
    row = _execution_row("exec-steps", sid)
    row["execution_steps"] = json.dumps([step])
    await db.save_agent_execution(row)
    engine.session_cache.clear()

    session = await engine._hydrate_session(sid)

    assert len(session.agents_executed) == 1
    steps = session.agents_executed[0].execution_steps
    assert len(steps) == 1
    assert steps[0].operation == "edit_file"


async def test_context_is_synthesised_from_the_session(engine, db):
    sid = "hydrate-context-synthesised"
    row = _session_row(sid)
    await db.save_session(row)
    await db.save_agent_execution(_execution_row("exec-context", sid))
    engine.session_cache.clear()

    session = await engine._hydrate_session(sid)

    context = session.agents_executed[0].context
    assert context.session_id == sid
    assert context.project_path == row["project_path"]


async def test_session_with_no_executions_hydrates_cleanly(engine, db):
    sid = "hydrate-no-executions"
    await db.save_session(_session_row(sid))
    engine.session_cache.clear()

    session = await engine._hydrate_session(sid)

    assert session.agents_executed == []
