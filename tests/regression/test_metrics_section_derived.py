"""Regression tests for deriving the Performance Metrics table from lists.

A cold-loaded notebook contradicted itself: the "Agents Executed" section
listed 26 entries (rendered from `session.agents_executed`, an in-memory
list) while the "Performance Metrics" table reported "Agents Executed: 2"
(read from `session.performance_metrics`, a stored blob refreshed only on
the live track_execution path and written mid-session). The two disagreed
because they came from different sources of truth.

`_generate_metrics_section` now derives every count-like row from the same
lists the rest of the notebook renders, so the sections can never disagree
again. Timing and efficiency have no list-based equivalent and still come
from the stored blob.

Every test below stores a deliberately WRONG counter in the stored blob and
asserts the rendered table shows the TRUE count from the lists. A test that
stores the same number in both places would be a tautology and would prove
nothing about the fix.

Match the house style: in-memory SQLite backend, asyncio_mode = "auto" (from
pyproject.toml) -- no @pytest.mark.asyncio decorators needed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "test_metrics_section_derived.db"))
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
        "project_path": "/tmp/metrics-section-derived-project",
        "project_name": "metrics-section-derived-project",
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


def _decision_row(decision_id: str, session_id: str, description: str) -> dict:
    return {
        "id": decision_id,
        "session_id": session_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "category": "implementation",
        "description": description,
        "rationale": None,
        "context": {},
        "impact_level": "medium",
        "artifacts": [],
    }


async def test_agents_row_counts_the_list_not_the_stored_counter(engine, db):
    sid = "metrics-agents-row-list-not-counter"
    await db.save_session(_session_row(sid, performance_metrics={"agents_executed": 3}))
    for i in range(7):
        await db.save_agent_execution(_execution_row(f"exec-{i}", sid, agent_name=f"agent-{i}"))
    engine.session_cache.clear()

    session = await engine._hydrate_session(sid)
    table = engine._generate_metrics_section(session)

    assert "| Agents Executed | 7 |" in table
    assert "| Agents Executed | 3 |" not in table


async def test_successful_and_failed_are_derived_from_statuses(engine, db):
    sid = "metrics-success-fail-derived"
    await db.save_session(
        _session_row(
            sid,
            performance_metrics={
                "successful_executions": 0,
                "failed_executions": 0,
            },
        )
    )
    await db.save_agent_execution(
        _execution_row("exec-ok-1", sid, agent_name="agent-ok-1", status="success")
    )
    await db.save_agent_execution(
        _execution_row("exec-ok-2", sid, agent_name="agent-ok-2", status="success")
    )
    await db.save_agent_execution(
        _execution_row("exec-bad", sid, agent_name="agent-bad", status="error")
    )
    engine.session_cache.clear()

    session = await engine._hydrate_session(sid)
    table = engine._generate_metrics_section(session)

    assert "| Successful Executions | 2 |" in table
    assert "| Failed Executions | 1 |" in table


async def test_decisions_row_counts_the_decisions_list(engine, db):
    sid = "metrics-decisions-row-list"
    await db.save_session(_session_row(sid, performance_metrics={"decisions_made": 0}))
    await db.save_decision(_decision_row("dec-1", sid, "First decision made in this session"))
    await db.save_decision(_decision_row("dec-2", sid, "Second decision made in this session"))
    engine.session_cache.clear()

    session = await engine._hydrate_session(sid)
    table = engine._generate_metrics_section(session)

    assert "| Decisions Made | 2 |" in table


async def test_notebook_sections_agree_end_to_end(engine, db):
    sid = "metrics-notebook-sections-agree"
    await db.save_session(_session_row(sid, performance_metrics={"agents_executed": 1}))
    for i in range(4):
        await db.save_agent_execution(_execution_row(f"exec-{i}", sid, agent_name=f"agent-{i}"))
    engine.session_cache.clear()

    result = await engine.session_create_notebook(
        session_id=sid, save_to_file=False, save_to_database=False
    )

    assert result.status == "success"
    markdown = result.markdown_output

    agents_section_lines = 0
    in_agents_section = False
    for line in markdown.split("\n"):
        if line.startswith("## "):
            in_agents_section = line.strip() == "## Agents Executed"
            continue
        if in_agents_section and line.startswith("- "):
            agents_section_lines += 1

    metrics_row = next(
        row for row in markdown.split("\n") if "Agents Executed" in row and "|" in row
    )
    reported_count = int(metrics_row.split("|")[2].strip())

    assert agents_section_lines == reported_count
    assert reported_count == 4


async def test_timing_and_efficiency_still_come_from_the_stored_blob(engine, db):
    sid = "metrics-timing-efficiency-from-blob"
    await db.save_session(
        _session_row(
            sid,
            performance_metrics={
                "total_execution_time_ms": 4500,
                "efficiency_score": 87.5,
            },
        )
    )
    engine.session_cache.clear()

    session = await engine._hydrate_session(sid)
    table = engine._generate_metrics_section(session)

    assert "| Total Execution Time | 4.5s |" in table
    assert "| Efficiency Score | 87.5% |" in table


async def test_session_with_no_activity_renders_zeros(engine, db):
    sid = "metrics-no-activity-renders-zeros"
    await db.save_session(_session_row(sid))
    engine.session_cache.clear()

    session = await engine._hydrate_session(sid)
    table = engine._generate_metrics_section(session)

    assert "| Agents Executed | 0 |" in table
    assert "| Successful Executions | 0 |" in table
    assert "| Decisions Made | 0 |" in table
