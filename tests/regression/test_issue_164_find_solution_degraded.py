"""
Regression tests for issue #164.

`session_find_solution` swallowed any query failure and returned a result
byte-identical to a successful search that matched nothing. The failure must now
be visible via `degraded` / `degraded_reason` while staying a degraded (not
raised) outcome.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.sqlite import SQLiteBackend


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    """SessionIntelligenceEngine wired to a fresh in-process SQLite backend."""
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issue164.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


def _break_query(engine: SessionIntelligenceEngine, monkeypatch) -> None:
    async def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine.database, "find_error_solutions", boom)


@pytest.mark.regression
async def test_query_failure_is_marked_degraded(engine, monkeypatch):
    _break_query(engine, monkeypatch)

    result = await engine.session_find_solution(error_text="anything")

    assert result.degraded is True
    assert result.degraded_reason is not None
    assert "RuntimeError" in result.degraded_reason
    assert result.total_found == 0


@pytest.mark.regression
async def test_clean_empty_search_is_not_degraded(engine):
    result = await engine.session_find_solution(error_text="no such error anywhere")

    assert result.total_found == 0
    assert result.degraded is False
    assert result.degraded_reason is None


VALID_SOLUTION = {
    "id": "sol-1",
    "error_pattern": "boom",
    "created_at": "2026-01-01T00:00:00",
    "project_path": None,
}


def _patch_solutions(engine, monkeypatch, rows) -> None:
    async def fake(*args, **kwargs):
        return [dict(r) for r in rows]

    monkeypatch.setattr(engine.database, "find_error_solutions", fake)


def _break_learnings(engine, monkeypatch) -> None:
    async def boom(*args, **kwargs):
        raise RuntimeError("learnings down")

    monkeypatch.setattr(engine.database, "query_project_learnings", boom)


@pytest.mark.regression
async def test_unusable_project_path_is_degraded(engine):
    result = await engine.session_find_solution(error_text="x", project_path="relative/path")

    assert result.total_found == 0
    assert result.degraded is True
    assert "unusable project_path" in result.degraded_reason
    assert "relative/path" in result.degraded_reason


@pytest.mark.regression
async def test_no_database_is_degraded(engine):
    engine.database = None

    result = await engine.session_find_solution(error_text="x")

    assert result.degraded is True
    assert result.degraded_reason == "no database configured"


@pytest.mark.regression
async def test_learnings_failure_keeps_solutions_but_is_degraded(engine, monkeypatch):
    _patch_solutions(engine, monkeypatch, [VALID_SOLUTION])
    _break_learnings(engine, monkeypatch)

    result = await engine.session_find_solution(error_text="boom")

    assert len(result.solutions) == 1
    assert result.degraded is True
    assert result.degraded_reason.startswith("project_learnings query failed: RuntimeError")


@pytest.mark.regression
async def test_malformed_records_are_counted_as_degraded(engine, monkeypatch):
    _patch_solutions(engine, monkeypatch, [VALID_SOLUTION, {"bogus": 1}])

    result = await engine.session_find_solution(error_text="boom")

    assert len(result.solutions) == 1
    assert result.degraded is True
    assert result.degraded_reason == "skipped 1 malformed error_solutions record(s)"


@pytest.mark.regression
async def test_both_partial_failures_join_reasons(engine, monkeypatch):
    _patch_solutions(engine, monkeypatch, [VALID_SOLUTION, {"bogus": 1}])
    _break_learnings(engine, monkeypatch)

    result = await engine.session_find_solution(error_text="boom")

    assert result.degraded is True
    assert "skipped 1 malformed" in result.degraded_reason
    assert "project_learnings query failed" in result.degraded_reason
    assert "; " in result.degraded_reason


@pytest.mark.regression
async def test_degraded_flag_reaches_execute_tool_response(engine, monkeypatch):
    _break_query(engine, monkeypatch)
    interface = LeanMCPInterface(engine)
    execute = interface.app._tool_manager._tools["execute_tool"].fn

    response = await execute("session_find_solution", {"error_text": "anything"})

    assert response["status"] == "success"
    assert response["result"]["degraded"] is True
    assert "RuntimeError" in response["result"]["degraded_reason"]
