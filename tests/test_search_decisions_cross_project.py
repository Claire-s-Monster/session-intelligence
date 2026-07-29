"""Cross-project decision search via session_search(search_type='decisions').

Verifies that the new 'decisions' branch in `search_sessions` returns matches
across project_name boundaries — the missing capability called out when
quarantining the PR #17 corruption.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from src.core.session_engine import SessionIntelligenceEngine
from src.persistence.sqlite import SQLiteBackend


@pytest.fixture
async def engine_with_decisions(tmp_path):
    """Engine backed by a fresh SQLite DB with two projects, two decisions each."""
    db_path = tmp_path / "search_decisions.db"
    backend = SQLiteBackend(db_path=str(db_path))
    await backend.initialize()

    engine = SessionIntelligenceEngine(repository_path=str(tmp_path))
    engine.database = backend

    now = datetime.now(UTC)

    # Two sessions, two distinct project_names
    sessions = [
        ("sess-alpha", "alpha-proj", str(tmp_path / "alpha")),
        ("sess-beta", "beta-proj", str(tmp_path / "beta")),
    ]
    for sid, pname, ppath in sessions:
        await backend.save_session(
            {
                "id": sid,
                "started_at": now,
                "project_path": ppath,
                "project_name": pname,
                "mode": "local",
                "status": "active",
                "metadata": {},
                "performance_metrics": {},
                "health_status": {},
            }
        )

    # Decisions split across both projects, all containing the keyword
    # "quarantine" so a single query should surface every one of them.
    decisions = [
        ("sess-alpha", "Designed quarantine label schema", "architecture"),
        ("sess-alpha", "Picked Postgres over Mongo", "infra"),
        ("sess-beta", "Wrote quarantine migration script", "ops"),
        ("sess-beta", "Skipped quarantine for empty rows", "ops"),
    ]
    for sid, desc, cat in decisions:
        await backend.save_decision(
            {
                "id": f"dec-{uuid.uuid4().hex[:8]}",
                "session_id": sid,
                "timestamp": now,
                "category": cat,
                "description": desc,
                "rationale": None,
                "context": {},
                "impact_level": "medium",
                "artifacts": [],
            }
        )

    yield engine
    await backend.close()


async def test_decisions_search_returns_matches_across_projects(engine_with_decisions):
    """A keyword query must surface decisions from both projects."""
    results = await engine_with_decisions.session_search(
        query="quarantine", search_type="decisions", limit=10
    )

    assert results.total_results >= 3
    project_names = {r.project_name for r in results.results}
    assert project_names == {"alpha-proj", "beta-proj"}


async def test_decisions_search_ignores_non_matching_decisions(engine_with_decisions):
    """Decisions not matching the keyword must be excluded."""
    results = await engine_with_decisions.session_search(
        query="quarantine", search_type="decisions", limit=10
    )

    snippets = " ".join((r.snippet or "") for r in results.results)
    assert "Postgres over Mongo" not in snippets


async def test_decisions_search_empty_when_no_match(engine_with_decisions):
    """Query with no hits returns zero results, not an error."""
    results = await engine_with_decisions.session_search(
        query="nonexistent-keyword-xyz", search_type="decisions", limit=10
    )

    assert results.total_results == 0
    assert results.results == []
