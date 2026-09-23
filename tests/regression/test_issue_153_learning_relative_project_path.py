"""
Regression tests for issue #153: session_log_learning's effective_project
fallback (session_engine.py:3538-3543) stores the caller-supplied
project_path verbatim with no absoluteness check and no sentinel check.

A relative path is meaningless server-side -- it resolves against the
SERVER's cwd, not the caller's -- yet _resolve_session_context already
rejects relative paths for the SESSION row (session_engine.py:335-346), so
the learning row ends up disagreeing with the session row it is FK-bound to.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from core.session_engine import UNKNOWN_PROJECT_PATH, SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db():
    """In-memory SQLite database, initialized and cleaned up per test."""
    backend = SQLiteBackend(db_path=":memory:")
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
def engine(db, monkeypatch: pytest.MonkeyPatch) -> SessionIntelligenceEngine:
    """Engine wired to in-memory SQLite, no filesystem."""
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENTS_DIR", "/tmp/nonexistent-agents")
    return SessionIntelligenceEngine(
        repository_path=None,
        use_filesystem=False,
        database=db,
    )


async def _seed_prior_session(
    db: SQLiteBackend, project_name: str, project_path: str, *, status: str = "completed"
) -> None:
    """Directly insert a session row as evidence of where a project lives.

    Bypasses the engine so the seeded row's project_path and status are
    exactly what the test controls, and inserted with an old started_at so
    it reads as prior history relative to anything the test triggers
    afterward. status defaults to "completed" (NOT active) so that
    _resolve_session_context's project_name branch takes the
    create-new-session path, which is where _derive_project_path (and thus
    the resolved_ctx.project_path the fix must fall through to) comes in.
    """
    await db.save_session(
        {
            "id": f"seed-{uuid.uuid4().hex[:12]}",
            "started_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            "project_path": project_path,
            "project_name": project_name,
            "status": status,
        }
    )


async def _stored_learning_project_path(
    db: SQLiteBackend, project_name: str, learning_id: str
) -> str:
    """Read a learning row back from the database by id.

    project_path="__nonexistent__" is intentional: the row's stored
    project_path is exactly what's under test (it may or may not be the
    sentinel/relative value passed in), so we can't filter by it. Instead
    we rely on query_project_learnings' project_name OR-widening (issue
    #120) to match on project_name alone, then pick the row by id.
    """
    rows = await db.query_project_learnings(
        project_path="__nonexistent__", project_name=project_name, limit=50
    )
    matches = [row for row in rows if row["id"] == learning_id]
    assert matches, f"learning {learning_id!r} not found for project_name={project_name!r}"
    return matches[0]["project_path"]


# ---------------------------------------------------------------------------
# Test 1: a relative caller-supplied project_path is not stored verbatim
# ---------------------------------------------------------------------------


async def test_relative_project_path_not_stored_verbatim(engine, db):
    """A relative project_path resolves against the SERVER's cwd, not the
    caller's, so it must not be trusted and stored as-is on the learning
    row -- the same absoluteness guard _resolve_session_context already
    applies to the session row it is FK-bound to."""
    result = await engine.session_log_learning(
        project_name="proj-153",
        project_path="relative/caller/path",
        category="pattern",
        learning_content="some learning content",
        trigger_context="some trigger",
    )

    stored_path = await _stored_learning_project_path(db, "proj-153", result.learning.id)
    assert stored_path != "relative/caller/path"


# ---------------------------------------------------------------------------
# Test 2: rejected relative path falls through to the resolved session path
# ---------------------------------------------------------------------------


async def test_relative_project_path_falls_through_to_resolved_session_path(engine, db):
    """Once the relative path is rejected, the learning row must agree with
    the resolved session's project_path instead of storing nothing useful
    or contradicting the session row it is bound to."""
    await _seed_prior_session(db, "proj-153", "/abs/real/proj-153", status="completed")

    result = await engine.session_log_learning(
        project_name="proj-153",
        project_path="relative/caller/path",
        category="pattern",
        learning_content="some learning content",
        trigger_context="some trigger",
    )

    stored_path = await _stored_learning_project_path(db, "proj-153", result.learning.id)
    assert stored_path == "/abs/real/proj-153"


# ---------------------------------------------------------------------------
# Test 3: an absolute caller-supplied path still wins (guards over-rejection)
# ---------------------------------------------------------------------------


async def test_absolute_caller_path_still_wins(engine, db):
    """A trustworthy, absolute caller-supplied project_path must still be
    stored verbatim -- the fix must not over-reject legitimate paths."""
    result = await engine.session_log_learning(
        project_name="proj-153",
        project_path="/abs/explicit/path",
        category="pattern",
        learning_content="some learning content",
        trigger_context="some trigger",
    )

    stored_path = await _stored_learning_project_path(db, "proj-153", result.learning.id)
    assert stored_path == "/abs/explicit/path"


# ---------------------------------------------------------------------------
# Test 4: the "_unknown_" sentinel is not trusted as a caller value either
# ---------------------------------------------------------------------------


async def test_sentinel_project_path_not_stored_as_caller_value(engine, db):
    """Passing the '_unknown_' sentinel explicitly as project_path must not
    be treated as trustworthy caller data (it is truthy, so a naive
    ``project_path or ...`` fallback stores it verbatim) -- it must fall
    through to the resolved session's project_path when one exists, exactly
    like a relative path does."""
    await _seed_prior_session(db, "proj-153", "/abs/real/proj-153", status="completed")

    result = await engine.session_log_learning(
        project_name="proj-153",
        project_path=UNKNOWN_PROJECT_PATH,
        category="pattern",
        learning_content="some learning content",
        trigger_context="some trigger",
    )

    stored_path = await _stored_learning_project_path(db, "proj-153", result.learning.id)
    assert stored_path == "/abs/real/proj-153"
