"""
Unit tests for SessionIntelligenceEngine._resolve_session_context.

Uses an in-memory SQLite backend so all tests are fully isolated from
real filesystem state and PostgreSQL.
"""

import pytest

from core.session_engine import SessionContextRequiredError, SessionIntelligenceEngine
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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _create_and_persist(engine, db, project_name, session_name=None):
    """Create a session in the engine cache and persist it to the DB."""
    result = engine._create_session(
        mode="local",
        project_name=project_name,
        metadata={},
        session_name=session_name,
    )
    assert result.status == "success"
    await db.save_session(result.session_data.model_dump(mode="python"))
    return result.session_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveBySessionId:
    async def test_resolve_by_session_id_returns_id(self, engine, db):
        """Resolving by a known session_id returns that same id."""
        sid = await _create_and_persist(engine, db, "proj-a")
        resolved = await engine._resolve_session_context(session_id=sid)
        assert resolved == sid

    async def test_resolve_by_session_id_unknown_raises(self, engine, db):
        """Resolving by a nonexistent session_id raises ValueError."""
        with pytest.raises(ValueError, match="not found in database"):
            await engine._resolve_session_context(session_id="nonexistent-id")


class TestResolveBySessionName:
    async def test_resolve_by_session_name_finds_existing(self, engine, db):
        """Resolving by name returns the id of the named session."""
        sid = await _create_and_persist(engine, db, "proj-b", session_name="my-debug-session")
        resolved = await engine._resolve_session_context(session_name="my-debug-session")
        assert resolved == sid

    async def test_resolve_by_session_name_with_project_filter(self, engine, db):
        """When two sessions share a name in different projects, project_name scopes correctly."""
        sid_a = await _create_and_persist(engine, db, "proj-x", session_name="shared-name")
        sid_b = await _create_and_persist(engine, db, "proj-y", session_name="shared-name")

        resolved_x = await engine._resolve_session_context(
            session_name="shared-name", project_name="proj-x"
        )
        resolved_y = await engine._resolve_session_context(
            session_name="shared-name", project_name="proj-y"
        )

        assert resolved_x == sid_a
        assert resolved_y == sid_b
        assert resolved_x != resolved_y

    async def test_resolve_by_session_name_creates_when_missing_and_create_if_missing_true(
        self, engine, db
    ):
        """When name not in DB and create_if_missing=True, a new session is created."""
        resolved = await engine._resolve_session_context(
            session_name="brand-new-session",
            create_if_missing=True,
        )
        assert resolved is not None
        assert resolved != ""

        # Verify persisted in cache
        assert resolved in engine.session_cache

    async def test_resolve_by_session_name_raises_when_missing_and_create_if_missing_false(
        self, engine, db
    ):
        """When name not in DB and create_if_missing=False, ValueError is raised."""
        with pytest.raises(ValueError, match="not found"):
            await engine._resolve_session_context(
                session_name="ghost-session",
                create_if_missing=False,
            )


class TestResolveByProjectName:
    async def test_resolve_by_project_name_returns_most_recent_active(self, engine, db):
        """When multiple sessions exist for a project, the most-recent active is returned."""
        # Create two sessions for the same project
        sid_old = await _create_and_persist(engine, db, "proj-multi")
        # Mark the first as completed so it isn't 'active'
        await db._connection.execute(
            "UPDATE sessions SET status='completed' WHERE id=?", (sid_old,)
        )
        await db._connection.commit()
        sid_new = await _create_and_persist(engine, db, "proj-multi")

        resolved = await engine._resolve_session_context(project_name="proj-multi")
        assert resolved == sid_new

    async def test_resolve_by_project_name_creates_when_no_active(self, engine, db):
        """When no active session exists for a project, a new one is created."""
        resolved = await engine._resolve_session_context(
            project_name="brand-new-project",
            create_if_missing=True,
        )
        assert resolved is not None
        assert resolved != ""
        assert resolved in engine.session_cache


class TestResolveAllNone:
    async def test_resolve_all_none_raises_session_context_required(self, engine, db):
        """Passing no identifier and allow_unbound=False raises SessionContextRequiredError."""
        with pytest.raises(SessionContextRequiredError):
            await engine._resolve_session_context()

    async def test_resolve_all_none_with_allow_unbound_uses_legacy(self, engine, db):
        """allow_unbound=True falls back to _get_or_create_current_session_id."""
        resolved = await engine._resolve_session_context(allow_unbound=True)
        # Should return a valid session id (not raise)
        assert resolved is not None
        assert resolved != ""


class TestResolvePriority:
    async def test_resolve_priority_id_over_name_over_project(self, engine, db):
        """When all three identifiers are passed, session_id wins."""
        sid = await _create_and_persist(engine, db, "proj-prio", session_name="prio-name")
        # Also create a session for a different project with same name
        await _create_and_persist(engine, db, "other-proj", session_name="other-name")

        # Pass all three — session_id must win
        resolved = await engine._resolve_session_context(
            session_id=sid,
            session_name="other-name",
            project_name="other-proj",
        )
        assert resolved == sid
