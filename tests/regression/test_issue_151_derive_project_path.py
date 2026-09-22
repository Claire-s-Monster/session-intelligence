"""
Regression tests for issue #151: sessions created through the session_name /
project_name branches of ``_resolve_session_context`` recorded the
"_unknown_" project_path sentinel even when a prior session for the same
project_name had already recorded a real, absolute path.

Verifies that ``SessionIntelligenceEngine._derive_project_path`` now falls
back to the most recent USABLE session for the same project_name (the same
evidence project-filtered recall matches against), instead of unconditionally
stamping the sentinel -- while a caller-supplied absolute path still always
wins, and the sentinel is still stored honestly when there is truly no prior
evidence to derive from.

Uses an in-memory SQLite backend so all tests are fully isolated from real
filesystem state and PostgreSQL.
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
    afterward. status defaults to "completed" (NOT active) since issue #151
    is specifically that non-active history was being ignored.
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeriveProjectPathFromPriorSession:
    async def test_project_name_branch_derives_from_prior_completed_session(self, engine, db):
        """Issue #151 core scenario: session_log_learning(project_name=X) with
        no project_path, where a NOT-active prior session for X recorded an
        absolute path, must derive that path -- not stamp '_unknown_'."""
        project = f"proj-151-{uuid.uuid4().hex[:8]}"
        await _seed_prior_session(db, project, "/home/user/repos/my-project", status="completed")

        result = await engine.session_log_learning(
            category="pattern",
            learning_content="some learning content",
            project_name=project,
        )

        assert result.learning.project_path == "/home/user/repos/my-project"
        assert result.learning.project_path != UNKNOWN_PROJECT_PATH

    async def test_session_name_branch_derives_from_prior_session(self, engine, db):
        """Branch 2 (session_name + project_name) must derive the same way as
        branch 3 (project_name alone) -- a brand-new session_name means the
        resolver hits the create path, which is exactly what's under test."""
        project = f"proj-151-name-{uuid.uuid4().hex[:8]}"
        await _seed_prior_session(db, project, "/home/user/repos/named-project")

        result = await engine.session_log_learning(
            category="pattern",
            learning_content="some learning content",
            session_name=f"brand-new-session-{uuid.uuid4().hex[:8]}",
            project_name=project,
        )

        assert result.learning.project_path == "/home/user/repos/named-project"

    async def test_no_prior_session_still_uses_sentinel(self, engine, db):
        """Documents the honest limit of the fix: with no prior evidence at
        all for this project_name, the sentinel is still stored -- there is
        nothing to derive from."""
        project = f"proj-151-none-{uuid.uuid4().hex[:8]}"

        result = await engine.session_log_learning(
            category="pattern",
            learning_content="some learning content",
            project_name=project,
        )

        assert result.learning.project_path == UNKNOWN_PROJECT_PATH

    async def test_caller_supplied_absolute_path_wins_over_derived(self, engine, db):
        """A caller-supplied absolute project_path always wins, even when a
        different path could be derived from prior history."""
        project = f"proj-151-override-{uuid.uuid4().hex[:8]}"
        await _seed_prior_session(db, project, "/home/user/repos/old-location")

        result = await engine.session_log_learning(
            category="pattern",
            learning_content="some learning content",
            project_name=project,
            project_path="/home/user/repos/new-location",
        )

        assert result.learning.project_path == "/home/user/repos/new-location"

    async def test_caller_supplied_relative_path_rejected_derivation_used(self, engine, db):
        """A relative caller-supplied project_path is not trustworthy (it
        would resolve against the SERVER's cwd, not the caller's) and must be
        discarded by _resolve_session_context in favor of derivation from
        prior history -- checked on the created session row itself, since
        that is what _derive_project_path controls. (session_log_learning's
        own effective_project fallback has a separate, pre-existing quirk of
        passing a caller-supplied string through verbatim regardless of
        absoluteness; that quirk is outside issue #151's scope.)"""
        project = f"proj-151-relative-{uuid.uuid4().hex[:8]}"
        await _seed_prior_session(db, project, "/home/user/repos/derived-location")

        result = await engine.session_log_learning(
            category="pattern",
            learning_content="some learning content",
            project_name=project,
            project_path="relative/caller/path",
        )

        session_id = result.learning.source_session_id
        assert session_id is not None
        stored_session = await db.get_session(session_id)
        assert stored_session is not None
        assert stored_session["project_path"] == "/home/user/repos/derived-location"
