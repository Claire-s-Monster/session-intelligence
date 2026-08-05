"""
Unit tests for the "_unknown_" project_path sentinel.

Verifies that when no project_path is supplied (directly or via a
resolvable session context), the engine records the explicit
UNKNOWN_PROJECT_PATH sentinel rather than silently falling back to the
server process's own cwd.

Uses an in-memory SQLite backend so all tests are fully isolated from
real filesystem state and PostgreSQL.
"""

from pathlib import Path

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateSessionProjectPathSentinel:
    def test_create_session_empty_metadata_uses_sentinel(self, engine):
        """With no project_path in metadata, the sentinel is used, not cwd."""
        result = engine._create_session(
            mode="local",
            project_name="proj-sentinel-empty",
            metadata={},
        )
        assert result.status == "success"
        assert result.session_data.project_path == UNKNOWN_PROJECT_PATH
        assert result.session_data.project_path != str(Path.cwd())

    def test_create_session_explicit_project_path_preserved(self, engine):
        """An explicitly-supplied project_path is preserved verbatim."""
        result = engine._create_session(
            mode="local",
            project_name="proj-sentinel-explicit",
            metadata={"project_path": "/tmp/some/caller/path"},
        )
        assert result.status == "success"
        assert result.session_data.project_path == "/tmp/some/caller/path"

    def test_create_session_none_project_path_uses_sentinel(self, engine):
        """An explicitly-passed None must also fall through to the sentinel."""
        result = engine._create_session(
            mode="local",
            project_name="proj-sentinel-none",
            metadata={"project_path": None},
        )
        assert result.status == "success"
        assert result.session_data.project_path == UNKNOWN_PROJECT_PATH


class TestSessionLogLearningProjectPathSentinel:
    async def test_session_log_learning_unresolvable_context_uses_sentinel(
        self, engine, db
    ):
        """No project_path and no resolvable session context -> sentinel stored."""
        result = await engine.session_log_learning(
            category="pattern",
            learning_content="some learning content",
            allow_unbound=True,
        )
        assert result.learning.project_path == UNKNOWN_PROJECT_PATH
        assert result.learning.project_path != str(Path.cwd())
