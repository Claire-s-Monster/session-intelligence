"""
Tests for session-intelligence knowledge system methods.

Tests session_log_learning, session_find_solution, and
session_update_solution_outcome engine methods.

Note: session_log_learning and session_update_solution_outcome are sync methods
that use asyncio.create_task() internally to persist to the database. Tests that
verify database persistence must be async (to provide a running event loop).
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import (
    LearningCategory,
    LearningResult,
    SolutionResult,
    SolutionSearchResult,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def engine(tmp_path):
    """Engine without database (no persistence)."""
    return SessionIntelligenceEngine(repository_path=str(tmp_path))


@pytest.fixture
def mock_database():
    """Mock database with async methods."""
    db = AsyncMock()
    db.save_project_learning = AsyncMock(
        return_value={"id": "test", "status": "saved"}
    )
    db.get_session = AsyncMock(return_value=None)
    db.find_error_solutions = AsyncMock(return_value=[])
    db.query_project_learnings = AsyncMock(return_value=[])
    db.update_solution_outcome = AsyncMock(
        return_value={"id": "test", "status": "updated"}
    )
    return db


@pytest.fixture
def engine_with_db(tmp_path, mock_database):
    """Engine with mock database attached."""
    return SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        database=mock_database,
    )


# ============================================================================
# session_log_learning Tests
# ============================================================================


class TestSessionLogLearning:
    """Tests for session_log_learning method."""

    @pytest.mark.asyncio
    async def test_returns_learning_result(self, engine):
        """Returns a LearningResult with correct fields."""
        result = await engine.session_log_learning(
            category="pattern",
            learning_content="Use fixtures for test data",
            allow_unbound=True,
        )

        assert isinstance(result, LearningResult)
        assert result.id.startswith("learn_")
        assert result.learning is not None
        assert result.learning.learning_content == "Use fixtures for test data"
        assert result.learning.category == LearningCategory.PATTERN

    @pytest.mark.asyncio
    async def test_without_database_returns_pending(self, engine):
        """Without a database, status should be pending_save."""
        result = await engine.session_log_learning(
            category="error_fix",
            learning_content="Fix import errors with sys.path",
            allow_unbound=True,
        )

        assert result.status == "pending_save"

    @pytest.mark.asyncio
    async def test_with_database_saves_and_returns_saved(
        self, engine_with_db, mock_database
    ):
        """With a database and running event loop, status is 'saved'."""
        result = await engine_with_db.session_log_learning(
            category="workflow",
            learning_content="Run lint before commit",
            allow_unbound=True,
        )

        assert result.status == "saved"
        assert "saved" in result.message.lower()

        # Let the async task execute
        await asyncio.sleep(0)
        mock_database.save_project_learning.assert_called_once()

    @pytest.mark.asyncio
    async def test_learning_content_preserved(self, engine):
        """Learning content and trigger context are preserved."""
        result = await engine.session_log_learning(
            category="pattern",
            learning_content="Validate FK references before insert",
            trigger_context="Database FK violation encountered",
            allow_unbound=True,
        )

        assert result.learning.learning_content == "Validate FK references before insert"
        assert result.learning.trigger_context == "Database FK violation encountered"

    @pytest.mark.asyncio
    async def test_project_path_defaults_to_repository(self, engine):
        """Project path defaults to engine's repository path."""
        result = await engine.session_log_learning(
            category="preference",
            learning_content="Use ruff for linting",
            allow_unbound=True,
        )

        assert result.learning.project_path is not None
        assert len(result.learning.project_path) > 0

    @pytest.mark.asyncio
    async def test_custom_project_path(self, engine):
        """Custom project path overrides default."""
        result = await engine.session_log_learning(
            category="workflow",
            learning_content="Use TDD workflow",
            project_path="/custom/project",
            allow_unbound=True,
        )

        assert result.learning.project_path == "/custom/project"

    @pytest.mark.asyncio
    async def test_all_categories_accepted(self, engine):
        """All learning categories are accepted."""
        for cat in ["error_fix", "pattern", "preference", "workflow"]:
            result = await engine.session_log_learning(
                category=cat,
                learning_content=f"Test learning for {cat}",
                allow_unbound=True,
            )
            assert result.learning.category == LearningCategory(cat)

    @pytest.mark.asyncio
    async def test_invalid_category_raises(self, engine):
        """Invalid category raises ValueError."""
        with pytest.raises(ValueError):
            await engine.session_log_learning(
                category="invalid_category",
                learning_content="This should fail",
            )

    @pytest.mark.asyncio
    async def test_unique_ids(self, engine):
        """Each call generates a unique learning ID."""
        ids = set()
        for _ in range(10):
            result = await engine.session_log_learning(
                category="pattern",
                learning_content="Repeated learning",
                allow_unbound=True,
            )
            ids.add(result.id)

        assert len(ids) == 10

    @pytest.mark.asyncio
    async def test_no_source_session_saves_with_null_session(
        self, engine_with_db, mock_database
    ):
        """Without active session, source_session_id is None in save call."""
        assert engine_with_db._current_session_id is None

        result = await engine_with_db.session_log_learning(
            category="pattern",
            learning_content="No session learning",
            allow_unbound=True,
        )

        assert result.status == "saved"

        # Let async task run and verify the call
        await asyncio.sleep(0)
        mock_database.save_project_learning.assert_called_once()
        call_kwargs = mock_database.save_project_learning.call_args.kwargs
        assert call_kwargs["source_session_id"] is None

    @pytest.mark.asyncio
    async def test_with_source_session_validates_fk(
        self, engine_with_db, mock_database
    ):
        """With active session, validates session exists before FK insert."""
        engine_with_db._current_session_id = "sess_123"
        mock_database.get_session = AsyncMock(return_value={"id": "sess_123"})

        result = await engine_with_db.session_log_learning(
            category="workflow",
            learning_content="Session-linked learning",
            allow_unbound=True,
        )

        assert result.status == "saved"

        # Let async validation+save task run
        await asyncio.sleep(0)
        mock_database.get_session.assert_called_once_with("sess_123")
        mock_database.save_project_learning.assert_called_once()
        call_kwargs = mock_database.save_project_learning.call_args.kwargs
        assert call_kwargs["source_session_id"] == "sess_123"

    @pytest.mark.asyncio
    async def test_invalid_source_session_saves_with_null(
        self, engine_with_db, mock_database
    ):
        """If source session doesn't exist in DB, saves with None."""
        engine_with_db._current_session_id = "nonexistent_session"
        mock_database.get_session = AsyncMock(return_value=None)

        result = await engine_with_db.session_log_learning(
            category="pattern",
            learning_content="Orphan session learning",
            allow_unbound=True,
        )

        assert result.status == "saved"

        await asyncio.sleep(0)
        mock_database.save_project_learning.assert_called_once()
        call_kwargs = mock_database.save_project_learning.call_args.kwargs
        assert call_kwargs["source_session_id"] is None

    @pytest.mark.asyncio
    async def test_category_value_extracted_for_db(
        self, engine_with_db, mock_database
    ):
        """Category enum value (not enum object) is passed to database."""
        await engine_with_db.session_log_learning(
            category="error_fix",
            learning_content="Test category extraction",
            allow_unbound=True,
        )

        await asyncio.sleep(0)
        call_kwargs = mock_database.save_project_learning.call_args.kwargs
        assert call_kwargs["category"] == "error_fix"
        assert isinstance(call_kwargs["category"], str)


# ============================================================================
# session_find_solution Tests
# ============================================================================


class TestSessionFindSolution:
    """Tests for session_find_solution method."""

    @pytest.mark.asyncio
    async def test_without_database_returns_empty(self, engine):
        """Without database, returns empty results."""
        result = await engine.session_find_solution(
            error_text="ImportError: No module named 'foo'"
        )

        assert isinstance(result, SolutionSearchResult)
        assert result.total_found == 0
        assert result.solutions == []
        assert result.error_text == "ImportError: No module named 'foo'"

    @pytest.mark.asyncio
    async def test_with_database_queries_solutions(
        self, engine_with_db, mock_database
    ):
        """With database, queries error_solutions table."""
        mock_database.find_error_solutions = AsyncMock(
            return_value=[
                {
                    "id": "sol_1",
                    "error_pattern": "ImportError",
                    "solution_steps": ["pip install missing-package"],
                    "success_rate": 0.95,
                    "usage_count": 10,
                    "project_path": None,
                    "created_at": "2026-01-01T00:00:00",
                },
            ]
        )
        mock_database.query_project_learnings = AsyncMock(return_value=[])

        result = await engine_with_db.session_find_solution(
            error_text="ImportError: No module named 'requests'"
        )

        assert result.total_found == 1
        assert len(result.solutions) == 1
        mock_database.find_error_solutions.assert_called_once()

    @pytest.mark.asyncio
    async def test_also_queries_learnings(self, engine_with_db, mock_database):
        """Also queries project_learnings table for matching content."""
        mock_database.find_error_solutions = AsyncMock(return_value=[])
        mock_database.query_project_learnings = AsyncMock(
            return_value=[
                {
                    "id": "learn_1",
                    "project_path": "/test/project",
                    "category": "error_fix",
                    "learning_content": "Fix ImportError by adding to sys.path",
                    "trigger_context": "ImportError in tests",
                },
            ]
        )

        result = await engine_with_db.session_find_solution(
            error_text="ImportError"
        )

        # Learning matches because "ImportError" is in both fields
        assert result.total_found == 1
        mock_database.query_project_learnings.assert_called_once()

    @pytest.mark.asyncio
    async def test_filters_learnings_by_text(self, engine_with_db, mock_database):
        """Only includes learnings where error_text matches content."""
        mock_database.find_error_solutions = AsyncMock(return_value=[])
        mock_database.query_project_learnings = AsyncMock(
            return_value=[
                {
                    "id": "learn_1",
                    "learning_content": "Fix ImportError by checking path",
                    "trigger_context": "",
                    "project_path": "/test",
                },
                {
                    "id": "learn_2",
                    "learning_content": "Use black for formatting",
                    "trigger_context": "Style issue",
                    "project_path": "/test",
                },
            ]
        )

        result = await engine_with_db.session_find_solution(
            error_text="ImportError"
        )

        # Only learn_1 matches
        assert result.total_found == 1

    @pytest.mark.asyncio
    async def test_counts_project_vs_universal(self, engine_with_db, mock_database):
        """Correctly counts project-specific vs universal solutions."""
        project_path = str(engine_with_db.claude_sessions_path.parent)
        mock_database.find_error_solutions = AsyncMock(
            return_value=[
                {
                    "id": "sol_1",
                    "error_pattern": "TypeError",
                    "solution_steps": [],
                    "success_rate": 1.0,
                    "usage_count": 1,
                    "project_path": project_path,
                    "created_at": "2026-01-01T00:00:00",
                },
                {
                    "id": "sol_2",
                    "error_pattern": "TypeError",
                    "solution_steps": [],
                    "success_rate": 0.8,
                    "usage_count": 5,
                    "project_path": None,
                    "created_at": "2026-01-01T00:00:00",
                },
            ]
        )
        mock_database.query_project_learnings = AsyncMock(return_value=[])

        result = await engine_with_db.session_find_solution(
            error_text="TypeError"
        )

        assert result.total_found == 2
        assert result.project_specific_count == 1
        assert result.universal_count == 1

    @pytest.mark.asyncio
    async def test_handles_database_error_gracefully(
        self, engine_with_db, mock_database
    ):
        """Database errors return empty results, not exceptions."""
        mock_database.find_error_solutions = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        result = await engine_with_db.session_find_solution(
            error_text="some error"
        )

        assert result.total_found == 0
        assert result.solutions == []

    @pytest.mark.asyncio
    async def test_custom_project_path(self, engine_with_db, mock_database):
        """Custom project_path is forwarded to database queries."""
        mock_database.find_error_solutions = AsyncMock(return_value=[])
        mock_database.query_project_learnings = AsyncMock(return_value=[])

        await engine_with_db.session_find_solution(
            error_text="Error",
            project_path="/custom/path",
        )

        call_args = mock_database.find_error_solutions.call_args
        assert call_args.kwargs["project_path"] == "/custom/path"

    @pytest.mark.asyncio
    async def test_category_filter_passed(self, engine_with_db, mock_database):
        """error_category is forwarded to query_project_learnings."""
        mock_database.find_error_solutions = AsyncMock(return_value=[])
        mock_database.query_project_learnings = AsyncMock(return_value=[])

        await engine_with_db.session_find_solution(
            error_text="Error",
            error_category="compile",
        )

        call_args = mock_database.query_project_learnings.call_args
        assert call_args.kwargs["category"] == "compile"


# ============================================================================
# session_update_solution_outcome Tests
# ============================================================================


class TestSessionUpdateSolutionOutcome:
    """Tests for session_update_solution_outcome method."""

    @pytest.mark.asyncio
    async def test_without_database_returns_pending(self, engine):
        """Without database, returns pending_update status."""
        result = await engine.session_update_solution_outcome(
            solution_id="sol_123",
            success=True,
        )

        assert isinstance(result, SolutionResult)
        assert result.id == "sol_123"
        assert result.status == "pending_update"

    @pytest.mark.asyncio
    async def test_with_database_returns_updated(self, engine_with_db):
        """With database and event loop, returns updated status."""
        result = await engine_with_db.session_update_solution_outcome(
            solution_id="sol_456",
            success=True,
        )

        assert result.status == "updated"
        assert "success" in result.message.lower()

    @pytest.mark.asyncio
    async def test_failure_outcome_message(self, engine_with_db):
        """Failure outcome is reflected in message."""
        result = await engine_with_db.session_update_solution_outcome(
            solution_id="sol_789",
            success=False,
        )

        assert result.status == "updated"
        assert "failure" in result.message.lower()

    @pytest.mark.asyncio
    async def test_preserves_solution_id(self, engine):
        """Solution ID is preserved in result."""
        result = await engine.session_update_solution_outcome(
            solution_id="my_solution_id",
            success=True,
        )

        assert result.id == "my_solution_id"

    @pytest.mark.asyncio
    async def test_calls_database_update(self, engine_with_db, mock_database):
        """Verifies database.update_solution_outcome is called."""
        await engine_with_db.session_update_solution_outcome(
            solution_id="sol_abc",
            success=True,
        )

        # Let async task execute
        await asyncio.sleep(0)
        mock_database.update_solution_outcome.assert_called_once_with(
            solution_id="sol_abc",
            success=True,
        )


# ============================================================================
# Regression: Original bug detection
# ============================================================================


class TestOriginalBugDetection:
    """
    These tests catch the original bug where:
    - session_log_learning returned 'pending_save' and never saved
    - session_find_solution always returned empty results
    - session_update_solution_outcome returned 'pending_update' and never updated
    """

    @pytest.mark.asyncio
    async def test_learning_not_permanently_pending(self, engine_with_db):
        """Learning must not remain in 'pending_save' when DB is available."""
        result = await engine_with_db.session_log_learning(
            category="pattern",
            learning_content="Test content",
            allow_unbound=True,
        )

        assert result.status != "pending_save", (
            "REGRESSION: session_log_learning is returning 'pending_save' "
            "with a database attached. Learnings are NOT being persisted."
        )

    @pytest.mark.asyncio
    async def test_find_solution_not_always_empty(
        self, engine_with_db, mock_database
    ):
        """find_solution must return results when database has matches."""
        mock_database.find_error_solutions = AsyncMock(
            return_value=[
                {
                    "id": "sol_1",
                    "error_pattern": "TestError",
                    "solution_steps": ["fix it"],
                    "success_rate": 1.0,
                    "usage_count": 1,
                    "project_path": None,
                    "created_at": "2026-01-01T00:00:00",
                },
            ]
        )
        mock_database.query_project_learnings = AsyncMock(return_value=[])

        result = await engine_with_db.session_find_solution(
            error_text="TestError"
        )

        assert result.total_found > 0, (
            "REGRESSION: session_find_solution returns 0 results even "
            "though the database has matching solutions."
        )

    @pytest.mark.asyncio
    async def test_outcome_not_permanently_pending(self, engine_with_db):
        """Outcome must not remain in 'pending_update' when DB is available."""
        result = await engine_with_db.session_update_solution_outcome(
            solution_id="sol_1",
            success=True,
        )

        assert result.status != "pending_update", (
            "REGRESSION: session_update_solution_outcome is returning "
            "'pending_update' with a database attached."
        )
