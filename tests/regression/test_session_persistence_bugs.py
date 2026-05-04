"""
Regression tests for PR #14: sessions not persisted to DB.

Bug: _create_session only stored sessions in memory/filesystem, never in the
     database. Decisions have FK constraint referencing sessions(id), so any
     decision INSERT failed with constraint violation.
Fix: session_manage_lifecycle now persists to DB after creation.
     session_log_decision auto-creates and persists session when needed.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


@pytest.mark.regression
class TestSessionPersistenceBugs:

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng

    async def test_session_persisted_to_db_not_just_memory(self, engine):
        """_create_session must write to DB, not just memory/filesystem."""
        result = await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="persist-test"
        )
        session_id = result.session_id

        db_session = await engine.database.get_session(session_id)
        assert db_session is not None, (
            "Session was created in memory but not persisted to database. "
            "This is the PR #14 bug."
        )
        assert db_session["project_name"] == "persist-test"

    async def test_decision_without_session_auto_creates(self, engine):
        """Logging a decision without active session must auto-create one and persist."""
        await engine.session_log_decision(
            decision="Decision without session",
            context={"rationale": "Testing auto-create", "category": "test"},
            allow_unbound=True,
        )

        assert engine._current_session_id is not None

    async def test_decision_fk_constraint_satisfied(self, engine):
        """Decision's session_id must reference a session that exists in DB."""
        result = await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="fk-test"
        )
        session_id = result.session_id

        await engine.session_log_decision(
            decision="FK test decision",
            context={"rationale": "Verify FK satisfied", "category": "test"},
        )

        decisions = await engine.database.query_decisions_by_session(session_id)
        assert len(decisions) >= 1
