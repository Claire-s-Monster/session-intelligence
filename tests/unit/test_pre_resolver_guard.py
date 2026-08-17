"""
Unit tests for the pre-resolver guard in session_log_decision.

Verifies that a stale disk-cached session ID does NOT silently hijack calls
that provide no explicit identifier (leak mode 1) and does NOT override an
explicit project_name/session_id (leak mode 2).

Since issue #72 the in-process flag no longer licenses an unbound call
either: _current_session_set_in_process only proves *this process* created a
session, and the HTTP transport shares one engine across every project, so it
never proved the session belonged to the caller's project.
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
# Tests
# ---------------------------------------------------------------------------


class TestPreResolverGuard:
    async def test_no_identifier_after_cold_start_raises(self, engine, db):
        """
        Leak mode 1: After a server restart the engine loads a stale session
        ID from disk into _current_session_id WITHOUT setting the in-process
        flag.  A no-identifier call must raise SessionContextRequiredError
        rather than silently binding to the stale session.
        """
        # Simulate disk-load: set _current_session_id directly (no flag)
        engine._current_session_id = "stale-from-disk-20260418-170137"
        assert engine._current_session_set_in_process is False

        with pytest.raises(SessionContextRequiredError):
            await engine.session_log_decision(decision="some decision")

    async def test_no_identifier_after_in_process_create_now_raises(self, engine, db):
        """
        Issue #72 reversed this case.  It used to be the "happy path": an
        in-process _create_session set both _current_session_id and
        _current_session_set_in_process, and a no-identifier call bound to
        that session.

        The premise was "this process created it, so it is mine" — true for a
        single-project stdio server, false for the HTTP transport, which builds
        ONE engine shared by every project (http_server.py lifespan()).  There
        the flag turns True as soon as ANY project creates a session, so an
        unbound call bound to whichever project most recently created one.
        That is how session-intelligence decisions ended up filed under
        package-incubator.

        The contract now matches session_log_learning: supply a scope, or pass
        allow_unbound=True.  Callers that want the old create-then-log flow
        pass the session_id that _create_session returned.
        """
        result = engine._create_session(
            mode="local",
            project_name="test-project",
            metadata={},
            session_name=None,
        )
        assert result.status == "success"
        await db.save_session(result.session_data.model_dump(mode="python"))

        # The ambient state the old guard keyed on is still set ...
        assert engine._current_session_id == result.session_id
        assert engine._current_session_set_in_process is True

        # ... but it is no longer sufficient to bind a decision.
        with pytest.raises(SessionContextRequiredError):
            await engine.session_log_decision(
                decision="a decision without explicit identifier"
            )

        # The documented replacement: pass the id the create returned.
        decision_result = await engine.session_log_decision(
            decision="a decision with the created session's id",
            session_id=result.session_id,
        )
        assert decision_result.decision_id != "error"
        assert decision_result.session_id == result.session_id

    async def test_explicit_project_name_overrides_stale_current(self, engine, db):
        """
        Leak mode 2: Even with a stale _current_session_id (disk-loaded, no
        in-process flag), passing project_name must route through the resolver
        and bind to a new/existing session for that project — NOT the stale one.
        """
        # Simulate disk-load
        engine._current_session_id = "stale-id-should-not-be-used"
        assert engine._current_session_set_in_process is False

        # Call with explicit project_name; resolver creates a fresh session
        decision_result = await engine.session_log_decision(
            decision="decision with project_name",
            project_name="myproject",
        )
        assert decision_result.decision_id != "error"
        # The resolved session must NOT be the stale one
        assert decision_result.session_id != "stale-id-should-not-be-used"

    async def test_explicit_session_id_overrides_stale_current(self, engine, db):
        """
        Explicit session_id always wins regardless of _current_session_id state.
        """
        # Create a real session to pass to session_log_decision
        result = engine._create_session(
            mode="local",
            project_name="proj-explicit",
            metadata={},
        )
        assert result.status == "success"
        await db.save_session(result.session_data.model_dump(mode="python"))
        real_sid = result.session_id

        # Overwrite current session with a stale one (no flag)
        engine._current_session_id = "stale-id-should-not-be-used"
        engine._current_session_set_in_process = False

        decision_result = await engine.session_log_decision(
            decision="decision with explicit session_id",
            session_id=real_sid,
        )
        assert decision_result.decision_id != "error"
        assert decision_result.session_id == real_sid
