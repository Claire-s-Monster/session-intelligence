"""
Regression tests for issue #33: hook-supplied Claude Code native session
UUIDs are rejected by execution tracking because they were never created
via `session_manage_lifecycle create`.

https://github.com/Claire-s-Monster/session-intelligence/issues/33

Verifies the fix:
  `_track_execution_sync` no longer returns `status="error-session-not-found"`
  on a cache miss. Instead it auto-creates and binds a session under the
  externally-supplied `session_id` (via `_create_session(..., session_id=...)`)
  so hook-driven callers (SubagentStart/SubagentStop) that pass a native
  Claude Code session UUID succeed on first use and reuse the same cached
  session on subsequent calls.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend

NATIVE_SESSION_ID = "cd2b76d0-37cc-4768-a466-2b61d3fd8947"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    """SessionIntelligenceEngine wired to a fresh in-process SQLite backend.

    Filesystem persistence is OFF - cache assertions are the focus of the
    issue #33 fix verification.
    """
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issue33.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


# ---------------------------------------------------------------------------
# Issue #33: hook-supplied native session ids auto-bind instead of erroring
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_hook_session_id_auto_binds_on_first_track_execution(engine):
    """A native Claude Code session UUID that was never created via
    `session_manage_lifecycle create` must auto-bind on first use instead
    of failing with status='error-session-not-found'."""
    result = engine.session_track_execution(
        session_id=NATIVE_SESSION_ID,
        agent_name="focused-code-modifier",
        step_data={"operation": "start", "description": "SubagentStart hook"},
    )

    assert result.status != "error-session-not-found", (
        "Hook-supplied native session id was rejected instead of being "
        "auto-bound. This is issue #33."
    )
    assert result.status == "success"

    assert NATIVE_SESSION_ID in engine.session_cache, (
        "Native session id was not registered in session_cache after "
        "auto-creation."
    )

    bound_session = engine.session_cache[NATIVE_SESSION_ID]
    assert bound_session.id == NATIVE_SESSION_ID
    assert "hook-bound" in bound_session.metadata.tags
    assert "claude-native-session" in bound_session.metadata.tags


@pytest.mark.regression
async def test_hook_session_id_reused_on_second_track_execution(engine):
    """Two calls with the same native session id (simulating SubagentStart
    then SubagentStop) must reuse the exact same cached Session rather than
    creating a duplicate or erroring on the second call."""
    first_result = engine.session_track_execution(
        session_id=NATIVE_SESSION_ID,
        agent_name="focused-code-modifier",
        step_data={"operation": "start", "description": "SubagentStart hook"},
    )
    assert first_result.status == "success"
    assert NATIVE_SESSION_ID in engine.session_cache
    first_session = engine.session_cache[NATIVE_SESSION_ID]

    second_result = engine.session_track_execution(
        session_id=NATIVE_SESSION_ID,
        agent_name="focused-code-modifier",
        step_data={"operation": "stop", "description": "SubagentStop hook"},
    )
    assert second_result.status == "success"
    assert second_result.status != "error-session-not-found"

    assert list(engine.session_cache.keys()).count(NATIVE_SESSION_ID) == 1
    second_session = engine.session_cache[NATIVE_SESSION_ID]

    assert second_session is first_session, (
        "Second call to session_track_execution with the same native "
        "session id created a distinct Session object instead of reusing "
        "the cached one."
    )
    assert second_session.id == first_session.id
    # Both hook calls should now be recorded as steps on the same session.
    assert len(second_session.agents_executed) >= 1
