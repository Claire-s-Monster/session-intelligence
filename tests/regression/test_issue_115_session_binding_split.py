"""
Regression tests for issue #115, cause 1: "session-binding split".

`session_manage_lifecycle(operation="create")` used to always mint a fresh
`session-<ts>-<hex>` id, even when a caller supplied `session_id`. Meanwhile
SubagentStop/SubagentStart hooks call `session_track_execution(session_id=
<Claude Code's native session UUID>)`, and `_track_execution_sync`
auto-creates a session keyed by that literal UUID (issue #33). The two id
spaces could never meet, so the caller's later `create()` call would mint a
*different* id, and the caller's session would forever render
`agents_executed: 0`.

Verifies the fix:
  `create` honors a caller-supplied `session_id` verbatim (no minting), and
  if a session already exists under that id, `create` adopts it idempotently
  -- preserving its recorded executions/decisions/started timestamp --
  instead of overwriting it with a blank session.

https://github.com/Claire-s-Monster/session-intelligence/issues/115
"""

from datetime import UTC, datetime

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
    issue #115 fix verification.
    """
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issue115.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


@pytest.fixture
def memory_engine(tmp_path):
    """SessionIntelligenceEngine with NO database (memory-only mode)."""
    return SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=None,
    )


# ---------------------------------------------------------------------------
# Issue #115 (cause 1): create() honors a caller-supplied session_id
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_create_with_new_session_id_uses_it_verbatim(engine):
    """`create` with a session_id that does not already exist must create a
    session whose id is EXACTLY that string -- no minting."""
    result = await engine.session_manage_lifecycle(
        operation="create",
        project_name="demo-project",
        session_id=NATIVE_SESSION_ID,
    )

    assert result.status == "success"
    assert result.session_id == NATIVE_SESSION_ID
    assert result.operation == "create"
    assert NATIVE_SESSION_ID in engine.session_cache
    assert engine.session_cache[NATIVE_SESSION_ID].id == NATIVE_SESSION_ID


@pytest.mark.regression
async def test_create_with_existing_session_id_adopts_and_preserves_data(engine):
    """`create` with a session_id that already exists (e.g. created by a
    SubagentStart hook via session_track_execution) must idempotently adopt
    it: status='success', same id, and the EXISTING session's recorded
    execution, started timestamp, etc. must be preserved -- not clobbered
    with a blank session."""
    # Simulate the hook path: SubagentStart auto-creates the session and
    # records an execution under the native session id (issue #33).
    track_result = await engine.session_track_execution(
        session_id=NATIVE_SESSION_ID,
        agent_name="focused-code-modifier",
        step_data={"operation": "start", "description": "SubagentStart hook"},
    )
    assert track_result.status == "success"
    assert NATIVE_SESSION_ID in engine.session_cache

    pre_existing_session = engine.session_cache[NATIVE_SESSION_ID]
    original_started = pre_existing_session.started
    original_execution_count = len(pre_existing_session.agents_executed)
    assert original_execution_count >= 1, (
        "Setup failed to seed a recorded execution on the pre-existing session."
    )

    # Now the caller calls create() with the same id (e.g. the primary
    # session lifecycle bootstrapping after the hook already fired).
    create_result = await engine.session_manage_lifecycle(
        operation="create",
        project_name="demo-project",
        session_id=NATIVE_SESSION_ID,
    )

    assert create_result.status == "success"
    assert create_result.session_id == NATIVE_SESSION_ID
    assert create_result.operation == "create"

    adopted_session = engine.session_cache[NATIVE_SESSION_ID]
    assert adopted_session is pre_existing_session or adopted_session.id == NATIVE_SESSION_ID
    assert len(adopted_session.agents_executed) == original_execution_count, (
        "create() clobbered the pre-existing session's recorded executions."
    )
    assert adopted_session.started == original_started, (
        "create() overwrote the pre-existing session's original started timestamp."
    )

    # The returned session_data must reflect the EXISTING (preserved) session.
    assert create_result.session_data is not None
    assert len(create_result.session_data.agents_executed) == original_execution_count
    assert create_result.session_data.started == original_started


@pytest.mark.regression
async def test_create_with_existing_session_id_db_only_adopts_and_hydrates(engine):
    """A session that exists ONLY in the database (no session_cache entry --
    e.g. after a `systemctl --user restart`, which empties the in-memory
    cache while the hook-bound row persists in the database) must still be
    adopted with its recorded executions RECOVERED, not returned empty. A
    caller that adopts and reads back `agents_executed: 0` here still has
    the #115 bug, just moved one hop later."""
    started = datetime.now(UTC)
    await engine.database.save_session(
        {
            "id": NATIVE_SESSION_ID,
            "started": started.isoformat(),
            "project_name": "demo-project",
            "project_path": "",
            "mode": "local",
            "status": "active",
            "metadata": {},
            "performance_metrics": {},
            "health_status": {},
        }
    )
    await engine.database.save_agent_execution(
        {
            "id": "exec-1",
            "session_id": NATIVE_SESSION_ID,
            "agent_name": "focused-code-modifier",
            "agent_type": "focused",
            "started": started.isoformat(),
            "status": "completed",
        }
    )
    # Precondition: nothing in the cache -- this is the DB-only path.
    assert NATIVE_SESSION_ID not in engine.session_cache

    result = await engine.session_manage_lifecycle(
        operation="create",
        project_name="demo-project",
        session_id=NATIVE_SESSION_ID,
    )

    assert result.status == "success"
    assert result.session_id == NATIVE_SESSION_ID
    assert result.session_data is not None
    assert len(result.session_data.agents_executed) == 1, (
        "DB-only adopt path did not hydrate agent executions from the "
        "database -- the #115 bug survives a process restart."
    )
    assert result.session_data.agents_executed[0].agent_name == "focused-code-modifier"


@pytest.mark.regression
async def test_create_without_session_id_mints_as_before(engine):
    """`create` with NO session_id must be unchanged: mints a
    `session-<ts>-<hex>` id, not a raw UUID."""
    result = await engine.session_manage_lifecycle(
        operation="create",
        project_name="demo-project",
    )

    assert result.status == "success"
    assert result.session_id.startswith("session-")
    assert result.session_id in engine.session_cache


@pytest.mark.regression
async def test_create_with_new_session_id_memory_only_mode(memory_engine):
    """`create` with a session_id and NO database (self.database is None)
    must not crash and must fall back to checking session_cache."""
    result = await memory_engine.session_manage_lifecycle(
        operation="create",
        project_name="demo-project",
        session_id=NATIVE_SESSION_ID,
    )

    assert result.status == "success"
    assert result.session_id == NATIVE_SESSION_ID
    assert NATIVE_SESSION_ID in memory_engine.session_cache


@pytest.mark.regression
async def test_create_with_existing_session_id_memory_only_mode_adopts(memory_engine):
    """In memory-only mode (self.database is None), a second `create` call
    with the same session_id must still adopt the cached session rather than
    crashing or overwriting it."""
    first = await memory_engine.session_manage_lifecycle(
        operation="create",
        project_name="demo-project",
        session_id=NATIVE_SESSION_ID,
    )
    assert first.status == "success"
    first_session = memory_engine.session_cache[NATIVE_SESSION_ID]

    second = await memory_engine.session_manage_lifecycle(
        operation="create",
        project_name="demo-project",
        session_id=NATIVE_SESSION_ID,
    )

    assert second.status == "success"
    assert second.session_id == NATIVE_SESSION_ID
    assert memory_engine.session_cache[NATIVE_SESSION_ID] is first_session
