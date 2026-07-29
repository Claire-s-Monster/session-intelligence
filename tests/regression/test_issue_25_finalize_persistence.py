"""
Regression tests for issue #25: _finalize_session doesn't persist
status='completed' to DB; cache reload resurrects completed sessions.

https://github.com/Claire-s-Monster/session-intelligence/issues/25

Verifies the fix:
  Bug 1 - _finalize_session now awaits database.save_session() so the
          row transitions to status='completed' in the DB, not just
          in the in-memory cache.
  Bug 2 - Finalized sessions are popped from session_cache and the
          disk-reload path skips any session whose persisted status
          is COMPLETED, so subsequent calls don't resurrect stale
          state.
"""

import json

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import SessionStatus
from persistence.sqlite import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    """SessionIntelligenceEngine wired to a fresh in-process SQLite backend.

    Filesystem persistence is OFF - DB and cache assertions are the focus
    of Bug 1 / Bug 2 verification.
    """
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issue25.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


@pytest.fixture
async def engine_with_fs(tmp_path, monkeypatch):
    """Engine variant with filesystem persistence enabled.

    Used for the disk-reload-skip-completed scenario: we need to
    materialise a `current-session-id` pointer file and a
    `session-metadata.json` with status='completed' on disk, then
    invoke `_get_or_create_current_session_id` and verify the engine
    does not resurrect the completed session.
    """
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issue25_fs.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=True,
        database=db,
    )
    yield eng
    await db.close()


# ---------------------------------------------------------------------------
# Bug 1: finalize persists status='completed' to the database
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_finalize_persists_status_completed_to_db(engine):
    """After finalize, the DB row's status column is 'completed'."""
    create_result = await engine.session_manage_lifecycle(
        operation="create", mode="local", project_name="issue-25-bug-1"
    )
    session_id = create_result.session_id

    # Sanity: row exists and is active before finalize.
    pre = await engine.database.get_session(session_id)
    assert pre is not None, "Session row missing immediately after create"
    assert pre["status"] == "active"

    engine._current_session_id = session_id
    finalize_result = await engine.session_manage_lifecycle(operation="finalize")
    assert finalize_result.status == "success"

    # Post-finalize: DB row must reflect completed status. Pre-fix this
    # assertion fails - the row stays 'active' because _finalize_session
    # was synchronous and never awaited save_session.
    post = await engine.database.get_session(session_id)
    assert post is not None
    assert post["status"] == "completed", (
        "Finalize did not persist status='completed' to DB. "
        "This is issue #25 Bug 1."
    )


@pytest.mark.regression
async def test_finalize_clears_active_session_for_find_recent(engine):
    """find_recent_session_by_project(status='active') stops returning
    a finalized session - the symptom that drove issue #25."""
    create_result = await engine.session_manage_lifecycle(
        operation="create", mode="local", project_name="issue-25-recent"
    )
    session_id = create_result.session_id

    # Before finalize: the active session is the most recent for this project.
    active_before = await engine.database.find_recent_session_by_project(
        project_name="issue-25-recent", status="active"
    )
    assert active_before is not None
    assert active_before["id"] == session_id

    engine._current_session_id = session_id
    await engine.session_manage_lifecycle(operation="finalize")

    # After finalize: no active session matches the project anymore.
    active_after = await engine.database.find_recent_session_by_project(
        project_name="issue-25-recent", status="active"
    )
    assert active_after is None, (
        "find_recent_session_by_project(status='active') still returned "
        "the finalized session - stale 'active' row was not cleared."
    )


# ---------------------------------------------------------------------------
# Bug 2: finalize removes the session from the in-memory cache
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_finalize_removes_session_from_cache(engine):
    """Finalized sessions are popped from session_cache so subsequent
    disk reloads or pre-resolver guards don't pick them up as current."""
    create_result = await engine.session_manage_lifecycle(
        operation="create", mode="local", project_name="issue-25-cache"
    )
    session_id = create_result.session_id
    assert session_id in engine.session_cache

    engine._current_session_id = session_id
    await engine.session_manage_lifecycle(operation="finalize")

    assert session_id not in engine.session_cache, (
        "Finalized session is still in session_cache. This violates "
        "issue #25 Bug 2 invariant: completed sessions must not remain "
        "in the in-memory cache."
    )
    assert engine._current_session_id is None


# ---------------------------------------------------------------------------
# Bug 2: disk-reload skips sessions with persisted status=COMPLETED
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_disk_reload_skips_completed_session(engine_with_fs):
    """When a stale `current-session-id` pointer references a session
    whose persisted metadata says status='completed',
    _get_or_create_current_session_id must NOT resurrect it as the
    current session. It should clean up the stale pointer and create
    a fresh session instead."""
    # Create + finalize a session through normal lifecycle so a metadata
    # file with status='completed' exists on disk.
    create_result = await engine_with_fs.session_manage_lifecycle(
        operation="create", mode="local", project_name="issue-25-disk"
    )
    completed_id = create_result.session_id
    engine_with_fs._current_session_id = completed_id
    await engine_with_fs.session_manage_lifecycle(operation="finalize")

    # Sanity: metadata file says completed.
    session_dir = engine_with_fs.claude_sessions_path / completed_id
    metadata_file = session_dir / "session-metadata.json"
    assert metadata_file.exists()
    with open(metadata_file) as fh:
        on_disk = json.load(fh)
    assert on_disk["status"] == SessionStatus.COMPLETED.value

    # Simulate a stale pointer: re-create the `current-session-id` file
    # pointing at the completed session, and clear engine state so the
    # next `_get_or_create_current_session_id` call must consult disk.
    pointer = engine_with_fs.claude_sessions_path.parent / "current-session-id"
    pointer.write_text(completed_id + "\n")
    engine_with_fs.session_cache.pop(completed_id, None)
    engine_with_fs._current_session_id = None

    next_id = engine_with_fs._get_or_create_current_session_id()

    assert next_id is not None
    assert next_id != completed_id, (
        "Disk-reload path resurrected a completed session as the current "
        "active session. This is issue #25 Bug 2."
    )
