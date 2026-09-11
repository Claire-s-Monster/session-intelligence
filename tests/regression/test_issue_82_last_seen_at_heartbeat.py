"""
Regression tests for issue #82: sessions and agent_executions gain a
last_seen_at heartbeat column so staleness is measured from last activity,
not from started_at alone.

Background:
    #69 / #70 introduced started_at-based staleness cutoffs for
    get_active_session_for_project, find_recent_session_by_project,
    reap_abandoned_sessions, and reap_stale_executions. That makes a
    long-running session (or execution) that has been active the whole time
    indistinguishable from one that was genuinely abandoned hours ago --
    both have an old started_at.

Fix (expected, may not be landed yet):
    - New last_seen_at column on both sessions and agent_executions.
    - The four staleness sites above gate on
      COALESCE(last_seen_at, started_at) instead of started_at alone.
    - Thresholds still come from get_session_max_age_hours() /
      get_execution_max_age_hours() (src/persistence/base.py) -- unchanged.
    - Idempotent migration backfills last_seen_at = started_at for
      pre-existing rows, mirroring the session_name / project_name
      migrations already in SQLiteBackend.initialize().

Uses the SQLite backend directly, following the existing regression-test
style (see test_issue_69_reap_abandoned_sessions.py,
test_issue_70_reconcile_stuck_executions.py).

NOTE: these tests are expected to FAIL until issue #82 lands -- they were
written against the target behavior, not the current implementation.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from persistence.base import DEFAULT_EXECUTION_MAX_AGE_HOURS, DEFAULT_SESSION_MAX_AGE_HOURS
from persistence.sqlite import SQLiteBackend

AGENT_NAME = "focused-code-modifier"


def _iso(hours_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()


def _session_dict(
    *,
    status: str = "active",
    age_hours: float = 0.0,
    last_seen_hours: float | None = None,
    project_path: str = "/tmp/proj",
    project_name: str = "proj",
) -> dict:
    """Build a raw session dict ready for SQLiteBackend.save_session.

    age_hours controls how far in the past started_at is set (as in #69's
    fixture). last_seen_hours independently controls last_seen_at -- issue
    #82's heartbeat column -- so tests can place a session on either side of
    the staleness threshold for EACH timestamp separately.
    last_seen_hours=None means "no heartbeat ever recorded" (NULL in the
    DB), exercising the COALESCE(last_seen_at, started_at) fallback.
    """
    sid = f"session-{uuid.uuid4().hex[:8]}"
    return {
        "id": sid,
        "started_at": _iso(age_hours),
        "ended_at": None,
        "last_seen_at": _iso(last_seen_hours) if last_seen_hours is not None else None,
        "project_path": project_path,
        "project_name": project_name,
        "mode": "local",
        "status": status,
        "metadata": {},
        "performance_metrics": {},
        "health_status": {},
    }


def _execution_dict(
    *,
    session_id: str,
    status: str = "running",
    age_hours: float = 0.0,
    last_seen_hours: float | None = None,
    agent_name: str = AGENT_NAME,
    agent_type: str = "focused",
) -> dict:
    """Build a raw agent_executions dict ready for SQLiteBackend.save_agent_execution.

    Mirrors _session_dict's age_hours / last_seen_hours split -- see #70's
    fixture for the FK-on-session_id note (a session row must exist first).
    """
    return {
        "id": f"exec-{uuid.uuid4().hex[:8]}",
        "session_id": session_id,
        "agent_name": agent_name,
        "agent_type": agent_type,
        "started_at": _iso(age_hours),
        "completed_at": None,
        "last_seen_at": _iso(last_seen_hours) if last_seen_hours is not None else None,
        "status": status,
        "execution_steps": [],
        "performance": {},
        "errors": [],
    }


# ---------------------------------------------------------------------------
# get_active_session_for_project
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestGetActiveSessionForProjectHonorsLastSeenAt:
    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db
        await db.close()

    async def test_old_started_at_recent_last_seen_at_still_returned(self, backend):
        """THE HEADLINE CASE: a long-running session (old started_at) that
        has been heartbeating recently (recent last_seen_at) must not be
        indistinguishable from a genuinely abandoned one."""
        session = _session_dict(
            age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 5,
            last_seen_hours=0.1,
            project_name="proj-headline",
        )
        await backend.save_session(session)

        result = await backend.get_active_session_for_project(session["project_path"])

        assert result is not None, (
            "Issue #82: staleness must be measured from last_seen_at (via "
            "COALESCE(last_seen_at, started_at)), not started_at alone -- "
            "a session with a recent heartbeat must stay resolvable as "
            "the active session for its project regardless of its age."
        )
        assert result["id"] == session["id"]

    async def test_old_started_at_old_last_seen_at_still_excluded(self, backend):
        session = _session_dict(
            age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 5,
            last_seen_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 5,
            project_name="proj-old-old",
        )
        await backend.save_session(session)

        result = await backend.get_active_session_for_project(session["project_path"])

        assert result is None, (
            "No regression of #69: a session stale on BOTH started_at and "
            "last_seen_at must still be excluded."
        )

    async def test_null_last_seen_at_falls_back_to_started_at(self, backend):
        session = _session_dict(
            age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 5,
            last_seen_hours=None,
            project_name="proj-null",
        )
        await backend.save_session(session)

        result = await backend.get_active_session_for_project(session["project_path"])

        assert result is None, (
            "Issue #82: COALESCE(last_seen_at, started_at) must fall back "
            "to started_at for un-backfilled (NULL last_seen_at) rows, so "
            "an old row with no heartbeat ever recorded stays excluded -- "
            "the change must not make anything MORE resurrectable."
        )


# ---------------------------------------------------------------------------
# find_recent_session_by_project
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestFindRecentSessionByProjectHonorsLastSeenAt:
    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db
        await db.close()

    async def test_old_started_at_recent_last_seen_at_still_returned(self, backend):
        session = _session_dict(
            age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 5,
            last_seen_hours=0.1,
            project_name="proj-frsbp-fresh",
        )
        await backend.save_session(session)

        result = await backend.find_recent_session_by_project(
            "proj-frsbp-fresh", status="active"
        )

        assert result is not None, (
            "find_recent_session_by_project must apply the same "
            "last_seen_at-aware staleness guard as "
            "get_active_session_for_project, or the headline case is "
            "reintroduced by a second lookup path."
        )
        assert result["id"] == session["id"]

    async def test_old_started_at_old_last_seen_at_still_excluded(self, backend):
        session = _session_dict(
            age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 5,
            last_seen_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 5,
            project_name="proj-frsbp-stale",
        )
        await backend.save_session(session)

        result = await backend.find_recent_session_by_project(
            "proj-frsbp-stale", status="active"
        )

        assert result is None


# ---------------------------------------------------------------------------
# reap_abandoned_sessions
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestReapAbandonedSessionsHonorsLastSeenAt:
    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db
        await db.close()

    async def test_recent_last_seen_at_not_reaped(self, backend):
        session = _session_dict(
            age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 5,
            last_seen_hours=0.1,
            project_name="proj-reap-fresh",
        )
        await backend.save_session(session)

        reaped = await backend.reap_abandoned_sessions()

        assert reaped == 0, (
            "A session heartbeating recently must not be reaped just "
            "because its started_at is old."
        )
        row = await backend.get_session(session["id"])
        assert row["status"] == "active"

    async def test_old_last_seen_at_still_reaped(self, backend):
        session = _session_dict(
            age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 5,
            last_seen_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 5,
            project_name="proj-reap-stale",
        )
        await backend.save_session(session)

        reaped = await backend.reap_abandoned_sessions()

        assert reaped == 1, "No regression of #69: a genuinely stale session must still be reaped."
        row = await backend.get_session(session["id"])
        assert row["status"] == "abandoned", (
            "Reaped rows must become 'abandoned', never 'completed' -- "
            "guard against regressing #69's core invariant while adding "
            "the last_seen_at heartbeat."
        )

    async def test_null_last_seen_at_falls_back_and_is_reaped(self, backend):
        session = _session_dict(
            age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 5,
            last_seen_hours=None,
            project_name="proj-reap-null",
        )
        await backend.save_session(session)

        reaped = await backend.reap_abandoned_sessions()

        assert reaped == 1, (
            "COALESCE fallback: a NULL-heartbeat row with an old "
            "started_at must still be reaped, proving the change is safe "
            "for un-backfilled data."
        )
        row = await backend.get_session(session["id"])
        assert row["status"] == "abandoned"


# ---------------------------------------------------------------------------
# reap_stale_executions
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestReapStaleExecutionsHonorsLastSeenAt:
    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db
        await db.close()

    async def _save_session(self, backend, project_name: str) -> str:
        session = _session_dict(project_name=project_name)
        await backend.save_session(session)
        return session["id"]

    async def test_recent_last_seen_at_not_reaped(self, backend):
        session_id = await self._save_session(backend, "proj-exec-fresh")
        execution = _execution_dict(
            session_id=session_id,
            age_hours=DEFAULT_EXECUTION_MAX_AGE_HOURS + 5,
            last_seen_hours=0.1,
        )
        await backend.save_agent_execution(execution)

        reaped = await backend.reap_stale_executions()

        assert reaped == 0, (
            "An execution heartbeating recently must not be swept just "
            "because its started_at is old."
        )
        rows = await backend.query_agent_executions(session_id=session_id)
        assert rows[0]["status"] == "running"

    async def test_old_last_seen_at_still_reaped(self, backend):
        session_id = await self._save_session(backend, "proj-exec-stale")
        execution = _execution_dict(
            session_id=session_id,
            age_hours=DEFAULT_EXECUTION_MAX_AGE_HOURS + 5,
            last_seen_hours=DEFAULT_EXECUTION_MAX_AGE_HOURS + 5,
        )
        await backend.save_agent_execution(execution)

        reaped = await backend.reap_stale_executions()

        assert reaped == 1, "No regression of #70: a genuinely stale execution must still be reaped."
        rows = await backend.query_agent_executions(session_id=session_id)
        assert rows[0]["status"] == "abandoned"

    async def test_null_last_seen_at_falls_back_and_is_reaped(self, backend):
        session_id = await self._save_session(backend, "proj-exec-null")
        execution = _execution_dict(
            session_id=session_id,
            age_hours=DEFAULT_EXECUTION_MAX_AGE_HOURS + 5,
            last_seen_hours=None,
        )
        await backend.save_agent_execution(execution)

        reaped = await backend.reap_stale_executions()

        assert reaped == 1, (
            "COALESCE fallback: a NULL-heartbeat execution with an old "
            "started_at must still be reaped."
        )
        rows = await backend.query_agent_executions(session_id=session_id)
        assert rows[0]["status"] == "abandoned"


# ---------------------------------------------------------------------------
# Backfill migration
# ---------------------------------------------------------------------------


@pytest.mark.regression
class TestBackfillMigration:
    async def test_existing_null_last_seen_at_backfilled_to_started_at(self, tmp_path):
        """A row saved before the last_seen_at heartbeat existed (or before
        it was populated) must, after the migration in
        SQLiteBackend.initialize() runs, end up with
        last_seen_at == started_at. This proves the migration makes nothing
        MORE reap-able than it was before #82 landed."""
        db_path = str(tmp_path / "backfill.db")

        pre_migration_db = SQLiteBackend(db_path)
        await pre_migration_db.initialize()
        session = _session_dict(
            age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 20,
            last_seen_hours=None,
            project_name="proj-backfill",
        )
        await pre_migration_db.save_session(session)
        await pre_migration_db.close()

        # Re-open against the same file: this is where issue #82's
        # idempotent migration must run (ALTER TABLE ... ADD COLUMN
        # last_seen_at + UPDATE ... SET last_seen_at = started_at WHERE
        # last_seen_at IS NULL), mirroring the session_name / project_name
        # migrations already present in SQLiteBackend.initialize().
        migrated_db = SQLiteBackend(db_path)
        await migrated_db.initialize()
        stored = await migrated_db.get_session(session["id"])
        await migrated_db.close()

        assert stored is not None
        assert stored["last_seen_at"] == stored["started"], (
            "Issue #82: migration must backfill last_seen_at = started_at "
            "for pre-existing rows so COALESCE(last_seen_at, started_at) "
            "staleness checks are not MORE aggressive than before the "
            "migration for un-backfilled data."
        )
