"""
Regression tests for issue #69: nothing ever transitioned an abandoned
session out of status='active'. Sessions only became 'completed' via an
explicit finalize, so any Claude session ending without one stayed 'active'
forever. get_active_session_for_project (and find_recent_session_by_project)
then loaded a MONTHS-OLD row as "the" active session for a project, silently
attaching new work to a stale session's lineage.

Fix:
- New SessionStatus.ABANDONED, distinct from COMPLETED -- the data stays
  honest about never having been finalized.
- get_active_session_for_project / find_recent_session_by_project(status=
  "active") now exclude sessions older than get_session_max_age_hours()
  (default 24h, overridable via SESSION_INTELLIGENCE_SESSION_MAX_AGE_HOURS).
- New reap_abandoned_sessions() flips stale 'active' rows to 'abandoned' and
  is called once at HTTP transport startup.

Uses the SQLite backend directly, following the existing regression-test
style (see test_issue_77_finalize_scope_binding.py).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from persistence.base import DEFAULT_SESSION_MAX_AGE_HOURS
from persistence.sqlite import SQLiteBackend


def _session_dict(
    *,
    status: str = "active",
    age_hours: float = 0.0,
    project_path: str = "/tmp/proj",
    project_name: str = "proj",
) -> dict:
    """Build a raw session dict ready for SQLiteBackend.save_session.

    age_hours controls how far in the past started_at is set, so tests can
    place a session on either side of the staleness threshold.
    """
    sid = f"session-{uuid.uuid4().hex[:8]}"
    started_at = datetime.now(UTC) - timedelta(hours=age_hours)
    return {
        "id": sid,
        "started_at": started_at.isoformat(),
        "ended_at": None,
        "project_path": project_path,
        "project_name": project_name,
        "mode": "local",
        "status": status,
        "metadata": {},
        "performance_metrics": {},
        "health_status": {},
    }


@pytest.mark.regression
class TestGetActiveSessionForProjectExcludesStale:
    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db
        await db.close()

    async def test_stale_active_session_not_returned(self, backend):
        stale = _session_dict(age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 1)
        await backend.save_session(stale)

        result = await backend.get_active_session_for_project(stale["project_path"])

        assert result is None, (
            "Issue #69: a stale 'active' session must not be resurrected as "
            "the current session for its project."
        )

    async def test_fresh_active_session_is_returned(self, backend):
        fresh = _session_dict(age_hours=1)
        await backend.save_session(fresh)

        result = await backend.get_active_session_for_project(fresh["project_path"])

        assert result is not None
        assert result["id"] == fresh["id"]


@pytest.mark.regression
class TestFindRecentSessionByProjectExcludesStale:
    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db
        await db.close()

    async def test_stale_active_session_excluded(self, backend):
        stale = _session_dict(
            age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 1, project_name="proj-x"
        )
        await backend.save_session(stale)

        result = await backend.find_recent_session_by_project("proj-x", status="active")

        assert result is None, (
            "find_recent_session_by_project must apply the same staleness "
            "guard as get_active_session_for_project, or the bug is "
            "reintroduced by a second path."
        )

    async def test_fresh_active_session_returned(self, backend):
        fresh = _session_dict(age_hours=1, project_name="proj-y")
        await backend.save_session(fresh)

        result = await backend.find_recent_session_by_project("proj-y", status="active")

        assert result is not None
        assert result["id"] == fresh["id"]

    async def test_non_active_status_lookup_unaffected_by_staleness(self, backend):
        old_completed = _session_dict(
            status="completed",
            age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 100,
            project_name="proj-z",
        )
        await backend.save_session(old_completed)

        result = await backend.find_recent_session_by_project("proj-z", status="completed")

        assert result is not None
        assert result["id"] == old_completed["id"]


@pytest.mark.regression
class TestReapAbandonedSessions:
    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        yield db
        await db.close()

    async def test_reap_flips_only_stale_active_rows(self, backend):
        stale = _session_dict(age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 1, project_name="p1")
        fresh = _session_dict(age_hours=1, project_name="p2")
        stale_but_completed = _session_dict(
            status="completed",
            age_hours=DEFAULT_SESSION_MAX_AGE_HOURS + 100,
            project_name="p3",
        )
        await backend.save_session(stale)
        await backend.save_session(fresh)
        await backend.save_session(stale_but_completed)

        reaped = await backend.reap_abandoned_sessions()

        assert reaped == 1, "Only the stale 'active' row should be reaped."

        stale_row = await backend.get_session(stale["id"])
        fresh_row = await backend.get_session(fresh["id"])
        completed_row = await backend.get_session(stale_but_completed["id"])

        assert stale_row["status"] == "abandoned", (
            "Issue #69: reaped rows must become 'abandoned', NOT 'completed' "
            "-- the distinction is the point: this session was never "
            "explicitly finalized."
        )
        assert fresh_row["status"] == "active"
        assert completed_row["status"] == "completed"

    async def test_reap_respects_custom_older_than_hours(self, backend):
        borderline = _session_dict(age_hours=5, project_name="p4")
        await backend.save_session(borderline)

        # Threshold tighter than default: a 5h-old session now counts as stale.
        reaped = await backend.reap_abandoned_sessions(older_than_hours=1)

        assert reaped == 1
        row = await backend.get_session(borderline["id"])
        assert row["status"] == "abandoned"

    async def test_reap_returns_zero_when_nothing_stale(self, backend):
        fresh = _session_dict(age_hours=1, project_name="p5")
        await backend.save_session(fresh)

        reaped = await backend.reap_abandoned_sessions()

        assert reaped == 0
        row = await backend.get_session(fresh["id"])
        assert row["status"] == "active"
