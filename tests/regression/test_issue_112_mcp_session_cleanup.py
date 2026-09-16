"""
Regression tests for issue #112: `cleanup_inactive_sessions` must accept
every representation of `last_activity` that can end up in
`MCPSessionManager._active_sessions`.

Bug: `_active_sessions` entries are populated from two different sources.
     `create_mcp_session`/`update_activity` store an aware UTC ISO string,
     but `get_engine_session_id`/`validate_session` cache the raw DB row
     returned by `database.get_mcp_session(...)` verbatim. A live probe
     showed the PostgreSQL backend returns `last_activity` as an aware
     `datetime` object (not a string), while SQLite returns a string that
     may be naive for legacy, pre-#112 rows. `cleanup_inactive_sessions`
     unconditionally called `datetime.fromisoformat(...)` on that value,
     so a rehydrated PostgreSQL session raised
     `TypeError: fromisoformat: argument must be str` and aborted the
     entire cleanup loop.
Fix: only call `datetime.fromisoformat` when the value is a `str`; pass
     `datetime` objects straight to `_as_aware_utc`.
"""

from datetime import UTC, datetime, timedelta

import pytest

from transport.mcp_session_manager import MCPSessionManager

pytestmark = pytest.mark.regression


def _session(last_activity):
    return {
        "mcp_session_id": "mcp-test",
        "engine_session_id": None,
        "created_at": last_activity,
        "last_activity": last_activity,
        "client_info": {},
    }


async def test_cleanup_accepts_datetime_last_activity_from_postgresql_rows():
    """PostgreSQL rows cache `last_activity` as an aware datetime object."""
    manager = MCPSessionManager(database=None)
    now = datetime.now(UTC)
    manager._active_sessions["old"] = _session(now - timedelta(hours=2))
    manager._active_sessions["fresh"] = _session(now - timedelta(minutes=1))

    removed = await manager.cleanup_inactive_sessions(max_age_seconds=3600)

    assert removed == 1
    assert "old" not in manager._active_sessions
    assert "fresh" in manager._active_sessions


async def test_cleanup_accepts_naive_legacy_string():
    """Naive local-time ISO strings from pre-#112 SQLite rows are handled."""
    manager = MCPSessionManager(database=None)
    stale_naive = (datetime.now() - timedelta(hours=2)).isoformat()
    manager._active_sessions["legacy"] = _session(stale_naive)

    removed = await manager.cleanup_inactive_sessions(max_age_seconds=3600)

    assert removed == 1
    assert "legacy" not in manager._active_sessions


async def test_cleanup_accepts_aware_iso_string():
    """Aware UTC ISO strings from `create_mcp_session`/`update_activity`."""
    manager = MCPSessionManager(database=None)
    fresh_aware = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    manager._active_sessions["fresh"] = _session(fresh_aware)

    removed = await manager.cleanup_inactive_sessions(max_age_seconds=3600)

    assert removed == 0
    assert "fresh" in manager._active_sessions


async def test_cleanup_handles_mixed_representations_in_one_pass():
    """All three representations coexist in `_active_sessions` at once."""
    manager = MCPSessionManager(database=None)
    now = datetime.now(UTC)

    manager._active_sessions["pg_old"] = _session(now - timedelta(hours=2))
    manager._active_sessions["pg_fresh"] = _session(now - timedelta(minutes=1))
    manager._active_sessions["legacy_old"] = _session(
        (datetime.now() - timedelta(hours=2)).isoformat()
    )
    manager._active_sessions["aware_fresh"] = _session(
        (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    )

    removed = await manager.cleanup_inactive_sessions(max_age_seconds=3600)

    assert removed == 2
    assert "pg_old" not in manager._active_sessions
    assert "legacy_old" not in manager._active_sessions
    assert "pg_fresh" in manager._active_sessions
    assert "aware_fresh" in manager._active_sessions
