"""
Regression tests for issue #57: reader offset pagination.

Covers the headline acceptance criteria:
- `offset: int = 0` was added as the last parameter of five readers
  (query_sessions, query_decisions_by_category, query_decisions_by_session,
  query_metrics_by_session, query_agent_executions) in base.py and both
  backends, each gaining a stable `id` tiebreaker in its ORDER BY.
- MigrationManager.migrate_all() no longer takes `scan_limit`; every
  `_migrate_*` now pages through MigrationManager._paginate() by offset
  until a short page, so batch_size no longer caps the total migrated.
- MAX_PAGES is the only remaining truncation path, tripped only by a
  reader that ignores `offset` and always returns a full page.

Match the house style: in-memory SQLite backend, asyncio_mode = "auto"
(from pyproject.toml) — no @pytest.mark.asyncio needed. See
tests/regression/test_issue_50_migrate_all_honest.py.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from persistence.migration import MigrationManager
from persistence.sqlite import SQLiteBackend
from tests.persistence.contract_tests import _agent_execution, _decision, _session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def source():
    """Source in-memory SQLite backend."""
    db = SQLiteBackend(db_path=":memory:")
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
async def target():
    """Target in-memory SQLite backend."""
    db = SQLiteBackend(db_path=":memory:")
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
def manager(source, target):
    return MigrationManager(source, target)


# ---------------------------------------------------------------------------
# Helper: drain a backend's offset-paginated query into a flat id list
# ---------------------------------------------------------------------------


async def _collect_all_session_ids(backend, page_size: int = 100) -> list[str]:
    """Page through backend.query_sessions(limit=page_size, offset=n) until a
    short page, returning the concatenated list of session ids seen."""
    ids: list[str] = []
    offset = 0
    while True:
        page = await backend.query_sessions(limit=page_size, offset=offset)
        if not page:
            break
        ids.extend(row["id"] for row in page)
        if len(page) < page_size:
            break
        offset += page_size
    return ids


# ---------------------------------------------------------------------------
# Headline: no more scan ceiling on migrate_all
# ---------------------------------------------------------------------------


async def test_sessions_beyond_one_page_are_all_migrated(source, target, manager):
    """250 sessions > batch_size=100 and there is no longer any scan_limit
    ceiling — migrate_all() must migrate all of them via offset pagination."""
    total = 250
    for i in range(total):
        await source.save_session(_session(session_id=f"sess-page-{i:04d}"))

    result = await manager.migrate_all(batch_size=100)

    assert result["records_migrated"]["sessions"] == total
    assert result["status"] == "success"


async def test_decisions_beyond_one_page_for_a_single_session_are_all_migrated(
    source, target, manager
):
    """250 decisions on a single session exceed batch_size=100 for the
    query_decisions_by_session fallback loop used for uncategorized
    decisions; all must still arrive in the target."""
    sid = "sess-many-decisions-001"
    await source.save_session(_session(session_id=sid))

    total = 250
    for i in range(total):
        await source.save_decision(_decision(session_id=sid, decision_id=f"dec-{i:04d}"))

    result = await manager.migrate_all(batch_size=100)

    assert result["status"] == "success"
    assert result["records_migrated"]["decisions"] == total
    target_decisions = await target.query_decisions_by_session(sid, limit=1000)
    assert len(target_decisions) == total


# ---------------------------------------------------------------------------
# Backend-level: offset pagination is gap-free and repeat-free
# ---------------------------------------------------------------------------


async def test_query_sessions_offset_paginates_without_gaps_or_repeats(source):
    """No migration involved: page directly through
    source.query_sessions(limit=100, offset=n) and confirm the ids collected
    across pages are exactly the 250 seeded rows with no duplicates."""
    total = 250
    for i in range(total):
        await source.save_session(_session(session_id=f"sess-offset-{i:04d}"))

    ids = await _collect_all_session_ids(source, page_size=100)

    assert len(ids) == total
    assert len(set(ids)) == total


async def test_query_sessions_pagination_is_stable_when_timestamps_collide(source):
    """The tiebreaker guard. All 250 seeded sessions share one identical
    started_at value. Without the `, id DESC` secondary sort key added by
    issue #57, ORDER BY started_at DESC alone gives SQLite no guaranteed
    order among tied rows across separate LIMIT/OFFSET queries, so paging
    could skip or repeat rows when timestamps collide. THIS test is the one
    that fails if the id tiebreaker is removed."""
    total = 250
    collision = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(total):
        await source.save_session(
            _session(session_id=f"sess-collide-{i:04d}", start_time=collision)
        )

    ids = await _collect_all_session_ids(source, page_size=100)

    assert len(ids) == total
    assert len(set(ids)) == total


# ---------------------------------------------------------------------------
# Agent executions
# ---------------------------------------------------------------------------


async def test_agent_executions_beyond_one_page_are_all_migrated(source, target, manager):
    """150 agent_execution rows on one session exceed batch_size=100; all
    must be migrated via the now-paginated query_agent_executions reader."""
    sid = "sess-agent-exec-001"
    await source.save_session(_session(session_id=sid))

    total = 150
    for i in range(total):
        await source.save_agent_execution(
            _agent_execution(session_id=sid, agent_name=f"agent-{i:04d}")
        )

    result = await manager.migrate_all(batch_size=100)

    assert result["records_migrated"]["agent_executions"] == total
    target_rows = await target.query_agent_executions(session_id=sid, limit=1000)
    assert len(target_rows) == total


# ---------------------------------------------------------------------------
# MAX_PAGES safety valve
# ---------------------------------------------------------------------------


async def test_paginate_safety_valve_reports_truncation_instead_of_hanging(manager, monkeypatch):
    """A reader that ignores `offset` and always returns a full page would
    make the offset-pagination loop spin forever. MAX_PAGES bounds that: it
    must terminate, mark manager.truncated True, and record a warning
    mentioning `offset`, instead of hanging. Locks in that removing
    scan_limit did not introduce an infinite loop."""
    monkeypatch.setattr(MigrationManager, "MAX_PAGES", 3)

    async def stub_reader(*args, limit, offset=0):
        return [{"id": "same-row"} for _ in range(limit)]

    async def _drain():
        rows = []
        async for row in manager._paginate("sessions", stub_reader, 2):
            rows.append(row)
        return rows

    rows = await asyncio.wait_for(_drain(), timeout=5)

    assert len(rows) == 6  # MAX_PAGES(3) * batch_size(2)
    assert manager.truncated is True
    assert any("offset" in w for w in manager.warnings)
