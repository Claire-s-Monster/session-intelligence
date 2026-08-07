"""
Regression tests for issue #50: make migrate_all honest.

Covers the headline acceptance criteria from the issue and its two review
comments (all line numbers verified against de9fff5):

- notes older than the old 365-day horizon are migrated
- a day with more than 1000 notes is migrated in full (paginated reader)
- re-running migrate_all does not duplicate notes
- an orphaned note (dangling session_id) is surfaced by the migrator instead
  of being silently invisible
- save_decision is idempotent on SQLite (previously PostgreSQL-only)
- migrate_all reports status="partial" with failure detail when an entity
  fails, instead of silently claiming "success"

Match the house style: in-memory SQLite backend, asyncio_mode = "auto" (from
pyproject.toml) — no @pytest.mark.asyncio needed. See
tests/regression/test_issue_53_decision_project_path.py.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from persistence.migration import MigrationManager
from persistence.sqlite import SQLiteBackend
from tests.persistence.contract_tests import _decision, _session


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
# Headline criterion: notes older than 365 days are migrated
# ---------------------------------------------------------------------------


async def test_note_older_than_365_days_is_migrated(source, target, manager):
    """A note dated well past the old 365-day scan horizon is still migrated,
    since _migrate_notes now paginates query_notes() instead of walking a
    fixed one-year window of individual dates."""
    sid = "sess-old-note-001"
    await source.save_session(_session(session_id=sid))

    old_date = (date.today() - timedelta(days=400)).isoformat()
    await source.save_note({
        "session_id": sid,
        "date": old_date,
        "content": "ancient note",
        "tags": [],
    })

    result = await manager.migrate_all()

    assert result["records_migrated"]["notes"] == 1
    target_notes = await target.query_notes(limit=10)
    assert any(n["content"] == "ancient note" for n in target_notes)


# ---------------------------------------------------------------------------
# A day with more than 1000 notes is migrated in full
# ---------------------------------------------------------------------------


async def test_more_than_1000_notes_in_a_day_migrated_in_full(source, target, manager):
    """The old code capped each day's notes at limit=1000 via
    query_notes_by_date. The paginated query_notes reader has no such cap."""
    sid = "sess-many-notes-001"
    await source.save_session(_session(session_id=sid))

    total = 1005
    same_date = "2026-01-01"
    for i in range(total):
        await source.save_note({
            "session_id": sid,
            "date": same_date,
            "content": f"note-{i}",
            "tags": [],
        })

    result = await manager.migrate_all(batch_size=500)

    assert result["status"] == "success"
    assert result["records_migrated"]["notes"] == total
    target_notes = await target.query_notes(limit=2000)
    assert len(target_notes) == total


# ---------------------------------------------------------------------------
# Re-running migrate_all does not duplicate notes
# ---------------------------------------------------------------------------


async def test_rerunning_migrate_all_does_not_duplicate_notes(source, target, manager):
    """save_note is now idempotent by id, so a second migration run does not
    produce duplicate rows in the target."""
    sid = "sess-rerun-notes-001"
    await source.save_session(_session(session_id=sid))
    for i in range(5):
        await source.save_note({
            "session_id": sid,
            "date": "2026-08-01",
            "content": f"note-{i}",
            "tags": [],
        })

    await manager.migrate_all()
    first_count = len(await target.query_notes(limit=100))

    manager2 = MigrationManager(source, target)
    await manager2.migrate_all()
    second_count = len(await target.query_notes(limit=100))

    assert first_count == 5
    assert second_count == first_count


# ---------------------------------------------------------------------------
# Orphaned note: dangling session_id
# ---------------------------------------------------------------------------


async def test_orphaned_note_surfaced_and_attempted(source, target, manager):
    """An orphaned note (session_id with no matching session anywhere) was
    previously invisible: query_notes_by_date INNER JOINs sessions, so the
    migrator's own reader could never see it. query_notes (Part A, no join)
    fixes that visibility gap.

    FINDING: SQLiteBackend.initialize() sets PRAGMA foreign_keys=ON
    (src/persistence/sqlite.py:331), and PostgreSQL's notes.session_id FK
    (src/persistence/postgresql.py) is not optional. So a note whose
    session does not exist ANYWHERE cannot actually be written to a target
    that enforces referential integrity — inserting it raises a foreign key
    violation regardless of how honest the migrator's bookkeeping is. The
    honest outcome is therefore a *reported* failure (status="partial", one
    notes failure counted, a specific warning message) rather than the old
    behavior of the note vanishing with no trace at all.
    """
    sid = "sess-orphan-001"
    await source.save_session(_session(session_id=sid))
    await source.save_note({
        "session_id": sid,
        "date": "2026-05-01",
        "content": "orphan note",
        "tags": [],
    })

    # Break referential integrity in the source on purpose: disable FK
    # enforcement just long enough to delete the session directly (bypassing
    # delete_session's manual cascade), leaving the note dangling.
    conn = source._connection
    await conn.execute("PRAGMA foreign_keys=OFF")
    await conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    await conn.commit()
    await conn.execute("PRAGMA foreign_keys=ON")

    assert await source.get_session(sid) is None  # confirmed gone from source

    # Part A fix: the non-joined reader surfaces the orphan...
    source_notes = await source.query_notes(limit=100)
    assert any(n["content"] == "orphan note" for n in source_notes)
    # ...while the old joined reader still cannot see it (unchanged contract).
    by_date = await source.query_notes_by_date("2026-05-01", limit=100)
    assert not any(n["content"] == "orphan note" for n in by_date)

    result = await manager.migrate_all()

    assert result["status"] == "partial"
    assert result["failed"]["notes"] == 1
    assert any("Failed to migrate note" in w for w in result["warnings"])
    target_notes = await target.query_notes(limit=100)
    assert not any(n["content"] == "orphan note" for n in target_notes)


# ---------------------------------------------------------------------------
# save_decision idempotence on SQLite
# ---------------------------------------------------------------------------


async def test_save_decision_twice_same_id_updates_not_duplicates_sqlite(source):
    """save_decision called twice with the same id produces one row on
    SQLite (updated, not duplicated) — matching PostgreSQL's existing
    ON CONFLICT (id) DO UPDATE behaviour."""
    sid = "sess-decision-idem-001"
    await source.save_session(_session(session_id=sid))

    dec = _decision(session_id=sid, decision_id="dec-fixed-001")
    await source.save_decision(dec)

    updated = dict(dec)
    updated["description"] = "updated description"
    await source.save_decision(updated)

    rows = await source.query_decisions_by_session(sid, limit=10)
    assert len(rows) == 1
    assert rows[0]["description"] == "updated description"


# ---------------------------------------------------------------------------
# Honest failure reporting
# ---------------------------------------------------------------------------


async def test_batch_size_does_not_cap_total_records_migrated(source, target, manager):
    """Regression guard: `batch_size` must never act as a hard cap on the
    total number of records migrated. `migrate_all()` with its DEFAULT
    arguments (batch_size=100, scan_limit=10000) must migrate all 150
    sessions and all 150 notes seeded here — well above batch_size=100 —
    with nothing truncated at the much larger scan_limit."""
    total = 150
    for i in range(total):
        sid = f"sess-batch-cap-{i:03d}"
        await source.save_session(_session(session_id=sid))
        await source.save_note({
            "session_id": sid,
            "date": "2026-08-01",
            "content": f"note-{i}",
            "tags": [],
        })

    result = await manager.migrate_all()

    assert result["records_migrated"]["sessions"] == total
    assert result["records_migrated"]["notes"] == total
    assert result["status"] == "success"


async def test_migrate_all_reports_partial_status_on_entity_failure(
    source, target, manager, monkeypatch
):
    """When a save to the target raises, migrate_all must report
    status != "success" with the failure surfaced in the result, instead of
    swallowing the exception and claiming success."""
    sid = "sess-failure-001"
    await source.save_session(_session(session_id=sid))

    async def failing_save_session(session_data):
        raise RuntimeError("simulated target failure")

    monkeypatch.setattr(target, "save_session", failing_save_session)

    result = await manager.migrate_all()

    assert result["status"] == "partial"
    assert result["failed"]["sessions"] == 1
    assert any("simulated target failure" in w for w in result["warnings"])
    # records_migrated / total_records keys are preserved for existing callers
    assert "records_migrated" in result
    assert "total_records" in result
