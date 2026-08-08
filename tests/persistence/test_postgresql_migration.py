"""PostgreSQL-target migration tests (issue #56).

Every pre-existing migration test is SQLite -> SQLite, so the PostgreSQL-only
branches of the migrator and backend were exercised by nothing. The one that
matters most is ``resync_notes_sequence()``: ``notes.id`` is SERIAL, and the
explicit-id upsert that ``save_note`` uses to stay idempotent does not advance
the backing sequence. Without a resync, migration itself succeeds and the
*first ordinary note write afterwards* fails on a duplicate key -- a failure
that surfaces far from its cause.

``_migrate_notes`` reaches the resync through ``getattr``, so on the
SQLite -> SQLite path it is silently skipped. These tests run against a real
PostgreSQL server so it actually executes.

Isolation note: each test gets its own throwaway database rather than reusing
the one in POSTGRES_DSN. That is not tidiness. ``save_note``'s explicit-id path
is ``ON CONFLICT (id) DO UPDATE``, so seeding ids into a shared database would
overwrite existing notes rows -- destructive if POSTGRES_DSN ever points at a
real database. A fresh database also puts the sequence at 1, which is what
makes the collision assertions deterministic.
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest

from tests.persistence.conftest import POSTGRES_AVAILABLE
from tests.persistence.contract_tests import _note, _session

pytestmark = [
    pytest.mark.postgresql,
    pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not available"),
]


# ---------------------------------------------------------------------------
# Throwaway-database helpers
# ---------------------------------------------------------------------------


def _with_database(dsn: str, name: str) -> str:
    """Return ``dsn`` repointed at database ``name``."""
    return urlunsplit(urlsplit(dsn)._replace(path=f"/{name}"))


async def _create_scratch_database(admin_dsn: str, name: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(admin_dsn)
    try:
        # CREATE DATABASE cannot run inside a transaction block; asyncpg's
        # execute() is autocommit outside an explicit transaction.
        await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()


async def _drop_scratch_database(admin_dsn: str, name: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(admin_dsn)
    try:
        try:
            # FORCE (PostgreSQL 13+) evicts any connection the pool left behind.
            await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        except asyncpg.PostgresSyntaxError:
            await conn.execute(f'DROP DATABASE IF EXISTS "{name}"')
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def pg_backend():
    """A PostgreSQL backend on its own freshly created database."""
    from persistence.postgresql import PostgreSQLBackend

    admin_dsn = os.environ["POSTGRES_DSN"]
    name = f"si_test_{uuid.uuid4().hex[:12]}"

    await _create_scratch_database(admin_dsn, name)
    db = PostgreSQLBackend(dsn=_with_database(admin_dsn, name))
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()
        await _drop_scratch_database(admin_dsn, name)


@pytest.fixture
async def sqlite_source(tmp_path):
    """Source SQLite backend (file-based, matching tests/unit/test_migration.py)."""
    from persistence.sqlite import SQLiteBackend

    db = SQLiteBackend(str(tmp_path / "source.db"))
    await db.initialize()
    yield db
    await db.close()


async def _seed_notes(backend, session_id: str, count: int) -> list[dict]:
    """Save a session plus ``count`` notes, returning the stored note rows."""
    await backend.save_session(_session(session_id=session_id))
    for i in range(count):
        await backend.save_note(_note(session_id=session_id, content=f"note {i}"))
    return await backend.query_notes(limit=100, offset=0)


# ---------------------------------------------------------------------------
# The regression guard for the sequence desync
# ---------------------------------------------------------------------------


async def test_ordinary_save_note_succeeds_after_migration_preserved_ids(
    sqlite_source, pg_backend
):
    """The acceptance criterion: a normal note write after a migration works.

    Without resync_notes_sequence() this raises UniqueViolationError -- see the
    negative control below, which asserts exactly that failure.
    """
    from persistence.migration import MigrationManager

    sid = "sess-issue-56-resync"
    source_notes = await _seed_notes(sqlite_source, sid, 5)
    source_ids = sorted(n["id"] for n in source_notes)
    assert source_ids == [1, 2, 3, 4, 5], "source ids should start at 1"

    result = await MigrationManager(sqlite_source, pg_backend).migrate_all()

    assert result["status"] == "success"
    assert result["records_migrated"]["notes"] == 5

    migrated = await pg_backend.query_notes(limit=100, offset=0)
    assert sorted(n["id"] for n in migrated) == source_ids, "ids must be preserved"

    # The write that would fail if the sequence were still sitting at 1.
    await pg_backend.save_note(_note(session_id=sid, content="written after migration"))

    after = await pg_backend.query_notes(limit=100, offset=0)
    assert len(after) == 6
    new_ids = {n["id"] for n in after} - set(source_ids)
    assert new_ids == {6}, "the new note should take the id after the migrated maximum"


async def test_explicit_id_inserts_without_resync_collide(pg_backend):
    """Negative control: proves the resync in the test above is load-bearing.

    This is the production failure mode from issue #56 reproduced directly --
    explicit-id inserts leave the SERIAL sequence at 1, so the next
    auto-assigned insert collides with an already-present row.
    """
    import asyncpg

    sid = "sess-issue-56-no-resync"
    await pg_backend.save_session(_session(session_id=sid))

    for note_id in range(1, 6):
        note = _note(session_id=sid, content=f"preserved {note_id}")
        note["id"] = note_id
        await pg_backend.save_note(note)

    # No resync_notes_sequence() call here -- the sequence is untouched.
    with pytest.raises(asyncpg.exceptions.UniqueViolationError):
        await pg_backend.save_note(_note(session_id=sid, content="collides"))


async def test_resync_notes_sequence_recovers_from_the_collision(pg_backend):
    """resync_notes_sequence() makes the previously failing write succeed."""
    sid = "sess-issue-56-recover"
    await pg_backend.save_session(_session(session_id=sid))

    for note_id in range(1, 6):
        note = _note(session_id=sid, content=f"preserved {note_id}")
        note["id"] = note_id
        await pg_backend.save_note(note)

    await pg_backend.resync_notes_sequence()

    await pg_backend.save_note(_note(session_id=sid, content="after resync"))
    rows = await pg_backend.query_notes(limit=100, offset=0)
    assert {n["id"] for n in rows} == {1, 2, 3, 4, 5, 6}


# NB: do not rename this to `test_` + exactly 35 [a-z0-9_] characters.
# That is the shape of a Lob test-mode API key, and TruffleHog's Lob
# detector flags such a name as a *verified* secret, failing CI secret
# scanning. The previous name hit it exactly (see PR #59).
async def test_repeated_resync_does_not_burn_ids(pg_backend):
    """Calling it repeatedly must not burn ids or skip ahead."""
    sid = "sess-issue-56-idempotent"
    await pg_backend.save_session(_session(session_id=sid))

    note = _note(session_id=sid, content="preserved 1")
    note["id"] = 1
    await pg_backend.save_note(note)

    await pg_backend.resync_notes_sequence()
    await pg_backend.resync_notes_sequence()
    await pg_backend.resync_notes_sequence()

    await pg_backend.save_note(_note(session_id=sid, content="next"))
    rows = await pg_backend.query_notes(limit=100, offset=0)
    assert {n["id"] for n in rows} == {1, 2}


async def test_resync_notes_sequence_on_empty_table(pg_backend):
    """COALESCE(MAX(id), 0) + 1 must leave an empty table starting at 1."""
    sid = "sess-issue-56-empty"
    await pg_backend.save_session(_session(session_id=sid))

    await pg_backend.resync_notes_sequence()

    await pg_backend.save_note(_note(session_id=sid, content="first ever note"))
    rows = await pg_backend.query_notes(limit=100, offset=0)
    assert [n["id"] for n in rows] == [1]


# ---------------------------------------------------------------------------
# PostgreSQL-only DATE handling
# ---------------------------------------------------------------------------


async def test_save_note_coerces_iso_date_string_to_date_column(pg_backend):
    """notes.date is DATE; SQLite stores TEXT, so only PostgreSQL hits this."""
    sid = "sess-issue-56-date"
    await pg_backend.save_session(_session(session_id=sid))

    note = _note(session_id=sid, content="dated note")
    note["date"] = "2026-08-07"
    await pg_backend.save_note(note)

    rows = await pg_backend.query_notes(limit=100, offset=0)
    assert len(rows) == 1
    stored = rows[0]["date"]
    assert isinstance(stored, str), "query_notes normalises DATE back to str"
    assert stored == "2026-08-07"


async def test_query_notes_by_date_normalises_date_to_str(pg_backend):
    """The same normalisation on the by-date reader."""
    sid = "sess-issue-56-by-date"
    await pg_backend.save_session(_session(session_id=sid))

    note = _note(session_id=sid, content="findable by date")
    note["date"] = "2026-08-07"
    await pg_backend.save_note(note)

    rows = await pg_backend.query_notes_by_date("2026-08-07", limit=10)
    assert len(rows) == 1
    assert rows[0]["date"] == "2026-08-07"
    assert isinstance(rows[0]["date"], str)


# ---------------------------------------------------------------------------
# End-to-end SQLite -> PostgreSQL
# ---------------------------------------------------------------------------


async def test_migrate_all_sqlite_to_postgres_is_clean(sqlite_source, pg_backend):
    """A full cross-backend migration reports success and lands every row."""
    from persistence.migration import MigrationManager
    from tests.persistence.contract_tests import _decision

    sid = "sess-issue-56-e2e"
    await sqlite_source.save_session(_session(session_id=sid))
    await sqlite_source.save_decision(_decision(session_id=sid, description="cross-backend"))
    await sqlite_source.save_note(_note(session_id=sid, content="cross-backend note"))

    result = await MigrationManager(sqlite_source, pg_backend).migrate_all()

    assert result["status"] == "success", result.get("warnings")
    assert result["records_migrated"]["sessions"] >= 1
    assert result["records_migrated"]["decisions"] >= 1
    assert result["records_migrated"]["notes"] >= 1

    migrated_session = await pg_backend.get_session(sid)
    assert migrated_session is not None
    assert migrated_session["id"] == sid


async def test_migrate_all_is_idempotent_against_postgres(sqlite_source, pg_backend):
    """Re-running a migration must not duplicate notes or break the sequence."""
    from persistence.migration import MigrationManager

    sid = "sess-issue-56-twice"
    await _seed_notes(sqlite_source, sid, 3)

    first = await MigrationManager(sqlite_source, pg_backend).migrate_all()
    second = await MigrationManager(sqlite_source, pg_backend).migrate_all()

    assert first["status"] == "success"
    assert second["status"] == "success"

    rows = await pg_backend.query_notes(limit=100, offset=0)
    assert sorted(n["id"] for n in rows) == [1, 2, 3], "upsert by id, not re-insert"

    # And an ordinary write still works after the second pass.
    await pg_backend.save_note(_note(session_id=sid, content="after second migration"))
    rows = await pg_backend.query_notes(limit=100, offset=0)
    assert sorted(n["id"] for n in rows) == [1, 2, 3, 4]
