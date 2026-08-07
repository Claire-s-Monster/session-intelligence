"""
Unit tests for MigrationManager.

Tests SQLite-to-SQLite migration using two in-memory databases.
Uses adapter helpers from contract_tests to build properly-keyed dicts.

asyncio_mode = "auto" — no @pytest.mark.asyncio decorators needed.
"""

from __future__ import annotations

import pytest

from persistence.migration import MigrationManager
from persistence.sqlite import SQLiteBackend

# Re-use the adapter helpers defined in the contract test suite.
from tests.persistence.contract_tests import (
    _agent,
    _agent_decision,
    _agent_execution,
    _agent_learning,
    _agent_notebook,
    _decision,
    _mcp_session,
    _metrics,
    _note,
    _session,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def source(tmp_path):
    """Source SQLite backend (file-based for WAL safety)."""
    db = SQLiteBackend(str(tmp_path / "source.db"))
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
async def target(tmp_path):
    """Target SQLite backend (separate file)."""
    db = SQLiteBackend(str(tmp_path / "target.db"))
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
def manager(source, target):
    return MigrationManager(source, target)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_migrate_empty_source(manager):
    """Migrating an empty source returns success with zero records."""
    result = await manager.migrate_all()

    assert result["status"] == "success"
    assert result["total_records"] == 0
    assert result["records_migrated"]["sessions"] == 0
    assert result["records_migrated"]["decisions"] == 0
    assert result["records_migrated"]["metrics"] == 0
    assert result["records_migrated"]["notes"] == 0
    assert result["records_migrated"]["agent_executions"] == 0


async def test_migrate_sessions(source, target, manager):
    """A session saved to source appears in target after migration."""
    sid = "sess-migrate-001"
    await source.save_session(_session(session_id=sid))

    result = await manager.migrate_all()

    assert result["status"] == "success"
    assert result["records_migrated"]["sessions"] >= 1

    migrated = await target.get_session(sid)
    assert migrated is not None
    assert migrated["id"] == sid


async def test_migrate_decisions(source, target, manager):
    """Decisions linked to a session are migrated to the target."""
    sid = "sess-migrate-dec-001"
    await source.save_session(_session(session_id=sid))

    dec = _decision(session_id=sid, category="architecture")
    dec_id = dec["id"]
    await source.save_decision(dec)

    result = await manager.migrate_all()

    assert result["status"] == "success"
    assert result["records_migrated"]["decisions"] >= 1

    target_decisions = await target.query_decisions_by_session(sid)
    ids = [d["id"] for d in target_decisions]
    assert dec_id in ids


async def test_migrate_all_entity_types(source, target, manager):
    """All six entity types (sessions, decisions, metrics, notes,
    agent_executions, mcp_sessions) are migrated."""
    sid = "sess-migrate-all-001"
    await source.save_session(_session(session_id=sid))
    await source.save_decision(_decision(session_id=sid))
    await source.save_metrics(_metrics(session_id=sid))
    await source.save_note(_note(session_id=sid))
    await source.save_agent_execution(_agent_execution(session_id=sid))
    await source.save_mcp_session(_mcp_session(mcp_session_id="mcp-migrate-all-001"))

    result = await manager.migrate_all()

    rm = result["records_migrated"]
    assert rm["sessions"] >= 1
    assert rm["decisions"] >= 1
    assert rm["metrics"] >= 1
    assert rm["notes"] >= 1
    assert rm["agent_executions"] >= 1
    assert rm["mcp_sessions"] >= 1
    assert result["total_records"] >= 6


async def test_migrate_idempotent(source, target, manager):
    """Migrating twice does not duplicate sessions, decisions, or notes in
    the target."""
    sid = "sess-migrate-idem-001"
    await source.save_session(_session(session_id=sid))
    dec = _decision(session_id=sid, category="architecture")
    await source.save_decision(dec)
    await source.save_note(_note(session_id=sid))

    # First migration
    await manager.migrate_all()
    # Second migration — uses a fresh manager so stats reset
    manager2 = MigrationManager(source, target)
    await manager2.migrate_all()

    sessions = await target.query_sessions(limit=1000)
    matching = [s for s in sessions if s["id"] == sid]
    # INSERT OR REPLACE means exactly one row
    assert len(matching) == 1

    decisions = await target.query_decisions_by_session(sid, limit=1000)
    matching_decisions = [d for d in decisions if d["id"] == dec["id"]]
    assert len(matching_decisions) == 1

    notes = await target.query_notes(limit=1000)
    matching_notes = [n for n in notes if n["session_id"] == sid]
    assert len(matching_notes) == 1


async def test_migrate_preserves_data_integrity(source, target, manager):
    """Migrated session data matches the source exactly for key fields."""
    sid = "sess-integrity-001"
    src_data = _session(
        session_id=sid,
        project_path="/home/user/myproject",
        status="completed",
        mode="local",
    )
    await source.save_session(src_data)

    await manager.migrate_all()

    src_row = await source.get_session(sid)
    tgt_row = await target.get_session(sid)

    assert tgt_row is not None
    assert tgt_row["id"] == src_row["id"]
    assert tgt_row["project_path"] == src_row["project_path"]
    assert tgt_row["status"] == src_row["status"]
    assert tgt_row["mode"] == src_row["mode"]


async def test_migrate_multiple_sessions(source, target, manager):
    """All sessions present in source are migrated to target."""
    session_ids = [f"sess-multi-{i:03d}" for i in range(5)]
    for sid in session_ids:
        await source.save_session(_session(session_id=sid))

    result = await manager.migrate_all()

    assert result["records_migrated"]["sessions"] == 5
    for sid in session_ids:
        row = await target.get_session(sid)
        assert row is not None, f"Session {sid} missing from target"


async def test_migrate_decisions_by_category(source, target, manager):
    """Decisions from multiple categories are all migrated."""
    sid = "sess-cat-001"
    await source.save_session(_session(session_id=sid))

    categories = ["architecture", "implementation", "testing"]
    decision_ids = []
    for cat in categories:
        dec = _decision(session_id=sid, category=cat)
        decision_ids.append(dec["id"])
        await source.save_decision(dec)

    result = await manager.migrate_all()

    # _migrate_decisions collects by id before saving, so a decision found by
    # both the category loop and the uncategorized-session fallback is saved
    # exactly once.
    assert result["records_migrated"]["decisions"] == len(categories)
    target_decisions = await target.query_decisions_by_session(sid)
    target_ids = {d["id"] for d in target_decisions}
    for did in decision_ids:
        assert did in target_ids, f"Decision {did} missing from target"
