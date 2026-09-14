"""
Regression tests: ``NotebookResult.search_indexed`` must report what
actually happened during ``session_create_notebook_async`` instead of a
hardcoded ``True``.

Prior to this fix, the async notebook-creation path
(``session_engine.session_create_notebook_async``, the method actually
wired to the MCP tool in ``lean_mcp_interface.py``) always returned
``search_indexed=True`` regardless of whether ``save_to_database`` was
requested, whether a database was configured, or whether the persist call
actually succeeded. Live reproduction: calling with
``save_to_database=False, save_to_file=False`` returned ``file_path: null``
(honest) alongside ``search_indexed: true`` (dishonest).

The sync path (``session_create_notebook`` -> ``_create_notebook_impl``)
already computes ``search_indexed`` via a local variable and is
deliberately left untouched by this fix -- it is out of scope and always
False today because it never persists.

Style follows tests/regression/test_issue_106_authored_body.py: SQLite-backed
engine fixture, asyncio_mode = "auto" (no @pytest.mark.asyncio needed).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


def _session_row(session_id: str, status: str = "active") -> dict:
    started = datetime.now(UTC).isoformat()
    return {
        "id": session_id,
        "started": started,
        "project_path": "/tmp/issue-search-indexed-project",
        "project_name": "issue-search-indexed-project",
        "mode": "local",
        "status": status,
        "metadata": {},
        "performance_metrics": {},
        "health_status": {},
    }


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "test_issue_search_indexed.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def engine(tmp_path, db):
    return SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )


async def test_save_to_database_false_reports_not_indexed(engine, db):
    """save_to_database=False -> search_indexed is False (no persist attempt
    was made, so the field must not claim otherwise)."""
    sid = "issue-search-indexed-no-db-save"
    await db.save_session(_session_row(sid))
    engine.session_cache.clear()

    result = await engine.session_create_notebook_async(
        session_id=sid, save_to_database=False, save_to_file=False
    )

    assert result.status == "success"
    assert result.search_indexed is False


async def test_successful_save_reports_indexed(engine, db):
    """save_to_database=True with a real backend and a successful
    save_session_summary -> search_indexed is True."""
    sid = "issue-search-indexed-successful-save"
    await db.save_session(_session_row(sid))
    engine.session_cache.clear()

    result = await engine.session_create_notebook_async(
        session_id=sid, save_to_database=True, save_to_file=False
    )

    assert result.status == "success"
    assert result.search_indexed is True

    # Sanity: it was genuinely indexed, not just flagged as such.
    stored = await db.get_session_summary(sid)
    assert stored is not None


async def test_save_session_summary_raising_reports_not_indexed(engine, db, monkeypatch):
    """If save_session_summary raises, the async path's outer try/except
    catches it and returns a status="error" NotebookResult -- whose
    search_indexed defaults to False on the model. This pins that the
    error path never reaches the search_indexed=True line."""
    sid = "issue-search-indexed-raises"
    await db.save_session(_session_row(sid))
    engine.session_cache.clear()

    async def _raise(*args, **kwargs):
        raise RuntimeError("simulated persistence failure")

    monkeypatch.setattr(db, "save_session_summary", _raise)

    result = await engine.session_create_notebook_async(
        session_id=sid, save_to_database=True, save_to_file=False
    )

    assert result.search_indexed is False
