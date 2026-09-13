"""
Regression tests for issue #106: session_summaries gained a caller-authored
``authored_body`` column, distinct from the regenerated ``summary_markdown``
snapshot.

Two persistence bugs would have made this dangerous if reverted:

1. SQLite's ``save_session_summary`` used ``INSERT OR REPLACE``, which
   deletes and re-inserts the whole row. Since routine notebook
   regeneration never supplies ``authored_body`` (it only refreshes the
   derived snapshot), an unconditional ``INSERT OR REPLACE`` would silently
   null out any previously-authored narrative on every regeneration. The
   fix converts SQLite to ``INSERT ... ON CONFLICT(session_id) DO UPDATE
   SET`` with ``authored_body = COALESCE(excluded.authored_body,
   session_summaries.authored_body)``, matching PostgreSQL's existing
   ON CONFLICT upsert.
2. A new ``update_session_summary_body`` method allows updating only the
   caller-authored fields (body/title) without touching the regenerated
   ``summary_markdown``/``key_changes``/``tags`` columns, and must not
   fabricate a row for a session that has no notebook yet.

On the engine side, ``session_create_notebook`` gained a ``body`` parameter
(caller-authored narrative, rendered verbatim) and an
``include_derived_sections`` parameter that independently controls whether
the compiled sections (agents, decisions, metrics, learnings, files) are
rendered into the markdown output. The default wiring makes derived
sections opt-out once a body is supplied (``render_sections = flag if flag
is not None else body is None``), but the section *data* is always
gathered regardless of whether it is rendered.

Match the house style: SQLite/PostgreSQL parity tests follow the
``PersistenceContractTests`` subclassing convention in
tests/persistence/contract_tests.py (see tests/persistence/test_sqlite_contract.py
and tests/persistence/test_postgresql_contract.py) -- PostgreSQL tests are
skipped when POSTGRES_DSN/asyncpg are unavailable. Engine-level rendering
tests use the SQLite-backed engine fixture pattern from
tests/regression/test_issue_103_notebook_cold_load.py. asyncio_mode = "auto"
(pyproject.toml) -- no @pytest.mark.asyncio decorators needed.

Distinctive literals (e.g. "AUTHORED-NARRATIVE-SENTINEL-...") are used
throughout so an assertion cannot pass by accident on incidental text.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend
from tests.persistence.conftest import POSTGRES_AVAILABLE

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _session_row(session_id: str, status: str = "active") -> dict:
    started = datetime.now(UTC).isoformat()
    return {
        "id": session_id,
        "started": started,
        "project_path": "/tmp/issue-106-project",
        "project_name": "issue-106-project",
        "mode": "local",
        "status": status,
        "metadata": {},
        "performance_metrics": {},
        "health_status": {},
    }


def _summary_row(session_id: str, **overrides) -> dict:
    defaults = {
        "session_id": session_id,
        "title": "Original Title",
        "summary_markdown": "# Original summary\n\nOriginal content.",
        "authored_body": None,
        "key_changes": ["src/original.py"],
        "tags": ["original-tag"],
        "created_at": datetime.now(UTC),
    }
    return {**defaults, **overrides}


def _as_list(value):
    """Normalise a key_changes/tags column value to a list.

    SQLite stores these as TEXT (JSON-encoded); PostgreSQL's JSONB columns
    come back from asyncpg as native lists for some fields and raw JSON
    strings for others (see persistence/postgresql.py::_from_record, which
    only parses a fixed field allowlist). Normalising here keeps the
    assertions about issue #106's collateral-nulling guard independent of
    that unrelated backend inconsistency.
    """
    if isinstance(value, str):
        return json.loads(value)
    return value or []


# ---------------------------------------------------------------------------
# Persistence parity tests (SQLite + PostgreSQL)
# ---------------------------------------------------------------------------


class _NotebookAuthoredBodyPersistenceTests:
    """Shared parity tests for issue #106 persistence behaviour.

    Subclass with a ``backend`` fixture (SQLite or PostgreSQL), matching
    the PersistenceContractTests convention in
    tests/persistence/contract_tests.py.
    """

    async def test_authored_body_round_trips(self, backend):
        """Pins: save_session_summary(authored_body=...) ->
        get_session_summary returns it verbatim."""
        sid = f"issue-106-roundtrip-{uuid.uuid4().hex[:8]}"
        await backend.save_session(_session_row(sid))
        await backend.save_session_summary(
            _summary_row(sid, authored_body="AUTHORED-NARRATIVE-SENTINEL-ROUNDTRIP")
        )

        result = await backend.get_session_summary(sid)

        assert result is not None
        assert result["authored_body"] == "AUTHORED-NARRATIVE-SENTINEL-ROUNDTRIP"

    async def test_authored_body_survives_regeneration_without_body(self, backend):
        """Pins the core case: a second save with authored_body absent and a
        different summary_markdown must NOT null the original body. This is
        the case that fails if sqlite reverts to INSERT OR REPLACE."""
        sid = f"issue-106-preserve-{uuid.uuid4().hex[:8]}"
        await backend.save_session(_session_row(sid))
        await backend.save_session_summary(
            _summary_row(
                sid,
                authored_body="AUTHORED-NARRATIVE-SENTINEL-PRESERVE",
                summary_markdown="# Original\n\nfirst pass",
            )
        )

        await backend.save_session_summary(
            _summary_row(
                sid,
                authored_body=None,
                summary_markdown="# Regenerated\n\nsecond pass",
            )
        )

        result = await backend.get_session_summary(sid)
        assert result is not None
        assert result["authored_body"] == "AUTHORED-NARRATIVE-SENTINEL-PRESERVE"
        assert result["summary_markdown"] == "# Regenerated\n\nsecond pass"

    async def test_authored_body_overwritten_when_new_body_supplied(self, backend):
        """Pins: saving again WITH a new authored_body replaces it."""
        sid = f"issue-106-overwrite-{uuid.uuid4().hex[:8]}"
        await backend.save_session(_session_row(sid))
        await backend.save_session_summary(
            _summary_row(sid, authored_body="AUTHORED-NARRATIVE-SENTINEL-OLD")
        )

        await backend.save_session_summary(
            _summary_row(sid, authored_body="AUTHORED-NARRATIVE-SENTINEL-NEW")
        )

        result = await backend.get_session_summary(sid)
        assert result is not None
        assert result["authored_body"] == "AUTHORED-NARRATIVE-SENTINEL-NEW"

    async def test_regeneration_does_not_null_key_changes_or_tags(self, backend):
        """Pins the INSERT OR REPLACE -> ON CONFLICT conversion specifically:
        after a body-preserving regeneration, key_changes and tags must
        still hold the values from that regeneration (no collateral
        nulling of unrelated columns)."""
        sid = f"issue-106-collateral-{uuid.uuid4().hex[:8]}"
        await backend.save_session(_session_row(sid))
        await backend.save_session_summary(
            _summary_row(
                sid,
                authored_body="AUTHORED-NARRATIVE-SENTINEL-COLLATERAL",
                key_changes=["src/keep_me.py"],
                tags=["keep-tag"],
            )
        )

        await backend.save_session_summary(
            _summary_row(
                sid,
                authored_body=None,
                summary_markdown="# Regenerated again",
                key_changes=["src/keep_me.py"],
                tags=["keep-tag"],
            )
        )

        result = await backend.get_session_summary(sid)
        assert result is not None
        assert result["authored_body"] == "AUTHORED-NARRATIVE-SENTINEL-COLLATERAL"
        assert _as_list(result["key_changes"]) == ["src/keep_me.py"]
        assert _as_list(result["tags"]) == ["keep-tag"]

    async def test_update_body_on_missing_summary_returns_false_and_creates_nothing(self, backend):
        """Pins: update_session_summary_body returns False for a session_id
        with no summary row, and does not create one."""
        sid = f"issue-106-missing-{uuid.uuid4().hex[:8]}"
        await backend.save_session(_session_row(sid))

        updated = await backend.update_session_summary_body(
            sid, authored_body="AUTHORED-NARRATIVE-SENTINEL-SHOULD-NOT-EXIST"
        )

        assert updated is False
        result = await backend.get_session_summary(sid)
        assert result is None

    async def test_update_body_with_both_args_none_returns_false(self, backend):
        """Pins: update_session_summary_body(authored_body=None, title=None)
        returns False and leaves the existing row untouched."""
        sid = f"issue-106-noop-{uuid.uuid4().hex[:8]}"
        await backend.save_session(_session_row(sid))
        await backend.save_session_summary(
            _summary_row(sid, authored_body="AUTHORED-NARRATIVE-SENTINEL-NOOP")
        )

        updated = await backend.update_session_summary_body(sid)

        assert updated is False
        result = await backend.get_session_summary(sid)
        assert result["authored_body"] == "AUTHORED-NARRATIVE-SENTINEL-NOOP"

    async def test_update_body_leaves_other_fields_byte_identical(self, backend):
        """Pins: update_session_summary_body(authored_body=...) updates only
        the body and leaves summary_markdown, key_changes and tags
        byte-identical."""
        sid = f"issue-106-scoped-update-{uuid.uuid4().hex[:8]}"
        await backend.save_session(_session_row(sid))
        original_markdown = "# Untouched\n\nThis summary_markdown must not change."
        await backend.save_session_summary(
            _summary_row(
                sid,
                authored_body="AUTHORED-NARRATIVE-SENTINEL-OLD-BODY",
                summary_markdown=original_markdown,
                key_changes=["src/untouched.py"],
                tags=["untouched-tag"],
            )
        )

        updated = await backend.update_session_summary_body(
            sid, authored_body="AUTHORED-NARRATIVE-SENTINEL-NEW-BODY"
        )

        assert updated is True
        result = await backend.get_session_summary(sid)
        assert result["authored_body"] == "AUTHORED-NARRATIVE-SENTINEL-NEW-BODY"
        assert result["summary_markdown"] == original_markdown
        assert _as_list(result["key_changes"]) == ["src/untouched.py"]
        assert _as_list(result["tags"]) == ["untouched-tag"]


class TestSQLiteNotebookAuthoredBody(_NotebookAuthoredBodyPersistenceTests):
    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "test_issue_106.db"))
        await db.initialize()
        yield db
        await db.close()


@pytest.mark.postgresql
@pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not available")
class TestPostgreSQLNotebookAuthoredBody(_NotebookAuthoredBodyPersistenceTests):
    @pytest.fixture
    async def backend(self):
        from persistence.postgresql import PostgreSQLBackend

        dsn = os.environ["POSTGRES_DSN"]
        db = PostgreSQLBackend(dsn=dsn)
        await db.initialize()
        yield db
        await db.close()


# ---------------------------------------------------------------------------
# Engine / rendering tests (SQLite-backed engine)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "test_issue_106_engine.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def engine(tmp_path, db):
    return SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )


async def test_no_body_renders_derived_sections_by_default(engine, db):
    """Backward-compatibility guard: with no body supplied, output is
    unchanged from current behaviour -- derived section headings are
    present."""
    sid = "issue-106-no-body"
    await db.save_session(_session_row(sid))
    engine.session_cache.clear()

    result = await engine.session_create_notebook(session_id=sid)

    assert result.status == "success"
    assert "## Performance Metrics" in result.markdown_output
    assert "## Overview" in result.markdown_output


async def test_body_supplied_without_flag_suppresses_derived_sections(engine, db):
    """body supplied, include_derived_sections not passed -> body text
    appears in markdown_output AND derived section headings do NOT."""
    sid = "issue-106-body-only"
    await db.save_session(_session_row(sid))
    engine.session_cache.clear()

    result = await engine.session_create_notebook(
        session_id=sid, body="AUTHORED-NARRATIVE-SENTINEL-RENDER"
    )

    assert result.status == "success"
    assert "AUTHORED-NARRATIVE-SENTINEL-RENDER" in result.markdown_output
    assert "## Performance Metrics" not in result.markdown_output


async def test_body_with_include_derived_sections_true_renders_both(engine, db):
    """body supplied with include_derived_sections=True -> BOTH the body and
    the derived headings appear."""
    sid = "issue-106-body-and-sections"
    await db.save_session(_session_row(sid))
    engine.session_cache.clear()

    result = await engine.session_create_notebook(
        session_id=sid,
        body="AUTHORED-NARRATIVE-SENTINEL-BOTH",
        include_derived_sections=True,
    )

    assert result.status == "success"
    assert "AUTHORED-NARRATIVE-SENTINEL-BOTH" in result.markdown_output
    assert "## Performance Metrics" in result.markdown_output


async def test_include_derived_sections_false_without_body_suppresses_sections(engine, db):
    """include_derived_sections=False with NO body -> sections suppressed,
    proving the flag is independent of body presence rather than merely
    correlated with it."""
    sid = "issue-106-flag-independent"
    await db.save_session(_session_row(sid))
    engine.session_cache.clear()

    result = await engine.session_create_notebook(session_id=sid, include_derived_sections=False)

    assert result.status == "success"
    assert "## Performance Metrics" not in result.markdown_output


async def test_notebook_model_carries_body_and_populated_sections_even_when_unrendered(engine, db):
    """The returned SessionNotebook.authored_body equals the supplied body,
    and sections are still POPULATED on the model even when not rendered
    (data gathering is unchanged; only rendering is gated)."""
    sid = "issue-106-data-vs-render"
    await db.save_session(_session_row(sid))
    engine.session_cache.clear()

    result = await engine.session_create_notebook(
        session_id=sid, body="AUTHORED-NARRATIVE-SENTINEL-DATA"
    )

    assert result.status == "success"
    assert result.notebook is not None
    assert result.notebook.authored_body == "AUTHORED-NARRATIVE-SENTINEL-DATA"
    assert len(result.notebook.sections) > 0
    # Sanity: confirm this is genuinely the unrendered case, not an
    # accidental default-True render.
    assert "## Performance Metrics" not in result.markdown_output


async def test_update_notebook_without_existing_notebook_errors(engine, db):
    """session_update_notebook on a session with no notebook returns
    status "error" and a message about creating one first -- NOT status
    success."""
    sid = "issue-106-no-notebook-yet"
    await db.save_session(_session_row(sid))

    result = await engine.session_update_notebook(
        session_id=sid, body="AUTHORED-NARRATIVE-SENTINEL-UPDATE"
    )

    assert result["status"] == "error"
    assert result["status"] != "success"
    assert "session_create_notebook" in result["message"]
