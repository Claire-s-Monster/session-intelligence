"""
Regression tests for the notebook cluster: issues #125, #126, #127, #128,
plus the FK persistence gap in ``_resolve_session_context`` fixed alongside
them.

Summary of the bugs pinned here:

1. Issue #125 — a dead synchronous notebook-generation path
   (``session_create_notebook``, ``_create_notebook_impl``,
   ``_generate_learnings_section``, ``_generate_files_section``) was removed
   in favour of the async implementations. The MCP registry entry named
   ``"session_create_notebook"`` must dispatch to
   ``session_create_notebook_async``.
2. Issue #126 — ``session_query_notebooks`` overflowed client tool-result
   caps by returning full rows (``summary_markdown``/``authored_body``/
   ``key_changes``) by default. It now projects down to a lightweight
   summary unless ``summary_only=False`` is passed.
3. Issue #127 — two bugs: (a) ``session_update_notebook(project_name=...)``
   resolved to the most-recent *active* session, which may not be the one
   that actually owns a notebook, so it must instead resolve to the most
   recent session with a notebook; (b) the lean MCP ``execute_tool``
   envelope always reported the outer ``status`` as ``"success"`` even when
   the inner tool result signalled ``status == "error"``.
4. Issue #128 — two bugs: (a) ``include_derived_sections=False`` only
   suppressed *rendering* of the compiled sections, not their *generation*,
   so DB queries (e.g. ``query_project_learnings``) still ran even when the
   caller asked to skip them; (b) ``file_status`` was inferred from a
   possibly-null ``file_path`` instead of being reported explicitly, which
   conflated "not requested", "filesystem disabled", and "write failed"
   into the same signal.
5. FK persistence gap (fixed alongside this cluster) — ``_create_session``
   only writes to ``self.session_cache`` (and, optionally, the filesystem);
   it never inserts a row into the ``sessions`` table. Every auto-create
   branch in ``_resolve_session_context`` (session_name, project_name, and
   the legacy unbound fallback) called ``_create_session`` directly without
   persisting to the database, so a later ``save_session_summary`` failed
   ``FOREIGN KEY constraint failed`` (``session_summaries.session_id``
   REFERENCES ``sessions(id)``).

Follows the fixture/style conventions of
tests/regression/test_issue_106_authored_body.py (SQLite-backed engine
fixture, asyncio_mode = "auto" from pyproject.toml -- no
@pytest.mark.asyncio decorators needed) and
tests/engine/test_mcp_tool_dispatch.py (the ``_get_meta_tool``/
``_extract_session_id`` helpers for driving the lean MCP interface
directly).
"""

from __future__ import annotations

import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.sqlite import SQLiteBackend

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "test_issue_125_128.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def engine(tmp_path, db):
    return SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )


def _get_meta_tool(interface: LeanMCPInterface, name: str):
    """Return the callable registered as a FastMCP tool by name.

    Mirrors tests/engine/test_mcp_tool_dispatch.py::_get_meta_tool.
    """
    manager = interface.app._tool_manager
    tool_obj = manager._tools[name]
    return tool_obj.fn


def _extract_session_id(create_result: dict) -> str:
    """Pull session_id out of a session_manage_lifecycle(create) envelope.

    Mirrors tests/engine/test_mcp_tool_dispatch.py::_extract_session_id.
    """
    inner = create_result["result"]
    if hasattr(inner, "session_id"):
        return inner.session_id
    return inner.get("session_id") or inner.get("result", {}).get("session_id")


# ---------------------------------------------------------------------------
# Issue #125: dead sync notebook path removed
# ---------------------------------------------------------------------------


class TestIssue125DeadSyncPathRemoved:
    def test_dead_sync_attributes_absent(self, engine):
        assert not hasattr(engine, "session_create_notebook")
        assert not hasattr(engine, "_create_notebook_impl")
        assert not hasattr(engine, "_generate_learnings_section")
        assert not hasattr(engine, "_generate_files_section")

    def test_async_replacements_still_present(self, engine):
        assert hasattr(engine, "session_create_notebook_async")
        assert hasattr(engine, "_generate_learnings_section_async")

    def test_registry_wires_session_create_notebook_to_async_impl(self, engine):
        interface = LeanMCPInterface(engine)
        tool_info = interface.tool_registry["session_create_notebook"]
        # _wrap_async_tool uses functools.wraps, which sets __wrapped__ to
        # the exact bound method it wrapped.
        wrapped = tool_info["implementation"].__wrapped__
        assert wrapped == engine.session_create_notebook_async


# ---------------------------------------------------------------------------
# Issue #128a: include_derived_sections=False skips GENERATION, not just
# rendering
# ---------------------------------------------------------------------------


class TestIssue128GenerationGatedNotRenderingOnly:
    async def test_include_derived_sections_false_skips_generation_and_query(
        self, engine, db, monkeypatch
    ):
        create = await engine.session_manage_lifecycle(
            operation="create", project_name="issue128-proj-off"
        )
        session_id = create.session_id
        await engine.session_log_decision(
            decision="Adopted the async notebook path", session_id=session_id
        )
        await engine.session_log_learning(
            category="pattern",
            learning_content="Async notebook generation must gate DB queries.",
            session_id=session_id,
            project_name="issue128-proj-off",
        )

        calls = {"n": 0}
        original_query = db.query_project_learnings

        async def spy(*args, **kwargs):
            calls["n"] += 1
            return await original_query(*args, **kwargs)

        monkeypatch.setattr(db, "query_project_learnings", spy)

        result = await engine.session_create_notebook_async(
            session_id=session_id,
            body="AMENDED-BODY-128-OFF",
            include_derived_sections=False,
        )

        assert result.status == "success"
        assert result.notebook is not None
        assert result.notebook.sections == []
        assert calls["n"] == 0

    async def test_include_derived_sections_true_generates_and_queries(
        self, engine, db, monkeypatch
    ):
        """Mirror case: True DOES populate sections and DOES query learnings."""
        create = await engine.session_manage_lifecycle(
            operation="create", project_name="issue128-proj-on"
        )
        session_id = create.session_id
        await engine.session_log_decision(
            decision="Adopted the async notebook path", session_id=session_id
        )
        await engine.session_log_learning(
            category="pattern",
            learning_content="Async notebook generation must gate DB queries.",
            session_id=session_id,
            project_name="issue128-proj-on",
        )

        calls = {"n": 0}
        original_query = db.query_project_learnings

        async def spy(*args, **kwargs):
            calls["n"] += 1
            return await original_query(*args, **kwargs)

        monkeypatch.setattr(db, "query_project_learnings", spy)

        result = await engine.session_create_notebook_async(
            session_id=session_id,
            include_derived_sections=True,
        )

        assert result.status == "success"
        assert result.notebook is not None
        assert len(result.notebook.sections) > 0
        assert calls["n"] >= 1


# ---------------------------------------------------------------------------
# Issue #128b: file_status is reported explicitly
# ---------------------------------------------------------------------------


class TestIssue128FileStatus:
    async def test_file_status_written_with_real_path_when_filesystem_enabled(self, tmp_path, db):
        fs_engine = SessionIntelligenceEngine(
            repository_path=str(tmp_path), use_filesystem=True, database=db
        )
        create = await fs_engine.session_manage_lifecycle(
            operation="create", project_name="issue128-fs-on"
        )
        session_id = create.session_id

        result = await fs_engine.session_create_notebook_async(
            session_id=session_id, save_to_file=True
        )

        assert result.status == "success"
        assert result.file_status == "written"
        assert result.file_path is not None

    async def test_file_status_reports_skip_reason_when_filesystem_disabled(self, engine, db):
        create = await engine.session_manage_lifecycle(
            operation="create", project_name="issue128-fs-off"
        )
        session_id = create.session_id

        result = await engine.session_create_notebook_async(
            session_id=session_id, save_to_file=True
        )

        assert result.status == "success"
        assert result.file_path is None
        assert result.file_status is not None
        assert "filesystem" in result.file_status.lower()


# ---------------------------------------------------------------------------
# Issue #126: session_query_notebooks projection
# ---------------------------------------------------------------------------


class TestIssue126Projection:
    async def test_default_projection_omits_full_notebook_fields(self, engine, db):
        create1 = await engine.session_manage_lifecycle(
            operation="create", project_name="issue126-proj"
        )
        sid1 = create1.session_id
        await engine.session_create_notebook_async(session_id=sid1, title="Notebook One")

        create2 = await engine.session_manage_lifecycle(
            operation="create", project_name="issue126-proj"
        )
        sid2 = create2.session_id
        await engine.session_create_notebook_async(session_id=sid2, title="Notebook Two")

        rows = await engine.session_query_notebooks(project_name="issue126-proj")

        assert len(rows) == 2
        allowed_keys = {"session_id", "title", "tags", "created_at", "project_name"}
        for row in rows:
            assert set(row.keys()) <= allowed_keys
            assert "summary_markdown" not in row
            assert "authored_body" not in row
            assert "key_changes" not in row

    async def test_summary_only_false_includes_full_notebook_fields(self, engine, db):
        create = await engine.session_manage_lifecycle(
            operation="create", project_name="issue126-proj-full"
        )
        sid = create.session_id
        await engine.session_create_notebook_async(session_id=sid, title="Full Notebook")

        rows = await engine.session_query_notebooks(
            project_name="issue126-proj-full", summary_only=False
        )

        assert len(rows) == 1
        assert "summary_markdown" in rows[0]


# ---------------------------------------------------------------------------
# Issue #127a: session_update_notebook resolves to the notebook OWNER, not
# merely the newest active session
# ---------------------------------------------------------------------------


class TestIssue127UpdateResolution:
    async def test_update_targets_notebook_owner_not_newer_notebook_less_session(self, engine, db):
        """Exact scenario from issue #127: session A (older, HAS a notebook)
        and session B (newer, active, NO notebook) both belong to project P.
        session_update_notebook(project_name=P, ...) must target A."""
        create_a = await engine.session_manage_lifecycle(
            operation="create", project_name="issue127-proj"
        )
        sid_a = create_a.session_id
        await engine.session_create_notebook_async(session_id=sid_a, body="original body")

        create_b = await engine.session_manage_lifecycle(
            operation="create", project_name="issue127-proj"
        )
        sid_b = create_b.session_id
        assert sid_b != sid_a

        result = await engine.session_update_notebook(project_name="issue127-proj", body="amended")

        assert result["status"] == "success"
        assert result["session_id"] == sid_a

        summary_a = await db.get_session_summary(sid_a)
        assert summary_a is not None
        assert summary_a["authored_body"] == "amended"

        summary_b = await db.get_session_summary(sid_b)
        assert summary_b is None


# ---------------------------------------------------------------------------
# Issue #127b: lean MCP execute_tool envelope reflects inner failure status
# ---------------------------------------------------------------------------


class TestIssue127EnvelopeDowngrade:
    @pytest.fixture
    async def lean_interface(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "test_issue_127_envelope.db"))
        await backend.initialize()
        eng = SessionIntelligenceEngine(
            repository_path=str(tmp_path), use_filesystem=False, database=backend
        )
        interface = LeanMCPInterface(eng)
        yield interface
        await backend.close()

    async def test_inner_error_status_downgrades_outer_envelope(self, lean_interface):
        execute = _get_meta_tool(lean_interface, "execute_tool")
        create_result = await execute(
            "session_manage_lifecycle",
            {"operation": "create", "project_name": "issue127-envelope"},
        )
        assert create_result["status"] == "success"
        session_id = _extract_session_id(create_result)

        # No notebook exists yet for session_id, so the inner call fails.
        result = await execute(
            "session_update_notebook",
            {"session_id": session_id, "body": "no notebook yet"},
        )

        inner = result["result"]
        inner_status = (
            inner.get("status") if isinstance(inner, dict) else getattr(inner, "status", None)
        )
        assert inner_status == "error"
        assert result["status"] == "error"

    async def test_successful_call_still_yields_outer_success(self, lean_interface):
        """Guards against an indiscriminate downgrade: a genuinely
        successful call must still report outer status success."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute(
            "session_manage_lifecycle",
            {"operation": "create", "project_name": "issue127-envelope-ok"},
        )
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# FK persistence fix: _resolve_session_context auto-create branches now
# persist to the database
# ---------------------------------------------------------------------------


class TestFKPersistenceGapFixed:
    async def test_create_notebook_via_project_name_with_no_prior_lifecycle_create(
        self, engine, db
    ):
        """No session_manage_lifecycle(create) was ever called for this
        project -- session_create_notebook_async(project_name=...) must
        auto-create a session via _resolve_session_context AND persist it
        to the DB before save_session_summary runs, instead of failing
        FOREIGN KEY constraint failed."""
        result = await engine.session_create_notebook_async(
            project_name="fk-fix-never-created-project"
        )

        assert result.status == "success"
        assert result.search_indexed is True
