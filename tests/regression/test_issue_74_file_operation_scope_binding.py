"""
Regression tests for issue #74: unbound file operations silently bound to
whichever project most recently created a session in the shared engine.

Bug: session_track_file_operation used to pick its session with
     `session_id = list(self.session_cache.keys())[-1]` -- the last key in
     dict insertion order, with no project check at all. That is weaker even
     than the ambient-session fallback issue #72 removed from
     session_log_decision, which at least required that SOME session had
     been created in-process. The HTTP transport builds a SINGLE
     SessionIntelligenceEngine shared by every project (see
     http_server.py lifespan()), so session_cache is cross-project state and
     its last key belongs to whichever project most recently created a
     session in the process -- not the caller. A file op from project A
     could get recorded against project B's session.
Fix: session_track_file_operation now raises SessionContextRequiredError
     when none of session_id/session_name/project_name is supplied (unless
     allow_unbound=True is explicitly passed), and resolves scope through
     the same _resolve_session_context() path as session_log_decision and
     session_log_learning, instead of reaching into session_cache directly.
"""

import sqlite3

import pytest

from core.session_engine import SessionContextRequiredError, SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


@pytest.mark.regression
class TestUnboundFileOperationIsRejected:

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_unbound_file_operation_raises(self, engine):
        """The core #74 regression: an in-process session existing somewhere
        in session_cache must NOT silently satisfy an unscoped
        session_track_file_operation call."""
        create_result = await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-a"
        )
        assert create_result.status == "success", (
            "Precondition failed: session_manage_lifecycle(create) did not "
            "succeed, so this test cannot exercise the #74 last-cache-key "
            "bleed scenario."
        )

        with pytest.raises(SessionContextRequiredError) as exc_info:
            await engine.session_track_file_operation(
                operation="edit", file_path="/tmp/x.py"
            )
        assert "session_track_file_operation" in str(exc_info.value), (
            "The #74 regression message should identify "
            "session_track_file_operation as requiring an explicit scope, "
            "not silently fall back to the last session_cache key."
        )

    async def test_allow_unbound_escape_hatch_still_works(self, engine):
        await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-a"
        )

        result = await engine.session_track_file_operation(
            operation="edit",
            file_path="/tmp/x.py",
            allow_unbound=True,
        )
        assert result is not None, (
            "allow_unbound=True is the documented escape hatch for the #74 "
            "guard and must still permit an unscoped file operation."
        )

    async def test_relative_project_path_alone_still_raises(self, engine):
        """A relative project_path resolves against the SERVER's cwd, not
        the caller's, so it must not silently satisfy scope (#48/#49)."""
        await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-a"
        )

        with pytest.raises(SessionContextRequiredError):
            await engine.session_track_file_operation(
                operation="edit",
                file_path="/tmp/x.py",
                project_path="relative/proj-a",
            )


@pytest.mark.regression
class TestSharedEngineDoesNotBleedAcrossProjects:
    """Every pre-#72 resolver test built a FRESH engine per test, which is
    exactly why this bug class (#72, and now #74) kept shipping: the HTTP
    transport builds ONE engine shared across all projects. These tests
    reuse a single engine for two different projects, with proj-b created
    SECOND so it is the last key in session_cache -- the value the old
    buggy `list(session_cache.keys())[-1]` code would have picked."""

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_file_operation_binds_to_named_project_not_last_cache_key(
        self, engine, tmp_path
    ):
        proj_a_path = str(tmp_path / "proj-a")
        proj_b_path = str(tmp_path / "proj-b")

        proj_a_create = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-a",
            project_path=proj_a_path,
        )
        # proj-b is created SECOND, i.e. it is the last key in
        # session_cache when the file operation below is tracked.
        proj_b_create = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-b",
            project_path=proj_b_path,
        )

        cache_keys = list(engine.session_cache.keys())
        assert cache_keys[-1] == proj_b_create.session_id, (
            "Precondition failed: proj-b's session is not the last key in "
            "session_cache, so this test would not actually exercise the "
            "#74 last-cache-key bug even if the fix were reverted."
        )

        result = await engine.session_track_file_operation(
            operation="edit",
            file_path="/tmp/x.py",
            project_name="proj-a",
        )

        assert result["session_id"] == proj_a_create.session_id, (
            "Issue #74: the file operation resolved to the wrong session. "
            "It must bind to proj-a's session (the caller's project_name), "
            "not fall back to the last key in session_cache."
        )
        assert result["session_id"] != proj_b_create.session_id, (
            "Issue #74: binding this file operation to proj-b's session "
            "means the last-cache-key fallback is back -- proj-b was only "
            "created more recently, it was never named by the caller."
        )

    async def test_file_operation_binds_by_absolute_project_path(
        self, engine, tmp_path
    ):
        # derive_project_name() probes path.is_dir() and otherwise falls
        # back to path.parent (both proj-a and proj-b would then resolve
        # to tmp_path's own basename, collapsing the two projects into the
        # same derived name). The directories must exist on disk so each
        # path derives its own distinct project name, matching how a real
        # caller's cwd always exists.
        proj_a_dir = tmp_path / "proj-a"
        proj_a_dir.mkdir()
        proj_b_dir = tmp_path / "proj-b"
        proj_b_dir.mkdir()
        proj_a_path = str(proj_a_dir)
        proj_b_path = str(proj_b_dir)

        proj_a_create = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-a",
            project_path=proj_a_path,
        )
        proj_b_create = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-b",
            project_path=proj_b_path,
        )

        result = await engine.session_track_file_operation(
            operation="edit",
            file_path="/tmp/x.py",
            project_path=proj_a_path,
        )

        assert result["session_id"] == proj_a_create.session_id, (
            "Issue #74: an absolute project_path naming proj-a must derive "
            "proj-a's project_name and bind to proj-a's session, not to "
            "proj-b's merely because proj-b was created more recently."
        )
        assert result["session_id"] != proj_b_create.session_id

    async def test_explicit_session_id_wins_over_last_cache_key(
        self, engine, tmp_path
    ):
        proj_a_path = str(tmp_path / "proj-a")
        proj_b_path = str(tmp_path / "proj-b")

        proj_a_create = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-a",
            project_path=proj_a_path,
        )
        proj_b_create = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-b",
            project_path=proj_b_path,
        )

        result = await engine.session_track_file_operation(
            operation="edit",
            file_path="/tmp/x.py",
            session_id=proj_a_create.session_id,
        )

        assert result["session_id"] == proj_a_create.session_id, (
            "Issue #74: an explicit session_id must be honored even though "
            "proj-b's session sits last in session_cache."
        )
        assert result["session_id"] != proj_b_create.session_id


@pytest.mark.regression
class TestFileOperationIsPersistedUnderResolvedSession:

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_row_is_written_under_the_resolved_session(
        self, engine, tmp_path
    ):
        """SQLiteBackend.query_file_operations_by_session() exists (see
        src/persistence/sqlite.py), so this test uses that getter directly
        rather than a raw SQL SELECT."""
        proj_a_path = str(tmp_path / "proj-a")
        proj_b_path = str(tmp_path / "proj-b")

        proj_a_create = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-a",
            project_path=proj_a_path,
        )
        proj_b_create = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-b",
            project_path=proj_b_path,
        )

        result = await engine.session_track_file_operation(
            operation="edit",
            file_path="/tmp/x.py",
            project_name="proj-a",
        )
        assert result["status"] == "success"

        proj_a_rows = await engine.database.query_file_operations_by_session(
            proj_a_create.session_id
        )
        assert len(proj_a_rows) == 1, (
            "Issue #74: the file operation row should be persisted under "
            "proj-a's session_id in the file_operations table, but "
            f"query_file_operations_by_session(proj-a) returned "
            f"{len(proj_a_rows)} rows instead of 1."
        )
        assert proj_a_rows[0]["file_path"] == "/tmp/x.py"

        proj_b_rows = await engine.database.query_file_operations_by_session(
            proj_b_create.session_id
        )
        assert proj_b_rows == [], (
            "Issue #74: the file operation must not be persisted under "
            "proj-b's session_id -- that would mean the row was written "
            "under the last-created session instead of the caller's "
            "named project_name."
        )

        # Cross-check with a raw SQL SELECT against the underlying SQLite
        # file directly, independent of the getter above.
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute(
                "SELECT session_id, file_path FROM file_operations"
            )
            raw_rows = cursor.fetchall()
        finally:
            conn.close()
        assert raw_rows == [(proj_a_create.session_id, "/tmp/x.py")], (
            "Issue #74: a raw SQL SELECT against file_operations confirms "
            f"the persisted row(s) {raw_rows!r} must reference only "
            "proj-a's session_id, not proj-b's."
        )
