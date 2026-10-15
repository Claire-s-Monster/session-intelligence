"""
Regression tests for issue #241: sessions bound to the SERVER's cwd, explicit
session_id ignoring the caller's project_path, and project_path_derived
reporting True for caller-supplied paths.

SQLite is exercised for real; no PostgreSQL needed.
"""

from pathlib import Path

import pytest

from core.project_naming import UNBOUND
from core.session_engine import SessionIntelligenceEngine
from models.session_models import UNKNOWN_PROJECT_PATH
from persistence.sqlite import SQLiteBackend

NATIVE_ID = "11111111-2222-3333-4444-555555555555"
REAL_PATH = "/home/user/work/proj-241"
SERVER_CWD = str(Path.cwd().resolve())


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
    db = SQLiteBackend(db_path=str(tmp_path / "issue241.db"))
    await db.initialize()
    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )
    yield eng
    await db.close()


async def _hook_session(engine, **step):
    await engine.session_track_execution(
        session_id=NATIVE_ID,
        agent_name="primary",
        step_data={"phase": "post_tool_use", "command": "ls", **step},
    )
    return engine.session_cache[NATIVE_ID]


@pytest.mark.regression
class TestHookCreatedSession:
    async def test_no_working_directory_never_uses_server_cwd(self, engine):
        session = await _hook_session(engine)

        assert session.project_path == UNKNOWN_PROJECT_PATH
        assert session.project_path != SERVER_CWD
        assert session.project_name == UNBOUND
        assert session.project_name != "session-intelligence"

    async def test_working_directory_still_used_when_present(self, engine):
        session = await _hook_session(engine, working_directory=REAL_PATH)

        assert session.project_path == REAL_PATH
        assert session.project_name == "proj-241"

    async def test_relative_working_directory_is_ignored(self, engine):
        session = await _hook_session(engine, working_directory="relative/dir")

        assert session.project_path == UNKNOWN_PROJECT_PATH

    async def test_later_event_heals_sentinel_session(self, engine):
        await _hook_session(engine)
        session = await _hook_session(engine, working_directory=REAL_PATH)

        assert session.project_path == REAL_PATH
        assert session.project_name == "proj-241"
        assert NATIVE_ID in engine.dirty_snapshot()

    async def test_real_path_is_never_overwritten(self, engine):
        await _hook_session(engine, working_directory=REAL_PATH)
        session = await _hook_session(engine, working_directory="/somewhere/else")

        assert session.project_path == REAL_PATH
        assert session.project_name == "proj-241"


async def _sentinel_row(engine, name=UNBOUND):
    result = await engine.session_manage_lifecycle(
        "create", session_id=NATIVE_ID, project_name=name
    )
    assert result.session_data.project_path == UNKNOWN_PROJECT_PATH
    return result


@pytest.mark.regression
class TestExplicitSessionIdHealing:
    async def test_caller_path_heals_sentinel_row(self, engine):
        await _sentinel_row(engine)

        resolved = await engine._resolve_session_context(
            session_id=NATIVE_ID, project_path=REAL_PATH
        )

        assert resolved.project_path == REAL_PATH
        assert resolved.project_name == "proj-241"
        assert resolved.project_path_source == "caller"
        row = await engine.database.get_session(NATIVE_ID)
        assert row["project_path"] == REAL_PATH
        assert row["project_name"] == "proj-241"
        assert engine.session_cache[NATIVE_ID].project_path == REAL_PATH

    async def test_learning_with_session_id_and_path_is_recallable(self, engine):
        await _sentinel_row(engine)

        result = await engine.session_log_learning(
            category="pattern",
            learning_content="healed",
            session_id=NATIVE_ID,
            project_path=REAL_PATH,
        )

        assert result.learning.project_path == REAL_PATH
        assert result.project_path_source == "caller"
        assert result.project_path_derived is False

    async def test_different_real_path_is_kept_with_warning(self, engine):
        await engine.session_manage_lifecycle(
            "create", session_id=NATIVE_ID, project_name="proj-241", project_path=REAL_PATH
        )

        resolved = await engine._resolve_session_context(
            session_id=NATIVE_ID, project_path="/other/place"
        )

        assert resolved.project_path == REAL_PATH
        assert [w["kind"] for w in resolved.warnings] == ["project_path_mismatch"]
        row = await engine.database.get_session(NATIVE_ID)
        assert row["project_path"] == REAL_PATH

    async def test_mismatch_warning_reaches_decision_learning_notebook(self, engine):
        await engine.session_manage_lifecycle(
            "create", session_id=NATIVE_ID, project_name="proj-241", project_path=REAL_PATH
        )
        kwargs = {"session_id": NATIVE_ID, "project_path": "/other/place"}

        decision = await engine.session_log_decision("d", **kwargs)
        learning = await engine.session_log_learning(
            category="pattern", learning_content="l", **kwargs
        )
        notebook = await engine.session_create_notebook_async(
            save_to_file=False, save_to_database=False, **kwargs
        )

        for result in (decision, learning, notebook):
            assert "project_path_mismatch" in [w["kind"] for w in result.warnings]

    async def test_notebook_heals_sentinel_row(self, engine):
        await _sentinel_row(engine)

        notebook = await engine.session_create_notebook_async(
            session_id=NATIVE_ID,
            project_path=REAL_PATH,
            save_to_file=False,
            save_to_database=False,
        )

        assert notebook.notebook.project_path == REAL_PATH
        assert notebook.project_path_source == "caller"

    async def test_adopt_with_path_heals_sentinel_row(self, engine):
        await _sentinel_row(engine)

        adopted = await engine.session_manage_lifecycle(
            "create", session_id=NATIVE_ID, project_path=REAL_PATH
        )

        assert adopted.session_data["project_path"] == REAL_PATH
        row = await engine.database.get_session(NATIVE_ID)
        assert row["project_path"] == REAL_PATH


OTHER_NAME = "ocaml-feedstock"
OTHER_PATH = "/home/user/work/ocaml-feedstock"


async def _seed_server_cwd_row(engine, *, hook_bound: bool):
    """A row stamped with the server's cwd, as created before the #241 fix."""
    await engine.database.save_session(
        {
            "id": NATIVE_ID,
            "started_at": "2026-01-01T00:00:00+00:00",
            "project_path": SERVER_CWD,
            "project_name": "session-intelligence",
            "status": "active",
            "metadata": {"tags": ["hook-bound", "claude-native-session"] if hook_bound else []},
        }
    )


@pytest.mark.regression
class TestServerCwdStampedRows:
    async def test_hook_bound_server_cwd_row_is_healed_by_decision(self, engine):
        await _seed_server_cwd_row(engine, hook_bound=True)

        result = await engine.session_log_decision(
            "d", session_id=NATIVE_ID, project_name=OTHER_NAME, project_path=OTHER_PATH
        )

        assert result.warnings == []
        row = await engine.database.get_session(NATIVE_ID)
        assert row["project_path"] == OTHER_PATH
        assert row["project_name"] == OTHER_NAME

    async def test_heal_without_caller_name_derives_name_from_path(self, engine):
        await _seed_server_cwd_row(engine, hook_bound=True)

        resolved = await engine._resolve_session_context(
            session_id=NATIVE_ID, project_path=OTHER_PATH
        )

        assert resolved.project_path == OTHER_PATH
        assert resolved.project_name == OTHER_NAME
        assert resolved.warnings == []

    async def test_non_hook_bound_server_cwd_row_is_not_healed(self, engine):
        await _seed_server_cwd_row(engine, hook_bound=False)

        result = await engine.session_log_decision(
            "d", session_id=NATIVE_ID, project_name=OTHER_NAME, project_path=OTHER_PATH
        )

        kinds = [w["kind"] for w in result.warnings]
        assert "project_path_mismatch" in kinds
        assert "project_name_mismatch" in kinds
        row = await engine.database.get_session(NATIVE_ID)
        assert row["project_path"] == SERVER_CWD
        assert row["project_name"] == "session-intelligence"

    async def test_name_mismatch_reaches_learning_and_notebook(self, engine):
        await _seed_server_cwd_row(engine, hook_bound=False)
        kwargs = {"session_id": NATIVE_ID, "project_name": OTHER_NAME, "project_path": OTHER_PATH}

        learning = await engine.session_log_learning(
            category="pattern", learning_content="l", **kwargs
        )
        notebook = await engine.session_create_notebook_async(
            save_to_file=False, save_to_database=False, **kwargs
        )

        for result in (learning, notebook):
            assert "project_name_mismatch" in [w["kind"] for w in result.warnings]

    async def test_notebook_heals_hook_bound_server_cwd_row(self, engine):
        await _seed_server_cwd_row(engine, hook_bound=True)

        notebook = await engine.session_create_notebook_async(
            session_id=NATIVE_ID,
            project_name=OTHER_NAME,
            project_path=OTHER_PATH,
            save_to_file=False,
            save_to_database=False,
        )

        assert notebook.notebook.project_path == OTHER_PATH
        assert "project_name_mismatch" not in [w["kind"] for w in notebook.warnings]
        row = await engine.database.get_session(NATIVE_ID)
        assert row["project_path"] == OTHER_PATH
        assert row["project_name"] == OTHER_NAME

    async def test_adopt_heals_hook_bound_server_cwd_row(self, engine):
        await _seed_server_cwd_row(engine, hook_bound=True)

        await engine.session_manage_lifecycle(
            "create", session_id=NATIVE_ID, project_path=OTHER_PATH
        )

        row = await engine.database.get_session(NATIVE_ID)
        assert row["project_path"] == OTHER_PATH
        assert row["project_name"] == OTHER_NAME

    async def test_hook_event_heals_server_cwd_row(self, engine):
        engine._create_session(
            mode="auto",
            project_name="session-intelligence",
            metadata={"project_path": SERVER_CWD, "tags": ["hook-bound"]},
            session_id=NATIVE_ID,
        )

        session = await _hook_session(engine, working_directory=OTHER_PATH)

        assert session.project_path == OTHER_PATH
        assert session.project_name == OTHER_NAME

    async def test_hook_event_never_heals_non_hook_bound_server_cwd_row(self, engine):
        engine._create_session(
            mode="auto",
            project_name="session-intelligence",
            metadata={"project_path": SERVER_CWD},
            session_id=NATIVE_ID,
        )

        session = await _hook_session(engine, working_directory=OTHER_PATH)

        assert session.project_path == SERVER_CWD

    async def test_sqlite_heal_requires_hook_bound_tag_and_cwd(self, engine):
        db = engine.database
        await _seed_server_cwd_row(engine, hook_bound=False)
        assert not await db.heal_session_project(
            NATIVE_ID, OTHER_PATH, OTHER_NAME, server_cwd=SERVER_CWD
        )

        await _seed_server_cwd_row(engine, hook_bound=True)
        assert not await db.heal_session_project(NATIVE_ID, OTHER_PATH, OTHER_NAME)
        assert await db.heal_session_project(
            NATIVE_ID, OTHER_PATH, OTHER_NAME, server_cwd=SERVER_CWD
        )
        row = await db.get_session(NATIVE_ID)
        assert (row["project_path"], row["project_name"]) == (OTHER_PATH, OTHER_NAME)


@pytest.mark.regression
class TestProjectPathSource:
    async def test_caller_path(self, engine):
        result = await engine.session_manage_lifecycle(
            "create", project_name="p", project_path=REAL_PATH
        )

        assert result.project_path_source == "caller"
        assert result.project_path_derived is False

    async def test_unknown_path(self, engine):
        result = await engine.session_manage_lifecycle("create", project_name="p-no-path")

        assert result.project_path_source == "unknown"
        assert result.project_path_derived is False

    async def test_derived_from_session_row(self, engine):
        await engine.session_manage_lifecycle("create", project_name="p", project_path=REAL_PATH)

        learning = await engine.session_log_learning(
            category="pattern", learning_content="x", project_name="p"
        )

        assert learning.learning.project_path == REAL_PATH
        assert learning.project_path_source == "derived"
        assert learning.project_path_derived is True

    async def test_legacy_fallback_when_source_unset(self):
        from models.session_models import LearningResult, ProjectLearning

        learning = ProjectLearning(
            id="l",
            project_path=REAL_PATH,
            category="pattern",
            learning_content="c",
            created_at="2026-01-01T00:00:00+00:00",
        )

        result = LearningResult(id="l", status="saved", learning=learning)
        assert result.project_path_derived is True
