"""
Regression tests for the non-breaking part of issue #98: project_path scoping.

Two worktrees of one project share a project_name but have different
project_paths. Recall used to key on project_name alone, create ignored
session_name, create_notebook had no project_path, and log_decision did not
say which scope it bound to. (Project-name derivation itself is issue #216.)

SQLite is exercised for real; PostgreSQL through a fake asyncpg pool.
"""

import pytest

from core.project_naming import derive_project_name
from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.postgresql import PostgreSQLBackend
from persistence.sqlite import SQLiteBackend

PROJECT = "proj-98"


@pytest.fixture
async def engine(tmp_path):
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


@pytest.fixture
def paths(tmp_path):
    a, b = tmp_path / "wt-a", tmp_path / "wt-b"
    a.mkdir()
    b.mkdir()
    return str(a), str(b)


async def _seed(engine, path, tag):
    """One session at `path` carrying a decision, learning and notebook."""
    created = await engine.session_manage_lifecycle(
        "create", project_name=PROJECT, project_path=path, session_name=f"sess-{tag}"
    )
    sid = created.session_id
    await engine.session_log_decision(f"decision-{tag}", session_id=sid)
    await engine.session_log_learning(
        category="pattern",
        learning_content=f"learning-{tag}",
        trigger_context=f"trigger-{tag}",
        session_id=sid,
    )
    await engine.session_create_notebook_async(session_id=sid, title=f"notebook-{tag}", body="b")
    return sid


def _texts(recalled):
    return {
        "sessions": {s["id"] for s in recalled["sessions"]},
        "decisions": {d["description"] for d in recalled["decisions"]},
        "learnings": {item["learning_content"] for item in recalled["learnings"]},
        "notebooks": {n["title"] for n in recalled["notebooks"]},
    }


@pytest.mark.regression
class TestRecallProjectPath:
    async def test_project_path_narrows_every_tier(self, engine, paths):
        path_a, path_b = paths
        sid_a = await _seed(engine, path_a, "a")
        sid_b = await _seed(engine, path_b, "b")

        only_a = _texts(await engine.session_recall(project_name=PROJECT, project_path=path_a))
        assert only_a == {
            "sessions": {sid_a},
            "decisions": {"decision-a"},
            "learnings": {"learning-a"},
            "notebooks": {"notebook-a"},
        }
        only_b = _texts(await engine.session_recall(project_name=PROJECT, project_path=path_b))
        assert only_b["sessions"] == {sid_b}
        assert only_b["decisions"] == {"decision-b"}
        assert only_b["learnings"] == {"learning-b"}
        assert only_b["notebooks"] == {"notebook-b"}

    async def test_unscoped_recall_returns_both_and_items_carry_path(self, engine, paths):
        path_a, path_b = paths
        await _seed(engine, path_a, "a")
        await _seed(engine, path_b, "b")

        recalled = await engine.session_recall(project_name=PROJECT)
        texts = _texts(recalled)
        assert texts["decisions"] == {"decision-a", "decision-b"}
        assert texts["learnings"] == {"learning-a", "learning-b"}
        assert texts["notebooks"] == {"notebook-a", "notebook-b"}
        assert "project_path" not in recalled
        by_decision = {d["description"]: d["project_path"] for d in recalled["decisions"]}
        assert by_decision == {"decision-a": path_a, "decision-b": path_b}
        by_learning = {i["learning_content"]: i["project_path"] for i in recalled["learnings"]}
        assert by_learning == {"learning-a": path_a, "learning-b": path_b}

    async def test_scoped_recall_echoes_path_and_unknown_path_is_empty(self, engine, paths):
        path_a, _ = paths
        await _seed(engine, path_a, "a")

        recalled = await engine.session_recall(project_name=PROJECT, project_path=path_a)
        assert recalled["project_path"] == path_a
        assert recalled["counts"]["decisions"] == 1
        nowhere = await engine.session_recall(project_name=PROJECT, project_path="/no/such/wt")
        assert nowhere["counts"] == {"sessions": 0, "decisions": 0, "learnings": 0, "notebooks": 0}

    async def test_relative_project_path_is_an_error_not_an_unfiltered_recall(self, engine, paths):
        await _seed(engine, paths[0], "a")
        recalled = await engine.session_recall(project_name=PROJECT, project_path="relative/dir")
        assert "not an absolute path" in recalled["error"]
        assert recalled["counts"]["decisions"] == 0

    async def test_project_name_path_resolution_does_not_cross_worktrees(self, engine, paths):
        path_a, path_b = paths
        sid_a = await _seed(engine, path_a, "a")
        sid_b = await _seed(engine, path_b, "b")

        # B is the most recent session; a project_name+project_path write for
        # A must still land on A's session, not B's.
        in_a = await engine.session_log_decision(
            "decision-a2", project_name=PROJECT, project_path=path_a
        )
        assert in_a.session_id == sid_a
        in_b = await engine.session_log_decision(
            "decision-b2", project_name=PROJECT, project_path=path_b
        )
        assert in_b.session_id == sid_b

    async def test_lean_schema_exposes_project_path(self, engine):
        spec = LeanMCPInterface(engine).tool_registry["session_recall"]["schema"]
        assert "project_path" in spec["properties"]
        assert spec["required"] == ["project_name"]


@pytest.mark.regression
class TestCreateSessionName:
    async def test_create_persists_and_echoes_session_name(self, engine, paths):
        created = await engine.session_manage_lifecycle(
            "create", project_name=PROJECT, project_path=paths[0], session_name="named-one"
        )
        assert created.session_data.session_name == "named-one"
        row = await engine.database.get_session(created.session_id)
        assert row["session_name"] == "named-one"

        # Later name-based resolution finds it instead of creating a new session.
        logged = await engine.session_log_decision(
            "by-name", session_name="named-one", project_name=PROJECT
        )
        assert logged.session_id == created.session_id

    async def test_adopt_sets_missing_name_but_never_overwrites(self, engine, paths):
        first = await engine.session_manage_lifecycle(
            "create", project_name=PROJECT, project_path=paths[0], session_id="native-uuid-98"
        )
        assert first.session_data.session_name is None

        named = await engine.session_manage_lifecycle(
            "create", session_id="native-uuid-98", session_name="late-name"
        )
        assert "adopted" in named.message
        assert (await engine.database.get_session("native-uuid-98"))["session_name"] == "late-name"

        clash = await engine.session_manage_lifecycle(
            "create", session_id="native-uuid-98", session_name="other-name"
        )
        assert "kept existing session_name 'late-name'" in clash.message
        assert (await engine.database.get_session("native-uuid-98"))["session_name"] == "late-name"

    async def test_create_with_only_project_path_derives_a_name(self, engine, paths):
        created = await engine.session_manage_lifecycle("create", project_path=paths[0])
        expected = derive_project_name(paths[0])
        assert expected not in ("unknown", "_unbound_")
        assert created.session_data.project_name == expected
        assert (await engine.database.get_session(created.session_id))["project_name"] == expected

    async def test_create_with_nothing_still_falls_back_to_unknown(self, engine):
        created = await engine.session_manage_lifecycle("create")
        assert created.session_data.project_name == "unknown"

    async def test_lifecycle_schema_documents_create_semantics(self, engine):
        props = LeanMCPInterface(engine).tool_registry["session_manage_lifecycle"]["schema"][
            "properties"
        ]
        assert "create" in props["session_name"]["description"]


@pytest.mark.regression
class TestNotebookProjectPath:
    async def test_create_notebook_scoped_by_project_name_and_path(self, engine, paths):
        path_a, path_b = paths
        sid_a = await _seed(engine, path_a, "a")
        sid_b = await _seed(engine, path_b, "b")

        nb_a = await engine.session_create_notebook_async(
            project_name=PROJECT, project_path=path_a, title="extra-a", body="x"
        )
        nb_b = await engine.session_create_notebook_async(
            project_name=PROJECT, project_path=path_b, title="extra-b", body="x"
        )
        assert nb_a.session_id == sid_a
        assert nb_b.session_id == sid_b

    async def test_create_notebook_with_only_project_path_derives_name(self, engine, paths):
        nb = await engine.session_create_notebook_async(
            project_path=paths[0], title="path-only", body="x"
        )
        assert nb.status == "success"
        row = await engine.database.get_session(nb.session_id)
        assert row["project_name"] == derive_project_name(paths[0])
        assert row["project_path"] == paths[0]

    async def test_three_write_tools_share_one_scope_description(self, engine):
        registry = LeanMCPInterface(engine).tool_registry
        sentence = "Scope: pass session_id (preferred), or session_name, or project_name"
        for tool in ("session_log_decision", "session_log_learning", "session_create_notebook"):
            desc = registry[tool]["schema"]["properties"]["project_path"]["description"]
            assert sentence in desc, tool


@pytest.mark.regression
class TestLegacyPathlessAdoption:
    @pytest.mark.parametrize("legacy_path", ["", "_unknown_"])
    async def test_pathless_active_session_is_adopted(self, engine, paths, legacy_path):
        created = await engine.session_manage_lifecycle("create", project_name=PROJECT)
        await engine.database.save_session(
            {**(await engine.database.get_session(created.session_id)), "project_path": legacy_path}
        )
        logged = await engine.session_log_decision(
            "adopt", project_name=PROJECT, project_path=paths[0]
        )
        assert logged.session_id == created.session_id

    async def test_session_with_other_real_path_gets_new_session(self, engine, paths):
        path_a, path_b = paths
        sid_a = await _seed(engine, path_a, "a")
        logged = await engine.session_log_decision(
            "other", project_name=PROJECT, project_path=path_b
        )
        assert logged.session_id != sid_a


@pytest.mark.regression
class TestTrailingSlash:
    async def test_recall_with_trailing_slash_matches_unslashed_rows(self, engine, paths):
        await _seed(engine, paths[0], "a")
        recalled = await engine.session_recall(project_name=PROJECT, project_path=paths[0] + "/")
        assert recalled["counts"]["decisions"] == 1
        assert recalled["project_path"] == paths[0]


@pytest.mark.regression
class TestDecisionEchoesScope:
    async def test_log_decision_returns_project_name_and_path(self, engine, paths):
        await _seed(engine, paths[0], "a")
        result = await engine.session_log_decision(
            "echo me", project_name=PROJECT, project_path=paths[0]
        )
        assert result.project_name == PROJECT
        assert result.project_path == paths[0]
        dumped = result.model_dump()
        assert {"decision_id", "session_id", "supersedes"} <= set(dumped)


class _FakeConn:
    def __init__(self):
        self.calls = []

    async def fetch(self, query, *args):
        self.calls.append((query, args))
        return []

    async def fetchrow(self, query, *args):
        self.calls.append((query, args))
        return None


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)


@pytest.mark.regression
class TestPostgresSql:
    def _backend(self):
        backend = PostgreSQLBackend(dsn="postgresql://localhost/fake_issue_98")
        conn = _FakeConn()
        backend._pool = _FakePool(conn)  # type: ignore[assignment]
        return backend, conn

    async def test_recall_binds_project_path_in_every_tier(self):
        backend, conn = self._backend()
        result = await backend.recall_project(PROJECT, project_path="/wt/a")

        assert result["project_path"] == "/wt/a"
        assert len(conn.calls) == 4
        sessions, decisions, learnings, notebooks = conn.calls
        assert "project_path = $4" in sessions[0]
        # Decisions/notebooks compare the effective scope (override, else session).
        assert "COALESCE(d.project_path, s.project_path) = $4" in decisions[0]
        assert "COALESCE(ss.project_path, s.project_path) = $4" in notebooks[0]
        for _query, args in (sessions, decisions, notebooks):
            assert args[3] == "/wt/a"
        assert "COALESCE(d.project_path, s.project_path) AS project_path" in decisions[0]
        assert "project_path = $3" in learnings[0]
        assert learnings[1] == (PROJECT, 10, "/wt/a")

    async def test_recall_without_path_binds_null(self):
        backend, conn = self._backend()
        result = await backend.recall_project(PROJECT)

        assert "project_path" not in result
        sessions, decisions, learnings, notebooks = conn.calls
        assert sessions[1][3] is None
        assert decisions[1][3] is None
        assert learnings[1][2] is None
        assert notebooks[1][3] is None

    async def test_find_recent_session_binds_project_path(self):
        backend, conn = self._backend()

        await backend.find_recent_session_by_project(PROJECT, project_path="/wt/a")
        query, args = conn.calls[-1]
        assert "project_path = $4" in query
        assert args[-1] == "/wt/a"

        await backend.find_recent_session_by_project(PROJECT, status="completed")
        query, args = conn.calls[-1]
        assert "project_path = $3" in query
        assert args == (PROJECT, "completed", None)
