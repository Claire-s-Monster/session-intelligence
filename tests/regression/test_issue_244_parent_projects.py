"""
Regression tests for issue #244 part 2: parent/child projects.

SQLite is exercised for real; the new PostgreSQL SQL through a fake pool that
records SQL and bound args.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.base import MAX_PROJECT_DEPTH
from persistence.postgresql import PostgreSQLBackend
from persistence.sqlite import SQLiteBackend

TIERS = ("decisions", "learnings", "notebooks")


@pytest.fixture
async def engine(tmp_path):
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


def _dir(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return str(d)


async def _seed(engine, project, path, tag, decisions=1):
    created = await engine.session_manage_lifecycle(
        "create", project_name=project, project_path=path, session_name=f"sess-{tag}"
    )
    sid = created.session_id
    for i in range(decisions):
        await engine.session_log_decision(f"decision-{tag}-{i}", session_id=sid)
    await engine.session_log_learning(
        category="pattern",
        learning_content=f"learning-{tag}",
        trigger_context=f"trigger-{tag}",
        session_id=sid,
    )
    await engine.session_create_notebook_async(session_id=sid, title=f"notebook-{tag}", body="b")
    return sid


def _labels(recalled):
    return {
        "decisions": [d["description"] for d in recalled["decisions"]],
        "learnings": [x["learning_content"] for x in recalled["learnings"]],
        "notebooks": [n["title"] for n in recalled["notebooks"]],
    }


@pytest.mark.regression
class TestSetAndClear:
    async def test_set_returns_shape_and_get_matches(self, engine):
        out = await engine.session_set_project_parent("child", "parent", description="kid")
        assert out["project_name"] == "child"
        assert out["parent_project_name"] == "parent"
        assert out["ancestors"] == ["parent"]
        assert out["children"] == []
        got = await engine.session_get_project("child")
        assert got == out
        parent = await engine.session_get_project("parent")
        assert parent["parent_project_name"] is None
        assert parent["children"] == ["child"]
        assert parent["ancestors"] == []

    async def test_clear_parent_and_keep_description(self, engine):
        await engine.session_set_project_parent("child", "parent", description="kid")
        cleared = await engine.session_set_project_parent("child", None)
        assert cleared["parent_project_name"] is None
        assert cleared["ancestors"] == []
        assert cleared["description"] == "kid"
        assert (await engine.session_get_project("parent"))["children"] == []

    async def test_unknown_project_get_is_empty_not_error(self, engine):
        got = await engine.session_get_project("nobody")
        assert got["parent_project_name"] is None
        assert got["ancestors"] == [] and got["children"] == []

    @pytest.mark.parametrize(
        ("name", "parent"), [("", "p"), ("  ", "p"), ("c", ""), ("c", "   "), ("c", "c")]
    )
    async def test_bad_names_rejected(self, engine, name, parent):
        out = await engine.session_set_project_parent(name, parent)
        assert out["status"] == "error"
        assert await engine.database.get_project(name.strip() or "c") is None

    async def test_cycle_rejected_and_nothing_written(self, engine):
        await engine.session_set_project_parent("b", "a")
        await engine.session_set_project_parent("c", "b")
        out = await engine.session_set_project_parent("a", "c")
        assert out["status"] == "error"
        assert "cycle" in out["message"]
        assert await engine.database.get_project("a") is None
        direct = await engine.session_set_project_parent("a", "b")
        assert direct["status"] == "error"

    async def test_reparent_moves_child(self, engine):
        await engine.session_set_project_parent("c", "p1")
        out = await engine.session_set_project_parent("c", "p2")
        assert out["ancestors"] == ["p2"]
        assert (await engine.session_get_project("p1"))["children"] == []


@pytest.mark.regression
class TestAncestorChain:
    async def test_multi_level_order(self, engine):
        await engine.session_set_project_parent("c", "b")
        await engine.session_set_project_parent("b", "a")
        assert await engine.database.get_project_ancestors("c") == ["b", "a"]

    async def test_depth_cap(self, engine):
        for i in range(8):
            await engine.session_set_project_parent(f"p{i}", f"p{i + 1}")
        chain = await engine.database.get_project_ancestors("p0")
        assert len(chain) == MAX_PROJECT_DEPTH == 5
        assert chain == ["p1", "p2", "p3", "p4", "p5"]

    async def test_walk_stops_on_pre_existing_cycle(self, engine):
        # Bypass the engine guard to simulate corrupt data.
        await engine.database.set_project_parent("x", "y")
        await engine.database.set_project_parent("y", "x")
        assert await engine.database.get_project_ancestors("x") == ["y"]


@pytest.mark.regression
class TestRecallInheritance:
    async def _family(self, engine, tmp_path):
        parent_path = _dir(tmp_path, "parent-wt")
        child_path = _dir(tmp_path, "child-wt")
        await _seed(engine, "parent", parent_path, "p")
        await _seed(engine, "child", child_path, "c")
        await engine.session_set_project_parent("child", "parent")
        return parent_path, child_path

    async def test_child_recall_includes_parent_tagged_own_first(self, engine, tmp_path):
        await self._family(engine, tmp_path)
        recalled = await engine.session_recall("child")
        assert _labels(recalled) == {
            "decisions": ["decision-c-0", "decision-p-0"],
            "learnings": ["learning-c", "learning-p"],
            "notebooks": ["notebook-c", "notebook-p"],
        }
        for tier in TIERS:
            own, inherited = recalled[tier]
            assert not own.get("inherited_from")
            assert inherited["inherited_from"] == "parent"
        assert recalled["ancestors"] == ["parent"]
        assert recalled["counts"] == {"sessions": 1, "decisions": 2, "learnings": 2, "notebooks": 2}

    async def test_sessions_tier_stays_own_only(self, engine, tmp_path):
        await self._family(engine, tmp_path)
        recalled = await engine.session_recall("child")
        assert len(recalled["sessions"]) == 1
        assert all("inherited_from" not in s for s in recalled["sessions"])

    async def test_grandparent_included_after_parent(self, engine, tmp_path):
        await _seed(engine, "grand", _dir(tmp_path, "g"), "g")
        await _seed(engine, "mid", _dir(tmp_path, "m"), "m")
        await _seed(engine, "leaf", _dir(tmp_path, "l"), "l")
        await engine.session_set_project_parent("mid", "grand")
        await engine.session_set_project_parent("leaf", "mid")
        recalled = await engine.session_recall("leaf")
        assert [d["description"] for d in recalled["decisions"]] == [
            "decision-l-0",
            "decision-m-0",
            "decision-g-0",
        ]
        assert [d.get("inherited_from") for d in recalled["decisions"]] == [None, "mid", "grand"]
        assert recalled["ancestors"] == ["mid", "grand"]

    async def test_limit_respected_own_first(self, engine, tmp_path):
        await _seed(engine, "parent", _dir(tmp_path, "p"), "p", decisions=3)
        await _seed(engine, "child", _dir(tmp_path, "c"), "c", decisions=2)
        await engine.session_set_project_parent("child", "parent")

        recalled = await engine.session_recall("child", include=["decisions"], limit=3)
        got = [d["description"] for d in recalled["decisions"]]
        assert len(got) == 3
        assert all(g.startswith("decision-c") for g in got[:2])
        assert got[2].startswith("decision-p")
        assert recalled["decisions"][2]["inherited_from"] == "parent"

        full = await engine.session_recall("child", include=["decisions"], limit=2)
        assert len(full["decisions"]) == 2
        assert all("inherited_from" not in d for d in full["decisions"])

    async def test_ancestor_newest_first(self, engine, tmp_path):
        await _seed(engine, "parent", _dir(tmp_path, "p"), "p", decisions=3)
        await engine.session_set_project_parent("child", "parent")
        recalled = await engine.session_recall("child", include=["decisions"], limit=2)
        assert [d["description"] for d in recalled["decisions"]] == [
            "decision-p-2",
            "decision-p-1",
        ]

    async def test_child_project_path_does_not_drop_parent_rows(self, engine, tmp_path):
        _, child_path = await self._family(engine, tmp_path)
        recalled = await engine.session_recall("child", project_path=child_path)
        assert _labels(recalled)["decisions"] == ["decision-c-0", "decision-p-0"]
        assert _labels(recalled)["learnings"] == ["learning-c", "learning-p"]
        assert _labels(recalled)["notebooks"] == ["notebook-c", "notebook-p"]
        assert len(recalled["sessions"]) == 1

    async def test_exclude_project_paths_applies_to_ancestor_rows(self, engine, tmp_path):
        parent_path, _ = await self._family(engine, tmp_path)
        recalled = await engine.session_recall("child", exclude_project_paths=[parent_path])
        assert _labels(recalled) == {
            "decisions": ["decision-c-0"],
            "learnings": ["learning-c"],
            "notebooks": ["notebook-c"],
        }

    async def test_parent_recall_excludes_child_entries(self, engine, tmp_path):
        await self._family(engine, tmp_path)
        recalled = await engine.session_recall("parent")
        assert _labels(recalled) == {
            "decisions": ["decision-p-0"],
            "learnings": ["learning-p"],
            "notebooks": ["notebook-p"],
        }
        assert "ancestors" not in recalled
        assert all("inherited_from" not in r for t in TIERS for r in recalled[t])

    async def test_include_ancestors_false_is_todays_output(self, engine, tmp_path):
        await self._family(engine, tmp_path)
        off = await engine.session_recall("child", include_ancestors=False)
        assert _labels(off) == {
            "decisions": ["decision-c-0"],
            "learnings": ["learning-c"],
            "notebooks": ["notebook-c"],
        }
        assert "ancestors" not in off
        assert all("inherited_from" not in r for t in TIERS for r in off[t])
        direct = await engine.database.recall_project("child")
        assert off == direct

    async def test_no_ancestors_identical_to_database_recall(self, engine, tmp_path):
        await _seed(engine, "solo", _dir(tmp_path, "s"), "s")
        via_engine = await engine.session_recall("solo")
        assert via_engine == await engine.database.recall_project("solo")

    async def test_include_filter_limits_ancestor_tiers(self, engine, tmp_path):
        await self._family(engine, tmp_path)
        recalled = await engine.session_recall("child", include=["learnings"])
        assert recalled["decisions"] == [] and recalled["notebooks"] == []
        assert _labels(recalled)["learnings"] == ["learning-c", "learning-p"]


@pytest.mark.regression
class TestToolRegistry:
    def test_new_tools_registered_with_schemas(self, engine):
        registry = LeanMCPInterface(engine).tool_registry
        setter = registry["session_set_project_parent"]["schema"]
        assert setter["required"] == ["project_name"]
        assert set(setter["properties"]) == {"project_name", "parent_project_name", "description"}
        getter = registry["session_get_project"]["schema"]
        assert getter["required"] == ["project_name"]

    def test_recall_schema_has_include_ancestors(self, engine):
        prop = LeanMCPInterface(engine).tool_registry["session_recall"]["schema"]["properties"][
            "include_ancestors"
        ]
        assert prop["type"] == "boolean"
        assert prop["default"] is True

    async def test_dispatch_through_registry(self, engine):
        registry = LeanMCPInterface(engine).tool_registry
        impl = registry["session_set_project_parent"]["implementation"]
        out = await impl(project_name="c", parent_project_name="p")
        assert out["ancestors"] == ["p"]


class _FakeConn:
    def __init__(self):
        self.calls = []

    async def fetchrow(self, query, *args):
        self.calls.append(("fetchrow", query, args))
        return None

    async def fetch(self, query, *args):
        self.calls.append(("fetch", query, args))
        return []


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
        backend = PostgreSQLBackend(dsn="postgresql://localhost/fake_issue_244b")
        conn = _FakeConn()
        backend._pool = _FakePool(conn)  # type: ignore[assignment]
        return backend, conn

    def test_projects_table_in_schema(self):
        assert "CREATE TABLE IF NOT EXISTS projects" in PostgreSQLBackend.SCHEMA
        assert "parent_name TEXT" in PostgreSQLBackend.SCHEMA
        assert "REFERENCES" not in PostgreSQLBackend.SCHEMA.split("TABLE IF NOT EXISTS projects")[1]

    async def test_upsert_sql_and_args(self):
        backend, conn = self._backend()
        with pytest.raises(TypeError):  # fake returns no row; _project_row(None) fails
            await backend.set_project_parent("kid", "dad", "desc")
        kind, query, args = conn.calls[0]
        assert kind == "fetchrow"
        assert "INSERT INTO projects (name, parent_name, description)" in query
        assert "ON CONFLICT (name) DO UPDATE" in query
        assert "COALESCE(EXCLUDED.description, projects.description)" in query
        assert args == ("kid", "dad", "desc")

    async def test_get_project_and_children_sql(self):
        backend, conn = self._backend()
        assert await backend.get_project("kid") is None
        assert await backend.list_project_children("dad") == []
        assert "WHERE name = $1" in conn.calls[0][1]
        assert conn.calls[0][2] == ("kid",)
        assert "WHERE parent_name = $1" in conn.calls[1][1]
        assert conn.calls[1][2] == ("dad",)

    async def test_ancestors_walk_uses_get_project(self):
        backend, _ = self._backend()
        rows = {"c": "b", "b": "a"}

        async def fake_get(name):
            return {"parent_name": rows.get(name)}

        backend.get_project = fake_get  # type: ignore[method-assign]
        assert await backend.get_project_ancestors("c") == ["b", "a"]
