"""
Regression tests for issue #244 (server half): session_recall exclude_project_paths.

SQLite is exercised for real; PostgreSQL through a fake asyncpg pool that
records SQL and bound args (SQLite columns for project_path are NOT NULL, so
the NULL-path guarantee is asserted on the PG SQL shape).
"""

import json

import httpx
import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence import DatabaseConfig
from persistence.postgresql import PostgreSQLBackend
from persistence.sqlite import SQLiteBackend
from transport.http_server import HTTPSessionIntelligenceServer, NotificationManager
from transport.mcp_session_manager import MCPSessionManager
from transport.security import SecurityConfig

PROJECT = "proj-244"
SECTIONS = ("sessions", "decisions", "learnings", "notebooks")


@pytest.fixture
async def engine(tmp_path):
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
    eng.database = SQLiteBackend(str(tmp_path / "test.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


@pytest.fixture
def paths(tmp_path):
    out = []
    for name in ("wt-a", "wt-b", "wt-c"):
        d = tmp_path / name
        d.mkdir()
        out.append(str(d))
    return out


async def _seed(engine, path, tag, decisions=1):
    created = await engine.session_manage_lifecycle(
        "create", project_name=PROJECT, project_path=path, session_name=f"sess-{tag}"
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


def _texts(recalled):
    return {
        "sessions": {s["id"] for s in recalled["sessions"]},
        "decisions": {d["description"] for d in recalled["decisions"]},
        "learnings": {item["learning_content"] for item in recalled["learnings"]},
        "notebooks": {n["title"] for n in recalled["notebooks"]},
    }


@pytest.mark.regression
class TestSqliteExclusion:
    async def test_excluded_path_dropped_in_all_four_sections(self, engine, paths):
        a, b, _ = paths
        sid_a = await _seed(engine, a, "a")
        sid_b = await _seed(engine, b, "b")

        recalled = await engine.session_recall(PROJECT, exclude_project_paths=[a])
        assert _texts(recalled) == {
            "sessions": {sid_b},
            "decisions": {"decision-b-0"},
            "learnings": {"learning-b"},
            "notebooks": {"notebook-b"},
        }
        assert sid_a not in _texts(recalled)["sessions"]

    async def test_multiple_exclusions(self, engine, paths):
        a, b, c = paths
        await _seed(engine, a, "a")
        await _seed(engine, b, "b")
        sid_c = await _seed(engine, c, "c")

        recalled = await engine.session_recall(PROJECT, exclude_project_paths=[a, b])
        assert _texts(recalled)["sessions"] == {sid_c}
        assert _texts(recalled)["decisions"] == {"decision-c-0"}

    async def test_empty_list_equals_not_given(self, engine, paths):
        await _seed(engine, paths[0], "a")
        await _seed(engine, paths[1], "b")
        plain = await engine.session_recall(PROJECT)
        empty = await engine.session_recall(PROJECT, exclude_project_paths=[])
        assert plain["counts"] == empty["counts"]
        assert plain["counts"]["decisions"] == 2

    async def test_trailing_slash_is_normalised(self, engine, paths):
        a, b, _ = paths
        await _seed(engine, a, "a")
        await _seed(engine, b, "b")
        recalled = await engine.session_recall(PROJECT, exclude_project_paths=[a + "/"])
        assert _texts(recalled)["decisions"] == {"decision-b-0"}

    @pytest.mark.parametrize("bad", ["relative/dir", "", "   ", "_unknown_"])
    async def test_bad_entry_rejected(self, engine, paths, bad):
        await _seed(engine, paths[0], "a")
        recalled = await engine.session_recall(PROJECT, exclude_project_paths=[paths[0], bad])
        assert "exclude_project_paths" in recalled["error"]
        assert "not an absolute path" in recalled["error"]
        assert recalled["counts"] == {"sessions": 0, "decisions": 0, "learnings": 0, "notebooks": 0}

    async def test_more_than_50_rejected_and_50_accepted(self, engine, paths):
        await _seed(engine, paths[0], "a")
        too_many = [f"/x/{i}" for i in range(51)]
        recalled = await engine.session_recall(PROJECT, exclude_project_paths=too_many)
        assert "max 50" in recalled["error"]
        ok = await engine.session_recall(PROJECT, exclude_project_paths=too_many[:50])
        assert "error" not in ok
        assert ok["counts"]["decisions"] == 1

    async def test_combined_with_project_path_is_and(self, engine, paths):
        a, b, _ = paths
        await _seed(engine, a, "a")
        await _seed(engine, b, "b")

        same = await engine.session_recall(PROJECT, project_path=a, exclude_project_paths=[a])
        assert same["counts"] == {"sessions": 0, "decisions": 0, "learnings": 0, "notebooks": 0}
        other = await engine.session_recall(PROJECT, project_path=a, exclude_project_paths=[b])
        assert _texts(other)["decisions"] == {"decision-a-0"}

    async def test_limit_applies_after_exclusion(self, engine, paths):
        a, b, _ = paths
        await _seed(engine, b, "b", decisions=2)
        await _seed(engine, a, "a", decisions=3)  # newest decisions live in the excluded path

        recalled = await engine.session_recall(
            PROJECT, include=["decisions"], limit=2, exclude_project_paths=[a]
        )
        assert len(recalled["decisions"]) == 2
        assert all(d["description"].startswith("decision-b") for d in recalled["decisions"])


class _FakeConn:
    def __init__(self):
        self.calls = []

    async def fetch(self, query, *args):
        self.calls.append((query, args))
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
        backend = PostgreSQLBackend(dsn="postgresql://localhost/fake_issue_244")
        conn = _FakeConn()
        backend._pool = _FakePool(conn)  # type: ignore[assignment]
        return backend, conn

    async def test_exclusion_bound_in_every_section_with_null_guard(self):
        backend, conn = self._backend()
        await backend.recall_project(
            PROJECT, project_path="/wt/a", exclude_project_paths=["/wt/x", "/wt/y"]
        )
        sessions, decisions, learnings, notebooks = conn.calls
        for (query, args), col, n in (
            (sessions, "project_path", 5),
            (decisions, "COALESCE(d.project_path, s.project_path)", 5),
            (notebooks, "COALESCE(ss.project_path, s.project_path)", 5),
            (learnings, "project_path", 4),
        ):
            assert f"({col} IS NULL OR NOT ({col} = ANY(${n}::text[])))" in query
            assert args[n - 1] == ["/wt/x", "/wt/y"]
            assert args[2 if col == "project_path" and n == 4 else 3] == "/wt/a"

    async def test_no_exclusion_leaves_sql_and_args_unchanged(self):
        backend, conn = self._backend()
        await backend.recall_project(PROJECT, exclude_project_paths=[])
        for query, _args in conn.calls:
            assert "ANY(" not in query
        assert [len(a) for _, a in conn.calls] == [4, 4, 3, 4]

    async def test_limit_is_a_bound_param_not_pre_applied(self):
        backend, conn = self._backend()
        await backend.recall_project(PROJECT, limit=7, exclude_project_paths=["/wt/x"])
        sessions, _, learnings, _ = conn.calls
        assert sessions[1][2] == 7
        assert learnings[1][1] == 7


@pytest.mark.regression
class TestSchemaAndHttp:
    def test_lean_schema_exposes_exclude_project_paths(self, engine):
        spec = LeanMCPInterface(engine).tool_registry["session_recall"]["schema"]
        prop = spec["properties"]["exclude_project_paths"]
        assert prop["type"] == "array"
        assert prop["items"] == {"type": "string"}
        assert prop["maxItems"] == 50

    async def test_http_execute_tool_accepts_param(self, tmp_path, paths):
        db = SQLiteBackend(str(tmp_path / "http.db"))
        await db.initialize()
        eng = SessionIntelligenceEngine(
            repository_path=str(tmp_path), use_filesystem=False, database=db
        )
        try:
            await _seed(eng, paths[0], "a")
            await _seed(eng, paths[1], "b")
            server = HTTPSessionIntelligenceServer(
                host="127.0.0.1",
                port=4099,
                repository_path=str(tmp_path),
                db_config=DatabaseConfig(),
                security_config=SecurityConfig(
                    localhost_only=False, allowed_origins=["*"], require_api_key=False
                ),
            )
            app = server.create_app()
            app.state.database = db
            app.state.session_engine = eng
            app.state.lean_interface = LeanMCPInterface(eng)
            app.state.mcp_session_manager = MCPSessionManager(db)
            app.state.notification_manager = NotificationManager()

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                init = await client.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {"clientInfo": {"name": "pytest", "version": "0.0.1"}},
                    },
                )
                resp = await client.post(
                    "/mcp",
                    headers={"MCP-Session-Id": init.headers["MCP-Session-Id"]},
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "execute_tool",
                            "arguments": {
                                "tool_name": "session_recall",
                                "parameters": {
                                    "project_name": PROJECT,
                                    "exclude_project_paths": [paths[0]],
                                },
                            },
                        },
                    },
                )
            envelope = json.loads(resp.json()["result"]["content"][0]["text"])
            assert envelope["status"] == "success"
            assert _texts(envelope["result"])["decisions"] == {"decision-b-0"}
        finally:
            await db.close()
