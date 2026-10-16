"""Tests for notebook lookup by session_id/title and append/insert update modes.

- session_query_notebooks: ``session_id`` and ``title`` (case-insensitive
  literal substring, SQL-side) filters, AND-combined with project_name/tags.
- session_update_notebook: ``mode`` = replace | append | insert_after_section
  (+ ``section``), read-modify-write on authored_body.

asyncio_mode = "auto" -- no @pytest.mark.asyncio decorators needed.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import httpx
import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence import DatabaseConfig
from persistence.sqlite import SQLiteBackend
from tests.persistence.conftest import POSTGRES_AVAILABLE
from transport.http_server import HTTPSessionIntelligenceServer, NotificationManager
from transport.mcp_session_manager import MCPSessionManager
from transport.security import SecurityConfig

BODY = (
    "# Top\nintro\n## Alpha\nalpha text\n### Alpha Sub\nsub text\n"
    "## Beta\nbeta text\n## Gamma\ngamma text\n"
)


def _session_row(session_id: str, project: str = "proj-a") -> dict:
    return {
        "id": session_id,
        "started": datetime.now(UTC).isoformat(),
        "project_path": f"/tmp/{project}",
        "project_name": project,
        "mode": "local",
        "status": "active",
        "metadata": {},
        "performance_metrics": {},
        "health_status": {},
    }


def _summary_row(session_id: str, title: str, body: str | None = None, **extra) -> dict:
    return {
        "session_id": session_id,
        "title": title,
        "summary_markdown": "# Derived\n\nstuff",
        "authored_body": body,
        "key_changes": [],
        "tags": ["t1"],
        "created_at": datetime.now(UTC),
        **extra,
    }


async def _seed(db, session_id, title, body=None, project="proj-a"):
    await db.save_session(_session_row(session_id, project))
    await db.save_session_summary(_summary_row(session_id, title, body))


# ---------------------------------------------------------------------------
# Backend contract: session_id / title filters (SQLite + PostgreSQL)
# ---------------------------------------------------------------------------


class _LookupContract:
    async def _populate(self, backend):
        await _seed(backend, "lk-1", "Release Notes 100%_done", project="proj-a")
        await _seed(backend, "lk-2", "Release notes plain", project="proj-b")
        await _seed(backend, "lk-3", "Unrelated", project="proj-a")

    async def test_session_id_hit_has_project_name(self, backend):
        await self._populate(backend)
        rows = await backend.query_session_summaries(session_id="lk-1")
        assert [r["session_id"] for r in rows] == ["lk-1"]
        assert rows[0]["project_name"] == "proj-a"

    async def test_session_id_miss(self, backend):
        await self._populate(backend)
        assert await backend.query_session_summaries(session_id="nope") == []

    async def test_session_id_and_mismatched_project_is_empty(self, backend):
        await self._populate(backend)
        rows = await backend.query_session_summaries(session_id="lk-1", project_name="proj-b")
        assert rows == []

    async def test_title_substring_case_insensitive(self, backend):
        await self._populate(backend)
        rows = await backend.query_session_summaries(title="RELEASE NOTES")
        assert {r["session_id"] for r in rows} == {"lk-1", "lk-2"}

    async def test_title_and_project_name(self, backend):
        await self._populate(backend)
        rows = await backend.query_session_summaries(title="release", project_name="proj-b")
        assert [r["session_id"] for r in rows] == ["lk-2"]

    async def test_title_and_tags(self, backend):
        await self._populate(backend)
        rows = await backend.query_session_summaries(title="unrel", tags=["t1"])
        assert [r["session_id"] for r in rows] == ["lk-3"]
        assert await backend.query_session_summaries(title="unrel", tags=["zzz"]) == []

    async def test_title_percent_and_underscore_are_literal(self, backend):
        await self._populate(backend)
        pct = await backend.query_session_summaries(title="100%")
        assert [r["session_id"] for r in pct] == ["lk-1"]
        under = await backend.query_session_summaries(title="_done")
        assert [r["session_id"] for r in under] == ["lk-1"]
        # '%' / '_' must not act as wildcards
        assert await backend.query_session_summaries(title="Rel%Notes") == []
        assert await backend.query_session_summaries(title="Releas_ notes") == []

    async def test_limit_applies_after_title_filter(self, backend):
        await self._populate(backend)
        rows = await backend.query_session_summaries(title="release", limit=1)
        assert len(rows) == 1


class TestSQLiteLookup(_LookupContract):
    @pytest.fixture
    async def backend(self, tmp_path):
        db = SQLiteBackend(str(tmp_path / "lookup.db"))
        await db.initialize()
        yield db
        await db.close()


@pytest.mark.postgresql
@pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not available")
class TestPostgreSQLLookup(_LookupContract):
    @pytest.fixture
    async def backend(self):
        from persistence.postgresql import PostgreSQLBackend

        db = PostgreSQLBackend(dsn=os.environ["POSTGRES_DSN"])
        await db.initialize()
        yield db
        await db.close()

    async def _populate(self, backend):
        # Shared PG DB: unique ids/titles/projects would be ideal; ids are fixed
        # here so clear any leftovers first.
        pool = backend._ensure_connected()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM session_summaries WHERE session_id LIKE 'lk-%'")
            await conn.execute("DELETE FROM sessions WHERE id LIKE 'lk-%'")
        await super()._populate(backend)


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "nb_lookup_append.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def engine(tmp_path, db):
    return SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )


# ---------------------------------------------------------------------------
# session_query_notebooks
# ---------------------------------------------------------------------------


async def test_query_by_session_id_hit_and_miss(engine, db):
    await _seed(db, "q-1", "One", BODY)
    await _seed(db, "q-2", "Two", BODY)
    hit = await engine.session_query_notebooks(session_id="q-2")
    assert [r["session_id"] for r in hit] == ["q-2"]
    assert hit[0]["project_name"] == "proj-a"
    assert await engine.session_query_notebooks(session_id="nope") == []


async def test_query_by_session_id_with_section_and_outline(engine, db):
    await _seed(db, "q-1", "One", BODY)
    sec = await engine.session_query_notebooks(session_id="q-1", section="Beta")
    assert sec[0]["authored_body"] == "## Beta\nbeta text\n"
    outline = await engine.session_query_notebooks(session_id="q-1", outline=True)
    assert [h["heading"] for h in outline[0]["outline"]][:2] == ["Top", "Alpha"]


async def test_query_session_id_and_mismatching_project_is_empty(engine, db):
    await _seed(db, "q-1", "One", BODY, project="proj-a")
    assert await engine.session_query_notebooks(session_id="q-1", project_name="proj-b") == []
    both = await engine.session_query_notebooks(session_id="q-1", project_name="proj-a")
    assert len(both) == 1


async def test_query_title_substring_and_blank_rejected(engine, db):
    await _seed(db, "q-1", "Quarterly Review", BODY)
    await _seed(db, "q-2", "Other", BODY)
    rows = await engine.session_query_notebooks(title="quarterly")
    assert [r["session_id"] for r in rows] == ["q-1"]
    with pytest.raises(ValueError):
        await engine.session_query_notebooks(title="   ")


# ---------------------------------------------------------------------------
# session_update_notebook modes
# ---------------------------------------------------------------------------


async def _body(db, sid):
    return (await db.get_session_summary(sid))["authored_body"]


async def test_default_mode_replaces(engine, db):
    await _seed(db, "u-1", "T", BODY)
    result = await engine.session_update_notebook(session_id="u-1", body="new")
    assert result["status"] == "success"
    assert result["mode"] == "replace"
    assert await _body(db, "u-1") == "new"


async def test_append_to_existing(engine, db):
    await _seed(db, "u-1", "T", "line one\n\n")
    result = await engine.session_update_notebook(session_id="u-1", body="line two", mode="append")
    assert result["status"] == "success"
    assert result["mode"] == "append"
    assert await _body(db, "u-1") == "line one\n\nline two"


async def test_append_to_empty_body(engine, db):
    await _seed(db, "u-1", "T", None)
    await engine.session_update_notebook(session_id="u-1", body="first", mode="append")
    assert await _body(db, "u-1") == "first"


async def test_append_with_title_replaces_title(engine, db):
    await _seed(db, "u-1", "T", "x")
    await engine.session_update_notebook(
        session_id="u-1", body="y", mode="append", title="New Title"
    )
    row = await db.get_session_summary("u-1")
    assert row["title"] == "New Title"
    assert row["authored_body"] == "x\n\ny"


async def test_insert_after_middle_section(engine, db):
    await _seed(db, "u-1", "T", BODY)
    result = await engine.session_update_notebook(
        session_id="u-1", body="ADDED", mode="insert_after_section", section="alpha"
    )
    assert result["status"] == "success"
    assert result["inserted_after"] == "Alpha"
    assert await _body(db, "u-1") == (
        "# Top\nintro\n## Alpha\nalpha text\n### Alpha Sub\nsub text\n\nADDED\n\n"
        "## Beta\nbeta text\n## Gamma\ngamma text\n"
    )


async def test_insert_after_last_section(engine, db):
    await _seed(db, "u-1", "T", BODY)
    result = await engine.session_update_notebook(
        session_id="u-1", body="TAIL", mode="insert_after_section", section="Gamma"
    )
    assert result["inserted_after"] == "Gamma"
    assert await _body(db, "u-1") == BODY.rstrip("\n") + "\n\nTAIL\n"


@pytest.mark.parametrize("section", ["nonexistent", "a"])
async def test_insert_missing_or_ambiguous_section_errors_and_writes_nothing(engine, db, section):
    # "a" matches Alpha, Alpha Sub, Beta, Gamma (ambiguous); the other matches nothing
    await _seed(db, "u-1", "T", BODY)
    result = await engine.session_update_notebook(
        session_id="u-1", body="X", mode="insert_after_section", section=section
    )
    assert result["status"] == "error"
    assert "Available headings" in result["message"]
    assert await _body(db, "u-1") == BODY


async def test_section_with_replace_mode_errors(engine, db):
    await _seed(db, "u-1", "T", BODY)
    result = await engine.session_update_notebook(session_id="u-1", body="X", section="Alpha")
    assert result["status"] == "error"
    assert await _body(db, "u-1") == BODY


async def test_insert_requires_section_and_body(engine, db):
    await _seed(db, "u-1", "T", BODY)
    no_section = await engine.session_update_notebook(
        session_id="u-1", body="X", mode="insert_after_section"
    )
    no_body = await engine.session_update_notebook(
        session_id="u-1", mode="append", title="only title"
    )
    assert no_section["status"] == "error"
    assert no_body["status"] == "error"
    row = await db.get_session_summary("u-1")
    assert row["authored_body"] == BODY
    assert row["title"] == "T"


async def test_invalid_mode_errors(engine, db):
    await _seed(db, "u-1", "T", BODY)
    result = await engine.session_update_notebook(session_id="u-1", body="X", mode="bogus")
    assert result["status"] == "error"


async def test_append_without_notebook_errors(engine):
    result = await engine.session_update_notebook(session_id="ghost", body="X", mode="append")
    assert result["status"] == "error"
    assert "session_create_notebook" in result["message"]


# ---------------------------------------------------------------------------
# Schema + HTTP path validation
# ---------------------------------------------------------------------------


def test_schemas_declare_new_params(engine):
    registry = LeanMCPInterface(engine).tool_registry
    q = registry["session_query_notebooks"]["schema"]["properties"]
    assert q["session_id"]["type"] == "string"
    assert q["title"]["type"] == "string"
    u = registry["session_update_notebook"]["schema"]["properties"]
    assert u["mode"]["enum"] == ["replace", "append", "insert_after_section"]
    assert u["section"]["type"] == "string"


@pytest.fixture
async def asgi_client(tmp_path, db, engine):
    lean = LeanMCPInterface(engine)
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
    app.state.session_engine = engine
    app.state.lean_interface = lean
    app.state.mcp_session_manager = MCPSessionManager(db)
    app.state.notification_manager = NotificationManager()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client


async def _http_execute(client, tool_name, parameters):
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
                "arguments": {"tool_name": tool_name, "parameters": parameters},
            },
        },
    )
    assert resp.status_code == 200, resp.text
    return json.loads(resp.json()["result"]["content"][0]["text"])


async def test_http_accepts_new_query_and_update_params(asgi_client, db):
    await _seed(db, "h-1", "HTTP Nb", BODY)
    queried = await _http_execute(
        asgi_client, "session_query_notebooks", {"session_id": "h-1", "title": "http"}
    )
    assert queried["status"] == "success", queried
    assert [r["session_id"] for r in queried["result"]] == ["h-1"]

    updated = await _http_execute(
        asgi_client,
        "session_update_notebook",
        {
            "session_id": "h-1",
            "body": "VIA HTTP",
            "mode": "insert_after_section",
            "section": "Beta",
        },
    )
    assert updated["status"] == "success", updated
    assert updated["result"]["inserted_after"] == "Beta"
