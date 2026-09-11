"""
Integration tests for the HTTP transport layer.

Uses httpx.AsyncClient against the real ASGI app via ASGITransport.
We build a minimal FastAPI app that reuses the same routing functions
as HTTPSessionIntelligenceServer but injects SQLite state directly —
avoiding the PostgreSQL lifespan altogether.

asyncio_mode = "auto" — no @pytest.mark.asyncio decorators needed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.sqlite import SQLiteBackend
from transport.http_server import HTTPSessionIntelligenceServer, NotificationManager
from transport.mcp_session_manager import MCPSessionManager
from transport.security import SecurityConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    """Initialised SQLite backend."""
    backend = SQLiteBackend(str(tmp_path / "test_http.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def app(tmp_path, db):
    """
    FastAPI app with pre-populated state using SQLite.

    We create the HTTPSessionIntelligenceServer, call create_app() to get all
    routes/middleware registered, then directly set app.state before any
    request is made.  Because httpx.ASGITransport does NOT trigger the
    lifespan (startup/shutdown events) the state we inject here is the only
    state ever used.
    """
    engine = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    lean = LeanMCPInterface(engine)
    mcp_mgr = MCPSessionManager(db)
    notif_mgr = NotificationManager()

    sc = SecurityConfig(
        localhost_only=False,
        allowed_origins=["*"],
        require_api_key=False,
    )
    from persistence import DatabaseConfig

    server = HTTPSessionIntelligenceServer(
        host="127.0.0.1",
        port=4099,
        repository_path=str(tmp_path),
        db_config=DatabaseConfig(),
        security_config=sc,
    )
    # create_app() registers all routes and middleware
    fastapi_app = server.create_app()

    # Inject state directly — httpx ASGITransport never fires lifespan events
    fastapi_app.state.database = db
    fastapi_app.state.session_engine = engine
    fastapi_app.state.lean_interface = lean
    fastapi_app.state.mcp_session_manager = mcp_mgr
    fastapi_app.state.notification_manager = notif_mgr

    yield fastapi_app


@pytest.fixture
async def asgi_client(app):
    """AsyncClient wired to the pre-seeded ASGI app."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mcp_body(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}


async def _initialize_mcp(client: httpx.AsyncClient) -> str:
    """Run MCP initialize handshake, return the new session ID."""
    resp = await client.post(
        "/mcp",
        json=_mcp_body("initialize", {"clientInfo": {"name": "pytest", "version": "0.0.1"}}),
    )
    assert resp.status_code == 200, f"initialize failed: {resp.text}"
    return resp.headers["MCP-Session-Id"]


# ===========================================================================
# Health endpoint
# ===========================================================================


async def test_health_returns_200(asgi_client):
    """GET /health returns HTTP 200."""
    resp = await asgi_client.get("/health")
    assert resp.status_code == 200


async def test_health_body_has_status_healthy(asgi_client):
    """GET /health body contains status=healthy."""
    resp = await asgi_client.get("/health")
    assert resp.json()["status"] == "healthy"


async def test_health_body_has_timestamp(asgi_client):
    """GET /health body includes a non-empty timestamp field."""
    resp = await asgi_client.get("/health")
    body = resp.json()
    assert "timestamp" in body
    assert body["timestamp"]


async def test_health_database_connected(asgi_client):
    """GET /health reports database as connected when SQLite is live."""
    resp = await asgi_client.get("/health")
    assert resp.json()["database"] == "connected"


# ===========================================================================
# MCP initialize handshake
# ===========================================================================


async def test_mcp_initialize_returns_session_id_header(asgi_client):
    """POST /mcp initialize returns MCP-Session-Id response header."""
    resp = await asgi_client.post(
        "/mcp",
        json=_mcp_body("initialize", {"clientInfo": {"name": "pytest"}}),
    )
    assert resp.status_code == 200
    assert "MCP-Session-Id" in resp.headers
    assert resp.headers["MCP-Session-Id"]


async def test_mcp_initialize_result_has_protocol_version(asgi_client):
    """initialize result contains protocolVersion and serverInfo."""
    resp = await asgi_client.post(
        "/mcp",
        json=_mcp_body("initialize", {"clientInfo": {"name": "pytest"}}),
    )
    result = resp.json()["result"]
    assert "protocolVersion" in result
    assert result["serverInfo"]["name"] == "session-intelligence"


# ===========================================================================
# tools/list
# ===========================================================================


async def test_tools_list_returns_meta_tools(asgi_client):
    """tools/list response includes the 3 meta-tools by name."""
    session_id = await _initialize_mcp(asgi_client)
    resp = await asgi_client.post(
        "/mcp",
        headers={"MCP-Session-Id": session_id},
        json=_mcp_body("tools/list"),
    )
    assert resp.status_code == 200
    names = [t["name"] for t in resp.json()["result"]["tools"]]
    assert "discover_tools" in names
    assert "get_tool_spec" in names
    assert "execute_tool" in names


# ===========================================================================
# execute_tool via MCP tools/call
# ===========================================================================


async def test_call_discover_tools_returns_tool_list(asgi_client):
    """tools/call → discover_tools returns available_tools list."""
    session_id = await _initialize_mcp(asgi_client)
    resp = await asgi_client.post(
        "/mcp",
        headers={"MCP-Session-Id": session_id},
        json=_mcp_body(
            "tools/call",
            {
                "name": "discover_tools",
                "arguments": {"pattern": "session"},
            },
        ),
    )
    assert resp.status_code == 200
    payload = json.loads(resp.json()["result"]["content"][0]["text"])
    assert "available_tools" in payload


async def test_execute_tool_dispatches_session_lifecycle(asgi_client):
    """execute_tool dispatching session_manage_lifecycle returns a result."""
    session_id = await _initialize_mcp(asgi_client)
    resp = await asgi_client.post(
        "/mcp",
        headers={"MCP-Session-Id": session_id},
        json=_mcp_body(
            "tools/call",
            {
                "name": "execute_tool",
                "arguments": {
                    "tool_name": "session_manage_lifecycle",
                    "parameters": {"operation": "create", "project_name": "http-test"},
                },
            },
        ),
    )
    assert resp.status_code == 200
    payload = json.loads(resp.json()["result"]["content"][0]["text"])
    assert payload["status"] == "success"


async def test_execute_unknown_tool_returns_error(asgi_client):
    """execute_tool with an unknown tool_name returns an error payload."""
    session_id = await _initialize_mcp(asgi_client)
    resp = await asgi_client.post(
        "/mcp",
        headers={"MCP-Session-Id": session_id},
        json=_mcp_body(
            "tools/call",
            {
                "name": "execute_tool",
                "arguments": {
                    "tool_name": "nonexistent_tool_xyz",
                    "parameters": {},
                },
            },
        ),
    )
    assert resp.status_code == 200
    payload = json.loads(resp.json()["result"]["content"][0]["text"])
    assert "error" in payload


# ===========================================================================
# Error cases
# ===========================================================================


async def test_invalid_json_returns_400(asgi_client):
    """POST /mcp with malformed JSON returns 400 parse error (-32700)."""
    resp = await asgi_client.post(
        "/mcp",
        content=b"this is not json!!!",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == -32700


async def test_missing_session_id_returns_400(asgi_client):
    """Non-initialize MCP request without MCP-Session-Id header returns 400."""
    resp = await asgi_client.post(
        "/mcp",
        json=_mcp_body("tools/list"),
    )
    assert resp.status_code == 400
    assert "Missing MCP-Session-Id" in resp.json()["error"]["message"]


async def test_invalid_session_id_returns_401(asgi_client):
    """MCP request with a bogus session ID returns 401."""
    resp = await asgi_client.post(
        "/mcp",
        headers={"MCP-Session-Id": "totally-bogus-id-99999"},
        json=_mcp_body("tools/list"),
    )
    assert resp.status_code == 401


async def test_unknown_mcp_method_returns_500(asgi_client):
    """Unrecognised MCP method returns a 500 JSON-RPC error response."""
    session_id = await _initialize_mcp(asgi_client)
    resp = await asgi_client.post(
        "/mcp",
        headers={"MCP-Session-Id": session_id},
        json=_mcp_body("unknown/method"),
    )
    assert resp.status_code == 500
    assert "error" in resp.json()


# ===========================================================================
# REST session endpoints
# ===========================================================================


async def test_list_sessions_endpoint_returns_list(asgi_client):
    """GET /api/sessions returns a sessions list."""
    resp = await asgi_client.get("/api/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert "sessions" in body
    assert isinstance(body["sessions"], list)


async def test_get_unknown_session_returns_404(asgi_client):
    """GET /api/sessions/{id} with unknown id returns 404."""
    resp = await asgi_client.get("/api/sessions/does-not-exist-abc123xyz")
    assert resp.status_code == 404


# ===========================================================================
# Concurrent requests
# ===========================================================================


async def test_concurrent_health_checks(asgi_client):
    """Five simultaneous GET /health calls all return 200 and status=healthy."""
    responses = await asyncio.gather(*[asgi_client.get("/health") for _ in range(5)])
    for resp in responses:
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


# ===========================================================================
# Race condition regressions (#43): "dictionary changed size during iteration"
# ===========================================================================
#
# Both _persist_sessions_to_database and NotificationManager.broadcast iterate
# a shared dict while awaiting inside the loop body. If a concurrent coroutine
# mutates that same dict during the await, Python raises
# RuntimeError: dictionary changed size during iteration. The fix wraps the
# dict in list(...) before iterating to snapshot keys/values up front.


class _FakeState:
    def __init__(self, database, session_engine):
        self.database = database
        self.session_engine = session_engine


class _FakeApp:
    def __init__(self, state):
        self.state = state


class _FakeRequest:
    """Minimal stand-in for fastapi.Request — only app.state is used."""

    def __init__(self, database, session_engine):
        self.app = _FakeApp(_FakeState(database, session_engine))


def _make_session(session_id: str, project_path: str):
    from models.session_models import Session, SessionMetadata

    return Session(
        id=session_id,
        started=datetime.now(),
        project_name="race-test",
        project_path=project_path,
        metadata=SessionMetadata(session_type="development", environment="local", user="user"),
    )


async def test_persist_sessions_survives_cache_mutation_during_iteration(tmp_path, db):
    """_persist_sessions_to_database must not raise when the session cache is
    mutated (a new session inserted) by a concurrent request mid-iteration."""
    engine = SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )
    engine.session_cache["s1"] = _make_session("s1", str(tmp_path))
    engine.session_cache["s2"] = _make_session("s2", str(tmp_path))

    persisted_ids: list[str] = []
    original_save_session = db.save_session
    call_count = {"n": 0}

    async def racy_save_session(session_data):
        call_count["n"] += 1
        persisted_ids.append(session_data["id"])
        if call_count["n"] == 1:
            # Simulate a concurrent request adding a new session mid-iteration.
            engine.session_cache["s3-concurrent"] = _make_session(
                "s3-concurrent", str(tmp_path)
            )
        await original_save_session(session_data)

    db.save_session = racy_save_session

    from persistence import DatabaseConfig

    server = HTTPSessionIntelligenceServer(
        host="127.0.0.1",
        port=4099,
        repository_path=str(tmp_path),
        db_config=DatabaseConfig(),
        security_config=SecurityConfig(
            localhost_only=False, allowed_origins=["*"], require_api_key=False
        ),
    )
    request = _FakeRequest(database=db, session_engine=engine)

    # Should complete without RuntimeError: dictionary changed size during iteration
    await server._persist_sessions_to_database(request)

    assert "s1" in persisted_ids
    assert "s2" in persisted_ids


async def test_broadcast_survives_subscriber_mutation_during_iteration():
    """NotificationManager.broadcast must not raise when a subscriber queue is
    removed (client disconnect) by another coroutine mid-broadcast."""
    manager = NotificationManager()

    class DisconnectingQueue(asyncio.Queue):
        """A queue whose first .put() simulates another client disconnecting,
        removing a different entry from manager._subscribers mid-broadcast."""

        def __init__(self, notification_manager: NotificationManager, key_to_remove: str):
            super().__init__()
            self._manager = notification_manager
            self._key_to_remove = key_to_remove
            self._put_calls = 0

        async def put(self, item):
            self._put_calls += 1
            if self._put_calls == 1:
                self._manager._subscribers.pop(self._key_to_remove, None)
            await super().put(item)

    manager._subscribers["sub-a"] = DisconnectingQueue(manager, key_to_remove="sub-b")
    manager._subscribers["sub-b"] = asyncio.Queue()

    # Should complete without RuntimeError: dictionary changed size during iteration
    await manager.broadcast("test_event", {"foo": "bar"})

    assert "sub-b" not in manager._subscribers
    assert manager._subscribers["sub-a"].qsize() == 1
