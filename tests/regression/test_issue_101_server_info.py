"""Regression test for issue #101 — server_info meta-tool.

Covers both transport paths (stdio/FastMCP registry building via
LeanMCPInterface.build_server_info, and the HTTP tools/list + tools/call
dispatch) to guard against the payload or tool registration drifting out
of sync between the two independent enumerations.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence import DatabaseConfig
from persistence.sqlite import SQLiteBackend
from transport.http_server import HTTPSessionIntelligenceServer, NotificationManager
from transport.mcp_session_manager import MCPSessionManager
from transport.security import SecurityConfig

REQUIRED_KEYS = {
    "name",
    "version",
    "description",
    "repository",
    "issues",
    "documentation",
    "support",
    "domains",
    "total_tools",
    "transport",
    "protocol_version",
}


# ---------------------------------------------------------------------------
# build_server_info() — single source of truth
# ---------------------------------------------------------------------------


@pytest.fixture
def interface():
    engine = MagicMock()
    return LeanMCPInterface(engine)


def test_build_server_info_has_required_keys(interface):
    info = interface.build_server_info()
    assert REQUIRED_KEYS.issubset(info.keys())


def test_build_server_info_total_tools_matches_registry(interface):
    info = interface.build_server_info()
    assert info["total_tools"] == len(interface.tool_registry)


def test_build_server_info_domains_sum_to_total(interface):
    info = interface.build_server_info()
    assert sum(info["domains"].values()) == info["total_tools"]


def test_build_server_info_default_transport_is_stdio(interface):
    info = interface.build_server_info()
    assert info["transport"] == "stdio"


def test_build_server_info_transport_reflects_argument(interface):
    info = interface.build_server_info(transport="HTTP (SSE)")
    assert info["transport"] == "HTTP (SSE)"


def test_build_server_info_version_is_nonempty_string(interface):
    info = interface.build_server_info()
    assert isinstance(info["version"], str)
    assert info["version"]


def test_build_server_info_name_and_urls(interface):
    info = interface.build_server_info()
    assert info["name"] == "session-intelligence"
    assert info["repository"] == "https://github.com/Claire-s-Monster/session-intelligence"
    assert info["issues"] == "https://github.com/Claire-s-Monster/session-intelligence/issues"
    assert info["support"]["bug_reports"].startswith(info["issues"])
    assert info["support"]["feature_requests"].startswith(info["issues"])


# ---------------------------------------------------------------------------
# HTTP transport — tools/list and tools/call
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "test_server_info.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def app(tmp_path, db):
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

    server = HTTPSessionIntelligenceServer(
        host="127.0.0.1",
        port=4099,
        repository_path=str(tmp_path),
        db_config=DatabaseConfig(),
        security_config=sc,
    )
    fastapi_app = server.create_app()

    fastapi_app.state.database = db
    fastapi_app.state.session_engine = engine
    fastapi_app.state.lean_interface = lean
    fastapi_app.state.mcp_session_manager = mcp_mgr
    fastapi_app.state.notification_manager = notif_mgr

    yield fastapi_app


@pytest.fixture
async def asgi_client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


def _mcp_body(method: str, params: dict | None = None, req_id: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}


async def _initialize_mcp(client: httpx.AsyncClient) -> str:
    resp = await client.post(
        "/mcp",
        json=_mcp_body("initialize", {"clientInfo": {"name": "pytest", "version": "0.0.1"}}),
    )
    assert resp.status_code == 200, f"initialize failed: {resp.text}"
    return resp.headers["MCP-Session-Id"]


async def test_http_tools_list_has_exactly_four_tools_including_server_info(asgi_client):
    session_id = await _initialize_mcp(asgi_client)
    resp = await asgi_client.post(
        "/mcp",
        headers={"MCP-Session-Id": session_id},
        json=_mcp_body("tools/list"),
    )
    assert resp.status_code == 200
    tools = resp.json()["result"]["tools"]
    assert len(tools) == 4

    by_name = {t["name"]: t for t in tools}
    assert "server_info" in by_name
    assert by_name["server_info"]["inputSchema"]["properties"] == {}


async def test_http_tools_call_server_info_returns_payload(asgi_client):
    session_id = await _initialize_mcp(asgi_client)
    resp = await asgi_client.post(
        "/mcp",
        headers={"MCP-Session-Id": session_id},
        json=_mcp_body(
            "tools/call",
            {"name": "server_info", "arguments": {}},
        ),
    )
    assert resp.status_code == 200
    payload = json.loads(resp.json()["result"]["content"][0]["text"])
    assert "error" not in payload
    assert REQUIRED_KEYS.issubset(payload.keys())
    assert payload["transport"] == "HTTP (SSE)"
