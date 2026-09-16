"""
Regression tests for issue #127: the HTTP transport's `_handle_tool_call`
reimplements the meta-tool trio inline instead of delegating to
`LeanMCPInterface.execute_tool`, and its `execute_tool` branch built the
outer envelope unconditionally as `status="success"` regardless of what the
dispatched tool itself reported:

    limited = apply_token_limits(tool_result, target)
    result = {"tool": target, "status": "success", "result": limited}

A tool that returns `{"status": "error", ...}` WITHOUT raising (e.g.
`session_update_notebook` given a session_id with no notebook) was reported
as an outer `status="success"` envelope over HTTP, even though
`lean_mcp_interface.py`'s stdio `execute_tool` had already been fixed (PR
#131) to reflect the inner status.

The stdio-only fix left the HTTP path broken because the two dispatchers
are separate code paths that do not share this envelope-construction logic
- unlike parameter validation (issue #61), which IS shared via
`validate_tool_parameters`. This is exactly the trap: a test against
`LeanMCPInterface.execute_tool` alone passes while production (HTTP,
port 4002) stays broken.

The fix mirrors lean_mcp_interface.py's inner-status reflection (~line
2158-2167) directly inside `_handle_tool_call` (~line 922-923), reading the
inner status from the ORIGINAL `tool_result` rather than the token-limited
`limited` copy.

These tests reuse the ASGI harness pattern from
tests/regression/test_issue_61_transport_validation_parity.py.

asyncio_mode = "auto" - no @pytest.mark.asyncio decorators needed.
"""

from __future__ import annotations

import json

import httpx
import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence import DatabaseConfig
from persistence.sqlite import SQLiteBackend
from transport.http_server import HTTPSessionIntelligenceServer, NotificationManager
from transport.mcp_session_manager import MCPSessionManager
from transport.security import SecurityConfig

# ---------------------------------------------------------------------------
# Fixtures (mirrors tests/regression/test_issue_61_transport_validation_parity.py)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    """Initialised SQLite backend."""
    backend = SQLiteBackend(str(tmp_path / "test_issue_127.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def app(tmp_path, db):
    """
    FastAPI app with pre-populated state using SQLite.

    Same construction as test_issue_61_transport_validation_parity.py:
    create the HTTPSessionIntelligenceServer, call create_app() for routes/
    middleware, then inject app.state directly (httpx.ASGITransport never
    fires lifespan events, so this injected state is the only state used).
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


async def _http_execute_tool(client: httpx.AsyncClient, tool_name: str, parameters: dict) -> dict:
    """Run execute_tool(tool_name, parameters) through `_handle_tool_call`
    over the real HTTP/MCP transport, and return the decoded envelope.

    This is the dispatcher under test: it goes through
    HTTPSessionIntelligenceServer._handle_tool_call, NOT
    LeanMCPInterface.execute_tool. That distinction is the entire point of
    #127 - the two are separate implementations of the same meta-tool.
    """
    session_id = await _initialize_mcp(client)
    resp = await client.post(
        "/mcp",
        headers={"MCP-Session-Id": session_id},
        json=_mcp_body(
            "tools/call",
            {
                "name": "execute_tool",
                "arguments": {"tool_name": tool_name, "parameters": parameters},
            },
        ),
    )
    assert resp.status_code == 200, f"tools/call failed: {resp.text}"
    return json.loads(resp.json()["result"]["content"][0]["text"])


def _get_meta_tool(interface: LeanMCPInterface, name: str):
    """Return the callable registered as a FastMCP tool by name."""
    manager = interface.app._tool_manager
    tool_obj = manager._tools[name]
    return tool_obj.fn


async def _stdio_execute_tool(
    interface: LeanMCPInterface, tool_name: str, parameters: dict
) -> dict:
    """Run execute_tool(tool_name, parameters) over the stdio meta-tool path
    (LeanMCPInterface.execute_tool), for parity comparison only."""
    execute = _get_meta_tool(interface, "execute_tool")
    return await execute(tool_name, parameters)


# ===========================================================================
# TEST 1: a tool reporting an inner error (no exception) over HTTP must
# yield an outer status="error" envelope (acceptance criterion)
# ===========================================================================


async def test_inner_error_status_over_http_yields_error_envelope(asgi_client):
    """
    Verbatim repro of #127 against the HTTP dispatcher: session_update_notebook
    given a session_id that owns no notebook returns
    `{"status": "error", ...}` as a normal return value - no exception is
    raised. Before the fix, `_handle_tool_call` unconditionally wrapped this
    in `{"status": "success", "result": {...status: error...}}`.
    """
    payload = await _http_execute_tool(
        asgi_client,
        "session_update_notebook",
        {"session_id": "does-not-exist-127", "body": "anything"},
    )

    assert payload["status"] == "error", (
        f"outer envelope must reflect the inner error status; got payload={payload}"
    )
    assert payload["result"]["status"] == "error"


# ===========================================================================
# TEST 2: a genuinely successful call over HTTP still yields status="success"
# ===========================================================================


async def test_inner_success_status_over_http_yields_success_envelope(asgi_client):
    """
    Guards against an overzealous fix: a tool call that succeeds (no
    exception, inner status != "error") must still produce an outer
    status="success" envelope over HTTP.
    """
    payload = await _http_execute_tool(
        asgi_client,
        "session_manage_lifecycle",
        {"operation": "create", "project_name": "issue-127-test-project"},
    )

    assert payload["status"] == "success", f"unexpected envelope: {payload}"


# ===========================================================================
# TEST 3: dispatcher parity - the guard that would have caught #127 in the
# first place (both dispatchers must agree on the same input)
# ===========================================================================


async def test_error_envelope_parity_between_stdio_and_http_dispatchers(app, asgi_client):
    """
    The same call, producing an inner `status="error"` result with no
    exception, must be reported identically by BOTH the stdio
    `LeanMCPInterface.execute_tool` meta-tool and the HTTP transport's
    `_handle_tool_call`. This is the parity guard: PR #131 fixed only the
    stdio side, and a test scoped to that side alone would have passed
    while HTTP (the actually-deployed transport on :4002) stayed broken.
    """
    lean_interface = app.state.lean_interface
    parameters = {"session_id": "does-not-exist-127-parity", "body": "anything"}

    stdio_result = await _stdio_execute_tool(lean_interface, "session_update_notebook", parameters)
    http_result = await _http_execute_tool(asgi_client, "session_update_notebook", parameters)

    assert stdio_result["status"] == "error"
    assert http_result["status"] == "error"
    assert stdio_result["status"] == http_result["status"]
