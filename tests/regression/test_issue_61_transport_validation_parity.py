"""
Regression tests for issue #61: an unknown/mistyped parameter passed to
execute_tool (e.g. `category` on session_log_decision, which has never been
a declared parameter of that tool's schema) was silently dropped instead of
refused.

Two independent bugs combined to make this dangerous:

1. Only the stdio meta-tool validated parameters against the declared schema
   before dispatch. The HTTP transport's `_handle_tool_call` dispatched
   straight through, so an extra/misspelled kwarg either blew up inside the
   engine call with a raw TypeError, or in some cases was simply ignored by
   `**kwargs`-style signatures - callers had no reliable signal that their
   call was wrong.
2. `_wrap_tool`/`_wrap_async_tool` caught ALL exceptions raised by a tool
   implementation and returned an `{"error": ...}` dict as the tool's
   *result*, which the transport then wrapped in a top-level
   `status="success"` envelope. A rejected/failed call could report success
   while writing nothing to the database - the exact shape of the issue #61
   report.

The fix: `LeanMCPInterface.validate_tool_parameters()` is now a single
shared pre-dispatch gate called by BOTH the stdio `execute_tool` meta-tool
and the HTTP transport's `_handle_tool_call`, and `_wrap_tool`/
`_wrap_async_tool` now log-and-reraise instead of swallowing exceptions, so
transport-level exception handlers report `status="error"` instead of a
success envelope with a buried error.

These tests reuse the ASGI harness pattern from
tests/integration/test_http_transport.py (minimal FastAPI app wired
directly to SQLite state, bypassing the PostgreSQL lifespan) rather than
inventing a new one.

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
# Fixtures (mirrors tests/integration/test_http_transport.py)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    """Initialised SQLite backend."""
    backend = SQLiteBackend(str(tmp_path / "test_issue_61.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def app(tmp_path, db):
    """
    FastAPI app with pre-populated state using SQLite.

    Same construction as tests/integration/test_http_transport.py: create
    the HTTPSessionIntelligenceServer, call create_app() for routes/
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


async def _http_execute_tool(
    client: httpx.AsyncClient, tool_name: str, parameters: dict
) -> dict:
    """Run execute_tool(tool_name, parameters) over the HTTP/MCP transport
    and return the decoded envelope."""
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
    # FastMCP stores tools in ._tool_manager._tools (dict keyed by name).
    # We access the underlying fn to call it directly in tests.
    manager = interface.app._tool_manager
    tool_obj = manager._tools[name]
    return tool_obj.fn


async def _stdio_execute_tool(
    interface: LeanMCPInterface, tool_name: str, parameters: dict
) -> dict:
    """Run execute_tool(tool_name, parameters) over the stdio meta-tool path."""
    execute = _get_meta_tool(interface, "execute_tool")
    return await execute(tool_name, parameters)


async def _count_decision_rows(db: SQLiteBackend) -> int:
    cursor = await db._connection.execute("SELECT COUNT(*) FROM decisions")
    row = await cursor.fetchone()
    return row[0]


# ===========================================================================
# TEST 1: verbatim repro from issue #61 (acceptance criterion)
# ===========================================================================


async def test_unknown_parameter_over_http_is_refused_and_writes_no_row(
    asgi_client, db, tmp_path
):
    """
    Verbatim repro of issue #61: session_log_decision does not declare a
    `category` parameter, yet a caller could pass one over HTTP and receive
    a status="success" envelope while the decision was never written -
    the extra kwarg either raised inside the engine or was dropped, and
    the exception-swallowing in _wrap_async_tool hid it either way.

    Asserts the call is now refused up front (status="error" naming the
    offending parameter) AND, critically, that no decision row is written -
    this is the actual point of the bug: the old behaviour reported success
    while persisting nothing.
    """
    before = await _count_decision_rows(db)

    # Verbatim repro from issue #61: `category` is not declared in
    # session_log_decision's schema (see lean_mcp_interface.py registry).
    payload = await _http_execute_tool(
        asgi_client,
        "session_log_decision",
        {
            "decision": "Use PostgreSQL for production",
            "context": {"reason": "scale"},
            "category": "architecture",
            "project_path": str(tmp_path),
        },
    )

    assert payload["status"] == "error"
    assert "category" in payload["error"]
    assert "unexpected" in payload["error"].lower()

    after = await _count_decision_rows(db)
    assert after == before, (
        "no decision row should have been written for a refused call; "
        f"before={before} after={after}"
    )


# ===========================================================================
# TEST 2: identical rejection over both transports (acceptance criterion)
# ===========================================================================


async def test_unknown_parameter_is_refused_identically_over_both_transports(
    app, asgi_client, tmp_path
):
    """
    The same invalid call, run through both the stdio meta-tool path and
    the HTTP transport against the SAME engine/registry, must produce not
    just the same status but the exact same error STRING. Equal strings
    prove the two transports share one validation implementation
    (validate_tool_parameters) rather than two independently-written
    checks that happen to agree today and could silently diverge later.
    """
    lean_interface = app.state.lean_interface
    parameters = {
        "decision": "Use PostgreSQL for production",
        "category": "architecture",
        "project_path": str(tmp_path),
    }

    stdio_result = await _stdio_execute_tool(
        lean_interface, "session_log_decision", parameters
    )
    http_result = await _http_execute_tool(
        asgi_client, "session_log_decision", parameters
    )

    assert stdio_result["status"] == "error"
    assert http_result["status"] == "error"
    assert stdio_result["error"] == http_result["error"]


# ===========================================================================
# TEST 3: divergence guard matrix (acceptance criterion)
# ===========================================================================
#
# Tool names below are taken directly from the registry built in
# src/lean_mcp_interface.py:
#   - session_manage_lifecycle: required=["operation"], several optional
#     params (mode, project_name, metadata, auto_recovery)
#   - session_analyze_patterns: no required params at all (scope,
#     pattern_types, include_agents, learning_mode, generate_insights are
#     all optional)

TRANSPORT_PARITY_CASES = [
    pytest.param(
        "session_manage_lifecycle",
        {"operation": "create", "bogus_param": "x"},
        id="unknown-param-on-tool-with-required-params",
    ),
    pytest.param(
        "session_analyze_patterns",
        {"bogus_param": "x"},
        id="unknown-param-on-tool-with-only-optional-params",
    ),
    pytest.param(
        "session_manage_lifecycle",
        {},
        id="missing-required-param",
    ),
    pytest.param(
        "session_manage_lifecycle",
        {"operation": "create", "foo": 1, "bar": 2},
        id="multiple-unknown-params-at-once",
    ),
    pytest.param(
        "session_manage_lifecycle",
        {"foo": 1},
        id="unknown-and-missing-together",
    ),
]


@pytest.mark.parametrize("tool_name,parameters", TRANSPORT_PARITY_CASES)
async def test_transport_validation_parity_matrix(app, asgi_client, tool_name, parameters):
    """
    Divergence guard: for a matrix of invalid-parameter shapes (unknown
    param on a tool with required params, unknown param on a tool with only
    optional params, missing required param, multiple unknown params at
    once, and unknown+missing combined), both transports must produce the
    identical (status, error) pair. This test must FAIL the moment either
    transport's dispatch path stops calling validate_tool_parameters, or
    calls it differently, since that is exactly how #61 happened - the
    stdio path validated and the HTTP path did not.
    """
    lean_interface = app.state.lean_interface

    stdio_result = await _stdio_execute_tool(lean_interface, tool_name, parameters)
    http_result = await _http_execute_tool(asgi_client, tool_name, parameters)

    stdio_pair = (stdio_result.get("status"), stdio_result.get("error"))
    http_pair = (http_result.get("status"), http_result.get("error"))

    assert stdio_pair[0] == "error"
    assert http_pair[0] == "error"
    assert stdio_pair == http_pair


# ===========================================================================
# TEST 4: an exception inside a tool never reports success over HTTP
# ===========================================================================


async def test_exception_inside_tool_is_not_reported_as_success_over_http(
    tmp_path, monkeypatch
):
    """
    Guards the second half of #61 independently of parameter validation: if
    a tool implementation raises AFTER passing the schema-validation gate
    (i.e. the call itself is well-formed but fails at runtime), the HTTP
    transport must report status="error" with the exception text, never
    status="success". Before the fix, _wrap_tool caught the exception and
    returned {"error": ...} as the tool's *result*, which the HTTP handler
    then wrapped in status="success".

    The engine method is patched BEFORE constructing LeanMCPInterface:
    the tool registry captures a direct reference to the bound method at
    construction time (`self._wrap_tool(self.session_engine.<method>)`), so
    patching the engine instance afterwards would not reach an
    already-built registry entry.
    """
    backend = SQLiteBackend(str(tmp_path / "test_issue_61_boom.db"))
    await backend.initialize()
    try:
        engine = SessionIntelligenceEngine(
            repository_path=str(tmp_path),
            use_filesystem=False,
            database=backend,
        )

        def _boom(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(engine, "session_analyze_patterns", _boom)

        lean = LeanMCPInterface(engine)
        mcp_mgr = MCPSessionManager(backend)
        notif_mgr = NotificationManager()
        sc = SecurityConfig(
            localhost_only=False, allowed_origins=["*"], require_api_key=False
        )

        server = HTTPSessionIntelligenceServer(
            host="127.0.0.1",
            port=4099,
            repository_path=str(tmp_path),
            db_config=DatabaseConfig(),
            security_config=sc,
        )
        fastapi_app = server.create_app()
        fastapi_app.state.database = backend
        fastapi_app.state.session_engine = engine
        fastapi_app.state.lean_interface = lean
        fastapi_app.state.mcp_session_manager = mcp_mgr
        fastapi_app.state.notification_manager = notif_mgr

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=fastapi_app),
            base_url="http://testserver",
        ) as client:
            # Schema-valid parameters: "scope" is a declared, optional
            # property of session_analyze_patterns, so this call must clear
            # validate_tool_parameters and actually reach the (patched)
            # engine method.
            payload = await _http_execute_tool(
                client, "session_analyze_patterns", {"scope": "recent"}
            )
    finally:
        await backend.close()

    assert payload["status"] == "error"
    assert payload["status"] != "success"
    assert "boom" in payload["error"]
