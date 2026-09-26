"""
Regression tests for issue #160: transport RESULT parity.

`src/transport/http_server.py`'s `_handle_tool_call` dispatches through the
same shared `tool_registry` as the stdio/lean path
(`LeanMCPInterface.execute_tool`), but for four tools it then OVERWRITES the
engine's `tool_result` in a post-call block (~:824-920):

  - `session_log_learning` (~:826-852)
  - `session_find_solution` (~:854-877)
  - `session_update_solution_outcome` (~:879-894)
  - `session_track_file_operation` (~:896-920)

`session_find_solution` and `session_update_solution_outcome` REBUILD their
result from scratch by re-querying the database directly with the RAW,
un-guarded `tool_params` -- discarding every safety guard the engine method
itself applies (e.g. issue #156's `_unknown_`/relative-`project_path` guard
in `session_find_solution`, and the engine's own accounting in
`session_update_solution_outcome`). Because these two `elif` branches are
NOT gated on the shape of `tool_result` (unlike the other two, see below),
they run unconditionally for every HTTP call to these tools, producing
results that diverge from the stdio/lean transport for the exact same
input. THIS is the live, reproducible half of #160, and is what tests 1-3
below cover.

IMPORTANT finding from investigation (do not re-derive): the other two
special-cased tools, `session_log_learning` and `session_track_file_operation`,
guard their override with `hasattr(tool_result, "learning")` /
`hasattr(tool_result, "operation")`. `tool_result` at that point is the
return value of `tool_registry[target]["implementation"]`
(http_server.py:817-822) -- and that implementation is
`LeanMCPInterface._wrap_async_tool(engine_method)` (the SAME wrapped
callable the stdio dispatcher uses, since `tool_registry` is literally
`lean_interface.tool_registry`, http_server.py:735). `_wrap_async_tool`
unconditionally pipes its result through `apply_token_limits()`, whose
`limit_response()` calls `_to_dict()` FIRST, before any size check -- so by
the time `_handle_tool_call`'s post-call block inspects `tool_result`, any
Pydantic model has ALREADY been flattened into a plain dict. A plain dict
has no `.learning`/`.operation` *attribute* (dict keys are not attributes),
so `hasattr(...)` is always False and these two branches are dead code
today: empirically verified (see scratch probe run against this branch) --
one HTTP `session_track_file_operation` call yields exactly one
`file_operations` row, and one HTTP `session_log_learning` call returns the
engine's own message untouched ("Learning saved to database for pattern."),
not the override's fixed "Learning saved to database" string. There is
therefore no CURRENTLY reproducible divergence to assert against for those
two tools without contriving a scenario that never happens today, so this
file intentionally covers only the three genuinely-diverging cases
(`session_find_solution` x2, `session_update_solution_outcome`).

These tests reuse the ASGI harness pattern from
tests/regression/test_issue_127_http_envelope_parity.py.

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
# Fixtures (mirrors tests/regression/test_issue_127_http_envelope_parity.py)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    """Initialised SQLite backend."""
    backend = SQLiteBackend(str(tmp_path / "test_issue_160.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def app(tmp_path, db):
    """
    FastAPI app with pre-populated state using SQLite.

    Same construction as test_issue_127_http_envelope_parity.py: create the
    HTTPSessionIntelligenceServer, call create_app() for routes/middleware,
    then inject app.state directly (httpx.ASGITransport never fires
    lifespan events, so this injected state is the only state used).
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
    over the real HTTP/MCP transport, and return the decoded envelope."""
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
# TEST 1: session_find_solution, project_path="_unknown_" sentinel
# ===========================================================================


async def test_find_solution_result_parity_between_stdio_and_http(app, asgi_client, db):
    """
    Core #160 repro. Seed an error_solutions row scoped LITERALLY to the
    "_unknown_" sentinel, then search for it with `project_path="_unknown_"`
    over both transports.

    Engine's `session_find_solution` (issue #156 guard) refuses to honour an
    unusable `project_path` and returns an empty result WITHOUT querying the
    database at all. The HTTP post-call block ignores that guard entirely
    and re-queries `database.find_error_solutions` with the raw
    `project_path="_unknown_"`, which literal-matches the seeded row's
    `project_path` column -- so HTTP finds it while stdio does not.

    NOTE: trigger_context is irrelevant here (no learnings are seeded), so
    issue #158's NULL-trigger_context trap does not apply to this test, but
    is avoided by construction anyway (no ProjectLearning rows written).
    """
    error_text = "ModuleNotFoundError: No module named 'issue160probe'"
    await db.save_error_solution(
        solution_id="sol-unknown-160",
        error_pattern=error_text,
        solution_steps=["pip install issue160probe"],
        project_path="_unknown_",
    )

    parameters = {"error_text": error_text, "project_path": "_unknown_"}

    lean_interface = app.state.lean_interface
    stdio_result = await _stdio_execute_tool(lean_interface, "session_find_solution", parameters)
    http_result = await _http_execute_tool(asgi_client, "session_find_solution", parameters)

    assert stdio_result["status"] == "success"
    assert http_result["status"] == "success"

    stdio_total = stdio_result["result"]["total_found"]
    http_total = http_result["result"]["total_found"]

    # This is the actual divergence: the engine's #156 guard yields 0,
    # while the HTTP rebuild bypasses it and finds the seeded row.
    assert stdio_total == 0, f"engine guard should suppress results, got {stdio_result}"
    assert http_total == stdio_total, (
        "transport parity violated: stdio (engine, #156-guarded) found "
        f"{stdio_total} solutions but HTTP (raw rebuild, unguarded) found "
        f"{http_total} for the same _unknown_ project_path input -- "
        f"stdio={stdio_result} http={http_result}"
    )


# ===========================================================================
# TEST 2: session_find_solution, relative project_path
# ===========================================================================


async def test_find_solution_relative_path_parity(app, asgi_client, db):
    """
    Same shape as test 1, but with a RELATIVE project_path instead of the
    "_unknown_" sentinel. A relative path resolves against the SERVER's cwd
    if honoured, so the engine's #156 guard (which reuses the same
    "unusable project_path" check as the sentinel case) refuses it and
    returns nothing. The HTTP rebuild again passes the raw relative string
    straight to `find_error_solutions`, literal-matching a row seeded with
    that exact string.
    """
    error_text = "ImportError: cannot import name 'issue160probe2'"
    relative_path = "relative/not/absolute"
    await db.save_error_solution(
        solution_id="sol-relative-160",
        error_pattern=error_text,
        solution_steps=["fix the import"],
        project_path=relative_path,
    )

    parameters = {"error_text": error_text, "project_path": relative_path}

    lean_interface = app.state.lean_interface
    stdio_result = await _stdio_execute_tool(lean_interface, "session_find_solution", parameters)
    http_result = await _http_execute_tool(asgi_client, "session_find_solution", parameters)

    stdio_total = stdio_result["result"]["total_found"]
    http_total = http_result["result"]["total_found"]

    assert stdio_total == 0, f"engine guard should suppress results, got {stdio_result}"
    assert http_total == stdio_total, (
        "transport parity violated: stdio (engine, #156-guarded) found "
        f"{stdio_total} solutions but HTTP (raw rebuild, unguarded) found "
        f"{http_total} for the same relative project_path input -- "
        f"stdio={stdio_result} http={http_result}"
    )


# ===========================================================================
# TEST 3: session_update_solution_outcome result parity
# ===========================================================================


async def test_update_solution_outcome_result_parity(app, asgi_client, db):
    """
    The HTTP post-call block for `session_update_solution_outcome` is NOT
    gated on the shape of `tool_result` (unlike the learning/file-operation
    branches) -- it unconditionally calls
    `database.update_solution_outcome(...)` a SECOND time (the engine
    already called it once inside the dispatched tool_func) and rebuilds a
    fresh `SolutionResult` with a differently-formatted message, discarding
    the engine's own result entirely.

    Two independent, freshly-seeded solution rows are used (one per
    transport) so each starts from an identical usage_count=1/
    success_rate=1.0 baseline and the comparison isn't contaminated by
    cross-call state.
    """
    await db.save_error_solution(
        solution_id="sol-outcome-stdio-160",
        error_pattern="stdio outcome probe",
        solution_steps=["step"],
    )
    await db.save_error_solution(
        solution_id="sol-outcome-http-160",
        error_pattern="http outcome probe",
        solution_steps=["step"],
    )

    lean_interface = app.state.lean_interface
    stdio_result = await _stdio_execute_tool(
        lean_interface,
        "session_update_solution_outcome",
        {"solution_id": "sol-outcome-stdio-160", "success": True},
    )
    http_result = await _http_execute_tool(
        asgi_client,
        "session_update_solution_outcome",
        {"solution_id": "sol-outcome-http-160", "success": True},
    )

    assert stdio_result["status"] == "success"
    assert http_result["status"] == "success"

    stdio_message = stdio_result["result"]["message"]
    http_message = http_result["result"]["message"]

    assert http_message == stdio_message, (
        "transport parity violated: the HTTP post-call block rebuilds "
        "SolutionResult with a different message format than the engine's "
        f"own result -- stdio message={stdio_message!r} "
        f"http message={http_message!r}"
    )

    # Corroborating evidence: the HTTP block's redundant second DB call
    # double-increments usage_count (engine call -> 1->2, HTTP block's own
    # extra call -> 2->3), while the stdio-only path increments it once
    # (1->2). Confirms this is a real double-write, not just a string
    # formatting difference.
    stdio_row = await db.find_error_solutions(
        error_text="stdio outcome probe", project_path=None, include_universal=False
    )
    http_row = await db.find_error_solutions(
        error_text="http outcome probe", project_path=None, include_universal=False
    )
    # find_error_solutions scopes to project_path=None -> universal only;
    # both seeded rows have project_path=None (default), so this matches.
    stdio_usage = next(r["usage_count"] for r in stdio_row if r["id"] == "sol-outcome-stdio-160")
    http_usage = next(r["usage_count"] for r in http_row if r["id"] == "sol-outcome-http-160")

    assert http_usage == stdio_usage, (
        "HTTP's redundant second update_solution_outcome() call "
        f"double-incremented usage_count: stdio usage_count={stdio_usage} "
        f"(one DB write) vs http usage_count={http_usage} (two DB writes) "
        "for an identical single HTTP tool call"
    )
