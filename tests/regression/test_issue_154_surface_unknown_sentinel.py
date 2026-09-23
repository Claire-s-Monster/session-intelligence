"""
Regression tests for issue #154: when a row is written with
project_path == UNKNOWN_PROJECT_PATH ("_unknown_"), the response envelope
must tell the caller, because such a row is not project-recallable (issue
#120 matches project_path exactly against a filter).

The agreed design is a boolean field ``project_path_derived: bool`` on the
write-result models (SessionResult, LearningResult, NotebookResult):
False when the stored project_path is the sentinel, True when a real
(caller-supplied or #151-derived) path was recorded instead.

session_log_decision is explicitly OUT OF SCOPE: it stores no project_path
of its own (it is scoped by session_id FK only), so DecisionResult is not
touched here.

Groups 1-3 use an in-memory SQLite backend (fixture style copied from
tests/regression/test_issue_151_derive_project_path.py). Group 4 exercises
both the stdio (LeanMCPInterface.execute_tool) and HTTP
(HTTPSessionIntelligenceServer._handle_tool_call) dispatchers over a real
ASGI harness (fixture style copied from
tests/regression/test_issue_127_http_envelope_parity.py), because the two
transports are separate implementations of the same meta-tool and a fix
scoped to only one of them would leave the other silently missing the
field.

asyncio_mode = "auto" - no @pytest.mark.asyncio decorators needed.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from core.session_engine import UNKNOWN_PROJECT_PATH, SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence import DatabaseConfig
from persistence.sqlite import SQLiteBackend
from transport.http_server import HTTPSessionIntelligenceServer, NotificationManager
from transport.mcp_session_manager import MCPSessionManager
from transport.security import SecurityConfig

# ---------------------------------------------------------------------------
# Fixtures: groups 1-3 (in-memory SQLite, direct engine calls)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db():
    """In-memory SQLite database, initialized and cleaned up per test."""
    backend = SQLiteBackend(db_path=":memory:")
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
def engine(db, monkeypatch: pytest.MonkeyPatch) -> SessionIntelligenceEngine:
    """Engine wired to in-memory SQLite, no filesystem."""
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENTS_DIR", "/tmp/nonexistent-agents")
    return SessionIntelligenceEngine(
        repository_path=None,
        use_filesystem=False,
        database=db,
    )


async def _seed_prior_session(
    db: SQLiteBackend, project_name: str, project_path: str, *, status: str = "completed"
) -> None:
    """Directly insert a session row as evidence of where a project lives.

    Copied from test_issue_151_derive_project_path.py: bypasses the engine
    so the seeded row's project_path/status are exactly what the test
    controls, with an old started_at so it reads as prior history.
    """
    await db.save_session(
        {
            "id": f"seed-{uuid.uuid4().hex[:12]}",
            "started_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            "project_path": project_path,
            "project_name": project_name,
            "status": status,
        }
    )


# ---------------------------------------------------------------------------
# Group 1: session_manage_lifecycle(operation="create") -> SessionResult
# ---------------------------------------------------------------------------


class TestSessionResultSurfacesSentinel:
    async def test_create_without_derivable_path_reports_not_derived(self, engine):
        """Breaking this hides the sentinel from the caller: a brand-new
        project_name with no prior history and no caller-supplied
        project_path stores UNKNOWN_PROJECT_PATH, and the caller has no way
        to know the row is unrecallable by project filter (#120) unless
        SessionResult.project_path_derived says so explicitly."""
        project = f"proj-154-fresh-{uuid.uuid4().hex[:8]}"

        result = await engine.session_manage_lifecycle(
            operation="create",
            project_name=project,
        )

        assert result.session_data is not None
        assert result.session_data.project_path == UNKNOWN_PROJECT_PATH
        assert result.project_path_derived is False

    async def test_create_with_absolute_path_reports_derived(self, engine):
        """Guards against an overzealous fix: a caller-supplied absolute
        project_path must NOT be reported as an undependable sentinel."""
        project = f"proj-154-abs-{uuid.uuid4().hex[:8]}"

        result = await engine.session_manage_lifecycle(
            operation="create",
            project_name=project,
            project_path="/abs/real/proj-154",
        )

        assert result.session_data is not None
        assert result.session_data.project_path == "/abs/real/proj-154"
        assert result.project_path_derived is True


# ---------------------------------------------------------------------------
# Group 2: session_log_learning -> LearningResult
# ---------------------------------------------------------------------------


class TestLearningResultSurfacesSentinel:
    async def test_learning_stored_with_sentinel_reports_not_derived(self, engine):
        """Documents the exact scenario PR #152/issue #151 explicitly left
        open (see test_no_prior_session_still_uses_sentinel in
        test_issue_151_derive_project_path.py): with no prior evidence at
        all for this project_name, the sentinel is honestly stored -- and
        the caller must be told the row will not surface via project-scoped
        recall (#120), or this silently reproduces the #120 trap."""
        project = f"proj-154-learn-none-{uuid.uuid4().hex[:8]}"

        result = await engine.session_log_learning(
            category="pattern",
            learning_content="some learning content",
            project_name=project,
        )

        assert result.learning is not None
        assert result.learning.project_path == UNKNOWN_PROJECT_PATH
        assert result.project_path_derived is False

    async def test_learning_with_derived_path_reports_derived(self, engine, db):
        """Guards against an overzealous fix: when #151's derivation logic
        successfully recovers a real path from prior session history, that
        must be reported as derived=True, not lumped in with the sentinel
        case."""
        project = f"proj-154-learn-derived-{uuid.uuid4().hex[:8]}"
        await _seed_prior_session(db, project, "/home/user/repos/proj-154-derived")

        result = await engine.session_log_learning(
            category="pattern",
            learning_content="some learning content",
            project_name=project,
        )

        assert result.learning is not None
        assert result.learning.project_path == "/home/user/repos/proj-154-derived"
        assert result.project_path_derived is True


# ---------------------------------------------------------------------------
# Group 3: session_create_notebook_async -> NotebookResult
# ---------------------------------------------------------------------------


class TestNotebookResultSurfacesSentinel:
    async def test_notebook_on_sentinel_session_reports_not_derived(self, engine):
        """A notebook inherits its session's project_path verbatim
        (session_create_notebook_async, session_engine.py:2714-2719). If the
        owning session stored the sentinel, the notebook result must say so
        too -- otherwise a notebook silently inherits an unrecallable
        project_path with no signal to the caller."""
        project = f"proj-154-nb-sentinel-{uuid.uuid4().hex[:8]}"
        create_result = await engine.session_manage_lifecycle(
            operation="create",
            project_name=project,
        )
        assert create_result.session_data.project_path == UNKNOWN_PROJECT_PATH

        notebook_result = await engine.session_create_notebook_async(
            session_id=create_result.session_id,
            save_to_file=False,
            save_to_database=False,
        )

        assert notebook_result.notebook is not None
        assert notebook_result.notebook.project_path == UNKNOWN_PROJECT_PATH
        assert notebook_result.project_path_derived is False

    async def test_notebook_on_absolute_path_session_reports_derived(self, engine):
        """Guards against an overzealous fix: a notebook built on a session
        with a real, absolute project_path must report derived=True."""
        project = f"proj-154-nb-abs-{uuid.uuid4().hex[:8]}"
        create_result = await engine.session_manage_lifecycle(
            operation="create",
            project_name=project,
            project_path="/abs/real/proj-154-notebook",
        )
        assert create_result.session_data.project_path == "/abs/real/proj-154-notebook"

        notebook_result = await engine.session_create_notebook_async(
            session_id=create_result.session_id,
            save_to_file=False,
            save_to_database=False,
        )

        assert notebook_result.notebook is not None
        assert notebook_result.notebook.project_path == "/abs/real/proj-154-notebook"
        assert notebook_result.project_path_derived is True


# ---------------------------------------------------------------------------
# Group 4: transport parity (stdio LeanMCPInterface.execute_tool vs HTTP
# HTTPSessionIntelligenceServer._handle_tool_call)
# ---------------------------------------------------------------------------
# Fixture/helper style copied from
# tests/regression/test_issue_127_http_envelope_parity.py: the two
# dispatchers are independent implementations of the same meta-tool, so a
# field added to one result-construction path can silently vanish on the
# other side unless a parity test pins both.


@pytest.fixture
async def transport_db(tmp_path):
    """Initialised SQLite backend (file-based, matches issue #127 harness)."""
    backend = SQLiteBackend(str(tmp_path / "test_issue_154.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def transport_app(tmp_path, transport_db):
    """FastAPI app with pre-populated state, mirroring test_issue_127's
    `app` fixture: httpx.ASGITransport never fires lifespan events, so the
    injected app.state below is the only state used."""
    engine = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=transport_db,
    )
    lean = LeanMCPInterface(engine)
    mcp_mgr = MCPSessionManager(transport_db)
    notif_mgr = NotificationManager()

    sc = SecurityConfig(
        localhost_only=False,
        allowed_origins=["*"],
        require_api_key=False,
    )

    server = HTTPSessionIntelligenceServer(
        host="127.0.0.1",
        port=4098,
        repository_path=str(tmp_path),
        db_config=DatabaseConfig(),
        security_config=sc,
    )
    fastapi_app = server.create_app()

    fastapi_app.state.database = transport_db
    fastapi_app.state.session_engine = engine
    fastapi_app.state.lean_interface = lean
    fastapi_app.state.mcp_session_manager = mcp_mgr
    fastapi_app.state.notification_manager = notif_mgr

    yield fastapi_app


@pytest.fixture
async def transport_client(transport_app):
    """AsyncClient wired to the pre-seeded ASGI app."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=transport_app),
        base_url="http://testserver",
    ) as client:
        yield client


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
    """Run execute_tool(tool_name, parameters) over the real HTTP/MCP
    transport (HTTPSessionIntelligenceServer._handle_tool_call), and return
    the decoded envelope."""
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
    """Run execute_tool(tool_name, parameters) over the stdio meta-tool
    path (LeanMCPInterface.execute_tool), for parity comparison."""
    execute = _get_meta_tool(interface, "execute_tool")
    return await execute(tool_name, parameters)


async def test_project_path_derived_present_over_both_transports(transport_app, transport_client):
    """Parity guard for #154: a sentinel-producing session_log_learning call
    (fresh project_name, no prior session, no project_path) must report
    project_path_derived=False identically through BOTH the stdio
    LeanMCPInterface.execute_tool meta-tool and the HTTP transport's
    _handle_tool_call. A fix applied only to the shared engine/model layer
    should pass this automatically (both dispatchers route through the same
    tool_registry + model_dump path); a fix that instead patches one
    dispatcher's response-shaping code directly would break this parity
    check even while a stdio-only or http-only test kept passing."""
    lean_interface = transport_app.state.lean_interface
    project = f"proj-154-parity-{uuid.uuid4().hex[:8]}"
    parameters = {
        "category": "pattern",
        "learning_content": "issue-154 transport parity probe",
        "project_name": project,
    }

    stdio_payload = await _stdio_execute_tool(lean_interface, "session_log_learning", parameters)
    http_payload = await _http_execute_tool(transport_client, "session_log_learning", parameters)

    assert "project_path_derived" in stdio_payload["result"], stdio_payload
    assert "project_path_derived" in http_payload["result"], http_payload
    assert stdio_payload["result"]["project_path_derived"] is False
    assert http_payload["result"]["project_path_derived"] is False
    assert (
        stdio_payload["result"]["project_path_derived"]
        == http_payload["result"]["project_path_derived"]
    )
