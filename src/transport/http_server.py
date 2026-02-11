"""
HTTP Server for Session-Intelligence MCP.

Provides both MCP protocol and direct REST API access for LLMs/scripts.

## REST API (Non-MCP, direct curl/HTTP access)

- GET /health                          - Server health and connection status
- GET /api/sessions                    - List sessions (?limit=N for pagination)
- GET /api/sessions/{session_id}       - Get specific session details
- POST /tools/agent_query_learnings    - Query agent learnings with text search
- POST /tools/query_project_learnings  - Query project learnings with text search
- POST /tools/session_find_solution    - Cross-agent solution search
- POST /tools/session_log_learning     - Log learning directly to database

## MCP Protocol (JSON-RPC 2.0)

- POST /mcp                            - MCP JSON-RPC requests (requires MCP-Session-Id)
- GET /mcp                             - SSE stream for server-to-client notifications

## Quick curl examples

    # Health check
    curl http://127.0.0.1:4002/health

    # List recent sessions
    curl http://127.0.0.1:4002/api/sessions?limit=10

    # Query agent learnings
    curl -X POST http://127.0.0.1:4002/tools/agent_query_learnings \\
      -H "Content-Type: application/json" \\
      -d '{"agent_name": "my-agent", "query": "patterns", "limit": 5}'
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any


class DataclassJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles dataclasses and common non-serializable types."""

    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if hasattr(obj, "model_dump"):  # Pydantic v2
            return obj.model_dump()
        if hasattr(obj, "dict"):  # Pydantic v1
            return obj.dict()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence import DatabaseConfig, create_database, sanitize_dsn
from transport.mcp_session_manager import MCPSessionManager
from transport.security import (
    LocalhostOnlyMiddleware,
    SecurityConfig,
    get_origin_validation_middleware,
    validate_api_key,
)
from utils.token_limiter import apply_token_limits

logger = logging.getLogger(__name__)


class NotificationManager:
    """Manages server-to-client notifications via SSE."""

    def __init__(self) -> None:
        self._subscribers: dict[str, asyncio.Queue[dict[str, Any]]] = {}

    async def subscribe(self, mcp_session_id: str) -> AsyncGenerator[dict[str, Any], None]:
        """Subscribe to notifications for an MCP session."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers[mcp_session_id] = queue

        try:
            while True:
                notification = await queue.get()
                yield notification
        finally:
            if mcp_session_id in self._subscribers:
                del self._subscribers[mcp_session_id]

    async def notify(self, mcp_session_id: str, event_type: str, data: dict[str, Any]) -> None:
        """Send notification to a specific MCP session."""
        if mcp_session_id in self._subscribers:
            await self._subscribers[mcp_session_id].put(
                {
                    "type": event_type,
                    "data": data,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    async def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcast notification to all connected sessions."""
        notification = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
        for queue in self._subscribers.values():
            await queue.put(notification)

    def get_subscriber_count(self) -> int:
        """Get count of active subscribers."""
        return len(self._subscribers)


class HTTPSessionIntelligenceServer:
    """HTTP server for Session Intelligence MCP with cross-session state sharing."""

    MCP_PROTOCOL_VERSION = "2024-11-05"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        repository_path: str = ".",
        db_config: DatabaseConfig | None = None,
        security_config: SecurityConfig | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.repository_path = repository_path
        self.security_config = security_config or SecurityConfig()

        # Database configuration - PostgreSQL only
        # Default location: ~/.claude/session-intelligence/ (global, cross-project)
        self.db_config = db_config or DatabaseConfig.load()

        # PostgreSQL DSN for logging (sanitized)
        self.db_path = (
            self.db_config.postgresql_dsn or "postgresql://localhost/session_intelligence"
        )

        self.database: Any | None = None
        self.session_engine: SessionIntelligenceEngine | None = None
        self.lean_interface: LeanMCPInterface | None = None
        self.mcp_session_manager: MCPSessionManager | None = None
        self.notification_manager: NotificationManager | None = None

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan manager."""
        logger.info(f"Starting HTTP server on {self.host}:{self.port}")
        logger.info("Database backend: postgresql")
        logger.info(f"Database: {sanitize_dsn(self.db_path)}")

        # Create and initialize database using factory
        self.database = create_database(config=self.db_config)
        await self.database.initialize()

        self.session_engine = SessionIntelligenceEngine(
            repository_path=self.repository_path,
            use_filesystem=False,  # HTTP transport uses database, not local filesystem
            database=self.database,  # Pass database for persistence
        )

        # Session continuity: Check for active session for this project
        active_session = await self.database.get_active_session_for_project(self.repository_path)
        if active_session:
            logger.info(
                f"Resuming active session: {active_session['id']} for {self.repository_path}"
            )
            # Load session into engine cache for continuity
            from models.session_models import Session

            try:
                session = Session.model_validate(active_session)
                self.session_engine.session_cache[session.id] = session
                self.session_engine._current_session_id = session.id
            except Exception as e:
                logger.warning(f"Could not restore session from database: {e}")
        else:
            logger.info(
                f"No active session found for {self.repository_path}, will create new on demand"
            )

        self.lean_interface = LeanMCPInterface(self.session_engine)
        self.mcp_session_manager = MCPSessionManager(self.database)
        self.notification_manager = NotificationManager()

        app.state.database = self.database
        app.state.session_engine = self.session_engine
        app.state.lean_interface = self.lean_interface
        app.state.mcp_session_manager = self.mcp_session_manager
        app.state.notification_manager = self.notification_manager

        logger.info("Session Intelligence HTTP server ready")

        yield

        logger.info("Shutting down HTTP server")
        await self.database.close()

    def create_app(self) -> FastAPI:
        """Create the FastAPI application with MCP endpoints."""
        app = FastAPI(
            title="Session Intelligence MCP Server",
            description="""HTTP transport for Session Intelligence MCP with REST API access.

## REST API (curl/HTTP access without MCP protocol)

LLMs and scripts can access these endpoints directly:

- `GET /health` - Server health check
- `GET /api/sessions` - List sessions (add `?limit=N`)
- `GET /api/sessions/{id}` - Get session by ID
- `POST /tools/agent_query_learnings` - Query agent learnings with text search
- `POST /tools/query_project_learnings` - Query project learnings with text search
- `POST /tools/session_find_solution` - Cross-agent solution search
- `POST /tools/session_log_learning` - Log learning directly (no MCP session needed)

## MCP Protocol (JSON-RPC 2.0)

- `POST /mcp` - MCP JSON-RPC requests (requires MCP-Session-Id header)
- `GET /mcp` - SSE stream for notifications

## Example curl commands

```bash
# Health check
curl http://127.0.0.1:4002/health

# List sessions
curl http://127.0.0.1:4002/api/sessions?limit=10

# Query learnings
curl -X POST http://127.0.0.1:4002/tools/agent_query_learnings \\
  -H "Content-Type: application/json" \\
  -d '{"agent_name": "my-agent", "query": "error handling", "limit": 5}'
```
""",
            version="1.0.0",
            lifespan=self.lifespan,
        )

        if self.security_config.localhost_only:
            app.add_middleware(LocalhostOnlyMiddleware)

        app.add_middleware(get_origin_validation_middleware(self.security_config.allowed_origins))

        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.security_config.allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["MCP-Session-Id", "MCP-Protocol-Version"],
        )

        self._add_mcp_endpoints(app)
        self._add_health_endpoint(app)
        self._add_session_endpoints(app)
        self._add_tools_endpoints(app)

        return app

    def _add_mcp_endpoints(self, app: FastAPI) -> None:
        """Add MCP protocol endpoints."""

        @app.post("/mcp")
        async def handle_mcp_post(
            request: Request,
            mcp_session_id: str | None = Header(None, alias="MCP-Session-Id"),
            x_api_key: str | None = Header(None, alias="X-API-Key"),
        ) -> JSONResponse:
            """Handle MCP JSON-RPC 2.0 requests."""
            if self.security_config.require_api_key and self.security_config.api_key:
                validate_api_key(x_api_key, self.security_config.api_key)

            try:
                body = await request.json()
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=400,
                    content={
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Parse error"},
                    },
                )

            method = body.get("method")
            params = body.get("params", {})
            req_id = body.get("id")

            mcp_manager = request.app.state.mcp_session_manager
            lean_interface = request.app.state.lean_interface

            if method == "initialize":
                new_session_id = await mcp_manager.create_mcp_session(
                    client_info=params.get("clientInfo")
                )
                response = JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "protocolVersion": self.MCP_PROTOCOL_VERSION,
                            "capabilities": {
                                "resources": {"subscribe": True, "listChanged": True},
                                "tools": {"listChanged": True},
                                "prompts": {"listChanged": True},
                            },
                            "serverInfo": {"name": "session-intelligence", "version": "1.0.0"},
                        },
                    }
                )
                response.headers["MCP-Session-Id"] = new_session_id
                response.headers["MCP-Protocol-Version"] = self.MCP_PROTOCOL_VERSION
                return response

            if not mcp_session_id:
                return JSONResponse(
                    status_code=400,
                    content={
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32600, "message": "Missing MCP-Session-Id"},
                    },
                )

            if not await mcp_manager.validate_session(mcp_session_id):
                return JSONResponse(
                    status_code=401,
                    content={
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32600, "message": "Invalid MCP-Session-Id"},
                    },
                )

            await mcp_manager.update_activity(mcp_session_id)

            try:
                result = await self._handle_mcp_method(
                    method, params, mcp_session_id, lean_interface, request
                )
                return JSONResponse(content={"jsonrpc": "2.0", "id": req_id, "result": result})
            except Exception as e:
                logger.exception(f"Error handling MCP method {method}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32603, "message": str(e)},
                    },
                )

        @app.get("/mcp")
        async def handle_mcp_sse(
            request: Request,
            mcp_session_id: str | None = Header(None, alias="MCP-Session-Id"),
        ) -> StreamingResponse:
            """Server-Sent Events stream for notifications."""
            if not mcp_session_id:
                raise HTTPException(status_code=400, detail="Missing MCP-Session-Id")

            mcp_manager = request.app.state.mcp_session_manager
            notification_manager = request.app.state.notification_manager

            if not await mcp_manager.validate_session(mcp_session_id):
                raise HTTPException(status_code=401, detail="Invalid MCP-Session-Id")

            async def event_generator() -> AsyncGenerator[str, None]:
                try:
                    async for notification in notification_manager.subscribe(mcp_session_id):
                        event_id = str(uuid.uuid4())[:8]
                        data = json.dumps(notification)
                        yield f"id: {event_id}\ndata: {data}\n\n"
                except asyncio.CancelledError:
                    pass

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

    async def _handle_mcp_method(
        self,
        method: str,
        params: dict[str, Any],
        mcp_session_id: str,
        lean_interface: LeanMCPInterface,
        request: Request,
    ) -> dict[str, Any]:
        """Handle individual MCP methods."""
        if method == "resources/templates/list":
            return {
                "resourceTemplates": [
                    {
                        "uriTemplate": "notes://session/{date}",
                        "name": "session-notes",
                        "description": "Session notes by date",
                    },
                    {
                        "uriTemplate": "decisions://category/{category}",
                        "name": "decisions",
                        "description": "Decisions by category",
                    },
                    {
                        "uriTemplate": "metrics://branch/{branch}",
                        "name": "metrics",
                        "description": "Metrics by branch",
                    },
                    {
                        "uriTemplate": "context://session/{session_id}",
                        "name": "context",
                        "description": "Session context",
                    },
                ]
            }

        if method == "resources/list":
            return {"resources": []}

        if method == "resources/read":
            return await self._handle_resource_read(params, request)

        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "discover_tools",
                        "description": (
                            "Discover session lifecycle, decision logging, agent tracking, "
                            "and learning tools (28 total). "
                            "USE WHEN: starting sessions, logging decisions, searching learnings"
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {"pattern": {"type": "string", "default": ""}},
                        },
                    },
                    {
                        "name": "get_tool_spec",
                        "description": (
                            "Get parameter schema for session/agent/learning tools. "
                            "USE WHEN: need exact parameters, debugging validation errors"
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {"tool_name": {"type": "string"}},
                            "required": ["tool_name"],
                        },
                    },
                    {
                        "name": "execute_tool",
                        "description": (
                            "Execute session management, agent tracking, or knowledge operations. "
                            "Returns domain-specific results for sessions, agents, and learnings"
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "tool_name": {"type": "string"},
                                "parameters": {"type": "object"},
                            },
                            "required": ["tool_name", "parameters"],
                        },
                    },
                ]
            }

        if method == "tools/call":
            return await self._handle_tool_call(params, mcp_session_id, lean_interface, request)

        if method == "notifications/initialized":
            return {}

        if method == "prompts/list":
            return {"prompts": []}

        if method == "prompts/get":
            return {"prompt": None}

        raise ValueError(f"Unknown method: {method}")

    async def _handle_resource_read(
        self, params: dict[str, Any], request: Request
    ) -> dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri", "")
        database = request.app.state.database

        if uri.startswith("notes://session/"):
            date = uri.replace("notes://session/", "")
            notes = await database.query_notes_by_date(date)
            return {
                "contents": [
                    {"uri": uri, "mimeType": "application/json", "text": json.dumps(notes)}
                ]
            }

        if uri.startswith("decisions://category/"):
            category = uri.replace("decisions://category/", "")
            decisions = await database.query_decisions_by_category(category)
            return {
                "contents": [
                    {"uri": uri, "mimeType": "application/json", "text": json.dumps(decisions)}
                ]
            }

        if uri.startswith("metrics://branch/"):
            branch = uri.replace("metrics://branch/", "")
            metrics = await database.query_metrics_by_branch(branch)
            return {
                "contents": [
                    {"uri": uri, "mimeType": "application/json", "text": json.dumps(metrics)}
                ]
            }

        if uri.startswith("context://session/"):
            session_id = uri.replace("context://session/", "")
            session = await database.get_session(session_id)
            return {
                "contents": [
                    {"uri": uri, "mimeType": "application/json", "text": json.dumps(session or {})}
                ]
            }

        return {"contents": []}

    async def _persist_sessions_to_database(self, request: Request) -> None:
        """Persist all sessions from engine cache to database.

        Saves session-level data AND related records (decisions, agent_executions).
        """
        database = request.app.state.database
        session_engine = request.app.state.session_engine

        for session_id, session in session_engine.session_cache.items():
            try:
                session_data = session.model_dump()
                await database.save_session(session_data)

                # Also persist decisions
                for decision in session.decisions:
                    try:
                        decision_data = (
                            decision.model_dump() if hasattr(decision, "model_dump") else decision
                        )
                        decision_data["session_id"] = session_id
                        await database.save_decision(decision_data)
                    except Exception as e:
                        logger.warning(f"Failed to persist decision: {e}")

                # Also persist agent executions
                for agent_exec in session.agents_executed:
                    try:
                        exec_data = (
                            agent_exec.model_dump()
                            if hasattr(agent_exec, "model_dump")
                            else agent_exec
                        )
                        exec_data["session_id"] = session_id
                        await database.save_agent_execution(exec_data)
                    except Exception as e:
                        logger.warning(f"Failed to persist agent execution: {e}")

                logger.debug(
                    f"Persisted session {session_id} with {len(session.decisions)} decisions"
                )
            except Exception as e:
                logger.error(f"Failed to persist session {session_id}: {e}")

    async def _ensure_sessions_loaded_from_database(self, request: Request) -> None:
        """Load active sessions from database into engine cache if cache is empty.

        This ensures session-modifying tools (like session_log_decision) can find
        sessions even across HTTP requests where the engine cache was cleared.
        """
        session_engine = request.app.state.session_engine
        database = request.app.state.database

        # If cache already has sessions, no need to load
        if session_engine.session_cache:
            logger.debug(
                f"Session cache has {len(session_engine.session_cache)} sessions, skipping DB load"
            )
            return

        # Get project path from engine
        project_path = str(session_engine.claude_sessions_path.parent)
        logger.debug(f"Loading active sessions for project: {project_path}")

        try:
            # Query for active sessions in this project
            session_data = await database.get_active_session_for_project(project_path)
            if not session_data:
                logger.debug("No active sessions found in database")
                return

            # Reconstruct Session object and add to cache
            from models.session_models import (
                HealthStatus,
                PerformanceMetrics,
                Session,
                SessionMetadata,
                SessionStatus,
            )

            # Build session from database data
            session_id = session_data["id"]
            session = Session(
                id=session_id,
                started=(
                    datetime.fromisoformat(session_data["started"])
                    if isinstance(session_data["started"], str)
                    else session_data["started"]
                ),
                completed=(
                    datetime.fromisoformat(session_data["completed"])
                    if session_data.get("completed") and isinstance(session_data["completed"], str)
                    else session_data.get("completed")
                ),
                mode=session_data.get("mode", "local"),
                project_name=session_data.get("project_name", ""),
                project_path=session_data.get("project_path", project_path),
                status=SessionStatus(session_data.get("status", "active")),
                metadata=SessionMetadata(
                    **session_data.get(
                        "metadata",
                        {"session_type": "development", "environment": "local", "user": "user"},
                    )
                ),
                health_status=(
                    HealthStatus(**session_data.get("health_status", {}))
                    if session_data.get("health_status")
                    else HealthStatus()
                ),
                performance_metrics=(
                    PerformanceMetrics(**session_data.get("performance_metrics", {}))
                    if session_data.get("performance_metrics")
                    else PerformanceMetrics()
                ),
            )

            # Load decisions for this session
            decisions = await database.query_decisions_by_session(session_id)
            if decisions:
                from models.session_models import Decision, DecisionContext, ImpactLevel

                for dec_data in decisions:
                    try:
                        decision = Decision(
                            decision_id=dec_data.get("id", dec_data.get("decision_id", "")),
                            timestamp=(
                                datetime.fromisoformat(dec_data["timestamp"])
                                if isinstance(dec_data["timestamp"], str)
                                else dec_data["timestamp"]
                            ),
                            description=dec_data.get("description", ""),
                            context=DecisionContext(
                                session_id=session_id, project_state=dec_data.get("context", {})
                            ),
                            impact_level=ImpactLevel(dec_data.get("impact_level", "medium")),
                            artifacts=dec_data.get("artifacts", []),
                        )
                        session.decisions.append(decision)
                    except Exception as e:
                        logger.warning(f"Failed to load decision: {e}")

            # Add to cache
            session_engine.session_cache[session_id] = session
            session_engine._current_session_id = session_id
            logger.info(
                f"Loaded session {session_id} from database with {len(session.decisions)} decisions"
            )

        except Exception as e:
            logger.error(f"Failed to load sessions from database: {e}")

    async def _handle_tool_call(
        self,
        params: dict[str, Any],
        mcp_session_id: str,
        lean_interface: LeanMCPInterface,
        request: Request,
    ) -> dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        tool_registry = lean_interface.tool_registry

        if tool_name == "discover_tools":
            pattern = arguments.get("pattern", "")
            tools = [
                {"name": n, "description": i["description"]}
                for n, i in tool_registry.items()
                if not pattern or pattern.lower() in n.lower()
            ]
            result = {
                "available_tools": tools,
                "total_tools": len(tool_registry),
                "filtered_count": len(tools),
            }

        elif tool_name == "get_tool_spec":
            target = arguments.get("tool_name")
            if target not in tool_registry:
                result = {
                    "error": f"Tool '{target}' not found",
                    "available_tools": list(tool_registry.keys()),
                }
            else:
                info = tool_registry[target]
                result = {
                    "name": target,
                    "description": info["description"],
                    "schema": info["schema"],
                    "examples": info.get("examples", []),
                }

        elif tool_name == "execute_tool":
            target = arguments.get("tool_name")
            tool_params = arguments.get("parameters", {})
            if target not in tool_registry:
                result = {
                    "error": f"Tool '{target}' not found",
                    "available_tools": list(tool_registry.keys()),
                }
            else:
                # Tools that read/write session state - need DB sync
                session_modifying_tools = {
                    "session_manage_lifecycle",
                    "session_track_execution",
                    "session_log_decision",
                    "session_coordinate_agents",
                    "session_log_learning",
                    "session_find_solution",
                    "session_update_solution_outcome",
                    "session_track_file_operation",
                }

                try:
                    import inspect

                    # Pre-load sessions from database before session-modifying tools
                    # This ensures decisions/executions can find their session even after
                    # the engine cache is cleared between HTTP requests
                    if target in session_modifying_tools:
                        await self._ensure_sessions_loaded_from_database(request)

                    tool_func = tool_registry[target]["implementation"]
                    # Handle async tool implementations
                    if inspect.iscoroutinefunction(tool_func):
                        tool_result = await tool_func(**tool_params)
                    else:
                        tool_result = tool_func(**tool_params)

                    # Handle knowledge system tools - persist to database
                    database = request.app.state.database
                    if (
                        target == "session_log_learning"
                        and hasattr(tool_result, "learning")
                        and tool_result.learning
                    ):
                        learning = tool_result.learning
                        # Validate source_session_id exists before FK insert, else use None
                        valid_session_id = None
                        if learning.source_session_id:
                            existing = await database.get_session(learning.source_session_id)
                            if existing:
                                valid_session_id = learning.source_session_id
                        await database.save_project_learning(
                            learning_id=learning.id,
                            project_path=learning.project_path,
                            category=(
                                learning.category.value
                                if hasattr(learning.category, "value")
                                else learning.category
                            ),
                            learning_content=learning.learning_content,
                            trigger_context=learning.trigger_context,
                            source_session_id=valid_session_id,
                        )
                        tool_result = tool_result.model_copy(
                            update={"status": "saved", "message": "Learning saved to database"}
                        )

                    elif target == "session_find_solution":
                        # Query database for solutions
                        solutions = await database.find_error_solutions(
                            error_text=tool_params.get("error_text", ""),
                            project_path=tool_params.get("project_path"),
                            include_universal=tool_params.get("include_universal", True),
                        )
                        from models.session_models import ErrorSolution, SolutionSearchResult

                        # Convert datetime fields to strings for Pydantic
                        for s in solutions:
                            if s.get("created_at") and hasattr(s["created_at"], "isoformat"):
                                s["created_at"] = s["created_at"].isoformat()
                            if s.get("last_used") and hasattr(s["last_used"], "isoformat"):
                                s["last_used"] = s["last_used"].isoformat()
                        tool_result = SolutionSearchResult(
                            error_text=tool_params.get("error_text", ""),
                            total_found=len(solutions),
                            solutions=[ErrorSolution(**s) for s in solutions] if solutions else [],
                            project_specific_count=sum(
                                1 for s in solutions if s.get("project_path")
                            ),
                            universal_count=sum(1 for s in solutions if not s.get("project_path")),
                        )

                    elif target == "session_update_solution_outcome":
                        db_result = await database.update_solution_outcome(
                            solution_id=tool_params.get("solution_id", ""),
                            success=tool_params.get("success", False),
                        )
                        from models.session_models import SolutionResult

                        tool_result = SolutionResult(
                            id=tool_params.get("solution_id", ""),
                            status="updated" if db_result.get("updated") else "error",
                            message=(
                                f"Success rate: {db_result.get('success_rate', 0):.2f}"
                                if db_result.get("updated")
                                else db_result.get("error", "")
                            ),
                        )

                    elif target == "session_track_file_operation":
                        # Persist file operation to database
                        if hasattr(tool_result, "operation") and tool_result.operation:
                            op = tool_result.operation
                            await database.save_file_operation(
                                {
                                    "session_id": op.session_id,
                                    "timestamp": (
                                        op.timestamp.isoformat()
                                        if hasattr(op.timestamp, "isoformat")
                                        else str(op.timestamp)
                                    ),
                                    "operation": (
                                        op.operation_type.value
                                        if hasattr(op.operation_type, "value")
                                        else op.operation_type
                                    ),
                                    "file_path": op.file_path,
                                    "lines_added": op.lines_added or 0,
                                    "lines_removed": op.lines_removed or 0,
                                    "summary": op.description or "",
                                    "tool_name": op.tool_name or "",
                                }
                            )
                            tool_result = tool_result.model_copy(update={"status": "saved"})

                    limited = apply_token_limits(tool_result, target)
                    result = {"tool": target, "status": "success", "result": limited}

                    # Persist session changes to database after session-modifying operations
                    if target in session_modifying_tools:
                        await self._persist_sessions_to_database(request)

                except Exception as e:
                    logger.exception(f"Error executing tool {target}")
                    result = {"tool": target, "status": "error", "error": str(e)}
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        return {"content": [{"type": "text", "text": json.dumps(result, cls=DataclassJSONEncoder)}]}

    def _add_health_endpoint(self, app: FastAPI) -> None:
        """Add health check endpoint."""

        @app.get("/health")
        async def health_check(request: Request) -> dict[str, Any]:
            db = request.app.state.database
            mcp_manager = request.app.state.mcp_session_manager
            notif_manager = request.app.state.notification_manager
            return {
                "status": "healthy",
                "database": "connected" if db and db.is_connected else "disconnected",
                "mcp_protocol_version": self.MCP_PROTOCOL_VERSION,
                "active_mcp_sessions": mcp_manager.get_active_session_count() if mcp_manager else 0,
                "sse_subscribers": notif_manager.get_subscriber_count() if notif_manager else 0,
                "timestamp": datetime.now().isoformat(),
            }

    def _add_session_endpoints(self, app: FastAPI) -> None:
        """Add REST endpoints for session queries."""

        @app.get("/api/sessions")
        async def list_sessions(
            request: Request,
            limit: int = 50,
            x_api_key: str | None = Header(None, alias="X-API-Key"),
        ) -> dict[str, Any]:
            if self.security_config.require_api_key and self.security_config.api_key:
                validate_api_key(x_api_key, self.security_config.api_key)
            sessions = await request.app.state.database.query_sessions(limit=limit)
            return {"sessions": sessions, "count": len(sessions)}

        @app.get("/api/sessions/{session_id}")
        async def get_session(
            request: Request,
            session_id: str,
            x_api_key: str | None = Header(None, alias="X-API-Key"),
        ) -> dict[str, Any]:
            if self.security_config.require_api_key and self.security_config.api_key:
                validate_api_key(x_api_key, self.security_config.api_key)
            session = await request.app.state.database.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            return session

    def _add_tools_endpoints(self, app: FastAPI) -> None:
        """Add REST API endpoints for tool-like queries."""

        @app.post("/tools/agent_query_learnings")
        async def agent_query_learnings(request: Request) -> dict[str, Any]:
            """Query learnings for a specific agent with text search and filtering."""
            body = await request.json()
            agent_name = body.get("agent_name")
            query = body.get("query")  # text search
            category = body.get("category")
            limit = body.get("limit", 5)
            min_success_rate = body.get("min_success_rate", 0.0)

            session_engine = request.app.state.session_engine

            # Get raw learnings
            learnings = await session_engine.agent_query_learnings(
                agent_name=agent_name,
                learning_type=category,
                limit=100,  # get more, filter below
            )

            # Apply text search filter
            if query:
                query_lower = query.lower()
                learnings = [
                    ln
                    for ln in learnings
                    if query_lower in ln.content.lower()
                    or query_lower in (ln.source_context or "").lower()
                ]

            # Apply min_success_rate filter
            learnings = [ln for ln in learnings if ln.success_rate >= min_success_rate]

            # Apply limit
            learnings = learnings[:limit]

            # Format response
            return {
                "status": "success",
                "learnings": [
                    {
                        "id": ln.id,
                        "category": ln.learning_type,
                        "learning_content": ln.content,
                        "trigger_context": ln.source_context,
                        "success_count": int(ln.times_applied * ln.success_rate),
                        "failure_count": int(ln.times_applied * (1 - ln.success_rate)),
                        "success_rate": ln.success_rate,
                        "last_used": ln.updated_at,
                    }
                    for ln in learnings
                ],
                "total_matches": len(learnings),
            }

        @app.post("/tools/session_find_solution")
        async def session_find_solution(request: Request) -> dict[str, Any]:
            """Cross-agent solution search, scoped to project or universal."""
            body = await request.json()
            error_context = body.get("error_context", "")
            project_path = body.get("project_path")
            include_universal = body.get("include_universal", True)
            limit = body.get("limit", 3)

            database = request.app.state.database

            solutions = []

            # Query project learnings
            if project_path:
                project_learnings = await database.query_project_learnings(
                    project_path=project_path,
                    limit=50,
                )
                # Text search filter
                query_lower = error_context.lower()
                for ln in project_learnings:
                    content = (ln.get("learning_content") or "").lower()
                    trigger = (ln.get("trigger_context") or "").lower()
                    if query_lower in content or query_lower in trigger:
                        solutions.append(
                            {
                                "id": ln["id"],
                                "source": "project",
                                "learning_content": ln["learning_content"],
                                "trigger_context": ln.get("trigger_context"),
                                "success_count": ln.get("success_count", 1),
                            }
                        )

            # Query universal learnings (from all projects with high success rate)
            # Note: Currently returns empty as there's no universal learnings table.
            # Future: Add promoted_to_universal column to project_learnings table.
            if include_universal:
                # Placeholder: universal learnings not yet implemented
                pass

            # Sort by success_count descending, limit
            solutions.sort(key=lambda x: x["success_count"], reverse=True)
            solutions = solutions[:limit]

            return {"status": "success", "solutions": solutions}

        @app.post("/tools/session_log_learning")
        async def session_log_learning(request: Request) -> dict[str, Any]:
            """Log a learning directly to database without MCP session requirement.

            Accepts:
                category: Learning category (pattern, anti-pattern, technique, etc.)
                learning_content: The learning content to store
                trigger_context: Optional context that triggered this learning
                project_path: Project path to scope the learning to

            Returns:
                status: success/error
                learning_id: Generated UUID for the learning
                message: Confirmation message
            """
            body = await request.json()
            category = body.get("category")
            learning_content = body.get("learning_content")
            trigger_context = body.get("trigger_context")
            project_path = body.get("project_path")

            # Validate required fields
            if not category:
                return {"status": "error", "message": "category is required"}
            if not learning_content:
                return {"status": "error", "message": "learning_content is required"}
            if not project_path:
                return {"status": "error", "message": "project_path is required"}

            database = request.app.state.database

            # Generate learning ID
            learning_id = f"learning-{uuid.uuid4().hex[:8]}"

            try:
                await database.save_project_learning(
                    learning_id=learning_id,
                    project_path=project_path,
                    category=category,
                    learning_content=learning_content,
                    trigger_context=trigger_context,
                    source_session_id=None,  # No MCP session required
                )

                return {
                    "status": "success",
                    "learning_id": learning_id,
                    "message": f"Learning saved to project {project_path}",
                }
            except Exception as e:
                logger.exception(f"Failed to save learning: {e}")
                return {"status": "error", "message": str(e)}

        @app.post("/tools/query_project_learnings")
        async def query_project_learnings(request: Request) -> dict[str, Any]:
            """Query project learnings from database.

            Accepts:
                project_path: Project path to query learnings for (required)
                category: Optional category filter
                query: Optional text search in learning_content
                limit: Max results (default 20)

            Returns:
                status: success/error
                learnings: List of matching learnings
            """
            body = await request.json()
            project_path = body.get("project_path")
            category = body.get("category")
            query = body.get("query")
            limit = body.get("limit", 20)

            if not project_path:
                return {"status": "error", "message": "project_path is required"}

            database = request.app.state.database

            try:
                learnings = await database.query_project_learnings(
                    project_path=project_path,
                    category=category,
                    limit=limit,
                )

                # Apply text search filter if provided
                if query:
                    query_lower = query.lower()
                    learnings = [
                        ln
                        for ln in learnings
                        if query_lower in (ln.get("learning_content") or "").lower()
                        or query_lower in (ln.get("trigger_context") or "").lower()
                    ]

                return {
                    "status": "success",
                    "learnings": learnings,
                    "count": len(learnings),
                }
            except Exception as e:
                logger.exception(f"Failed to query learnings: {e}")
                return {"status": "error", "message": str(e)}

    async def run(self) -> None:
        """Run the HTTP server."""
        app = self.create_app()
        config = uvicorn.Config(
            app=app, host=self.host, port=self.port, log_level="info", access_log=True
        )
        server = uvicorn.Server(config)
        await server.serve()
