"""
Database abstraction layer for session persistence.

Provides a Protocol interface that enables multiple backends:
- SQLite (development, single-user)
- PostgreSQL (production, multi-agent analysis)

Usage:
    from persistence.base import DatabaseBackend
    from persistence.sqlite import SQLiteBackend
    from persistence.postgresql import PostgreSQLBackend

    # Development
    db: DatabaseBackend = SQLiteBackend(path="~/.claude/session-intelligence/sessions.db")

    # Production
    db: DatabaseBackend = PostgreSQLBackend(dsn="postgresql://user:pass@localhost/sessions")
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


# Default global location for session intelligence data
DEFAULT_DATA_DIR = Path.home() / ".claude" / "session-intelligence"
DEFAULT_SQLITE_PATH = DEFAULT_DATA_DIR / "sessions.db"
DEFAULT_POSTGRES_DSN = "postgresql://localhost/session_intelligence"


def get_default_data_dir() -> Path:
    """Get the default data directory, creating it if needed."""
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DATA_DIR


@runtime_checkable
class DatabaseBackend(Protocol):
    """
    Protocol defining the database interface for session persistence.

    All backends must implement these async methods to support:
    - Session lifecycle (CRUD)
    - Decision tracking
    - Metrics storage
    - Notes management
    - Agent execution logging
    - MCP session mapping
    """

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        ...

    async def initialize(self) -> None:
        """Initialize database connection and apply schema."""
        ...

    async def close(self) -> None:
        """Close database connection."""
        ...

    # Session operations
    async def save_session(self, session_data: dict[str, Any]) -> None:
        """Save or update a session."""
        ...

    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get a session by ID."""
        ...

    async def query_sessions(
        self,
        limit: int = 50,
        project_path: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Query sessions with optional filters."""
        ...

    async def get_active_session_for_project(
        self, project_path: str
    ) -> Optional[dict[str, Any]]:
        """Get the most recent active session for a project path."""
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID. Returns True if deleted."""
        ...

    # Decision operations
    async def save_decision(self, decision_data: dict[str, Any]) -> None:
        """Save a decision."""
        ...

    async def query_decisions_by_category(
        self, category: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query decisions by category across sessions."""
        ...

    async def query_decisions_by_session(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query decisions for a specific session."""
        ...

    # Metrics operations
    async def save_metrics(self, metrics_data: dict[str, Any]) -> None:
        """Save metrics snapshot."""
        ...

    async def query_metrics_by_branch(
        self, branch: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query metrics by branch across sessions."""
        ...

    async def query_metrics_by_session(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query metrics for a specific session."""
        ...

    # Notes operations
    async def save_note(self, note_data: dict[str, Any]) -> None:
        """Save a session note."""
        ...

    async def query_notes_by_date(
        self, date: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query notes by date across sessions."""
        ...

    # Agent execution operations
    async def save_agent_execution(self, execution_data: dict[str, Any]) -> None:
        """Save agent execution record."""
        ...

    async def query_agent_executions(
        self,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query agent executions with optional filters."""
        ...

    # MCP session operations
    async def save_mcp_session(self, mcp_session_data: dict[str, Any]) -> None:
        """Save MCP session mapping."""
        ...

    async def get_mcp_session(self, mcp_session_id: str) -> Optional[dict[str, Any]]:
        """Get MCP session by ID."""
        ...

    async def update_mcp_session_activity(self, mcp_session_id: str) -> None:
        """Update last activity timestamp for MCP session."""
        ...

    async def link_mcp_to_engine_session(
        self, mcp_session_id: str, engine_session_id: str
    ) -> None:
        """Link MCP session to engine session."""
        ...

    # Maintenance operations
    async def vacuum(self) -> None:
        """Optimize database storage."""
        ...

    async def get_statistics(self) -> dict[str, Any]:
        """Get database statistics for monitoring."""
        ...


class BaseDatabaseBackend:
    """
    Base class with shared utilities for database backends.

    Provides common functionality used by both SQLite and PostgreSQL.
    """

    SCHEMA_VERSION = 2  # Bumped for PostgreSQL compatibility

    def __init__(self) -> None:
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def _serialize_json(self, obj: Any) -> str:
        """Serialize object to JSON string."""
        import json

        if obj is None:
            return "{}"
        if isinstance(obj, str):
            return obj
        return json.dumps(obj, default=str)

    def _deserialize_json(self, json_str: Optional[str]) -> dict[str, Any]:
        """Deserialize JSON string to dict."""
        import json

        if not json_str:
            return {}
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}

    def _normalize_session_data(self, row: dict[str, Any]) -> dict[str, Any]:
        """Normalize session row to consistent format."""
        return {
            "id": row.get("id"),
            "started": row.get("started_at") or row.get("started"),
            "completed": row.get("ended_at") or row.get("completed"),
            "project_path": row.get("project_path", ""),
            "project_name": row.get("project_name"),
            "mode": row.get("mode", "local"),
            "status": row.get("status", "active"),
            "metadata": self._deserialize_json(row.get("metadata")),
            "performance_metrics": self._deserialize_json(row.get("performance_metrics")),
            "health_status": self._deserialize_json(row.get("health_status")),
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()
