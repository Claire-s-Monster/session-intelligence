"""
Database abstraction layer for session persistence.

Uses PostgreSQL for production-grade session management with
connection pooling, concurrent access, and cross-session analytics.

Usage:
    from persistence.base import DatabaseBackend
    from persistence.postgresql import PostgreSQLBackend

    db: DatabaseBackend = PostgreSQLBackend(dsn="postgresql://localhost/session_intelligence")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Default global location for session intelligence data
DEFAULT_DATA_DIR = Path.home() / ".claude" / "session-intelligence"
DEFAULT_POSTGRES_DSN = "postgresql://localhost/session_intelligence"
# SQLite path for testing (SQLite is test-only, not for production)
DEFAULT_SQLITE_PATH = DEFAULT_DATA_DIR / "sessions.db"


def get_default_data_dir() -> Path:
    """Get the default data directory, creating it if needed."""
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DATA_DIR


def sanitize_dsn(dsn: str) -> str:
    """Remove password from DSN for safe logging.

    Args:
        dsn: Database connection string (e.g., postgresql://user:password@host:5432/db)

    Returns:
        Sanitized DSN with password replaced by '***'

    Examples:
        >>> sanitize_dsn("postgresql://user:secret@localhost:5432/db")
        'postgresql://user:***@localhost:5432/db'
        >>> sanitize_dsn("postgresql://localhost/db")
        'postgresql://localhost/db'
    """
    import re

    # Match pattern: ://user:password@ where password can contain any chars except ://
    # Use greedy match for password to handle @ in password, match until last @ before host
    return re.sub(r"://([^/:]+):(.+)@([^@]+)$", r"://\1:***@\3", dsn)


# Database retry decorator for transient failures
try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    # Build list of retryable exceptions
    _retryable_exceptions: list[type[Exception]] = [ConnectionError, TimeoutError, OSError]

    # Add asyncpg-specific exceptions if available
    try:
        import asyncpg

        _retryable_exceptions.extend([
            asyncpg.PostgresConnectionError,
            asyncpg.InterfaceError,
        ])
    except ImportError:
        pass

    db_retry = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(tuple(_retryable_exceptions)),
        reraise=True,
    )
except ImportError:
    # Fallback: no-op decorator if tenacity not installed
    from collections.abc import Callable
    from functools import wraps
    from typing import TypeVar

    F = TypeVar("F", bound=Callable[..., Any])

    def db_retry(func: F) -> F:
        """No-op decorator when tenacity is not available."""
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)
        return wrapper  # type: ignore


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

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get a session by ID."""
        ...

    async def query_sessions(
        self,
        limit: int = 50,
        project_path: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query sessions with optional filters."""
        ...

    async def get_active_session_for_project(self, project_path: str) -> dict[str, Any] | None:
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

    async def query_metrics_by_branch(self, branch: str, limit: int = 100) -> list[dict[str, Any]]:
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

    async def query_notes_by_date(self, date: str, limit: int = 100) -> list[dict[str, Any]]:
        """Query notes by date across sessions."""
        ...

    # Agent execution operations
    async def save_agent_execution(self, execution_data: dict[str, Any]) -> None:
        """Save agent execution record."""
        ...

    async def query_agent_executions(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query agent executions with optional filters."""
        ...

    # MCP session operations
    async def save_mcp_session(self, mcp_session_data: dict[str, Any]) -> None:
        """Save MCP session mapping."""
        ...

    async def get_mcp_session(self, mcp_session_id: str) -> dict[str, Any] | None:
        """Get MCP session by ID."""
        ...

    async def update_mcp_session_activity(self, mcp_session_id: str) -> None:
        """Update last activity timestamp for MCP session."""
        ...

    async def link_mcp_to_engine_session(self, mcp_session_id: str, engine_session_id: str) -> None:
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

    def _deserialize_json(self, json_str: str | dict | list | None) -> dict[str, Any]:
        """Deserialize JSON string to dict.

        Handles both string JSON and already-parsed dicts (from PostgreSQL JSONB).
        """
        import json

        if not json_str:
            return {}
        # Already a dict (PostgreSQL JSONB returns native Python types)
        if isinstance(json_str, dict):
            return json_str
        # Already a list - wrap in dict for consistency
        if isinstance(json_str, list):
            return {"items": json_str}
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
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
