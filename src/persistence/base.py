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

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Default global location for session intelligence data
DEFAULT_DATA_DIR = Path.home() / ".claude" / "session-intelligence"
DEFAULT_POSTGRES_DSN = "postgresql://localhost/session_intelligence"
# SQLite path for testing (SQLite is test-only, not for production)
DEFAULT_SQLITE_PATH = DEFAULT_DATA_DIR / "sessions.db"

# Issue #69: staleness threshold (hours) for 'active' sessions. A session
# that never received an explicit finalize call stays 'active' forever
# unless something reaps it. This single constant backs both the read guard
# (get_active_session_for_project / find_recent_session_by_project) and the
# startup sweep (reap_abandoned_sessions) in both backends, so they can never
# drift apart. Overridable via SESSION_INTELLIGENCE_SESSION_MAX_AGE_HOURS.
DEFAULT_SESSION_MAX_AGE_HOURS = 24


def get_session_max_age_hours() -> int:
    """Return the staleness threshold (hours) for 'active' sessions.

    Issue #82: guards and sweeps compare against COALESCE(last_seen_at,
    started_at), so a session that has been heartbeat-updated is judged by
    its most recent activity rather than only its creation time. Rows
    predating the last_seen_at migration fall back to started_at via the
    same COALESCE, so behavior is unaffected until they receive a heartbeat.
    """
    raw = os.environ.get("SESSION_INTELLIGENCE_SESSION_MAX_AGE_HOURS")
    if raw is None:
        return DEFAULT_SESSION_MAX_AGE_HOURS
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_SESSION_MAX_AGE_HOURS


# Issue #70: staleness threshold (hours) for 'running' agent_executions. The
# SubagentStop hook ("agent_stop" phase) is the only signal that transitions
# an execution out of RUNNING (see session_engine.py). If that event never
# arrives -- agent killed, session ends mid-flight, hook fails/times out,
# server restart between start and stop -- the row stays 'running' forever,
# permanently inflating the success_rate denominator (see get_agent_stats).
# Mirrors DEFAULT_SESSION_MAX_AGE_HOURS / get_session_max_age_hours() exactly
# so the two staleness sweeps cannot drift apart in behavior. Overridable via
# SESSION_INTELLIGENCE_EXECUTION_MAX_AGE_HOURS.
DEFAULT_EXECUTION_MAX_AGE_HOURS = 24


def get_execution_max_age_hours() -> int:
    """Return the staleness threshold (hours) for 'running' agent_executions."""
    raw = os.environ.get("SESSION_INTELLIGENCE_EXECUTION_MAX_AGE_HOURS")
    if raw is None:
        return DEFAULT_EXECUTION_MAX_AGE_HOURS
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_EXECUTION_MAX_AGE_HOURS


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
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query sessions with optional filters.

        Ordered by started_at DESC with `id` as a tiebreaker, so paginating by
        `offset` cannot skip or repeat rows that share a timestamp. Paginate by
        increasing offset until fewer than `limit` rows are returned.
        """
        ...

    async def get_active_session_for_project(self, project_path: str) -> dict[str, Any] | None:
        """Get the most recent active session for a project path."""
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID. Returns True if deleted."""
        ...

    async def reap_abandoned_sessions(self, older_than_hours: int | None = None) -> int:
        """Flip stale 'active' sessions to 'abandoned'. Returns rows affected.

        Distinct from 'completed': an abandoned session never received an
        explicit finalize call, so the status stays honest about that.
        Defaults to get_session_max_age_hours() when older_than_hours is None.
        """
        ...

    async def reap_stale_executions(self, older_than_hours: int | None = None) -> int:
        """Flip stale 'running' agent_executions to 'abandoned'. Returns rows affected.

        Issue #70: reconcile-on-finalize (see session_engine._finalize_session)
        catches executions whose session was explicitly finalized; this sweep
        catches the rest (server restart, crash, etc.) so no execution stays
        'running' forever. Defaults to get_execution_max_age_hours() when
        older_than_hours is None.
        """
        ...

    # Decision operations
    async def save_decision(self, decision_data: dict[str, Any]) -> None:
        """Save a decision."""
        ...

    async def query_decisions_by_category(
        self, category: str, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Query decisions by category across sessions.

        Ordered by timestamp DESC with `id` as a tiebreaker; paginated like
        query_sessions.
        """
        ...

    async def query_decisions_by_session(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Query decisions for a specific session.

        Ordered by timestamp DESC with `id` as a tiebreaker; paginated like
        query_sessions.
        """
        ...

    # Metrics operations
    async def save_metrics(self, metrics_data: dict[str, Any]) -> None:
        """Save metrics snapshot."""
        ...

    async def query_metrics_by_branch(self, branch: str, limit: int = 100) -> list[dict[str, Any]]:
        """Query metrics by branch across sessions."""
        ...

    async def query_metrics_by_session(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Query metrics for a specific session.

        Ordered by timestamp DESC with `id` as a tiebreaker; paginated like
        query_sessions.
        """
        ...

    # Notes operations
    async def save_note(self, note_data: dict[str, Any]) -> None:
        """Save a session note."""
        ...

    async def query_notes_by_date(self, date: str, limit: int = 100) -> list[dict[str, Any]]:
        """Query notes by date across sessions."""
        ...

    async def query_notes(self, limit: int = 1000, offset: int = 0) -> list[dict[str, Any]]:
        """Query notes across all sessions, ordered by id.

        Unlike query_notes_by_date this does NOT join sessions, so notes whose
        session row is missing (orphans) are still returned. Paginate by
        increasing offset until fewer than `limit` rows are returned.
        """
        ...

    async def query_mcp_sessions(self, limit: int = 1000, offset: int = 0) -> list[dict[str, Any]]:
        """Query MCP session mappings, ordered by a stable key. Paginated like query_notes."""
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
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query agent executions with optional filters.

        Ordered by started_at DESC with `id` as a tiebreaker; paginated like
        query_sessions.
        """
        ...

    async def get_agent_stats(self, time_window_hours: int = 168) -> dict[str, Any]:
        """Return per-agent-type usage statistics over the last time_window_hours hours.

        Returns a dict with "total_sessions_scanned" (int) and "agents" (list of
        per-agent-type stat dicts).
        """
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
            "last_seen_at": row.get("last_seen_at"),
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
