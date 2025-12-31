"""
PostgreSQL database backend for session persistence.

Optimal for:
- Production environments
- Multi-agent analysis
- Concurrent access from multiple processes
- Real-time notifications via LISTEN/NOTIFY

Requires: asyncpg

Connection string format:
    postgresql://user:password@host:port/database
    postgresql://localhost/session_intelligence
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore

from .base import BaseDatabaseBackend, DEFAULT_POSTGRES_DSN

logger = logging.getLogger(__name__)


class PostgreSQLBackend(BaseDatabaseBackend):
    """PostgreSQL database backend with async support for session persistence."""

    SCHEMA = """
    -- Schema version tracking
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    -- Sessions table
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        started_at TIMESTAMPTZ NOT NULL,
        ended_at TIMESTAMPTZ,
        project_path TEXT NOT NULL,
        project_name TEXT,
        mode TEXT DEFAULT 'local',
        status TEXT DEFAULT 'active',
        metadata JSONB DEFAULT '{}',
        performance_metrics JSONB DEFAULT '{}',
        health_status JSONB DEFAULT '{}'
    );

    CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_path);
    CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
    CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
    CREATE INDEX IF NOT EXISTS idx_sessions_metadata ON sessions USING GIN (metadata);

    -- Decisions table with category index for filtering
    CREATE TABLE IF NOT EXISTS decisions (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        category TEXT,
        description TEXT NOT NULL,
        rationale TEXT,
        context JSONB DEFAULT '{}',
        impact_level TEXT DEFAULT 'medium',
        artifacts JSONB DEFAULT '[]'
    );

    CREATE INDEX IF NOT EXISTS idx_decisions_session ON decisions(session_id);
    CREATE INDEX IF NOT EXISTS idx_decisions_category ON decisions(category);
    CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);

    -- Metrics table for time-series data
    CREATE TABLE IF NOT EXISTS metrics (
        id SERIAL PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        branch TEXT,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        coverage REAL,
        complexity REAL,
        test_count INTEGER,
        agents_executed INTEGER,
        execution_time_ms INTEGER,
        custom_metrics JSONB DEFAULT '{}'
    );

    CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics(session_id);
    CREATE INDEX IF NOT EXISTS idx_metrics_branch ON metrics(branch);
    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);

    -- Notes table for session notes
    CREATE TABLE IF NOT EXISTS notes (
        id SERIAL PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        date DATE NOT NULL DEFAULT CURRENT_DATE,
        content TEXT NOT NULL,
        tags JSONB DEFAULT '[]'
    );

    CREATE INDEX IF NOT EXISTS idx_notes_session ON notes(session_id);
    CREATE INDEX IF NOT EXISTS idx_notes_date ON notes(date);

    -- Agent executions table
    CREATE TABLE IF NOT EXISTS agent_executions (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        agent_name TEXT NOT NULL,
        agent_type TEXT,
        started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        completed_at TIMESTAMPTZ,
        status TEXT DEFAULT 'running',
        execution_steps JSONB DEFAULT '[]',
        performance JSONB DEFAULT '{}',
        errors JSONB DEFAULT '[]'
    );

    CREATE INDEX IF NOT EXISTS idx_agent_executions_session ON agent_executions(session_id);
    CREATE INDEX IF NOT EXISTS idx_agent_executions_agent ON agent_executions(agent_name);
    CREATE INDEX IF NOT EXISTS idx_agent_executions_status ON agent_executions(status);

    -- MCP session mapping table
    CREATE TABLE IF NOT EXISTS mcp_sessions (
        mcp_session_id TEXT PRIMARY KEY,
        engine_session_id TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        client_info JSONB DEFAULT '{}'
    );

    CREATE INDEX IF NOT EXISTS idx_mcp_sessions_engine ON mcp_sessions(engine_session_id);
    CREATE INDEX IF NOT EXISTS idx_mcp_sessions_activity ON mcp_sessions(last_activity);

    -- Analytical views for cross-session analysis
    CREATE OR REPLACE VIEW session_analytics AS
    SELECT
        s.id,
        s.project_path,
        s.project_name,
        s.started_at,
        s.ended_at,
        s.status,
        EXTRACT(EPOCH FROM (COALESCE(s.ended_at, NOW()) - s.started_at)) / 60 as duration_minutes,
        (SELECT COUNT(*) FROM decisions d WHERE d.session_id = s.id) as decision_count,
        (SELECT COUNT(*) FROM agent_executions ae WHERE ae.session_id = s.id) as agent_count,
        (SELECT COUNT(*) FROM metrics m WHERE m.session_id = s.id) as metrics_count
    FROM sessions s;

    CREATE OR REPLACE VIEW agent_performance_summary AS
    SELECT
        agent_name,
        COUNT(*) as total_executions,
        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful,
        COUNT(CASE WHEN status = 'error' THEN 1 END) as failed,
        AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds
    FROM agent_executions
    WHERE completed_at IS NOT NULL
    GROUP BY agent_name;

    CREATE OR REPLACE VIEW decision_summary AS
    SELECT
        category,
        impact_level,
        COUNT(*) as count,
        MIN(timestamp) as first_decision,
        MAX(timestamp) as last_decision
    FROM decisions
    GROUP BY category, impact_level;
    """

    def __init__(self, dsn: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize PostgreSQL backend.

        Args:
            dsn: PostgreSQL connection string. Defaults to postgresql://localhost/session_intelligence.
            **kwargs: Additional arguments passed to asyncpg.create_pool().
        """
        super().__init__()

        if asyncpg is None:
            raise ImportError(
                "asyncpg is required for PostgreSQL backend. "
                "Install with: pixi add asyncpg"
            )

        self.dsn = dsn or DEFAULT_POSTGRES_DSN
        self._pool_kwargs = kwargs
        self._pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        """Initialize database connection pool and apply schema."""
        self._pool = await asyncpg.create_pool(
            self.dsn,
            **self._pool_kwargs,
        )

        # Apply schema
        async with self._pool.acquire() as conn:
            # Split schema into individual statements for PostgreSQL
            statements = [s.strip() for s in self.SCHEMA.split(";") if s.strip()]
            for statement in statements:
                try:
                    await conn.execute(statement)
                except Exception as e:
                    # Ignore "already exists" errors
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Schema statement warning: {e}")

            # Track schema version
            await conn.execute(
                """
                INSERT INTO schema_version (version)
                VALUES ($1)
                ON CONFLICT (version) DO NOTHING
                """,
                self.SCHEMA_VERSION,
            )

        self._is_connected = True
        logger.info(f"PostgreSQL database initialized: {self.dsn.split('@')[-1]}")

    async def close(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._is_connected = False
            logger.info("PostgreSQL connection pool closed")

    def _ensure_connected(self) -> None:
        """Raise error if not connected."""
        if not self._pool:
            raise RuntimeError("Database not initialized")

    def _to_jsonb(self, obj: Any) -> Any:
        """Convert object to JSONB-compatible format."""
        if obj is None:
            return None
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except json.JSONDecodeError:
                return obj
        return obj

    def _from_record(self, record: asyncpg.Record) -> dict[str, Any]:
        """Convert asyncpg Record to dict."""
        return dict(record)

    # Session operations

    async def save_session(self, session_data: dict[str, Any]) -> None:
        """Save or update a session."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions
                (id, started_at, ended_at, project_path, project_name, mode, status,
                 metadata, performance_metrics, health_status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    ended_at = EXCLUDED.ended_at,
                    status = EXCLUDED.status,
                    metadata = EXCLUDED.metadata,
                    performance_metrics = EXCLUDED.performance_metrics,
                    health_status = EXCLUDED.health_status
                """,
                session_data["id"],
                session_data.get("started") or session_data.get("started_at"),
                session_data.get("completed") or session_data.get("ended_at"),
                session_data.get("project_path", ""),
                session_data.get("project_name"),
                session_data.get("mode", "local"),
                session_data.get("status", "active"),
                json.dumps(session_data.get("metadata", {})),
                json.dumps(session_data.get("performance_metrics", {})),
                json.dumps(session_data.get("health_status", {})),
            )

            # Notify listeners of session change
            await conn.execute(
                "SELECT pg_notify('session_changes', $1)",
                json.dumps({"session_id": session_data["id"], "action": "upsert"}),
            )

    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get a session by ID."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM sessions WHERE id = $1", session_id
            )
            if row:
                return self._normalize_session_data(self._from_record(row))
            return None

    async def query_sessions(
        self,
        limit: int = 50,
        project_path: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Query sessions with optional filters."""
        self._ensure_connected()

        query = "SELECT * FROM sessions WHERE 1=1"
        params: list[Any] = []
        param_idx = 1

        if project_path:
            query += f" AND project_path = ${param_idx}"
            params.append(project_path)
            param_idx += 1
        if status:
            query += f" AND status = ${param_idx}"
            params.append(status)
            param_idx += 1

        query += f" ORDER BY started_at DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._normalize_session_data(self._from_record(row)) for row in rows]

    async def get_active_session_for_project(
        self, project_path: str
    ) -> Optional[dict[str, Any]]:
        """Get the most recent active session for a project path."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM sessions
                WHERE project_path = $1 AND status = 'active'
                ORDER BY started_at DESC
                LIMIT 1
                """,
                project_path,
            )
            if row:
                return self._normalize_session_data(self._from_record(row))
            return None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID (cascades to related tables)."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM sessions WHERE id = $1", session_id
            )
            deleted = result == "DELETE 1"

            if deleted:
                await conn.execute(
                    "SELECT pg_notify('session_changes', $1)",
                    json.dumps({"session_id": session_id, "action": "delete"}),
                )

            return deleted

    # Decision operations

    async def save_decision(self, decision_data: dict[str, Any]) -> None:
        """Save a decision."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO decisions
                (id, session_id, timestamp, category, description, rationale,
                 context, impact_level, artifacts)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO UPDATE SET
                    description = EXCLUDED.description,
                    rationale = EXCLUDED.rationale,
                    context = EXCLUDED.context
                """,
                decision_data.get("decision_id") or decision_data.get("id"),
                decision_data["session_id"],
                decision_data.get("timestamp", self._get_timestamp()),
                decision_data.get("category"),
                decision_data.get("description") or decision_data.get("decision", ""),
                decision_data.get("rationale"),
                json.dumps(decision_data.get("context", {})),
                decision_data.get("impact_level", "medium"),
                json.dumps(decision_data.get("artifacts", [])),
            )

    async def query_decisions_by_category(
        self, category: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query decisions by category across sessions."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT d.*, s.project_name
                FROM decisions d
                JOIN sessions s ON d.session_id = s.id
                WHERE d.category = $1
                ORDER BY d.timestamp DESC
                LIMIT $2
                """,
                category,
                limit,
            )
            return [self._from_record(row) for row in rows]

    async def query_decisions_by_session(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query decisions for a specific session."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM decisions
                WHERE session_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                session_id,
                limit,
            )
            return [self._from_record(row) for row in rows]

    # Metrics operations

    async def save_metrics(self, metrics_data: dict[str, Any]) -> None:
        """Save metrics snapshot."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO metrics
                (session_id, branch, timestamp, coverage, complexity, test_count,
                 agents_executed, execution_time_ms, custom_metrics)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                metrics_data["session_id"],
                metrics_data.get("branch"),
                metrics_data.get("timestamp", self._get_timestamp()),
                metrics_data.get("coverage"),
                metrics_data.get("complexity"),
                metrics_data.get("test_count"),
                metrics_data.get("agents_executed"),
                metrics_data.get("execution_time_ms"),
                json.dumps(metrics_data.get("custom_metrics", {})),
            )

    async def query_metrics_by_branch(
        self, branch: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query metrics by branch across sessions."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM metrics
                WHERE branch = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                branch,
                limit,
            )
            return [self._from_record(row) for row in rows]

    async def query_metrics_by_session(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query metrics for a specific session."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM metrics
                WHERE session_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                session_id,
                limit,
            )
            return [self._from_record(row) for row in rows]

    # Notes operations

    async def save_note(self, note_data: dict[str, Any]) -> None:
        """Save a session note."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO notes (session_id, date, content, tags)
                VALUES ($1, $2, $3, $4)
                """,
                note_data["session_id"],
                note_data.get("date"),
                note_data["content"],
                json.dumps(note_data.get("tags", [])),
            )

    async def query_notes_by_date(
        self, date: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query notes by date across sessions."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT n.*, s.project_name
                FROM notes n
                JOIN sessions s ON n.session_id = s.id
                WHERE n.date = $1
                ORDER BY n.id DESC
                LIMIT $2
                """,
                date,
                limit,
            )
            return [self._from_record(row) for row in rows]

    # Agent execution operations

    async def save_agent_execution(self, execution_data: dict[str, Any]) -> None:
        """Save agent execution record."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_executions
                (id, session_id, agent_name, agent_type, started_at, completed_at,
                 status, execution_steps, performance, errors)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    completed_at = EXCLUDED.completed_at,
                    status = EXCLUDED.status,
                    execution_steps = EXCLUDED.execution_steps,
                    performance = EXCLUDED.performance,
                    errors = EXCLUDED.errors
                """,
                execution_data["id"],
                execution_data["session_id"],
                execution_data["agent_name"],
                execution_data.get("agent_type"),
                execution_data.get("started_at", self._get_timestamp()),
                execution_data.get("completed_at"),
                execution_data.get("status", "running"),
                json.dumps(execution_data.get("execution_steps", [])),
                json.dumps(execution_data.get("performance", {})),
                json.dumps(execution_data.get("errors", [])),
            )

    async def query_agent_executions(
        self,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query agent executions with optional filters."""
        self._ensure_connected()

        query = "SELECT * FROM agent_executions WHERE 1=1"
        params: list[Any] = []
        param_idx = 1

        if session_id:
            query += f" AND session_id = ${param_idx}"
            params.append(session_id)
            param_idx += 1
        if agent_name:
            query += f" AND agent_name = ${param_idx}"
            params.append(agent_name)
            param_idx += 1

        query += f" ORDER BY started_at DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._from_record(row) for row in rows]

    # MCP session operations

    async def save_mcp_session(self, mcp_session_data: dict[str, Any]) -> None:
        """Save MCP session mapping."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO mcp_sessions
                (mcp_session_id, engine_session_id, created_at, last_activity, client_info)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (mcp_session_id) DO UPDATE SET
                    engine_session_id = EXCLUDED.engine_session_id,
                    last_activity = EXCLUDED.last_activity,
                    client_info = EXCLUDED.client_info
                """,
                mcp_session_data["mcp_session_id"],
                mcp_session_data.get("engine_session_id"),
                mcp_session_data.get("created_at", self._get_timestamp()),
                mcp_session_data.get("last_activity", self._get_timestamp()),
                json.dumps(mcp_session_data.get("client_info", {})),
            )

    async def get_mcp_session(self, mcp_session_id: str) -> Optional[dict[str, Any]]:
        """Get MCP session by ID."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM mcp_sessions WHERE mcp_session_id = $1",
                mcp_session_id,
            )
            if row:
                return self._from_record(row)
            return None

    async def update_mcp_session_activity(self, mcp_session_id: str) -> None:
        """Update last activity timestamp for MCP session."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE mcp_sessions SET last_activity = NOW() WHERE mcp_session_id = $1",
                mcp_session_id,
            )

    async def link_mcp_to_engine_session(
        self, mcp_session_id: str, engine_session_id: str
    ) -> None:
        """Link MCP session to engine session."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE mcp_sessions SET engine_session_id = $1 WHERE mcp_session_id = $2",
                engine_session_id,
                mcp_session_id,
            )

    # Maintenance operations

    async def vacuum(self) -> None:
        """Optimize database storage (VACUUM ANALYZE)."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            # Use ANALYZE instead of VACUUM (VACUUM requires exclusive access)
            await conn.execute("ANALYZE")
        logger.info("PostgreSQL database analyzed")

    async def get_statistics(self) -> dict[str, Any]:
        """Get database statistics for monitoring."""
        self._ensure_connected()

        stats: dict[str, Any] = {
            "backend": "postgresql",
            "dsn": self.dsn.split("@")[-1] if "@" in self.dsn else self.dsn,
        }

        async with self._pool.acquire() as conn:
            # Get table counts
            tables = [
                "sessions",
                "decisions",
                "metrics",
                "notes",
                "agent_executions",
                "mcp_sessions",
            ]
            for table in tables:
                row = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {table}")
                stats[f"{table}_count"] = row["count"] if row else 0

            # Get database size
            row = await conn.fetchrow(
                "SELECT pg_database_size(current_database()) as size"
            )
            if row:
                stats["size_bytes"] = row["size"]

            # Get pool stats
            stats["pool_size"] = self._pool.get_size()
            stats["pool_free"] = self._pool.get_idle_size()

        return stats

    # PostgreSQL-specific analytical methods

    async def get_session_analytics(
        self, project_path: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get session analytics from the pre-built view."""
        self._ensure_connected()

        query = "SELECT * FROM session_analytics"
        params: list[Any] = []
        param_idx = 1

        if project_path:
            query += f" WHERE project_path = ${param_idx}"
            params.append(project_path)
            param_idx += 1

        query += f" ORDER BY started_at DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._from_record(row) for row in rows]

    async def get_agent_performance_summary(self) -> list[dict[str, Any]]:
        """Get agent performance summary from the pre-built view."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM agent_performance_summary ORDER BY total_executions DESC"
            )
            return [self._from_record(row) for row in rows]

    async def get_decision_summary(self) -> list[dict[str, Any]]:
        """Get decision summary from the pre-built view."""
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM decision_summary ORDER BY count DESC"
            )
            return [self._from_record(row) for row in rows]

    async def subscribe_to_changes(self, callback: Any) -> None:
        """Subscribe to real-time session changes via LISTEN/NOTIFY.

        This enables analysis agents to react to new data in real-time.
        """
        self._ensure_connected()

        async with self._pool.acquire() as conn:
            await conn.add_listener("session_changes", callback)
            logger.info("Subscribed to session_changes notifications")
