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
from typing import Any

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore

from .base import DEFAULT_POSTGRES_DSN, BaseDatabaseBackend, db_retry, sanitize_dsn

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

    -- File operations tracking
    CREATE TABLE IF NOT EXISTS file_operations (
        id SERIAL PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        operation TEXT NOT NULL,
        file_path TEXT NOT NULL,
        lines_added INTEGER DEFAULT 0,
        lines_removed INTEGER DEFAULT 0,
        summary TEXT,
        tool_name TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_file_ops_session ON file_operations(session_id);

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

    -- Session summaries table for narrative session documentation
    CREATE TABLE IF NOT EXISTS session_summaries (
        session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
        title TEXT,
        summary_markdown TEXT,
        key_changes JSONB DEFAULT '[]',
        tags JSONB DEFAULT '[]',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_summaries_created ON session_summaries(created_at);

    -- Project-specific learnings (trial/error history)
    CREATE TABLE IF NOT EXISTS project_learnings (
        id TEXT PRIMARY KEY,
        project_path TEXT NOT NULL,
        category TEXT NOT NULL,
        trigger_context TEXT,
        learning_content TEXT NOT NULL,
        source_session_id TEXT REFERENCES sessions(id),
        success_count INTEGER DEFAULT 1,
        failure_count INTEGER DEFAULT 0,
        last_used TIMESTAMPTZ,
        promoted_to_universal BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_learnings_project ON project_learnings(project_path);
    CREATE INDEX IF NOT EXISTS idx_learnings_category ON project_learnings(category);
    CREATE INDEX IF NOT EXISTS idx_learnings_promoted ON project_learnings(promoted_to_universal);

    -- Error→Solution quick lookup
    CREATE TABLE IF NOT EXISTS error_solutions (
        id TEXT PRIMARY KEY,
        error_pattern TEXT NOT NULL,
        error_hash TEXT,
        error_category TEXT,
        solution_steps JSONB NOT NULL DEFAULT '[]',
        context_requirements JSONB,
        success_rate REAL DEFAULT 1.0,
        usage_count INTEGER DEFAULT 1,
        project_path TEXT,
        source_session_id TEXT REFERENCES sessions(id),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_used TIMESTAMPTZ
    );

    CREATE INDEX IF NOT EXISTS idx_errors_hash ON error_solutions(error_hash);
    CREATE INDEX IF NOT EXISTS idx_errors_category ON error_solutions(error_category);
    CREATE INDEX IF NOT EXISTS idx_errors_project ON error_solutions(project_path);

    -- AGENTS TABLE (cross-session agent identity and statistics)
    CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        agent_type TEXT NOT NULL,
        display_name TEXT,
        description TEXT,
        metadata JSONB DEFAULT '{}',
        capabilities JSONB DEFAULT '[]',
        first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_active_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        total_executions INTEGER DEFAULT 0,
        total_decisions INTEGER DEFAULT 0,
        total_learnings INTEGER DEFAULT 0,
        total_notebooks INTEGER DEFAULT 0,
        is_active BOOLEAN DEFAULT TRUE
    );
    CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name);
    CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type);

    -- AGENT_DECISIONS TABLE (cross-session decision tracking)
    CREATE TABLE IF NOT EXISTS agent_decisions (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        description TEXT NOT NULL,
        rationale TEXT,
        category TEXT,
        impact_level TEXT DEFAULT 'medium',
        context JSONB DEFAULT '{}',
        artifacts JSONB DEFAULT '[]',
        source_session_id TEXT,
        source_project_path TEXT,
        outcome TEXT,
        outcome_notes TEXT,
        outcome_updated_at TIMESTAMPTZ
    );
    CREATE INDEX IF NOT EXISTS idx_agent_decisions_agent ON agent_decisions(agent_id);
    CREATE INDEX IF NOT EXISTS idx_agent_decisions_category ON agent_decisions(category);
    CREATE INDEX IF NOT EXISTS idx_agent_decisions_timestamp ON agent_decisions(timestamp);

    -- AGENT_LEARNINGS TABLE (cross-session agent learnings)
    CREATE TABLE IF NOT EXISTS agent_learnings (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
        category TEXT NOT NULL,
        trigger_context TEXT,
        learning_content TEXT NOT NULL,
        applies_to JSONB DEFAULT '{}',
        success_count INTEGER DEFAULT 1,
        failure_count INTEGER DEFAULT 0,
        last_used_at TIMESTAMPTZ,
        source_session_id TEXT,
        source_project_path TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_agent_learnings_agent ON agent_learnings(agent_id);
    CREATE INDEX IF NOT EXISTS idx_agent_learnings_category ON agent_learnings(category);

    -- AGENT_NOTEBOOKS TABLE (cross-session agent notebooks/summaries)
    CREATE TABLE IF NOT EXISTS agent_notebooks (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        summary_markdown TEXT NOT NULL,
        notebook_type TEXT DEFAULT 'summary',
        tags JSONB DEFAULT '[]',
        key_insights JSONB DEFAULT '[]',
        related_sessions JSONB DEFAULT '[]',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        covers_from TIMESTAMPTZ,
        covers_to TIMESTAMPTZ
    );
    CREATE INDEX IF NOT EXISTS idx_agent_notebooks_agent ON agent_notebooks(agent_id);
    CREATE INDEX IF NOT EXISTS idx_agent_notebooks_type ON agent_notebooks(notebook_type);
    """

    def __init__(self, dsn: str | None = None, **kwargs: Any) -> None:
        """Initialize PostgreSQL backend.

        Args:
            dsn: PostgreSQL connection string. Defaults to postgresql://localhost/session_intelligence.
            **kwargs: Additional arguments passed to asyncpg.create_pool().
        """
        super().__init__()

        if asyncpg is None:
            raise ImportError(
                "asyncpg is required for PostgreSQL backend. " "Install with: pixi add asyncpg"
            )

        self.dsn = dsn or DEFAULT_POSTGRES_DSN
        self._pool_kwargs = kwargs
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Initialize database connection pool and apply schema."""
        # Default pool configuration for production use
        # These can be overridden via __init__ kwargs
        pool_defaults = {
            "min_size": 2,        # Minimum connections to maintain
            "max_size": 10,       # Maximum connections allowed
            "timeout": 30,        # Connection acquisition timeout (seconds)
            "command_timeout": 60,  # Command execution timeout (seconds)
        }
        # Merge defaults with user-provided kwargs (user kwargs take precedence)
        pool_config = {**pool_defaults, **self._pool_kwargs}

        self._pool = await asyncpg.create_pool(
            self.dsn,
            **pool_config,
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
        logger.info(f"PostgreSQL database initialized: {sanitize_dsn(self.dsn)}")

    async def close(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._is_connected = False
            logger.info("PostgreSQL connection pool closed")

    def _ensure_connected(self) -> asyncpg.Pool:
        """Return connection pool or raise if not connected.

        Returns:
            The asyncpg connection pool.

        Raises:
            RuntimeError: If database not initialized.
        """
        if self._pool is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._pool

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
        """Convert asyncpg Record to dict, parsing JSON strings for JSONB fields."""
        result = dict(record)

        # Parse JSON strings that should be dicts/lists (JSONB fields stored as strings)
        json_fields = [
            "metadata",
            "capabilities",
            "tags",
            "alternatives",
            "applicability",
            "context",
            "decisions_referenced",
            "learnings_referenced",
            "performance_metrics",
            "health_status",
        ]

        for field in json_fields:
            if field in result and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    pass  # Keep as string if parsing fails

        return result

    # Session operations

    @db_retry
    async def save_session(self, session_data: dict[str, Any]) -> None:
        """Save or update a session."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
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

    @db_retry
    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get a session by ID."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM sessions WHERE id = $1", session_id)
            if row:
                return self._normalize_session_data(self._from_record(row))
            return None

    @db_retry
    async def query_sessions(
        self,
        limit: int = 50,
        project_path: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query sessions with optional filters."""
        pool = self._ensure_connected()

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

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._normalize_session_data(self._from_record(row)) for row in rows]

    @db_retry
    async def get_active_session_for_project(self, project_path: str) -> dict[str, Any] | None:
        """Get the most recent active session for a project path."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
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

    @db_retry
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID (cascades to related tables)."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM sessions WHERE id = $1", session_id)
            deleted = result == "DELETE 1"

            if deleted:
                await conn.execute(
                    "SELECT pg_notify('session_changes', $1)",
                    json.dumps({"session_id": session_id, "action": "delete"}),
                )

            return deleted

    # Decision operations

    @db_retry
    async def save_decision(self, decision_data: dict[str, Any]) -> None:
        """Save a decision."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
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

    @db_retry
    async def query_decisions_by_category(
        self, category: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query decisions by category across sessions."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
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
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
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

    @db_retry
    async def save_metrics(self, metrics_data: dict[str, Any]) -> None:
        """Save metrics snapshot."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
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

    async def query_metrics_by_branch(self, branch: str, limit: int = 100) -> list[dict[str, Any]]:
        """Query metrics by branch across sessions."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
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
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
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
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
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

    async def query_notes_by_date(self, date: str, limit: int = 100) -> list[dict[str, Any]]:
        """Query notes by date across sessions."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
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

    # File operations tracking

    async def save_file_operation(self, file_op_data: dict[str, Any]) -> None:
        """Save a file operation record."""
        pool = self._ensure_connected()

        # Parse timestamp if string
        timestamp = file_op_data.get("timestamp", self._get_timestamp())
        if isinstance(timestamp, str):
            from datetime import datetime

            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO file_operations
                (session_id, timestamp, operation, file_path, lines_added, lines_removed, summary, tool_name)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                file_op_data["session_id"],
                timestamp,
                file_op_data["operation"],
                file_op_data["file_path"],
                file_op_data.get("lines_added", 0),
                file_op_data.get("lines_removed", 0),
                file_op_data.get("summary"),
                file_op_data.get("tool_name"),
            )

    async def query_file_operations_by_session(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query file operations for a specific session."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM file_operations
                WHERE session_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                session_id,
                limit,
            )
            return [self._from_record(row) for row in rows]

    # Session summaries (notebooks)

    async def save_session_summary(self, summary_data: dict[str, Any]) -> None:
        """Save or update a session summary/notebook."""
        pool = self._ensure_connected()

        # Parse timestamp if string
        created_at = summary_data.get("created_at", self._get_timestamp())
        if isinstance(created_at, str):
            from datetime import datetime

            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO session_summaries
                (session_id, title, summary_markdown, key_changes, tags, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (session_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    summary_markdown = EXCLUDED.summary_markdown,
                    key_changes = EXCLUDED.key_changes,
                    tags = EXCLUDED.tags,
                    created_at = EXCLUDED.created_at
                """,
                summary_data["session_id"],
                summary_data.get("title"),
                summary_data.get("summary_markdown"),
                json.dumps(summary_data.get("key_changes", [])),
                json.dumps(summary_data.get("tags", [])),
                created_at,
            )

    async def get_session_summary(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a session summary by session ID."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM session_summaries WHERE session_id = $1",
                session_id,
            )
            return self._from_record(row) if row else None

    async def query_session_summaries(
        self,
        project_path: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query session summaries/notebooks with optional filters."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            if tags:
                # Query by tags using JSONB containment
                if project_path:
                    query = """
                        SELECT ss.*, s.project_path, s.project_name
                        FROM session_summaries ss
                        JOIN sessions s ON ss.session_id = s.id
                        WHERE ss.tags @> $1::jsonb AND s.project_path = $2
                        ORDER BY ss.created_at DESC
                        LIMIT $3
                    """
                    rows = await conn.fetch(query, json.dumps([tags[0]]), project_path, limit)
                else:
                    query = """
                        SELECT ss.*, s.project_path, s.project_name
                        FROM session_summaries ss
                        JOIN sessions s ON ss.session_id = s.id
                        WHERE ss.tags @> $1::jsonb
                        ORDER BY ss.created_at DESC
                        LIMIT $2
                    """
                    rows = await conn.fetch(query, json.dumps([tags[0]]), limit)
            elif project_path:
                # Query by project
                query = """
                    SELECT ss.*, s.project_path, s.project_name
                    FROM session_summaries ss
                    JOIN sessions s ON ss.session_id = s.id
                    WHERE s.project_path = $1
                    ORDER BY ss.created_at DESC
                    LIMIT $2
                """
                rows = await conn.fetch(query, project_path, limit)
            else:
                # Query all recent
                query = """
                    SELECT ss.*, s.project_path, s.project_name
                    FROM session_summaries ss
                    JOIN sessions s ON ss.session_id = s.id
                    ORDER BY ss.created_at DESC
                    LIMIT $1
                """
                rows = await conn.fetch(query, limit)

            return [self._from_record(row) for row in rows]

    # Agent execution operations

    async def save_agent_execution(self, execution_data: dict[str, Any]) -> None:
        """Save agent execution record."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
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
        session_id: str | None = None,
        agent_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query agent executions with optional filters."""
        pool = self._ensure_connected()

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

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._from_record(row) for row in rows]

    # MCP session operations

    @db_retry
    async def save_mcp_session(self, mcp_session_data: dict[str, Any]) -> None:
        """Save MCP session mapping."""
        pool = self._ensure_connected()

        # Parse timestamps if strings (MCP session manager passes ISO strings)
        created_at = mcp_session_data.get("created_at", self._get_timestamp())
        if isinstance(created_at, str):
            from datetime import datetime

            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

        last_activity = mcp_session_data.get("last_activity", self._get_timestamp())
        if isinstance(last_activity, str):
            from datetime import datetime

            last_activity = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))

        async with pool.acquire() as conn:
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
                created_at,
                last_activity,
                json.dumps(mcp_session_data.get("client_info", {})),
            )

    @db_retry
    async def get_mcp_session(self, mcp_session_id: str) -> dict[str, Any] | None:
        """Get MCP session by ID."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM mcp_sessions WHERE mcp_session_id = $1",
                mcp_session_id,
            )
            if row:
                return self._from_record(row)
            return None

    async def update_mcp_session_activity(self, mcp_session_id: str) -> None:
        """Update last activity timestamp for MCP session."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE mcp_sessions SET last_activity = NOW() WHERE mcp_session_id = $1",
                mcp_session_id,
            )

    async def link_mcp_to_engine_session(self, mcp_session_id: str, engine_session_id: str) -> None:
        """Link MCP session to engine session."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE mcp_sessions SET engine_session_id = $1 WHERE mcp_session_id = $2",
                engine_session_id,
                mcp_session_id,
            )

    # Maintenance operations

    async def vacuum(self) -> None:
        """Optimize database storage (VACUUM ANALYZE)."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            # Use ANALYZE instead of VACUUM (VACUUM requires exclusive access)
            await conn.execute("ANALYZE")
        logger.info("PostgreSQL database analyzed")

    async def get_statistics(self) -> dict[str, Any]:
        """Get database statistics for monitoring."""
        pool = self._ensure_connected()

        stats: dict[str, Any] = {
            "backend": "postgresql",
            "dsn": sanitize_dsn(self.dsn),
        }

        async with pool.acquire() as conn:
            # Get table counts
            tables = [
                "sessions",
                "decisions",
                "metrics",
                "notes",
                "agent_executions",
                "mcp_sessions",
                "agents",
                "agent_decisions",
                "agent_learnings",
                "agent_notebooks",
            ]
            for table in tables:
                row = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {table}")
                stats[f"{table}_count"] = row["count"] if row else 0

            # Get database size
            row = await conn.fetchrow("SELECT pg_database_size(current_database()) as size")
            if row:
                stats["size_bytes"] = row["size"]

            # Get pool stats
            stats["pool_size"] = pool.get_size()
            stats["pool_free"] = pool.get_idle_size()

        return stats

    # PostgreSQL-specific analytical methods

    async def get_session_analytics(
        self, project_path: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get session analytics from the pre-built view."""
        pool = self._ensure_connected()

        query = "SELECT * FROM session_analytics"
        params: list[Any] = []
        param_idx = 1

        if project_path:
            query += f" WHERE project_path = ${param_idx}"
            params.append(project_path)
            param_idx += 1

        query += f" ORDER BY started_at DESC LIMIT ${param_idx}"
        params.append(limit)

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._from_record(row) for row in rows]

    async def get_agent_performance_summary(self) -> list[dict[str, Any]]:
        """Get agent performance summary from the pre-built view."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM agent_performance_summary ORDER BY total_executions DESC"
            )
            return [self._from_record(row) for row in rows]

    async def get_decision_summary(self) -> list[dict[str, Any]]:
        """Get decision summary from the pre-built view."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM decision_summary ORDER BY count DESC")
            return [self._from_record(row) for row in rows]

    async def subscribe_to_changes(self, callback: Any) -> None:
        """Subscribe to real-time session changes via LISTEN/NOTIFY.

        This enables analysis agents to react to new data in real-time.
        """
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            await conn.add_listener("session_changes", callback)
            logger.info("Subscribed to session_changes notifications")

    # ===== Knowledge System Methods =====

    async def save_project_learning(
        self,
        learning_id: str,
        project_path: str,
        category: str,
        learning_content: str,
        trigger_context: str | None = None,
        source_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Save a project-specific learning."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO project_learnings (
                    id, project_path, category, trigger_context, learning_content,
                    source_session_id, success_count, failure_count, last_used,
                    promoted_to_universal, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, 1, 0, NOW(), FALSE, NOW())
                ON CONFLICT(id) DO UPDATE SET
                    learning_content = EXCLUDED.learning_content,
                    trigger_context = EXCLUDED.trigger_context,
                    last_used = NOW()
                """,
                learning_id,
                project_path,
                category,
                trigger_context,
                learning_content,
                source_session_id,
            )
        return {"id": learning_id, "status": "saved"}

    async def query_project_learnings(
        self,
        project_path: str,
        category: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query learnings for a project."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            if category:
                rows = await conn.fetch(
                    """
                    SELECT * FROM project_learnings
                    WHERE project_path = $1 AND category = $2
                    ORDER BY success_count DESC, last_used DESC
                    LIMIT $3
                    """,
                    project_path,
                    category,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM project_learnings
                    WHERE project_path = $1
                    ORDER BY success_count DESC, last_used DESC
                    LIMIT $2
                    """,
                    project_path,
                    limit,
                )
            return [self._from_record(row) for row in rows]

    async def update_learning_usage(self, learning_id: str, success: bool) -> dict[str, Any]:
        """Update success/failure count for a learning."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            if success:
                await conn.execute(
                    """
                    UPDATE project_learnings
                    SET success_count = success_count + 1, last_used = NOW()
                    WHERE id = $1
                    """,
                    learning_id,
                )
            else:
                await conn.execute(
                    """
                    UPDATE project_learnings
                    SET failure_count = failure_count + 1, last_used = NOW()
                    WHERE id = $1
                    """,
                    learning_id,
                )
        return {"id": learning_id, "updated": True, "success": success}

    async def save_error_solution(
        self,
        solution_id: str,
        error_pattern: str,
        solution_steps: list[str],
        error_category: str | None = None,
        context_requirements: dict[str, Any] | None = None,
        project_path: str | None = None,
        source_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Save an error→solution mapping."""
        pool = self._ensure_connected()
        import hashlib

        error_hash = hashlib.sha256(error_pattern.encode()).hexdigest()[:16]

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO error_solutions (
                    id, error_pattern, error_hash, error_category, solution_steps,
                    context_requirements, success_rate, usage_count, project_path,
                    source_session_id, created_at, last_used
                ) VALUES ($1, $2, $3, $4, $5, $6, 1.0, 1, $7, $8, NOW(), NOW())
                ON CONFLICT(id) DO UPDATE SET
                    solution_steps = EXCLUDED.solution_steps,
                    context_requirements = EXCLUDED.context_requirements,
                    last_used = NOW()
                """,
                solution_id,
                error_pattern,
                error_hash,
                error_category,
                json.dumps(solution_steps),
                json.dumps(context_requirements) if context_requirements else None,
                project_path,
                source_session_id,
            )
        return {"id": solution_id, "error_hash": error_hash, "status": "saved"}

    async def find_error_solutions(
        self,
        error_text: str,
        project_path: str | None = None,
        include_universal: bool = True,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find solutions matching an error pattern."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            if project_path and include_universal:
                rows = await conn.fetch(
                    """
                    SELECT * FROM error_solutions
                    WHERE (project_path = $1 OR project_path IS NULL)
                    AND (error_pattern ILIKE $2 OR $3 ILIKE '%' || error_pattern || '%')
                    ORDER BY
                        CASE WHEN project_path = $1 THEN 0 ELSE 1 END,
                        success_rate DESC,
                        usage_count DESC
                    LIMIT $4
                    """,
                    project_path,
                    f"%{error_text[:100]}%",
                    error_text,
                    limit,
                )
            elif project_path:
                rows = await conn.fetch(
                    """
                    SELECT * FROM error_solutions
                    WHERE project_path = $1
                    AND (error_pattern ILIKE $2 OR $3 ILIKE '%' || error_pattern || '%')
                    ORDER BY success_rate DESC, usage_count DESC
                    LIMIT $4
                    """,
                    project_path,
                    f"%{error_text[:100]}%",
                    error_text,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM error_solutions
                    WHERE project_path IS NULL
                    AND (error_pattern ILIKE $1 OR $2 ILIKE '%' || error_pattern || '%')
                    ORDER BY success_rate DESC, usage_count DESC
                    LIMIT $3
                    """,
                    f"%{error_text[:100]}%",
                    error_text,
                    limit,
                )

            results = []
            for row in rows:
                result = self._from_record(row)
                if result.get("solution_steps"):
                    result["solution_steps"] = (
                        json.loads(result["solution_steps"])
                        if isinstance(result["solution_steps"], str)
                        else result["solution_steps"]
                    )
                if result.get("context_requirements"):
                    result["context_requirements"] = (
                        json.loads(result["context_requirements"])
                        if isinstance(result["context_requirements"], str)
                        else result["context_requirements"]
                    )
                results.append(result)
            return results

    async def update_solution_outcome(self, solution_id: str, success: bool) -> dict[str, Any]:
        """Update success rate for a solution."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT usage_count, success_rate FROM error_solutions WHERE id = $1",
                solution_id,
            )
            if not row:
                return {"id": solution_id, "error": "Solution not found"}

            usage_count = row["usage_count"]
            current_rate = row["success_rate"]

            new_usage = usage_count + 1
            if success:
                new_rate = (current_rate * usage_count + 1.0) / new_usage
            else:
                new_rate = (current_rate * usage_count) / new_usage

            await conn.execute(
                """
                UPDATE error_solutions
                SET usage_count = $1, success_rate = $2, last_used = NOW()
                WHERE id = $3
                """,
                new_usage,
                new_rate,
                solution_id,
            )

        return {
            "id": solution_id,
            "usage_count": new_usage,
            "success_rate": new_rate,
            "updated": True,
        }

    # ===== Agent System Operations =====

    @db_retry
    async def save_agent(self, agent_data: dict[str, Any]) -> dict[str, Any]:
        """Save or update an agent (ON CONFLICT DO UPDATE)."""
        pool = self._ensure_connected()

        # Parse timestamps if strings
        first_seen_at = agent_data.get("first_seen_at", self._get_timestamp())
        if isinstance(first_seen_at, str):
            from datetime import datetime

            first_seen_at = datetime.fromisoformat(first_seen_at.replace("Z", "+00:00"))

        last_active_at = agent_data.get("last_active_at", self._get_timestamp())
        if isinstance(last_active_at, str):
            from datetime import datetime

            last_active_at = datetime.fromisoformat(last_active_at.replace("Z", "+00:00"))

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agents (
                    id, name, agent_type, display_name, description,
                    metadata, capabilities, first_seen_at, last_active_at,
                    total_executions, total_decisions, total_learnings,
                    total_notebooks, is_active
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    agent_type = EXCLUDED.agent_type,
                    display_name = EXCLUDED.display_name,
                    description = EXCLUDED.description,
                    metadata = EXCLUDED.metadata,
                    capabilities = EXCLUDED.capabilities,
                    last_active_at = EXCLUDED.last_active_at,
                    is_active = EXCLUDED.is_active
                """,
                agent_data["id"],
                agent_data["name"],
                agent_data["agent_type"],
                agent_data.get("display_name"),
                agent_data.get("description"),
                json.dumps(agent_data.get("metadata", {})),
                json.dumps(agent_data.get("capabilities", [])),
                first_seen_at,
                last_active_at,
                agent_data.get("total_executions", 0),
                agent_data.get("total_decisions", 0),
                agent_data.get("total_learnings", 0),
                agent_data.get("total_notebooks", 0),
                agent_data.get("is_active", True),
            )
        return {"id": agent_data["id"], "status": "saved"}

    @db_retry
    async def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Get agent by ID."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM agents WHERE id = $1",
                agent_id,
            )
            return self._from_record(row) if row else None

    async def get_agent_by_name(self, name: str) -> dict[str, Any] | None:
        """Get agent by unique name."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM agents WHERE name = $1",
                name,
            )
            return self._from_record(row) if row else None

    async def update_agent_stats(self, agent_id: str, stat_type: str) -> dict[str, Any]:
        """Increment total_* counters for an agent.

        Args:
            agent_id: The agent's ID
            stat_type: One of 'executions', 'decisions', 'learnings', 'notebooks'

        Returns:
            Updated stats dict
        """
        pool = self._ensure_connected()

        column_map = {
            "executions": "total_executions",
            "decisions": "total_decisions",
            "learnings": "total_learnings",
            "notebooks": "total_notebooks",
        }

        if stat_type not in column_map:
            return {"error": f"Invalid stat_type: {stat_type}"}

        column = column_map[stat_type]

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE agents
                SET {column} = {column} + 1, last_active_at = NOW()
                WHERE id = $1
                RETURNING id, {column} as new_value
                """,
                agent_id,
            )
            if row:
                return {
                    "id": agent_id,
                    "stat_type": stat_type,
                    "new_value": row["new_value"],
                    "updated": True,
                }
            return {"id": agent_id, "error": "Agent not found"}

    @db_retry
    async def save_agent_decision(self, decision_data: dict[str, Any]) -> dict[str, Any]:
        """Save an agent decision."""
        pool = self._ensure_connected()

        # Parse timestamps if strings
        timestamp = decision_data.get("timestamp", self._get_timestamp())
        if isinstance(timestamp, str):
            from datetime import datetime

            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        outcome_updated_at = decision_data.get("outcome_updated_at")
        if isinstance(outcome_updated_at, str):
            from datetime import datetime

            outcome_updated_at = datetime.fromisoformat(outcome_updated_at.replace("Z", "+00:00"))

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_decisions (
                    id, agent_id, timestamp, description, rationale, category,
                    impact_level, context, artifacts, source_session_id,
                    source_project_path, outcome, outcome_notes, outcome_updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                decision_data["id"],
                decision_data["agent_id"],
                timestamp,
                decision_data["description"],
                decision_data.get("rationale"),
                decision_data.get("category"),
                decision_data.get("impact_level", "medium"),
                json.dumps(decision_data.get("context", {})),
                json.dumps(decision_data.get("artifacts", [])),
                decision_data.get("source_session_id"),
                decision_data.get("source_project_path"),
                decision_data.get("outcome"),
                decision_data.get("outcome_notes"),
                outcome_updated_at,
            )
        return {"id": decision_data["id"], "status": "saved"}

    async def query_agent_decisions(
        self,
        agent_id: str,
        category: str | None = None,
        outcome: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query agent decisions with optional filters."""
        pool = self._ensure_connected()

        query = "SELECT * FROM agent_decisions WHERE agent_id = $1"
        params: list[Any] = [agent_id]
        param_idx = 2

        if category:
            query += f" AND category = ${param_idx}"
            params.append(category)
            param_idx += 1
        if outcome:
            query += f" AND outcome = ${param_idx}"
            params.append(outcome)
            param_idx += 1

        query += f" ORDER BY timestamp DESC LIMIT ${param_idx}"
        params.append(limit)

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._from_record(row) for row in rows]

    async def update_agent_decision_outcome(
        self,
        decision_id: str,
        outcome: str,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Update the outcome of an agent decision."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE agent_decisions
                SET outcome = $1, outcome_notes = $2, outcome_updated_at = NOW()
                WHERE id = $3
                """,
                outcome,
                notes,
                decision_id,
            )
            updated = result == "UPDATE 1"
        return {"id": decision_id, "outcome": outcome, "updated": updated}

    @db_retry
    async def save_agent_learning(self, learning_data: dict[str, Any]) -> dict[str, Any]:
        """Save an agent learning."""
        pool = self._ensure_connected()

        # Parse timestamps if strings
        from datetime import datetime as dt

        last_used_at = learning_data.get("last_used_at")
        if isinstance(last_used_at, str):
            last_used_at = dt.fromisoformat(last_used_at.replace("Z", "+00:00"))

        created_at = learning_data.get("created_at", self._get_timestamp())
        if isinstance(created_at, str):
            created_at = dt.fromisoformat(created_at.replace("Z", "+00:00"))

        updated_at = learning_data.get("updated_at", self._get_timestamp())
        if isinstance(updated_at, str):
            updated_at = dt.fromisoformat(updated_at.replace("Z", "+00:00"))

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_learnings (
                    id, agent_id, category, trigger_context, learning_content,
                    applies_to, success_count, failure_count, last_used_at,
                    source_session_id, source_project_path, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                learning_data["id"],
                learning_data["agent_id"],
                learning_data["category"],
                learning_data.get("trigger_context"),
                learning_data["learning_content"],
                json.dumps(learning_data.get("applies_to", {})),
                learning_data.get("success_count", 1),
                learning_data.get("failure_count", 0),
                last_used_at,
                learning_data.get("source_session_id"),
                learning_data.get("source_project_path"),
                created_at,
                updated_at,
            )
        return {"id": learning_data["id"], "status": "saved"}

    async def query_agent_learnings(
        self,
        agent_id: str,
        category: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query agent learnings with optional category filter."""
        pool = self._ensure_connected()

        if category:
            query = """
                SELECT * FROM agent_learnings
                WHERE agent_id = $1 AND category = $2
                ORDER BY success_count DESC, updated_at DESC
                LIMIT $3
            """
            params = [agent_id, category, limit]
        else:
            query = """
                SELECT * FROM agent_learnings
                WHERE agent_id = $1
                ORDER BY success_count DESC, updated_at DESC
                LIMIT $2
            """
            params = [agent_id, limit]

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._from_record(row) for row in rows]

    async def update_agent_learning_outcome(
        self, learning_id: str, success: bool
    ) -> dict[str, Any]:
        """Increment success or failure count for a learning."""
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            if success:
                row = await conn.fetchrow(
                    """
                    UPDATE agent_learnings
                    SET success_count = success_count + 1,
                        last_used_at = NOW(),
                        updated_at = NOW()
                    WHERE id = $1
                    RETURNING id, success_count, failure_count
                    """,
                    learning_id,
                )
            else:
                row = await conn.fetchrow(
                    """
                    UPDATE agent_learnings
                    SET failure_count = failure_count + 1,
                        last_used_at = NOW(),
                        updated_at = NOW()
                    WHERE id = $1
                    RETURNING id, success_count, failure_count
                    """,
                    learning_id,
                )

            if row:
                return {
                    "id": learning_id,
                    "success_count": row["success_count"],
                    "failure_count": row["failure_count"],
                    "updated": True,
                }
            return {"id": learning_id, "error": "Learning not found"}

    async def save_agent_notebook(self, notebook_data: dict[str, Any]) -> dict[str, Any]:
        """Save an agent notebook."""
        pool = self._ensure_connected()

        # Parse timestamps if strings
        from datetime import datetime as dt

        created_at = notebook_data.get("created_at", self._get_timestamp())
        if isinstance(created_at, str):
            created_at = dt.fromisoformat(created_at.replace("Z", "+00:00"))

        updated_at = notebook_data.get("updated_at", self._get_timestamp())
        if isinstance(updated_at, str):
            updated_at = dt.fromisoformat(updated_at.replace("Z", "+00:00"))

        covers_from = notebook_data.get("covers_from")
        if isinstance(covers_from, str):
            covers_from = dt.fromisoformat(covers_from.replace("Z", "+00:00"))

        covers_to = notebook_data.get("covers_to")
        if isinstance(covers_to, str):
            covers_to = dt.fromisoformat(covers_to.replace("Z", "+00:00"))

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_notebooks (
                    id, agent_id, title, summary_markdown, notebook_type,
                    tags, key_insights, related_sessions, created_at,
                    updated_at, covers_from, covers_to
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                notebook_data["id"],
                notebook_data["agent_id"],
                notebook_data["title"],
                notebook_data["summary_markdown"],
                notebook_data.get("notebook_type", "summary"),
                json.dumps(notebook_data.get("tags", [])),
                json.dumps(notebook_data.get("key_insights", [])),
                json.dumps(notebook_data.get("related_sessions", [])),
                created_at,
                updated_at,
                covers_from,
                covers_to,
            )
        return {"id": notebook_data["id"], "status": "saved"}

    async def query_agent_notebooks(
        self,
        agent_id: str,
        notebook_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query agent notebooks with optional type filter."""
        pool = self._ensure_connected()

        if notebook_type:
            query = """
                SELECT * FROM agent_notebooks
                WHERE agent_id = $1 AND notebook_type = $2
                ORDER BY created_at DESC
                LIMIT $3
            """
            params = [agent_id, notebook_type, limit]
        else:
            query = """
                SELECT * FROM agent_notebooks
                WHERE agent_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """
            params = [agent_id, limit]

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._from_record(row) for row in rows]

    async def search_sessions(
        self, query: str, search_type: str = "fulltext", limit: int = 20
    ) -> list[dict[str, Any]]:
        """
        Search session summaries using PostgreSQL full-text search.

        Args:
            query: Search query string
            search_type: Type of search - "fulltext", "tag", or "file" (only fulltext supported)
            limit: Maximum results to return

        Returns:
            List of dicts with session_id, title, summary, tags, relevance, snippet
        """
        pool = self._ensure_connected()

        async with pool.acquire() as conn:
            # PostgreSQL full-text search using to_tsvector and plainto_tsquery
            # Search in both title and summary_markdown fields
            # JOIN with sessions table to avoid N+1 query pattern
            rows = await conn.fetch(
                """
                SELECT
                    ss.session_id,
                    ss.title,
                    ss.summary_markdown as summary,
                    ss.tags,
                    ss.created_at,
                    s.project_name,
                    s.project_path,
                    s.started_at,
                    ts_rank(
                        to_tsvector('english', coalesce(ss.title, '') || ' ' || coalesce(ss.summary_markdown, '')),
                        plainto_tsquery('english', $1)
                    ) as relevance,
                    ts_headline(
                        'english',
                        coalesce(ss.summary_markdown, ''),
                        plainto_tsquery('english', $1),
                        'MaxWords=50, MinWords=25, MaxFragments=1'
                    ) as snippet
                FROM session_summaries ss
                JOIN sessions s ON ss.session_id = s.id
                WHERE to_tsvector('english', coalesce(ss.title, '') || ' ' || coalesce(ss.summary_markdown, ''))
                      @@ plainto_tsquery('english', $1)
                ORDER BY relevance DESC
                LIMIT $2
                """,
                query,
                limit,
            )

        return [dict(row) for row in rows]
