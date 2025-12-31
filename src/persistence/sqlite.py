"""
SQLite database backend for session persistence.

Optimal for:
- Development environments
- Single-user deployments
- Local testing

Uses WAL mode for better concurrent read access.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from .base import DEFAULT_SQLITE_PATH, BaseDatabaseBackend

logger = logging.getLogger(__name__)


class SQLiteBackend(BaseDatabaseBackend):
    """SQLite database backend with async support for session persistence."""

    SCHEMA = """
    -- Schema version tracking
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    );

    -- Sessions table
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        project_path TEXT NOT NULL,
        project_name TEXT,
        mode TEXT DEFAULT 'local',
        status TEXT DEFAULT 'active',
        metadata TEXT,
        performance_metrics TEXT,
        health_status TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_path);
    CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
    CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);

    -- Decisions table with category index for filtering
    CREATE TABLE IF NOT EXISTS decisions (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        category TEXT,
        description TEXT NOT NULL,
        rationale TEXT,
        context TEXT,
        impact_level TEXT DEFAULT 'medium',
        artifacts TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_decisions_session ON decisions(session_id);
    CREATE INDEX IF NOT EXISTS idx_decisions_category ON decisions(category);
    CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);

    -- Metrics table for time-series data
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        branch TEXT,
        timestamp TEXT NOT NULL,
        coverage REAL,
        complexity REAL,
        test_count INTEGER,
        agents_executed INTEGER,
        execution_time_ms INTEGER,
        custom_metrics TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics(session_id);
    CREATE INDEX IF NOT EXISTS idx_metrics_branch ON metrics(branch);
    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);

    -- Notes table for session notes
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        date TEXT NOT NULL,
        content TEXT NOT NULL,
        tags TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_notes_session ON notes(session_id);
    CREATE INDEX IF NOT EXISTS idx_notes_date ON notes(date);

    -- Agent executions table
    CREATE TABLE IF NOT EXISTS agent_executions (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        agent_name TEXT NOT NULL,
        agent_type TEXT,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        status TEXT DEFAULT 'running',
        execution_steps TEXT,
        performance TEXT,
        errors TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_agent_executions_session ON agent_executions(session_id);
    CREATE INDEX IF NOT EXISTS idx_agent_executions_agent ON agent_executions(agent_name);

    -- MCP session mapping table
    CREATE TABLE IF NOT EXISTS mcp_sessions (
        mcp_session_id TEXT PRIMARY KEY,
        engine_session_id TEXT,
        created_at TEXT NOT NULL,
        last_activity TEXT NOT NULL,
        client_info TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_mcp_sessions_engine ON mcp_sessions(engine_session_id);

    -- Session summaries table for narrative session documentation
    CREATE TABLE IF NOT EXISTS session_summaries (
        session_id TEXT PRIMARY KEY,
        title TEXT,
        summary_markdown TEXT,
        key_changes TEXT,
        tags TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_summaries_created ON session_summaries(created_at);

    -- Project-specific learnings (trial/error history)
    CREATE TABLE IF NOT EXISTS project_learnings (
        id TEXT PRIMARY KEY,
        project_path TEXT NOT NULL,
        category TEXT NOT NULL,           -- 'error_fix', 'pattern', 'preference', 'workflow'
        trigger_context TEXT,             -- What situation triggers this knowledge
        learning_content TEXT NOT NULL,   -- The actual knowledge/solution
        source_session_id TEXT,           -- Which session created this
        success_count INTEGER DEFAULT 1,
        failure_count INTEGER DEFAULT 0,
        last_used TEXT,
        promoted_to_universal BOOLEAN DEFAULT FALSE,
        created_at TEXT NOT NULL,
        FOREIGN KEY (source_session_id) REFERENCES sessions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_learnings_project ON project_learnings(project_path);
    CREATE INDEX IF NOT EXISTS idx_learnings_category ON project_learnings(category);
    CREATE INDEX IF NOT EXISTS idx_learnings_promoted ON project_learnings(promoted_to_universal);

    -- Error→Solution quick lookup
    CREATE TABLE IF NOT EXISTS error_solutions (
        id TEXT PRIMARY KEY,
        error_pattern TEXT NOT NULL,      -- Key phrases or regex
        error_hash TEXT,                  -- Hash for quick matching
        error_category TEXT,              -- 'compile', 'runtime', 'config', 'dependency'
        solution_steps TEXT NOT NULL,     -- JSON array of solution steps
        context_requirements TEXT,        -- When this solution applies (JSON)
        success_rate REAL DEFAULT 1.0,
        usage_count INTEGER DEFAULT 1,
        project_path TEXT,                -- NULL = universal, set = project-specific
        source_session_id TEXT,
        created_at TEXT NOT NULL,
        last_used TEXT,
        FOREIGN KEY (source_session_id) REFERENCES sessions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_errors_hash ON error_solutions(error_hash);
    CREATE INDEX IF NOT EXISTS idx_errors_category ON error_solutions(error_category);
    CREATE INDEX IF NOT EXISTS idx_errors_project ON error_solutions(project_path);
    """

    # FTS5 schema must be created separately (cannot use IF NOT EXISTS with virtual tables)
    FTS_SCHEMA = """
    CREATE VIRTUAL TABLE IF NOT EXISTS session_search USING fts5(
        session_id,
        title,
        summary,
        decisions,
        notes,
        tags,
        content='',
        tokenize='porter unicode61'
    );
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file. Defaults to ~/.claude/session-intelligence/sessions.db.
                    Use ":memory:" for testing.
        """
        super().__init__()

        if db_path == ":memory:":
            self.db_path = db_path
        else:
            self.db_path = Path(db_path) if db_path else DEFAULT_SQLITE_PATH
            if isinstance(self.db_path, Path):
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize database connection and apply schema."""
        db_path = str(self.db_path) if isinstance(self.db_path, Path) else self.db_path
        self._connection = await aiosqlite.connect(db_path)
        self._connection.row_factory = aiosqlite.Row

        # Enable WAL mode for better concurrent access
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA foreign_keys=ON")
        await self._connection.execute("PRAGMA synchronous=NORMAL")

        # Apply schema
        await self._connection.executescript(self.SCHEMA)
        await self._connection.commit()

        # Create FTS5 virtual table (handle gracefully if already exists)
        try:
            await self._connection.executescript(self.FTS_SCHEMA)
            await self._connection.commit()
        except Exception as e:
            # FTS5 table likely already exists
            logger.debug(f"FTS5 table creation: {e}")

        # Track schema version
        await self._connection.execute(
            "INSERT OR IGNORE INTO schema_version VALUES (?, ?)",
            (self.SCHEMA_VERSION, datetime.now().isoformat()),
        )
        await self._connection.commit()

        self._is_connected = True
        logger.info(f"SQLite database initialized: {self.db_path}")

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            # Checkpoint WAL before closing
            try:
                await self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as e:
                logger.warning(f"WAL checkpoint failed: {e}")

            await self._connection.close()
            self._connection = None
            self._is_connected = False
            logger.info("SQLite connection closed")

    def _ensure_connected(self) -> None:
        """Raise error if not connected."""
        if not self._connection:
            raise RuntimeError("Database not initialized")

    # Session operations

    async def save_session(self, session_data: dict[str, Any]) -> None:
        """Save or update a session."""
        self._ensure_connected()

        await self._connection.execute(
            """
            INSERT OR REPLACE INTO sessions
            (id, started_at, ended_at, project_path, project_name, mode, status,
             metadata, performance_metrics, health_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_data["id"],
                session_data.get("started") or session_data.get("started_at"),
                session_data.get("completed") or session_data.get("ended_at"),
                session_data.get("project_path", ""),
                session_data.get("project_name"),
                session_data.get("mode", "local"),
                session_data.get("status", "active"),
                self._serialize_json(session_data.get("metadata", {})),
                self._serialize_json(session_data.get("performance_metrics", {})),
                self._serialize_json(session_data.get("health_status", {})),
            ),
        )
        await self._connection.commit()

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get a session by ID."""
        self._ensure_connected()

        cursor = await self._connection.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if row:
            return self._normalize_session_data(dict(row))
        return None

    async def query_sessions(
        self,
        limit: int = 50,
        project_path: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query sessions with optional filters."""
        self._ensure_connected()

        query = "SELECT * FROM sessions WHERE 1=1"
        params: list[Any] = []

        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [self._normalize_session_data(dict(row)) for row in rows]

    async def get_active_session_for_project(
        self, project_path: str
    ) -> dict[str, Any] | None:
        """Get the most recent active session for a project path."""
        self._ensure_connected()

        cursor = await self._connection.execute(
            """
            SELECT * FROM sessions
            WHERE project_path = ? AND status = 'active'
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (project_path,),
        )
        row = await cursor.fetchone()
        if row:
            return self._normalize_session_data(dict(row))
        return None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID."""
        self._ensure_connected()

        # Delete related records first (cascade)
        await self._connection.execute(
            "DELETE FROM decisions WHERE session_id = ?", (session_id,)
        )
        await self._connection.execute(
            "DELETE FROM metrics WHERE session_id = ?", (session_id,)
        )
        await self._connection.execute(
            "DELETE FROM notes WHERE session_id = ?", (session_id,)
        )
        await self._connection.execute(
            "DELETE FROM agent_executions WHERE session_id = ?", (session_id,)
        )

        cursor = await self._connection.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
        await self._connection.commit()
        return cursor.rowcount > 0

    # Decision operations

    async def save_decision(self, decision_data: dict[str, Any]) -> None:
        """Save a decision."""
        self._ensure_connected()

        await self._connection.execute(
            """
            INSERT INTO decisions
            (id, session_id, timestamp, category, description, rationale,
             context, impact_level, artifacts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                decision_data.get("decision_id") or decision_data.get("id"),
                decision_data["session_id"],
                decision_data.get("timestamp", self._get_timestamp()),
                decision_data.get("category"),
                decision_data.get("description") or decision_data.get("decision", ""),
                decision_data.get("rationale"),
                self._serialize_json(decision_data.get("context", {})),
                decision_data.get("impact_level", "medium"),
                self._serialize_json(decision_data.get("artifacts", [])),
            ),
        )
        await self._connection.commit()

    async def query_decisions_by_category(
        self, category: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query decisions by category across sessions."""
        self._ensure_connected()

        cursor = await self._connection.execute(
            """
            SELECT d.*, s.project_name
            FROM decisions d
            JOIN sessions s ON d.session_id = s.id
            WHERE d.category = ?
            ORDER BY d.timestamp DESC
            LIMIT ?
        """,
            (category, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def query_decisions_by_session(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query decisions for a specific session."""
        self._ensure_connected()

        cursor = await self._connection.execute(
            """
            SELECT * FROM decisions
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # Metrics operations

    async def save_metrics(self, metrics_data: dict[str, Any]) -> None:
        """Save metrics snapshot."""
        self._ensure_connected()

        await self._connection.execute(
            """
            INSERT INTO metrics
            (session_id, branch, timestamp, coverage, complexity, test_count,
             agents_executed, execution_time_ms, custom_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metrics_data["session_id"],
                metrics_data.get("branch"),
                metrics_data.get("timestamp", self._get_timestamp()),
                metrics_data.get("coverage"),
                metrics_data.get("complexity"),
                metrics_data.get("test_count"),
                metrics_data.get("agents_executed"),
                metrics_data.get("execution_time_ms"),
                self._serialize_json(metrics_data.get("custom_metrics", {})),
            ),
        )
        await self._connection.commit()

    async def query_metrics_by_branch(
        self, branch: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query metrics by branch across sessions."""
        self._ensure_connected()

        cursor = await self._connection.execute(
            """
            SELECT * FROM metrics
            WHERE branch = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (branch, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def query_metrics_by_session(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query metrics for a specific session."""
        self._ensure_connected()

        cursor = await self._connection.execute(
            """
            SELECT * FROM metrics
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # Notes operations

    async def save_note(self, note_data: dict[str, Any]) -> None:
        """Save a session note."""
        self._ensure_connected()

        await self._connection.execute(
            """
            INSERT INTO notes (session_id, date, content, tags)
            VALUES (?, ?, ?, ?)
        """,
            (
                note_data["session_id"],
                note_data.get("date", datetime.now().strftime("%Y-%m-%d")),
                note_data["content"],
                self._serialize_json(note_data.get("tags", [])),
            ),
        )
        await self._connection.commit()

    async def query_notes_by_date(
        self, date: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Query notes by date across sessions."""
        self._ensure_connected()

        cursor = await self._connection.execute(
            """
            SELECT n.*, s.project_name
            FROM notes n
            JOIN sessions s ON n.session_id = s.id
            WHERE n.date = ?
            ORDER BY n.id DESC
            LIMIT ?
        """,
            (date, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # Agent execution operations

    async def save_agent_execution(self, execution_data: dict[str, Any]) -> None:
        """Save agent execution record."""
        self._ensure_connected()

        await self._connection.execute(
            """
            INSERT OR REPLACE INTO agent_executions
            (id, session_id, agent_name, agent_type, started_at, completed_at,
             status, execution_steps, performance, errors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                execution_data["id"],
                execution_data["session_id"],
                execution_data["agent_name"],
                execution_data.get("agent_type"),
                execution_data.get("started_at", self._get_timestamp()),
                execution_data.get("completed_at"),
                execution_data.get("status", "running"),
                self._serialize_json(execution_data.get("execution_steps", [])),
                self._serialize_json(execution_data.get("performance", {})),
                self._serialize_json(execution_data.get("errors", [])),
            ),
        )
        await self._connection.commit()

    async def query_agent_executions(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query agent executions with optional filters."""
        self._ensure_connected()

        query = "SELECT * FROM agent_executions WHERE 1=1"
        params: list[Any] = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # MCP session operations

    async def save_mcp_session(self, mcp_session_data: dict[str, Any]) -> None:
        """Save MCP session mapping."""
        self._ensure_connected()

        await self._connection.execute(
            """
            INSERT OR REPLACE INTO mcp_sessions
            (mcp_session_id, engine_session_id, created_at, last_activity, client_info)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                mcp_session_data["mcp_session_id"],
                mcp_session_data.get("engine_session_id"),
                mcp_session_data.get("created_at", self._get_timestamp()),
                mcp_session_data.get("last_activity", self._get_timestamp()),
                self._serialize_json(mcp_session_data.get("client_info", {})),
            ),
        )
        await self._connection.commit()

    async def get_mcp_session(self, mcp_session_id: str) -> dict[str, Any] | None:
        """Get MCP session by ID."""
        self._ensure_connected()

        cursor = await self._connection.execute(
            "SELECT * FROM mcp_sessions WHERE mcp_session_id = ?",
            (mcp_session_id,),
        )
        row = await cursor.fetchone()
        if row:
            return dict(row)
        return None

    async def update_mcp_session_activity(self, mcp_session_id: str) -> None:
        """Update last activity timestamp for MCP session."""
        self._ensure_connected()

        await self._connection.execute(
            "UPDATE mcp_sessions SET last_activity = ? WHERE mcp_session_id = ?",
            (self._get_timestamp(), mcp_session_id),
        )
        await self._connection.commit()

    async def link_mcp_to_engine_session(
        self, mcp_session_id: str, engine_session_id: str
    ) -> None:
        """Link MCP session to engine session."""
        self._ensure_connected()

        await self._connection.execute(
            "UPDATE mcp_sessions SET engine_session_id = ? WHERE mcp_session_id = ?",
            (engine_session_id, mcp_session_id),
        )
        await self._connection.commit()

    # Session summary operations

    async def save_session_summary(self, summary_data: dict[str, Any]) -> None:
        """Save or update a session summary."""
        self._ensure_connected()

        await self._connection.execute(
            """
            INSERT OR REPLACE INTO session_summaries
            (session_id, title, summary_markdown, key_changes, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                summary_data["session_id"],
                summary_data.get("title"),
                summary_data.get("summary_markdown"),
                self._serialize_json(summary_data.get("key_changes", [])),
                self._serialize_json(summary_data.get("tags", [])),
                summary_data.get("created_at", self._get_timestamp()),
            ),
        )
        await self._connection.commit()

        # Update FTS index
        await self._update_search_index(summary_data["session_id"])

    async def get_session_summary(self, session_id: str) -> dict[str, Any] | None:
        """Get session summary by session ID."""
        self._ensure_connected()

        cursor = await self._connection.execute(
            "SELECT * FROM session_summaries WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if row:
            result = dict(row)
            result["key_changes"] = self._deserialize_json(result.get("key_changes"))
            result["tags"] = self._deserialize_json(result.get("tags"))
            return result
        return None

    async def query_summaries_by_tag(
        self, tag: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Query session summaries that contain a specific tag."""
        self._ensure_connected()

        # SQLite JSON search using json_each
        cursor = await self._connection.execute(
            """
            SELECT ss.*, s.project_name, s.project_path
            FROM session_summaries ss
            JOIN sessions s ON ss.session_id = s.id
            WHERE EXISTS (
                SELECT 1 FROM json_each(ss.tags) WHERE value = ?
            )
            ORDER BY ss.created_at DESC
            LIMIT ?
        """,
            (tag, limit),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result["key_changes"] = self._deserialize_json(result.get("key_changes"))
            result["tags"] = self._deserialize_json(result.get("tags"))
            results.append(result)
        return results

    async def query_recent_summaries(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get most recent session summaries."""
        self._ensure_connected()

        cursor = await self._connection.execute(
            """
            SELECT ss.*, s.project_name, s.project_path
            FROM session_summaries ss
            JOIN sessions s ON ss.session_id = s.id
            ORDER BY ss.created_at DESC
            LIMIT ?
        """,
            (limit,),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result["key_changes"] = self._deserialize_json(result.get("key_changes"))
            result["tags"] = self._deserialize_json(result.get("tags"))
            results.append(result)
        return results

    # Full-text search operations

    async def _update_search_index(self, session_id: str) -> None:
        """Update FTS index for a session."""
        self._ensure_connected()

        # Gather all searchable content for this session
        session = await self.get_session(session_id)
        summary = await self.get_session_summary(session_id)
        decisions = await self.query_decisions_by_session(session_id)

        # Build searchable content
        title = summary.get("title", "") if summary else ""
        summary_text = summary.get("summary_markdown", "") if summary else ""
        decisions_text = " ".join(
            [d.get("description", "") + " " + (d.get("rationale") or "") for d in decisions]
        )

        # Get notes for this session
        cursor = await self._connection.execute(
            "SELECT content FROM notes WHERE session_id = ?", (session_id,)
        )
        notes_rows = await cursor.fetchall()
        notes_text = " ".join([row[0] for row in notes_rows])

        tags_text = " ".join(summary.get("tags", [])) if summary else ""

        # Delete existing entry and insert new one
        await self._connection.execute(
            "DELETE FROM session_search WHERE session_id = ?", (session_id,)
        )
        await self._connection.execute(
            """
            INSERT INTO session_search (session_id, title, summary, decisions, notes, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (session_id, title, summary_text, decisions_text, notes_text, tags_text),
        )
        await self._connection.commit()

    async def search_sessions(
        self, query: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Full-text search across sessions."""
        self._ensure_connected()

        # Use FTS5 MATCH syntax for full-text search
        cursor = await self._connection.execute(
            """
            SELECT
                ss.session_id,
                ss.title,
                ss.summary,
                ss.tags,
                bm25(session_search) as relevance,
                snippet(session_search, 2, '<mark>', '</mark>', '...', 32) as snippet
            FROM session_search ss
            WHERE session_search MATCH ?
            ORDER BY relevance
            LIMIT ?
        """,
            (query, limit),
        )
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            result = dict(row)
            # Enrich with session data
            session = await self.get_session(result["session_id"])
            if session:
                result["project_name"] = session.get("project_name")
                result["project_path"] = session.get("project_path")
                result["started_at"] = session.get("started_at")
            results.append(result)

        return results

    async def search_by_file_change(
        self, file_pattern: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Search sessions by file changes."""
        self._ensure_connected()

        # Search in key_changes JSON array
        cursor = await self._connection.execute(
            """
            SELECT ss.*, s.project_name, s.project_path
            FROM session_summaries ss
            JOIN sessions s ON ss.session_id = s.id
            WHERE EXISTS (
                SELECT 1 FROM json_each(ss.key_changes)
                WHERE value LIKE ?
            )
            ORDER BY ss.created_at DESC
            LIMIT ?
        """,
            (f"%{file_pattern}%", limit),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result["key_changes"] = self._deserialize_json(result.get("key_changes"))
            result["tags"] = self._deserialize_json(result.get("tags"))
            results.append(result)
        return results

    # Project Learnings operations

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
        self._ensure_connected()

        now = datetime.now(UTC).isoformat()
        await self._connection.execute(
            """
            INSERT INTO project_learnings (
                id, project_path, category, trigger_context, learning_content,
                source_session_id, success_count, failure_count, last_used,
                promoted_to_universal, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 1, 0, ?, FALSE, ?)
            ON CONFLICT(id) DO UPDATE SET
                learning_content = excluded.learning_content,
                trigger_context = excluded.trigger_context,
                last_used = excluded.last_used
        """,
            (
                learning_id,
                project_path,
                category,
                trigger_context,
                learning_content,
                source_session_id,
                now,
                now,
            ),
        )
        await self._connection.commit()
        return {"id": learning_id, "status": "saved"}

    async def query_project_learnings(
        self,
        project_path: str,
        category: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query learnings for a project."""
        self._ensure_connected()

        if category:
            cursor = await self._connection.execute(
                """
                SELECT * FROM project_learnings
                WHERE project_path = ? AND category = ?
                ORDER BY success_count DESC, last_used DESC
                LIMIT ?
            """,
                (project_path, category, limit),
            )
        else:
            cursor = await self._connection.execute(
                """
                SELECT * FROM project_learnings
                WHERE project_path = ?
                ORDER BY success_count DESC, last_used DESC
                LIMIT ?
            """,
                (project_path, limit),
            )

        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def update_learning_usage(
        self, learning_id: str, success: bool
    ) -> dict[str, Any]:
        """Update success/failure count for a learning."""
        self._ensure_connected()

        now = datetime.now(UTC).isoformat()
        if success:
            await self._connection.execute(
                """
                UPDATE project_learnings
                SET success_count = success_count + 1, last_used = ?
                WHERE id = ?
            """,
                (now, learning_id),
            )
        else:
            await self._connection.execute(
                """
                UPDATE project_learnings
                SET failure_count = failure_count + 1, last_used = ?
                WHERE id = ?
            """,
                (now, learning_id),
            )
        await self._connection.commit()
        return {"id": learning_id, "updated": True, "success": success}

    # Error Solutions operations

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
        self._ensure_connected()
        import hashlib

        now = datetime.now(UTC).isoformat()
        error_hash = hashlib.sha256(error_pattern.encode()).hexdigest()[:16]

        await self._connection.execute(
            """
            INSERT INTO error_solutions (
                id, error_pattern, error_hash, error_category, solution_steps,
                context_requirements, success_rate, usage_count, project_path,
                source_session_id, created_at, last_used
            ) VALUES (?, ?, ?, ?, ?, ?, 1.0, 1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                solution_steps = excluded.solution_steps,
                context_requirements = excluded.context_requirements,
                last_used = excluded.last_used
        """,
            (
                solution_id,
                error_pattern,
                error_hash,
                error_category,
                json.dumps(solution_steps),
                json.dumps(context_requirements) if context_requirements else None,
                project_path,
                source_session_id,
                now,
                now,
            ),
        )
        await self._connection.commit()
        return {"id": solution_id, "error_hash": error_hash, "status": "saved"}

    async def find_error_solutions(
        self,
        error_text: str,
        project_path: str | None = None,
        include_universal: bool = True,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find solutions matching an error pattern."""
        self._ensure_connected()

        # Build query based on scope
        if project_path and include_universal:
            cursor = await self._connection.execute(
                """
                SELECT * FROM error_solutions
                WHERE (project_path = ? OR project_path IS NULL)
                AND (error_pattern LIKE ? OR ? LIKE '%' || error_pattern || '%')
                ORDER BY
                    CASE WHEN project_path = ? THEN 0 ELSE 1 END,
                    success_rate DESC,
                    usage_count DESC
                LIMIT ?
            """,
                (project_path, f"%{error_text[:100]}%", error_text, project_path, limit),
            )
        elif project_path:
            cursor = await self._connection.execute(
                """
                SELECT * FROM error_solutions
                WHERE project_path = ?
                AND (error_pattern LIKE ? OR ? LIKE '%' || error_pattern || '%')
                ORDER BY success_rate DESC, usage_count DESC
                LIMIT ?
            """,
                (project_path, f"%{error_text[:100]}%", error_text, limit),
            )
        else:
            cursor = await self._connection.execute(
                """
                SELECT * FROM error_solutions
                WHERE project_path IS NULL
                AND (error_pattern LIKE ? OR ? LIKE '%' || error_pattern || '%')
                ORDER BY success_rate DESC, usage_count DESC
                LIMIT ?
            """,
                (f"%{error_text[:100]}%", error_text, limit),
            )

        rows = await cursor.fetchall()
        results = []
        for row in rows:
            result = dict(row)
            result["solution_steps"] = self._deserialize_json(result.get("solution_steps"))
            result["context_requirements"] = self._deserialize_json(
                result.get("context_requirements")
            )
            results.append(result)
        return results

    async def update_solution_outcome(
        self, solution_id: str, success: bool
    ) -> dict[str, Any]:
        """Update success rate for a solution."""
        self._ensure_connected()

        now = datetime.now(UTC).isoformat()

        # Get current stats
        cursor = await self._connection.execute(
            "SELECT usage_count, success_rate FROM error_solutions WHERE id = ?",
            (solution_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return {"id": solution_id, "error": "Solution not found"}

        usage_count = row["usage_count"]
        current_rate = row["success_rate"]

        # Calculate new success rate as weighted average
        new_usage = usage_count + 1
        if success:
            new_rate = (current_rate * usage_count + 1.0) / new_usage
        else:
            new_rate = (current_rate * usage_count) / new_usage

        await self._connection.execute(
            """
            UPDATE error_solutions
            SET usage_count = ?, success_rate = ?, last_used = ?
            WHERE id = ?
        """,
            (new_usage, new_rate, now, solution_id),
        )
        await self._connection.commit()
        return {
            "id": solution_id,
            "usage_count": new_usage,
            "success_rate": new_rate,
            "updated": True,
        }

    # Maintenance operations

    async def vacuum(self) -> None:
        """Optimize database storage."""
        self._ensure_connected()
        await self._connection.execute("VACUUM")
        await self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        logger.info("SQLite database vacuumed")

    async def get_statistics(self) -> dict[str, Any]:
        """Get database statistics for monitoring."""
        self._ensure_connected()

        stats: dict[str, Any] = {"backend": "sqlite", "path": str(self.db_path)}

        # Get table counts
        tables = ["sessions", "decisions", "metrics", "notes", "agent_executions", "mcp_sessions", "session_summaries"]
        for table in tables:
            cursor = await self._connection.execute(f"SELECT COUNT(*) FROM {table}")
            row = await cursor.fetchone()
            stats[f"{table}_count"] = row[0] if row else 0

        # Get database size
        if isinstance(self.db_path, Path) and self.db_path.exists():
            stats["size_bytes"] = self.db_path.stat().st_size
            wal_path = Path(str(self.db_path) + "-wal")
            if wal_path.exists():
                stats["wal_size_bytes"] = wal_path.stat().st_size

        return stats


# Backwards compatibility alias
Database = SQLiteBackend
