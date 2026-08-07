"""
Data migration utilities for session-intelligence database.

Supports:
- SQLite to PostgreSQL migration
- Local project database to global ~/.claude migration
- Data export/import for backups

Usage:
    # Migrate local SQLite to global SQLite
    pixi run python -m persistence.migration local-to-global

    # Migrate SQLite to PostgreSQL
    pixi run python -m persistence.migration sqlite-to-postgres --dsn "postgresql://..."

    # Export to JSON backup
    pixi run python -m persistence.migration export --output backup.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import DEFAULT_SQLITE_PATH, DatabaseBackend
from .config import DatabaseConfig, create_database

logger = logging.getLogger(__name__)


class MigrationManager:
    """Handles database migrations between backends and locations."""

    ENTITIES: tuple[str, ...] = (
        "sessions",
        "decisions",
        "metrics",
        "notes",
        "agent_executions",
        "mcp_sessions",
    )

    def __init__(
        self,
        source: DatabaseBackend,
        target: DatabaseBackend,
    ) -> None:
        self.source = source
        self.target = target
        self.stats: dict[str, int] = dict.fromkeys(self.ENTITIES, 0)
        # Per-entity counts of records that raised while saving to target.
        self.failed: dict[str, int] = dict.fromkeys(self.ENTITIES, 0)
        # Human-readable messages: both save failures and detected truncation.
        self.warnings: list[str] = []
        # Set when any query hit its limit and we cannot prove nothing was
        # left behind (no offset support on that reader).
        self.truncated = False

    def _record_failure(self, entity: str, message: str) -> None:
        """Record a record that raised while being saved to the target."""
        self.failed[entity] += 1
        self.warnings.append(message)
        logger.warning(message)

    def _record_truncation(self, entity: str, message: str) -> None:
        """Record that a query may have left records behind (hit its cap)."""
        self.truncated = True
        self.warnings.append(message)
        logger.warning(message)

    async def migrate_all(
        self, batch_size: int = 100, scan_limit: int = 10000
    ) -> dict[str, Any]:
        """Migrate all data from source to target.

        Two distinct concepts, not to be confused:

        - `batch_size` is the page size for the genuinely paginated readers
          (`query_notes`, `query_mcp_sessions`), which loop by `offset` until
          exhausted. It bounds memory per page, NOT the total number of rows
          migrated — pagination continues until the source is exhausted
          regardless of its value.
        - `scan_limit` is a hard cap for readers that expose no `offset`
          (`query_sessions`, `query_decisions_by_category`,
          `query_decisions_by_session`, `query_metrics_by_session`,
          `query_agent_executions`). These readers cannot be paginated with
          the current read interface, so hitting `scan_limit` is real
          truncation and is reported via `status="partial"` and a warning.
          Properly paginating them requires adding `offset` support to the
          read interface in `base.py`, which is deliberately out of scope
          for this change.
        """
        start_time = datetime.now()

        logger.info("Starting migration...")
        logger.info(f"Source: {type(self.source).__name__}")
        logger.info(f"Target: {type(self.target).__name__}")

        # Migrate in dependency order
        await self._migrate_sessions(batch_size, scan_limit)
        await self._migrate_decisions(batch_size, scan_limit)
        await self._migrate_metrics(batch_size, scan_limit)
        await self._migrate_notes(batch_size, scan_limit)
        await self._migrate_agent_executions(batch_size, scan_limit)
        await self._migrate_mcp_sessions(batch_size, scan_limit)

        duration = (datetime.now() - start_time).total_seconds()

        is_clean = sum(self.failed.values()) == 0 and not self.truncated
        result: dict[str, Any] = {
            "status": "success" if is_clean else "partial",
            "duration_seconds": duration,
            "records_migrated": self.stats,
            "total_records": sum(self.stats.values()),
        }
        if not is_clean:
            result["warnings"] = list(self.warnings)
            result["failed"] = {k: v for k, v in self.failed.items() if v}

        logger.info(f"Migration completed in {duration:.2f}s")
        logger.info(f"Total records migrated: {sum(self.stats.values())}")

        return result

    async def _migrate_sessions(self, batch_size: int, scan_limit: int) -> None:
        """Migrate sessions table."""
        logger.info("Migrating sessions...")
        sessions = await self.source.query_sessions(limit=scan_limit)
        if len(sessions) >= scan_limit:
            self._record_truncation(
                "sessions",
                f"sessions: query_sessions returned {len(sessions)} rows at "
                f"scan_limit={scan_limit}; query_sessions has no offset support, "
                "so the source may contain more sessions than were migrated",
            )

        for session in sessions:
            try:
                await self.target.save_session(session)
                self.stats["sessions"] += 1
            except Exception as e:
                self._record_failure(
                    "sessions", f"Failed to migrate session {session.get('id')}: {e}"
                )

        logger.info(f"  Migrated {self.stats['sessions']} sessions")

    async def _migrate_decisions(self, batch_size: int, scan_limit: int) -> None:
        """Migrate decisions table.

        Decisions are collected into a dict keyed by id BEFORE saving, so a
        decision found by both the category loop and the session fallback
        loop (uncategorized decisions) is saved and counted exactly once.
        """
        logger.info("Migrating decisions...")

        categories = [
            "architecture",
            "implementation",
            "testing",
            "deployment",
            "refactoring",
        ]

        collected: dict[Any, dict[str, Any]] = {}

        for category in categories:
            decisions = await self.source.query_decisions_by_category(category, limit=scan_limit)
            if len(decisions) >= scan_limit:
                self._record_truncation(
                    "decisions",
                    f"decisions: category '{category}' returned {len(decisions)} rows "
                    f"at scan_limit={scan_limit}; source may contain more",
                )
            for decision in decisions:
                collected[decision.get("id")] = decision

        # Uncategorized decisions: pull all decisions per session and keep
        # only the ones not already collected above.
        sessions = await self.source.query_sessions(limit=scan_limit)
        if len(sessions) >= scan_limit:
            self._record_truncation(
                "decisions",
                f"decisions: query_sessions returned {len(sessions)} rows at "
                f"scan_limit={scan_limit} while scanning for uncategorized decisions; "
                "source may contain more sessions than were scanned",
            )
        for session in sessions:
            session_decisions = await self.source.query_decisions_by_session(
                session["id"], limit=scan_limit
            )
            if len(session_decisions) >= scan_limit:
                self._record_truncation(
                    "decisions",
                    f"decisions: session {session['id']} returned "
                    f"{len(session_decisions)} decisions at scan_limit={scan_limit}; "
                    "source may contain more",
                )
            for decision in session_decisions:
                collected.setdefault(decision.get("id"), decision)

        for decision in collected.values():
            try:
                await self.target.save_decision(decision)
                self.stats["decisions"] += 1
            except Exception as e:
                self._record_failure(
                    "decisions", f"Failed to migrate decision {decision.get('id')}: {e}"
                )

        logger.info(f"  Migrated {self.stats['decisions']} decisions")

    async def _migrate_metrics(self, batch_size: int, scan_limit: int) -> None:
        """Migrate metrics table."""
        logger.info("Migrating metrics...")

        sessions = await self.source.query_sessions(limit=scan_limit)
        if len(sessions) >= scan_limit:
            self._record_truncation(
                "metrics",
                f"metrics: query_sessions returned {len(sessions)} rows at "
                f"scan_limit={scan_limit}; source may contain more sessions than were scanned",
            )
        for session in sessions:
            metrics = await self.source.query_metrics_by_session(session["id"], limit=scan_limit)
            if len(metrics) >= scan_limit:
                self._record_truncation(
                    "metrics",
                    f"metrics: session {session['id']} returned {len(metrics)} metrics "
                    f"at scan_limit={scan_limit}; source may contain more",
                )
            for metric in metrics:
                try:
                    await self.target.save_metrics(metric)
                    self.stats["metrics"] += 1
                except Exception as e:
                    self._record_failure("metrics", f"Failed to migrate metric: {e}")

        logger.info(f"  Migrated {self.stats['metrics']} metrics")

    async def _migrate_notes(self, batch_size: int, scan_limit: int) -> None:
        """Migrate notes table.

        Paginated full scan via query_notes (not joined to sessions, so
        orphaned notes are included), superseding the old 365-day walk which
        silently dropped anything older than a year, capped each day at
        1000, and never surfaced the cap.
        """
        logger.info("Migrating notes...")

        offset = 0
        while True:
            notes = await self.source.query_notes(limit=batch_size, offset=offset)
            if not notes:
                break

            for note in notes:
                try:
                    await self.target.save_note(note)
                    self.stats["notes"] += 1
                except Exception as e:
                    self._record_failure(
                        "notes", f"Failed to migrate note {note.get('id')}: {e}"
                    )

            if len(notes) < batch_size:
                break
            offset += batch_size

        # notes.id is SERIAL on PostgreSQL; explicit-id inserts (preserving
        # source ids for idempotency) don't advance the sequence. Resync it
        # if the target supports it. SQLite's AUTOINCREMENT bookkeeping is
        # updated automatically on explicit-id inserts, so no equivalent call
        # exists there — this is not an oversight.
        resync = getattr(self.target, "resync_notes_sequence", None)
        if resync is not None:
            await resync()

        logger.info(f"  Migrated {self.stats['notes']} notes")

    async def _migrate_agent_executions(self, batch_size: int, scan_limit: int) -> None:
        """Migrate agent_executions table."""
        logger.info("Migrating agent executions...")

        executions = await self.source.query_agent_executions(limit=scan_limit)
        if len(executions) >= scan_limit:
            self._record_truncation(
                "agent_executions",
                f"agent_executions: query_agent_executions returned "
                f"{len(executions)} rows at scan_limit={scan_limit}; source may contain more",
            )
        for execution in executions:
            try:
                await self.target.save_agent_execution(execution)
                self.stats["agent_executions"] += 1
            except Exception as e:
                self._record_failure(
                    "agent_executions",
                    f"Failed to migrate agent execution {execution.get('id')}: {e}",
                )

        logger.info(f"  Migrated {self.stats['agent_executions']} agent executions")

    async def _migrate_mcp_sessions(self, batch_size: int, scan_limit: int) -> None:
        """Migrate mcp_sessions table via a paginated full scan."""
        logger.info("Migrating MCP sessions...")

        offset = 0
        while True:
            mcp_sessions = await self.source.query_mcp_sessions(limit=batch_size, offset=offset)
            if not mcp_sessions:
                break

            for mcp_session in mcp_sessions:
                try:
                    await self.target.save_mcp_session(mcp_session)
                    self.stats["mcp_sessions"] += 1
                except Exception as e:
                    self._record_failure(
                        "mcp_sessions",
                        f"Failed to migrate MCP session "
                        f"{mcp_session.get('mcp_session_id')}: {e}",
                    )

            if len(mcp_sessions) < batch_size:
                break
            offset += batch_size

        logger.info(f"  Migrated {self.stats['mcp_sessions']} MCP sessions")


async def migrate_local_to_global(
    local_path: Path | None = None,
) -> dict[str, Any]:
    """Migrate from local project database to global ~/.claude database."""

    # Find local database
    if local_path is None:
        # Look for common local paths
        candidates = [
            Path.cwd() / ".claude" / "session-intelligence" / "sessions.db",
            Path.cwd() / ".claude" / "sessions.db",
        ]
        for candidate in candidates:
            if candidate.exists():
                local_path = candidate
                break

    if local_path is None or not local_path.exists():
        return {
            "status": "error",
            "message": "No local database found. Specify path with --source",
        }

    # Create backends
    from .sqlite import SQLiteBackend

    source = SQLiteBackend(db_path=str(local_path))
    target = SQLiteBackend(db_path=str(DEFAULT_SQLITE_PATH))

    await source.initialize()
    await target.initialize()

    try:
        manager = MigrationManager(source, target)
        result = await manager.migrate_all()
        result["source"] = str(local_path)
        result["target"] = str(DEFAULT_SQLITE_PATH)
        return result
    finally:
        await source.close()
        await target.close()


async def migrate_sqlite_to_postgres(
    sqlite_path: Path | None = None,
    postgres_dsn: str | None = None,
) -> dict[str, Any]:
    """Migrate from SQLite to PostgreSQL."""

    from .postgresql import PostgreSQLBackend
    from .sqlite import SQLiteBackend

    # Use defaults if not specified
    sqlite_path = sqlite_path or DEFAULT_SQLITE_PATH
    postgres_dsn = postgres_dsn or "postgresql://localhost/session_intelligence"

    if not Path(sqlite_path).exists():
        return {
            "status": "error",
            "message": f"SQLite database not found: {sqlite_path}",
        }

    source = SQLiteBackend(db_path=str(sqlite_path))
    target = PostgreSQLBackend(dsn=postgres_dsn)

    await source.initialize()
    await target.initialize()

    try:
        manager = MigrationManager(source, target)
        result = await manager.migrate_all()
        result["source"] = str(sqlite_path)
        result["target"] = postgres_dsn.split("@")[-1] if "@" in postgres_dsn else postgres_dsn
        return result
    finally:
        await source.close()
        await target.close()


async def export_to_json(
    source_config: DatabaseConfig | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Export all data to JSON file for backup."""

    config = source_config or DatabaseConfig.load()
    db = create_database(config=config)
    await db.initialize()

    try:
        data = {
            "exported_at": datetime.now().isoformat(),
            "source_backend": "postgresql",
            "sessions": await db.query_sessions(limit=100000),
            "statistics": await db.get_statistics(),
        }

        # Add related data for each session
        for session in data["sessions"]:
            session["decisions"] = await db.query_decisions_by_session(session["id"], limit=1000)
            session["metrics"] = await db.query_metrics_by_session(session["id"], limit=1000)
            session["agent_executions"] = await db.query_agent_executions(
                session_id=session["id"], limit=1000
            )

        output_path = output_path or Path(
            f"session-export-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        )

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return {
            "status": "success",
            "output_path": str(output_path),
            "sessions_exported": len(data["sessions"]),
            "file_size_bytes": output_path.stat().st_size,
        }

    finally:
        await db.close()


def main() -> None:
    """CLI entry point for migration commands."""
    parser = argparse.ArgumentParser(
        description="Session Intelligence Database Migration Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Migration command")

    # local-to-global command
    local_parser = subparsers.add_parser(
        "local-to-global",
        help="Migrate local project database to global ~/.claude location",
    )
    local_parser.add_argument(
        "--source",
        type=Path,
        help="Source SQLite database path",
    )

    # sqlite-to-postgres command
    pg_parser = subparsers.add_parser(
        "sqlite-to-postgres",
        help="Migrate from SQLite to PostgreSQL",
    )
    pg_parser.add_argument(
        "--source",
        type=Path,
        help="Source SQLite database path",
    )
    pg_parser.add_argument(
        "--dsn",
        required=True,
        help="PostgreSQL connection string",
    )

    # export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export database to JSON backup",
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file path",
    )

    # Parse and execute
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.command == "local-to-global":
        result = asyncio.run(migrate_local_to_global(args.source))
    elif args.command == "sqlite-to-postgres":
        result = asyncio.run(migrate_sqlite_to_postgres(args.source, args.dsn))
    elif args.command == "export":
        result = asyncio.run(export_to_json(output_path=args.output))
    else:
        parser.print_help()
        return

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
