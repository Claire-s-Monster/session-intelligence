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

    def __init__(
        self,
        source: DatabaseBackend,
        target: DatabaseBackend,
    ) -> None:
        self.source = source
        self.target = target
        self.stats: dict[str, int] = {
            "sessions": 0,
            "decisions": 0,
            "metrics": 0,
            "notes": 0,
            "agent_executions": 0,
            "mcp_sessions": 0,
        }

    async def migrate_all(self, batch_size: int = 100) -> dict[str, Any]:
        """Migrate all data from source to target."""
        start_time = datetime.now()

        logger.info("Starting migration...")
        logger.info(f"Source: {type(self.source).__name__}")
        logger.info(f"Target: {type(self.target).__name__}")

        # Migrate in dependency order
        await self._migrate_sessions(batch_size)
        await self._migrate_decisions(batch_size)
        await self._migrate_metrics(batch_size)
        await self._migrate_notes(batch_size)
        await self._migrate_agent_executions(batch_size)
        await self._migrate_mcp_sessions(batch_size)

        duration = (datetime.now() - start_time).total_seconds()

        result = {
            "status": "success",
            "duration_seconds": duration,
            "records_migrated": self.stats,
            "total_records": sum(self.stats.values()),
        }

        logger.info(f"Migration completed in {duration:.2f}s")
        logger.info(f"Total records migrated: {sum(self.stats.values())}")

        return result

    async def _migrate_sessions(self, batch_size: int) -> None:
        """Migrate sessions table."""
        logger.info("Migrating sessions...")
        sessions = await self.source.query_sessions(limit=10000)

        for session in sessions:
            try:
                await self.target.save_session(session)
                self.stats["sessions"] += 1
            except Exception as e:
                logger.warning(f"Failed to migrate session {session.get('id')}: {e}")

        logger.info(f"  Migrated {self.stats['sessions']} sessions")

    async def _migrate_decisions(self, batch_size: int) -> None:
        """Migrate decisions table."""
        logger.info("Migrating decisions...")

        # Query decisions by common categories
        categories = [
            "architecture",
            "implementation",
            "testing",
            "deployment",
            "refactoring",
            None,  # Uncategorized
        ]

        for category in categories:
            if category:
                decisions = await self.source.query_decisions_by_category(category, limit=10000)
            else:
                # Get all decisions not in known categories via session
                sessions = await self.source.query_sessions(limit=10000)
                decisions = []
                for session in sessions:
                    session_decisions = await self.source.query_decisions_by_session(
                        session["id"], limit=1000
                    )
                    decisions.extend(session_decisions)

            for decision in decisions:
                try:
                    await self.target.save_decision(decision)
                    self.stats["decisions"] += 1
                except Exception as e:
                    logger.warning(f"Failed to migrate decision {decision.get('id')}: {e}")

        logger.info(f"  Migrated {self.stats['decisions']} decisions")

    async def _migrate_metrics(self, batch_size: int) -> None:
        """Migrate metrics table."""
        logger.info("Migrating metrics...")

        # Get metrics via sessions
        sessions = await self.source.query_sessions(limit=10000)
        for session in sessions:
            metrics = await self.source.query_metrics_by_session(session["id"], limit=1000)
            for metric in metrics:
                try:
                    await self.target.save_metrics(metric)
                    self.stats["metrics"] += 1
                except Exception as e:
                    logger.warning(f"Failed to migrate metric: {e}")

        logger.info(f"  Migrated {self.stats['metrics']} metrics")

    async def _migrate_notes(self, batch_size: int) -> None:
        """Migrate notes table."""
        logger.info("Migrating notes...")

        # Get notes for recent dates
        from datetime import timedelta

        for days_ago in range(365):  # Last year
            date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            notes = await self.source.query_notes_by_date(date, limit=1000)

            for note in notes:
                try:
                    await self.target.save_note(note)
                    self.stats["notes"] += 1
                except Exception as e:
                    logger.warning(f"Failed to migrate note: {e}")

        logger.info(f"  Migrated {self.stats['notes']} notes")

    async def _migrate_agent_executions(self, batch_size: int) -> None:
        """Migrate agent_executions table."""
        logger.info("Migrating agent executions...")

        executions = await self.source.query_agent_executions(limit=10000)
        for execution in executions:
            try:
                await self.target.save_agent_execution(execution)
                self.stats["agent_executions"] += 1
            except Exception as e:
                logger.warning(f"Failed to migrate agent execution {execution.get('id')}: {e}")

        logger.info(f"  Migrated {self.stats['agent_executions']} agent executions")

    async def _migrate_mcp_sessions(self, batch_size: int) -> None:
        """Migrate mcp_sessions table."""
        logger.info("Migrating MCP sessions...")

        # MCP sessions don't have a query_all method, so we skip or implement differently
        # For now, we'll log that this needs manual handling if needed
        logger.info("  MCP sessions migration: manual review recommended")


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
            "source_backend": config.backend,
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
