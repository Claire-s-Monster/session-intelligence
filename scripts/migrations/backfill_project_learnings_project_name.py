"""Backfill project_learnings.project_name from source_session_id → sessions.project_name.

Idempotent — only updates rows where project_name IS NULL.
Run after the fix/issue-23-project-name-on-learnings PR is deployed to
retroactively make old rows recallable via project_name.

Usage (PostgreSQL):
    PYTHONPATH=src python scripts/migrations/backfill_project_learnings_project_name.py

Usage (SQLite):
    PYTHONPATH=src python scripts/migrations/backfill_project_learnings_project_name.py \
        --backend sqlite [--db-path /path/to/sessions.db]

Environment variables:
    DATABASE_URL  PostgreSQL DSN (default: postgresql://localhost/session_intelligence)
    SQLITE_PATH   SQLite DB path (default: ~/.claude/session-intelligence/sessions.db)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def backfill_postgresql(dsn: str) -> None:
    """Run the backfill on a PostgreSQL database."""
    try:
        import asyncpg
    except ImportError:
        logger.error("asyncpg is required for PostgreSQL. Install with: pip install asyncpg")
        sys.exit(1)

    logger.info(f"Connecting to PostgreSQL: {dsn}")
    conn = await asyncpg.connect(dsn)
    try:
        # Count rows to be updated
        null_count = await conn.fetchval(
            "SELECT COUNT(*) FROM project_learnings WHERE project_name IS NULL"
        )
        logger.info(f"Rows with project_name IS NULL: {null_count}")

        # Backfill: update project_name from source_session_id → sessions.project_name
        result = await conn.execute(
            """
            UPDATE project_learnings pl
            SET project_name = s.project_name
            FROM sessions s
            WHERE pl.source_session_id = s.id
              AND pl.project_name IS NULL
              AND s.project_name IS NOT NULL
            """
        )
        updated = int(result.split()[-1]) if result else 0
        logger.info(f"Updated {updated} rows with project_name from sessions")

        # Count still-null rows (no valid source_session_id)
        still_null = await conn.fetchval(
            "SELECT COUNT(*) FROM project_learnings WHERE project_name IS NULL"
        )
        logger.info(
            f"Rows still NULL after backfill (no valid source_session_id): {still_null}"
        )
        logger.info("Backfill complete (PostgreSQL).")
    finally:
        await conn.close()


async def backfill_sqlite(db_path: str) -> None:
    """Run the backfill on a SQLite database."""
    try:
        import aiosqlite
    except ImportError:
        logger.error("aiosqlite is required for SQLite. Install with: pip install aiosqlite")
        sys.exit(1)

    logger.info(f"Opening SQLite database: {db_path}")
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row

        # Count rows to be updated
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM project_learnings WHERE project_name IS NULL"
        )
        row = await cursor.fetchone()
        null_count = row[0] if row else 0
        logger.info(f"Rows with project_name IS NULL: {null_count}")

        # Backfill
        cursor = await conn.execute(
            """
            UPDATE project_learnings
            SET project_name = (
                SELECT s.project_name
                FROM sessions s
                WHERE s.id = project_learnings.source_session_id
                  AND s.project_name IS NOT NULL
                LIMIT 1
            )
            WHERE project_name IS NULL
              AND source_session_id IS NOT NULL
            """
        )
        updated = cursor.rowcount
        await conn.commit()
        logger.info(f"Updated {updated} rows with project_name from sessions")

        cursor = await conn.execute(
            "SELECT COUNT(*) FROM project_learnings WHERE project_name IS NULL"
        )
        row = await cursor.fetchone()
        still_null = row[0] if row else 0
        logger.info(
            f"Rows still NULL after backfill (no valid source_session_id): {still_null}"
        )
        logger.info("Backfill complete (SQLite).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill project_learnings.project_name from sessions table."
    )
    parser.add_argument(
        "--backend",
        choices=["postgresql", "sqlite"],
        default="postgresql",
        help="Database backend to target (default: postgresql)",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite database file path (overrides SQLITE_PATH env var)",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL DSN (overrides DATABASE_URL env var)",
    )
    args = parser.parse_args()

    if args.backend == "postgresql":
        dsn = args.dsn or os.environ.get(
            "DATABASE_URL", "postgresql://localhost/session_intelligence"
        )
        asyncio.run(backfill_postgresql(dsn))
    else:
        from pathlib import Path

        default_path = Path.home() / ".claude" / "session-intelligence" / "sessions.db"
        db_path = args.db_path or os.environ.get("SQLITE_PATH", str(default_path))
        asyncio.run(backfill_sqlite(db_path))


if __name__ == "__main__":
    main()
