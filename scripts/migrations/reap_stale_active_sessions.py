"""One-time reap of pre-existing stale 'active' sessions (issue #69).

Background:
    Nothing ever transitioned an abandoned session out of status='active'
    before this fix -- sessions only became 'completed' via an explicit
    finalize call. As of 2026-08-15, 630 sessions were status='active', 615
    of them more than a day old (dating back to Dec 2025), collectively
    owning 10,708 agent_executions (73% of that table).

    The application-level fix (reap_abandoned_sessions(), called once at
    HTTP transport startup, plus the read-path staleness guard in
    get_active_session_for_project / find_recent_session_by_project) only
    takes effect going forward and on process restart. This script is a
    ONE-TIME cleanup for rows that were already stale before the fix
    shipped. It intentionally duplicates the app's UPDATE rather than
    calling reap_abandoned_sessions() directly, so it can run standalone
    (dry-run) without importing the full server stack.

Known limitation (deferred, see issue #69 and persistence/base.py
get_session_max_age_hours()):
    started_at is the only staleness signal available today. A genuinely
    long-lived session that has stayed open past --older-than-hours will be
    incorrectly flagged as abandoned. A last_seen_at heartbeat column would
    fix this but requires a schema migration; that is DEFERRED.

Safety:
    - Default mode is DRY-RUN (report only, no writes).
    - --apply is required to write. There is no additional confirmation flag
      by design (this script is invoked deliberately, unlike the interactive
      backfill scripts) -- but per project policy this script is checked in
      and is NOT to be executed against the production database as part of
      landing this fix. Running it (even in dry-run) against production is a
      separate, explicit operational decision.
    - Rows are flipped to 'abandoned', NEVER 'completed' -- the distinction
      is the point: these sessions were never explicitly finalized.

Usage (report only, safe to run anytime):
    PYTHONPATH=src python scripts/migrations/reap_stale_active_sessions.py

Usage (apply, default 24h threshold):
    PYTHONPATH=src python scripts/migrations/reap_stale_active_sessions.py --apply

Usage (custom threshold):
    PYTHONPATH=src python scripts/migrations/reap_stale_active_sessions.py \
        --older-than-hours 72

Environment variables:
    SESSION_DB_DSN  PostgreSQL DSN (same variable the application reads via
                     persistence.config.DatabaseConfig; falls back to
                     DEFAULT_POSTGRES_DSN like the app does if unset)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import urllib.parse
from datetime import UTC, datetime, timedelta

from persistence.base import DEFAULT_SESSION_MAX_AGE_HOURS
from persistence.config import DatabaseConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_db_name(dsn: str) -> str:
    """Extract the database name from a PostgreSQL DSN for operator confirmation."""
    parsed = urllib.parse.urlparse(dsn)
    return parsed.path.lstrip("/") or "(unspecified)"


def print_month_summary(month_counts: dict[str, int], total: int, label: str) -> None:
    """Print a month -> row_count table, matching the issue's reporting format."""
    print()
    print(f"{label}")
    print(f"{'month':<10} {'row_count':>10}")
    print("-" * 21)
    for month in sorted(month_counts):
        print(f"{month:<10} {month_counts[month]:>10}")
    print("-" * 21)
    print(f"{'total':<10} {total:>10}")
    print()


async def reap(dsn: str, apply_changes: bool, older_than_hours: int) -> None:
    """Report (dry-run) or apply the reap of stale 'active' sessions."""
    try:
        import asyncpg
    except ImportError:
        logger.error("asyncpg is required for PostgreSQL. Install with: pip install asyncpg")
        sys.exit(1)

    db_name = get_db_name(dsn)
    logger.info(f"Target database: {db_name}")
    logger.info(f"Staleness threshold: {older_than_hours}h")

    cutoff = datetime.now(UTC) - timedelta(hours=older_than_hours)

    conn = await asyncpg.connect(dsn)
    try:
        total_active = await conn.fetchval(
            "SELECT COUNT(*) FROM sessions WHERE status = 'active'"
        )
        logger.info(f"Total rows currently status='active': {total_active}")

        before_rows = await conn.fetch(
            """
            SELECT to_char(started_at, 'YYYY-MM') AS month, COUNT(*) AS row_count
            FROM sessions
            WHERE status = 'active' AND started_at < $1
            GROUP BY month
            ORDER BY month
            """,
            cutoff,
        )
        before_counts = {row["month"]: row["row_count"] for row in before_rows}
        would_change = sum(before_counts.values())

        print_month_summary(
            before_counts,
            would_change,
            "Stale 'active' sessions BY MONTH (started_at < cutoff):",
        )
        logger.info(
            f"{would_change} of {total_active} active rows are older than "
            f"the {older_than_hours}h threshold and would move to 'abandoned'."
        )

        if not apply_changes:
            logger.info(
                "DRY RUN complete — no changes written. Re-run with --apply to write."
            )
            return

        result = await conn.execute(
            """
            UPDATE sessions
            SET status = 'abandoned'
            WHERE status = 'active' AND started_at < $1
            """,
            cutoff,
        )
        try:
            updated = int(result.split()[-1])
        except (IndexError, ValueError):
            updated = 0

        remaining_active = await conn.fetchval(
            "SELECT COUNT(*) FROM sessions WHERE status = 'active'"
        )
        total_abandoned = await conn.fetchval(
            "SELECT COUNT(*) FROM sessions WHERE status = 'abandoned'"
        )
        logger.info(f"APPLY complete: {updated} rows moved to 'abandoned'.")
        logger.info(f"Remaining status='active' rows: {remaining_active}")
        logger.info(f"Total status='abandoned' rows: {total_abandoned}")
    finally:
        await conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reap pre-existing stale 'active' sessions to 'abandoned' "
            "(one-time cleanup for issue #69). Dry-run by default."
        )
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help=(
            "PostgreSQL DSN. Defaults to the same config the app uses "
            "(SESSION_DB_DSN env var, or DEFAULT_POSTGRES_DSN)."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only, never write (default mode)",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Perform the UPDATE. Not run as part of shipping this fix; a separate, "
        "explicit operational decision.",
    )
    parser.add_argument(
        "--older-than-hours",
        type=int,
        default=DEFAULT_SESSION_MAX_AGE_HOURS,
        help=f"Staleness threshold in hours (default: {DEFAULT_SESSION_MAX_AGE_HOURS})",
    )
    args = parser.parse_args()

    if not args.dsn:
        args.dsn = (
            DatabaseConfig.load().postgresql_dsn
            or os.environ.get("SESSION_DB_DSN")
        )
    if not args.dsn:
        from persistence.base import DEFAULT_POSTGRES_DSN

        args.dsn = DEFAULT_POSTGRES_DSN

    return args


def main() -> None:
    args = parse_args()
    asyncio.run(
        reap(
            args.dsn,
            apply_changes=args.apply,
            older_than_hours=args.older_than_hours,
        )
    )


if __name__ == "__main__":
    main()
