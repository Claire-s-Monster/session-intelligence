"""One-time reap of pre-existing stale 'running' agent_executions (issue #70).

Background:
    Issue #39/#40 fixed the common case of AgentExecution status never
    transitioning out of RUNNING by having the SubagentStop hook
    (phase == "agent_stop") flip it to SUCCESS/ERROR. That fix is incomplete:
    if the stop event never arrives -- agent killed, session ends mid-flight,
    hook fails or times out, server restart between start and stop -- the row
    stays 'running' forever. As of 2026-08-15, 847 executions started AFTER
    the #40 fix are still 'running', alongside 13,688 'success'. Because
    get_agent_stats counted every row toward the success_rate denominator
    regardless of status, these rows silently understated success_rate for
    every agent -- the exact metric #39 existed to fix.

    The application-level fix is two-pronged:
    - reconcile-on-finalize: session_engine._finalize_session now flips any
      still-RUNNING execution (and its still-running execution_steps) to
      ABANDONED before the session is persisted.
    - reap_stale_executions(), called once at HTTP transport startup (mirrors
      reap_abandoned_sessions from issue #69), catches executions whose
      session was never explicitly finalized either (crash, restart, etc).
    Both only take effect going forward and on process restart/finalize. This
    script is a ONE-TIME cleanup for rows that were already stale before the
    fix shipped. It intentionally duplicates the app's UPDATE rather than
    calling reap_stale_executions() directly, so it can run standalone
    (dry-run) without importing the full server stack.

Known limitation (same caveat as issue #69, see persistence/base.py
get_execution_max_age_hours()):
    started_at is the only staleness signal available today. A genuinely
    long-running execution that has stayed open past --older-than-hours will
    be incorrectly flagged as abandoned.

Safety:
    - Default mode is DRY-RUN (report only, no writes).
    - --apply is required to write. There is no additional confirmation flag
      by design (this script is invoked deliberately, unlike the interactive
      backfill scripts) -- but per project policy this script is checked in
      and is NOT to be executed against the production database as part of
      landing this fix. Running it (even in dry-run) against production is a
      separate, explicit operational decision.
    - Rows are flipped to 'abandoned', NEVER 'error' -- the distinction is
      the point: "never reported a stop event" is not "failed".

Usage (report only, safe to run anytime):
    PYTHONPATH=src python scripts/migrations/reap_stale_executions.py

Usage (apply, default 24h threshold):
    PYTHONPATH=src python scripts/migrations/reap_stale_executions.py --apply

Usage (custom threshold):
    PYTHONPATH=src python scripts/migrations/reap_stale_executions.py \
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

from persistence.base import DEFAULT_EXECUTION_MAX_AGE_HOURS
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
    """Report (dry-run) or apply the reap of stale 'running' agent_executions."""
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
        total_running = await conn.fetchval(
            "SELECT COUNT(*) FROM agent_executions WHERE status = 'running'"
        )
        logger.info(f"Total rows currently status='running': {total_running}")

        before_rows = await conn.fetch(
            """
            SELECT to_char(started_at, 'YYYY-MM') AS month, COUNT(*) AS row_count
            FROM agent_executions
            WHERE status = 'running' AND started_at < $1
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
            "Stale 'running' agent_executions BY MONTH (started_at < cutoff):",
        )
        logger.info(
            f"{would_change} of {total_running} running rows are older than "
            f"the {older_than_hours}h threshold and would move to 'abandoned'."
        )

        if not apply_changes:
            logger.info(
                "DRY RUN complete — no changes written. Re-run with --apply to write."
            )
            return

        result = await conn.execute(
            """
            UPDATE agent_executions
            SET status = 'abandoned', completed_at = COALESCE(completed_at, NOW())
            WHERE status = 'running' AND started_at < $1
            """,
            cutoff,
        )
        try:
            updated = int(result.split()[-1])
        except (IndexError, ValueError):
            updated = 0

        remaining_running = await conn.fetchval(
            "SELECT COUNT(*) FROM agent_executions WHERE status = 'running'"
        )
        total_abandoned = await conn.fetchval(
            "SELECT COUNT(*) FROM agent_executions WHERE status = 'abandoned'"
        )
        logger.info(f"APPLY complete: {updated} rows moved to 'abandoned'.")
        logger.info(f"Remaining status='running' rows: {remaining_running}")
        logger.info(f"Total status='abandoned' rows: {total_abandoned}")
    finally:
        await conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reap pre-existing stale 'running' agent_executions to 'abandoned' "
            "(one-time cleanup for issue #70). Dry-run by default."
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
        default=DEFAULT_EXECUTION_MAX_AGE_HOURS,
        help=f"Staleness threshold in hours (default: {DEFAULT_EXECUTION_MAX_AGE_HOURS})",
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
