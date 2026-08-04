"""Backfill sessions.project_name for rows stranded under the '_unbound_' sentinel.

Re-derives project_name from each row's stored project_path using the shared
core.project_naming.derive_project_name() helper (git remote origin basename ->
git toplevel basename -> path basename -> UNBOUND). As of 2026-08-04, 5,797 of
7,286 rows in the sessions table are stranded under '_unbound_'.

Note: paths that no longer exist on disk fall back to basename derivation
(derive_project_name never raises); this is expected and acceptable.

Idempotent — only ever touches rows still matching
    WHERE project_name = '_unbound_' AND project_path IS NOT NULL AND project_path <> ''
so a rerun after a successful --apply naturally has nothing left to do for the
rows it already fixed.

Excluding untrustworthy paths:
    Some stranded rows got their project_path from the server process's own
    cwd rather than the caller's project (see session_engine.py's
    `_create_session`, which defaults project_path to the server cwd). A
    dry-run against production on 2026-08-04 found 5,767 of 5,835 candidate
    rows all sharing ONE such path:
    /home/memento/ClaudeCode/Servers/session-intelligence/development
    (the session-intelligence server's own working directory). Backfilling
    those would incorrectly stamp "session-intelligence" onto thousands of
    sessions that actually belong to many different projects. The remaining
    ~68 rows have genuine, distinct project paths and ARE safely recoverable.

    Use --exclude-path (repeatable) to exclude specific paths from the
    backfill entirely. Matching is an EXACT string match on project_path —
    NOT a prefix or glob match — so each untrustworthy path must be listed
    verbatim. Excluded rows are removed from consideration before --limit is
    applied, and are reported separately in the summary so the totals
    reconcile.

    PYTHONPATH=src python scripts/migrations/backfill_unbound_session_project_names.py \
        --exclude-path /home/memento/ClaudeCode/Servers/session-intelligence/development

Safety:
    - Default mode is --dry-run (report only, no writes).
    - --apply requires --yes, or it refuses and exits non-zero.
    - Rows whose re-derived name is still '_unbound_' are never written; they
      are counted and reported as skipped.
    - Updates run inside a transaction, batched (default 500 rows/batch), with
      progress logged per batch.

Usage (report only, safe to run anytime):
    PYTHONPATH=src python scripts/migrations/backfill_unbound_session_project_names.py

Usage (staged rollout, first 500 rows only):
    PYTHONPATH=src python scripts/migrations/backfill_unbound_session_project_names.py \
        --apply --yes --limit 500

Usage (full apply):
    PYTHONPATH=src python scripts/migrations/backfill_unbound_session_project_names.py \
        --apply --yes

Environment variables:
    POSTGRES_DSN  PostgreSQL DSN (required unless --dsn is passed explicitly)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import urllib.parse

from core.project_naming import UNBOUND, derive_project_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def get_db_name(dsn: str) -> str:
    """Extract the database name from a PostgreSQL DSN for operator confirmation."""
    parsed = urllib.parse.urlparse(dsn)
    return parsed.path.lstrip("/") or "(unspecified)"


def print_summary(
    derived_counts: dict[str, int],
    examined: int,
    would_change: int,
    still_unbound: int,
    skipped_empty_path: int,
    skipped_excluded_path: int,
) -> None:
    """Print the derived_name -> row_count table plus overall totals."""
    print()
    print(f"{'derived_name':<50} {'row_count':>10}")
    print("-" * 61)
    for name, count in sorted(derived_counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{name:<50} {count:>10}")
    print("-" * 61)
    print(f"Rows examined:                              {examined}")
    print(f"Rows that would change / did change:        {would_change}")
    print(f"Rows skipped (derivation still {UNBOUND!r}):    {still_unbound}")
    print(f"Rows skipped (empty/NULL project_path):     {skipped_empty_path}")
    print(f"Rows skipped (--exclude-path):               {skipped_excluded_path}")
    print()


async def _execute_batch(conn, batch: list[tuple[str, str]]) -> int:
    """Apply one batch of (project_path, derived_name) updates inside a transaction."""
    updated = 0
    async with conn.transaction():
        for project_path, derived_name in batch:
            result = await conn.execute(
                """
                UPDATE sessions
                SET project_name = $1
                WHERE project_name = $2
                  AND project_path = $3
                """,
                derived_name,
                UNBOUND,
                project_path,
            )
            updated += int(result.split()[-1]) if result else 0
    return updated


async def apply_updates(
    conn, path_updates: list[tuple[str, str, int]], batch_size: int = BATCH_SIZE
) -> int:
    """Apply updates grouped by distinct project_path, batched by ~batch_size rows.

    path_updates: list of (project_path, derived_name, row_count) for paths whose
    re-derived name is not UNBOUND.
    """
    total_updated = 0
    total_paths = len(path_updates)
    processed_paths = 0
    batch: list[tuple[str, str]] = []
    batch_rows = 0

    for project_path, derived_name, row_count in path_updates:
        batch.append((project_path, derived_name))
        batch_rows += row_count
        if batch_rows >= batch_size:
            total_updated += await _execute_batch(conn, batch)
            processed_paths += len(batch)
            logger.info(
                f"Progress: {processed_paths}/{total_paths} paths, "
                f"{total_updated} rows updated so far"
            )
            batch = []
            batch_rows = 0

    if batch:
        total_updated += await _execute_batch(conn, batch)
        processed_paths += len(batch)
        logger.info(
            f"Progress: {processed_paths}/{total_paths} paths, "
            f"{total_updated} rows updated so far"
        )

    return total_updated


async def backfill(
    dsn: str,
    apply_changes: bool,
    limit: int | None,
    exclude_paths: list[str] | None = None,
) -> None:
    """Run the backfill (or dry-run report) against a PostgreSQL database.

    exclude_paths: project_path values excluded from consideration via an
    EXACT string match (not a prefix/glob match). Use for paths known to be
    untrustworthy, e.g. a server process's own cwd rather than the caller's
    project.
    """
    try:
        import asyncpg
    except ImportError:
        logger.error("asyncpg is required for PostgreSQL. Install with: pip install asyncpg")
        sys.exit(1)

    exclude_paths = list(exclude_paths) if exclude_paths else []

    db_name = get_db_name(dsn)
    logger.info(f"Target database: {db_name}")

    conn = await asyncpg.connect(dsn)
    try:
        total_sessions = await conn.fetchval("SELECT COUNT(*) FROM sessions")
        logger.info(f"Total rows in sessions table: {total_sessions}")

        skipped_empty_path = await conn.fetchval(
            """
            SELECT COUNT(*) FROM sessions
            WHERE project_name = $1
              AND (project_path IS NULL OR project_path = '')
            """,
            UNBOUND,
        )

        skipped_excluded_path = await conn.fetchval(
            """
            SELECT COUNT(*) FROM sessions
            WHERE project_name = $1
              AND project_path IS NOT NULL
              AND project_path <> ''
              AND project_path = ANY($2::text[])
            """,
            UNBOUND,
            exclude_paths,
        )

        rows = await conn.fetch(
            """
            SELECT project_path, COUNT(*) AS row_count
            FROM (
                SELECT project_path FROM sessions
                WHERE project_name = $1
                  AND project_path IS NOT NULL
                  AND project_path <> ''
                  AND project_path <> ALL($3::text[])
                ORDER BY id
                LIMIT $2::bigint
            ) sub
            GROUP BY project_path
            ORDER BY row_count DESC
            """,
            UNBOUND,
            limit,
            exclude_paths,
        )

        examined = sum(row["row_count"] for row in rows)

        # Derivation is per-distinct-path, not per-row: far fewer paths than rows.
        to_update: list[tuple[str, str, int]] = []
        still_unbound_count = 0
        derived_counts: dict[str, int] = {}
        for row in rows:
            project_path = row["project_path"]
            row_count = row["row_count"]
            derived_name = derive_project_name(project_path)
            derived_counts[derived_name] = derived_counts.get(derived_name, 0) + row_count
            if derived_name == UNBOUND:
                still_unbound_count += row_count
            else:
                to_update.append((project_path, derived_name, row_count))

        would_change = sum(row_count for _, _, row_count in to_update)

        print_summary(
            derived_counts,
            examined,
            would_change,
            still_unbound_count,
            skipped_empty_path,
            skipped_excluded_path,
        )

        if apply_changes:
            updated = await apply_updates(conn, to_update)
            logger.info(f"APPLY complete: {updated} rows updated.")
        else:
            logger.info(
                "DRY RUN complete — no changes written. Re-run with --apply --yes to write."
            )
    finally:
        await conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill sessions.project_name for rows stranded under the "
            f"{UNBOUND!r} sentinel by re-deriving the name from project_path."
        )
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("POSTGRES_DSN"),
        help="PostgreSQL DSN (default: POSTGRES_DSN env var)",
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
        help="Perform the UPDATE. Requires --yes.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm --apply. Required alongside --apply.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of stranded rows processed (for a staged rollout)",
    )
    parser.add_argument(
        "--exclude-path",
        action="append",
        default=None,
        metavar="PATH",
        dest="exclude_path",
        help=(
            "Exact project_path to exclude from the backfill (repeatable). "
            "Use for paths that record the server's own cwd rather than the "
            "caller's project. Matching is an EXACT string match, not a "
            "prefix or glob match."
        ),
    )
    args = parser.parse_args()

    if not args.dsn:
        parser.error("--dsn is required (or set the POSTGRES_DSN environment variable)")
    if args.apply and not args.yes:
        parser.error("--apply requires --yes to confirm you intend to write to the database")

    return args


def main() -> None:
    args = parse_args()
    asyncio.run(
        backfill(
            args.dsn,
            apply_changes=args.apply,
            limit=args.limit,
            exclude_paths=args.exclude_path,
        )
    )


if __name__ == "__main__":
    main()
