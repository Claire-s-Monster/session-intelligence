"""Purge Claude Code progress-summarizer rows from agent_executions (issue #137).

The ~32s progress summarizer emits a typeless SubagentStop with no start, so the
hook created one-step, near-zero-duration 'success' rows named by a hex subagent
id. #136 stops new ones; this removes the existing ones.

Bucket A (default): agent_type in (NULL, '', 'unknown'), exactly one execution
step, status 'success', completed < 2s after start, hex agent_name.
Bucket B (--include-abandoned-typeless): agent_type = '', one step, status
'abandoned', completed_at set, hex agent_name (closed later by the abandon sweep).

Matching is ONE shared Python predicate (`classify`) over rows from a loose,
identical SQL pre-filter, so PostgreSQL and SQLite behave identically.

Dry run by default. --apply deletes in a single transaction and recomputes
sessions.performance_metrics.agents_executed for sessions that lost rows.

STOP THE SERVER FIRST: run `systemctl --user stop session-intelligence` before
--apply and start it again afterwards. A running server holds cached sessions with
their in-memory agent lists, and its persist sweep can write those back, which
re-inserts deleted rows and overwrites the recomputed counter.

Usage:
    PYTHONPATH=src python scripts/purge_progress_summarizer_rows.py \
        [--dsn DSN|PATH] [--apply] [--include-abandoned-typeless]

DSN: postgresql://..., sqlite:///path, or an existing .db file path. Defaults to
the app config (SESSION_DB_DSN / config.json). No fallback: unresolved = error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from persistence.config import DatabaseConfig

HEX_NAME = re.compile(r"[0-9a-f]{8,}")
MAX_DURATION = timedelta(seconds=2)
CHUNK = 500
CANDIDATES = """
SELECT e.id, e.session_id, e.agent_name, e.agent_type, e.status, e.started_at,
       e.completed_at, e.execution_steps, s.project_name
FROM agent_executions e LEFT JOIN sessions s ON s.id = e.session_id
WHERE e.status IN ('success', 'abandoned') AND e.completed_at IS NOT NULL
  AND (e.agent_type IS NULL OR e.agent_type IN ('', 'unknown'))
"""


def _steps_len(value: Any) -> int | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return len(value) if isinstance(value, list) else None


def _ts(value: Any) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def classify(row: dict[str, Any]) -> str | None:
    """Return 'A', 'B' or None for an agent_executions row."""
    if not HEX_NAME.fullmatch(row["agent_name"] or "") or _steps_len(row["execution_steps"]) != 1:
        return None
    if row["completed_at"] is None:
        return None
    agent_type, status = row["agent_type"], row["status"]
    if status == "success" and agent_type in (None, "", "unknown"):
        if _ts(row["completed_at"]) - _ts(row["started_at"]) < MAX_DURATION:
            return "A"
    if status == "abandoned" and agent_type == "":
        return "B"
    return None


class SqliteConn:
    json_cast = ""

    def __init__(self, path: str) -> None:
        self.path = path
        self.conn: Any = None

    @asynccontextmanager
    async def transaction(self):
        import aiosqlite

        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        try:
            yield self
            await self.conn.commit()
        except BaseException:
            await self.conn.rollback()
            raise
        finally:
            await self.conn.close()

    async def fetch(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        async with self.conn.execute(sql, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def execute(self, sql: str, params: tuple = ()) -> None:
        await self.conn.execute(sql, params)


class PgConn:
    json_cast = "::jsonb"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.conn: Any = None

    @asynccontextmanager
    async def transaction(self):
        import asyncpg

        self.conn = await asyncpg.connect(self.dsn)
        try:
            async with self.conn.transaction():
                yield self
        finally:
            await self.conn.close()

    @staticmethod
    def _sql(sql: str) -> str:
        parts = sql.split("?")
        return "".join(p + (f"${i + 1}" if i < len(parts) - 1 else "") for i, p in enumerate(parts))

    async def fetch(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in await self.conn.fetch(self._sql(sql), *params)]

    async def execute(self, sql: str, params: tuple = ()) -> None:
        await self.conn.execute(self._sql(sql), *params)


def resolve_target(dsn: str | None) -> SqliteConn | PgConn:
    dsn = dsn or DatabaseConfig.load().postgresql_dsn
    if not dsn:
        raise SystemExit("error: no database resolved (pass --dsn or set SESSION_DB_DSN)")
    if dsn.startswith(("postgresql://", "postgres://")):
        return PgConn(dsn)
    path = dsn.removeprefix("sqlite:///")
    if not Path(path).is_file():
        raise SystemExit(f"error: cannot resolve database from {dsn!r}")
    return SqliteConn(path)


def _in(ids: list[str]) -> str:
    return ",".join("?" * len(ids))


async def _apply(db: SqliteConn | PgConn, matched: list[dict[str, Any]]) -> tuple[int, int]:
    ids = [r["id"] for r in matched]
    sessions = sorted({r["session_id"] for r in matched})
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i : i + CHUNK]
        await db.execute(f"DELETE FROM agent_executions WHERE id IN ({_in(chunk)})", tuple(chunk))
    for sid in sessions:
        count = (
            await db.fetch(
                "SELECT COUNT(*) AS n FROM agent_executions WHERE session_id = ?", (sid,)
            )
        )[0]["n"]
        got = await db.fetch("SELECT performance_metrics AS pm FROM sessions WHERE id = ?", (sid,))
        if not got:
            continue
        pm = got[0]["pm"]
        metrics = json.loads(pm) if isinstance(pm, str) else (pm or {})
        metrics["agents_executed"] = count
        await db.execute(
            f"UPDATE sessions SET performance_metrics = ?{db.json_cast} WHERE id = ?",
            (json.dumps(metrics), sid),
        )
    return len(ids), len(sessions)


def _report(total: int, buckets: dict[str, list[dict[str, Any]]]) -> None:
    print(f"total agent_executions: {total}")
    for name, rows in buckets.items():
        starts = sorted(str(_ts(r["started_at"])) for r in rows)
        span = f"{starts[0]} .. {starts[-1]}" if starts else "-"
        print(f"bucket {name}: {len(rows)} rows, started_at {span}")
    matched = [r for rows in buckets.values() for r in rows]
    print("matched by project (top 20):")
    for project, n in Counter(r["project_name"] or "(none)" for r in matched).most_common(20):
        print(f"  {n:>8}  {project}")
    print("samples:")
    for r in matched[:5]:
        print(
            "  "
            + " | ".join(
                str(r[k])
                for k in ("id", "session_id", "agent_name", "agent_type", "status", "started_at")
            )
        )


async def run(target: SqliteConn | PgConn, apply: bool, include_b: bool) -> None:
    async with target.transaction() as db:
        total = (await db.fetch("SELECT COUNT(*) AS n FROM agent_executions"))[0]["n"]
        rows = await db.fetch(CANDIDATES)
        buckets: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
        for row in rows:
            if (kind := classify(row)) and (kind == "A" or include_b):
                buckets[kind].append(row)
        if not include_b:
            del buckets["B"]
        _report(total, buckets)
        matched = [r for rs in buckets.values() for r in rs]
        if not apply:
            print("DRY RUN: nothing deleted. Re-run with --apply.")
            return
        print("NOTE: --apply assumes the session-intelligence service is stopped.")
        deleted, updated = await _apply(db, matched)
        print(f"APPLIED: deleted {deleted} rows, updated {updated} sessions")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dsn", default=None, help="postgresql:// DSN, sqlite:/// URL or path")
    parser.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    parser.add_argument("--include-abandoned-typeless", action="store_true", dest="include_b")
    args = parser.parse_args(argv)
    try:
        target = resolve_target(args.dsn)
        asyncio.run(run(target, args.apply, args.include_b))
    except SystemExit as exc:
        print(exc.code, file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero; txn already rolled back
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
