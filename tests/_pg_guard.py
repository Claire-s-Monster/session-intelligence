"""
PostgreSQL test-database resolution and safety guard (issue #114).

The PostgreSQL-backed suites run against whatever ``POSTGRES_DSN`` names, and
some of them DELETE rows or CREATE/DROP databases. On a developer machine the
only database that exists by default is the production one the :4002 server
uses, so this module decides which DSN a test run gets and refuses to hand out
a production one.

Kept free of pytest imports so the logic is unit-testable directly.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable, Mapping
from urllib.parse import parse_qs, urlsplit

PRODUCTION_DATABASE = "session_intelligence"
DEFAULT_TEST_DSN = "postgresql://localhost/session_intelligence_test"

# Every PostgreSQL skipif in the suite carries this substring, and the terminal
# summary counts skips by it. New PostgreSQL skip reasons must contain it.
PG_SKIP_REASON = "PostgreSQL not available"

_TRUTHY = {"1", "true", "yes", "on"}


def database_name(dsn: str, environ: Mapping[str, str] | None = None) -> str | None:
    """Return the database ``dsn`` connects to, resolved the way libpq does.

    Order: URL path, then ``dbname``/``database`` query parameter, then
    ``PGDATABASE``, then the connecting user name (libpq's default).
    """
    env = os.environ if environ is None else environ
    parts = urlsplit(dsn)
    name = parts.path.strip("/")
    if not name:
        query = parse_qs(parts.query)
        for key in ("dbname", "database"):
            if query.get(key):
                name = query[key][0]
                break
    if not name:
        name = env.get("PGDATABASE", "")
    if not name:
        name = parts.username or env.get("PGUSER", "")
    return name or None


def production_dsn_reason(dsn: str, environ: Mapping[str, str] | None = None) -> str | None:
    """Return why ``dsn`` must not be used by the test suite, or None if it is safe."""
    env = os.environ if environ is None else environ
    if database_name(dsn, env) == PRODUCTION_DATABASE:
        return f"POSTGRES_DSN targets the production database {PRODUCTION_DATABASE!r}"
    served = env.get("SESSION_DB_DSN", "")
    if served and served.rstrip("/") == dsn.rstrip("/"):
        return "POSTGRES_DSN equals SESSION_DB_DSN, the DSN the server itself uses"
    return None


def require_pg(environ: Mapping[str, str] | None = None) -> bool:
    """Whether ``SESSION_TEST_REQUIRE_PG`` asks for a missing database to be an error."""
    env = os.environ if environ is None else environ
    return env.get("SESSION_TEST_REQUIRE_PG", "").strip().lower() in _TRUTHY


def database_reachable(dsn: str, timeout: float = 2.0) -> bool:
    """Whether a connection to ``dsn`` can actually be opened."""
    try:
        import asyncpg
    except ImportError:
        return False

    async def _probe() -> None:
        conn = await asyncpg.connect(dsn, timeout=timeout)
        await conn.close()

    try:
        asyncio.run(_probe())
    except Exception:
        return False
    return True


def count_pg_skips(reports: Iterable[object]) -> int:
    """Count skip reports whose reason marks a PostgreSQL-gated test."""
    count = 0
    for report in reports:
        longrepr = getattr(report, "longrepr", None)
        if isinstance(longrepr, tuple) and len(longrepr) == 3:
            reason = str(longrepr[2])
        else:
            reason = str(longrepr or "")
        if PG_SKIP_REASON in reason:
            count += 1
    return count
