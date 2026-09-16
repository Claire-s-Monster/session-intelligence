"""
Create the local PostgreSQL test database (issue #114).

The PostgreSQL-backed test suites skip unless a test database is reachable.
This creates ``session_intelligence_test`` alongside the production database so
``pixi run -e ci test`` exercises both backends. Idempotent.

Usage:
    pixi run -e ci test-db-create
    python scripts/create_test_db.py --admin-dsn postgresql://localhost/postgres
"""

import argparse
import asyncio
import re

import asyncpg

TEST_DATABASE = "session_intelligence_test"


async def create_database(admin_dsn: str, name: str) -> bool:
    """Create ``name`` if it does not exist. Return True if it was created."""
    conn = await asyncpg.connect(admin_dsn)
    try:
        if await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name):
            return False
        # CREATE DATABASE cannot take a bind parameter; ``name`` is validated in main().
        await conn.execute(f'CREATE DATABASE "{name}"')
        return True
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--admin-dsn", default="postgresql://localhost/postgres")
    parser.add_argument("--name", default=TEST_DATABASE)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.name):
        parser.error(f"invalid database name: {args.name!r}")
    if "test" not in args.name:
        parser.error("refusing to create a database whose name does not contain 'test'")

    created = asyncio.run(create_database(args.admin_dsn, args.name))
    state = "created" if created else "already exists"
    print(f"{args.name}: {state}")
    print(f"POSTGRES_DSN defaults to postgresql://localhost/{TEST_DATABASE} when unset")


if __name__ == "__main__":
    main()
