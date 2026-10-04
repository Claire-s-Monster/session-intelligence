"""Schema-fingerprint fast path and DDL lock_timeout tests (issue #174).

``initialize()`` used to re-run all DDL on every call. ``ALTER TABLE ... ADD
COLUMN IF NOT EXISTS`` takes ACCESS EXCLUSIVE even when the column exists, so
one initialize() against a live DB queued behind any open query and stalled
every later query on that table. Each test gets its own throwaway database
(never the one in POSTGRES_DSN), so the fingerprint row cannot outlive it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from tests.persistence.conftest import POSTGRES_AVAILABLE
from tests.persistence.test_postgresql_migration import (
    _create_scratch_database,
    _drop_scratch_database,
    _with_database,
)

pytestmark = [
    pytest.mark.postgresql,
    pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not available"),
]


@pytest.fixture
async def scratch_dsn():
    admin_dsn = os.environ["POSTGRES_DSN"]
    name = f"si_test_{uuid.uuid4().hex[:12]}"
    await _create_scratch_database(admin_dsn, name)
    try:
        yield _with_database(admin_dsn, name)
    finally:
        await _drop_scratch_database(admin_dsn, name)


class _LockHolder:
    """A connection holding ACCESS SHARE on sessions inside an open txn."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self._tx = conn.transaction()
        self._open = False

    async def hold(self) -> None:
        await self._tx.start()
        self._open = True
        await self._conn.fetchval("SELECT 1 FROM sessions LIMIT 1")

    async def release(self) -> None:
        if self._open:
            self._open = False
            await self._tx.rollback()


@pytest.fixture
async def lock_holder(scratch_dsn):
    import asyncpg

    conn = await asyncpg.connect(scratch_dsn)
    holder = _LockHolder(conn)
    try:
        yield holder
    finally:
        await holder.release()
        await conn.close()


async def _fingerprint_rows(dsn: str) -> int:
    import asyncpg

    conn = await asyncpg.connect(dsn)
    try:
        return await conn.fetchval("SELECT count(*) FROM schema_fingerprint")
    finally:
        await conn.close()


async def test_second_initialize_takes_fast_path(scratch_dsn, caplog):
    from persistence.postgresql import PostgreSQLBackend

    caplog.set_level(logging.INFO, logger="persistence.postgresql")
    first = PostgreSQLBackend(dsn=scratch_dsn)
    await first.initialize()
    await first.close()
    assert "slow path" in caplog.text
    assert await _fingerprint_rows(scratch_dsn) == 1

    caplog.clear()
    second = PostgreSQLBackend(dsn=scratch_dsn)
    with patch.object(PostgreSQLBackend, "_apply_schema", new=AsyncMock()) as apply_schema:
        await second.initialize()
    await second.close()
    apply_schema.assert_not_called()
    assert "fast path" in caplog.text
    assert "slow path" not in caplog.text


async def test_fast_path_does_not_stall_behind_open_query(scratch_dsn, lock_holder):
    from persistence.postgresql import PostgreSQLBackend

    seed = PostgreSQLBackend(dsn=scratch_dsn)
    await seed.initialize()
    await seed.close()

    await lock_holder.hold()
    db = PostgreSQLBackend(dsn=scratch_dsn)
    await asyncio.wait_for(db.initialize(), 5)
    await db.close()


async def test_slow_path_fails_fast_under_lock_then_recovers(scratch_dsn, lock_holder):
    import asyncpg

    from persistence.postgresql import PostgreSQLBackend

    seed = PostgreSQLBackend(dsn=scratch_dsn)
    await seed.initialize()
    async with seed._pool.acquire() as conn:
        await conn.execute("DELETE FROM schema_fingerprint")
    await seed.close()

    await lock_holder.hold()
    blocked = PostgreSQLBackend(dsn=scratch_dsn, lock_timeout_ms=200, ddl_max_attempts=2)
    with pytest.raises(asyncpg.exceptions.LockNotAvailableError):
        await asyncio.wait_for(blocked.initialize(), 10)

    await lock_holder.release()
    recovered = PostgreSQLBackend(dsn=scratch_dsn)
    await asyncio.wait_for(recovered.initialize(), 10)
    await recovered.close()
    assert await _fingerprint_rows(scratch_dsn) == 1


async def test_slow_path_does_not_leak_lock_timeout(scratch_dsn):
    from persistence.postgresql import PostgreSQLBackend

    db = PostgreSQLBackend(dsn=scratch_dsn, min_size=1, max_size=1)
    await db.initialize()
    try:
        async with db._pool.acquire() as conn:
            assert await conn.fetchval("SHOW lock_timeout") == "0"
    finally:
        await db.close()
