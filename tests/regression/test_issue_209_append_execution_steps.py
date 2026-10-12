"""Regression tests for issue #209: persisting an AgentExecution rewrote all its steps.

Bug: every persist of a changed AgentExecution re-sent the whole
     ``execution_steps`` JSON, so total bytes written grew quadratically with
     the number of steps.
Fix: the persist tracker remembers (count, digest) of the steps last written;
     when only new steps were appended the sweep marks the payload for a PostgreSQL
     ``execution_steps || $new`` UPDATE guarded by the expected array length, with
     a full upsert fallback. SQLite always rewrites the whole list.
"""

import json
import os
import uuid
from types import SimpleNamespace

import pytest
import pytest_asyncio

from core.session_engine import SessionIntelligenceEngine
from persistence.postgresql import PostgreSQLBackend
from persistence.sqlite import SQLiteBackend
from transport.http_server import HTTPSessionIntelligenceServer
from transport.persist_sweep import PendingBatch, collect_session, write_pending
from transport.persist_tracker import PersistDigestTracker

POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "")


# --------------------------------------------------------------------------
# (a) unit: tracker + collect, no database
# --------------------------------------------------------------------------


class StubExecution:
    def __init__(self, steps: list[dict]) -> None:
        self.id = "exec-1"
        self.status = "running"
        self.steps = steps

    def model_dump(self):
        return {
            "id": self.id,
            "agent_name": "a",
            "status": self.status,
            "execution_steps": [dict(s) for s in self.steps],
        }


class StubSession:
    def __init__(self, execution: StubExecution) -> None:
        self.id = "s1"
        self.decisions: list = []
        self.agents_executed = [execution]

    def model_dump(self):
        return {"id": self.id, "status": "active"}


class BatchDB:
    """persist_batch recorder; ``fail`` makes every write raise."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.executions: list[dict] = []

    async def persist_batch(self, sessions, decisions, executions):
        if self.fail:
            raise RuntimeError("boom")
        self.executions = list(executions)

    async def save_session(self, data):
        if self.fail:
            raise RuntimeError("boom")

    async def save_decision(self, data):
        if self.fail:
            raise RuntimeError("boom")

    async def save_agent_execution(self, data):
        if self.fail:
            raise RuntimeError("boom")
        self.executions = [data]


def _steps(n: int, start: int = 0) -> list[dict]:
    return [{"step_id": f"x{i}", "status": "running", "n": i} for i in range(start, start + n)]


async def _sweep(tracker, session, db) -> dict | None:
    batch = PendingBatch()
    collect_session(tracker, "s1", session, batch)
    await write_pending(db, tracker, batch)
    return batch.executions[0].data if batch.executions else None


@pytest.mark.regression
class TestAppendPlanning:
    async def test_first_write_is_full(self):
        tracker = PersistDigestTracker()
        data = await _sweep(tracker, StubSession(StubExecution(_steps(3))), BatchDB())
        assert "_append_from" not in data

    async def test_appended_steps_are_marked_with_only_the_slice(self):
        tracker = PersistDigestTracker()
        ex = StubExecution(_steps(3))
        session = StubSession(ex)
        await _sweep(tracker, session, BatchDB())

        ex.steps = _steps(5)
        data = await _sweep(tracker, session, BatchDB())

        assert data["_append_from"] == 3
        assert data["_append_steps"] == _steps(2, start=3)
        assert data["execution_steps"] == _steps(5)

    async def test_full_after_forget(self):
        tracker = PersistDigestTracker()
        ex = StubExecution(_steps(3))
        session = StubSession(ex)
        await _sweep(tracker, session, BatchDB())
        tracker.forget("s1")

        ex.steps = _steps(5)
        data = await _sweep(tracker, session, BatchDB())

        assert "_append_from" not in data

    async def test_full_when_earlier_step_changed(self):
        tracker = PersistDigestTracker()
        ex = StubExecution(_steps(3))
        session = StubSession(ex)
        await _sweep(tracker, session, BatchDB())

        ex.steps = _steps(5)
        ex.steps[1]["status"] = "abandoned"  # finalize reconcile
        data = await _sweep(tracker, session, BatchDB())

        assert "_append_from" not in data

    async def test_full_when_steps_shrank(self):
        tracker = PersistDigestTracker()
        ex = StubExecution(_steps(4))
        session = StubSession(ex)
        await _sweep(tracker, session, BatchDB())

        ex.steps = _steps(2)
        data = await _sweep(tracker, session, BatchDB())

        assert "_append_from" not in data

    async def test_full_when_length_equal_and_other_field_changed(self):
        tracker = PersistDigestTracker()
        ex = StubExecution(_steps(3))
        session = StubSession(ex)
        await _sweep(tracker, session, BatchDB())

        ex.status = "completed"
        data = await _sweep(tracker, session, BatchDB())

        assert "_append_from" not in data

    async def test_failed_write_does_not_record_step_state(self):
        tracker = PersistDigestTracker()
        ex = StubExecution(_steps(3))
        session = StubSession(ex)
        await _sweep(tracker, session, BatchDB(fail=True))

        ex.steps = _steps(5)
        data = await _sweep(tracker, session, BatchDB())

        assert "_append_from" not in data


# --------------------------------------------------------------------------
# Engine-driven fixtures shared by PG and SQLite round trips
# --------------------------------------------------------------------------


def _make_server() -> HTTPSessionIntelligenceServer:
    server = HTTPSessionIntelligenceServer.__new__(HTTPSessionIntelligenceServer)
    server.persist_tracker = PersistDigestTracker()
    return server


def _request(database, engine):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(database=database, session_engine=engine))
    )


async def _start(engine, project: str) -> str:
    result = await engine.session_manage_lifecycle(
        operation="create", mode="local", project_name=project
    )
    sid = result.session_id
    await engine.session_track_execution(
        session_id=sid,
        agent_name="agent-209",
        step_data={"phase": "agent_start", "agent_type": "t"},
        allow_unbound=True,
    )
    return sid


async def _add_steps(engine, sid: str, count: int, tag: str) -> None:
    for i in range(count):
        await engine.session_track_execution(
            session_id=sid,
            agent_name="agent-209",
            step_data={"phase": "tool_use", "tool": f"{tag}{i}"},
            allow_unbound=True,
        )


def _cached_steps(engine, sid: str) -> list:
    return json.loads(
        json.dumps(
            [s.model_dump() for s in engine.session_cache[sid].agents_executed[0].execution_steps],
            default=str,
        )
    )


def _as_list(value) -> list:
    return json.loads(value) if isinstance(value, str) else value


class SpyConn:
    def __init__(self, conn, log: list[str]) -> None:
        self._conn = conn
        self._log = log

    async def execute(self, sql, *args):
        self._log.append(sql)
        return await self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


class _Acquire:
    def __init__(self, ctx, log: list[str]) -> None:
        self._ctx = ctx
        self._log = log

    async def __aenter__(self):
        return SpyConn(await self._ctx.__aenter__(), self._log)

    async def __aexit__(self, *exc):
        return await self._ctx.__aexit__(*exc)


class SpyPool:
    def __init__(self, pool, log: list[str]) -> None:
        self._pool = pool
        self._log = log

    def acquire(self):
        return _Acquire(self._pool.acquire(), self._log)

    def __getattr__(self, name):
        return getattr(self._pool, name)


def _appended(log: list[str]) -> bool:
    return any("execution_steps ||" in sql for sql in log)


def _full_upserted(log: list[str]) -> bool:
    return any("INSERT INTO agent_executions" in sql for sql in log)


# --------------------------------------------------------------------------
# PostgreSQL
# --------------------------------------------------------------------------


@pytest_asyncio.fixture
async def pg(monkeypatch):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
    database = PostgreSQLBackend(POSTGRES_DSN)
    await database.initialize()
    project = f"issue209-{uuid.uuid4().hex[:8]}"
    engine = SessionIntelligenceEngine(repository_path=".", database=database, use_filesystem=False)
    yield SimpleNamespace(db=database, engine=engine, project=project)
    pool = database._ensure_connected()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM agent_executions WHERE session_id IN "
            "(SELECT id FROM sessions WHERE project_name = $1)",
            project,
        )
        await conn.execute("DELETE FROM sessions WHERE project_name = $1", project)
    await database.close()


async def _stored_pg(db, sid: str) -> list:
    pool = db._ensure_connected()
    async with pool.acquire() as conn:
        raw = await conn.fetchval(
            "SELECT execution_steps::text FROM agent_executions WHERE session_id = $1", sid
        )
    return json.loads(raw)


def _spy(monkeypatch, db) -> list[str]:
    log: list[str] = []
    real = db._ensure_connected()
    monkeypatch.setattr(db, "_ensure_connected", lambda: SpyPool(real, log))
    return log


@pytest.mark.regression
@pytest.mark.postgresql
@pytest.mark.skipif(not POSTGRES_DSN, reason="POSTGRES_DSN not set; PostgreSQL not available")
class TestPostgresAppend:
    async def test_append_path_stores_all_steps_in_order(self, pg, monkeypatch):
        sid = await _start(pg.engine, pg.project)
        await _add_steps(pg.engine, sid, 4, "a")
        server, request = _make_server(), _request(pg.db, pg.engine)
        await server._persist_sessions_to_database(request)
        assert await _stored_pg(pg.db, sid) == _cached_steps(pg.engine, sid)

        await _add_steps(pg.engine, sid, 3, "b")
        log = _spy(monkeypatch, pg.db)
        await server._persist_sessions_to_database(request)

        stored = await _stored_pg(pg.db, sid)
        assert stored == _cached_steps(pg.engine, sid)
        assert len(stored) >= 7
        assert _appended(log)
        assert not _full_upserted(log)

    async def test_length_guard_falls_back_to_full_rewrite(self, pg, monkeypatch):
        sid = await _start(pg.engine, pg.project)
        await _add_steps(pg.engine, sid, 4, "a")
        server, request = _make_server(), _request(pg.db, pg.engine)
        await server._persist_sessions_to_database(request)

        pool = pg.db._ensure_connected()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE agent_executions SET execution_steps = '[]'::jsonb WHERE session_id = $1",
                sid,
            )

        await _add_steps(pg.engine, sid, 2, "b")
        log = _spy(monkeypatch, pg.db)
        await server._persist_sessions_to_database(request)

        assert _appended(log)
        assert _full_upserted(log)
        assert await _stored_pg(pg.db, sid) == _cached_steps(pg.engine, sid)

    async def test_finalize_reconcile_then_sweep_rewrites_fully(self, pg, monkeypatch):
        sid = await _start(pg.engine, pg.project)
        await _add_steps(pg.engine, sid, 3, "a")
        server, request = _make_server(), _request(pg.db, pg.engine)
        await server._persist_sessions_to_database(request)
        await _add_steps(pg.engine, sid, 2, "b")
        await server._persist_sessions_to_database(request)

        await pg.engine.session_manage_lifecycle(operation="finalize", session_id=sid)
        log = _spy(monkeypatch, pg.db)
        if sid in pg.engine.session_cache:
            await server._persist_sessions_to_database(request)
            assert not _appended(log)

        stored = await _stored_pg(pg.db, sid)
        assert stored
        assert all(s["status"] != "running" for s in stored)

    async def test_retried_batch_cannot_double_append(self, pg):
        sid = await _start(pg.engine, pg.project)
        await _add_steps(pg.engine, sid, 3, "a")
        server, request = _make_server(), _request(pg.db, pg.engine)
        await server._persist_sessions_to_database(request)
        await _add_steps(pg.engine, sid, 2, "b")

        tracker = server.persist_tracker
        batch = PendingBatch()
        collect_session(tracker, sid, pg.engine.session_cache[sid], batch)
        assert batch.executions[0].data.get("_append_from") is not None
        # Same batch written twice (first commit "lost"): the guard rejects the
        # second append and the full upsert restores the exact list.
        await write_pending(pg.db, PersistDigestTracker(), batch)
        await write_pending(pg.db, PersistDigestTracker(), batch)

        assert await _stored_pg(pg.db, sid) == _cached_steps(pg.engine, sid)


# --------------------------------------------------------------------------
# SQLite: always a full rewrite, still correct
# --------------------------------------------------------------------------


@pytest.mark.regression
class TestSqliteRoundTrip:
    async def test_append_sweep_round_trip_is_correct(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
        engine = SessionIntelligenceEngine(repository_path=str(tmp_path), use_filesystem=False)
        engine.database = SQLiteBackend(str(tmp_path / "issue209.db"))
        await engine.database.initialize()
        try:
            sid = await _start(engine, "issue209-sqlite")
            await _add_steps(engine, sid, 3, "a")
            server, request = _make_server(), _request(engine.database, engine)
            await server._persist_sessions_to_database(request)
            await _add_steps(engine, sid, 2, "b")
            await server._persist_sessions_to_database(request)

            rows = await engine.database.query_agent_executions(session_id=sid)
            assert len(rows) == 1
            assert _as_list(rows[0]["execution_steps"]) == _cached_steps(engine, sid)
        finally:
            await engine.database.close()
