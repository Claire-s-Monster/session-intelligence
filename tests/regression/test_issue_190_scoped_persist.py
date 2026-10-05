"""Regression tests for issue #190: the persist sweep walked the whole cache.

Fix under test:
* the engine marks the sessions it creates/mutates dirty, and the HTTP sweep
  persists only those (full walk on the first sweep and whenever nothing is
  marked, as a fail-safe; the #67 digest filter applies in every mode);
* changed entities are written in one ``persist_batch`` transaction, with an
  entity-by-entity retry if the batch fails (poison-row fallback);
* the sweep's counts/timings ride on the ``persist_sessions`` slow_db event and
  ``session_cache_size`` is part of the stall snapshot.
"""

import logging
from types import SimpleNamespace

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence import DatabaseConfig
from persistence.sqlite import SQLiteBackend
from transport.http_server import HTTPSessionIntelligenceServer
from transport.persist_tracker import PersistDigestTracker
from transport.security import SecurityConfig
from transport.stall_monitor import StallMonitor


class CountingDatabase:
    """Per-entity saves only (no persist_batch), recording every call."""

    def __init__(self) -> None:
        self.sessions: list[str] = []
        self.decisions: list[str] = []
        self.agent_executions: list[str] = []

    def reset(self) -> None:
        self.sessions.clear()
        self.decisions.clear()
        self.agent_executions.clear()

    async def save_session(self, session_data):
        self.sessions.append(session_data["id"])

    async def save_decision(self, decision_data):
        self.decisions.append(decision_data.get("id"))

    async def save_agent_execution(self, execution_data):
        self.agent_executions.append(execution_data.get("id") or execution_data.get("execution_id"))


class FailingDecisionDatabase(CountingDatabase):
    async def save_decision(self, decision_data):
        raise RuntimeError("simulated decision write failure")


def make_server() -> HTTPSessionIntelligenceServer:
    server = HTTPSessionIntelligenceServer.__new__(HTTPSessionIntelligenceServer)
    server.persist_tracker = PersistDigestTracker()
    return server


def make_request(database, engine):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(database=database, session_engine=engine))
    )


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path), use_filesystem=False)
    eng.database = SQLiteBackend(str(tmp_path / "issue190.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


async def _create(engine, name: str) -> str:
    result = await engine.session_manage_lifecycle(
        operation="create", mode="local", project_name=name
    )
    return result.session_id


async def _log_decision(engine, session_id: str, text: str = "decide"):
    return await engine.session_log_decision(
        decision=text, context={"category": "test"}, session_id=session_id
    )


@pytest.mark.regression
class TestDirtyMarking:
    async def test_create_marks_session_dirty(self, engine):
        sid = await _create(engine, "p-create")
        assert sid in engine.dirty_snapshot()

    async def test_log_decision_marks_its_session_only(self, engine):
        sid_a = await _create(engine, "p-a")
        sid_b = await _create(engine, "p-b")
        for sid, ver in engine.dirty_snapshot().items():
            engine.clear_dirty(sid, ver)

        await _log_decision(engine, sid_a)

        assert set(engine.dirty_snapshot()) == {sid_a}
        assert sid_b not in engine.dirty_snapshot()

    async def test_track_execution_marks_its_session(self, engine):
        sid = await _create(engine, "p-track")
        for s, ver in engine.dirty_snapshot().items():
            engine.clear_dirty(s, ver)

        await engine.session_track_execution(
            session_id=sid,
            agent_name="a",
            step_data={"phase": "agent_start", "agent_type": "t"},
            allow_unbound=True,
        )

        assert sid in engine.dirty_snapshot()

    async def test_validate_and_finalize_mark_their_session(self, engine):
        sid = await _create(engine, "p-life")
        for s, ver in engine.dirty_snapshot().items():
            engine.clear_dirty(s, ver)

        await engine.session_manage_lifecycle(operation="validate", session_id=sid)
        assert sid in engine.dirty_snapshot()

        for s, ver in engine.dirty_snapshot().items():
            engine.clear_dirty(s, ver)
        await engine.session_manage_lifecycle(operation="finalize", session_id=sid)
        assert sid in engine.dirty_snapshot()

    async def test_hydrating_an_uncached_session_marks_it(self, engine):
        sid = await _create(engine, "p-hydrate")
        engine.session_cache.pop(sid)
        for s, ver in engine.dirty_snapshot().items():
            engine.clear_dirty(s, ver)

        await engine.session_track_execution(
            session_id=sid,
            agent_name="a",
            step_data={"phase": "agent_start", "agent_type": "t"},
            allow_unbound=True,
        )

        assert sid in engine.dirty_snapshot()

    async def test_clear_dirty_ignores_a_stale_version(self, engine):
        sid = await _create(engine, "p-version")
        stale = engine.dirty_snapshot()[sid]
        engine.mark_dirty(sid)  # a mutation landed while a write was awaited

        engine.clear_dirty(sid, stale)

        assert sid in engine.dirty_snapshot()


@pytest.mark.regression
class TestScopedSweep:
    async def _three_sessions(self, engine):
        return [await _create(engine, f"p{i}") for i in range(3)]

    async def test_first_persist_is_a_full_walk(self, engine):
        await self._three_sessions(engine)
        engine._dirty_sessions.clear()  # isolate the first-sweep rule
        server = make_server()

        stats = await server._persist_sessions_to_database(make_request(CountingDatabase(), engine))

        assert stats["persist_mode"] == "full_first"
        assert stats["sessions_walked"] == 3
        assert server.persist_metrics["full_walk_first"] == 1
        assert server.persist_metrics["full_walk_fallbacks"] == 0

    async def test_only_the_dirty_session_is_walked_and_written(self, engine):
        sids = await self._three_sessions(engine)
        database = CountingDatabase()
        server = make_server()
        request = make_request(database, engine)
        await server._persist_sessions_to_database(request)  # full first walk
        assert not engine.dirty_snapshot()
        database.reset()

        await _log_decision(engine, sids[1], "only this one")
        stats = await server._persist_sessions_to_database(request, tool="session_log_decision")

        assert stats["persist_mode"] == "dirty"
        assert stats["sessions_walked"] == 1, "the other two sessions must not be digested"
        assert stats["entities_digested"] == 2  # session row + decision
        assert stats["entities_written"] == 1  # session row unchanged: digest filter skips it
        assert database.sessions == []
        assert len(database.decisions) == 1
        assert not engine.dirty_snapshot(), "a successful write clears the dirty mark"
        assert server.persist_metrics["dirty_walks"] == 1

    async def test_empty_dirty_set_falls_back_to_full_walk(self, engine):
        await self._three_sessions(engine)
        database = CountingDatabase()
        server = make_server()
        request = make_request(database, engine)
        await server._persist_sessions_to_database(request)
        database.reset()

        stats = await server._persist_sessions_to_database(request, tool="session_log_learning")

        assert stats["persist_mode"] == "full_fallback"
        assert stats["sessions_walked"] == 3
        assert stats["entities_written"] == 0  # digest filter still skips everything
        assert server.persist_metrics["full_walk_fallbacks"] == 1
        assert server.persist_metrics["full_walk_fallbacks_by_tool"] == {"session_log_learning": 1}

    async def test_failed_write_stays_dirty_and_is_retried(self, engine):
        sid = await _create(engine, "p-fail")
        await _log_decision(engine, sid)
        database = FailingDecisionDatabase()
        server = make_server()
        request = make_request(database, engine)

        await server._persist_sessions_to_database(request)

        assert sid in engine.dirty_snapshot()
        stats = await server._persist_sessions_to_database(request)
        assert stats["persist_mode"] == "dirty"
        assert stats["sessions_walked"] == 1


@pytest.mark.regression
class TestFullWalkBackstops:
    """A mutation that bypasses mark_dirty must still reach the DB."""

    async def _setup(self, engine):
        sid_a = await _create(engine, "p-unmarked")
        sid_b = await _create(engine, "p-dirty")
        database = CountingDatabase()
        server = make_server()
        request = make_request(database, engine)
        await server._persist_sessions_to_database(request)  # first full walk
        database.reset()
        # Unmarkable path: direct edit on a held Session, no mark_dirty.
        engine.session_cache[sid_a].metadata.tags = ["sneaky"]
        engine.mark_dirty(sid_b)
        return sid_a, sid_b, server, request, database

    async def test_shutdown_full_walk_persists_unmarked_mutation(self, engine):
        sid_a, _, server, request, database = await self._setup(engine)

        await server._persist_full_walk_at_shutdown(request.app)

        assert sid_a in database.sessions
        assert server.persist_metrics["full_walk_shutdown"] == 1

    async def test_periodic_full_walk_persists_unmarked_mutation(self, engine, monkeypatch):
        import transport.http_server as http_server

        clock = {"t": 1000.0}
        monkeypatch.setattr(http_server, "_now", lambda: clock["t"])
        sid_a, _, server, request, database = await self._setup(engine)

        # Within the interval: dirty-only, A is missed.
        clock["t"] += 30
        stats = await server._persist_sessions_to_database(request, tool="t")
        assert stats["persist_mode"] == "dirty"
        assert sid_a not in database.sessions

        # Past the interval: full walk picks A up.
        clock["t"] += 31
        stats = await server._persist_sessions_to_database(request, tool="t")
        assert stats["persist_mode"] == "full_periodic"
        assert sid_a in database.sessions
        assert server.persist_metrics["full_walk_periodic"] == 1


class StubEntity:
    def __init__(self, entity_id, **fields):
        self._data = {"id": entity_id, **fields}

    def model_dump(self):
        return dict(self._data)


class StubSession:
    def __init__(self, session_id, decisions=(), executions=()):
        self.id = session_id
        self.decisions = list(decisions)
        self.agents_executed = list(executions)

    def model_dump(self):
        return {
            "id": self.id,
            "started": "2026-01-01T00:00:00+00:00",
            "project_path": "/tmp/p",
            "project_name": "p",
            "status": "active",
            "decisions": [d.model_dump() for d in self.decisions],
            "agents_executed": [a.model_dump() for a in self.agents_executed],
        }

    def executions_dump(self):
        return [{**a.model_dump(), "session_id": self.id} for a in self.agents_executed]


@pytest.mark.regression
class TestBatchedWrite:
    @pytest.fixture
    async def db(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "batch190.db"))
        await backend.initialize()
        yield backend
        await backend.close()

    async def test_clean_sweep_uses_one_batch(self, db):
        calls = []
        original = db.persist_batch

        async def spy(sessions, decisions, executions):
            calls.append((len(sessions), len(decisions), len(executions)))
            await original(sessions, decisions, executions)

        db.persist_batch = spy
        session = StubSession(
            "s1",
            decisions=[StubEntity("d1", description="x", timestamp="2026-01-01T00:00:00")],
            executions=[StubEntity("e1", agent_name="a")],
        )
        server = make_server()

        stats = await server._persist_sessions_to_database(
            make_request(db, SimpleNamespace(session_cache={"s1": session}))
        )

        assert calls == [(1, 1, 1)]
        assert stats["entities_written"] == 3
        assert stats["batch_fallback"] is False

    async def test_poison_row_falls_back_and_only_its_digest_stays_uncommitted(self, db, caplog):
        session = StubSession(
            "s1",
            decisions=[StubEntity("d1", description="x", timestamp="2026-01-01T00:00:00")],
            executions=[
                StubEntity("e-good", agent_name="a"),
                StubEntity("e-bad"),  # no agent_name: malformed
            ],
        )
        server = make_server()
        request = make_request(db, SimpleNamespace(session_cache={"s1": session}))

        with caplog.at_level(logging.WARNING):
            stats = await server._persist_sessions_to_database(request)

        assert stats["batch_fallback"] is True
        assert stats["entities_written"] == 3  # session + decision + good execution
        assert server.persist_metrics["batch_failures"] == 1
        assert any("retrying entity by entity" in r.getMessage() for r in caplog.records)
        assert await db.get_session("s1") is not None
        assert len(await db.query_decisions_by_session("s1")) == 1
        assert [r["id"] for r in await db.query_agent_executions(session_id="s1")] == ["e-good"]

        tracker = server.persist_tracker
        assert (
            tracker.digest_if_changed("s1", "execution:e-good", session.executions_dump()[0])
            is None
        )
        assert (
            tracker.digest_if_changed("s1", "execution:e-bad", session.executions_dump()[1])
            is not None
        )


@pytest.mark.regression
class TestStallEvidence:
    async def test_slow_db_event_carries_persist_counts(self, engine):
        await _create(engine, "p-evidence")
        server = make_server()
        monitor = StallMonitor(slow_db_ms=0)
        stats: dict = {}

        async with monitor.timed_db("persist_sessions", extra=stats):
            await server._persist_sessions_to_database(
                make_request(CountingDatabase(), engine), tool="t", stats=stats
            )

        event = [e for e in monitor.snapshot()["events"] if e["type"] == "slow_db"][-1]
        assert event["label"] == "persist_sessions"
        assert event["persist_mode"] == "full_first"
        assert event["sessions_walked"] == 1
        assert event["entities_digested"] == 1
        assert event["entities_written"] == 1
        assert "walk_digest_ms" in event
        assert "write_ms" in event

    def test_snapshot_includes_extra_stats(self):
        monitor = StallMonitor(extra_stats=lambda: {"session_cache_size": 7})
        assert monitor.snapshot()["session_cache_size"] == 7

    def test_failing_extra_stats_provider_does_not_break_snapshot(self):
        def boom():
            raise RuntimeError("nope")

        assert "thresholds" in StallMonitor(extra_stats=boom).snapshot()

    async def test_server_snapshot_reports_session_cache_size(self, engine, tmp_path):
        await _create(engine, "p-size")
        server = HTTPSessionIntelligenceServer(
            host="127.0.0.1",
            port=4099,
            repository_path=str(tmp_path),
            db_config=DatabaseConfig(),
            security_config=SecurityConfig(
                localhost_only=False, allowed_origins=["*"], require_api_key=False
            ),
        )
        server.session_engine = engine

        snapshot = server.stall_monitor.snapshot()

        assert snapshot["session_cache_size"] == 1
        assert snapshot["persist"]["calls"] == 0
