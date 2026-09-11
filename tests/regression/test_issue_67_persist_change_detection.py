"""Regression tests for issue #67: persist loop re-upserted the entire cache.

Bug: ``_persist_sessions_to_database`` walked the whole ``session_cache`` and
     unconditionally re-upserted every session, every decision and every agent
     execution after *each* session-modifying tool call. Measured at ~12.8 no-op
     UPDATE/sec on ``agent_executions`` while the server was idle, with 1,626
     autovacuum runs on that one table.
Fix: a content digest is recorded per entity; a payload identical to the last
     successful write is skipped, and digests are committed only *after* the
     write succeeds so failures are retried.
"""

from types import SimpleNamespace

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend
from transport.http_server import HTTPSessionIntelligenceServer
from transport.persist_tracker import PersistDigestTracker, compute_digest


class CountingDatabase:
    """Records every save call so tests can assert on write volume."""

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
    """Fails every ``save_decision`` to prove digests are not committed on failure."""

    async def save_decision(self, decision_data):
        self.decisions.append(decision_data.get("id"))
        raise RuntimeError("simulated decision write failure")


class StubEntity:
    """Minimal stand-in for a Decision / AgentExecution pydantic model."""

    def __init__(self, entity_id: str, **fields) -> None:
        self._data = {"id": entity_id, **fields}

    def model_dump(self):
        return dict(self._data)

    def mutate(self, key: str, value) -> None:
        self._data[key] = value


class StubSession:
    """Minimal stand-in for a Session pydantic model."""

    def __init__(self, session_id: str, decisions=None, agents_executed=None) -> None:
        self.id = session_id
        self.status = "active"
        self.decisions = decisions if decisions is not None else []
        self.agents_executed = agents_executed if agents_executed is not None else []

    def model_dump(self):
        return {
            "id": self.id,
            "status": self.status,
            "decisions": [d.model_dump() for d in self.decisions],
            "agents_executed": [a.model_dump() for a in self.agents_executed],
        }


def make_server() -> HTTPSessionIntelligenceServer:
    """Build a server without running ``__init__`` (which would load DB config)."""
    server = HTTPSessionIntelligenceServer.__new__(HTTPSessionIntelligenceServer)
    server.persist_tracker = PersistDigestTracker()
    return server


def make_request(database, session_cache):
    engine = SimpleNamespace(session_cache=session_cache)
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(database=database, session_engine=engine))
    )


@pytest.mark.regression
class TestPersistDigestTracker:
    def test_unchanged_payload_is_skipped_after_commit(self):
        tracker = PersistDigestTracker()
        payload = {"id": "d1", "text": "hello"}

        digest = tracker.digest_if_changed("s1", "decision:d1", payload)
        assert digest is not None
        tracker.commit("s1", "decision:d1", digest)

        assert tracker.digest_if_changed("s1", "decision:d1", payload) is None

    def test_changed_payload_is_not_skipped(self):
        tracker = PersistDigestTracker()
        digest = tracker.digest_if_changed("s1", "decision:d1", {"id": "d1", "n": 1})
        tracker.commit("s1", "decision:d1", digest)

        assert tracker.digest_if_changed("s1", "decision:d1", {"id": "d1", "n": 2}) is not None

    def test_uncommitted_digest_leaves_entity_dirty(self):
        """A digest that is never committed (failed write) must stay dirty."""
        tracker = PersistDigestTracker()
        payload = {"id": "d1"}

        assert tracker.digest_if_changed("s1", "decision:d1", payload) is not None
        assert tracker.digest_if_changed("s1", "decision:d1", payload) is not None

    def test_retain_drops_digests_for_evicted_sessions(self):
        tracker = PersistDigestTracker()
        for session_id in ("s1", "s2"):
            digest = tracker.digest_if_changed(session_id, "session", {"id": session_id})
            tracker.commit(session_id, "session", digest)
        assert tracker.tracked_entity_count() == 2

        tracker.retain(["s1"])

        assert tracker.tracked_entity_count() == 1
        assert tracker.digest_if_changed("s1", "session", {"id": "s1"}) is None
        assert tracker.digest_if_changed("s2", "session", {"id": "s2"}) is not None

    def test_forget_drops_a_single_session(self):
        tracker = PersistDigestTracker()
        digest = tracker.digest_if_changed("s1", "session", {"id": "s1"})
        tracker.commit("s1", "session", digest)

        tracker.forget("s1")

        assert tracker.digest_if_changed("s1", "session", {"id": "s1"}) is not None

    def test_digest_ignores_key_insertion_order(self):
        assert compute_digest({"a": 1, "b": 2}) == compute_digest({"b": 2, "a": 1})

    def test_uncanonicalisable_payload_always_looks_changed(self):
        """Mixed key types break sort_keys; fall back to always-write, never skip."""
        payload = {"a": 1, 2: "b"}
        assert compute_digest(payload) != compute_digest(payload)


@pytest.mark.regression
class TestPersistSessionsToDatabase:
    async def test_first_pass_writes_every_entity(self):
        session = StubSession(
            "s1",
            decisions=[StubEntity("d1"), StubEntity("d2")],
            agents_executed=[StubEntity("e1"), StubEntity("e2"), StubEntity("e3")],
        )
        database = CountingDatabase()
        server = make_server()

        await server._persist_sessions_to_database(make_request(database, {"s1": session}))

        assert database.sessions == ["s1"]
        assert database.decisions == ["d1", "d2"]
        assert database.agent_executions == ["e1", "e2", "e3"]

    async def test_second_pass_without_changes_writes_nothing(self):
        """The #67 bug: this pass used to re-upsert all six rows again."""
        session = StubSession(
            "s1",
            decisions=[StubEntity("d1"), StubEntity("d2")],
            agents_executed=[StubEntity("e1"), StubEntity("e2"), StubEntity("e3")],
        )
        database = CountingDatabase()
        server = make_server()
        request = make_request(database, {"s1": session})

        await server._persist_sessions_to_database(request)
        database.reset()
        await server._persist_sessions_to_database(request)

        assert database.sessions == []
        assert database.decisions == []
        assert database.agent_executions == []

    async def test_only_the_changed_execution_is_rewritten(self):
        executions = [StubEntity("e1"), StubEntity("e2"), StubEntity("e3")]
        session = StubSession("s1", agents_executed=executions)
        database = CountingDatabase()
        server = make_server()
        request = make_request(database, {"s1": session})

        await server._persist_sessions_to_database(request)
        database.reset()
        executions[1].mutate("status", "completed")
        await server._persist_sessions_to_database(request)

        assert database.agent_executions == ["e2"]

    async def test_appended_decision_does_not_rewrite_existing_ones(self):
        decisions = [StubEntity("d1"), StubEntity("d2")]
        session = StubSession("s1", decisions=decisions)
        database = CountingDatabase()
        server = make_server()
        request = make_request(database, {"s1": session})

        await server._persist_sessions_to_database(request)
        database.reset()
        decisions.append(StubEntity("d3"))
        await server._persist_sessions_to_database(request)

        assert database.decisions == ["d3"]

    async def test_failed_write_is_retried_on_the_next_pass(self):
        session = StubSession("s1", decisions=[StubEntity("d1")])
        database = FailingDecisionDatabase()
        server = make_server()
        request = make_request(database, {"s1": session})

        await server._persist_sessions_to_database(request)
        database.reset()
        await server._persist_sessions_to_database(request)

        assert database.decisions == [
            "d1"
        ], "A decision whose write failed must not have its digest committed"

    async def test_evicted_session_is_rewritten_when_it_returns(self):
        session = StubSession("s1", decisions=[StubEntity("d1")])
        database = CountingDatabase()
        server = make_server()

        await server._persist_sessions_to_database(make_request(database, {"s1": session}))
        await server._persist_sessions_to_database(make_request(database, {}))
        assert server.persist_tracker.tracked_entity_count() == 0

        database.reset()
        await server._persist_sessions_to_database(make_request(database, {"s1": session}))

        assert database.sessions == ["s1"]
        assert database.decisions == ["d1"]

    async def test_tracker_does_not_outgrow_the_cache(self):
        database = CountingDatabase()
        server = make_server()

        for index in range(5):
            session_id = f"s{index}"
            session = StubSession(session_id, decisions=[StubEntity(f"d{index}")])
            await server._persist_sessions_to_database(
                make_request(database, {session_id: session})
            )

        assert server.persist_tracker.tracked_entity_count() == 2


@pytest.mark.regression
class TestPersistWithRealSessionModel:
    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_repeat_persist_of_real_session_writes_nothing(self, engine):
        await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="issue-67"
        )
        await engine.session_log_decision(
            decision="Digest the payload instead of always upserting",
            context={"rationale": "issue #67", "category": "test"},
            project_name="issue-67",
        )
        assert engine.session_cache

        database = CountingDatabase()
        server = make_server()
        request = make_request(database, engine.session_cache)

        await server._persist_sessions_to_database(request)
        assert database.sessions, "first pass must persist the session"
        database.reset()

        await server._persist_sessions_to_database(request)

        assert database.sessions == []
        assert database.decisions == []
        assert database.agent_executions == []
