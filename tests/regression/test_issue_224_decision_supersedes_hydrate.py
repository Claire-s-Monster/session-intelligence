"""Regression tests for issue #224: supersedes/rationale NULLed after hydrate + sweep.

_decision_from_row dropped both columns, so a hydrated Decision carried None and the
persist sweep's unconditional upsert overwrote the stored values with NULL.
"""

from types import SimpleNamespace

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend
from transport.http_server import HTTPSessionIntelligenceServer
from transport.persist_tracker import PersistDigestTracker

RATIONALE = "because of reasons"


def _server() -> HTTPSessionIntelligenceServer:
    server = HTTPSessionIntelligenceServer.__new__(HTTPSessionIntelligenceServer)
    server.persist_tracker = PersistDigestTracker()
    return server


async def _sweep(engine) -> None:
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(database=engine.database, session_engine=engine))
    )
    await _server()._persist_sessions_to_database(request)


async def _raw(engine, decision_id: str):
    conn = engine.database._ensure_connected()
    cursor = await conn.execute(
        "SELECT supersedes, rationale FROM decisions WHERE id = ?", (decision_id,)
    )
    row = await cursor.fetchone()
    return row[0], row[1]


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path), use_filesystem=False)
    eng.database = SQLiteBackend(str(tmp_path / "issue224.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


async def _chain(engine):
    """Log A, then B superseding A with a rationale; persist both."""
    created = await engine.session_manage_lifecycle(
        operation="create", mode="local", project_name="p224"
    )
    sid = created.session_id
    await engine.session_log_decision(decision="decA", context={"k": 1}, session_id=sid)
    id_a = engine.session_cache[sid].decisions[0].decision_id
    await engine.session_log_decision(
        decision="decB", context={"k": 2}, session_id=sid, supersedes=id_a
    )
    id_b = engine.session_cache[sid].decisions[1].decision_id
    await _sweep(engine)
    updated = await engine.session_update_decision(decision_id=id_b, rationale=RATIONALE)
    assert updated.get("status") == "success"
    await _sweep(engine)
    return sid, id_a, id_b


async def _cycle(engine, sid):
    engine.session_cache.pop(sid)
    session = await engine._hydrate_session(sid)
    await _sweep(engine)
    return session


@pytest.mark.regression
class TestRoundTrip:
    async def test_db_row_keeps_supersedes_and_rationale(self, engine):
        sid, id_a, id_b = await _chain(engine)
        assert await _raw(engine, id_b) == (id_a, RATIONALE)
        session = await _cycle(engine, sid)
        hydrated = next(d for d in session.decisions if d.decision_id == id_b)
        assert hydrated.supersedes == id_a
        assert hydrated.rationale == RATIONALE
        assert await _raw(engine, id_b) == (id_a, RATIONALE)

    async def test_superseded_decision_stays_excluded_from_recall(self, engine):
        sid, id_a, id_b = await _chain(engine)
        await _cycle(engine, sid)
        recalled = await engine.database.recall_project("p224", include=["decisions"])
        descriptions = [d["description"] for d in recalled["decisions"]]
        assert "decB" in descriptions
        assert "decA" not in descriptions


@pytest.mark.regression
class TestDecisionFromRow:
    def test_reads_supersedes_and_rationale(self, engine):
        row = {
            "id": "d1",
            "description": "x",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "impact_level": "medium",
            "artifacts": [],
            "context": {},
            "supersedes": "d0",
            "rationale": "why",
        }
        decision = engine._decision_from_row(row, "s")
        assert decision.supersedes == "d0"
        assert decision.rationale == "why"

    def test_missing_columns_are_none(self, engine):
        row = {"id": "d1", "description": "x", "timestamp": "2026-01-01T00:00:00+00:00"}
        decision = engine._decision_from_row(row, "s")
        assert decision.supersedes is None
        assert decision.rationale is None


@pytest.mark.regression
class TestUpsertKeepsSupersedes:
    async def test_none_supersedes_does_not_clobber(self, engine):
        created = await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="p224"
        )
        sid = created.session_id
        await _sweep(engine)
        base = {"decision_id": "dec-up", "session_id": sid, "description": "d"}
        await engine.database.save_decision({**base, "supersedes": "dec-old"})
        await engine.database.save_decision({**base, "supersedes": None})
        assert (await _raw(engine, "dec-up"))[0] == "dec-old"
