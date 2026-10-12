"""Regression tests for issue #217: decisions.context nested itself every cycle.

Canonical stored shape is the flat caller dict (== DecisionContext.project_state).
The sweep used to store the whole DecisionContext dump, and hydrate wrapped the
stored dict as project_state again, adding one layer per persist/hydrate cycle.
"""

import json
from types import SimpleNamespace

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend
from transport.http_server import HTTPSessionIntelligenceServer
from transport.persist_tracker import PersistDigestTracker

CALLER_CONTEXT = {"category": "test", "tags": ["a", "b"], "nested": {"k": 1}}


def _server() -> HTTPSessionIntelligenceServer:
    server = HTTPSessionIntelligenceServer.__new__(HTTPSessionIntelligenceServer)
    server.persist_tracker = PersistDigestTracker()
    return server


async def _sweep(engine) -> None:
    # A fresh tracker per sweep forces every entity to be re-written.
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(database=engine.database, session_engine=engine))
    )
    await _server()._persist_sessions_to_database(request)


async def _raw_context(engine, decision_id: str):
    conn = engine.database._ensure_connected()
    cursor = await conn.execute("SELECT context FROM decisions WHERE id = ?", (decision_id,))
    row = await cursor.fetchone()
    return row[0]


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path), use_filesystem=False)
    eng.database = SQLiteBackend(str(tmp_path / "issue217.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


async def _logged(engine, context=None):
    created = await engine.session_manage_lifecycle(
        operation="create", mode="local", project_name="p217"
    )
    sid = created.session_id
    await engine.session_log_decision(
        decision="d", context=dict(context or CALLER_CONTEXT), session_id=sid
    )
    decision_id = engine.session_cache[sid].decisions[0].decision_id
    return sid, decision_id


@pytest.mark.regression
class TestRoundTrip:
    async def test_three_cycles_stay_flat(self, engine):
        sid, did = await _logged(engine)
        for _ in range(3):
            await _sweep(engine)
            engine.session_cache.pop(sid)
            session = await engine._hydrate_session(sid)
            assert session.decisions[0].context.project_state == CALLER_CONTEXT
        await _sweep(engine)
        assert session.decisions[0].context.project_state == CALLER_CONTEXT

    async def test_stored_column_is_flat_dict_not_string(self, engine):
        sid, did = await _logged(engine)
        assert json.loads(await _raw_context(engine, did)) == CALLER_CONTEXT
        await _sweep(engine)
        raw = json.loads(await _raw_context(engine, did))
        assert isinstance(raw, dict)
        assert raw == CALLER_CONTEXT


@pytest.mark.regression
class TestUnwrapOnHydrate:
    def _wrap(self, inner: dict, sid: str = "s") -> dict:
        return {
            "session_id": sid,
            "agent_name": None,
            "workflow_phase": None,
            "project_state": inner,
            "available_options": [],
            "constraints": [],
        }

    def _row(self, context):
        return {
            "id": "d1",
            "description": "x",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "impact_level": "medium",
            "artifacts": [],
            "context": context,
        }

    def test_two_level_nested_legacy_row_unwraps(self, engine):
        nested = self._wrap(self._wrap(CALLER_CONTEXT))
        decision = engine._decision_from_row(self._row(nested), "s")
        assert decision.context.project_state == CALLER_CONTEXT

    def test_two_level_nested_json_text_unwraps(self, engine):
        nested = self._wrap(self._wrap(CALLER_CONTEXT))
        decision = engine._decision_from_row(self._row(json.dumps(nested)), "s")
        assert decision.context.project_state == CALLER_CONTEXT

    def test_double_encoded_string_decodes(self, engine):
        # PG JSONB string scalar: asyncpg hands back '"{...}"'.
        row = self._row(json.dumps(json.dumps(CALLER_CONTEXT)))
        decision = engine._decision_from_row(row, "s")
        assert decision.context.project_state == CALLER_CONTEXT

    def test_double_encoded_wrapped_string_decodes(self, engine):
        # json-text of json-text: _json_or_default yields a str, helper decodes again.
        row = self._row(json.dumps(json.dumps(self._wrap(CALLER_CONTEXT))))
        decision = engine._decision_from_row(row, "s")
        assert decision.context.project_state == CALLER_CONTEXT

    def test_caller_dict_with_project_state_key_preserved(self, engine):
        ctx = {"project_state": {"a": 1}, "session_id": "mine", "note": "unrelated"}
        decision = engine._decision_from_row(self._row(ctx), "s")
        assert decision.context.project_state == ctx

    def test_flat_legacy_row_unchanged(self, engine):
        decision = engine._decision_from_row(self._row(dict(CALLER_CONTEXT)), "s")
        assert decision.context.project_state == CALLER_CONTEXT
