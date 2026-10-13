"""Regression tests for issue #226: decision artifacts were double-encoded on first write.

session_log_decision passed json.dumps(link_artifacts) to the backend, and the
PostgreSQL backend json.dumps'd it again into a JSONB string scalar, which
hydrated back as [] (silent loss). The engine now passes the raw list and
hydrate tolerates legacy double-encoded rows.
"""

import json

import pytest

from core.session_engine import SessionIntelligenceEngine, _unwrap_decision_artifacts
from persistence.sqlite import SQLiteBackend

ARTIFACTS = ["a.py", "b.py"]


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
    eng = SessionIntelligenceEngine(repository_path=str(tmp_path), use_filesystem=False)
    eng.database = SQLiteBackend(str(tmp_path / "issue226.db"))
    await eng.database.initialize()
    yield eng
    await eng.database.close()


async def _logged(engine):
    created = await engine.session_manage_lifecycle(
        operation="create", mode="local", project_name="p226"
    )
    sid = created.session_id
    await engine.session_log_decision(
        decision="d", context={"k": 1}, session_id=sid, link_artifacts=list(ARTIFACTS)
    )
    return sid


@pytest.mark.regression
class TestUnwrapDecisionArtifacts:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (["a.py"], ["a.py"]),
            (json.dumps(["a.py"]), ["a.py"]),
            (json.dumps(json.dumps(["a.py"])), ["a.py"]),
            ("not json", []),
            ('"x"', []),
            ("{}", []),
            ({"a": 1}, []),
            (None, []),
        ],
    )
    def test_unwrap(self, value, expected):
        assert _unwrap_decision_artifacts(value) == expected


@pytest.mark.regression
class TestFirstWrite:
    async def test_decision_data_artifacts_is_list(self, engine):
        captured = []
        original = engine.database.save_decision

        async def spy(decision_data):
            captured.append(decision_data)
            return await original(decision_data)

        engine.database.save_decision = spy
        await _logged(engine)
        assert captured
        assert captured[0]["artifacts"] == ARTIFACTS
        assert isinstance(captured[0]["artifacts"], list)

    async def test_round_trip_through_hydrate(self, engine):
        sid = await _logged(engine)
        engine.session_cache.pop(sid)
        session = await engine._hydrate_session(sid)
        assert session.decisions[0].artifacts == ARTIFACTS


@pytest.mark.regression
class TestLegacyRow:
    def test_double_encoded_row_hydrates(self, engine):
        row = {
            "id": "d1",
            "description": "x",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "impact_level": "medium",
            "artifacts": json.dumps(json.dumps(["a.py"])),
            "context": {},
        }
        assert engine._decision_from_row(row, "s").artifacts == ["a.py"]
