"""Regression tests for issue #144: resume must hydrate DB-only sessions.

`_resume_session` checked `session_cache` and a filesystem fallback but never
`_hydrate_session`, so after a restart (cache empty, use_filesystem=False)
resuming an existing session returned no session data, or failed outright.
"""

import uuid
from datetime import UTC, datetime

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend

SESSION_ID = "c1d2e3f4-0144-4a53-9a57-6a1c2b3d4e51"
MISSING_ID = "9d4f7a10-0144-4e0a-b1d8-3f5e6a7b8c92"
PROJECT = "resume-hydrate-project"


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issue144.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


async def _seed_session_with_history(engine: SessionIntelligenceEngine) -> None:
    now = datetime.now(UTC).isoformat()
    await engine.database.save_session(
        {
            "id": SESSION_ID,
            "started": now,
            "project_name": PROJECT,
            "project_path": "",
            "mode": "local",
            "status": "active",
            "metadata": {},
            "performance_metrics": {},
            "health_status": {},
        }
    )
    await engine.database.save_agent_execution(
        {
            "id": f"exec-{uuid.uuid4().hex[:8]}",
            "session_id": SESSION_ID,
            "agent_name": "focused-code-modifier",
            "agent_type": "focused",
            "started_at": now,
            "completed_at": now,
            "status": "success",
            "execution_steps": [],
            "performance": {},
            "errors": [],
        }
    )
    await engine.database.save_decision(
        {
            "id": f"dec-{uuid.uuid4().hex[:8]}",
            "session_id": SESSION_ID,
            "timestamp": now,
            "category": "implementation",
            "description": "seeded decision",
            "rationale": None,
            "context": {},
            "impact_level": "medium",
            "artifacts": [],
        }
    )


@pytest.mark.regression
async def test_resume_hydrates_db_only_session_history(engine):
    await _seed_session_with_history(engine)
    engine.session_cache.clear()

    result = await engine.session_manage_lifecycle(operation="resume", session_id=SESSION_ID)

    assert result.status == "success"
    assert result.session_id == SESSION_ID
    assert result.session_data is not None
    assert len(result.session_data.agents_executed) >= 1
    assert len(result.session_data.decisions) >= 1


@pytest.mark.regression
async def test_resume_unknown_explicit_id_is_error_not_raise(engine):
    result = await engine.session_manage_lifecycle(operation="resume", session_id=MISSING_ID)

    assert result.status == "error"
