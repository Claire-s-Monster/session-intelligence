"""
Regression tests for issues #145 and #148 (explicit scope handling).

#145: `session_manage_lifecycle(validate, session_id=X)` discarded an explicit
id that exists in the DB but not in `session_cache` (e.g. an abandoned session)
and answered "No active session to validate" with session_id="none".

#148: `session_monitor_health` rejected a project_name-only call with
"missing required parameter(s): ['session_id']" because the registry schema
listed session_id as required.
"""

from datetime import UTC, datetime

import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.sqlite import SQLiteBackend

ABANDONED_ID = "5b0e1c5a-8f3d-4a53-9a57-6a1c2b3d4e51"
MISSING_ID = "9d4f7a10-2c6b-4e0a-b1d8-3f5e6a7b8c92"
PROJECT = "scope-project"


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    """SessionIntelligenceEngine wired to a fresh in-process SQLite backend."""
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issues145-148.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


async def _save_db_only_abandoned_session(engine: SessionIntelligenceEngine) -> None:
    await _save_db_only_session(engine, ABANDONED_ID, "abandoned")


async def _save_db_only_session(
    engine: SessionIntelligenceEngine, session_id: str, status: str
) -> None:
    await engine.database.save_session(
        {
            "id": session_id,
            "started": datetime.now(UTC).isoformat(),
            "project_name": PROJECT,
            "project_path": "",
            "mode": "local",
            "status": status,
            "metadata": {},
            "performance_metrics": {},
            "health_status": {},
        }
    )


# ---------------------------------------------------------------------------
# #145
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_validate_explicit_id_of_uncached_abandoned_session(engine):
    await _save_db_only_abandoned_session(engine)
    assert ABANDONED_ID not in engine.session_cache

    result = await engine.session_manage_lifecycle(operation="validate", session_id=ABANDONED_ID)

    assert result.status == "success"
    assert result.session_id == ABANDONED_ID
    assert result.session_data is not None
    assert str(result.session_data.status).lower().endswith("abandoned")


@pytest.mark.regression
async def test_validate_explicit_id_that_does_not_exist_names_the_id(engine):
    result = await engine.session_manage_lifecycle(operation="validate", session_id=MISSING_ID)

    assert result.status == "error"
    assert result.session_id == MISSING_ID
    assert MISSING_ID in result.message
    assert "not found" in result.message


@pytest.mark.regression
async def test_finalize_explicit_id_of_uncached_abandoned_session(engine):
    await _save_db_only_abandoned_session(engine)
    assert ABANDONED_ID not in engine.session_cache

    result = await engine.session_manage_lifecycle(operation="finalize", session_id=ABANDONED_ID)

    assert result.status == "success"
    assert result.session_id == ABANDONED_ID
    assert result.session_data is not None
    assert str(result.session_data.status).lower().endswith("completed")

    row = await engine.database.get_session(ABANDONED_ID)
    assert row is not None
    assert row["status"] == "completed"
    assert row["completed"]


@pytest.mark.regression
async def test_finalize_explicit_id_that_does_not_exist_names_the_id(engine):
    result = await engine.session_manage_lifecycle(operation="finalize", session_id=MISSING_ID)

    assert result.status == "error"
    assert result.session_id == MISSING_ID
    assert MISSING_ID in result.message
    assert "not found" in result.message


# ---------------------------------------------------------------------------
# #148
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_monitor_health_project_name_only_via_dispatch(engine):
    interface = LeanMCPInterface(engine)
    execute = interface.app._tool_manager._tools["execute_tool"].fn
    await engine.session_manage_lifecycle(operation="create", project_name=PROJECT)

    result = await execute("session_monitor_health", {"project_name": PROJECT})

    assert "missing required" not in str(result.get("error", ""))
    assert result["status"] == "success"


@pytest.mark.regression
async def test_monitor_health_engine_level_without_session_id(engine):
    created = await engine.session_manage_lifecycle(operation="create", project_name=PROJECT)

    result = await engine.session_monitor_health(project_name=PROJECT)

    assert result.session_id == created.session_id
    assert not any("Health monitoring error" in issue for issue in result.issues)


# The two tests above create the session in-process, so it is always cached.
# After a restart the cache is empty; monitor_health must hydrate from the DB
# the way validate/finalize do (#145), not report "No active session found".


@pytest.mark.regression
async def test_monitor_health_explicit_id_of_uncached_session(engine):
    await _save_db_only_abandoned_session(engine)
    assert ABANDONED_ID not in engine.session_cache

    result = await engine.session_monitor_health(session_id=ABANDONED_ID)

    assert result.session_id == ABANDONED_ID
    assert "No active session found" not in result.issues
    assert result.health_score > 0.0


@pytest.mark.regression
async def test_monitor_health_project_name_only_after_restart(engine):
    active_id = "3c7e2b90-1d4a-4f6e-8b2c-5a9d0e1f2a33"
    await _save_db_only_session(engine, active_id, "active")
    assert not engine.session_cache

    result = await engine.session_monitor_health(project_name=PROJECT)

    assert result.session_id == active_id
    assert "No active session found" not in result.issues
    assert result.health_score > 0.0


@pytest.mark.regression
async def test_monitor_health_explicit_id_that_does_not_exist(engine):
    result = await engine.session_monitor_health(session_id=MISSING_ID)

    assert result.session_id == MISSING_ID
    assert result.health_score == 0.0


# ---------------------------------------------------------------------------
# #195
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_monitor_health_skips_filesystem_checks_when_filesystem_disabled(engine):
    active_id = "7a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"
    await _save_db_only_session(engine, active_id, "active")

    result = await engine.session_monitor_health(session_id=active_id)

    assert result.health_score == 100.0
    assert result.issues == []
    assert result.recovery_actions == []


@pytest.mark.regression
async def test_monitor_health_does_not_claim_unimplemented_auto_recovery(engine):
    failed_id = "8b2c3d4e-5f6a-4b7c-9d8e-0f1a2b3c4d5e"
    await _save_db_only_session(engine, failed_id, "failed")

    result = await engine.session_monitor_health(session_id=failed_id, auto_recover=True)

    assert result.recovery_actions
    assert result.auto_recovery_attempted is False
