"""
Regression tests for issue #147: session_get_dashboard was a placeholder that
ignored dashboard_type and returned empty metrics. All five views are now real,
JSON-only, and scoped like the other session_* tools.
"""

import pytest

from core.session_engine import SessionContextRequiredError, SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from models.session_models import DashboardResult, DashboardType
from persistence.sqlite import SQLiteBackend

PROJECT = "dash-project"
MISSING_ID = "9d4f7a10-2c6b-4e0a-b1d8-3f5e6a7b8c92"
DECISION_TEXT = "use sqlite for the dashboard test"
VIEWS = ["overview", "performance", "agents", "decisions", "health"]


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
    db = SQLiteBackend(db_path=str(tmp_path / "issue147.db"))
    await db.initialize()
    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


@pytest.fixture
async def seeded(engine):
    """Session with 1 successful agent execution and 1 decision."""
    created = await engine.session_manage_lifecycle(operation="create", project_name=PROJECT)
    sid = created.session_id
    for phase, extra in (("agent_start", {}), ("agent_stop", {"success": True})):
        await engine.session_track_execution(
            session_id=sid,
            agent_name="dash-agent",
            step_data={"phase": phase, "agent_type": "micro-test", **extra},
            allow_unbound=True,
        )
    await engine.session_log_decision(
        decision=DECISION_TEXT, context={"category": "test"}, session_id=sid
    )
    return sid


@pytest.mark.regression
@pytest.mark.parametrize("view", VIEWS)
async def test_type_and_session_id_echoed(engine, seeded, view):
    result = await engine.session_get_dashboard(dashboard_type=view, session_id=seeded)

    assert isinstance(result, DashboardResult)
    assert result.dashboard_type == DashboardType(view)
    assert result.session_id == seeded
    assert result.real_time_data is False
    assert result.metrics


@pytest.mark.regression
async def test_overview_counts_match_seed(engine, seeded):
    result = await engine.session_get_dashboard(session_id=seeded)

    assert result.dashboard_type == DashboardType.OVERVIEW
    assert result.metrics["agents_executed"] == 1
    assert result.metrics["decisions"] == 1


@pytest.mark.regression
async def test_agents_view_per_agent_info(engine, seeded):
    result = await engine.session_get_dashboard(dashboard_type="agents", session_id=seeded)

    assert result.metrics["agents_executed"] == 1
    assert result.metrics["status_breakdown"] == {"success": 1}
    [agent] = result.metrics["agents"]
    assert agent["agent_name"] == "dash-agent"
    assert agent["status"] == "success"


@pytest.mark.regression
async def test_decisions_view_lists_recent(engine, seeded):
    result = await engine.session_get_dashboard(dashboard_type="decisions", session_id=seeded)

    assert result.metrics["decisions"] == 1
    assert [d["description"] for d in result.metrics["recent_decisions"]] == [DECISION_TEXT]


@pytest.mark.regression
async def test_performance_view_fields(engine, seeded):
    result = await engine.session_get_dashboard(dashboard_type="performance", session_id=seeded)

    for key in (
        "agents_executed",
        "successful_executions",
        "failed_executions",
        "abandoned_executions",
        "indeterminate_executions",
        "efficiency_score",
        "average_execution_time_ms",
    ):
        assert key in result.metrics
    assert result.metrics["agents_executed"] == 1
    assert result.metrics["successful_executions"] == 1


@pytest.mark.regression
async def test_health_view_matches_monitor_health(engine, seeded):
    health = await engine.session_monitor_health(session_id=seeded)

    result = await engine.session_get_dashboard(dashboard_type="health", session_id=seeded)

    assert result.metrics["health_score"] == health.health_score
    assert result.metrics["issues"] == health.issues
    assert "recovery_actions" in result.metrics


@pytest.mark.regression
async def test_project_name_only_resolves_active_session(engine, seeded):
    result = await engine.session_get_dashboard(project_name=PROJECT)

    assert result.session_id == seeded
    assert result.metrics["agents_executed"] == 1


@pytest.mark.regression
async def test_no_scope_raises(engine, seeded):
    with pytest.raises(SessionContextRequiredError):
        await engine.session_get_dashboard()


@pytest.mark.regression
async def test_db_only_session_is_hydrated(engine, seeded):
    session = engine.session_cache[seeded]
    await engine.database.save_session(session.model_dump())
    for execution in session.agents_executed:
        data = execution.model_dump()
        data["session_id"] = seeded
        await engine.database.save_agent_execution(data)
    engine.session_cache.clear()

    result = await engine.session_get_dashboard(session_id=seeded)

    assert result.metrics["agents_executed"] == 1
    assert result.metrics["decisions"] == 1


@pytest.mark.regression
async def test_unknown_session_id_is_not_an_empty_success(engine):
    with pytest.raises(ValueError, match="not found"):
        await engine.session_get_dashboard(session_id=MISSING_ID)


@pytest.mark.regression
async def test_invalid_dashboard_type_raises(engine, seeded):
    with pytest.raises(ValueError, match="overview"):
        await engine.session_get_dashboard(dashboard_type="bogus", session_id=seeded)


@pytest.mark.regression
async def test_dispatch_health_by_project_name(engine, seeded):
    execute = LeanMCPInterface(engine).app._tool_manager._tools["execute_tool"].fn

    result = await execute(
        "session_get_dashboard", {"dashboard_type": "health", "project_name": PROJECT}
    )

    assert result["status"] == "success", result
    assert result["result"]["dashboard_type"] == "health"


@pytest.mark.regression
async def test_dispatch_rejects_removed_real_time_param(engine, seeded):
    interface = LeanMCPInterface(engine)

    error = interface.validate_tool_parameters("session_get_dashboard", {"real_time": True})

    assert error is not None
    assert "real_time" in str(error)
