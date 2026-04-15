"""
Tests for the LeanMCPInterface meta-tool dispatch layer.

Covers:
- discover_tools() — listing and filtering
- get_tool_spec() — schema retrieval, error handling
- execute_tool() — dispatch for all 27 registered tools

asyncio_mode = "auto" (from pyproject.toml) — no @pytest.mark.asyncio needed.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.sqlite import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def lean_interface(tmp_path):
    """Yield a LeanMCPInterface backed by a fresh in-process SQLite database."""
    db = SQLiteBackend(str(tmp_path / "test.db"))
    await db.initialize()
    engine = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    interface = LeanMCPInterface(engine)
    yield interface


# ---------------------------------------------------------------------------
# Helpers — access the inner functions registered on the FastMCP app
# ---------------------------------------------------------------------------


def _get_meta_tool(interface: LeanMCPInterface, name: str):
    """Return the callable registered as a FastMCP tool by name."""
    # FastMCP stores tools in ._tool_manager._tools (dict keyed by name).
    # We access the underlying fn to call it directly in tests.
    manager = interface.app._tool_manager
    tool_obj = manager._tools[name]
    return tool_obj.fn


# ---------------------------------------------------------------------------
# TestDiscoverTools
# ---------------------------------------------------------------------------


class TestDiscoverTools:
    async def test_discover_all_tools_returns_dict(self, lean_interface):
        """discover_tools('') returns a dict with required keys."""
        discover = _get_meta_tool(lean_interface, "discover_tools")
        result = discover("")
        assert isinstance(result, dict)
        assert "available_tools" in result
        assert "total_tools" in result
        assert "filtered_count" in result

    async def test_discover_all_tools_count(self, lean_interface):
        """Registry contains all registered tools and counts are consistent."""
        discover = _get_meta_tool(lean_interface, "discover_tools")
        result = discover("")
        total = result["total_tools"]
        assert total > 0
        assert result["filtered_count"] == total
        assert len(result["available_tools"]) == total

    async def test_discover_all_tools_have_name_and_description(self, lean_interface):
        """Each tool entry exposes 'name' and 'description' keys."""
        discover = _get_meta_tool(lean_interface, "discover_tools")
        result = discover("")
        for tool in result["available_tools"]:
            assert "name" in tool
            assert "description" in tool
            assert tool["name"]
            assert tool["description"]

    async def test_discover_with_pattern_filter_session(self, lean_interface):
        """Pattern 'session' returns only session-prefixed tools."""
        discover = _get_meta_tool(lean_interface, "discover_tools")
        result = discover("session")
        names = [t["name"] for t in result["available_tools"]]
        assert all("session" in n for n in names)
        assert result["filtered_count"] < result["total_tools"]

    async def test_discover_with_pattern_filter_agent(self, lean_interface):
        """Pattern 'agent' returns only agent-prefixed tools."""
        discover = _get_meta_tool(lean_interface, "discover_tools")
        result = discover("agent")
        names = [t["name"] for t in result["available_tools"]]
        assert all("agent" in n for n in names)
        assert result["filtered_count"] > 0

    async def test_discover_with_nonmatching_pattern(self, lean_interface):
        """A pattern that matches nothing returns empty list with filtered_count=0."""
        discover = _get_meta_tool(lean_interface, "discover_tools")
        total_before = discover("")["total_tools"]
        result = discover("zzz_no_such_tool_zzz")
        assert result["filtered_count"] == 0
        assert result["available_tools"] == []
        assert result["total_tools"] == total_before  # total unchanged by filter

    async def test_discover_empty_pattern_same_as_no_filter(self, lean_interface):
        """Empty-string pattern behaves identically to no filter."""
        discover = _get_meta_tool(lean_interface, "discover_tools")
        r1 = discover("")
        r2 = discover("   ")  # whitespace-only should also mean "no filter"
        assert r1["filtered_count"] == r2["filtered_count"]


# ---------------------------------------------------------------------------
# TestGetToolSpec
# ---------------------------------------------------------------------------


class TestGetToolSpec:
    async def test_valid_tool_returns_spec(self, lean_interface):
        """get_tool_spec for a known tool returns name, description, schema, examples."""
        get_spec = _get_meta_tool(lean_interface, "get_tool_spec")
        result = get_spec("session_manage_lifecycle")
        assert result["name"] == "session_manage_lifecycle"
        assert "description" in result
        assert "schema" in result
        assert "examples" in result

    async def test_valid_tool_schema_has_required(self, lean_interface):
        """Schema for session_manage_lifecycle lists 'operation' as required."""
        get_spec = _get_meta_tool(lean_interface, "get_tool_spec")
        result = get_spec("session_manage_lifecycle")
        assert "required" in result["schema"]
        assert "operation" in result["schema"]["required"]

    async def test_invalid_tool_returns_error(self, lean_interface):
        """get_tool_spec for an unknown name returns an error key."""
        get_spec = _get_meta_tool(lean_interface, "get_tool_spec")
        result = get_spec("no_such_tool_xyz")
        assert "error" in result
        assert "available_tools" in result

    async def test_get_spec_for_every_registered_tool(self, lean_interface):
        """get_tool_spec succeeds (no 'error') for all tools in the registry."""
        get_spec = _get_meta_tool(lean_interface, "get_tool_spec")
        for tool_name in lean_interface.tool_registry:
            result = get_spec(tool_name)
            assert "error" not in result, f"get_tool_spec failed for {tool_name!r}: {result}"


# ---------------------------------------------------------------------------
# TestExecuteTool — meta-dispatch
# ---------------------------------------------------------------------------


class TestExecuteToolDispatch:
    async def test_execute_invalid_tool_returns_error(self, lean_interface):
        """execute_tool with unknown tool name returns an error response with available_tools."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("no_such_tool_xyz", {})
        # The not-found path returns {"error": ..., "available_tools": [...]}
        # (no "status" key — different from the execution-error path)
        assert "error" in result
        assert "available_tools" in result

    async def test_execute_parameters_must_be_dict(self, lean_interface):
        """execute_tool rejects non-dict (non-string) parameters."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("session_manage_lifecycle", 42)  # type: ignore[arg-type]
        assert result["status"] == "error"

    async def test_execute_string_parameters_parsed_as_json(self, lean_interface):
        """execute_tool accepts a JSON string for parameters and parses it."""
        import json

        execute = _get_meta_tool(lean_interface, "execute_tool")
        params_str = json.dumps({"operation": "create"})
        result = await execute("session_manage_lifecycle", params_str)
        assert result["status"] == "success"

    async def test_tool_count_matches_registry(self, lean_interface):
        """The number of tools in the registry matches total_tools from discover."""
        discover = _get_meta_tool(lean_interface, "discover_tools")
        result = discover("")
        assert len(lean_interface.tool_registry) == result["total_tools"]


# ---------------------------------------------------------------------------
# TestExecuteTool — per-tool smoke tests
# ---------------------------------------------------------------------------


class TestExecuteSessionManageLifecycle:
    async def test_execute_session_manage_lifecycle_create(self, lean_interface):
        """execute_tool session_manage_lifecycle create returns session_id."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("session_manage_lifecycle", {"operation": "create"})
        assert result["status"] == "success"
        inner = result["result"]
        # Result is a Pydantic model dict or object — handle both
        if hasattr(inner, "session_id"):
            assert inner.session_id
        else:
            assert inner.get("session_id") or inner.get("result", {}).get("session_id")

    async def test_execute_session_manage_lifecycle_validate(self, lean_interface):
        """execute_tool session_manage_lifecycle validate runs without error."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("session_manage_lifecycle", {"operation": "validate"})
        assert result["status"] == "success"


class TestExecuteSessionTrackExecution:
    async def test_execute_session_track_execution(self, lean_interface):
        """session_track_execution succeeds with minimal params."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute(
            "session_track_execution",
            {"agent_name": "test-agent", "step_data": {"phase": "start"}},
        )
        assert result["status"] == "success"


class TestExecuteSessionCoordinateAgents:
    async def test_execute_session_coordinate_agents(self, lean_interface):
        """session_coordinate_agents succeeds with a minimal agents list."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute(
            "session_coordinate_agents",
            {"agents": [{"name": "agent-a"}]},
        )
        assert result["status"] == "success"


class TestExecuteSessionLogDecision:
    async def test_execute_session_log_decision(self, lean_interface):
        """session_log_decision succeeds with minimal params."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute(
            "session_log_decision",
            {"decision": "Use pytest for all new tests"},
        )
        assert result["status"] == "success"


class TestExecuteSessionTrackFileOperation:
    async def test_execute_session_track_file_operation(self, lean_interface):
        """session_track_file_operation succeeds with operation + file_path."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute(
            "session_track_file_operation",
            {"operation": "create", "file_path": "src/new_module.py", "lines_added": 10},
        )
        assert result["status"] == "success"


class TestExecuteSessionAnalyzePatterns:
    async def test_execute_session_analyze_patterns(self, lean_interface):
        """session_analyze_patterns succeeds with no required params."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("session_analyze_patterns", {})
        assert result["status"] == "success"


class TestExecuteSessionMonitorHealth:
    async def test_execute_session_monitor_health(self, lean_interface):
        """session_monitor_health succeeds with session_id=None (current session)."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("session_monitor_health", {"session_id": None})
        assert result["status"] == "success"


class TestExecuteSessionOrchestrateWorkflow:
    async def test_execute_session_orchestrate_workflow(self, lean_interface):
        """session_orchestrate_workflow succeeds with a valid workflow_type."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute(
            "session_orchestrate_workflow",
            {"workflow_type": "tdd"},
        )
        assert result["status"] == "success"


class TestExecuteSessionAnalyzeCommands:
    async def test_execute_session_analyze_commands(self, lean_interface):
        """session_analyze_commands succeeds with no required params."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("session_analyze_commands", {})
        assert result["status"] == "success"


class TestExecuteSessionTrackMissingFunctions:
    async def test_execute_session_track_missing_functions(self, lean_interface):
        """session_track_missing_functions succeeds with no required params."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("session_track_missing_functions", {})
        assert result["status"] == "success"


class TestExecuteSessionGetDashboard:
    async def test_execute_session_get_dashboard(self, lean_interface):
        """session_get_dashboard succeeds with no required params."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("session_get_dashboard", {})
        assert result["status"] == "success"


class TestExecuteSessionCreateNotebook:
    async def test_execute_session_create_notebook(self, lean_interface):
        """session_create_notebook succeeds with no required params."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("session_create_notebook", {})
        assert result["status"] == "success"


class TestExecuteSessionSearch:
    async def test_execute_session_search(self, lean_interface):
        """session_search succeeds with a query string."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("session_search", {"query": "test query"})
        assert result["status"] == "success"


class TestExecuteSessionQueryNotebooks:
    async def test_execute_session_query_notebooks(self, lean_interface):
        """session_query_notebooks succeeds with no required params."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("session_query_notebooks", {})
        assert result["status"] == "success"


class TestExecuteSessionRecall:
    async def test_execute_session_recall(self, lean_interface):
        """session_recall succeeds with a project_name."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("session_recall", {"project_name": "test-project"})
        assert result["status"] == "success"


class TestExecuteSessionLogLearning:
    async def test_execute_session_log_learning(self, lean_interface):
        """session_log_learning succeeds with category + learning_content."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute(
            "session_log_learning",
            {
                "category": "pattern",
                "learning_content": "Always use fixtures for shared test data.",
            },
        )
        assert result["status"] == "success"


class TestExecuteSessionFindSolution:
    async def test_execute_session_find_solution(self, lean_interface):
        """session_find_solution succeeds with an error_text."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute(
            "session_find_solution",
            {"error_text": "ModuleNotFoundError: No module named 'foo'"},
        )
        assert result["status"] == "success"


class TestExecuteSessionUpdateSolutionOutcome:
    async def test_execute_session_update_solution_outcome(self, lean_interface):
        """session_update_solution_outcome succeeds with a fake solution_id."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        # A non-existent ID is fine — the tool should not crash
        result = await execute(
            "session_update_solution_outcome",
            {"solution_id": "sol_nonexistent", "success": True},
        )
        # May succeed or return a graceful error — either way no exception
        assert result["status"] in ("success", "error")


class TestExecuteAgentRegister:
    async def test_execute_agent_register(self, lean_interface):
        """agent_register creates a new agent and returns agent_id."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute(
            "agent_register",
            {"agent_name": "test-dispatch-agent", "agent_type": "focused"},
        )
        assert result["status"] == "success"


class TestExecuteAgentGetInfo:
    async def test_execute_agent_get_info(self, lean_interface):
        """agent_get_info returns info for a registered agent."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        # Register first
        await execute(
            "agent_register",
            {"agent_name": "info-test-agent", "agent_type": "micro"},
        )
        result = await execute("agent_get_info", {"agent_name": "info-test-agent"})
        assert result["status"] == "success"

    async def test_execute_agent_get_info_unknown(self, lean_interface):
        """agent_get_info returns an error or not-found result for unknown agents."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute("agent_get_info", {"agent_name": "agent-does-not-exist"})
        # Should not raise — either success (empty) or error
        assert result["status"] in ("success", "error")


class TestExecuteAgentLogDecision:
    async def test_execute_agent_log_decision(self, lean_interface):
        """agent_log_decision succeeds after agent is registered."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        await execute(
            "agent_register",
            {"agent_name": "decision-agent", "agent_type": "focused"},
        )
        result = await execute(
            "agent_log_decision",
            {
                "agent_name": "decision-agent",
                "decision_type": "tool_selection",
                "context": "Multiple lint errors in file",
                "decision": "Use ruff --fix",
            },
        )
        assert result["status"] == "success"


class TestExecuteAgentQueryDecisions:
    async def test_execute_agent_query_decisions(self, lean_interface):
        """agent_query_decisions returns a list for a registered agent."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        await execute(
            "agent_register",
            {"agent_name": "query-dec-agent", "agent_type": "focused"},
        )
        result = await execute("agent_query_decisions", {"agent_name": "query-dec-agent"})
        assert result["status"] == "success"


class TestExecuteAgentUpdateDecisionOutcome:
    async def test_execute_agent_update_decision_outcome(self, lean_interface):
        """agent_update_decision_outcome handles unknown decision_id gracefully."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute(
            "agent_update_decision_outcome",
            {"decision_id": "dec_nonexistent", "outcome": "worked", "success": True},
        )
        assert result["status"] in ("success", "error")


class TestExecuteAgentLogLearning:
    async def test_execute_agent_log_learning(self, lean_interface):
        """agent_log_learning succeeds after agent is registered."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        await execute(
            "agent_register",
            {"agent_name": "learning-agent", "agent_type": "focused"},
        )
        result = await execute(
            "agent_log_learning",
            {
                "agent_name": "learning-agent",
                "learning_type": "pattern",
                "title": "Ruff handles import sorting",
                "content": "Ruff with isort rules can replace separate isort step.",
            },
        )
        assert result["status"] == "success"


class TestExecuteAgentQueryLearnings:
    async def test_execute_agent_query_learnings(self, lean_interface):
        """agent_query_learnings returns a list for a registered agent."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        await execute(
            "agent_register",
            {"agent_name": "query-lrn-agent", "agent_type": "focused"},
        )
        result = await execute("agent_query_learnings", {"agent_name": "query-lrn-agent"})
        assert result["status"] == "success"


class TestExecuteAgentUpdateLearningOutcome:
    async def test_execute_agent_update_learning_outcome(self, lean_interface):
        """agent_update_learning_outcome handles unknown learning_id gracefully."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        result = await execute(
            "agent_update_learning_outcome",
            {"learning_id": "lrn_nonexistent", "times_applied_increment": 1},
        )
        assert result["status"] in ("success", "error")


class TestExecuteAgentCreateNotebook:
    async def test_execute_agent_create_notebook(self, lean_interface):
        """agent_create_notebook succeeds after agent is registered."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        await execute(
            "agent_register",
            {"agent_name": "notebook-agent", "agent_type": "focused"},
        )
        result = await execute(
            "agent_create_notebook",
            {
                "agent_name": "notebook-agent",
                "title": "Test Notebook",
                "content": "## Summary\n\nThis is a test notebook.",
            },
        )
        assert result["status"] == "success"


class TestExecuteAgentQueryNotebooks:
    async def test_execute_agent_query_notebooks(self, lean_interface):
        """agent_query_notebooks returns results for a registered agent."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        await execute(
            "agent_register",
            {"agent_name": "query-nb-agent", "agent_type": "focused"},
        )
        result = await execute("agent_query_notebooks", {"agent_name": "query-nb-agent"})
        assert result["status"] == "success"


class TestExecuteAgentSearchAll:
    async def test_execute_agent_search_all(self, lean_interface):
        """agent_search_all returns results for a registered agent."""
        execute = _get_meta_tool(lean_interface, "execute_tool")
        await execute(
            "agent_register",
            {"agent_name": "search-all-agent", "agent_type": "focused"},
        )
        result = await execute(
            "agent_search_all",
            {"agent_name": "search-all-agent", "query": "ruff lint"},
        )
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# TestAllAsyncToolsProperlyWrapped — structural integrity
# ---------------------------------------------------------------------------


class TestToolWrapperIntegrity:
    async def test_all_async_tools_are_coroutine_functions(self, lean_interface):
        """
        All tools wrapped with _wrap_async_tool must be awaitable.

        Tools wrapped with _wrap_tool (sync) are intentionally excluded.
        The execute_tool dispatcher uses inspect.iscoroutinefunction to choose
        await vs direct call, so the wrapper type must match the underlying
        engine method's async-ness.
        """
        import inspect

        sync_tools = {
            "session_track_execution",
            "session_coordinate_agents",
            "session_analyze_patterns",
            "session_monitor_health",
            "session_orchestrate_workflow",
            "session_analyze_commands",
            "session_track_missing_functions",
            "session_get_dashboard",
        }

        for tool_name, tool_info in lean_interface.tool_registry.items():
            impl = tool_info["implementation"]
            if tool_name in sync_tools:
                assert not inspect.iscoroutinefunction(impl), (
                    f"{tool_name} should be sync-wrapped but is a coroutinefunction"
                )
            else:
                assert inspect.iscoroutinefunction(impl), (
                    f"{tool_name} should be async-wrapped but is not a coroutinefunction"
                )
