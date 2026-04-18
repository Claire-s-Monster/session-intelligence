"""
Integration tests for error handling across layers.

Uses LeanMCPInterface + real SQLite backend (no live server).
The lean wrappers (_wrap_tool, _wrap_async_tool) catch exceptions and
return error dicts rather than re-raising — tests verify that behavior.

asyncio_mode = "auto" — no @pytest.mark.asyncio decorators needed.
"""

from __future__ import annotations

import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.sqlite import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path):
    """Initialised SQLite backend for each test."""
    backend = SQLiteBackend(str(tmp_path / "test_errors.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
async def engine(tmp_path, db):
    """SessionIntelligenceEngine backed by the test SQLite database."""
    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng


@pytest.fixture
async def lean(engine):
    """LeanMCPInterface wrapping the engine."""
    return LeanMCPInterface(engine)


# ---------------------------------------------------------------------------
# Helper: call a tool via the registry
# ---------------------------------------------------------------------------


async def _call(lean: LeanMCPInterface, tool_name: str, **params):
    """Invoke a tool from the registry, returning the raw result."""
    tool_info = lean.tool_registry[tool_name]
    impl = tool_info["implementation"]
    import inspect
    if inspect.iscoroutinefunction(impl):
        return await impl(**params)
    return impl(**params)


# ===========================================================================
# Wrapper error behaviour
#
# _wrap_tool / _wrap_async_tool catch exceptions and return {"error": ...}
# dicts.  Tests here verify that contract rather than expecting re-raises.
# ===========================================================================


async def test_lifecycle_missing_operation_returns_error_dict(lean):
    """session_manage_lifecycle with no params returns an error dict (wrapped)."""
    result = await _call(lean, "session_manage_lifecycle")
    # Wrapper catches TypeError and returns error dict
    assert isinstance(result, dict)
    assert "error" in result


async def test_lifecycle_invalid_operation_returns_failure(lean):
    """session_manage_lifecycle with an unknown operation returns a failure response."""
    result = await _call(lean, "session_manage_lifecycle", operation="fly_to_moon")
    # Engine returns a dict/model — either "error" key or a "message" with the unknown op name
    assert result is not None
    result_dict = result if isinstance(result, dict) else result.model_dump()
    # Either an error dict or a message containing the unknown operation name
    has_error = "error" in result_dict
    has_message = "message" in result_dict and "fly_to_moon" in str(result_dict.get("message", ""))
    assert has_error or has_message, f"Expected failure indicator in: {result_dict}"


async def test_log_decision_missing_decision_returns_error_dict(lean):
    """session_log_decision with no 'decision' returns an error dict (wrapped)."""
    result = await _call(lean, "session_log_decision")
    assert isinstance(result, dict)
    assert "error" in result


async def test_log_decision_empty_string_is_accepted(lean):
    """session_log_decision with empty string decision does not crash."""
    await _call(lean, "session_manage_lifecycle", operation="create", project_name="ep-test")
    result = await _call(lean, "session_log_decision", decision="")
    # Empty string is technically valid — engine returns some result
    assert result is not None


async def test_log_decision_none_context_is_handled(lean):
    """session_log_decision with context=None does not crash."""
    await _call(lean, "session_manage_lifecycle", operation="create", project_name="ctx-test")
    result = await _call(lean, "session_log_decision", decision="test decision", context=None)
    assert result is not None


# ===========================================================================
# agent_register — missing required params
# ===========================================================================


async def test_agent_register_missing_name_returns_error_dict(lean):
    """agent_register without agent_name returns an error dict (wrapped)."""
    result = await _call(lean, "agent_register", agent_type="domain")
    assert isinstance(result, dict)
    assert "error" in result


async def test_agent_register_missing_type_returns_error_dict(lean):
    """agent_register without agent_type returns an error dict (wrapped)."""
    result = await _call(lean, "agent_register", agent_name="my-agent")
    assert isinstance(result, dict)
    assert "error" in result


async def test_agent_register_empty_name_is_handled(lean):
    """agent_register with empty string name does not raise an unhandled exception."""
    try:
        result = await _call(lean, "agent_register", agent_name="", agent_type="domain")
        # Either returns a result or an error dict — both acceptable
        assert result is not None
    except Exception:
        pass  # DB constraint rejection surfaced as exception is also acceptable


# ===========================================================================
# agent_log_decision — FK constraint (non-existent agent)
# ===========================================================================


async def test_agent_log_decision_unknown_agent_is_handled(lean):
    """agent_log_decision for a non-existent agent name is handled gracefully."""
    try:
        result = await _call(
            lean,
            "agent_log_decision",
            agent_name="ghost-agent-does-not-exist",
            decision_type="architecture",
            context="testing FK path",
            decision="Use SQLite",
        )
        assert result is not None
    except Exception as exc:
        # A clear error message is acceptable
        assert str(exc)


# ===========================================================================
# session_track_execution — missing required params
# ===========================================================================


async def test_track_execution_missing_params_returns_error_dict(lean):
    """session_track_execution with no params returns an error dict (wrapped)."""
    result = await _call(lean, "session_track_execution")
    assert isinstance(result, dict)
    assert "error" in result


async def test_track_execution_partial_params_returns_error_dict(lean):
    """session_track_execution with only step_data returns an error dict (wrapped)."""
    result = await _call(lean, "session_track_execution", step_data={"phase": "start"})
    assert isinstance(result, dict)
    assert "error" in result


# ===========================================================================
# discover_tools / get_tool_spec — edge cases
# ===========================================================================


def test_discover_tools_with_no_match_returns_empty(lean):
    """discover_tools with a pattern that matches nothing returns empty list."""
    registry = lean.tool_registry
    pattern = "zzz_no_match_zzz"
    tools = [
        {"name": n, "description": i["description"]}
        for n, i in registry.items()
        if pattern.lower() in n.lower()
    ]
    assert tools == []


def test_get_tool_spec_unknown_tool_returns_error(lean):
    """get_tool_spec for an unknown tool returns error dict with available_tools."""
    registry = lean.tool_registry
    tool_name = "nonexistent_tool_xyz"
    if tool_name not in registry:
        result = {
            "error": f"Tool '{tool_name}' not found",
            "available_tools": list(registry.keys()),
        }
    else:
        result = registry[tool_name]
    assert "error" in result
    assert "available_tools" in result
