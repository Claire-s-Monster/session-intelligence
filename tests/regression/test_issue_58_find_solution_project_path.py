"""
Regression tests for issue #58: session_find_solution's engine fully
implements project_path (core/session_engine.py), but the registered
schema in lean_mcp_interface.py never declared it.

Since execute_tool() validates parameters against the declared schema
before dispatch (lean_mcp_interface.py, execute_tool nested function),
an undeclared project_path is unreachable over stdio/HTTP: callers never
learn it exists (discover_tools/get_tool_spec never advertise it), so
they silently omit it and the engine falls back to a path derived from
the SERVER's own cwd instead of the caller's project.

These tests assert the SCHEMA/ENGINE agreement directly, since that is
the actual defect — not just that the engine method itself accepts the
parameter (it already did, that was never in question).
"""

from __future__ import annotations

import inspect

import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence.sqlite import SQLiteBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def lean_interface(tmp_path):
    """LeanMCPInterface backed by a fresh in-process SQLite database."""
    db = SQLiteBackend(str(tmp_path / "test.db"))
    await db.initialize()
    engine = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    interface = LeanMCPInterface(engine)
    yield interface
    await db.close()


def _get_meta_tool(interface: LeanMCPInterface, name: str):
    """Return the callable registered as a FastMCP tool by name."""
    # FastMCP stores tools in ._tool_manager._tools (dict keyed by name).
    # We access the underlying fn to call it directly in tests.
    manager = interface.app._tool_manager
    tool_obj = manager._tools[name]
    return tool_obj.fn


# ---------------------------------------------------------------------------
# Test 1: the schema declares project_path
# ---------------------------------------------------------------------------


async def test_find_solution_schema_declares_project_path(lean_interface):
    """
    session_find_solution's registered schema must declare project_path,
    or the parameter is unreachable through execute_tool()'s pre-dispatch
    validation gate — this is the exact defect reported in #58.
    """
    schema = lean_interface.tool_registry["session_find_solution"]["schema"]
    assert "project_path" in schema["properties"]


# ---------------------------------------------------------------------------
# Test 2: schema properties stay in sync with the engine signature
# ---------------------------------------------------------------------------


async def test_find_solution_schema_matches_engine_signature(lean_interface):
    """
    Generalized guard against schema/engine drift in either direction:
    every schema-declared property must be an accepted parameter of
    SessionIntelligenceEngine.session_find_solution, and project_path
    specifically must appear in both.
    """
    schema = lean_interface.tool_registry["session_find_solution"]["schema"]
    schema_params = set(schema["properties"])

    engine_params = set(
        inspect.signature(SessionIntelligenceEngine.session_find_solution).parameters
    ) - {"self"}

    undeclared_in_engine = schema_params - engine_params
    assert not undeclared_in_engine, (
        f"Schema declares parameters the engine method does not accept: "
        f"{undeclared_in_engine}"
    )

    assert "project_path" in schema_params
    assert "project_path" in engine_params


# ---------------------------------------------------------------------------
# Test 3: execute_tool dispatch actually accepts project_path
# ---------------------------------------------------------------------------


async def test_find_solution_accepts_project_path_when_dispatched(lean_interface, tmp_path):
    """
    Calling session_find_solution through the real execute_tool dispatch
    path with a project_path argument must not be rejected. Before the
    fix, execute_tool's pre-dispatch schema validation (lean_mcp_interface
    .py ~1762-1766) would reject project_path as an unexpected parameter,
    since it was absent from the declared schema.
    """
    execute = _get_meta_tool(lean_interface, "execute_tool")
    result = await execute(
        "session_find_solution",
        {
            "error_text": "ModuleNotFoundError: No module named 'foo'",
            "project_path": str(tmp_path),
        },
    )

    # Top-level rejection path: execute_tool's schema-validation gate
    # returns status="error" with an "error" message naming the
    # unexpected/unknown parameter before the engine is ever called.
    top_level_error = str(result.get("error", ""))
    assert "unexpected" not in top_level_error.lower(), (
        f"execute_tool rejected project_path before dispatch: {top_level_error}"
    )
    assert "unknown" not in top_level_error.lower(), (
        f"execute_tool rejected project_path before dispatch: {top_level_error}"
    )

    # Nested rejection path: _wrap_async_tool catches TypeErrors raised by
    # the underlying engine call (e.g. an actual unexpected-keyword-argument
    # TypeError) and reports them inside result["result"]["error"] while
    # the envelope status still reads "success". Checking only the
    # top-level status would miss this — the whole point of #58 is that a
    # dropped/rejected call can still look successful at a glance.
    inner = result.get("result")
    inner_error = ""
    if isinstance(inner, dict):
        inner_error = str(inner.get("error", ""))
    elif hasattr(inner, "error"):
        inner_error = str(inner.error or "")
    assert "unexpected keyword argument" not in inner_error
    assert "unknown" not in inner_error.lower()

    # With the schema fixed and the engine already accepting project_path,
    # dispatch should fully succeed.
    assert result.get("status") == "success"
