"""
Data builder functions for persistence-layer tests.

Each builder returns a dict with sensible defaults.  Any field can be
overridden by passing it as a keyword argument.

Usage::

    session = make_session_data()
    session_custom = make_session_data(session_type="live", tags=["prod"])
"""

import uuid
from datetime import UTC, datetime


def _uid() -> str:
    """Return an 8-character hex unique ID."""
    return uuid.uuid4().hex[:8]


def _now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Core session entities
# ---------------------------------------------------------------------------


def make_session_data(**overrides) -> dict:
    """Build a minimal valid session data dict."""
    uid = _uid()
    defaults = {
        "session_id": f"session-{uid}",
        "session_type": "development",
        "start_time": _now(),
        "end_time": None,
        "status": "active",
        "project_path": f"/tmp/project-{uid}",
        "git_branch": "main",
        "git_commit": uid * 5,  # 40-char hex
        "tags": [],
        "metadata": {},
    }
    return {**defaults, **overrides}


def make_decision_data(**overrides) -> dict:
    """Build a minimal valid session decision data dict."""
    defaults = {
        "decision_id": f"dec-{_uid()}",
        "session_id": f"session-{_uid()}",
        "decision_type": "tool_selection",
        "context": "Test decision context",
        "decision": "Use pytest",
        "reasoning": "Standard Python testing framework",
        "alternatives": ["unittest"],
        "confidence": 0.9,
        "outcome": None,
        "outcome_success": None,
        "tags": ["testing"],
        "created_at": _now(),
    }
    return {**defaults, **overrides}


def make_metrics_data(**overrides) -> dict:
    """Build a minimal valid session metrics data dict."""
    defaults = {
        "metrics_id": f"metrics-{_uid()}",
        "session_id": f"session-{_uid()}",
        "total_tokens": 1000,
        "input_tokens": 600,
        "output_tokens": 400,
        "total_cost": 0.01,
        "tools_used": {"read": 3, "write": 1},
        "files_modified": 2,
        "errors_encountered": 0,
        "recorded_at": _now(),
    }
    return {**defaults, **overrides}


def make_note_data(**overrides) -> dict:
    """Build a minimal valid session note data dict."""
    defaults = {
        "note_id": f"note-{_uid()}",
        "session_id": f"session-{_uid()}",
        "content": "Test note content",
        "note_type": "general",
        "tags": [],
        "created_at": _now(),
    }
    return {**defaults, **overrides}


def make_file_operation_data(**overrides) -> dict:
    """Build a minimal valid file operation data dict."""
    defaults = {
        "operation_id": f"op-{_uid()}",
        "session_id": f"session-{_uid()}",
        "file_path": f"/tmp/test-{_uid()}.py",
        "operation_type": "write",
        "lines_changed": 10,
        "timestamp": _now(),
    }
    return {**defaults, **overrides}


# ---------------------------------------------------------------------------
# Agent entities
# ---------------------------------------------------------------------------


def make_agent_data(**overrides) -> dict:
    """Build a minimal valid agent registry data dict."""
    uid = _uid()
    defaults = {
        "id": str(uuid.uuid4()),
        "name": f"test-agent-{uid}",
        "agent_type": "focused",
        "display_name": f"Test Agent {uid}",
        "description": "An agent created by the test builder",
        "capabilities": ["test", "validate"],
        "metadata": {"version": "1.0.0"},
        "first_seen_at": _now(),
        "last_active_at": _now(),
        "total_executions": 0,
        "total_decisions": 0,
        "total_learnings": 0,
        "total_notebooks": 0,
        "is_active": True,
    }
    return {**defaults, **overrides}


def make_agent_decision_data(**overrides) -> dict:
    """Build a minimal valid agent decision data dict."""
    defaults = {
        "id": str(uuid.uuid4()),
        "agent_id": str(uuid.uuid4()),
        "decision_type": "architecture",
        "context": "Test decision context",
        "decision": "Use hexagonal architecture",
        "reasoning": "Separation of concerns",
        "alternatives": ["layered", "microservices"],
        "confidence": 0.85,
        "outcome": None,
        "outcome_success": None,
        "tags": ["architecture"],
        "created_at": _now(),
        "updated_at": _now(),
    }
    return {**defaults, **overrides}


def make_agent_learning_data(**overrides) -> dict:
    """Build a minimal valid agent learning data dict."""
    defaults = {
        "id": str(uuid.uuid4()),
        "agent_id": str(uuid.uuid4()),
        "learning_type": "pattern",
        "title": "Test Learning Pattern",
        "content": "Always validate inputs before processing",
        "source_context": "Code review session",
        "applicability": ["validation", "input-handling"],
        "confidence": 0.9,
        "times_applied": 0,
        "success_rate": 0.0,
        "tags": ["best-practice"],
        "created_at": _now(),
        "updated_at": _now(),
    }
    return {**defaults, **overrides}


def make_agent_notebook_data(**overrides) -> dict:
    """Build a minimal valid agent notebook data dict."""
    defaults = {
        "id": str(uuid.uuid4()),
        "agent_id": str(uuid.uuid4()),
        "title": "Test Notebook",
        "summary": "A notebook created by the test builder",
        "content": "# Test Notebook\n\nSome content here.",
        "notebook_type": "execution",
        "context": {},
        "decisions_referenced": [],
        "learnings_referenced": [],
        "tags": ["test"],
        "created_at": _now(),
        "updated_at": _now(),
    }
    return {**defaults, **overrides}


# ---------------------------------------------------------------------------
# MCP / workflow entities
# ---------------------------------------------------------------------------


def make_mcp_session_data(**overrides) -> dict:
    """Build a minimal valid MCP session data dict."""
    uid = _uid()
    defaults = {
        "mcp_session_id": f"mcp-{uid}",
        "engine_session_id": None,
        "created_at": _now(),
        "last_activity": _now(),
        "client_info": {},
    }
    return {**defaults, **overrides}


def make_summary_data(**overrides) -> dict:
    """Build a minimal valid session summary data dict."""
    defaults = {
        "summary_id": f"sum-{_uid()}",
        "session_id": f"session-{_uid()}",
        "summary_type": "end_of_session",
        "content": "Session completed successfully.",
        "key_decisions": [],
        "key_learnings": [],
        "files_modified": [],
        "created_at": _now(),
    }
    return {**defaults, **overrides}


def make_agent_execution_data(**overrides) -> dict:
    """Build a minimal valid agent execution data dict."""
    defaults = {
        "execution_id": f"exec-{_uid()}",
        "agent_id": str(uuid.uuid4()),
        "session_id": f"session-{_uid()}",
        "phase": "start",
        "command": "pytest tests/",
        "status": "running",
        "started_at": _now(),
        "ended_at": None,
        "metadata": {},
    }
    return {**defaults, **overrides}


def make_project_learning_data(**overrides) -> dict:
    """Build a minimal valid project learning data dict."""
    defaults = {
        "learning_id": f"pl-{_uid()}",
        "project_path": f"/tmp/project-{_uid()}",
        "category": "pattern",
        "learning_content": "Use fixtures for shared test state",
        "trigger_context": "Writing unit tests",
        "confidence": 0.8,
        "times_applied": 0,
        "created_at": _now(),
    }
    return {**defaults, **overrides}


def make_error_solution_data(**overrides) -> dict:
    """Build a minimal valid error/solution data dict."""
    defaults = {
        "solution_id": f"sol-{_uid()}",
        "error_pattern": "ImportError: No module named 'foo'",
        "solution": "Run `pip install foo`",
        "context": "Missing dependency",
        "tags": ["dependency", "import"],
        "success_count": 1,
        "created_at": _now(),
    }
    return {**defaults, **overrides}
