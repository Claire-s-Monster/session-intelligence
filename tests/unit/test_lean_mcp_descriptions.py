"""Regression test for tool description framing.

Each registered tool's description must communicate: which system, which
data tier, and (for write tools) anti-dupe guidance. This test catches
description rewrites that drop the framing model.
"""

from unittest.mock import MagicMock

import pytest

from lean_mcp_interface import LeanMCPInterface


@pytest.fixture
def interface():
    """Build interface with mock engine."""
    engine = MagicMock()
    return LeanMCPInterface(engine)


SESSION_TOOLS = {
    "session_log_decision",
    "session_log_learning",
    "session_create_notebook",
    "session_query_notebooks",
    "session_recall",
    "session_search",
}

AGENT_TOOLS = {
    "agent_register",
    "agent_log_decision",
    "agent_log_learning",
    "agent_create_notebook",
    "agent_query_decisions",
    "agent_query_learnings",
    "agent_query_notebooks",
    "agent_search_all",
}

WRITE_TOOLS = {
    "session_log_decision",
    "session_log_learning",
    "session_create_notebook",
    "agent_register",
    "agent_log_decision",
    "agent_log_learning",
    "agent_create_notebook",
}


@pytest.mark.parametrize("tool_name", sorted(SESSION_TOOLS))
def test_session_tool_description_mentions_system(interface, tool_name):
    desc = interface.tool_registry[tool_name]["description"]
    assert "SYSTEM" in desc and "session" in desc.lower(), (
        f"{tool_name} description missing session-system framing"
    )


@pytest.mark.parametrize("tool_name", sorted(AGENT_TOOLS))
def test_agent_tool_description_mentions_system(interface, tool_name):
    desc = interface.tool_registry[tool_name]["description"]
    assert "SYSTEM" in desc and "agent" in desc.lower(), (
        f"{tool_name} description missing agent-system framing"
    )


# agent_register is a registry-identity operation; it writes to the agent
# registry itself (not a data tier), so TIER/READS are not expected there.
TIER_EXEMPT_TOOLS = {"agent_register"}


@pytest.mark.parametrize("tool_name", sorted((SESSION_TOOLS | AGENT_TOOLS) - TIER_EXEMPT_TOOLS))
def test_tool_description_mentions_tier(interface, tool_name):
    desc = interface.tool_registry[tool_name]["description"]
    assert "TIER" in desc or "READS" in desc, (
        f"{tool_name} description missing TIER or READS tag"
    )


@pytest.mark.parametrize("tool_name", sorted(WRITE_TOOLS))
def test_write_tool_description_includes_anti_dupe_hint(interface, tool_name):
    desc = interface.tool_registry[tool_name]["description"]
    desc_lower = desc.lower()
    # Accept any of: explicit ANTI-DUPE tag, "query" (query-first protocol),
    # or "check if already" (registry idempotency wording used by agent_register).
    assert (
        "ANTI-DUPE" in desc
        or "anti-dupe" in desc_lower
        or "query" in desc_lower
        or "check if already" in desc_lower
    ), f"{tool_name} (write) description missing anti-dupe / query-first hint"


@pytest.mark.parametrize(
    "tool_name",
    sorted({"session_log_decision", "session_log_learning", "session_create_notebook"}),
)
def test_session_write_tool_mentions_project_scope_discipline(interface, tool_name):
    """Session write tools must mention project scoping (project_name or project_path)."""
    desc = interface.tool_registry[tool_name]["description"]
    # Accept either spelling — both express the same bind-to-project discipline
    assert "project_name" in desc or "project_path" in desc or "project" in desc.lower(), (
        f"{tool_name} description missing project-scope discipline reminder"
    )


@pytest.mark.parametrize(
    "tool_name",
    sorted({"agent_register", "agent_log_decision", "agent_log_learning", "agent_create_notebook"}),
)
def test_agent_write_tool_mentions_validation(interface, tool_name):
    desc = interface.tool_registry[tool_name]["description"]
    assert (
        "~/.claude/agents" in desc
        or "AgentNotFoundError" in desc
        or "validated" in desc.lower()
    ), f"{tool_name} description missing agent-name validation reminder"


def test_discover_tools_includes_framework_guide_on_bare_call(interface):
    result = interface._discover_tools(pattern="")
    assert "_framework_guide" in result, "discover_tools() bare call missing _framework_guide"
    guide = result["_framework_guide"]
    assert "systems" in guide
    assert "data_tiers" in guide
    assert "anti_dupe_protocol" in guide
    assert "project_name_discipline" in guide
    assert "session_*" in guide["systems"]
    assert "agent_*" in guide["systems"]
    assert "decision" in guide["data_tiers"]
    assert "learning" in guide["data_tiers"]
    assert "notebook" in guide["data_tiers"]


def test_discover_tools_omits_framework_guide_on_filtered_call(interface):
    result = interface._discover_tools(pattern="agent")
    assert "_framework_guide" not in result, (
        "_framework_guide should be absent when caller filters by pattern"
    )


def test_write_tool_examples_include_workflow_hint(interface):
    """Write tools should have at least one example with _workflow_hint."""
    for tool_name in sorted(WRITE_TOOLS):
        examples = interface.tool_registry[tool_name].get("examples", [])
        has_hint = any(
            "_workflow_hint" in (ex if isinstance(ex, dict) else {}) for ex in examples
        )
        assert has_hint, f"{tool_name} examples missing _workflow_hint workflow guidance"


def test_notebook_type_description_distinguishes_entity_from_narrative(interface):
    """The notebook_type field for agent_create_notebook should clarify it's not the same
    as the 'learning' entity."""
    schema = interface.tool_registry["agent_create_notebook"]["schema"]
    nt_desc = schema["properties"]["notebook_type"]["description"]
    assert "narrative" in nt_desc.lower() or "entity" in nt_desc.lower(), (
        f"notebook_type description should clarify entity-vs-narrative distinction; got: {nt_desc}"
    )
