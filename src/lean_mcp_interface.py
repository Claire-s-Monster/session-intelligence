"""
Lean MCP Interface with Dynamic Tool Discovery.

This module implements the meta-tool pattern to reduce context consumption
from 20-50K tokens down to minimal context while maintaining full functionality.

The Problem:
- Traditional MCP servers expose 10-50 verbose tool definitions
- This saturates agent context before any real work begins
- With 10+ MCP servers, agents hit context limits from tool definitions alone

Solution: Meta-Tool Pattern
- Expose only 3 standard meta-tools with minimal definitions
- Tools are discovered dynamically on-demand
- Full schemas retrieved only when needed
- Zero functionality loss with massive context savings
"""

import json
import logging
from functools import wraps
from typing import Any

from fastmcp import FastMCP

from core.session_engine import SessionIntelligenceEngine
from models.session_models import *  # noqa: F403
from utils.token_limiter import apply_token_limits

logger = logging.getLogger(__name__)


class LeanMCPInterface:
    """
    Lean MCP Interface implementing the meta-tool pattern for dynamic tool discovery.

    Instead of exposing 10+ verbose tool definitions (20-50K tokens),
    exposes only 3 compact meta-tools (~500 tokens) with on-demand discovery.
    """

    def __init__(self, session_engine: SessionIntelligenceEngine):
        self.session_engine = session_engine
        self.app = FastMCP("session-intelligence-lean")

        # Tool registry: maps tool names to their implementations and metadata
        self.tool_registry = self._build_tool_registry()

        # Setup the 3 meta-tools
        self._setup_meta_tools()

    def _build_tool_registry(self) -> dict[str, dict[str, Any]]:
        """
        Build comprehensive tool registry with metadata for dynamic discovery.

        Each tool entry contains:
        - implementation: The actual function
        - schema: Full parameter schema
        - domain: Tool domain (session, workflow, analytics, etc.)
        - complexity: Tool complexity (micro, focused, comprehensive)
        - description: Brief description
        - examples: Usage examples
        """
        registry = {}

        # Session Management Tools
        registry["session_manage_lifecycle"] = {
            "implementation": self._wrap_async_tool(self.session_engine.session_manage_lifecycle),
            "description": "Complete session lifecycle management with recovery",
            "schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "resume", "finalize", "validate"],
                        "description": "Lifecycle operation to perform",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["local", "remote", "hybrid", "auto"],
                        "default": "local",
                        "description": "Session mode",
                    },
                    "project_name": {"type": "string", "description": "Project context (optional)"},
                    "metadata": {"description": "Additional session metadata"},
                    "auto_recovery": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable automatic recovery",
                    },
                },
                "required": ["operation"],
            },
            "examples": [
                {"operation": "create", "project_name": "my-project"},
                {"operation": "resume", "mode": "hybrid"},
            ],
        }

        registry["session_track_execution"] = {
            "implementation": self._wrap_tool(self.session_engine.session_track_execution),
            "description": "Track agent execution with pattern detection",
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "Agent being executed"},
                    "step_data": {"type": "object", "description": "ExecutionStep details"},
                    "session_id": {"type": "string", "description": "Session ID (optional)"},
                    "track_patterns": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable pattern detection",
                    },
                    "suggest_optimizations": {
                        "type": "boolean",
                        "default": True,
                        "description": "Generate optimization suggestions",
                    },
                },
                "required": ["agent_name", "step_data"],
            },
            "examples": [
                {"agent_name": "test-runner", "step_data": {"phase": "start", "command": "pytest"}}
            ],
        }

        registry["session_coordinate_agents"] = {
            "implementation": self._wrap_tool(self.session_engine.session_coordinate_agents),
            "description": "Multi-agent coordination with dependency management",
            "schema": {
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Agents to coordinate",
                    },
                    "session_id": {"type": "string", "description": "Session context"},
                    "execution_mode": {
                        "type": "string",
                        "enum": ["sequential", "parallel", "adaptive"],
                        "default": "sequential",
                        "description": "Execution strategy",
                    },
                    "dependency_graph": {"type": "object", "description": "Agent dependencies"},
                    "optimization_level": {
                        "type": "string",
                        "enum": ["conservative", "balanced", "aggressive"],
                        "default": "balanced",
                        "description": "Optimization approach",
                    },
                },
                "required": ["agents"],
            },
            "examples": [
                {
                    "agents": [{"name": "quality-check"}, {"name": "test-runner"}],
                    "execution_mode": "parallel",
                }
            ],
        }

        registry["session_log_decision"] = {
            "implementation": self._wrap_async_tool(self.session_engine.session_log_decision),
            "description": (
                "Log a decision made on this project. "
                "**SYSTEM**: session — project-scoped work history bound to project_name. "
                "**TIER**: decision — atomic choice (what was decided + reasoning). "
                "**ANTI-DUPE**: call session_recall(project_name=X) or "
                "session_search(query=Y) FIRST to find existing decisions covering the "
                "same ground; extend or supersede rather than re-log. "
                "**DISCIPLINE**: pass project_name explicitly — never let it fall back "
                "to _unbound_."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "description": "Decision description"},
                    "session_id": {"type": "string", "description": "Session context"},
                    "context": {"type": "object", "description": "Decision context and rationale"},
                    "impact_analysis": {
                        "type": "boolean",
                        "default": True,
                        "description": "Analyze decision impact",
                    },
                    "link_artifacts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Related files or commits",
                    },
                    "project_name": {
                        "type": "string",
                        "description": (
                            "Project name to bind decision to — pass EXPLICITLY. "
                            "Creates/selects a session with this project_name if needed. "
                            "Omitting silently falls back to _unbound_ and corrupts retrieval."
                        ),
                    },
                    "session_name": {
                        "type": "string",
                        "description": (
                            "Human-readable session label. Resolved to a session ID; "
                            "creates a new session if not found (and create_if_missing=True)."
                        ),
                    },
                    "project_path": {
                        "type": "string",
                        "description": (
                            "Absolute path to the caller's project. When no session_id, "
                            "session_name, or project_name is given, project_name is "
                            "derived from this path. Relative paths are ignored (they "
                            "would resolve against the server's cwd, not the caller's)."
                        ),
                    },
                    "allow_unbound": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "If True, opt into the legacy '_unbound_' fallback when no "
                            "session identifier is provided. Deprecated — pass project_name "
                            "or session_name explicitly instead."
                        ),
                    },
                },
                "required": ["decision"],
            },
            "examples": [
                {
                    "_workflow_hint": "STEP 1: query first to dedupe",
                    "project_name": "my-project",
                    "decision": "session_recall or session_search goes here first",
                },
                {
                    "_workflow_hint": "STEP 2: log only if no matching decision found above",
                    "decision": "Switch to pytest for testing",
                    "context": {"reason": "Better async support"},
                    "project_name": "my-project",
                },
            ],
        }

        registry["session_track_file_operation"] = {
            "implementation": self._wrap_async_tool(
                self.session_engine.session_track_file_operation
            ),
            "description": "Track file create/edit/delete operations for session notebook",
            "schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "edit", "delete", "read"],
                        "description": "File operation type",
                    },
                    "file_path": {"type": "string", "description": "Path to the file"},
                    "lines_added": {
                        "type": "integer",
                        "default": 0,
                        "description": "Number of lines added",
                    },
                    "lines_removed": {
                        "type": "integer",
                        "default": 0,
                        "description": "Number of lines removed",
                    },
                    "summary": {"type": "string", "description": "Brief description of changes"},
                    "tool_name": {"type": "string", "description": "Tool that made the change"},
                },
                "required": ["operation", "file_path"],
            },
            "examples": [{"operation": "create", "file_path": "src/module.py", "lines_added": 150}],
        }

        registry["session_analyze_patterns"] = {
            "implementation": self._wrap_tool(self.session_engine.session_analyze_patterns),
            "description": "Cross-session pattern analysis with ML insights",
            "schema": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["current", "recent", "historical", "all"],
                        "default": "current",
                        "description": "Analysis scope",
                    },
                    "pattern_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Patterns to analyze",
                    },
                    "include_agents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific agents to analyze",
                    },
                    "learning_mode": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable ML-based learning",
                    },
                    "generate_insights": {
                        "type": "boolean",
                        "default": True,
                        "description": "Generate actionable insights",
                    },
                },
            },
            "examples": [{"scope": "recent", "pattern_types": ["execution", "errors"]}],
        }

        registry["session_monitor_health"] = {
            "implementation": self._wrap_tool(self.session_engine.session_monitor_health),
            "description": "Real-time session health monitoring with auto-recovery",
            "schema": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": ["string", "null"],
                        "description": "Session to monitor (use null for current session)",
                    },
                    "health_checks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Checks to perform",
                    },
                    "auto_recover": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable automatic recovery",
                    },
                    "alert_thresholds": {
                        "type": "object",
                        "description": "Custom alert thresholds",
                    },
                    "include_diagnostics": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed diagnostics",
                    },
                },
                "required": ["session_id"],
            },
            "examples": [
                {"session_id": "session-123", "health_checks": ["continuity", "files"]},
                {"session_id": None, "auto_recover": True},
            ],
        }

        registry["session_orchestrate_workflow"] = {
            "implementation": self._wrap_tool(self.session_engine.session_orchestrate_workflow),
            "description": "Advanced workflow orchestration with optimization",
            "schema": {
                "type": "object",
                "properties": {
                    "workflow_type": {
                        "type": "string",
                        "enum": ["tdd", "atomic", "quality", "prime", "custom"],
                        "description": "Workflow type",
                    },
                    "session_id": {"type": "string", "description": "Session context"},
                    "workflow_config": {"type": "object", "description": "Workflow configuration"},
                    "parallel_execution": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable parallel execution",
                    },
                    "optimize_execution": {
                        "type": "boolean",
                        "default": True,
                        "description": "Optimize execution order",
                    },
                },
                "required": ["workflow_type"],
            },
            "examples": [{"workflow_type": "tdd", "parallel_execution": True}],
        }

        registry["session_analyze_commands"] = {
            "implementation": self._wrap_tool(self.session_engine.session_analyze_commands),
            "description": "Analyze hook-based commands for inefficiencies",
            "schema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session to analyze"},
                    "command_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command categories",
                    },
                    "detect_inefficiencies": {
                        "type": "boolean",
                        "default": True,
                        "description": "Detect inefficient patterns",
                    },
                    "suggest_alternatives": {
                        "type": "boolean",
                        "default": True,
                        "description": "Suggest better approaches",
                    },
                    "include_timing": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include timing analysis",
                    },
                },
            },
            "examples": [{"command_types": ["git", "test"], "detect_inefficiencies": True}],
        }

        registry["session_track_missing_functions"] = {
            "implementation": self._wrap_tool(self.session_engine.session_track_missing_functions),
            "description": "Track missing functions for ecosystem improvement",
            "schema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session context"},
                    "auto_suggest": {
                        "type": "boolean",
                        "default": True,
                        "description": "Suggest function implementations",
                    },
                    "priority_analysis": {
                        "type": "boolean",
                        "default": True,
                        "description": "Analyze implementation priority",
                    },
                    "generate_report": {
                        "type": "boolean",
                        "default": True,
                        "description": "Generate missing function report",
                    },
                },
            },
            "examples": [{"auto_suggest": True, "priority_analysis": True}],
        }

        registry["session_get_dashboard"] = {
            "implementation": self._wrap_tool(self.session_engine.session_get_dashboard),
            "description": "Comprehensive intelligence dashboard with real-time insights",
            "schema": {
                "type": "object",
                "properties": {
                    "dashboard_type": {
                        "type": "string",
                        "enum": ["overview", "performance", "agents", "decisions", "health"],
                        "default": "overview",
                        "description": "Dashboard view type",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session or cross-session view",
                    },
                    "real_time": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable real-time updates",
                    },
                    "export_format": {
                        "type": "string",
                        "enum": ["json", "html", "markdown"],
                        "description": "Export format",
                    },
                },
            },
            "examples": [{"dashboard_type": "performance", "real_time": True}],
        }

        registry["session_create_notebook"] = {
            "implementation": self._wrap_async_tool(
                self.session_engine.session_create_notebook_async
            ),
            "description": (
                "Create a reasoning narrative for this project session. "
                "**SYSTEM**: session — project-scoped work history. "
                "**TIER**: notebook — reasoning narrative recording abandoned paths, "
                "hypotheses, and context that lets future readers judge whether stored "
                "decisions/learnings are still valid. Call at session end or at a "
                "natural breakpoint. Use session_query_notebooks first if you want to "
                "avoid duplicate session records. "
                "**DISCIPLINE**: pass at least one of session_id, session_name, or "
                "project_name. Use allow_unbound=true to opt into the legacy unbound "
                "fallback (deprecated)."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Explicit session ID to summarize.",
                    },
                    "session_name": {
                        "type": "string",
                        "description": "Named session to summarize.",
                    },
                    "project_name": {
                        "type": "string",
                        "description": (
                            "Project name — summarizes the most-recent active session "
                            "for that project, or creates one if needed."
                        ),
                    },
                    "allow_unbound": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "If True, opt into the legacy '_unbound_' fallback when no "
                            "session identifier is provided. Deprecated."
                        ),
                    },
                    "title": {"type": "string", "description": "Custom title for the notebook"},
                    "include_decisions": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include decision log section",
                    },
                    "include_agents": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include agent execution summary",
                    },
                    "include_metrics": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include performance metrics",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for cross-session search",
                    },
                    "save_to_file": {
                        "type": "boolean",
                        "default": True,
                        "description": "Save markdown to session directory",
                    },
                    "save_to_database": {
                        "type": "boolean",
                        "default": True,
                        "description": "Persist to database for FTS search",
                    },
                },
            },
            "examples": [
                {
                    "_workflow_hint": "STEP 1: check existing notebooks to avoid duplicates",
                    "project_name": "session-intelligence",
                },
                {
                    "_workflow_hint": "STEP 2: log only if no recent notebook for this session",
                    "project_name": "session-intelligence",
                    "title": "Feature Implementation Session",
                    "tags": ["feature", "python"],
                },
            ],
        }

        registry["session_search"] = {
            "implementation": self._wrap_async_tool(self.session_engine.session_search),
            "description": (
                "Full-text search across session notebooks, decisions, and learnings. "
                "**SYSTEM**: session (with cross-project reach for decisions/learnings). "
                "**READS**: all three tiers — notebook narratives (fulltext/tag/file), "
                "project decisions (search_type='decisions', cross-project), "
                "project learnings (search_type='learnings', cross-project). "
                "**USE AS STEP 1** in the anti-dupe protocol before logging any "
                "new session decision or learning. Supports FTS5 syntax: AND, OR, NOT, phrases."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (supports FTS5 syntax: AND, OR, NOT, phrases)",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["fulltext", "tag", "file", "learnings", "decisions"],
                        "default": "fulltext",
                        "description": (
                            "Type of search to perform. 'decisions' and "
                            "'learnings' search across all projects (cross-project)."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum results to return",
                    },
                },
                "required": ["query"],
            },
            "examples": [
                {"query": "authentication bug fix"},
                {"query": "python", "search_type": "tag", "limit": 10},
                {"query": "pytest migration", "search_type": "decisions"},
            ],
        }

        registry["session_query_notebooks"] = {
            "implementation": self._wrap_async_tool(self.session_engine.session_query_notebooks),
            "description": (
                "Query session notebooks (reasoning narratives) with optional filters. "
                "**SYSTEM**: session — project-scoped work history. "
                "**READS**: notebook tier — session narrative records. "
                "Use as STEP 1 in the anti-dupe protocol before creating a new session "
                "notebook. Filter by project_path or tags to find existing narratives "
                "for the current work context."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Filter notebooks by project path",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter notebooks by tags",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum results to return",
                    },
                },
            },
            "examples": [
                {"limit": 10},
                {"project_path": "/home/user/my-project", "limit": 5},
                {"tags": ["feature", "bugfix"]},
            ],
        }

        registry["session_recall"] = {
            "implementation": self._wrap_async_tool(self.session_engine.session_recall),
            "description": (
                "Recall all stored knowledge for a specific project across all sessions. "
                "**SYSTEM**: session — project-scoped work history. "
                "**READS**: all three tiers — sessions, decisions, learnings, and notebooks "
                "for the named project. "
                "**USE AS STEP 1** in the anti-dupe protocol: call this before logging "
                "new decisions or learnings to surface what was already recorded. "
                "Requires project_name (explicit — do not omit)."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Project name to recall knowledge for",
                    },
                    "include": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["sessions", "decisions", "learnings", "notebooks"],
                        },
                        "description": "Sections to include (default: all)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Max items per section",
                    },
                    "days": {
                        "type": "integer",
                        "default": 30,
                        "description": "How far back to look in days",
                    },
                },
                "required": ["project_name"],
            },
            "examples": [
                {"project_name": "session-intelligence"},
                {
                    "project_name": "session-intelligence",
                    "include": ["decisions", "learnings"],
                    "limit": 5,
                    "days": 7,
                },
            ],
        }

        # ===== KNOWLEDGE SYSTEM TOOLS =====

        registry["session_log_learning"] = {
            "implementation": self._wrap_async_tool(self.session_engine.session_log_learning),
            "description": (
                "Log a reusable pattern or fix discovered while working on this project. "
                "**SYSTEM**: session — project-scoped work history. "
                "**TIER**: learning — reusable pattern with when/how-to-apply guidance. "
                "**ANTI-DUPE**: call session_search(query=X, search_type='learnings') "
                "or session_recall(project_name=Y) FIRST; if a similar pattern exists, "
                "prefer updating or extending it over creating a duplicate. "
                "**DISCIPLINE**: pass at least one of session_id, session_name, or "
                "project_name. Use allow_unbound=true to opt into the legacy unbound "
                "fallback (deprecated)."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["error_fix", "pattern", "preference", "workflow"],
                        "description": "Learning category",
                    },
                    "learning_content": {
                        "type": "string",
                        "description": "The actual knowledge/solution",
                    },
                    "trigger_context": {
                        "type": "string",
                        "description": "When to apply this learning (optional)",
                    },
                    "project_path": {
                        "type": "string",
                        "description": (
                            "Project path for scoping the learning row (back-compat). "
                            "Does not control session binding; use project_name for that."
                        ),
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Explicit session ID to bind this learning to.",
                    },
                    "session_name": {
                        "type": "string",
                        "description": "Named session to bind this learning to.",
                    },
                    "project_name": {
                        "type": "string",
                        "description": (
                            "Project name — binds to the most-recent active session "
                            "for that project, or creates one if needed."
                        ),
                    },
                    "allow_unbound": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "If True, opt into the legacy '_unbound_' fallback when no "
                            "session identifier is provided. Deprecated."
                        ),
                    },
                },
                "required": ["category", "learning_content"],
            },
            "examples": [
                {
                    "_workflow_hint": "STEP 1: search existing learnings before logging",
                    "query": "ModuleNotFoundError",
                    "search_type": "learnings",
                },
                {
                    "_workflow_hint": "STEP 2: log only if no matching learning found above",
                    "category": "error_fix",
                    "learning_content": "ImportError for module X: install via pip install X",
                    "trigger_context": "When seeing 'ModuleNotFoundError: X'",
                    "project_name": "session-intelligence",
                },
            ],
        }

        registry["session_find_solution"] = {
            "implementation": self._wrap_async_tool(self.session_engine.session_find_solution),
            "description": "Find solutions for an error from project and universal knowledge",
            "schema": {
                "type": "object",
                "properties": {
                    "error_text": {
                        "type": "string",
                        "description": "The error message/pattern to search for",
                    },
                    "error_category": {
                        "type": "string",
                        "enum": ["compile", "runtime", "config", "dependency", "test", "lint"],
                        "description": "Optional category hint",
                    },
                    "include_universal": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to include universal (cross-project) solutions",
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Absolute path to the caller's project, used to scope project-specific solutions. Omitting it falls back to the server's own working directory, which is almost never the caller's project. Relative paths are ignored (they would resolve against the server's cwd, not the caller's).",
                    },
                },
                "required": ["error_text"],
            },
            "examples": [
                {"error_text": "ModuleNotFoundError: No module named 'foo'"},
                {"error_text": "TypeError: expected str, got int", "error_category": "runtime"},
                {"error_text": "ModuleNotFoundError: No module named 'foo'", "project_path": "/home/user/projects/my-project"},
            ],
        }

        registry["session_update_solution_outcome"] = {
            "implementation": self._wrap_async_tool(
                self.session_engine.session_update_solution_outcome
            ),
            "description": "Update success/failure count for a solution after trying it",
            "schema": {
                "type": "object",
                "properties": {
                    "solution_id": {
                        "type": "string",
                        "description": "ID of the solution to update",
                    },
                    "success": {"type": "boolean", "description": "Whether the solution worked"},
                },
                "required": ["solution_id", "success"],
            },
            "examples": [
                {"solution_id": "sol_abc123", "success": True},
                {"solution_id": "sol_xyz789", "success": False},
            ],
        }

        # ===== AGENT SYSTEM TOOLS =====

        registry["agent_register"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_register),
            "description": (
                "Register or update an agent in the global cross-project agent registry. "
                "**SYSTEM**: agent — global pattern library, not bound to any project. "
                "agent_name MUST match the filename stem of a file under "
                "~/.claude/agents/{type}/{name}.md (validated; raises AgentNotFoundError "
                "on typo). Call agent_get_info(agent_name=X) first to check if already "
                "registered before re-registering."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Unique agent name (e.g., 'focused-quality-resolver')",
                    },
                    "agent_type": {
                        "type": "string",
                        "description": (
                            "Agent type (e.g., 'focused', 'comprehensive', 'micro', 'meta')"
                        ),
                    },
                    "display_name": {
                        "type": "string",
                        "description": "Human-friendly display name",
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of agent's purpose",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional agent metadata (version, author, etc.)",
                    },
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of agent capabilities",
                    },
                },
                "required": ["agent_name", "agent_type"],
            },
            "examples": [
                {
                    "_workflow_hint": "STEP 1: check if agent already registered",
                    "agent_name": "focused-quality-resolver",
                },
                {
                    "_workflow_hint": "STEP 2: register only if agent_get_info returned not-found",
                    "agent_name": "focused-quality-resolver",
                    "agent_type": "focused",
                    "display_name": "Quality Resolver",
                    "description": "Resolves code quality issues",
                    "capabilities": ["lint-fix", "format", "type-check"],
                },
            ],
        }

        registry["agent_get_info"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_get_info),
            "description": "Get agent information by name or UUID",
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Agent name (e.g., 'focused-quality-resolver') or UUID",
                    }
                },
                "required": ["agent_name"],
            },
            "examples": [
                {"agent_name": "focused-quality-resolver"},
                {"agent_name": "550e8400-e29b-41d4-a716-446655440000"},
            ],
        }

        registry["agent_log_decision"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_log_decision),
            "description": (
                "Log a decision made by this agent that applies across projects. "
                "**SYSTEM**: agent — global cross-project pattern library; agent_name "
                "MUST match ~/.claude/agents/{type}/{name}.md (validated). "
                "**TIER**: decision — atomic choice (what was decided + reasoning), "
                "queryable and valid-now. "
                "**ANTI-DUPE**: call agent_query_decisions(agent_name=X, "
                "decision_type=Y) FIRST; if a similar decision exists, update its "
                "outcome via agent_update_decision_outcome instead of re-logging."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the agent making the decision",
                    },
                    "decision_type": {
                        "type": "string",
                        "description": (
                            "Category of decision "
                            "(e.g., 'tool_selection', 'error_handling', 'strategy')"
                        ),
                    },
                    "context": {
                        "type": "string",
                        "description": "The situation or problem that required a decision",
                    },
                    "decision": {"type": "string", "description": "The decision that was made"},
                    "reasoning": {"type": "string", "description": "Why this decision was made"},
                    "alternatives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Other options that were considered",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8,
                        "description": "Confidence level in the decision (0.0-1.0)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization and search",
                    },
                },
                "required": ["agent_name", "decision_type", "context", "decision"],
            },
            "examples": [
                {
                    "_workflow_hint": "STEP 1: query existing decisions before logging",
                    "agent_name": "focused-quality-resolver",
                    "decision_type": "tool_selection",
                },
                {
                    "_workflow_hint": "STEP 2: log only if no matching decision found above",
                    "agent_name": "focused-quality-resolver",
                    "decision_type": "tool_selection",
                    "context": "Multiple lint errors in Python file",
                    "decision": "Use ruff --fix for auto-fixable issues",
                    "reasoning": "Ruff is faster and handles most common issues",
                    "alternatives": ["Manual fixes", "Black + isort separately"],
                    "confidence": 0.9,
                    "tags": ["python", "linting"],
                },
            ],
        }

        registry["agent_query_decisions"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_query_decisions),
            "description": (
                "Query decisions logged for an agent across all projects. "
                "**SYSTEM**: agent — global cross-project pattern library. "
                "**READS**: decision tier — atomic choices this agent has recorded. "
                "**USE AS STEP 1** in the anti-dupe protocol before calling "
                "agent_log_decision; if a matching decision exists, update its outcome "
                "via agent_update_decision_outcome rather than creating a duplicate."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "Name of the agent to query"},
                    "decision_type": {
                        "type": "string",
                        "description": "Filter by decision type/category",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of results",
                    },
                },
                "required": ["agent_name"],
            },
            "examples": [
                {"agent_name": "focused-quality-resolver"},
                {
                    "agent_name": "focused-quality-resolver",
                    "decision_type": "error_handling",
                    "limit": 10,
                },
            ],
        }

        registry["agent_update_decision_outcome"] = {
            "implementation": self._wrap_async_tool(
                self.session_engine.agent_update_decision_outcome
            ),
            "description": "Update the outcome of a decision after execution",
            "schema": {
                "type": "object",
                "properties": {
                    "decision_id": {
                        "type": "string",
                        "description": "ID of the decision to update",
                    },
                    "outcome": {"type": "string", "description": "Description of the outcome"},
                    "success": {
                        "type": "boolean",
                        "description": "Whether the decision led to a successful outcome",
                    },
                },
                "required": ["decision_id", "outcome", "success"],
            },
            "examples": [
                {
                    "decision_id": "dec_abc123",
                    "outcome": "All lint errors fixed successfully",
                    "success": True,
                },
                {
                    "decision_id": "dec_xyz789",
                    "outcome": "Auto-fix introduced new errors, needed manual intervention",
                    "success": False,
                },
            ],
        }

        registry["agent_log_learning"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_log_learning),
            "description": (
                "Log a reusable pattern this agent has discovered. "
                "**SYSTEM**: agent — global cross-project pattern library; agent_name "
                "MUST match ~/.claude/agents/{type}/{name}.md (validated; raises "
                "AgentNotFoundError on typo). "
                "**TIER**: learning — reusable pattern with when/how-to-apply guidance. "
                "**ANTI-DUPE**: call agent_query_learnings(agent_name=X, "
                "learning_type=Y) FIRST; if a similar pattern exists, update its "
                "outcome stats via agent_update_learning_outcome instead of logging "
                "a duplicate."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the agent logging the learning",
                    },
                    "learning_type": {
                        "type": "string",
                        "description": (
                            "Type of learning "
                            "(e.g., 'pattern', 'anti_pattern', 'technique', 'preference')"
                        ),
                    },
                    "title": {"type": "string", "description": "Brief title for the learning"},
                    "content": {
                        "type": "string",
                        "description": "Detailed content of the learning",
                    },
                    "source_context": {
                        "type": "string",
                        "description": "The context where this was learned",
                    },
                    "applicability": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Situations where this learning applies",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8,
                        "description": "Confidence level in the learning (0.0-1.0)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization and search",
                    },
                },
                "required": ["agent_name", "learning_type", "title", "content"],
            },
            "examples": [
                {
                    "_workflow_hint": "STEP 1: query existing learnings before logging",
                    "agent_name": "focused-quality-resolver",
                    "learning_type": "pattern",
                },
                {
                    "_workflow_hint": "STEP 2: log only if no matching learning found above",
                    "agent_name": "focused-quality-resolver",
                    "learning_type": "pattern",
                    "title": "Ruff handles import sorting",
                    "content": "Ruff with isort rules enabled can replace separate isort step",
                    "applicability": ["python-projects", "lint-workflows"],
                    "confidence": 0.95,
                    "tags": ["python", "tooling"],
                },
            ],
        }

        registry["agent_query_learnings"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_query_learnings),
            "description": (
                "Query reusable patterns logged for an agent across all projects. "
                "**SYSTEM**: agent — global cross-project pattern library. "
                "**READS**: learning tier — reusable patterns this agent has recorded. "
                "**USE AS STEP 1** in the anti-dupe protocol before calling "
                "agent_log_learning; if a similar pattern exists, update its stats "
                "via agent_update_learning_outcome rather than creating a duplicate."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "Name of the agent to query"},
                    "learning_type": {
                        "type": "string",
                        "description": "Filter by learning type/category",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of results",
                    },
                },
                "required": ["agent_name"],
            },
            "examples": [
                {"agent_name": "focused-quality-resolver"},
                {"agent_name": "focused-quality-resolver", "learning_type": "pattern", "limit": 5},
            ],
        }

        registry["agent_update_learning_outcome"] = {
            "implementation": self._wrap_async_tool(
                self.session_engine.agent_update_learning_outcome
            ),
            "description": "Update application stats for a learning after it was applied",
            "schema": {
                "type": "object",
                "properties": {
                    "learning_id": {
                        "type": "string",
                        "description": "ID of the learning to update",
                    },
                    "times_applied_increment": {
                        "type": "integer",
                        "default": 1,
                        "description": "How many times to increment the application count",
                    },
                    "new_success_rate": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Updated success rate (0.0-1.0)",
                    },
                },
                "required": ["learning_id"],
            },
            "examples": [
                {"learning_id": "lrn_abc123", "times_applied_increment": 1},
                {
                    "learning_id": "lrn_xyz789",
                    "times_applied_increment": 1,
                    "new_success_rate": 0.85,
                },
            ],
        }

        registry["agent_create_notebook"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_create_notebook),
            "description": (
                "Create a reasoning narrative notebook for this agent. "
                "**SYSTEM**: agent — global cross-project pattern library; agent_name "
                "MUST match ~/.claude/agents/{type}/{name}.md (validated). "
                "**TIER**: notebook — reasoning narrative recording abandoned paths, "
                "hypotheses, and context that lets future readers judge whether stored "
                "decisions/learnings are still valid. Use agent_query_notebooks first "
                "to avoid duplicate notebook records for the same work."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the agent creating the notebook",
                    },
                    "title": {"type": "string", "description": "Title of the notebook"},
                    "content": {
                        "type": "string",
                        "description": "Markdown content of the notebook",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary for search/display",
                    },
                    "notebook_type": {
                        "type": "string",
                        "default": "execution",
                        "description": (
                            "Notebook category. Canonical values: 'execution' (chronicle of "
                            "work done), 'research' (investigation/exploration), 'learning' "
                            "(reasoning narrative for a discovered pattern). Default: "
                            "'execution'. NOTE: this differs from the 'learning' entity — a "
                            "notebook of type 'learning' is the NARRATIVE explaining how/why "
                            "a learning was discovered; a learning entity is the atomic "
                            "pattern itself."
                        ),
                    },
                    "context": {"type": "object", "description": "Additional context metadata"},
                    "decisions_referenced": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of decisions referenced in this notebook",
                    },
                    "learnings_referenced": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of learnings referenced in this notebook",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization and search",
                    },
                },
                "required": ["agent_name", "title", "content"],
            },
            "examples": [
                {
                    "_workflow_hint": "STEP 1: query existing notebooks to avoid duplicates",
                    "agent_name": "focused-quality-resolver",
                    "notebook_type": "execution",
                },
                {
                    "_workflow_hint": "STEP 2: create notebook only if no recent one exists",
                    "agent_name": "focused-quality-resolver",
                    "title": "Quality Resolution Session - 2025-01-05",
                    "content": "## Summary\n\nFixed 15 lint issues...",
                    "summary": "Resolved lint issues in src/module.py",
                    "notebook_type": "execution",
                    "tags": ["quality", "python"],
                },
            ],
        }

        registry["agent_query_notebooks"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_query_notebooks),
            "description": (
                "Query reasoning narrative notebooks logged for an agent. "
                "**SYSTEM**: agent — global cross-project pattern library. "
                "**READS**: notebook tier — narrative records of this agent's work. "
                "**USE AS STEP 1** in the anti-dupe protocol before calling "
                "agent_create_notebook; filter by notebook_type and tags to find "
                "existing narratives for the current work context."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "Name of the agent to query"},
                    "notebook_type": {"type": "string", "description": "Filter by notebook type"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of results",
                    },
                },
                "required": ["agent_name"],
            },
            "examples": [
                {"agent_name": "focused-quality-resolver"},
                {
                    "agent_name": "focused-quality-resolver",
                    "notebook_type": "execution",
                    "limit": 5,
                },
            ],
        }

        registry["session_agent_stats"] = {
            "implementation": self._wrap_async_tool(
                self._session_agent_stats_handler
            ),
            "description": (
                "Return per-agent-type usage statistics over a configurable time window. "
                "Aggregates invocations, successes, failures, and average duration from "
                "agent_executions records. Use this for data-driven decisions about which "
                "agent types are actually being used."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "time_window_hours": {
                        "type": "integer",
                        "default": 168,
                        "description": "Lookback window in hours (default 168 = 7 days)",
                    },
                    "min_invocations": {
                        "type": "integer",
                        "default": 1,
                        "description": "Filter out agents with fewer than this many invocations",
                    },
                },
            },
            "examples": [
                {"time_window_hours": 168},
                {"time_window_hours": 720, "min_invocations": 3},
            ],
        }

        registry["agent_search_all"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_search_all),
            "description": (
                "Search across all data for an agent: decisions, learnings, and notebooks. "
                "**SYSTEM**: agent — global cross-project pattern library. "
                "**READS**: all three tiers simultaneously for the named agent. "
                "**USE AS STEP 1** in the anti-dupe protocol when you are unsure which "
                "tier to check — a single call surfaces matches across decisions, "
                "learnings, and notebooks so you can extend or supersede rather than "
                "create duplicates."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "Name of the agent to search"},
                    "query": {"type": "string", "description": "Search query string"},
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum results per content type",
                    },
                },
                "required": ["agent_name", "query"],
            },
            "examples": [
                {"agent_name": "focused-quality-resolver", "query": "ruff lint"},
                {"agent_name": "focused-quality-resolver", "query": "import sorting", "limit": 10},
            ],
        }

        return registry

    async def _session_agent_stats_handler(self, **params) -> dict[str, Any]:
        """Handler for session_agent_stats that applies min_invocations filter."""
        time_window_hours = params.get("time_window_hours", 168)
        min_invocations = params.get("min_invocations", 1)

        result = await self.session_engine.session_agent_stats(
            time_window_hours=time_window_hours,
        )

        if "agent_stats" in result and min_invocations > 1:
            result["agent_stats"] = [
                s for s in result["agent_stats"]
                if s["invocations"] >= min_invocations
            ]

        return result

    def _wrap_tool(self, tool_func):
        """Wrap tool function with token limiting and error handling."""

        @wraps(tool_func)
        def wrapper(*args, **kwargs):
            try:
                result = tool_func(*args, **kwargs)
                return apply_token_limits(result, tool_func.__name__)
            except Exception as e:
                logger.error(f"Error in {tool_func.__name__}: {e}")
                return {"error": str(e), "tool": tool_func.__name__}

        return wrapper

    def _wrap_async_tool(self, async_tool_func):
        """Wrap async tool function with token limiting and error handling."""

        @wraps(async_tool_func)
        async def wrapper(*args, **kwargs):
            try:
                result = await async_tool_func(*args, **kwargs)
                return apply_token_limits(result, async_tool_func.__name__)
            except Exception as e:
                logger.error(f"Error in {async_tool_func.__name__}: {e}")
                return {"error": str(e), "tool": async_tool_func.__name__}

        return wrapper

    def _discover_tools(self, pattern: str = "") -> dict[str, Any]:
        """
        Core implementation of the discover_tools meta-tool.

        Extracted as a method so it can be called directly in tests without
        going through the FastMCP tool dispatcher.
        """
        tools = []

        for name, info in self.tool_registry.items():
            # Apply pattern filter if provided
            if pattern and pattern.strip() and pattern.lower() not in name.lower():
                continue

            tools.append({"name": name, "description": info["description"]})

        result: dict[str, Any] = {
            "available_tools": tools,
            "total_tools": len(self.tool_registry),
            "filtered_count": len(tools),
        }
        if not pattern:
            result["_framework_guide"] = {
                "systems": {
                    "session_*": (
                        "Project-scoped work history. Use when recording what was tried "
                        "on a SPECIFIC project. Bind to project_name explicitly."
                    ),
                    "agent_*": (
                        "Global cross-project agent pattern library. Use when refining "
                        "how YOU (the agent) approach a class of problem regardless of "
                        "project. agent_name is validated against "
                        "~/.claude/agents/{type}/{name}.md."
                    ),
                    "knowledge_*": "Cross-project searchable knowledge base.",
                },
                "data_tiers": {
                    "decision": "Atomic choice — what was decided. Queryable, valid-now.",
                    "learning": "Reusable pattern — when/how to apply.",
                    "notebook": (
                        "Reasoning narrative — abandoned paths, hypotheses, the context "
                        "that lets future readers judge whether stored decisions/learnings "
                        "are still valid."
                    ),
                },
                "anti_dupe_protocol": (
                    "Before logging, QUERY existing tools (session_search, session_recall, "
                    "agent_query_*) for matching content. Re-logging the same fact creates "
                    "duplicates that pollute future retrievals."
                ),
                "project_name_discipline": (
                    "Every session_* write tool accepts project_name. Pass it EXPLICITLY "
                    "— the silent _unbound_ fallback corrupts retrieval (see PR #17)."
                ),
            }
        return result

    def _setup_meta_tools(self):
        """Setup the 3 meta-tools for dynamic discovery."""

        @self.app.tool(
            description=(
                "Discover session lifecycle, decision logging, agent tracking, "
                "and learning tools. "
                "USE WHEN: starting sessions, logging decisions, searching learnings"
            )
        )
        def discover_tools(pattern: str = "") -> dict[str, Any]:
            """
            [STEP 1] Discover available tools in the session-intelligence MCP server.

            USE WHEN:
            - You need to find session management operations (create, resume, finalize)
            - You want to log decisions, learnings, or track agent execution
            - You need to search across session notebooks or agent knowledge
            - You're exploring what session/agent operations are available
            - You don't know the exact tool name for an operation

            COMMON TASKS:
            - Session lifecycle: session_manage_lifecycle, session_monitor_health
            - Execution tracking: session_track_execution, session_track_file_operation
            - Decision/learning: session_log_decision, session_log_learning, agent_log_decision
            - Agent registry: agent_register, agent_get_info, agent_query_decisions
            - Search: session_search, agent_search_all

            This lean interface provides tools across 3 domains (session, agent, knowledge),
            saving ~25k tokens vs loading all tool schemas upfront.

            WORKFLOW:
            1. discover_tools(pattern) <- YOU ARE HERE
            2. get_tool_spec(tool_name) <- Get schema/parameters for a specific tool
            3. execute_tool(tool_name, params) <- Execute the operation

            Args:
                pattern: Filter tools by name (e.g., "session", "agent", "learning")
                         Leave empty "" to see all tools

            Returns:
                Dictionary containing:
                - available_tools: List of tools, each with:
                  * name: Tool name to use in get_tool_spec() or execute_tool()
                  * description: What the tool does
                - total_tools: Total tools in registry (dynamic)
                - filtered_count: How many matched your pattern

                Example output for discover_tools("session"):
                {
                  "available_tools": [
                    {
                        "name": "session_manage_lifecycle",
                        "description": "Complete session lifecycle management with recovery"
                    },
                    {
                        "name": "session_track_execution",
                        "description": "Track agent execution with pattern detection"
                    },
                    {
                        "name": "session_log_decision",
                        "description": "Log decisions with context and impact analysis"
                    }
                  ],
                  "filtered_count": 13,
                  "total_tools": <dynamic>
                }

            Examples:
                discover_tools("")              # List all tools
                discover_tools("session")       # Find session management tools
                discover_tools("agent")         # Find agent registry tools
                discover_tools("learning")      # Find learning/knowledge tools

            MISSING TOOL? If you need an operation that's not available:
            File an issue at https://github.com/Claire-s-Monster/session-intelligence
            """
            return self._discover_tools(pattern)

        @self.app.tool(
            description=(
                "Get parameter schema for session/agent/learning tools. "
                "USE WHEN: need exact parameters, debugging validation errors"
            )
        )
        def get_tool_spec(tool_name: str) -> dict[str, Any]:
            """
            [STEP 2] Get detailed schema and parameters for a specific tool.

            USE WHEN:
            - You found a tool via discover_tools() but need to see its parameters
            - You need to understand required vs optional parameters before calling execute_tool()
            - You want to see parameter types and valid values (enums, defaults, etc.)
            - You're debugging parameter validation errors from execute_tool()

            DON'T SKIP THIS STEP! Calling execute_tool() without checking the schema first
            will likely fail parameter validation. This tool shows you exactly what to pass.

            WORKFLOW:
            1. discover_tools(pattern) <- Already done
            2. get_tool_spec(tool_name) <- YOU ARE HERE
            3. execute_tool(tool_name, params) <- Execute with correct parameters

            Args:
                tool_name: Exact tool name from discover_tools() output
                           Examples: "session_manage_lifecycle", "agent_log_decision"

            Returns:
                Dictionary containing:
                - name: Tool name (same as input)
                - description: What the tool does
                - schema: JSON Schema with:
                  * properties: Each parameter's type, description, default value
                  * required: List of required parameters
                - examples: Usage examples showing common parameter combinations

                Example output for get_tool_spec("session_manage_lifecycle"):
                {
                  "name": "session_manage_lifecycle",
                  "description": "Complete session lifecycle management with recovery",
                  "schema": {
                    "properties": {
                      "operation": {
                          "type": "string",
                          "enum": ["create", "resume", "finalize", "validate"]
                      },
                      "mode": {
                          "type": "string",
                          "enum": ["local", "remote", "hybrid", "auto"],
                          "default": "local"
                      },
                      "project_name": {
                          "type": "string",
                          "description": "Project context (optional)"
                      }
                    },
                    "required": ["operation"]
                  },
                  "examples": [
                    {"operation": "create", "project_name": "my-project"},
                    {"operation": "resume", "mode": "hybrid"}
                  ]
                }

                Example for get_tool_spec("agent_log_decision"):
                {
                  "name": "agent_log_decision",
                  "schema": {
                    "properties": {
                      "agent_name": {"type": "string"},
                      "decision_type": {"type": "string"},
                      "context": {"type": "string"},
                      "decision": {"type": "string"},
                      "confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.8}
                    },
                    "required": ["agent_name", "decision_type", "context", "decision"]
                  }
                }

            Examples:
                get_tool_spec("session_manage_lifecycle")  # See lifecycle operations
                get_tool_spec("agent_log_decision")        # See decision logging params
                get_tool_spec("session_search")            # See search query options

            TOOL NOT FOUND? Use discover_tools() first to find available tools.
            If the tool should exist but doesn't, file a feature request at:
            https://github.com/Claire-s-Monster/session-intelligence
            """
            if tool_name not in self.tool_registry:
                available_tools = list(self.tool_registry.keys())
                return {
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": available_tools,
                }

            tool_info = self.tool_registry[tool_name]
            return {
                "name": tool_name,
                "description": tool_info["description"],
                "schema": tool_info["schema"],
                "examples": tool_info["examples"],
            }

        @self.app.tool(
            description=(
                "Execute session management, agent tracking, or knowledge operations. "
                "Returns domain-specific results for sessions, agents, and learnings"
            )
        )
        async def execute_tool(tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
            """
            [STEP 3] Execute a session-intelligence operation.

            USE WHEN: You have the tool name and parameters ready to perform an operation.

            WORKFLOW:
            1. discover_tools(pattern) <- Found the right tool
            2. get_tool_spec(tool_name) <- Got the parameter schema
            3. execute_tool(tool_name, params) <- YOU ARE HERE

            VALIDATION: Parameters are validated against the tool schema before execution.
            Unexpected parameters will be rejected with an error listing valid parameters.

            Args:
                tool_name: Exact tool name from discover_tools() or get_tool_spec()
                parameters: Dictionary of parameters matching the tool schema
                           Use get_tool_spec() if unsure what parameters are needed

            Returns:
                SUCCESS: Tool execution result containing:
                - tool: Name of tool that was executed
                - status: "success"
                - result: Tool-specific output, varies by tool:
                  * session_manage_lifecycle: {
                      "session_id": "...", "status": "active", "mode": "local"
                  }
                  * session_log_decision: {"decision_id": "dec_abc123", "logged": true}
                  * session_log_learning: {"learning_id": "learn_xyz789", "category": "error_fix"}
                  * agent_register: {"agent_id": "uuid", "name": "agent-name", "registered": true}
                  * session_search: {"results": [...], "total_matches": 15}
                  * session_get_dashboard: {"overview": {...}, "health": {...}, "metrics": {...}}

                ERROR: Validation/execution failure with:
                - tool: Name of tool that failed
                - status: "error"
                - error: Error message explaining what went wrong
                - (if tool not found): available_tools list

            Examples:
                # Create a new session
                execute_tool("session_manage_lifecycle", {
                    "operation": "create",
                    "project_name": "my-project"
                })

                # Log a decision made by an agent
                execute_tool("agent_log_decision", {
                    "agent_name": "focused-quality-resolver",
                    "decision_type": "tool_selection",
                    "context": "Multiple lint errors found",
                    "decision": "Use ruff --fix for auto-fixable issues",
                    "confidence": 0.9
                })

                # Search across session notebooks
                execute_tool("session_search", {
                    "query": "authentication bug fix",
                    "search_type": "fulltext",
                    "limit": 10
                })

            DON'T KNOW WHAT TOOL TO USE?
            Call discover_tools(pattern) first to find the right tool for your task.

            FOUND A BUG OR MISSING FEATURE?
            File an issue at https://github.com/Claire-s-Monster/session-intelligence
            Include: tool name, parameters used, error message, expected vs actual behavior
            """
            import inspect

            if tool_name not in self.tool_registry:
                available_tools = list(self.tool_registry.keys())
                return {
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": available_tools,
                }

            tool_info = self.tool_registry[tool_name]
            tool_func = tool_info["implementation"]

            # Coerce JSON string parameters to dict (MCP proxies may serialize objects as strings)
            if isinstance(parameters, str):
                try:
                    parameters = json.loads(parameters)
                except (json.JSONDecodeError, TypeError) as e:
                    return {
                        "tool": tool_name,
                        "status": "error",
                        "error": f"Invalid parameters JSON: {e}",
                    }
            if not isinstance(parameters, dict):
                return {
                    "tool": tool_name,
                    "status": "error",
                    "error": f"parameters must be a mapping, got {type(parameters).__name__}",
                }

            # Validate parameters against the declared schema before dispatch, so a
            # mistyped name surfaces as an actionable error instead of a raw TypeError
            # leaking out of the engine. Schemas without declared properties are not
            # validated (nothing to check against).
            schema = tool_info.get("schema") or {}
            valid_params = set(schema.get("properties", {}))
            if valid_params:
                unknown = sorted(set(parameters) - valid_params)
                missing = sorted(set(schema.get("required", [])) - set(parameters))
                if unknown or missing:
                    problems = []
                    if unknown:
                        problems.append(f"unexpected parameter(s): {unknown}")
                    if missing:
                        problems.append(f"missing required parameter(s): {missing}")
                    return {
                        "tool": tool_name,
                        "status": "error",
                        "error": (
                            f"Invalid parameters for '{tool_name}' - "
                            f"{'; '.join(problems)}. "
                            f"Valid parameters: {sorted(valid_params)}. "
                            f"Call get_tool_spec('{tool_name}') for the full schema."
                        ),
                    }

            try:
                # Execute tool - await if async, call directly if sync
                if inspect.iscoroutinefunction(tool_func):
                    result = await tool_func(**parameters)
                else:
                    result = tool_func(**parameters)
                return {"tool": tool_name, "status": "success", "result": result}
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                return {"tool": tool_name, "status": "error", "error": str(e)}

    def get_app(self) -> FastMCP:
        """Get the FastMCP app instance."""
        return self.app


def create_lean_interface(
    session_engine: SessionIntelligenceEngine,
    database: Any | None = None,
) -> FastMCP:
    """
    Create a lean MCP interface with minimal context consumption.

    Args:
        session_engine: Initialized session intelligence engine
        database: Optional database instance for persistence

    Returns:
        FastMCP app with 3 meta-tools exposing full functionality
    """
    lean_interface = LeanMCPInterface(session_engine)
    app = lean_interface.get_app()

    # Add database lifecycle hooks if database is provided
    if database:
        original_run_stdio = app.run_stdio_async

        async def run_stdio_with_db(show_banner: bool = True) -> None:
            """Wrap run_stdio_async to initialize/cleanup database."""
            # Initialize database connection pool
            try:
                await database.initialize()
                logger.info("Database initialized successfully")
            except Exception as e:
                logger.error(f"Database initialization failed: {e}")
                session_engine.database = None

            try:
                # Run the original stdio server
                await original_run_stdio(show_banner=show_banner)
            finally:
                # Cleanup database on shutdown
                if session_engine.database:
                    try:
                        await database.close()
                        logger.info("Database connection closed")
                    except Exception as e:
                        logger.warning(f"Error closing database: {e}")

        app.run_stdio_async = run_stdio_with_db

    return app
