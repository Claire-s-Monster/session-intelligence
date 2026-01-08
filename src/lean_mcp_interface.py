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

import logging
from functools import wraps
from typing import Any

from fastmcp import FastMCP

from core.session_engine import SessionIntelligenceEngine
from models.session_models import *
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
            "implementation": self._wrap_tool(self.session_engine.session_manage_lifecycle),
            "description": "Complete session lifecycle management with recovery",
            "schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "resume", "finalize", "validate"],
                        "description": "Lifecycle operation to perform"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["local", "remote", "hybrid", "auto"],
                        "default": "local",
                        "description": "Session mode"
                    },
                    "project_name": {
                        "type": "string",
                        "description": "Project context (optional)"
                    },
                    "metadata": {
                        "description": "Additional session metadata"
                    },
                    "auto_recovery": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable automatic recovery"
                    }
                },
                "required": ["operation"]
            },
            "examples": [
                {"operation": "create", "project_name": "my-project"},
                {"operation": "resume", "mode": "hybrid"}
            ]
        }

        registry["session_track_execution"] = {
            "implementation": self._wrap_tool(self.session_engine.session_track_execution),
            "description": "Track agent execution with pattern detection",
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Agent being executed"
                    },
                    "step_data": {
                        "type": "object",
                        "description": "ExecutionStep details"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (optional)"
                    },
                    "track_patterns": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable pattern detection"
                    },
                    "suggest_optimizations": {
                        "type": "boolean",
                        "default": True,
                        "description": "Generate optimization suggestions"
                    }
                },
                "required": ["agent_name", "step_data"]
            },
            "examples": [
                {"agent_name": "test-runner", "step_data": {"phase": "start", "command": "pytest"}}
            ]
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
                        "description": "Agents to coordinate"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session context"
                    },
                    "execution_mode": {
                        "type": "string",
                        "enum": ["sequential", "parallel", "adaptive"],
                        "default": "sequential",
                        "description": "Execution strategy"
                    },
                    "dependency_graph": {
                        "type": "object",
                        "description": "Agent dependencies"
                    },
                    "optimization_level": {
                        "type": "string",
                        "enum": ["conservative", "balanced", "aggressive"],
                        "default": "balanced",
                        "description": "Optimization approach"
                    }
                },
                "required": ["agents"]
            },
            "examples": [
                {"agents": [{"name": "quality-check"}, {"name": "test-runner"}], "execution_mode": "parallel"}
            ]
        }

        registry["session_log_decision"] = {
            "implementation": self._wrap_tool(self.session_engine.session_log_decision),
            "description": "Log decisions with context and impact analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "description": "Decision description"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session context"
                    },
                    "context": {
                        "type": "object",
                        "description": "Decision context and rationale"
                    },
                    "impact_analysis": {
                        "type": "boolean",
                        "default": True,
                        "description": "Analyze decision impact"
                    },
                    "link_artifacts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Related files or commits"
                    }
                },
                "required": ["decision"]
            },
            "examples": [
                {"decision": "Switch to pytest for testing", "context": {"reason": "Better async support"}}
            ]
        }

        registry["session_track_file_operation"] = {
            "implementation": self._wrap_tool(self.session_engine.session_track_file_operation),
            "description": "Track file create/edit/delete operations for session notebook",
            "schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "edit", "delete", "read"],
                        "description": "File operation type"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "lines_added": {
                        "type": "integer",
                        "default": 0,
                        "description": "Number of lines added"
                    },
                    "lines_removed": {
                        "type": "integer",
                        "default": 0,
                        "description": "Number of lines removed"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief description of changes"
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Tool that made the change"
                    }
                },
                "required": ["operation", "file_path"]
            },
            "examples": [
                {"operation": "create", "file_path": "src/module.py", "lines_added": 150}
            ]
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
                        "description": "Analysis scope"
                    },
                    "pattern_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Patterns to analyze"
                    },
                    "include_agents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific agents to analyze"
                    },
                    "learning_mode": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable ML-based learning"
                    },
                    "generate_insights": {
                        "type": "boolean",
                        "default": True,
                        "description": "Generate actionable insights"
                    }
                }
            },
            "examples": [
                {"scope": "recent", "pattern_types": ["execution", "errors"]}
            ]
        }

        registry["session_monitor_health"] = {
            "implementation": self._wrap_tool(self.session_engine.session_monitor_health),
            "description": "Real-time session health monitoring with auto-recovery",
            "schema": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session to monitor"
                    },
                    "health_checks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Checks to perform"
                    },
                    "auto_recover": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable automatic recovery"
                    },
                    "alert_thresholds": {
                        "type": "object",
                        "description": "Custom alert thresholds"
                    },
                    "include_diagnostics": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include detailed diagnostics"
                    }
                }
            },
            "examples": [
                {"health_checks": ["continuity", "files"], "auto_recover": True}
            ]
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
                        "description": "Workflow type"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session context"
                    },
                    "workflow_config": {
                        "type": "object",
                        "description": "Workflow configuration"
                    },
                    "parallel_execution": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable parallel execution"
                    },
                    "optimize_execution": {
                        "type": "boolean",
                        "default": True,
                        "description": "Optimize execution order"
                    }
                },
                "required": ["workflow_type"]
            },
            "examples": [
                {"workflow_type": "tdd", "parallel_execution": True}
            ]
        }

        registry["session_analyze_commands"] = {
            "implementation": self._wrap_tool(self.session_engine.session_analyze_commands),
            "description": "Analyze hook-based commands for inefficiencies",
            "schema": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session to analyze"
                    },
                    "command_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command categories"
                    },
                    "detect_inefficiencies": {
                        "type": "boolean",
                        "default": True,
                        "description": "Detect inefficient patterns"
                    },
                    "suggest_alternatives": {
                        "type": "boolean",
                        "default": True,
                        "description": "Suggest better approaches"
                    },
                    "include_timing": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include timing analysis"
                    }
                }
            },
            "examples": [
                {"command_types": ["git", "test"], "detect_inefficiencies": True}
            ]
        }

        registry["session_track_missing_functions"] = {
            "implementation": self._wrap_tool(self.session_engine.session_track_missing_functions),
            "description": "Track missing functions for ecosystem improvement",
            "schema": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session context"
                    },
                    "auto_suggest": {
                        "type": "boolean",
                        "default": True,
                        "description": "Suggest function implementations"
                    },
                    "priority_analysis": {
                        "type": "boolean",
                        "default": True,
                        "description": "Analyze implementation priority"
                    },
                    "generate_report": {
                        "type": "boolean",
                        "default": True,
                        "description": "Generate missing function report"
                    }
                }
            },
            "examples": [
                {"auto_suggest": True, "priority_analysis": True}
            ]
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
                        "description": "Dashboard view type"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session or cross-session view"
                    },
                    "real_time": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable real-time updates"
                    },
                    "export_format": {
                        "type": "string",
                        "enum": ["json", "html", "markdown"],
                        "description": "Export format"
                    }
                }
            },
            "examples": [
                {"dashboard_type": "performance", "real_time": True}
            ]
        }

        registry["session_create_notebook"] = {
            "implementation": self._wrap_async_tool(self.session_engine.session_create_notebook_async),
            "description": "Generate markdown notebook/summary at session end with searchable content",
            "schema": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session to summarize (defaults to current)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Custom title for the notebook"
                    },
                    "include_decisions": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include decision log section"
                    },
                    "include_agents": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include agent execution summary"
                    },
                    "include_metrics": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include performance metrics"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for cross-session search"
                    },
                    "save_to_file": {
                        "type": "boolean",
                        "default": True,
                        "description": "Save markdown to session directory"
                    },
                    "save_to_database": {
                        "type": "boolean",
                        "default": True,
                        "description": "Persist to database for FTS search"
                    }
                }
            },
            "examples": [
                {"title": "Feature Implementation Session", "tags": ["feature", "python"]},
                {"include_metrics": False, "save_to_file": True}
            ]
        }

        registry["session_search"] = {
            "implementation": self._wrap_tool(self.session_engine.session_search),
            "description": "Full-text search across session notebooks and summaries",
            "schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (supports FTS5 syntax: AND, OR, NOT, phrases)"
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["fulltext", "tag", "file"],
                        "default": "fulltext",
                        "description": "Type of search to perform"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum results to return"
                    }
                },
                "required": ["query"]
            },
            "examples": [
                {"query": "authentication bug fix"},
                {"query": "python", "search_type": "tag", "limit": 10}
            ]
        }

        # ===== KNOWLEDGE SYSTEM TOOLS =====

        registry["session_log_learning"] = {
            "implementation": self._wrap_tool(self.session_engine.session_log_learning),
            "description": "Log a project-specific learning (pattern, fix, preference, workflow)",
            "schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["error_fix", "pattern", "preference", "workflow"],
                        "description": "Learning category"
                    },
                    "learning_content": {
                        "type": "string",
                        "description": "The actual knowledge/solution"
                    },
                    "trigger_context": {
                        "type": "string",
                        "description": "When to apply this learning (optional)"
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Project scope (uses current if not specified)"
                    }
                },
                "required": ["category", "learning_content"]
            },
            "examples": [
                {
                    "category": "error_fix",
                    "learning_content": "ImportError for module X: install via pip install X",
                    "trigger_context": "When seeing 'ModuleNotFoundError: X'"
                },
                {
                    "category": "pattern",
                    "learning_content": "Always run lint before commit in this project"
                }
            ]
        }

        registry["session_find_solution"] = {
            "implementation": self._wrap_tool(self.session_engine.session_find_solution),
            "description": "Find solutions for an error from project and universal knowledge",
            "schema": {
                "type": "object",
                "properties": {
                    "error_text": {
                        "type": "string",
                        "description": "The error message/pattern to search for"
                    },
                    "error_category": {
                        "type": "string",
                        "enum": ["compile", "runtime", "config", "dependency", "test", "lint"],
                        "description": "Optional category hint"
                    },
                    "include_universal": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to include universal (cross-project) solutions"
                    }
                },
                "required": ["error_text"]
            },
            "examples": [
                {"error_text": "ModuleNotFoundError: No module named 'foo'"},
                {
                    "error_text": "TypeError: expected str, got int",
                    "error_category": "runtime"
                }
            ]
        }

        registry["session_update_solution_outcome"] = {
            "implementation": self._wrap_tool(
                self.session_engine.session_update_solution_outcome
            ),
            "description": "Update success/failure count for a solution after trying it",
            "schema": {
                "type": "object",
                "properties": {
                    "solution_id": {
                        "type": "string",
                        "description": "ID of the solution to update"
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether the solution worked"
                    }
                },
                "required": ["solution_id", "success"]
            },
            "examples": [
                {"solution_id": "sol_abc123", "success": True},
                {"solution_id": "sol_xyz789", "success": False}
            ]
        }

        # ===== AGENT SYSTEM TOOLS =====

        registry["agent_register"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_register),
            "description": "Register or update an agent in the global agent registry",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique agent name (e.g., 'focused-quality-resolver')"
                    },
                    "agent_type": {
                        "type": "string",
                        "description": "Agent type (e.g., 'focused', 'comprehensive', 'micro', 'meta')"
                    },
                    "display_name": {
                        "type": "string",
                        "description": "Human-friendly display name"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of agent's purpose"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional agent metadata (version, author, etc.)"
                    },
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of agent capabilities"
                    }
                },
                "required": ["name", "agent_type"]
            },
            "examples": [
                {
                    "name": "focused-quality-resolver",
                    "agent_type": "focused",
                    "display_name": "Quality Resolver",
                    "description": "Resolves code quality issues",
                    "capabilities": ["lint-fix", "format", "type-check"]
                },
                {
                    "name": "micro-test-runner",
                    "agent_type": "micro"
                }
            ]
        }

        registry["agent_get_info"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_get_info),
            "description": "Get agent information by name or UUID",
            "schema": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Agent name (e.g., 'focused-quality-resolver') or UUID"
                    }
                },
                "required": ["identifier"]
            },
            "examples": [
                {"identifier": "focused-quality-resolver"},
                {"identifier": "550e8400-e29b-41d4-a716-446655440000"}
            ]
        }

        registry["agent_log_decision"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_log_decision),
            "description": "Log a decision made by an agent with context and reasoning",
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the agent making the decision"
                    },
                    "decision_type": {
                        "type": "string",
                        "description": "Category of decision (e.g., 'tool_selection', 'error_handling', 'strategy')"
                    },
                    "context": {
                        "type": "string",
                        "description": "The situation or problem that required a decision"
                    },
                    "decision": {
                        "type": "string",
                        "description": "The decision that was made"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this decision was made"
                    },
                    "alternatives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Other options that were considered"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8,
                        "description": "Confidence level in the decision (0.0-1.0)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization and search"
                    }
                },
                "required": ["agent_name", "decision_type", "context", "decision"]
            },
            "examples": [
                {
                    "agent_name": "focused-quality-resolver",
                    "decision_type": "tool_selection",
                    "context": "Multiple lint errors in Python file",
                    "decision": "Use ruff --fix for auto-fixable issues",
                    "reasoning": "Ruff is faster and handles most common issues",
                    "alternatives": ["Manual fixes", "Black + isort separately"],
                    "confidence": 0.9,
                    "tags": ["python", "linting"]
                }
            ]
        }

        registry["agent_query_decisions"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_query_decisions),
            "description": "Query decisions made by an agent with optional filters",
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the agent to query"
                    },
                    "decision_type": {
                        "type": "string",
                        "description": "Filter by decision type/category"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of results"
                    }
                },
                "required": ["agent_name"]
            },
            "examples": [
                {"agent_name": "focused-quality-resolver"},
                {
                    "agent_name": "focused-quality-resolver",
                    "decision_type": "error_handling",
                    "limit": 10
                }
            ]
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
                        "description": "ID of the decision to update"
                    },
                    "outcome": {
                        "type": "string",
                        "description": "Description of the outcome"
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether the decision led to a successful outcome"
                    }
                },
                "required": ["decision_id", "outcome", "success"]
            },
            "examples": [
                {
                    "decision_id": "dec_abc123",
                    "outcome": "All lint errors fixed successfully",
                    "success": True
                },
                {
                    "decision_id": "dec_xyz789",
                    "outcome": "Auto-fix introduced new errors, needed manual intervention",
                    "success": False
                }
            ]
        }

        registry["agent_log_learning"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_log_learning),
            "description": "Log a learning or knowledge item discovered by an agent",
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the agent logging the learning"
                    },
                    "learning_type": {
                        "type": "string",
                        "description": "Type of learning (e.g., 'pattern', 'anti_pattern', 'technique', 'preference')"
                    },
                    "title": {
                        "type": "string",
                        "description": "Brief title for the learning"
                    },
                    "content": {
                        "type": "string",
                        "description": "Detailed content of the learning"
                    },
                    "source_context": {
                        "type": "string",
                        "description": "The context where this was learned"
                    },
                    "applicability": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Situations where this learning applies"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8,
                        "description": "Confidence level in the learning (0.0-1.0)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization and search"
                    }
                },
                "required": ["agent_name", "learning_type", "title", "content"]
            },
            "examples": [
                {
                    "agent_name": "focused-quality-resolver",
                    "learning_type": "pattern",
                    "title": "Ruff handles import sorting",
                    "content": "Ruff with isort rules enabled can replace separate isort step",
                    "applicability": ["python-projects", "lint-workflows"],
                    "confidence": 0.95,
                    "tags": ["python", "tooling"]
                }
            ]
        }

        registry["agent_query_learnings"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_query_learnings),
            "description": "Query learnings for an agent with optional filters",
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the agent to query"
                    },
                    "learning_type": {
                        "type": "string",
                        "description": "Filter by learning type/category"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum number of results"
                    }
                },
                "required": ["agent_name"]
            },
            "examples": [
                {"agent_name": "focused-quality-resolver"},
                {
                    "agent_name": "focused-quality-resolver",
                    "learning_type": "pattern",
                    "limit": 5
                }
            ]
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
                        "description": "ID of the learning to update"
                    },
                    "times_applied_increment": {
                        "type": "integer",
                        "default": 1,
                        "description": "How many times to increment the application count"
                    },
                    "new_success_rate": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Updated success rate (0.0-1.0)"
                    }
                },
                "required": ["learning_id"]
            },
            "examples": [
                {"learning_id": "lrn_abc123", "times_applied_increment": 1},
                {
                    "learning_id": "lrn_xyz789",
                    "times_applied_increment": 1,
                    "new_success_rate": 0.85
                }
            ]
        }

        registry["agent_create_notebook"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_create_notebook),
            "description": "Create a notebook document for an agent",
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the agent creating the notebook"
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the notebook"
                    },
                    "content": {
                        "type": "string",
                        "description": "Markdown content of the notebook"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary for search/display"
                    },
                    "notebook_type": {
                        "type": "string",
                        "default": "execution",
                        "description": "Type of notebook (e.g., 'execution', 'analysis', 'retrospective')"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context metadata"
                    },
                    "decisions_referenced": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of decisions referenced in this notebook"
                    },
                    "learnings_referenced": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of learnings referenced in this notebook"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization and search"
                    }
                },
                "required": ["agent_name", "title", "content"]
            },
            "examples": [
                {
                    "agent_name": "focused-quality-resolver",
                    "title": "Quality Resolution Session - 2025-01-05",
                    "content": "## Summary\n\nFixed 15 lint issues...",
                    "summary": "Resolved lint issues in src/module.py",
                    "notebook_type": "execution",
                    "tags": ["quality", "python"]
                }
            ]
        }

        registry["agent_query_notebooks"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_query_notebooks),
            "description": "Query notebooks for an agent with optional filters",
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the agent to query"
                    },
                    "notebook_type": {
                        "type": "string",
                        "description": "Filter by notebook type"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of results"
                    }
                },
                "required": ["agent_name"]
            },
            "examples": [
                {"agent_name": "focused-quality-resolver"},
                {
                    "agent_name": "focused-quality-resolver",
                    "notebook_type": "execution",
                    "limit": 5
                }
            ]
        }

        registry["agent_search_all"] = {
            "implementation": self._wrap_async_tool(self.session_engine.agent_search_all),
            "description": "Search across all agent data (decisions, learnings, notebooks)",
            "schema": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the agent to search"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum results per content type"
                    }
                },
                "required": ["agent_name", "query"]
            },
            "examples": [
                {
                    "agent_name": "focused-quality-resolver",
                    "query": "ruff lint"
                },
                {
                    "agent_name": "focused-quality-resolver",
                    "query": "import sorting",
                    "limit": 10
                }
            ]
        }

        return registry

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

    def _setup_meta_tools(self):
        """Setup the 3 meta-tools for dynamic discovery."""

        @self.app.tool(
            description=(
                "Discover session lifecycle, decision logging, agent tracking, "
                "and learning tools (28 total). "
                "USE WHEN: starting sessions, logging decisions, searching learnings"
            )
        )
        def discover_tools(
            pattern: str = ""
        ) -> dict[str, Any]:
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

            This lean interface provides 28 tools across 3 domains (session, agent, knowledge),
            saving ~25k tokens vs loading all tool schemas upfront.

            WORKFLOW:
            1. discover_tools(pattern) <- YOU ARE HERE
            2. get_tool_spec(tool_name) <- Get schema/parameters for a specific tool
            3. execute_tool(tool_name, params) <- Execute the operation

            Args:
                pattern: Filter tools by name (e.g., "session", "agent", "learning")
                         Leave empty "" to see all 28 tools

            Returns:
                Dictionary containing:
                - available_tools: List of tools, each with:
                  * name: Tool name to use in get_tool_spec() or execute_tool()
                  * description: What the tool does
                - total_tools: Total tools in registry (28)
                - filtered_count: How many matched your pattern

                Example output for discover_tools("session"):
                {
                  "available_tools": [
                    {"name": "session_manage_lifecycle", "description": "Complete session lifecycle management with recovery"},
                    {"name": "session_track_execution", "description": "Track agent execution with pattern detection"},
                    {"name": "session_log_decision", "description": "Log decisions with context and impact analysis"}
                  ],
                  "filtered_count": 12,
                  "total_tools": 28
                }

            Examples:
                discover_tools("")              # List all 28 tools
                discover_tools("session")       # Find session management tools (12 tools)
                discover_tools("agent")         # Find agent registry tools (11 tools)
                discover_tools("learning")      # Find learning/knowledge tools

            MISSING TOOL? If you need an operation that's not available:
            File an issue at https://github.com/MementoRC/session-intelligence
            """
            tools = []

            for name, info in self.tool_registry.items():
                # Apply pattern filter if provided
                if pattern and pattern.strip() and pattern.lower() not in name.lower():
                    continue

                tools.append({
                    "name": name,
                    "description": info["description"]
                })

            return {
                "available_tools": tools,
                "total_tools": len(self.tool_registry),
                "filtered_count": len(tools)
            }

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
                      "operation": {"type": "string", "enum": ["create", "resume", "finalize", "validate"]},
                      "mode": {"type": "string", "enum": ["local", "remote", "hybrid", "auto"], "default": "local"},
                      "project_name": {"type": "string", "description": "Project context (optional)"}
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
            https://github.com/MementoRC/session-intelligence
            """
            if tool_name not in self.tool_registry:
                available_tools = list(self.tool_registry.keys())
                return {
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": available_tools
                }

            tool_info = self.tool_registry[tool_name]
            return {
                "name": tool_name,
                "description": tool_info["description"],
                "schema": tool_info["schema"],
                "examples": tool_info["examples"]
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
                  * session_manage_lifecycle: {"session_id": "...", "status": "active", "mode": "local"}
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
            File an issue at https://github.com/MementoRC/session-intelligence
            Include: tool name, parameters used, error message, expected vs actual behavior
            """
            import inspect

            if tool_name not in self.tool_registry:
                available_tools = list(self.tool_registry.keys())
                return {
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": available_tools
                }

            tool_info = self.tool_registry[tool_name]
            tool_func = tool_info["implementation"]

            try:
                # Execute tool - await if async, call directly if sync
                if inspect.iscoroutinefunction(tool_func):
                    result = await tool_func(**parameters)
                else:
                    result = tool_func(**parameters)
                return {
                    "tool": tool_name,
                    "status": "success",
                    "result": result
                }
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                return {
                    "tool": tool_name,
                    "status": "error",
                    "error": str(e)
                }

    def get_app(self) -> FastMCP:
        """Get the FastMCP app instance."""
        return self.app


def create_lean_interface(session_engine: SessionIntelligenceEngine) -> FastMCP:
    """
    Create a lean MCP interface with minimal context consumption.
    
    Args:
        session_engine: Initialized session intelligence engine
        
    Returns:
        FastMCP app with 3 meta-tools exposing full functionality
    """
    lean_interface = LeanMCPInterface(session_engine)
    return lean_interface.get_app()
