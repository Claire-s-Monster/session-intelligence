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
            "implementation": self._wrap_tool(self.session_engine.session_create_notebook),
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

    def _setup_meta_tools(self):
        """Setup the 3 meta-tools for dynamic discovery."""

        @self.app.tool()
        def discover_tools(
            pattern: str = ""
        ) -> dict[str, Any]:
            """
            Tools available with session-intelligence MCP server.
            
            Args:
                pattern: Filter by name pattern (substring match, empty string for all tools)
            
            Returns:
                Compact tool list with names and brief descriptions
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

        @self.app.tool()
        def get_tool_spec(tool_name: str) -> dict[str, Any]:
            """
            Get full specification for specific session-intelligence tool including schema and examples.
            
            Args:
                tool_name: Name of tool to get specification for
            
            Returns:
                Complete tool specification with schema, examples, and usage notes
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

        @self.app.tool()
        def execute_tool(tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
            """
            Execute session-intelligence tool with parameters using dynamic dispatch.
            
            Args:
                tool_name: Name of tool to execute
                parameters: Tool parameters as object
            
            Returns:
                Tool execution result with standard error handling
            """
            if tool_name not in self.tool_registry:
                available_tools = list(self.tool_registry.keys())
                return {
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": available_tools
                }

            tool_info = self.tool_registry[tool_name]
            tool_func = tool_info["implementation"]

            try:
                # Execute tool with parameters
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
