"""
Session Intelligence MCP Server - Unified Session Management and Analytics.

Consolidates 42+ scattered claudecode session management functions into 10 unified MCP functions
providing comprehensive session lifecycle management, execution tracking, decision logging,
pattern analysis, and intelligence operations for the Claude Code framework.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Setup file logging for debugging
debug_log_file = Path("/tmp/session-intelligence-debug.log")
debug_logger = logging.getLogger("session_intelligence_debug")
debug_handler = logging.FileHandler(debug_log_file)
debug_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
debug_logger.addHandler(debug_handler)
debug_logger.setLevel(logging.INFO)

from core.session_engine import SessionIntelligenceEngine
from models.session_models import (
    AnalysisScope,
    CommandAnalysisResult,
    CoordinationResult,
    DashboardResult,
    DashboardType,
    DecisionResult,
    ExecutionMode,
    ExecutionTrackingResult,
    MissingFunctionResult,
    OptimizationLevel,
    PatternAnalysisResult,
    SessionHealthResult,
    SessionResult,
    WorkflowResult,
    WorkflowType,
)
from utils.token_limiter import apply_token_limits


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Session Intelligence MCP Server")
    parser.add_argument(
        "--repository",
        type=str,
        default=".",
        help="Repository root path (default: current directory)",
    )
    return parser.parse_args()


# Initialize FastMCP app
app = FastMCP("session-intelligence")

# Global variables for server state
session_engine = None
repository_path = None


def safe_response(response_data: Any, operation: str) -> dict[str, Any]:
    """Helper function to safely return response with token limits."""
    try:
        if hasattr(response_data, "model_dump"):
            response = response_data.model_dump()
        elif isinstance(response_data, dict):
            response = response_data
        else:
            response = {"data": str(response_data)}

        return apply_token_limits(response, operation)
    except Exception as e:
        logger.error(f"Error processing response for {operation}: {e}")
        return {"error": f"Response processing failed: {str(e)}", "operation": operation}


@app.tool()
def session_manage_lifecycle(
    operation: str,
    mode: str = "local",
    project_name: str | None = None,
    metadata: Any | None = None,
    auto_recovery: bool = True,
) -> dict[str, Any]:
    """
    Comprehensive session lifecycle management with intelligent tracking.

    Consolidates: claudecode_create_session_metadata, claudecode_get_or_create_session_id,
                 claudecode_create_session_notes, claudecode_finalize_session_summary,
                 claudecode_save_session_state, claudecode_capture_enhanced_state

    Args:
        operation: Lifecycle operation ("create", "resume", "finalize", "validate")
        mode: Session mode ("local", "remote", "hybrid", "auto")
        project_name: Project context (optional)
        metadata: Additional session metadata as dict or JSON string (optional)
        auto_recovery: Enable automatic recovery (default: True)

    Returns:
        SessionResult with session ID, status, metadata, recovery options
    """
    try:
        # Parse metadata if it's a JSON string
        parsed_metadata = {}
        if metadata:
            if isinstance(metadata, str):
                try:
                    parsed_metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat it as a single key-value pair
                    parsed_metadata = {"raw_metadata": metadata}
            elif isinstance(metadata, dict):
                parsed_metadata = metadata
            else:
                parsed_metadata = {"metadata": str(metadata)}

        result = session_engine.session_manage_lifecycle(
            operation=operation,
            mode=mode,
            project_name=project_name,
            metadata=parsed_metadata,
            auto_recovery=auto_recovery,
        )
        response = result
        return apply_token_limits(response, "session_manage_lifecycle")
    except Exception as e:
        error_response = SessionResult(
            session_id="error",
            operation=operation,
            status="error",
            message=f"Session lifecycle error: {str(e)}",
        )
        return apply_token_limits(error_response, "session_manage_lifecycle")


@app.tool()
def session_track_execution(
    agent_name: str,
    step_data: dict[str, Any],
    session_id: str | None = None,
    track_patterns: bool = True,
    suggest_optimizations: bool = True,
) -> dict[str, Any]:
    """
    Advanced execution tracking with pattern detection and optimization.

    Consolidates: claudecode_initialize_agent_execution_log, claudecode_add_execution_step,
                 claudecode_log_execution_step, claudecode_write_agent_execution_log,
                 claudecode_update_agent_status

    Args:
        agent_name: Agent being executed (required)
        step_data: ExecutionStep details (required)
        session_id: Session ID or current session (optional)
        track_patterns: Enable pattern detection (default: True)
        suggest_optimizations: Generate optimization suggestions (default: True)

    Returns:
        ExecutionTrackingResult with step ID, patterns detected, optimizations
    """
    try:
        debug_logger.info(f"[EXEC-TRACK] Starting execution tracking for agent: {agent_name}")
        debug_logger.info(f"[EXEC-TRACK] Session ID: {session_id}")
        debug_logger.info(f"[EXEC-TRACK] Repository path: {repository_path}")
        debug_logger.info(f"[EXEC-TRACK] Step data: {step_data}")
        debug_logger.info(
            f"[EXEC-TRACK] Session engine claude_sessions_path: {session_engine.claude_sessions_path}"
        )

        result = session_engine.session_track_execution(
            session_id=session_id,
            agent_name=agent_name,
            step_data=step_data,
            track_patterns=track_patterns,
            suggest_optimizations=suggest_optimizations,
        )

        debug_logger.info(f"[EXEC-TRACK] Execution tracking result: {result}")
        debug_logger.info(f"[EXEC-TRACK] Result status: {result.status}")

        return safe_response(result, "session_track_execution")
    except Exception as e:
        debug_logger.error(f"[EXEC-TRACK] Exception in session_track_execution: {e}")
        debug_logger.error(f"[EXEC-TRACK] Exception type: {type(e)}")
        import traceback

        debug_logger.error(f"[EXEC-TRACK] Full traceback: {traceback.format_exc()}")

        return safe_response(
            ExecutionTrackingResult(
                step_id="error",
                session_id=session_id or "unknown",
                agent_name=agent_name,
                status="error",
                patterns_detected=[],
                optimizations=[],
            ),
            "session_manage_lifecycle",
        )


@app.tool()
def session_coordinate_agents(
    agents: list[dict[str, Any]],
    session_id: str | None = None,
    execution_mode: str = "sequential",
    dependency_graph: dict[str, Any] | None = None,
    optimization_level: str = "balanced",
) -> dict[str, Any]:
    """
    Multi-agent coordination with dependency management and parallel execution.

    Consolidates: claudecode_log_agent_start, claudecode_log_agent_complete,
                 claudecode_log_agent_error, claudecode_create_agent_context,
                 claudecode_workflow_dispatch_parallel

    Args:
        agents: Agents to coordinate (required)
        session_id: Session context (optional)
        execution_mode: Execution strategy ("sequential", "parallel", "adaptive")
        dependency_graph: Agent dependencies (optional)
        optimization_level: Optimization approach ("conservative", "balanced", "aggressive")

    Returns:
        CoordinationResult with execution plan, timing, dependency resolution
    """
    try:
        exec_mode = ExecutionMode(execution_mode)
        opt_level = OptimizationLevel(optimization_level)

        result = session_engine.session_coordinate_agents(
            session_id=session_id,
            agents=agents,
            execution_mode=exec_mode,
            dependency_graph=dependency_graph,
            optimization_level=opt_level,
        )
        return safe_response(result, "session_manage_lifecycle")
    except Exception as e:
        return CoordinationResult(
            coordination_id="error",
            session_id=session_id or "unknown",
            execution_plan={"error": str(e)},
            timing_estimate=0,
        )


@app.tool()
def session_log_decision(
    decision: str,
    session_id: str | None = None,
    context: dict[str, Any] | None = None,
    impact_analysis: bool = True,
    link_artifacts: list[str] | None = None,
) -> dict[str, Any]:
    """
    Intelligent decision logging with context and impact analysis.

    Consolidates: claudecode_log_decision, claudecode_log_workflow_step
    Enhanced: Adds decision impact analysis and relationship mapping

    Args:
        decision: Decision description (required)
        session_id: Session context (optional)
        context: Decision context and rationale (optional)
        impact_analysis: Analyze decision impact (default: True)
        link_artifacts: Related files or commits (optional)

    Returns:
        DecisionResult with decision ID, impact analysis, linked decisions
    """
    try:
        result = session_engine.session_log_decision(
            session_id=session_id,
            decision=decision,
            context=context,
            impact_analysis=impact_analysis,
            link_artifacts=link_artifacts,
        )
        return safe_response(result, "session_manage_lifecycle")
    except Exception as e:
        return DecisionResult(
            decision_id="error",
            session_id=session_id or "unknown",
            impact_analysis={"error": str(e)},
        )


@app.tool()
def session_analyze_patterns(
    scope: str = "current",
    pattern_types: list[str] = None,
    include_agents: list[str] | None = None,
    learning_mode: bool = True,
    generate_insights: bool = True,
) -> dict[str, Any]:
    """
    Cross-session pattern analysis with learning and recommendations.

    New Functionality: Advanced analytics not present in individual functions

    Args:
        scope: Analysis scope ("current", "recent", "historical", "all")
        pattern_types: Patterns to analyze (default: ["execution", "errors", "performance"])
        include_agents: Specific agents to analyze (optional)
        learning_mode: Enable ML-based learning (default: True)
        generate_insights: Generate actionable insights (default: True)

    Returns:
        PatternAnalysisResult with patterns, trends, recommendations, learning model
    """
    try:
        if pattern_types is None:
            pattern_types = ["execution", "errors", "performance"]

        analysis_scope = AnalysisScope(scope)

        result = session_engine.session_analyze_patterns(
            scope=analysis_scope,
            pattern_types=pattern_types,
            include_agents=include_agents,
            learning_mode=learning_mode,
            generate_insights=generate_insights,
        )
        return safe_response(result, "session_manage_lifecycle")
    except Exception:
        return safe_response(
            PatternAnalysisResult(
                analysis_id="error",
                scope=AnalysisScope.CURRENT,
                patterns=[],
                trends=[],
                recommendations=[],
            ),
            "session_manage_lifecycle",
        )


@app.tool()
def session_monitor_health(
    session_id: str | None = None,
    health_checks: list[str] = None,
    auto_recover: bool = True,
    alert_thresholds: dict[str, float] | None = None,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """
    Real-time session health monitoring with auto-recovery capabilities.

    Consolidates: claudecode_check_session_health, claudecode_validate_session_files,
                 claudecode_session_continuity_check, claudecode_meta_session_health

    Args:
        session_id: Session to monitor (optional)
        health_checks: Checks to perform (default: ["continuity", "files", "state", "agents"])
        auto_recover: Enable automatic recovery (default: True)
        alert_thresholds: Custom alert thresholds (optional)
        include_diagnostics: Include detailed diagnostics (default: True)

    Returns:
        SessionHealthResult with health score, issues, recovery actions, diagnostics
    """
    try:
        if health_checks is None:
            health_checks = ["continuity", "files", "state", "agents"]

        result = session_engine.session_monitor_health(
            session_id=session_id,
            health_checks=health_checks,
            auto_recover=auto_recover,
            alert_thresholds=alert_thresholds,
            include_diagnostics=include_diagnostics,
        )
        return safe_response(result, "session_manage_lifecycle")
    except Exception as e:
        return SessionHealthResult(
            session_id=session_id or "unknown",
            health_score=0.0,
            issues=[f"Health monitoring error: {str(e)}"],
        )


@app.tool()
def session_orchestrate_workflow(
    workflow_type: str,
    session_id: str | None = None,
    workflow_config: dict[str, Any] | None = None,
    parallel_execution: bool = False,
    optimize_execution: bool = True,
) -> dict[str, Any]:
    """
    Advanced workflow orchestration with state management and optimization.

    Consolidates: claudecode_workflow_init, claudecode_workflow_status,
                 claudecode_workflow_complete, claudecode_workflow_orchestrate_prime

    Args:
        workflow_type: Workflow type ("tdd", "atomic", "quality", "prime", "custom")
        session_id: Session context (optional)
        workflow_config: Workflow configuration (optional)
        parallel_execution: Enable parallel execution (default: False)
        optimize_execution: Optimize execution order (default: True)

    Returns:
        WorkflowResult with execution plan, state, progress, optimizations
    """
    try:
        wf_type = WorkflowType(workflow_type)

        result = session_engine.session_orchestrate_workflow(
            workflow_type=wf_type,
            session_id=session_id,
            workflow_config=workflow_config,
            parallel_execution=parallel_execution,
            optimize_execution=optimize_execution,
        )
        return safe_response(result, "session_manage_lifecycle")
    except Exception as e:
        return WorkflowResult(
            workflow_id="error",
            session_id=session_id or "unknown",
            execution_plan={"error": str(e)},
            state=None,
        )


@app.tool()
def session_analyze_commands(
    session_id: str | None = None,
    command_types: list[str] = None,
    detect_inefficiencies: bool = True,
    suggest_alternatives: bool = True,
    include_timing: bool = True,
) -> dict[str, Any]:
    """
    Analyze hook-based command logs for patterns and inefficiencies.

    New Functionality: Advanced command analysis from hook logs

    Args:
        session_id: Session to analyze (optional)
        command_types: Command categories (default: ["git", "test", "quality"])
        detect_inefficiencies: Detect inefficient patterns (default: True)
        suggest_alternatives: Suggest better approaches (default: True)
        include_timing: Include timing analysis (default: True)

    Returns:
        CommandAnalysisResult with patterns, inefficiencies, suggestions, metrics
    """
    try:
        if command_types is None:
            command_types = ["git", "test", "quality"]

        result = session_engine.session_analyze_commands(
            session_id=session_id,
            command_types=command_types,
            detect_inefficiencies=detect_inefficiencies,
            suggest_alternatives=suggest_alternatives,
            include_timing=include_timing,
        )
        return safe_response(result, "session_manage_lifecycle")
    except Exception:
        return safe_response(
            CommandAnalysisResult(
                session_id=session_id or "unknown",
                analysis_period="current",
                patterns=[],
                inefficiencies=[],
                suggestions=[],
                metrics={},
            ),
            "session_manage_lifecycle",
        )


@app.tool()
def session_track_missing_functions(
    session_id: str | None = None,
    auto_suggest: bool = True,
    priority_analysis: bool = True,
    generate_report: bool = True,
) -> dict[str, Any]:
    """
    Track and analyze missing functions for ecosystem improvement.

    Consolidates: claudecode_find_missing_functions, claudecode_log_missing_script,
                 claudecode_write_missing_scripts

    Args:
        session_id: Session context (optional)
        auto_suggest: Suggest function implementations (default: True)
        priority_analysis: Analyze implementation priority (default: True)
        generate_report: Generate missing function report (default: True)

    Returns:
        MissingFunctionResult with functions, priorities, suggestions, impact
    """
    try:
        result = session_engine.session_track_missing_functions(
            session_id=session_id,
            auto_suggest=auto_suggest,
            priority_analysis=priority_analysis,
            generate_report=generate_report,
        )
        return safe_response(result, "session_manage_lifecycle")
    except Exception:
        return safe_response(
            MissingFunctionResult(
                session_id=session_id or "unknown",
                functions=[],
                priorities={},
                suggestions=[],
                impact={},
            ),
            "session_manage_lifecycle",
        )


@app.tool()
def session_get_dashboard(
    dashboard_type: str = "overview",
    session_id: str | None = None,
    real_time: bool = False,
    export_format: str | None = None,
) -> dict[str, Any]:
    """
    Comprehensive session intelligence dashboard with real-time insights.

    New Functionality: Unified intelligence dashboard

    Args:
        dashboard_type: View type ("overview", "performance", "agents", "decisions", "health")
        session_id: Session or cross-session view (optional)
        real_time: Enable real-time updates (default: False)
        export_format: Export format ("json", "html", "markdown") (optional)

    Returns:
        DashboardResult with metrics, visualizations, insights, recommendations
    """
    try:
        dash_type = DashboardType(dashboard_type)

        result = session_engine.session_get_dashboard(
            dashboard_type=dash_type,
            session_id=session_id,
            real_time=real_time,
            export_format=export_format,
        )
        return safe_response(result, "session_manage_lifecycle")
    except Exception as e:
        return DashboardResult(
            dashboard_type=DashboardType.OVERVIEW,
            session_id=session_id,
            metrics={"error": str(e)},
            visualizations=[],
            insights=[],
            recommendations=[],
        )


def initialize_server():
    """Initialize the server with command line arguments."""
    global session_engine, repository_path

    try:
        args = parse_args()
        repository_path = Path(args.repository).resolve()
    except SystemExit:
        # Fallback when argparse fails (e.g., in MCP mode)
        repository_path = Path(".").resolve()

    # Initialize session intelligence engine with repository path
    session_engine = SessionIntelligenceEngine(repository_path=str(repository_path))


if __name__ == "__main__":
    # Initialize server with command line arguments
    initialize_server()

    # For development/testing
    app.run()
else:
    # When imported as a module, initialize with current directory
    initialize_server()
