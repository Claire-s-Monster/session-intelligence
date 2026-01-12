"""
Session Intelligence Engine - Core business logic for session management and analytics.

This engine consolidates the functionality of 42+ scattered claudecode session functions
into a unified, intelligent system with pattern recognition, optimization, and learning capabilities.
"""

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Setup file logging for debugging
debug_log_file = Path("/tmp/session-intelligence-debug.log")
debug_logger = logging.getLogger("session_intelligence_engine_debug")
debug_handler = logging.FileHandler(debug_log_file)
debug_handler.setFormatter(logging.Formatter("%(asctime)s [ENGINE-DEBUG] %(message)s"))
debug_logger.addHandler(debug_handler)
debug_logger.setLevel(logging.INFO)

from models.session_models import (
    Agent,
    AgentDecision,
    AgentDecisionResult,
    AgentExecution,
    AgentLearning,
    AgentLearningResult,
    AgentNotebook,
    AgentNotebookResult,
    AgentRegistrationResult,
    AnalysisScope,
    CommandAnalysisResult,
    CoordinationResult,
    DashboardResult,
    DashboardType,
    Decision,
    DecisionResult,
    ExecutionMode,
    ExecutionStatus,
    ExecutionStep,
    ExecutionTrackingResult,
    HealthStatus,
    ImpactLevel,
    LearningCategory,
    LearningResult,
    MissingFunctionResult,
    NotebookResult,
    NotebookSection,
    Optimization,
    OptimizationLevel,
    Pattern,
    PatternAnalysis,
    PatternAnalysisResult,
    PatternType,
    PerformanceMetrics,
    ProjectLearning,
    SearchResult,
    SearchResults,
    Session,
    SessionHealthResult,
    SessionMetadata,
    SessionNotebook,
    SessionResult,
    SessionStatus,
    SolutionResult,
    SolutionSearchResult,
    WorkflowResult,
    WorkflowState,
    WorkflowType,
)


class SessionIntelligenceEngine:
    """
    Core session intelligence engine providing unified session management,
    execution tracking, pattern analysis, and optimization capabilities.

    Consolidates 42+ claudecode session functions into intelligent operations.
    """

    def __init__(
        self,
        repository_path: str | None = None,
        use_filesystem: bool = True,
        database: Any | None = None,
    ):
        """Initialize the session intelligence engine.

        Args:
            repository_path: Path to the repository root. If None, auto-detects project root.
            use_filesystem: If True, persist sessions to filesystem. If False, use memory only.
                           Set to False for HTTP transport to avoid creating local folders.
            database: Optional async database for persistence (used by HTTP server).
        """
        debug_logger.info(
            f"SessionIntelligenceEngine.__init__ called with repository_path: {repository_path}, use_filesystem: {use_filesystem}"
        )

        self.session_cache: dict[str, Session] = {}
        self.pattern_cache: dict[str, list[PatternAnalysis]] = {}
        self.use_filesystem = use_filesystem
        self.database = database  # Optional database for persistence
        self._current_session_id: str | None = None

        # Use provided repository path or auto-detect project directory
        if repository_path:
            self.claude_sessions_path = Path(repository_path) / ".claude" / "session-intelligence"
            debug_logger.info(f"Using provided repository_path: {repository_path}")
        else:
            self.claude_sessions_path = self._get_project_session_path()
            debug_logger.info(
                f"Auto-detected project path, claude_sessions_path: {self.claude_sessions_path}"
            )

        debug_logger.info(f"Final claude_sessions_path: {self.claude_sessions_path}")

        # Only create filesystem directories if filesystem persistence is enabled
        if self.use_filesystem:
            self.claude_sessions_path.mkdir(parents=True, exist_ok=True)
            debug_logger.info(
                f"Created/ensured session directory exists at: {self.claude_sessions_path}"
            )
        else:
            debug_logger.info("Filesystem persistence disabled - using memory only")

    def _get_or_create_current_session_id(self) -> str | None:
        """Get current session ID from cache/file, or create new session if needed."""
        # Check in-memory cache first
        if self._current_session_id and self._current_session_id in self.session_cache:
            debug_logger.info(f"Using cached current session ID: {self._current_session_id}")
            return self._current_session_id

        # If filesystem enabled, try to read from file
        if self.use_filesystem:
            current_session_file = self.claude_sessions_path.parent / "current-session-id"
            debug_logger.info(f"Looking for current session ID in: {current_session_file}")

            if current_session_file.exists():
                try:
                    session_id = current_session_file.read_text().strip()
                    debug_logger.info(f"Found existing session ID: {session_id}")

                    if session_id in self.session_cache:
                        debug_logger.info(f"Session {session_id} found in cache")
                        self._current_session_id = session_id
                        return session_id

                    # Try to load session from disk
                    session_dir = self.claude_sessions_path / session_id
                    metadata_file = session_dir / "session-metadata.json"

                    if session_dir.exists() and metadata_file.exists():
                        debug_logger.info(f"Session {session_id} found on disk, loading to cache")
                        with open(metadata_file) as f:
                            session_data = json.load(f)
                        from models.session_models import Session

                        session = Session.model_validate(session_data)
                        self.session_cache[session_id] = session
                        self._current_session_id = session_id
                        return session_id
                    else:
                        debug_logger.warning(
                            f"Session {session_id} not found on disk, creating new session"
                        )

                except Exception as e:
                    debug_logger.error(f"Error reading current session ID: {e}")

        # Create new session if none exists or is valid
        debug_logger.info("Creating new session")
        result = self._create_session(
            mode="auto",
            project_name=self.claude_sessions_path.parent.name,
            metadata={"project_path": str(self.claude_sessions_path.parent)},
        )

        if result.status == "success":
            self._current_session_id = result.session_id
            # Only write to file if filesystem is enabled
            if self.use_filesystem:
                current_session_file = self.claude_sessions_path.parent / "current-session-id"
                current_session_file.write_text(result.session_id + "\n")
                debug_logger.info(f"Saved new session ID to file: {result.session_id}")
            else:
                debug_logger.info(f"New session ID (memory only): {result.session_id}")
            return result.session_id

        debug_logger.error(f"Failed to create session: {result.message}")
        return None

    def _get_project_session_path(self) -> Path:
        """
        Get the session intelligence path for the current project.

        Looks for project root by finding common markers, then uses
        .claude/session-intelligence within that project.
        """
        current_path = Path.cwd()

        # Look for common project root markers
        project_markers = [
            ".git",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            ".project",
            "composer.json",
        ]

        # Start from current directory and walk up to find project root
        for path in [current_path] + list(current_path.parents):
            for marker in project_markers:
                if (path / marker).exists():
                    return path / ".claude" / "session-intelligence"

        # Fallback to current directory if no project root found
        return current_path / ".claude" / "session-intelligence"

    # ===== SESSION LIFECYCLE MANAGEMENT =====

    def session_manage_lifecycle(
        self,
        operation: str,
        mode: str = "local",
        project_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        auto_recovery: bool = True,
    ) -> SessionResult:
        """
        Comprehensive session lifecycle management with intelligent tracking.

        Consolidates: claudecode_create_session_metadata, claudecode_get_or_create_session_id,
                     claudecode_create_session_notes, claudecode_finalize_session_summary,
                     claudecode_save_session_state, claudecode_capture_enhanced_state
        """
        try:
            return self._manage_lifecycle_sync(
                operation, mode, project_name, metadata, auto_recovery
            )
        except Exception as e:
            return SessionResult(
                session_id="error",
                operation=operation,
                status="error",
                message=f"Session lifecycle error: {str(e)}",
            )

    def _manage_lifecycle_sync(
        self,
        operation: str,
        mode: str,
        project_name: str | None,
        metadata: dict[str, Any] | None,
        auto_recovery: bool,
    ) -> SessionResult:
        """Synchronous session lifecycle management."""

        if operation == "create":
            return self._create_session(mode, project_name, metadata or {})
        elif operation == "resume":
            return self._resume_session(auto_recovery)
        elif operation == "finalize":
            return self._finalize_session()
        elif operation == "validate":
            return self._validate_session()
        else:
            return SessionResult(
                session_id="error",
                operation=operation,
                status="error",
                message=f"Unknown operation: {operation}",
            )

    def _create_session(
        self, mode: str, project_name: str, metadata: dict[str, Any]
    ) -> SessionResult:
        """Create a new session with comprehensive setup."""
        session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Create session metadata
        session_metadata = SessionMetadata(
            session_type="development",
            environment=mode,
            user=metadata.get("user", "claude"),
            git_branch=metadata.get("git_branch"),
            git_commit=metadata.get("git_commit"),
            tags=metadata.get("tags", []),
        )

        # Create session object
        session = Session(
            id=session_id,
            started=datetime.now(UTC),
            mode=mode,
            project_name=project_name or "unknown",
            project_path=metadata.get("project_path", str(Path.cwd())),
            metadata=session_metadata,
            health_status=HealthStatus(),
            performance_metrics=PerformanceMetrics(),
        )

        # Cache session in memory
        self.session_cache[session_id] = session

        # Only create filesystem artifacts if enabled
        if self.use_filesystem:
            session_dir = self.claude_sessions_path / session_id
            session_dir.mkdir(exist_ok=True)
            (session_dir / "agents").mkdir(exist_ok=True)

            metadata_file = session_dir / "session-metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(session.model_dump(), f, indent=2, default=str)
            debug_logger.info(f"Saved session to filesystem: {session_dir}")
        else:
            debug_logger.info(f"Session {session_id} created in memory only")

        return SessionResult(
            session_id=session_id,
            operation="create",
            status="success",
            message=f"Session {session_id} created successfully",
            session_data=session,
            next_steps=["Initialize agent tracking", "Set up workflow state"],
        )

    def _resume_session(self, auto_recovery: bool) -> SessionResult:
        """Resume an existing session with recovery if needed."""
        # First check in-memory cache
        if self.session_cache:
            session_id = list(self.session_cache.keys())[-1]
            self._current_session_id = session_id
            return SessionResult(
                session_id=session_id,
                operation="resume",
                status="success",
                message=f"Resumed session {session_id} from cache",
                recovery_options=["Validate continuity", "Check health"] if auto_recovery else [],
            )

        # If filesystem enabled, try to load from disk
        if self.use_filesystem:
            try:
                session_dirs = [d for d in self.claude_sessions_path.iterdir() if d.is_dir()]
                if session_dirs:
                    latest_session_dir = max(session_dirs, key=lambda x: x.stat().st_mtime)
                    session_id = latest_session_dir.name

                    metadata_file = latest_session_dir / "session-metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            session_data = json.load(f)
                            session = Session.model_validate(session_data)
                            self.session_cache[session_id] = session
                            self._current_session_id = session_id

                    return SessionResult(
                        session_id=session_id,
                        operation="resume",
                        status="success",
                        message=f"Resumed session {session_id} from filesystem",
                        recovery_options=(
                            ["Validate continuity", "Check health"] if auto_recovery else []
                        ),
                    )
            except Exception as e:
                debug_logger.error(f"Error resuming from filesystem: {e}")

        return SessionResult(
            session_id="none",
            operation="resume",
            status="error",
            message="No existing sessions found",
        )

    def _finalize_session(self) -> SessionResult:
        """Finalize current session with comprehensive summary."""
        # Get current session ID
        session_id = self._get_or_create_current_session_id()

        if not session_id or session_id not in self.session_cache:
            return SessionResult(
                session_id="none",
                operation="finalize",
                status="error",
                message="No active session to finalize",
            )

        session = self.session_cache[session_id]

        # Update session status
        session.status = SessionStatus.COMPLETED
        session.completed = datetime.now(UTC)

        # Calculate final metrics
        total_time = (session.completed - session.started).total_seconds() * 1000
        session.performance_metrics.total_execution_time_ms = int(total_time)

        # Clear in-memory current session
        self._current_session_id = None

        # Only do filesystem operations if enabled
        if self.use_filesystem:
            session_dir = self.claude_sessions_path / session_id
            metadata_file = session_dir / "session-metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(session.model_dump(), f, indent=2, default=str)

            current_session_file = self.claude_sessions_path.parent / "current-session-id"
            try:
                if current_session_file.exists():
                    current_session_file.unlink()
                    debug_logger.info(f"Removed current-session-id file: {current_session_file}")
            except Exception as e:
                debug_logger.error(f"Error removing current-session-id file: {e}")
        else:
            debug_logger.info(f"Session {session_id} finalized in memory only")

        return SessionResult(
            session_id=session_id,
            operation="finalize",
            status="success",
            message=f"Session {session_id} finalized successfully",
            session_data=session,
        )

    def _validate_session(self) -> SessionResult:
        """Validate session continuity and health."""
        if not self.session_cache:
            return SessionResult(
                session_id="none",
                operation="validate",
                status="error",
                message="No active session to validate",
            )

        session_id = list(self.session_cache.keys())[-1]
        session = self.session_cache[session_id]

        # Perform validation checks
        issues = []

        # Only check filesystem if enabled
        if self.use_filesystem:
            session_dir = self.claude_sessions_path / session_id
            if not session_dir.exists():
                issues.append("Session directory missing")
            if not (session_dir / "session-metadata.json").exists():
                issues.append("Session metadata file missing")

        # Update health status
        session.health_status.continuity_valid = len(issues) == 0
        session.health_status.files_valid = len(issues) == 0
        session.health_status.overall_score = 100.0 if not issues else 50.0
        session.health_status.issues = issues

        status = "success" if not issues else "warning"
        message = (
            "Session validation passed" if not issues else f"Validation issues: {', '.join(issues)}"
        )

        return SessionResult(
            session_id=session_id,
            operation="validate",
            status=status,
            message=message,
            session_data=session,
        )

    # ===== EXECUTION TRACKING =====

    def session_track_execution(
        self,
        session_id: str | None,
        agent_name: str,
        step_data: dict[str, Any],
        track_patterns: bool = True,
        suggest_optimizations: bool = True,
    ) -> ExecutionTrackingResult:
        """
        Advanced execution tracking with pattern detection and optimization.

        Consolidates: claudecode_initialize_agent_execution_log, claudecode_add_execution_step,
                     claudecode_log_execution_step, claudecode_write_agent_execution_log,
                     claudecode_update_agent_status
        """
        try:
            return self._track_execution_sync(
                session_id, agent_name, step_data, track_patterns, suggest_optimizations
            )
        except Exception:
            return ExecutionTrackingResult(
                step_id="error",
                session_id=session_id or "unknown",
                agent_name=agent_name,
                status="error",
                patterns_detected=[],
                optimizations=[],
            )

    def _track_execution_sync(
        self,
        session_id: str | None,
        agent_name: str,
        step_data: dict[str, Any],
        track_patterns: bool,
        suggest_optimizations: bool,
    ) -> ExecutionTrackingResult:
        """Synchronous execution tracking."""

        debug_logger.info("_track_execution_sync called")
        debug_logger.info(f"session_id: {session_id}")
        debug_logger.info(f"agent_name: {agent_name}")
        debug_logger.info(f"step_data: {step_data}")
        debug_logger.info(f"claude_sessions_path: {self.claude_sessions_path}")
        debug_logger.info(f"session_cache keys: {list(self.session_cache.keys())}")

        # Get current session ID (auto-detect from file if not provided)
        if not session_id:
            session_id = self._get_or_create_current_session_id()
            debug_logger.info(f"Auto-detected session_id: {session_id}")

        if not session_id:
            debug_logger.error("ERROR: No session_id available after auto-detection")
            return ExecutionTrackingResult(
                step_id="error",
                session_id="unknown",
                agent_name=agent_name,
                status="error-no-session",
                patterns_detected=[],
                optimizations=[],
            )

        if session_id not in self.session_cache:
            debug_logger.error(f"ERROR: session_id {session_id} not in cache")
            debug_logger.error(f"Available sessions: {list(self.session_cache.keys())}")
            return ExecutionTrackingResult(
                step_id="error",
                session_id=session_id,
                agent_name=agent_name,
                status="error-session-not-found",
                patterns_detected=[],
                optimizations=[],
            )

        session = self.session_cache[session_id]
        debug_logger.info(f"Found session in cache: {session.id}")
        debug_logger.info(f"Session project_path: {session.project_path}")
        debug_logger.info(f"Session agents_executed count: {len(session.agents_executed)}")

        # Create execution step
        step_id = f"{agent_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        debug_logger.info(f"Created step_id: {step_id}")

        execution_step = ExecutionStep(
            step_id=step_id,
            step_number=len(session.agents_executed) + 1,
            agent=agent_name,
            operation=step_data.get("operation", "unknown"),
            description=step_data.get("description", ""),
            tools_used=step_data.get("tools_used", []),
            started=datetime.now(UTC),
            status=ExecutionStatus.RUNNING,
        )
        debug_logger.info(f"Created execution_step: {execution_step}")

        # Pattern detection
        patterns = []
        if track_patterns:
            patterns = self._detect_patterns(agent_name, step_data)

        # Optimization suggestions
        optimizations = []
        if suggest_optimizations:
            optimizations = self._suggest_optimizations(agent_name, step_data)

        execution_step.patterns_detected = patterns
        execution_step.optimizations_available = optimizations

        # Find or create agent execution
        agent_execution = None
        for agent_exec in session.agents_executed:
            if agent_exec.agent_name == agent_name and agent_exec.status == ExecutionStatus.RUNNING:
                agent_execution = agent_exec
                break

        if not agent_execution:
            from models.session_models import AgentContext, AgentPerformance

            agent_execution = AgentExecution(
                agent_name=agent_name,
                agent_type=step_data.get("agent_type", "unknown"),
                execution_id=f"{agent_name}-{uuid.uuid4().hex[:8]}",
                started=datetime.now(UTC),
                context=AgentContext(
                    session_id=session_id,
                    project_path=session.project_path,
                    working_directory=step_data.get("working_directory", session.project_path),
                ),
                performance=AgentPerformance(),
            )
            session.agents_executed.append(agent_execution)

        # Add step to agent execution
        agent_execution.execution_steps.append(execution_step)
        debug_logger.info("Added execution_step to agent_execution")
        debug_logger.info(f"Agent execution steps count: {len(agent_execution.execution_steps)}")

        # Update performance metrics
        session.performance_metrics.agents_executed = len(session.agents_executed)
        debug_logger.info(
            f"Updated performance metrics - agents executed: {session.performance_metrics.agents_executed}"
        )

        # Save session data to disk only if filesystem is enabled
        if self.use_filesystem:
            try:
                session_dir = self.claude_sessions_path / session_id
                session_dir.mkdir(parents=True, exist_ok=True)
                debug_logger.info(f"Ensured session directory exists: {session_dir}")

                # Save session metadata
                metadata_file = session_dir / "session-metadata.json"
                with open(metadata_file, "w") as f:
                    json.dump(session.model_dump(), f, indent=2, default=str)
                debug_logger.info(f"Saved session metadata to: {metadata_file}")

                # Save individual agent execution log
                agent_dir = session_dir / "agents" / agent_name
                agent_dir.mkdir(parents=True, exist_ok=True)
                agent_log_file = agent_dir / "execution-log.json"
                with open(agent_log_file, "w") as f:
                    json.dump(agent_execution.model_dump(), f, indent=2, default=str)
                debug_logger.info(f"Saved agent execution log to: {agent_log_file}")

            except Exception as save_error:
                debug_logger.error(f"ERROR saving session/agent data: {save_error}")
                import traceback

                debug_logger.error(f"Save error traceback: {traceback.format_exc()}")
        else:
            debug_logger.info("Execution tracking updated in memory only (filesystem disabled)")

        result = ExecutionTrackingResult(
            step_id=step_id,
            session_id=session_id,
            agent_name=agent_name,
            status="success",
            patterns_detected=patterns,
            optimizations=optimizations,
        )
        debug_logger.info(f"Returning ExecutionTrackingResult: {result}")
        return result

    def _detect_patterns(self, agent_name: str, step_data: dict[str, Any]) -> list[Pattern]:
        """Detect execution patterns for optimization."""
        patterns = []

        # Simple pattern detection (can be enhanced with ML)
        if "error" in step_data.get("description", "").lower():
            patterns.append(
                Pattern(
                    pattern_id=f"error-pattern-{uuid.uuid4().hex[:8]}",
                    pattern_type=PatternType.ERROR,
                    description="Error pattern detected in execution",
                    frequency=1,
                    confidence=0.8,
                    impact="negative",
                )
            )

        if step_data.get("duration_ms", 0) > 30000:  # >30 seconds
            patterns.append(
                Pattern(
                    pattern_id=f"performance-pattern-{uuid.uuid4().hex[:8]}",
                    pattern_type=PatternType.PERFORMANCE,
                    description="Long execution time detected",
                    frequency=1,
                    confidence=0.9,
                    impact="negative",
                )
            )

        return patterns

    def _suggest_optimizations(
        self, agent_name: str, step_data: dict[str, Any]
    ) -> list[Optimization]:
        """Suggest optimizations based on execution data."""
        optimizations = []

        # Simple optimization suggestions
        if step_data.get("tools_used") and len(step_data["tools_used"]) > 5:
            optimizations.append(
                Optimization(
                    optimization_id=f"tool-opt-{uuid.uuid4().hex[:8]}",
                    description="Consider batching tool calls to reduce overhead",
                    potential_impact="Reduce execution time by 20-30%",
                    effort_level="low",
                    confidence=0.7,
                )
            )

        return optimizations

    # ===== AGENT COORDINATION =====

    def session_coordinate_agents(
        self,
        session_id: str | None,
        agents: list[dict[str, Any]],
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        dependency_graph: dict[str, Any] | None = None,
        optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
    ) -> CoordinationResult:
        """
        Multi-agent coordination with dependency management and parallel execution.

        Consolidates: claudecode_log_agent_start, claudecode_log_agent_complete,
                     claudecode_log_agent_error, claudecode_create_agent_context,
                     claudecode_workflow_dispatch_parallel
        """
        try:
            return self._coordinate_agents_sync(
                session_id, agents, execution_mode, dependency_graph, optimization_level
            )
        except Exception as e:
            return CoordinationResult(
                coordination_id="error",
                session_id=session_id or "unknown",
                execution_plan={"error": str(e)},
                timing_estimate=0,
            )

    def _coordinate_agents_sync(
        self,
        session_id: str | None,
        agents: list[dict[str, Any]],
        execution_mode: ExecutionMode,
        dependency_graph: dict[str, Any] | None,
        optimization_level: OptimizationLevel,
    ) -> CoordinationResult:
        """Synchronous agent coordination."""

        coordination_id = f"coord-{uuid.uuid4().hex[:8]}"

        # Create execution plan
        execution_plan = {
            "mode": execution_mode.value,
            "agents": [agent.get("name", "unknown") for agent in agents],
            "optimization_level": optimization_level.value,
            "created_at": datetime.now().isoformat(),
        }

        # Dependency resolution
        dependency_resolution = []
        if dependency_graph:
            dependency_resolution = self._resolve_dependencies(agents, dependency_graph)

        # Parallel execution grouping
        parallel_groups = []
        if execution_mode == ExecutionMode.PARALLEL:
            parallel_groups = [agent.get("name", "unknown") for agent in agents]
        elif execution_mode == ExecutionMode.SEQUENTIAL:
            parallel_groups = [[agent.get("name", "unknown")] for agent in agents]

        # Estimate timing
        timing_estimate = len(agents) * 5000  # 5 seconds per agent estimate
        if execution_mode == ExecutionMode.PARALLEL:
            timing_estimate = max(5000, timing_estimate // len(agents))

        return CoordinationResult(
            coordination_id=coordination_id,
            session_id=session_id or "unknown",
            execution_plan=execution_plan,
            timing_estimate=timing_estimate,
            dependency_resolution=dependency_resolution,
            parallel_execution_groups=[parallel_groups] if parallel_groups else [],
        )

    def _resolve_dependencies(
        self, agents: list[dict[str, Any]], dependency_graph: dict[str, Any]
    ) -> list[str]:
        """Resolve agent dependencies for optimal execution order."""
        # Simple dependency resolution (can be enhanced)
        resolved = []
        agent_names = [agent.get("name", "unknown") for agent in agents]

        for agent_name in agent_names:
            dependencies = dependency_graph.get(agent_name, [])
            if all(dep in resolved for dep in dependencies):
                resolved.append(agent_name)

        return resolved

    # ===== DECISION LOGGING =====

    def session_log_decision(
        self,
        decision: str,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
        impact_analysis: bool = True,
        link_artifacts: list[str] | None = None,
    ) -> DecisionResult:
        """
        Intelligent decision logging with context and impact analysis.

        Consolidates: claudecode_log_decision, claudecode_log_workflow_step
        Enhanced: Adds decision impact analysis and relationship mapping
        """
        try:
            return self._log_decision_sync(
                decision, session_id, context, impact_analysis, link_artifacts
            )
        except Exception as e:
            return DecisionResult(
                decision_id="error",
                session_id=session_id or "unknown",
                impact_analysis={"error": str(e)},
            )

    def _log_decision_sync(
        self,
        decision: str,
        session_id: str | None,
        context: dict[str, Any] | None,
        impact_analysis: bool,
        link_artifacts: list[str] | None,
    ) -> DecisionResult:
        """Synchronous decision logging."""

        decision_id = f"decision-{uuid.uuid4().hex[:8]}"

        # Get current session
        if not session_id and self.session_cache:
            session_id = list(self.session_cache.keys())[-1]

        if session_id and session_id in self.session_cache:
            session = self.session_cache[session_id]

            from models.session_models import DecisionContext

            decision_context = DecisionContext(session_id=session_id, project_state=context or {})

            decision_obj = Decision(
                decision_id=decision_id,
                timestamp=datetime.now(UTC),
                description=decision,
                context=decision_context,
                impact_level=ImpactLevel.MEDIUM,
                artifacts=link_artifacts or [],
            )

            session.decisions.append(decision_obj)

            # Persist to database
            if self.database:
                decision_data = {
                    "id": decision_id,
                    "session_id": session_id,
                    "timestamp": decision_obj.timestamp.isoformat(),
                    "description": decision,
                    "context": json.dumps(context or {}),
                    "impact_level": decision_obj.impact_level.value,
                    "artifacts": json.dumps(link_artifacts or []),
                }
                try:
                    import asyncio

                    asyncio.get_running_loop()
                    asyncio.create_task(self.database.save_decision(decision_data))
                except RuntimeError:
                    pass  # No event loop in sync context

        # Impact analysis
        impact_analysis_result = {}
        if impact_analysis:
            impact_analysis_result = {
                "estimated_impact": "medium",
                "affected_components": [],
                "risk_assessment": "low",
            }

        return DecisionResult(
            decision_id=decision_id,
            session_id=session_id or "unknown",
            impact_analysis=impact_analysis_result,
            linked_decisions=[],
            predicted_outcomes=["Continue with planned execution"],
        )

    def session_track_file_operation(
        self,
        operation: str,
        file_path: str,
        session_id: str | None = None,
        lines_added: int = 0,
        lines_removed: int = 0,
        summary: str | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any]:
        """Track a file operation for the session notebook."""
        if not session_id and self.session_cache:
            session_id = list(self.session_cache.keys())[-1]

        if not session_id:
            return {"status": "error", "message": "No active session"}

        file_op_data = {
            "session_id": session_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "operation": operation,
            "file_path": file_path,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "summary": summary,
            "tool_name": tool_name,
        }

        if self.database:
            try:
                import asyncio

                asyncio.get_running_loop()
                asyncio.create_task(self.database.save_file_operation(file_op_data))
            except RuntimeError:
                pass

        return {
            "status": "success",
            "session_id": session_id,
            "operation": operation,
            "file_path": file_path,
        }

    # ===== HEALTH MONITORING =====

    def session_monitor_health(
        self,
        session_id: str | None,
        health_checks: list[str] = None,
        auto_recover: bool = True,
        alert_thresholds: dict[str, float] | None = None,
        include_diagnostics: bool = True,
    ) -> SessionHealthResult:
        """
        Real-time session health monitoring with auto-recovery capabilities.

        Consolidates: claudecode_check_session_health, claudecode_validate_session_files,
                     claudecode_session_continuity_check, claudecode_meta_session_health
        """
        if health_checks is None:
            health_checks = ["continuity", "files", "state", "agents"]

        try:
            return self._monitor_health_sync(
                session_id, health_checks, auto_recover, alert_thresholds, include_diagnostics
            )
        except Exception as e:
            return SessionHealthResult(
                session_id=session_id or "unknown",
                health_score=0.0,
                issues=[f"Health monitoring error: {str(e)}"],
            )

    def _monitor_health_sync(
        self,
        session_id: str | None,
        health_checks: list[str],
        auto_recover: bool,
        alert_thresholds: dict[str, float] | None,
        include_diagnostics: bool,
    ) -> SessionHealthResult:
        """Synchronous health monitoring."""

        # Get current session
        if not session_id and self.session_cache:
            session_id = list(self.session_cache.keys())[-1]

        if not session_id or session_id not in self.session_cache:
            return SessionHealthResult(
                session_id=session_id or "unknown",
                health_score=0.0,
                issues=["No active session found"],
            )

        session = self.session_cache[session_id]
        issues = []
        recovery_actions = []
        health_score = 100.0

        # Continuity check
        if "continuity" in health_checks:
            session_dir = self.claude_sessions_path / session_id
            if not session_dir.exists():
                issues.append("Session directory missing")
                recovery_actions.append("Recreate session directory")
                health_score -= 25.0

        # Files check
        if "files" in health_checks:
            session_dir = self.claude_sessions_path / session_id
            required_files = ["session-metadata.json"]
            for file_name in required_files:
                if not (session_dir / file_name).exists():
                    issues.append(f"Required file missing: {file_name}")
                    recovery_actions.append(f"Recreate {file_name}")
                    health_score -= 10.0

        # State check
        if "state" in health_checks:
            if session.status == SessionStatus.FAILED:
                issues.append("Session in failed state")
                recovery_actions.append("Reset session state")
                health_score -= 30.0

        # Agents check
        if "agents" in health_checks:
            failed_agents = [
                agent for agent in session.agents_executed if agent.status == ExecutionStatus.ERROR
            ]
            if failed_agents:
                issues.append(f"Failed agents: {len(failed_agents)}")
                recovery_actions.append("Restart failed agents")
                health_score -= len(failed_agents) * 5.0

        # Auto-recovery
        auto_recovery_attempted = False
        if auto_recover and recovery_actions:
            auto_recovery_attempted = True
            # Implement basic auto-recovery logic here

        # Diagnostics
        diagnostics = {}
        if include_diagnostics:
            diagnostics = {
                "session_age_minutes": (datetime.now(UTC) - session.started).total_seconds() / 60,
                "agents_count": len(session.agents_executed),
                "decisions_count": len(session.decisions),
                "performance_score": session.performance_metrics.efficiency_score,
            }

        return SessionHealthResult(
            session_id=session_id,
            health_score=max(0.0, health_score),
            issues=issues,
            recovery_actions=recovery_actions,
            diagnostics=diagnostics,
            auto_recovery_attempted=auto_recovery_attempted,
        )

    # ===== PLACEHOLDER IMPLEMENTATIONS FOR OTHER FUNCTIONS =====

    def session_orchestrate_workflow(self, **kwargs) -> WorkflowResult:
        """Workflow orchestration - placeholder implementation."""
        return WorkflowResult(
            workflow_id="placeholder",
            session_id=kwargs.get("session_id", "unknown"),
            execution_plan={},
            state=WorkflowState(
                workflow_type=WorkflowType.CUSTOM, current_phase="placeholder", state_machine={}
            ),
        )

    def session_analyze_patterns(self, **kwargs) -> PatternAnalysisResult:
        """Pattern analysis - placeholder implementation."""
        return PatternAnalysisResult(
            analysis_id="placeholder",
            scope=AnalysisScope.CURRENT,
            patterns=[],
            trends=[],
            recommendations=[],
        )

    def session_analyze_commands(self, **kwargs) -> CommandAnalysisResult:
        """Command analysis - placeholder implementation."""
        return CommandAnalysisResult(
            session_id=kwargs.get("session_id", "unknown"),
            analysis_period="current",
            patterns=[],
            inefficiencies=[],
            suggestions=[],
            metrics={},
        )

    def session_track_missing_functions(self, **kwargs) -> MissingFunctionResult:
        """Missing function tracking - placeholder implementation."""
        return MissingFunctionResult(
            session_id=kwargs.get("session_id", "unknown"),
            functions=[],
            priorities={},
            suggestions=[],
            impact={},
        )

    def session_get_dashboard(self, **kwargs) -> DashboardResult:
        """Dashboard generation - placeholder implementation."""
        return DashboardResult(
            dashboard_type=DashboardType.OVERVIEW,
            session_id=kwargs.get("session_id"),
            metrics={},
            visualizations=[],
            insights=[],
            recommendations=[],
        )

    # ===== SESSION NOTEBOOK =====

    def session_create_notebook(
        self,
        session_id: str | None = None,
        title: str | None = None,
        include_decisions: bool = True,
        include_agents: bool = True,
        include_metrics: bool = True,
        tags: list[str] | None = None,
        save_to_file: bool = True,
        save_to_database: bool = True,
    ) -> NotebookResult:
        """
        Generate a comprehensive markdown notebook/summary for a session.

        Creates a narrative summary of all work done during the session,
        including decisions made, agents executed, file changes, and metrics.

        Args:
            session_id: Session to summarize (defaults to current session)
            title: Custom title for the notebook
            include_decisions: Include decision log section
            include_agents: Include agent execution summary
            include_metrics: Include performance metrics
            tags: Tags for cross-session search
            save_to_file: Save markdown to file in session directory
            save_to_database: Persist summary to database for search

        Returns:
            NotebookResult with generated notebook and file path
        """
        try:
            return self._create_notebook_sync(
                session_id,
                title,
                include_decisions,
                include_agents,
                include_metrics,
                tags,
                save_to_file,
                save_to_database,
            )
        except Exception as e:
            debug_logger.error(f"Error creating notebook: {e}")
            return NotebookResult(
                session_id=session_id or "unknown",
                status="error",
                message=f"Failed to create notebook: {str(e)}",
            )

    async def session_create_notebook_async(
        self,
        session_id: str | None = None,
        title: str | None = None,
        include_decisions: bool = True,
        include_agents: bool = True,
        include_metrics: bool = True,
        tags: list[str] | None = None,
        save_to_file: bool = True,
        save_to_database: bool = True,
    ) -> NotebookResult:
        """Async version: Generate notebook with full database queries."""
        try:
            # Get session
            if not session_id:
                session_id = self._get_or_create_current_session_id()
            if not session_id or session_id not in self.session_cache:
                return NotebookResult(
                    session_id=session_id or "unknown", status="error", message="No session found"
                )

            session = self.session_cache[session_id]

            # Merge decisions from database
            if self.database:
                db_decisions = await self.database.query_decisions_by_session(session_id)
                existing_ids = {d.decision_id for d in session.decisions}
                for db_dec in db_decisions:
                    dec_id = db_dec.get("id") or db_dec.get("decision_id")
                    if dec_id and dec_id not in existing_ids:
                        from models.session_models import Decision, DecisionContext

                        session.decisions.append(
                            Decision(
                                decision_id=dec_id,
                                timestamp=(
                                    datetime.fromisoformat(db_dec["timestamp"])
                                    if db_dec.get("timestamp")
                                    else datetime.now(UTC)
                                ),
                                description=db_dec.get("description", ""),
                                context=DecisionContext(session_id=session_id, project_state={}),
                                impact_level=ImpactLevel(db_dec.get("impact_level", "medium")),
                                artifacts=(
                                    json.loads(db_dec.get("artifacts", "[]"))
                                    if isinstance(db_dec.get("artifacts"), str)
                                    else db_dec.get("artifacts", [])
                                ),
                            )
                        )

            # Build sections
            end_time = session.completed or datetime.now(UTC)
            duration_minutes = (end_time - session.started).total_seconds() / 60
            if not title:
                title = f"Session: {session.project_name} - {session.started.strftime('%Y-%m-%d %H:%M')}"

            sections: list[NotebookSection] = []
            sections.append(
                NotebookSection(
                    heading="Overview",
                    content=self._generate_overview_section(session, duration_minutes),
                    level=2,
                )
            )

            # File operations from database (async)
            files_changed: list[str] = []
            if self.database:
                files_content, files_changed = await self._generate_files_section_async(session_id)
                if files_content:
                    sections.append(
                        NotebookSection(heading="Work Completed", content=files_content, level=2)
                    )

            # Agents
            agents_used: list[str] = []
            if include_agents and session.agents_executed:
                agents_content, agents_used = self._generate_agents_section(session)
                sections.append(
                    NotebookSection(heading="Agents Executed", content=agents_content, level=2)
                )

            # Decisions
            decisions_made: list[str] = []
            if include_decisions and session.decisions:
                decisions_content, decisions_made = self._generate_decisions_section(session)
                sections.append(
                    NotebookSection(heading="Decisions Made", content=decisions_content, level=2)
                )

            # Metrics
            if include_metrics:
                sections.append(
                    NotebookSection(
                        heading="Performance Metrics",
                        content=self._generate_metrics_section(session),
                        level=2,
                    )
                )

            # Learnings from database (async)
            if self.database:
                learnings_content = await self._generate_learnings_section_async(
                    session.project_path
                )
                if learnings_content:
                    sections.append(
                        NotebookSection(
                            heading="Project Learnings", content=learnings_content, level=2
                        )
                    )

            key_changes = list(set(self._extract_key_changes(session)) | set(files_changed))[:20]
            if tags is None:
                tags = self._auto_generate_tags(session, agents_used, key_changes)

            summary_markdown = self._generate_summary_markdown(
                title, sections, session, duration_minutes
            )
            notebook = SessionNotebook(
                session_id=session_id,
                title=title,
                created_at=datetime.now().isoformat(),
                project_name=session.project_name,
                project_path=session.project_path,
                duration_minutes=round(duration_minutes, 2),
                sections=sections,
                summary_markdown=summary_markdown,
                key_changes=key_changes,
                agents_used=agents_used,
                decisions_made=decisions_made,
                tags=tags,
            )

            file_path = None
            if save_to_file and self.use_filesystem:
                file_path = self._save_notebook_to_file(session_id, notebook)

            # Save to database
            if save_to_database and self.database:
                await self.database.save_session_summary(
                    {
                        "session_id": session_id,
                        "title": title,
                        "summary_markdown": summary_markdown,
                        "key_changes": key_changes,
                        "tags": tags,
                        "created_at": datetime.now(UTC).isoformat(),
                    }
                )

            return NotebookResult(
                session_id=session_id,
                status="success",
                notebook=notebook,
                markdown_output=summary_markdown,
                file_path=file_path,
                search_indexed=True,
                message=f"Notebook created with {len(sections)} sections",
            )
        except Exception as e:
            debug_logger.error(f"Error creating async notebook: {e}")
            return NotebookResult(
                session_id=session_id or "unknown", status="error", message=str(e)
            )

    def _create_notebook_sync(
        self,
        session_id: str | None,
        title: str | None,
        include_decisions: bool,
        include_agents: bool,
        include_metrics: bool,
        tags: list[str] | None,
        save_to_file: bool,
        save_to_database: bool,
    ) -> NotebookResult:
        """Synchronous notebook creation."""

        # Get session (current or specified)
        if not session_id:
            session_id = self._get_or_create_current_session_id()

        if not session_id or session_id not in self.session_cache:
            return NotebookResult(
                session_id=session_id or "unknown",
                status="error",
                message="No session found to create notebook for",
            )

        session = self.session_cache[session_id]

        # Merge decisions from database if available
        if self.database:
            try:
                import asyncio

                asyncio.get_running_loop()
                asyncio.create_task(
                    self.database.query_decisions_by_session(session_id)
                )
                # We can't await here in sync context, so use run_until_complete alternative
                # Instead, we'll check if we're in async context
            except RuntimeError:
                pass  # No event loop - skip database merge in sync context
            else:
                # If we have an event loop, schedule the merge
                async def merge_db_decisions():
                    db_decisions_list = await self.database.query_decisions_by_session(session_id)
                    existing_ids = {d.decision_id for d in session.decisions}
                    for db_dec in db_decisions_list:
                        dec_id = db_dec.get("id") or db_dec.get("decision_id")
                        if dec_id and dec_id not in existing_ids:
                            from models.session_models import Decision, DecisionContext

                            session.decisions.append(
                                Decision(
                                    decision_id=dec_id,
                                    timestamp=(
                                        datetime.fromisoformat(db_dec["timestamp"])
                                        if db_dec.get("timestamp")
                                        else datetime.now(UTC)
                                    ),
                                    description=db_dec.get("description", ""),
                                    context=DecisionContext(
                                        session_id=session_id, project_state={}
                                    ),
                                    impact_level=ImpactLevel(db_dec.get("impact_level", "medium")),
                                    artifacts=(
                                        json.loads(db_dec.get("artifacts", "[]"))
                                        if isinstance(db_dec.get("artifacts"), str)
                                        else db_dec.get("artifacts", [])
                                    ),
                                )
                            )

                asyncio.create_task(merge_db_decisions())

        # Calculate duration
        end_time = session.completed or datetime.now(UTC)
        duration_minutes = (end_time - session.started).total_seconds() / 60

        # Generate title if not provided
        if not title:
            title = (
                f"Session: {session.project_name} - {session.started.strftime('%Y-%m-%d %H:%M')}"
            )

        # Build notebook sections
        sections: list[NotebookSection] = []

        # Overview section
        overview_content = self._generate_overview_section(session, duration_minutes)
        sections.append(NotebookSection(heading="Overview", content=overview_content, level=2))

        # Work Completed section (file operations)
        files_content, files_changed = self._generate_files_section(session)
        if files_content:
            sections.append(
                NotebookSection(heading="Work Completed", content=files_content, level=2)
            )

        # Agents section
        agents_used: list[str] = []
        if include_agents and session.agents_executed:
            agents_content, agents_used = self._generate_agents_section(session)
            sections.append(
                NotebookSection(heading="Agents Executed", content=agents_content, level=2)
            )

        # Decisions section
        decisions_made: list[str] = []
        if include_decisions and session.decisions:
            decisions_content, decisions_made = self._generate_decisions_section(session)
            sections.append(
                NotebookSection(heading="Decisions Made", content=decisions_content, level=2)
            )

        # Metrics section
        if include_metrics:
            metrics_content = self._generate_metrics_section(session)
            sections.append(
                NotebookSection(heading="Performance Metrics", content=metrics_content, level=2)
            )

        # Learnings section (from database)
        learnings_content = self._generate_learnings_section(session.project_path)
        if learnings_content:
            sections.append(
                NotebookSection(heading="Project Learnings", content=learnings_content, level=2)
            )

        # Gather key file changes from agent executions and file operations
        key_changes = self._extract_key_changes(session)
        # Merge with files from file_operations tracking
        key_changes = list(set(key_changes) | set(files_changed))[:20]

        # Auto-generate tags if not provided
        if tags is None:
            tags = self._auto_generate_tags(session, agents_used, key_changes)

        # Generate summary markdown
        summary_markdown = self._generate_summary_markdown(
            title, sections, session, duration_minutes
        )

        # Create notebook object
        notebook = SessionNotebook(
            session_id=session_id,
            title=title,
            created_at=datetime.now().isoformat(),
            project_name=session.project_name,
            project_path=session.project_path,
            duration_minutes=round(duration_minutes, 2),
            sections=sections,
            summary_markdown=summary_markdown,
            key_changes=key_changes,
            agents_used=agents_used,
            decisions_made=decisions_made,
            tags=tags,
        )

        # Save to file if requested
        file_path = None
        if save_to_file and self.use_filesystem:
            file_path = self._save_notebook_to_file(session_id, notebook)

        # Save to database if requested (for search indexing)
        search_indexed = False
        if save_to_database and self.database:
            # This would be async in the HTTP server context
            # For now, just mark as not indexed
            search_indexed = False
            debug_logger.info("Database persistence requires async context")

        return NotebookResult(
            session_id=session_id,
            status="success",
            notebook=notebook,
            markdown_output=summary_markdown,
            file_path=file_path,
            search_indexed=search_indexed,
            message=f"Notebook created successfully with {len(sections)} sections",
        )

    def _generate_overview_section(self, session: Session, duration_minutes: float) -> str:
        """Generate the overview section content."""
        return f"""
**Project**: {session.project_name}
**Path**: `{session.project_path}`
**Started**: {session.started.strftime('%Y-%m-%d %H:%M:%S')}
**Duration**: {duration_minutes:.1f} minutes
**Status**: {session.status.value}
**Mode**: {session.mode}

### Session Health
- Overall Score: {session.health_status.overall_score:.0f}%
- Continuity Valid: {'Yes' if session.health_status.continuity_valid else 'No'}
- Files Valid: {'Yes' if session.health_status.files_valid else 'No'}
""".strip()

    def _generate_agents_section(self, session: Session) -> tuple[str, list[str]]:
        """Generate agents section and return agent names."""
        agents_used = []
        lines = []

        for agent in session.agents_executed:
            agents_used.append(agent.agent_name)
            status_emoji = (
                "✅"
                if agent.status == ExecutionStatus.SUCCESS
                else "⚠️" if agent.status == ExecutionStatus.RUNNING else "❌"
            )
            lines.append(f"- {status_emoji} **{agent.agent_name}** ({agent.agent_type})")

            if agent.execution_steps:
                lines.append(f"  - Steps: {len(agent.execution_steps)}")
                for step in agent.execution_steps[:3]:  # Show first 3 steps
                    lines.append(f"    - {step.operation}: {step.description[:50]}...")

        return "\n".join(lines), agents_used

    def _generate_decisions_section(self, session: Session) -> tuple[str, list[str]]:
        """Generate decisions section and return decision descriptions."""
        decisions_made = []
        lines = []

        for decision in session.decisions:
            decisions_made.append(decision.description)
            impact_emoji = {
                ImpactLevel.LOW: "🟢",
                ImpactLevel.MEDIUM: "🟡",
                ImpactLevel.HIGH: "🟠",
                ImpactLevel.CRITICAL: "🔴",
            }.get(decision.impact_level, "⚪")

            lines.append(f"- {impact_emoji} **{decision.description}**")
            if decision.rationale:
                lines.append(f"  - Rationale: {decision.rationale}")
            if decision.artifacts:
                lines.append(f"  - Artifacts: {', '.join(decision.artifacts[:3])}")

        return "\n".join(lines), decisions_made

    def _generate_metrics_section(self, session: Session) -> str:
        """Generate performance metrics section."""
        metrics = session.performance_metrics
        return f"""
| Metric | Value |
|--------|-------|
| Total Execution Time | {metrics.total_execution_time_ms / 1000:.1f}s |
| Agents Executed | {metrics.agents_executed} |
| Successful Executions | {metrics.successful_executions} |
| Failed Executions | {metrics.failed_executions} |
| Commands Executed | {metrics.commands_executed} |
| Decisions Made | {metrics.decisions_made} |
| Efficiency Score | {metrics.efficiency_score:.1f}% |
""".strip()

    def _generate_files_section(self, session: Session) -> tuple[str | None, list[str]]:
        """Generate files section from database.

        Queries file operations for the session and formats as markdown table.
        Returns tuple of (markdown_content, list_of_changed_files).
        Returns (None, []) if no database or no file operations found.
        """
        changed_files: list[str] = []

        if not self.database:
            return None, changed_files

        # We need async context to query, return placeholder for sync
        try:
            import asyncio

            asyncio.get_running_loop()
        except RuntimeError:
            return None, changed_files

        return None, changed_files

    async def _generate_files_section_async(self, session_id: str) -> tuple[str | None, list[str]]:
        """Async version: Generate files section from database."""
        changed_files: list[str] = []

        if not self.database:
            return None, changed_files

        file_ops = await self.database.query_file_operations_by_session(session_id)
        if not file_ops:
            return None, changed_files

        # Group by operation type
        by_type: dict[str, list[dict]] = {"create": [], "edit": [], "delete": [], "read": []}
        for op in file_ops:
            op_type = op.get("operation", "edit").lower()
            if op_type in by_type:
                by_type[op_type].append(op)
            changed_files.append(op.get("file_path", ""))

        lines = ["| Operation | File | Lines | Summary |", "|-----------|------|-------|---------|"]

        for op_type in ["create", "edit", "delete"]:
            ops = by_type.get(op_type, [])
            for op in ops:
                file_path = op.get("file_path", "")
                lines_info = f"+{op.get('lines_added', 0)}/-{op.get('lines_removed', 0)}"
                summary = (op.get("summary") or "")[:50]
                lines.append(f"| {op_type} | `{file_path}` | {lines_info} | {summary} |")

        if len(lines) == 2:  # Only header
            return None, changed_files

        return "\n".join(lines), changed_files

    def _generate_learnings_section(self, project_path: str) -> str | None:
        """Generate learnings section from database.

        Queries project-specific learnings and formats as markdown.
        Returns None if no database or no learnings found.
        """
        if not self.database:
            return None

        # We need async context to query, so return placeholder
        # The actual query happens in async context of HTTP server
        try:
            import asyncio

            asyncio.get_running_loop()
        except RuntimeError:
            return None  # No event loop - skip learnings in sync context

        # Return a placeholder that will be populated async
        # For sync context, we return None and let HTTP server handle it
        return None

    async def _generate_learnings_section_async(self, project_path: str) -> str | None:
        """Async version: Generate learnings section from database."""
        if not self.database:
            return None

        learnings = await self.database.query_project_learnings(project_path, limit=10)
        if not learnings:
            return None

        lines = ["This project has accumulated the following learnings:\n"]

        for learning in learnings:
            category = learning.get("category", "general")
            content = learning.get("learning_content", "")
            trigger = learning.get("trigger_context", "")
            success_count = learning.get("success_count", 0)

            category_emoji = {
                "error_fix": "🔧",
                "pattern": "📋",
                "preference": "⚙️",
                "workflow": "🔄",
            }.get(category, "💡")

            lines.append(
                f"- {category_emoji} **{category}**: {content[:100]}{'...' if len(content) > 100 else ''}"
            )
            if trigger:
                lines.append(f"  - *Trigger*: {trigger[:80]}{'...' if len(trigger) > 80 else ''}")
            if success_count > 1:
                lines.append(f"  - *Used successfully*: {success_count} times")

        return "\n".join(lines)

    def _extract_key_changes(self, session: Session) -> list[str]:
        """Extract key file changes from agent executions."""
        changes = set()

        for agent in session.agents_executed:
            for step in agent.execution_steps:
                # Extract tools that typically modify files
                for tool in step.tools_used:
                    if any(
                        action in tool.lower() for action in ["write", "edit", "create", "modify"]
                    ):
                        changes.add(tool)

        # Also check decision artifacts
        for decision in session.decisions:
            for artifact in decision.artifacts:
                if artifact.endswith((".py", ".js", ".ts", ".toml", ".yaml", ".yml", ".md")):
                    changes.add(artifact)

        return list(changes)[:20]  # Limit to 20 changes

    def _auto_generate_tags(
        self, session: Session, agents_used: list[str], key_changes: list[str]
    ) -> list[str]:
        """Auto-generate tags based on session content."""
        tags = set()

        # Add project name as tag
        tags.add(session.project_name.lower().replace(" ", "-"))

        # Add agent types as tags
        for agent in agents_used:
            if "test" in agent.lower():
                tags.add("testing")
            if "quality" in agent.lower():
                tags.add("quality")
            if "security" in agent.lower():
                tags.add("security")
            if "deploy" in agent.lower():
                tags.add("deployment")

        # Add file type tags
        extensions = set()
        for change in key_changes:
            if "." in change:
                ext = change.split(".")[-1]
                if ext in ("py", "python"):
                    extensions.add("python")
                elif ext in ("js", "ts", "jsx", "tsx"):
                    extensions.add("javascript")
                elif ext in ("yaml", "yml", "toml"):
                    extensions.add("config")

        tags.update(extensions)

        return list(tags)[:10]  # Limit to 10 tags

    def _generate_summary_markdown(
        self, title: str, sections: list[NotebookSection], session: Session, duration_minutes: float
    ) -> str:
        """Generate the complete markdown document."""
        lines = [
            f"# {title}",
            "",
            f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"> Duration: {duration_minutes:.1f} minutes",
            "",
        ]

        for section in sections:
            heading_prefix = "#" * section.level
            lines.append(f"{heading_prefix} {section.heading}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

        # Add footer
        lines.extend(
            [
                "---",
                "",
                f"*Session ID: `{session.id}`*",
                "*Generated by session-intelligence MCP server*",
            ]
        )

        return "\n".join(lines)

    def _save_notebook_to_file(self, session_id: str, notebook: SessionNotebook) -> str:
        """Save notebook markdown to file."""
        session_dir = self.claude_sessions_path / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save markdown file
        notebook_file = session_dir / "session-notebook.md"
        with open(notebook_file, "w") as f:
            f.write(notebook.summary_markdown)

        # Save JSON metadata
        metadata_file = session_dir / "notebook-metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(notebook.model_dump(), f, indent=2, default=str)

        debug_logger.info(f"Saved notebook to: {notebook_file}")
        return str(notebook_file)

    # ===== SESSION SEARCH =====

    async def session_search(
        self, query: str, search_type: str = "fulltext", limit: int = 20
    ) -> SearchResults:
        """
        Search across sessions using full-text search.

        Args:
            query: Search query (supports FTS5 syntax for SQLite, simple text for PostgreSQL)
            search_type: Type of search - "fulltext", "tag", or "file"
            limit: Maximum results to return

        Returns:
            SearchResults with matching sessions
        """
        if not self.database:
            debug_logger.warning("No database configured for session search")
            return SearchResults(query=query, total_results=0, results=[])

        try:
            # Call database search_sessions method
            search_results = await self.database.search_sessions(
                query=query, search_type=search_type, limit=limit
            )

            # Convert database results to SearchResult model instances
            results = []
            for result in search_results:
                # Convert datetime to ISO string if present
                started_at = result.get("started_at")
                if started_at and hasattr(started_at, "isoformat"):
                    started_at = started_at.isoformat()
                elif started_at:
                    started_at = str(started_at)

                # Extract tags if they're in JSON format
                tags = result.get("tags", [])
                if isinstance(tags, str):
                    import json

                    try:
                        tags = json.loads(tags)
                    except (json.JSONDecodeError, TypeError):
                        tags = []

                results.append(
                    SearchResult(
                        session_id=result.get("session_id", ""),
                        title=result.get("title"),
                        snippet=result.get("snippet", ""),
                        relevance=float(result.get("relevance", 0.0)),
                        project_name=result.get("project_name"),
                        project_path=result.get("project_path"),
                        started_at=started_at,
                        tags=tags,
                    )
                )

            debug_logger.info(f"Session search for '{query}' returned {len(results)} results")
            return SearchResults(query=query, total_results=len(results), results=results)

        except Exception as e:
            debug_logger.error(f"Error in session_search: {e}")
            return SearchResults(query=query, total_results=0, results=[])

    async def session_query_notebooks(
        self,
        project_path: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Query session notebooks/summaries with optional filters.

        Args:
            project_path: Filter by project path
            tags: Filter by tags
            limit: Maximum results to return

        Returns:
            List of session notebook summaries
        """
        if not self.database:
            debug_logger.warning("No database configured for session_query_notebooks")
            return []

        try:
            results = await self.database.query_session_summaries(
                project_path=project_path,
                tags=tags,
                limit=limit,
            )

            # Convert datetime objects to ISO strings for JSON serialization
            for result in results:
                for key in ["created_at", "updated_at"]:
                    if key in result and hasattr(result[key], "isoformat"):
                        result[key] = result[key].isoformat()

            debug_logger.info(f"session_query_notebooks returned {len(results)} results")
            return results

        except Exception as e:
            debug_logger.error(f"Error in session_query_notebooks: {e}")
            return []

    # ===== KNOWLEDGE SYSTEM =====

    def session_log_learning(
        self,
        category: str,
        learning_content: str,
        trigger_context: str | None = None,
        project_path: str | None = None,
    ) -> LearningResult:
        """
        Log a project-specific learning (pattern, fix, preference).

        Args:
            category: Learning category - error_fix, pattern, preference, workflow
            learning_content: The actual knowledge/solution
            trigger_context: When to apply this learning
            project_path: Project scope (uses current if not specified)

        Returns:
            LearningResult with saved learning
        """
        import uuid

        learning_id = f"learn_{uuid.uuid4().hex[:12]}"
        effective_project = project_path or str(self.claude_sessions_path.parent)

        # Get current session if available
        source_session = self._current_session_id

        debug_logger.info(f"Logging learning: {category} for {effective_project}")

        return LearningResult(
            id=learning_id,
            status="pending_save",
            message=f"Learning logged for {category}. Requires async save to database.",
            learning=ProjectLearning(
                id=learning_id,
                project_path=effective_project,
                category=LearningCategory(category),
                trigger_context=trigger_context,
                learning_content=learning_content,
                source_session_id=source_session,
                created_at=datetime.now().isoformat(),
            ),
        )

    def session_find_solution(
        self,
        error_text: str,
        error_category: str | None = None,
        include_universal: bool = True,
        project_path: str | None = None,
    ) -> SolutionSearchResult:
        """
        Find solutions for an error from project and universal knowledge.

        Args:
            error_text: The error message/pattern to search for
            error_category: Optional category hint (compile, runtime, config, dependency)
            include_universal: Whether to include universal solutions
            project_path: Project scope for solutions

        Returns:
            SolutionSearchResult with matching solutions
        """
        debug_logger.info(f"Finding solutions for error: {error_text[:100]}...")

        # This requires database access which is async
        # Return placeholder indicating need for async call
        return SolutionSearchResult(
            error_text=error_text,
            total_found=0,
            solutions=[],
            project_specific_count=0,
            universal_count=0,
        )

    def session_update_solution_outcome(
        self,
        solution_id: str,
        success: bool,
    ) -> SolutionResult:
        """
        Update success/failure count for a solution.

        Args:
            solution_id: ID of the solution to update
            success: Whether the solution worked

        Returns:
            SolutionResult with updated status
        """
        debug_logger.info(
            f"Updating solution outcome: {solution_id} -> {'success' if success else 'failure'}"
        )

        return SolutionResult(
            id=solution_id,
            status="pending_update",
            message="Solution outcome recorded. Requires async update to database.",
        )

    # ===== AGENT SYSTEM METHODS =====
    # These methods manage cross-session agent identity, decisions, learnings, and notebooks.
    # Agents persist globally (not session-scoped) to accumulate knowledge over time.

    async def agent_register(
        self,
        name: str,
        agent_type: str,
        display_name: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        capabilities: list[str] | None = None,
    ) -> AgentRegistrationResult:
        """
        Register or update an agent in the global agent registry.

        Creates a new agent if one doesn't exist with this name, otherwise updates
        the existing agent's metadata and marks it as active.

        Args:
            name: Unique agent name (e.g., "focused-quality-resolver")
            agent_type: Agent type category (e.g., "meta", "domain", "specialized")
            display_name: Human-readable display name
            description: Agent description and purpose
            metadata: Additional metadata dict
            capabilities: List of agent capabilities

        Returns:
            AgentRegistrationResult with status 'created', 'updated', or 'error'
        """
        if not self.database:
            debug_logger.warning("agent_register called without database")
            return AgentRegistrationResult(
                agent_id="",
                name=name,
                status="error",
                message="Database not available for agent registration",
            )

        try:
            # Check if agent already exists by name
            existing_agent = await self.database.get_agent_by_name(name)

            now = datetime.now(UTC).isoformat()

            if existing_agent:
                # Update existing agent
                agent_id = existing_agent["id"]
                agent_data = {
                    "id": agent_id,
                    "name": name,
                    "agent_type": agent_type,
                    "display_name": display_name or existing_agent.get("display_name"),
                    "description": description or existing_agent.get("description"),
                    "metadata": metadata or existing_agent.get("metadata", {}),
                    "capabilities": capabilities or existing_agent.get("capabilities", []),
                    "first_seen_at": existing_agent.get("first_seen_at", now),
                    "last_active_at": now,
                    "total_executions": existing_agent.get("total_executions", 0),
                    "total_decisions": existing_agent.get("total_decisions", 0),
                    "total_learnings": existing_agent.get("total_learnings", 0),
                    "total_notebooks": existing_agent.get("total_notebooks", 0),
                    "is_active": True,
                }
                await self.database.save_agent(agent_data)
                debug_logger.info(f"Updated existing agent: {name} ({agent_id})")
                return AgentRegistrationResult(
                    agent_id=agent_id,
                    name=name,
                    status="updated",
                    message=f"Agent '{name}' updated successfully",
                )
            else:
                # Create new agent
                agent_id = str(uuid.uuid4())
                agent_data = {
                    "id": agent_id,
                    "name": name,
                    "agent_type": agent_type,
                    "display_name": display_name,
                    "description": description,
                    "metadata": metadata or {},
                    "capabilities": capabilities or [],
                    "first_seen_at": now,
                    "last_active_at": now,
                    "total_executions": 0,
                    "total_decisions": 0,
                    "total_learnings": 0,
                    "total_notebooks": 0,
                    "is_active": True,
                }
                await self.database.save_agent(agent_data)
                debug_logger.info(f"Created new agent: {name} ({agent_id})")
                return AgentRegistrationResult(
                    agent_id=agent_id,
                    name=name,
                    status="created",
                    message=f"Agent '{name}' created successfully",
                )

        except Exception as e:
            debug_logger.error(f"Error registering agent {name}: {e}")
            return AgentRegistrationResult(
                agent_id="",
                name=name,
                status="error",
                message=f"Failed to register agent: {str(e)}",
            )

    async def agent_get_info(self, identifier: str) -> Agent | None:
        """
        Get agent information by name or UUID.

        Automatically detects whether the identifier is a UUID or a name
        and queries accordingly.

        Args:
            identifier: Agent name (e.g., "focused-quality-resolver") or UUID

        Returns:
            Agent model if found, None otherwise
        """
        if not self.database:
            debug_logger.warning("agent_get_info called without database")
            return None

        try:
            # Detect if identifier is a UUID (contains hyphens and matches UUID format)
            is_uuid = False
            try:
                uuid.UUID(identifier)
                is_uuid = True
            except ValueError:
                is_uuid = False

            if is_uuid:
                agent_data = await self.database.get_agent(identifier)
            else:
                agent_data = await self.database.get_agent_by_name(identifier)

            if not agent_data:
                debug_logger.info(f"Agent not found: {identifier}")
                return None

            # Convert timestamps to ISO strings if they're datetime objects
            first_seen = agent_data.get("first_seen_at")
            if first_seen and hasattr(first_seen, "isoformat"):
                first_seen = first_seen.isoformat()

            last_active = agent_data.get("last_active_at")
            if last_active and hasattr(last_active, "isoformat"):
                last_active = last_active.isoformat()

            # Ensure metadata and capabilities are correct types
            metadata = agent_data.get("metadata", {})
            if isinstance(metadata, str):
                import json

                metadata = json.loads(metadata)

            capabilities = agent_data.get("capabilities", [])
            if isinstance(capabilities, str):
                import json

                capabilities = json.loads(capabilities)

            # Convert to Agent model
            return Agent(
                id=agent_data["id"],
                name=agent_data["name"],
                agent_type=agent_data["agent_type"],
                display_name=agent_data.get("display_name"),
                description=agent_data.get("description"),
                metadata=metadata,
                capabilities=capabilities,
                first_seen_at=first_seen,
                last_active_at=last_active,
                total_executions=agent_data.get("total_executions", 0),
                total_decisions=agent_data.get("total_decisions", 0),
                total_learnings=agent_data.get("total_learnings", 0),
                total_notebooks=agent_data.get("total_notebooks", 0),
                is_active=agent_data.get("is_active", True),
            )

        except Exception as e:
            debug_logger.error(f"Error getting agent {identifier}: {e}")
            return None

    async def agent_log_decision(
        self,
        agent_name: str,
        decision_type: str,
        context: str,
        decision: str,
        reasoning: str | None = None,
        alternatives: list[str] | None = None,
        confidence: float = 0.8,
        tags: list[str] | None = None,
    ) -> AgentDecisionResult:
        """
        Log a decision made by an agent.

        Creates a decision record associated with the agent and updates
        the agent's total_decisions counter.

        Args:
            agent_name: Name of the agent making the decision
            decision_type: Category of decision (e.g., "architecture", "implementation", "pattern")
            context: The situation/context that led to this decision
            decision: The actual decision made
            reasoning: Explanation of why this decision was made
            alternatives: Alternative options that were considered
            confidence: Confidence level (0.0-1.0)
            tags: Tags for categorization and search

        Returns:
            AgentDecisionResult with decision_id and status
        """
        if not self.database:
            debug_logger.warning("agent_log_decision called without database")
            return AgentDecisionResult(
                decision_id="",
                agent_id="",
                status="error",
                message="Database not available for logging decision",
            )

        try:
            # Look up agent by name
            agent_data = await self.database.get_agent_by_name(agent_name)
            if not agent_data:
                return AgentDecisionResult(
                    decision_id="",
                    agent_id="",
                    status="error",
                    message=f"Agent '{agent_name}' not found. Register the agent first.",
                )

            agent_id = agent_data["id"]
            decision_id = str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()

            # Build decision data for database
            decision_data = {
                "id": decision_id,
                "agent_id": agent_id,
                "timestamp": now,
                "description": decision,
                "rationale": reasoning,
                "category": decision_type,
                "impact_level": "medium",  # Default
                "context": {"situation": context, "alternatives": alternatives or []},
                "artifacts": tags or [],
                "source_session_id": self._current_session_id,
                "source_project_path": (
                    str(self.claude_sessions_path.parent) if self.use_filesystem else None
                ),
            }

            await self.database.save_agent_decision(decision_data)

            # Update agent stats
            await self.database.update_agent_stats(agent_id, "decisions")

            debug_logger.info(f"Logged decision {decision_id} for agent {agent_name}")
            return AgentDecisionResult(
                decision_id=decision_id,
                agent_id=agent_id,
                status="success",
                message=f"Decision logged successfully for agent '{agent_name}'",
            )

        except Exception as e:
            debug_logger.error(f"Error logging decision for {agent_name}: {e}")
            return AgentDecisionResult(
                decision_id="",
                agent_id="",
                status="error",
                message=f"Failed to log decision: {str(e)}",
            )

    async def agent_query_decisions(
        self,
        agent_name: str,
        decision_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[AgentDecision]:
        """
        Query decisions made by an agent.

        Args:
            agent_name: Name of the agent to query
            decision_type: Filter by decision type/category
            tags: Filter by tags (not yet implemented in DB layer)
            limit: Maximum number of results

        Returns:
            List of AgentDecision models
        """
        if not self.database:
            debug_logger.warning("agent_query_decisions called without database")
            return []

        try:
            # Look up agent by name
            agent_data = await self.database.get_agent_by_name(agent_name)
            if not agent_data:
                debug_logger.info(f"Agent '{agent_name}' not found for decision query")
                return []

            agent_id = agent_data["id"]

            # Query decisions
            decision_rows = await self.database.query_agent_decisions(
                agent_id=agent_id,
                category=decision_type,
                limit=limit,
            )

            # Convert to AgentDecision models
            decisions = []
            for row in decision_rows:
                # Parse context - may be JSON string from PostgreSQL
                context_raw = row.get("context", {})
                if isinstance(context_raw, str):
                    try:
                        context_data = json.loads(context_raw)
                    except (json.JSONDecodeError, TypeError):
                        context_data = {"situation": context_raw}
                else:
                    context_data = context_raw if isinstance(context_raw, dict) else {}

                # Parse artifacts - may be JSON string from PostgreSQL
                artifacts_raw = row.get("artifacts", [])
                if isinstance(artifacts_raw, str):
                    try:
                        artifacts = json.loads(artifacts_raw)
                    except (json.JSONDecodeError, TypeError):
                        artifacts = []
                else:
                    artifacts = artifacts_raw if isinstance(artifacts_raw, list) else []

                # Convert datetime to ISO string if needed
                created_at = row.get("timestamp")
                if hasattr(created_at, "isoformat"):
                    created_at = created_at.isoformat()
                updated_at = row.get("outcome_updated_at")
                if hasattr(updated_at, "isoformat"):
                    updated_at = updated_at.isoformat()

                decisions.append(
                    AgentDecision(
                        id=row["id"],
                        agent_id=row["agent_id"],
                        decision_type=row.get("category", "unknown"),
                        context=(
                            context_data.get("situation", "")
                            if isinstance(context_data, dict)
                            else str(context_data)
                        ),
                        decision=row.get("description", ""),
                        reasoning=row.get("rationale"),
                        alternatives=(
                            context_data.get("alternatives", [])
                            if isinstance(context_data, dict)
                            else []
                        ),
                        confidence=0.8,  # Default, not stored in current schema
                        outcome=row.get("outcome"),
                        outcome_success=None,  # Would need to parse outcome
                        tags=artifacts,
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                )

            return decisions

        except Exception as e:
            debug_logger.error(f"Error querying decisions for {agent_name}: {e}")
            return []

    async def agent_update_decision_outcome(
        self,
        decision_id: str,
        outcome: str,
        success: bool,
    ) -> dict[str, Any]:
        """
        Update the outcome of a decision.

        Args:
            decision_id: ID of the decision to update
            outcome: Description of the outcome
            success: Whether the decision led to a successful outcome

        Returns:
            Dict with status and message
        """
        if not self.database:
            debug_logger.warning("agent_update_decision_outcome called without database")
            return {"status": "error", "message": "Database not available"}

        try:
            notes = f"Success: {success}"
            await self.database.update_agent_decision_outcome(decision_id, outcome, notes)
            debug_logger.info(f"Updated decision outcome: {decision_id}")
            return {
                "status": "success",
                "decision_id": decision_id,
                "outcome": outcome,
                "success": success,
            }

        except Exception as e:
            debug_logger.error(f"Error updating decision outcome {decision_id}: {e}")
            return {"status": "error", "message": str(e)}

    async def agent_log_learning(
        self,
        agent_name: str,
        learning_type: str,
        title: str,
        content: str,
        source_context: str | None = None,
        applicability: list[str] | None = None,
        confidence: float = 0.8,
        tags: list[str] | None = None,
    ) -> AgentLearningResult:
        """
        Log a learning/knowledge item for an agent.

        Creates a learning record that captures patterns, techniques,
        anti-patterns, or preferences discovered by the agent.

        Args:
            agent_name: Name of the agent
            learning_type: Type of learning (e.g., "pattern", "anti-pattern", "technique", "preference")
            title: Short title for the learning
            content: Detailed content of the learning
            source_context: Context where this learning was discovered
            applicability: List of contexts where this learning applies
            confidence: Confidence level (0.0-1.0)
            tags: Tags for categorization and search

        Returns:
            AgentLearningResult with learning_id and status
        """
        if not self.database:
            debug_logger.warning("agent_log_learning called without database")
            return AgentLearningResult(
                learning_id="",
                agent_id="",
                status="error",
                message="Database not available for logging learning",
            )

        try:
            # Look up agent by name
            agent_data = await self.database.get_agent_by_name(agent_name)
            if not agent_data:
                return AgentLearningResult(
                    learning_id="",
                    agent_id="",
                    status="error",
                    message=f"Agent '{agent_name}' not found. Register the agent first.",
                )

            agent_id = agent_data["id"]
            learning_id = str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()

            # Build learning data for database
            learning_data = {
                "id": learning_id,
                "agent_id": agent_id,
                "category": learning_type,
                "trigger_context": source_context,
                "learning_content": f"# {title}\n\n{content}",
                "applies_to": {
                    "contexts": applicability or [],
                    "tags": tags or [],
                    "confidence": confidence,
                },
                "success_count": 1,
                "failure_count": 0,
                "source_session_id": self._current_session_id,
                "source_project_path": (
                    str(self.claude_sessions_path.parent) if self.use_filesystem else None
                ),
                "created_at": now,
                "updated_at": now,
            }

            await self.database.save_agent_learning(learning_data)

            # Update agent stats
            await self.database.update_agent_stats(agent_id, "learnings")

            debug_logger.info(f"Logged learning {learning_id} for agent {agent_name}")
            return AgentLearningResult(
                learning_id=learning_id,
                agent_id=agent_id,
                status="success",
                message=f"Learning logged successfully for agent '{agent_name}'",
            )

        except Exception as e:
            debug_logger.error(f"Error logging learning for {agent_name}: {e}")
            return AgentLearningResult(
                learning_id="",
                agent_id="",
                status="error",
                message=f"Failed to log learning: {str(e)}",
            )

    async def agent_query_learnings(
        self,
        agent_name: str,
        learning_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[AgentLearning]:
        """
        Query learnings for an agent.

        Args:
            agent_name: Name of the agent to query
            learning_type: Filter by learning type/category
            tags: Filter by tags (not yet implemented in DB layer)
            limit: Maximum number of results

        Returns:
            List of AgentLearning models
        """
        if not self.database:
            debug_logger.warning("agent_query_learnings called without database")
            return []

        try:
            # Look up agent by name
            agent_data = await self.database.get_agent_by_name(agent_name)
            if not agent_data:
                debug_logger.info(f"Agent '{agent_name}' not found for learning query")
                return []

            agent_id = agent_data["id"]

            # Query learnings
            learning_rows = await self.database.query_agent_learnings(
                agent_id=agent_id,
                category=learning_type,
                limit=limit,
            )

            # Convert to AgentLearning models
            learnings = []
            for row in learning_rows:
                # Parse applies_to - may be JSON string from PostgreSQL
                applies_to_raw = row.get("applies_to", {})
                if isinstance(applies_to_raw, str):
                    try:
                        applies_to = json.loads(applies_to_raw)
                    except (json.JSONDecodeError, TypeError):
                        applies_to = {}
                else:
                    applies_to = applies_to_raw if isinstance(applies_to_raw, dict) else {}

                content = row.get("learning_content", "")

                # Parse title from content if it starts with "# "
                title = "Untitled"
                if content.startswith("# "):
                    lines = content.split("\n", 1)
                    title = lines[0][2:].strip()
                    content = lines[1].strip() if len(lines) > 1 else ""

                # Calculate success rate
                success_count = row.get("success_count", 1)
                failure_count = row.get("failure_count", 0)
                total = success_count + failure_count
                success_rate = success_count / total if total > 0 else 0.0

                # Convert datetime to ISO string if needed
                created_at = row.get("created_at")
                if hasattr(created_at, "isoformat"):
                    created_at = created_at.isoformat()
                updated_at = row.get("updated_at")
                if hasattr(updated_at, "isoformat"):
                    updated_at = updated_at.isoformat()

                learnings.append(
                    AgentLearning(
                        id=row["id"],
                        agent_id=row["agent_id"],
                        learning_type=row.get("category", "unknown"),
                        title=title,
                        content=content,
                        source_context=row.get("trigger_context"),
                        applicability=(
                            applies_to.get("contexts", []) if isinstance(applies_to, dict) else []
                        ),
                        confidence=(
                            applies_to.get("confidence", 0.8)
                            if isinstance(applies_to, dict)
                            else 0.8
                        ),
                        times_applied=success_count + failure_count,
                        success_rate=success_rate,
                        tags=applies_to.get("tags", []) if isinstance(applies_to, dict) else [],
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                )

            return learnings

        except Exception as e:
            debug_logger.error(f"Error querying learnings for {agent_name}: {e}")
            return []

    async def agent_update_learning_outcome(
        self,
        learning_id: str,
        times_applied_increment: int = 1,
        new_success_rate: float | None = None,
    ) -> dict[str, Any]:
        """
        Update the application stats for a learning.

        Args:
            learning_id: ID of the learning to update
            times_applied_increment: How many times to increment application count (default 1)
            new_success_rate: If provided, indicates success (True) or failure (False)
                            via the sign - positive for success, we use a simpler bool

        Returns:
            Dict with status and message
        """
        if not self.database:
            debug_logger.warning("agent_update_learning_outcome called without database")
            return {"status": "error", "message": "Database not available"}

        try:
            # Determine success based on new_success_rate
            # If new_success_rate is provided and > 0.5, consider it a success
            success = new_success_rate is None or (
                new_success_rate is not None and new_success_rate > 0.5
            )

            # Apply updates for each increment
            for _ in range(times_applied_increment):
                await self.database.update_agent_learning_outcome(learning_id, success)

            debug_logger.info(
                f"Updated learning outcome: {learning_id}, increments: {times_applied_increment}"
            )
            return {
                "status": "success",
                "learning_id": learning_id,
                "times_applied_increment": times_applied_increment,
                "success": success,
            }

        except Exception as e:
            debug_logger.error(f"Error updating learning outcome {learning_id}: {e}")
            return {"status": "error", "message": str(e)}

    async def agent_create_notebook(
        self,
        agent_name: str,
        title: str,
        content: str,
        summary: str | None = None,
        notebook_type: str = "execution",
        context: dict[str, Any] | None = None,
        decisions_referenced: list[str] | None = None,
        learnings_referenced: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> AgentNotebookResult:
        """
        Create a notebook for an agent.

        Notebooks are narrative documents that capture agent execution summaries,
        research findings, or accumulated learnings.

        Args:
            agent_name: Name of the agent
            title: Notebook title
            content: Markdown content of the notebook
            summary: Short summary of the notebook
            notebook_type: Type of notebook (e.g., "execution", "research", "learning")
            context: Additional context dict
            decisions_referenced: List of decision IDs referenced in this notebook
            learnings_referenced: List of learning IDs referenced in this notebook
            tags: Tags for categorization and search

        Returns:
            AgentNotebookResult with notebook_id and status
        """
        if not self.database:
            debug_logger.warning("agent_create_notebook called without database")
            return AgentNotebookResult(
                notebook_id="",
                agent_id="",
                title=title,
                status="error",
                message="Database not available for creating notebook",
            )

        try:
            # Look up agent by name
            agent_data = await self.database.get_agent_by_name(agent_name)
            if not agent_data:
                return AgentNotebookResult(
                    notebook_id="",
                    agent_id="",
                    title=title,
                    status="error",
                    message=f"Agent '{agent_name}' not found. Register the agent first.",
                )

            agent_id = agent_data["id"]
            notebook_id = str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()

            # Build notebook data for database
            notebook_data = {
                "id": notebook_id,
                "agent_id": agent_id,
                "title": title,
                "summary_markdown": content,
                "notebook_type": notebook_type,
                "tags": tags or [],
                "key_insights": [],  # Could extract from content
                "related_sessions": [self._current_session_id] if self._current_session_id else [],
                "decisions_referenced": decisions_referenced or [],
                "learnings_referenced": learnings_referenced or [],
                "created_at": now,
                "updated_at": now,
            }

            await self.database.save_agent_notebook(notebook_data)

            # Update agent stats
            await self.database.update_agent_stats(agent_id, "notebooks")

            debug_logger.info(f"Created notebook {notebook_id} for agent {agent_name}")
            return AgentNotebookResult(
                notebook_id=notebook_id,
                agent_id=agent_id,
                title=title,
                status="success",
                message=f"Notebook '{title}' created successfully for agent '{agent_name}'",
            )

        except Exception as e:
            debug_logger.error(f"Error creating notebook for {agent_name}: {e}")
            return AgentNotebookResult(
                notebook_id="",
                agent_id="",
                title=title,
                status="error",
                message=f"Failed to create notebook: {str(e)}",
            )

    async def agent_query_notebooks(
        self,
        agent_name: str,
        notebook_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[AgentNotebook]:
        """
        Query notebooks for an agent.

        Args:
            agent_name: Name of the agent to query
            notebook_type: Filter by notebook type
            tags: Filter by tags (not yet implemented in DB layer)
            limit: Maximum number of results

        Returns:
            List of AgentNotebook models
        """
        if not self.database:
            debug_logger.warning("agent_query_notebooks called without database")
            return []

        try:
            # Look up agent by name
            agent_data = await self.database.get_agent_by_name(agent_name)
            if not agent_data:
                debug_logger.info(f"Agent '{agent_name}' not found for notebook query")
                return []

            agent_id = agent_data["id"]

            # Query notebooks
            notebook_rows = await self.database.query_agent_notebooks(
                agent_id=agent_id,
                notebook_type=notebook_type,
                limit=limit,
            )

            # Convert to AgentNotebook models
            notebooks = []
            for row in notebook_rows:
                notebooks.append(
                    AgentNotebook(
                        id=row["id"],
                        agent_id=row["agent_id"],
                        title=row.get("title", "Untitled"),
                        summary=None,  # Not stored separately
                        content=row.get("summary_markdown", ""),
                        notebook_type=row.get("notebook_type", "execution"),
                        context={},
                        decisions_referenced=[],  # Not stored in current schema
                        learnings_referenced=[],  # Not stored in current schema
                        tags=row.get("tags", []) if isinstance(row.get("tags"), list) else [],
                        created_at=row.get("created_at"),
                        updated_at=row.get("updated_at"),
                    )
                )

            return notebooks

        except Exception as e:
            debug_logger.error(f"Error querying notebooks for {agent_name}: {e}")
            return []

    async def agent_search_all(
        self,
        agent_name: str,
        query: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Search across decisions, learnings, and notebooks for an agent.

        Performs a simple text search across all agent content types.

        Args:
            agent_name: Name of the agent to search
            query: Search query string
            limit: Maximum results per content type

        Returns:
            Dict with 'decisions', 'learnings', and 'notebooks' lists
        """
        if not self.database:
            debug_logger.warning("agent_search_all called without database")
            return {
                "decisions": [],
                "learnings": [],
                "notebooks": [],
                "error": "Database not available",
            }

        try:
            # Look up agent by name
            agent_data = await self.database.get_agent_by_name(agent_name)
            if not agent_data:
                return {
                    "decisions": [],
                    "learnings": [],
                    "notebooks": [],
                    "error": f"Agent '{agent_name}' not found",
                }

            agent_id = agent_data["id"]
            query_lower = query.lower()

            # Query all content types and filter by search query
            # Note: This is a simple in-memory filter; for production, use FTS

            # Search decisions
            all_decisions = await self.database.query_agent_decisions(agent_id=agent_id, limit=100)
            matching_decisions = []
            for d in all_decisions:
                desc = (d.get("description") or "").lower()
                rationale = (d.get("rationale") or "").lower()
                if query_lower in desc or query_lower in rationale:
                    matching_decisions.append(d)
                    if len(matching_decisions) >= limit:
                        break

            # Search learnings
            all_learnings = await self.database.query_agent_learnings(agent_id=agent_id, limit=100)
            matching_learnings = []
            for l in all_learnings:
                content = (l.get("learning_content") or "").lower()
                trigger = (l.get("trigger_context") or "").lower()
                if query_lower in content or query_lower in trigger:
                    matching_learnings.append(l)
                    if len(matching_learnings) >= limit:
                        break

            # Search notebooks
            all_notebooks = await self.database.query_agent_notebooks(agent_id=agent_id, limit=100)
            matching_notebooks = []
            for n in all_notebooks:
                title = (n.get("title") or "").lower()
                content = (n.get("summary_markdown") or "").lower()
                if query_lower in title or query_lower in content:
                    matching_notebooks.append(n)
                    if len(matching_notebooks) >= limit:
                        break

            debug_logger.info(
                f"Search for '{query}' in agent {agent_name}: "
                f"{len(matching_decisions)} decisions, {len(matching_learnings)} learnings, "
                f"{len(matching_notebooks)} notebooks"
            )

            return {
                "agent_name": agent_name,
                "agent_id": agent_id,
                "query": query,
                "decisions": matching_decisions,
                "learnings": matching_learnings,
                "notebooks": matching_notebooks,
                "total_matches": len(matching_decisions)
                + len(matching_learnings)
                + len(matching_notebooks),
            }

        except Exception as e:
            debug_logger.error(f"Error searching agent {agent_name}: {e}")
            return {
                "decisions": [],
                "learnings": [],
                "notebooks": [],
                "error": str(e),
            }
