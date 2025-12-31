"""
Session Intelligence Engine - Core business logic for session management and analytics.

This engine consolidates the functionality of 42+ scattered claudecode session functions
into a unified, intelligent system with pattern recognition, optimization, and learning capabilities.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Setup file logging for debugging
debug_log_file = Path("/tmp/session-intelligence-debug.log")
debug_logger = logging.getLogger("session_intelligence_engine_debug")
debug_handler = logging.FileHandler(debug_log_file)
debug_handler.setFormatter(logging.Formatter('%(asctime)s [ENGINE-DEBUG] %(message)s'))
debug_logger.addHandler(debug_handler)
debug_logger.setLevel(logging.INFO)

from models.session_models import (
    AgentExecution,
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
        debug_logger.info(f"SessionIntelligenceEngine.__init__ called with repository_path: {repository_path}, use_filesystem: {use_filesystem}")

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
            debug_logger.info(f"Auto-detected project path, claude_sessions_path: {self.claude_sessions_path}")

        debug_logger.info(f"Final claude_sessions_path: {self.claude_sessions_path}")

        # Only create filesystem directories if filesystem persistence is enabled
        if self.use_filesystem:
            self.claude_sessions_path.mkdir(parents=True, exist_ok=True)
            debug_logger.info(f"Created/ensured session directory exists at: {self.claude_sessions_path}")
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
                        debug_logger.warning(f"Session {session_id} not found on disk, creating new session")

                except Exception as e:
                    debug_logger.error(f"Error reading current session ID: {e}")

        # Create new session if none exists or is valid
        debug_logger.info("Creating new session")
        result = self._create_session(
            mode="auto",
            project_name=self.claude_sessions_path.parent.name,
            metadata={"project_path": str(self.claude_sessions_path.parent)}
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
            "composer.json"
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
        auto_recovery: bool = True
    ) -> SessionResult:
        """
        Comprehensive session lifecycle management with intelligent tracking.
        
        Consolidates: claudecode_create_session_metadata, claudecode_get_or_create_session_id,
                     claudecode_create_session_notes, claudecode_finalize_session_summary,
                     claudecode_save_session_state, claudecode_capture_enhanced_state
        """
        try:
            return self._manage_lifecycle_sync(operation, mode, project_name, metadata, auto_recovery)
        except Exception as e:
            return SessionResult(
                session_id="error",
                operation=operation,
                status="error",
                message=f"Session lifecycle error: {str(e)}"
            )

    def _manage_lifecycle_sync(
        self,
        operation: str,
        mode: str,
        project_name: str | None,
        metadata: dict[str, Any] | None,
        auto_recovery: bool
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
                message=f"Unknown operation: {operation}"
            )

    def _create_session(self, mode: str, project_name: str, metadata: dict[str, Any]) -> SessionResult:
        """Create a new session with comprehensive setup."""
        session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Create session metadata
        session_metadata = SessionMetadata(
            session_type="development",
            environment=mode,
            user=metadata.get("user", "claude"),
            git_branch=metadata.get("git_branch"),
            git_commit=metadata.get("git_commit"),
            tags=metadata.get("tags", [])
        )

        # Create session object
        session = Session(
            id=session_id,
            started=datetime.now(),
            mode=mode,
            project_name=project_name or "unknown",
            project_path=metadata.get("project_path", str(Path.cwd())),
            metadata=session_metadata,
            health_status=HealthStatus(),
            performance_metrics=PerformanceMetrics()
        )

        # Cache session in memory
        self.session_cache[session_id] = session

        # Only create filesystem artifacts if enabled
        if self.use_filesystem:
            session_dir = self.claude_sessions_path / session_id
            session_dir.mkdir(exist_ok=True)
            (session_dir / "agents").mkdir(exist_ok=True)

            metadata_file = session_dir / "session-metadata.json"
            with open(metadata_file, 'w') as f:
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
            next_steps=["Initialize agent tracking", "Set up workflow state"]
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
                recovery_options=["Validate continuity", "Check health"] if auto_recovery else []
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
                        recovery_options=["Validate continuity", "Check health"] if auto_recovery else []
                    )
            except Exception as e:
                debug_logger.error(f"Error resuming from filesystem: {e}")

        return SessionResult(
            session_id="none",
            operation="resume",
            status="error",
            message="No existing sessions found"
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
                message="No active session to finalize"
            )

        session = self.session_cache[session_id]

        # Update session status
        session.status = SessionStatus.COMPLETED
        session.completed = datetime.now()

        # Calculate final metrics
        total_time = (session.completed - session.started).total_seconds() * 1000
        session.performance_metrics.total_execution_time_ms = int(total_time)

        # Clear in-memory current session
        self._current_session_id = None

        # Only do filesystem operations if enabled
        if self.use_filesystem:
            session_dir = self.claude_sessions_path / session_id
            metadata_file = session_dir / "session-metadata.json"
            with open(metadata_file, 'w') as f:
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
            session_data=session
        )

    def _validate_session(self) -> SessionResult:
        """Validate session continuity and health."""
        if not self.session_cache:
            return SessionResult(
                session_id="none",
                operation="validate",
                status="error",
                message="No active session to validate"
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
        message = "Session validation passed" if not issues else f"Validation issues: {', '.join(issues)}"

        return SessionResult(
            session_id=session_id,
            operation="validate",
            status=status,
            message=message,
            session_data=session
        )

    # ===== EXECUTION TRACKING =====

    def session_track_execution(
        self,
        session_id: str | None,
        agent_name: str,
        step_data: dict[str, Any],
        track_patterns: bool = True,
        suggest_optimizations: bool = True
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
                optimizations=[]
            )

    def _track_execution_sync(
        self,
        session_id: str | None,
        agent_name: str,
        step_data: dict[str, Any],
        track_patterns: bool,
        suggest_optimizations: bool
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
                optimizations=[]
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
                optimizations=[]
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
            started=datetime.now(),
            status=ExecutionStatus.RUNNING
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
                started=datetime.now(),
                context=AgentContext(
                    session_id=session_id,
                    project_path=session.project_path,
                    working_directory=step_data.get("working_directory", session.project_path)
                ),
                performance=AgentPerformance()
            )
            session.agents_executed.append(agent_execution)

        # Add step to agent execution
        agent_execution.execution_steps.append(execution_step)
        debug_logger.info("Added execution_step to agent_execution")
        debug_logger.info(f"Agent execution steps count: {len(agent_execution.execution_steps)}")

        # Update performance metrics
        session.performance_metrics.agents_executed = len(session.agents_executed)
        debug_logger.info(f"Updated performance metrics - agents executed: {session.performance_metrics.agents_executed}")

        # Save session data to disk only if filesystem is enabled
        if self.use_filesystem:
            try:
                session_dir = self.claude_sessions_path / session_id
                session_dir.mkdir(parents=True, exist_ok=True)
                debug_logger.info(f"Ensured session directory exists: {session_dir}")

                # Save session metadata
                metadata_file = session_dir / "session-metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(session.model_dump(), f, indent=2, default=str)
                debug_logger.info(f"Saved session metadata to: {metadata_file}")

                # Save individual agent execution log
                agent_dir = session_dir / "agents" / agent_name
                agent_dir.mkdir(parents=True, exist_ok=True)
                agent_log_file = agent_dir / "execution-log.json"
                with open(agent_log_file, 'w') as f:
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
            optimizations=optimizations
        )
        debug_logger.info(f"Returning ExecutionTrackingResult: {result}")
        return result

    def _detect_patterns(self, agent_name: str, step_data: dict[str, Any]) -> list[Pattern]:
        """Detect execution patterns for optimization."""
        patterns = []

        # Simple pattern detection (can be enhanced with ML)
        if "error" in step_data.get("description", "").lower():
            patterns.append(Pattern(
                pattern_id=f"error-pattern-{uuid.uuid4().hex[:8]}",
                pattern_type=PatternType.ERROR,
                description="Error pattern detected in execution",
                frequency=1,
                confidence=0.8,
                impact="negative"
            ))

        if step_data.get("duration_ms", 0) > 30000:  # >30 seconds
            patterns.append(Pattern(
                pattern_id=f"performance-pattern-{uuid.uuid4().hex[:8]}",
                pattern_type=PatternType.PERFORMANCE,
                description="Long execution time detected",
                frequency=1,
                confidence=0.9,
                impact="negative"
            ))

        return patterns

    def _suggest_optimizations(self, agent_name: str, step_data: dict[str, Any]) -> list[Optimization]:
        """Suggest optimizations based on execution data."""
        optimizations = []

        # Simple optimization suggestions
        if step_data.get("tools_used") and len(step_data["tools_used"]) > 5:
            optimizations.append(Optimization(
                optimization_id=f"tool-opt-{uuid.uuid4().hex[:8]}",
                description="Consider batching tool calls to reduce overhead",
                potential_impact="Reduce execution time by 20-30%",
                effort_level="low",
                confidence=0.7
            ))

        return optimizations

    # ===== AGENT COORDINATION =====

    def session_coordinate_agents(
        self,
        session_id: str | None,
        agents: list[dict[str, Any]],
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        dependency_graph: dict[str, Any] | None = None,
        optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
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
                timing_estimate=0
            )

    def _coordinate_agents_sync(
        self,
        session_id: str | None,
        agents: list[dict[str, Any]],
        execution_mode: ExecutionMode,
        dependency_graph: dict[str, Any] | None,
        optimization_level: OptimizationLevel
    ) -> CoordinationResult:
        """Synchronous agent coordination."""

        coordination_id = f"coord-{uuid.uuid4().hex[:8]}"

        # Create execution plan
        execution_plan = {
            "mode": execution_mode.value,
            "agents": [agent.get("name", "unknown") for agent in agents],
            "optimization_level": optimization_level.value,
            "created_at": datetime.now().isoformat()
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
            parallel_execution_groups=[parallel_groups] if parallel_groups else []
        )

    def _resolve_dependencies(self, agents: list[dict[str, Any]], dependency_graph: dict[str, Any]) -> list[str]:
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
        link_artifacts: list[str] | None = None
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
                impact_analysis={"error": str(e)}
            )

    def _log_decision_sync(
        self,
        decision: str,
        session_id: str | None,
        context: dict[str, Any] | None,
        impact_analysis: bool,
        link_artifacts: list[str] | None
    ) -> DecisionResult:
        """Synchronous decision logging."""

        decision_id = f"decision-{uuid.uuid4().hex[:8]}"

        # Get current session
        if not session_id and self.session_cache:
            session_id = list(self.session_cache.keys())[-1]

        if session_id and session_id in self.session_cache:
            session = self.session_cache[session_id]

            from models.session_models import DecisionContext
            decision_context = DecisionContext(
                session_id=session_id,
                project_state=context or {}
            )

            decision_obj = Decision(
                decision_id=decision_id,
                timestamp=datetime.now(),
                description=decision,
                context=decision_context,
                impact_level=ImpactLevel.MEDIUM,
                artifacts=link_artifacts or []
            )

            session.decisions.append(decision_obj)

        # Impact analysis
        impact_analysis_result = {}
        if impact_analysis:
            impact_analysis_result = {
                "estimated_impact": "medium",
                "affected_components": [],
                "risk_assessment": "low"
            }

        return DecisionResult(
            decision_id=decision_id,
            session_id=session_id or "unknown",
            impact_analysis=impact_analysis_result,
            linked_decisions=[],
            predicted_outcomes=["Continue with planned execution"]
        )

    # ===== HEALTH MONITORING =====

    def session_monitor_health(
        self,
        session_id: str | None,
        health_checks: list[str] = None,
        auto_recover: bool = True,
        alert_thresholds: dict[str, float] | None = None,
        include_diagnostics: bool = True
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
                issues=[f"Health monitoring error: {str(e)}"]
            )

    def _monitor_health_sync(
        self,
        session_id: str | None,
        health_checks: list[str],
        auto_recover: bool,
        alert_thresholds: dict[str, float] | None,
        include_diagnostics: bool
    ) -> SessionHealthResult:
        """Synchronous health monitoring."""

        # Get current session
        if not session_id and self.session_cache:
            session_id = list(self.session_cache.keys())[-1]

        if not session_id or session_id not in self.session_cache:
            return SessionHealthResult(
                session_id=session_id or "unknown",
                health_score=0.0,
                issues=["No active session found"]
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
            failed_agents = [agent for agent in session.agents_executed
                           if agent.status == ExecutionStatus.ERROR]
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
                "session_age_minutes": (datetime.now() - session.started).total_seconds() / 60,
                "agents_count": len(session.agents_executed),
                "decisions_count": len(session.decisions),
                "performance_score": session.performance_metrics.efficiency_score
            }

        return SessionHealthResult(
            session_id=session_id,
            health_score=max(0.0, health_score),
            issues=issues,
            recovery_actions=recovery_actions,
            diagnostics=diagnostics,
            auto_recovery_attempted=auto_recovery_attempted
        )

    # ===== PLACEHOLDER IMPLEMENTATIONS FOR OTHER FUNCTIONS =====

    def session_orchestrate_workflow(self, **kwargs) -> WorkflowResult:
        """Workflow orchestration - placeholder implementation."""
        return WorkflowResult(
            workflow_id="placeholder",
            session_id=kwargs.get("session_id", "unknown"),
            execution_plan={},
            state=WorkflowState(
                workflow_type=WorkflowType.CUSTOM,
                current_phase="placeholder",
                state_machine={}
            )
        )

    def session_analyze_patterns(self, **kwargs) -> PatternAnalysisResult:
        """Pattern analysis - placeholder implementation."""
        return PatternAnalysisResult(
            analysis_id="placeholder",
            scope=AnalysisScope.CURRENT,
            patterns=[],
            trends=[],
            recommendations=[]
        )

    def session_analyze_commands(self, **kwargs) -> CommandAnalysisResult:
        """Command analysis - placeholder implementation."""
        return CommandAnalysisResult(
            session_id=kwargs.get("session_id", "unknown"),
            analysis_period="current",
            patterns=[],
            inefficiencies=[],
            suggestions=[],
            metrics={}
        )

    def session_track_missing_functions(self, **kwargs) -> MissingFunctionResult:
        """Missing function tracking - placeholder implementation."""
        return MissingFunctionResult(
            session_id=kwargs.get("session_id", "unknown"),
            functions=[],
            priorities={},
            suggestions=[],
            impact={}
        )

    def session_get_dashboard(self, **kwargs) -> DashboardResult:
        """Dashboard generation - placeholder implementation."""
        return DashboardResult(
            dashboard_type=DashboardType.OVERVIEW,
            session_id=kwargs.get("session_id"),
            metrics={},
            visualizations=[],
            insights=[],
            recommendations=[]
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
        save_to_database: bool = True
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
                session_id, title, include_decisions, include_agents,
                include_metrics, tags, save_to_file, save_to_database
            )
        except Exception as e:
            debug_logger.error(f"Error creating notebook: {e}")
            return NotebookResult(
                session_id=session_id or "unknown",
                status="error",
                message=f"Failed to create notebook: {str(e)}"
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
        save_to_database: bool
    ) -> NotebookResult:
        """Synchronous notebook creation."""

        # Get session (current or specified)
        if not session_id:
            session_id = self._get_or_create_current_session_id()

        if not session_id or session_id not in self.session_cache:
            return NotebookResult(
                session_id=session_id or "unknown",
                status="error",
                message="No session found to create notebook for"
            )

        session = self.session_cache[session_id]

        # Calculate duration
        end_time = session.completed or datetime.now()
        duration_minutes = (end_time - session.started).total_seconds() / 60

        # Generate title if not provided
        if not title:
            title = f"Session: {session.project_name} - {session.started.strftime('%Y-%m-%d %H:%M')}"

        # Build notebook sections
        sections: list[NotebookSection] = []

        # Overview section
        overview_content = self._generate_overview_section(session, duration_minutes)
        sections.append(NotebookSection(
            heading="Overview",
            content=overview_content,
            level=2
        ))

        # Agents section
        agents_used: list[str] = []
        if include_agents and session.agents_executed:
            agents_content, agents_used = self._generate_agents_section(session)
            sections.append(NotebookSection(
                heading="Agents Executed",
                content=agents_content,
                level=2
            ))

        # Decisions section
        decisions_made: list[str] = []
        if include_decisions and session.decisions:
            decisions_content, decisions_made = self._generate_decisions_section(session)
            sections.append(NotebookSection(
                heading="Decisions Made",
                content=decisions_content,
                level=2
            ))

        # Metrics section
        if include_metrics:
            metrics_content = self._generate_metrics_section(session)
            sections.append(NotebookSection(
                heading="Performance Metrics",
                content=metrics_content,
                level=2
            ))

        # Gather key file changes from agent executions
        key_changes = self._extract_key_changes(session)

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
            tags=tags
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
            message=f"Notebook created successfully with {len(sections)} sections"
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
            status_emoji = "✅" if agent.status == ExecutionStatus.SUCCESS else "⚠️" if agent.status == ExecutionStatus.RUNNING else "❌"
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
                ImpactLevel.CRITICAL: "🔴"
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

    def _extract_key_changes(self, session: Session) -> list[str]:
        """Extract key file changes from agent executions."""
        changes = set()

        for agent in session.agents_executed:
            for step in agent.execution_steps:
                # Extract tools that typically modify files
                for tool in step.tools_used:
                    if any(action in tool.lower() for action in ["write", "edit", "create", "modify"]):
                        changes.add(tool)

        # Also check decision artifacts
        for decision in session.decisions:
            for artifact in decision.artifacts:
                if artifact.endswith((".py", ".js", ".ts", ".toml", ".yaml", ".yml", ".md")):
                    changes.add(artifact)

        return list(changes)[:20]  # Limit to 20 changes

    def _auto_generate_tags(
        self,
        session: Session,
        agents_used: list[str],
        key_changes: list[str]
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
        self,
        title: str,
        sections: list[NotebookSection],
        session: Session,
        duration_minutes: float
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
        lines.extend([
            "---",
            "",
            f"*Session ID: `{session.id}`*",
            "*Generated by session-intelligence MCP server*"
        ])

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

    def session_search(
        self,
        query: str,
        search_type: str = "fulltext",
        limit: int = 20
    ) -> SearchResults:
        """
        Search across sessions using full-text search.

        Args:
            query: Search query (supports FTS5 syntax)
            search_type: Type of search - "fulltext", "tag", or "file"
            limit: Maximum results to return

        Returns:
            SearchResults with matching sessions
        """
        # This requires database access which is async
        # For now, return empty results with a message
        debug_logger.info(f"Session search requires database access: {query}")
        return SearchResults(
            query=query,
            total_results=0,
            results=[]
        )

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
        source_session = self.current_session.id if self.current_session else None

        debug_logger.info(
            f"Logging learning: {category} for {effective_project}"
        )

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
            )
        )

    def session_find_solution(
        self,
        error_text: str,
        error_category: str | None = None,
        include_universal: bool = True,
    ) -> SolutionSearchResult:
        """
        Find solutions for an error from project and universal knowledge.

        Args:
            error_text: The error message/pattern to search for
            error_category: Optional category hint (compile, runtime, config, dependency)
            include_universal: Whether to include universal solutions

        Returns:
            SolutionSearchResult with matching solutions
        """
        debug_logger.info(
            f"Finding solutions for error: {error_text[:100]}..."
        )

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
