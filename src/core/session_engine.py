"""
Session Intelligence Engine - Core business logic for session management and analytics.

This engine consolidates the functionality of 42+ scattered claudecode session functions
into a unified, intelligent system with pattern recognition, optimization, and learning capabilities.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Setup file logging for debugging
debug_log_file = Path("/tmp/session-intelligence-debug.log")
debug_logger = logging.getLogger("session_intelligence_engine_debug")
debug_handler = logging.FileHandler(debug_log_file)
debug_handler.setFormatter(logging.Formatter('%(asctime)s [ENGINE-DEBUG] %(message)s'))
debug_logger.addHandler(debug_handler)
debug_logger.setLevel(logging.INFO)

from models.session_models import (
    Session, SessionMetadata, SessionResult, SessionStatus, ExecutionStep, 
    AgentExecution, Decision, WorkflowState, PatternAnalysis, SessionIntelligence,
    CommandAnalysis, ExecutionTrackingResult, CoordinationResult, DecisionResult,
    PatternAnalysisResult, SessionHealthResult, WorkflowResult, CommandAnalysisResult,
    MissingFunctionResult, DashboardResult, HealthStatus, PerformanceMetrics,
    ExecutionStatus, PatternType, WorkflowType, DashboardType, AnalysisScope,
    ExecutionMode, OptimizationLevel, ImpactLevel, Pattern, Optimization
)


class SessionIntelligenceEngine:
    """
    Core session intelligence engine providing unified session management,
    execution tracking, pattern analysis, and optimization capabilities.
    
    Consolidates 42+ claudecode session functions into intelligent operations.
    """
    
    def __init__(
        self,
        repository_path: Optional[str] = None,
        use_filesystem: bool = True,
        database: Optional[Any] = None,
    ):
        """Initialize the session intelligence engine.

        Args:
            repository_path: Path to the repository root. If None, auto-detects project root.
            use_filesystem: If True, persist sessions to filesystem. If False, use memory only.
                           Set to False for HTTP transport to avoid creating local folders.
            database: Optional async database for persistence (used by HTTP server).
        """
        debug_logger.info(f"SessionIntelligenceEngine.__init__ called with repository_path: {repository_path}, use_filesystem: {use_filesystem}")

        self.session_cache: Dict[str, Session] = {}
        self.pattern_cache: Dict[str, List[PatternAnalysis]] = {}
        self.use_filesystem = use_filesystem
        self.database = database  # Optional database for persistence
        self._current_session_id: Optional[str] = None

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

    def _get_or_create_current_session_id(self) -> Optional[str]:
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
                        with open(metadata_file, 'r') as f:
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
        project_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
        project_name: Optional[str],
        metadata: Optional[Dict[str, Any]],
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
    
    def _create_session(self, mode: str, project_name: str, metadata: Dict[str, Any]) -> SessionResult:
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
                        with open(metadata_file, 'r') as f:
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
        session_id: Optional[str],
        agent_name: str,
        step_data: Dict[str, Any],
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
        except Exception as e:
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
        session_id: Optional[str],
        agent_name: str,
        step_data: Dict[str, Any],
        track_patterns: bool,
        suggest_optimizations: bool
    ) -> ExecutionTrackingResult:
        """Synchronous execution tracking."""
        
        debug_logger.info(f"_track_execution_sync called")
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
            debug_logger.error(f"ERROR: No session_id available after auto-detection")
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
        debug_logger.info(f"Added execution_step to agent_execution")
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
            debug_logger.info(f"Execution tracking updated in memory only (filesystem disabled)")
        
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
    
    def _detect_patterns(self, agent_name: str, step_data: Dict[str, Any]) -> List[Pattern]:
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
    
    def _suggest_optimizations(self, agent_name: str, step_data: Dict[str, Any]) -> List[Optimization]:
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
        session_id: Optional[str],
        agents: List[Dict[str, Any]],
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        dependency_graph: Optional[Dict[str, Any]] = None,
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
        session_id: Optional[str],
        agents: List[Dict[str, Any]],
        execution_mode: ExecutionMode,
        dependency_graph: Optional[Dict[str, Any]],
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
    
    def _resolve_dependencies(self, agents: List[Dict[str, Any]], dependency_graph: Dict[str, Any]) -> List[str]:
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
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        impact_analysis: bool = True,
        link_artifacts: Optional[List[str]] = None
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
        session_id: Optional[str],
        context: Optional[Dict[str, Any]],
        impact_analysis: bool,
        link_artifacts: Optional[List[str]]
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
        session_id: Optional[str],
        health_checks: List[str] = None,
        auto_recover: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None,
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
        session_id: Optional[str],
        health_checks: List[str],
        auto_recover: bool,
        alert_thresholds: Optional[Dict[str, float]],
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