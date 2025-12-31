"""
Session Intelligence Data Models.

Comprehensive data models for session lifecycle management, execution tracking,
decision logging, analytics, and intelligence operations based on the PRD requirements.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


# ===== ENUMS =====

class SessionStatus(str, Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERED = "recovered"


class ExecutionStatus(str, Enum):
    """Execution step status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class ImpactLevel(str, Enum):
    """Decision impact level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PatternType(str, Enum):
    """Pattern analysis type enumeration."""
    EXECUTION = "execution"
    ERROR = "error"
    PERFORMANCE = "performance"
    WORKFLOW = "workflow"


class WorkflowType(str, Enum):
    """Workflow type enumeration."""
    TDD = "tdd"
    ATOMIC = "atomic"
    QUALITY = "quality"
    PRIME = "prime"
    CUSTOM = "custom"


class DashboardType(str, Enum):
    """Dashboard view type enumeration."""
    OVERVIEW = "overview"
    PERFORMANCE = "performance"
    AGENTS = "agents"
    DECISIONS = "decisions"
    HEALTH = "health"


class AnalysisScope(str, Enum):
    """Pattern analysis scope enumeration."""
    CURRENT = "current"
    RECENT = "recent"
    HISTORICAL = "historical"
    ALL = "all"


class ExecutionMode(str, Enum):
    """Agent execution mode enumeration."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"


class OptimizationLevel(str, Enum):
    """Optimization level enumeration."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


# ===== CORE SESSION MODELS =====

class SessionMetadata(BaseModel):
    """Session metadata and configuration."""
    session_type: str
    environment: str
    user: str
    git_branch: Optional[str] = None
    git_commit: Optional[str] = None
    parent_session: Optional[str] = None
    recovery_from: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)


class PerformanceMetrics(BaseModel):
    """Session performance metrics."""
    total_execution_time_ms: int = 0
    agents_executed: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time_ms: float = 0.0
    commands_executed: int = 0
    decisions_made: int = 0
    efficiency_score: float = 0.0


class HealthStatus(BaseModel):
    """Session health status."""
    overall_score: float = Field(default=100.0, ge=0.0, le=100.0)
    continuity_valid: bool = True
    files_valid: bool = True
    state_consistent: bool = True
    agents_healthy: bool = True
    last_health_check: Optional[datetime] = None
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CommandExecution(BaseModel):
    """Command execution details."""
    command: str
    started: datetime
    completed: Optional[datetime] = None
    duration_ms: int = 0
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    success: bool = True


class Pattern(BaseModel):
    """Detected execution pattern."""
    pattern_id: str
    pattern_type: PatternType
    description: str
    frequency: int = 1
    confidence: float = Field(ge=0.0, le=1.0)
    impact: str = "neutral"  # positive, neutral, negative


class Optimization(BaseModel):
    """Available optimization suggestion."""
    optimization_id: str
    description: str
    potential_impact: str
    effort_level: str = "medium"  # low, medium, high
    confidence: float = Field(ge=0.0, le=1.0)
    implementation_hints: List[str] = Field(default_factory=list)


class ExecutionStep(BaseModel):
    """Detailed execution step tracking."""
    step_id: str
    step_number: int
    agent: str
    operation: str
    description: str
    tools_used: List[str] = Field(default_factory=list)
    commands_executed: List[CommandExecution] = Field(default_factory=list)
    started: datetime
    completed: Optional[datetime] = None
    duration_ms: int = 0
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    patterns_detected: List[Pattern] = Field(default_factory=list)
    optimizations_available: List[Optimization] = Field(default_factory=list)


# ===== AGENT AND WORKFLOW MODELS =====

class AgentContext(BaseModel):
    """Agent execution context."""
    session_id: str
    project_path: str
    working_directory: str
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)


class AgentPerformance(BaseModel):
    """Agent performance metrics."""
    total_execution_time_ms: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    tools_used_count: int = 0
    commands_executed_count: int = 0
    average_step_time_ms: float = 0.0
    efficiency_score: float = 0.0


class AgentError(BaseModel):
    """Agent execution error details."""
    error_id: str
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    step_id: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False


class AgentExecution(BaseModel):
    """Agent execution tracking."""
    agent_name: str
    agent_type: str
    execution_id: str
    started: datetime
    completed: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.RUNNING
    execution_steps: List[ExecutionStep] = Field(default_factory=list)
    context: AgentContext
    dependencies: List[str] = Field(default_factory=list)
    performance: AgentPerformance = Field(default_factory=AgentPerformance)
    errors: List[AgentError] = Field(default_factory=list)


class StateMachine(BaseModel):
    """Workflow state machine representation."""
    current_state: str
    available_transitions: List[str] = Field(default_factory=list)
    state_data: Dict[str, Any] = Field(default_factory=dict)
    transition_history: List[Dict[str, Any]] = Field(default_factory=list)


class ParallelExecution(BaseModel):
    """Parallel execution tracking."""
    execution_id: str
    agents: List[str]
    started: datetime
    completed: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.RUNNING
    coordination_data: Dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """Workflow state management."""
    workflow_type: WorkflowType
    current_phase: str
    phases_completed: List[str] = Field(default_factory=list)
    parallel_executions: List[ParallelExecution] = Field(default_factory=list)
    state_machine: StateMachine
    optimizations_applied: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)


# ===== DECISION MODELS =====

class DecisionContext(BaseModel):
    """Decision context and rationale."""
    session_id: str
    agent_name: Optional[str] = None
    workflow_phase: Optional[str] = None
    project_state: Dict[str, Any] = Field(default_factory=dict)
    available_options: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)


class DecisionOutcome(BaseModel):
    """Decision outcome tracking."""
    outcome_id: str
    timestamp: datetime
    actual_result: str
    expected_result: str
    success: bool
    impact_measured: Optional[str] = None
    lessons_learned: List[str] = Field(default_factory=list)


class Decision(BaseModel):
    """Decision tracking and analysis."""
    decision_id: str
    timestamp: datetime
    description: str
    rationale: Optional[str] = None
    context: DecisionContext
    impact_level: ImpactLevel = ImpactLevel.MEDIUM
    related_decisions: List[str] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    outcomes: List[DecisionOutcome] = Field(default_factory=list)


# ===== ANALYTICS AND INTELLIGENCE MODELS =====

class Recommendation(BaseModel):
    """Optimization recommendation."""
    recommendation_id: str
    description: str
    rationale: str
    impact_estimate: str
    effort_estimate: str
    confidence: float = Field(ge=0.0, le=1.0)
    implementation_steps: List[str] = Field(default_factory=list)


class PatternAnalysis(BaseModel):
    """Pattern analysis results."""
    pattern_id: str
    pattern_type: PatternType
    frequency: int
    sessions_affected: List[str] = Field(default_factory=list)
    agents_involved: List[str] = Field(default_factory=list)
    description: str
    impact: str = "neutral"  # positive, neutral, negative
    recommendations: List[Recommendation] = Field(default_factory=list)
    learning_confidence: float = Field(ge=0.0, le=1.0)


class Bottleneck(BaseModel):
    """Performance bottleneck identification."""
    bottleneck_id: str
    description: str
    location: str  # agent, workflow, command, etc.
    impact_score: float = Field(ge=0.0, le=10.0)
    frequency: int = 0
    suggested_fixes: List[str] = Field(default_factory=list)


class PredictedIssue(BaseModel):
    """Predicted issue based on patterns."""
    issue_id: str
    description: str
    probability: float = Field(ge=0.0, le=1.0)
    potential_impact: ImpactLevel = ImpactLevel.MEDIUM
    prevention_steps: List[str] = Field(default_factory=list)
    early_warning_signs: List[str] = Field(default_factory=list)


class LearningInsight(BaseModel):
    """Machine learning insight."""
    insight_id: str
    description: str
    model_confidence: float = Field(ge=0.0, le=1.0)
    data_points_used: int = 0
    actionable_recommendations: List[str] = Field(default_factory=list)
    validation_status: str = "pending"  # pending, validated, rejected


class Trend(BaseModel):
    """Cross-session trend analysis."""
    trend_id: str
    description: str
    trend_direction: str  # improving, declining, stable
    trend_strength: float = Field(ge=0.0, le=1.0)
    time_period: str
    data_points: List[Dict[str, Any]] = Field(default_factory=list)


class SessionIntelligence(BaseModel):
    """Session intelligence analytics."""
    session_id: str
    efficiency_score: float = Field(ge=0.0, le=100.0)
    patterns_detected: List[PatternAnalysis] = Field(default_factory=list)
    bottlenecks: List[Bottleneck] = Field(default_factory=list)
    optimization_opportunities: List[Optimization] = Field(default_factory=list)
    predicted_issues: List[PredictedIssue] = Field(default_factory=list)
    learning_insights: List[LearningInsight] = Field(default_factory=list)
    cross_session_trends: List[Trend] = Field(default_factory=list)


class CommandAlternative(BaseModel):
    """Command alternative suggestion."""
    alternative_command: str
    expected_improvement: str
    risk_level: str = "low"  # low, medium, high
    compatibility_notes: List[str] = Field(default_factory=list)


class CommandAnalysis(BaseModel):
    """Command execution analysis."""
    command: str
    frequency: int = 0
    average_duration_ms: float = 0.0
    success_rate: float = Field(ge=0.0, le=1.0)
    error_patterns: List[str] = Field(default_factory=list)
    inefficiency_score: float = Field(ge=0.0, le=10.0)
    alternatives: List[CommandAlternative] = Field(default_factory=list)
    optimization_impact: str = "none"


# ===== MAIN SESSION MODEL =====

class Session(BaseModel):
    """Complete session representation."""
    id: str
    started: datetime
    completed: Optional[datetime] = None
    mode: str = "local"  # local, remote, hybrid
    project_name: str
    project_path: str
    status: SessionStatus = SessionStatus.ACTIVE
    metadata: SessionMetadata
    agents_executed: List[AgentExecution] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    workflow_state: Optional[WorkflowState] = None
    health_status: HealthStatus = Field(default_factory=HealthStatus)
    performance_metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)


# ===== RESULT MODELS FOR MCP FUNCTIONS =====

class SessionResult(BaseModel):
    """Result from session lifecycle management."""
    session_id: str
    operation: str
    status: str
    message: str
    session_data: Optional[Session] = None
    recovery_options: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)


class ExecutionTrackingResult(BaseModel):
    """Result from execution tracking."""
    step_id: str
    session_id: str
    agent_name: str
    status: str
    patterns_detected: List[Pattern] = Field(default_factory=list)
    optimizations: List[Optimization] = Field(default_factory=list)
    performance_impact: Optional[str] = None


class CoordinationResult(BaseModel):
    """Result from agent coordination."""
    coordination_id: str
    session_id: str
    execution_plan: Dict[str, Any]
    timing_estimate: int = 0  # estimated duration in ms
    dependency_resolution: List[str] = Field(default_factory=list)
    parallel_execution_groups: List[List[str]] = Field(default_factory=list)


class DecisionResult(BaseModel):
    """Result from decision logging."""
    decision_id: str
    session_id: str
    impact_analysis: Dict[str, Any] = Field(default_factory=dict)
    linked_decisions: List[str] = Field(default_factory=list)
    predicted_outcomes: List[str] = Field(default_factory=list)


class PatternAnalysisResult(BaseModel):
    """Result from pattern analysis."""
    analysis_id: str
    scope: AnalysisScope
    patterns: List[PatternAnalysis] = Field(default_factory=list)
    trends: List[Trend] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    learning_model_updated: bool = False


class SessionHealthResult(BaseModel):
    """Result from session health monitoring."""
    session_id: str
    health_score: float = Field(ge=0.0, le=100.0)
    issues: List[str] = Field(default_factory=list)
    recovery_actions: List[str] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    auto_recovery_attempted: bool = False


class WorkflowResult(BaseModel):
    """Result from workflow orchestration."""
    workflow_id: str
    session_id: str
    execution_plan: Dict[str, Any]
    state: WorkflowState
    progress: Dict[str, Any] = Field(default_factory=dict)
    optimizations: List[str] = Field(default_factory=list)


class CommandAnalysisResult(BaseModel):
    """Result from command analysis."""
    session_id: str
    analysis_period: str
    patterns: List[str] = Field(default_factory=list)
    inefficiencies: List[str] = Field(default_factory=list)
    suggestions: List[CommandAlternative] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)


class MissingFunctionResult(BaseModel):
    """Result from missing function tracking."""
    session_id: str
    functions: List[str] = Field(default_factory=list)
    priorities: Dict[str, int] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)
    impact: Dict[str, str] = Field(default_factory=dict)


class DashboardResult(BaseModel):
    """Result from dashboard generation."""
    dashboard_type: DashboardType
    session_id: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    visualizations: List[Dict[str, Any]] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    real_time_data: bool = False