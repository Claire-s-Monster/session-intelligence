"""
Session Intelligence Data Models.

Comprehensive data models for session lifecycle management, execution tracking,
decision logging, analytics, and intelligence operations based on the PRD requirements.
"""

from datetime import datetime
from enum import Enum
from typing import Any

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
    git_branch: str | None = None
    git_commit: str | None = None
    parent_session: str | None = None
    recovery_from: str | None = None
    tags: list[str] = Field(default_factory=list)
    custom_attributes: dict[str, Any] = Field(default_factory=dict)


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
    last_health_check: datetime | None = None
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CommandExecution(BaseModel):
    """Command execution details."""
    command: str
    started: datetime
    completed: datetime | None = None
    duration_ms: int = 0
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    success: bool = True


class FileOperation(BaseModel):
    """File operation tracking for session notebooks."""
    session_id: str
    timestamp: datetime
    operation: str  # create, edit, delete, read
    file_path: str
    lines_added: int = 0
    lines_removed: int = 0
    summary: str | None = None
    tool_name: str | None = None


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
    implementation_hints: list[str] = Field(default_factory=list)


class ExecutionStep(BaseModel):
    """Detailed execution step tracking."""
    step_id: str
    step_number: int
    agent: str
    operation: str
    description: str
    tools_used: list[str] = Field(default_factory=list)
    commands_executed: list[CommandExecution] = Field(default_factory=list)
    started: datetime
    completed: datetime | None = None
    duration_ms: int = 0
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: str | None = None
    error: str | None = None
    patterns_detected: list[Pattern] = Field(default_factory=list)
    optimizations_available: list[Optimization] = Field(default_factory=list)


# ===== AGENT AND WORKFLOW MODELS =====

class AgentContext(BaseModel):
    """Agent execution context."""
    session_id: str
    project_path: str
    working_directory: str
    environment_variables: dict[str, str] = Field(default_factory=dict)
    configuration: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)


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
    stack_trace: str | None = None
    step_id: str | None = None
    recovery_attempted: bool = False
    recovery_successful: bool = False


class AgentExecution(BaseModel):
    """Agent execution tracking."""
    agent_name: str
    agent_type: str
    execution_id: str
    started: datetime
    completed: datetime | None = None
    status: ExecutionStatus = ExecutionStatus.RUNNING
    execution_steps: list[ExecutionStep] = Field(default_factory=list)
    context: AgentContext
    dependencies: list[str] = Field(default_factory=list)
    performance: AgentPerformance = Field(default_factory=AgentPerformance)
    errors: list[AgentError] = Field(default_factory=list)


class StateMachine(BaseModel):
    """Workflow state machine representation."""
    current_state: str
    available_transitions: list[str] = Field(default_factory=list)
    state_data: dict[str, Any] = Field(default_factory=dict)
    transition_history: list[dict[str, Any]] = Field(default_factory=list)


class ParallelExecution(BaseModel):
    """Parallel execution tracking."""
    execution_id: str
    agents: list[str]
    started: datetime
    completed: datetime | None = None
    status: ExecutionStatus = ExecutionStatus.RUNNING
    coordination_data: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """Workflow state management."""
    workflow_type: WorkflowType
    current_phase: str
    phases_completed: list[str] = Field(default_factory=list)
    parallel_executions: list[ParallelExecution] = Field(default_factory=list)
    state_machine: StateMachine
    optimizations_applied: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


# ===== DECISION MODELS =====

class DecisionContext(BaseModel):
    """Decision context and rationale."""
    session_id: str
    agent_name: str | None = None
    workflow_phase: str | None = None
    project_state: dict[str, Any] = Field(default_factory=dict)
    available_options: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class DecisionOutcome(BaseModel):
    """Decision outcome tracking."""
    outcome_id: str
    timestamp: datetime
    actual_result: str
    expected_result: str
    success: bool
    impact_measured: str | None = None
    lessons_learned: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    """Decision tracking and analysis."""
    decision_id: str
    timestamp: datetime
    description: str
    rationale: str | None = None
    context: DecisionContext
    impact_level: ImpactLevel = ImpactLevel.MEDIUM
    related_decisions: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    outcomes: list[DecisionOutcome] = Field(default_factory=list)


# ===== ANALYTICS AND INTELLIGENCE MODELS =====

class Recommendation(BaseModel):
    """Optimization recommendation."""
    recommendation_id: str
    description: str
    rationale: str
    impact_estimate: str
    effort_estimate: str
    confidence: float = Field(ge=0.0, le=1.0)
    implementation_steps: list[str] = Field(default_factory=list)


class PatternAnalysis(BaseModel):
    """Pattern analysis results."""
    pattern_id: str
    pattern_type: PatternType
    frequency: int
    sessions_affected: list[str] = Field(default_factory=list)
    agents_involved: list[str] = Field(default_factory=list)
    description: str
    impact: str = "neutral"  # positive, neutral, negative
    recommendations: list[Recommendation] = Field(default_factory=list)
    learning_confidence: float = Field(ge=0.0, le=1.0)


class Bottleneck(BaseModel):
    """Performance bottleneck identification."""
    bottleneck_id: str
    description: str
    location: str  # agent, workflow, command, etc.
    impact_score: float = Field(ge=0.0, le=10.0)
    frequency: int = 0
    suggested_fixes: list[str] = Field(default_factory=list)


class PredictedIssue(BaseModel):
    """Predicted issue based on patterns."""
    issue_id: str
    description: str
    probability: float = Field(ge=0.0, le=1.0)
    potential_impact: ImpactLevel = ImpactLevel.MEDIUM
    prevention_steps: list[str] = Field(default_factory=list)
    early_warning_signs: list[str] = Field(default_factory=list)


class LearningInsight(BaseModel):
    """Machine learning insight."""
    insight_id: str
    description: str
    model_confidence: float = Field(ge=0.0, le=1.0)
    data_points_used: int = 0
    actionable_recommendations: list[str] = Field(default_factory=list)
    validation_status: str = "pending"  # pending, validated, rejected


class Trend(BaseModel):
    """Cross-session trend analysis."""
    trend_id: str
    description: str
    trend_direction: str  # improving, declining, stable
    trend_strength: float = Field(ge=0.0, le=1.0)
    time_period: str
    data_points: list[dict[str, Any]] = Field(default_factory=list)


class SessionIntelligence(BaseModel):
    """Session intelligence analytics."""
    session_id: str
    efficiency_score: float = Field(ge=0.0, le=100.0)
    patterns_detected: list[PatternAnalysis] = Field(default_factory=list)
    bottlenecks: list[Bottleneck] = Field(default_factory=list)
    optimization_opportunities: list[Optimization] = Field(default_factory=list)
    predicted_issues: list[PredictedIssue] = Field(default_factory=list)
    learning_insights: list[LearningInsight] = Field(default_factory=list)
    cross_session_trends: list[Trend] = Field(default_factory=list)


class CommandAlternative(BaseModel):
    """Command alternative suggestion."""
    alternative_command: str
    expected_improvement: str
    risk_level: str = "low"  # low, medium, high
    compatibility_notes: list[str] = Field(default_factory=list)


class CommandAnalysis(BaseModel):
    """Command execution analysis."""
    command: str
    frequency: int = 0
    average_duration_ms: float = 0.0
    success_rate: float = Field(ge=0.0, le=1.0)
    error_patterns: list[str] = Field(default_factory=list)
    inefficiency_score: float = Field(ge=0.0, le=10.0)
    alternatives: list[CommandAlternative] = Field(default_factory=list)
    optimization_impact: str = "none"


# ===== MAIN SESSION MODEL =====

class Session(BaseModel):
    """Complete session representation."""
    id: str
    started: datetime
    completed: datetime | None = None
    mode: str = "local"  # local, remote, hybrid
    project_name: str
    project_path: str
    status: SessionStatus = SessionStatus.ACTIVE
    metadata: SessionMetadata
    agents_executed: list[AgentExecution] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    workflow_state: WorkflowState | None = None
    health_status: HealthStatus = Field(default_factory=HealthStatus)
    performance_metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)


# ===== RESULT MODELS FOR MCP FUNCTIONS =====

class SessionResult(BaseModel):
    """Result from session lifecycle management."""
    session_id: str
    operation: str
    status: str
    message: str
    session_data: Session | None = None
    recovery_options: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class ExecutionTrackingResult(BaseModel):
    """Result from execution tracking."""
    step_id: str
    session_id: str
    agent_name: str
    status: str
    patterns_detected: list[Pattern] = Field(default_factory=list)
    optimizations: list[Optimization] = Field(default_factory=list)
    performance_impact: str | None = None


class CoordinationResult(BaseModel):
    """Result from agent coordination."""
    coordination_id: str
    session_id: str
    execution_plan: dict[str, Any]
    timing_estimate: int = 0  # estimated duration in ms
    dependency_resolution: list[str] = Field(default_factory=list)
    parallel_execution_groups: list[list[str]] = Field(default_factory=list)


class DecisionResult(BaseModel):
    """Result from decision logging."""
    decision_id: str
    session_id: str
    impact_analysis: dict[str, Any] = Field(default_factory=dict)
    linked_decisions: list[str] = Field(default_factory=list)
    predicted_outcomes: list[str] = Field(default_factory=list)


class PatternAnalysisResult(BaseModel):
    """Result from pattern analysis."""
    analysis_id: str
    scope: AnalysisScope
    patterns: list[PatternAnalysis] = Field(default_factory=list)
    trends: list[Trend] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    learning_model_updated: bool = False


class SessionHealthResult(BaseModel):
    """Result from session health monitoring."""
    session_id: str
    health_score: float = Field(ge=0.0, le=100.0)
    issues: list[str] = Field(default_factory=list)
    recovery_actions: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    auto_recovery_attempted: bool = False


class WorkflowResult(BaseModel):
    """Result from workflow orchestration."""
    workflow_id: str
    session_id: str
    execution_plan: dict[str, Any]
    state: WorkflowState
    progress: dict[str, Any] = Field(default_factory=dict)
    optimizations: list[str] = Field(default_factory=list)


class CommandAnalysisResult(BaseModel):
    """Result from command analysis."""
    session_id: str
    analysis_period: str
    patterns: list[str] = Field(default_factory=list)
    inefficiencies: list[str] = Field(default_factory=list)
    suggestions: list[CommandAlternative] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class MissingFunctionResult(BaseModel):
    """Result from missing function tracking."""
    session_id: str
    functions: list[str] = Field(default_factory=list)
    priorities: dict[str, int] = Field(default_factory=dict)
    suggestions: list[str] = Field(default_factory=list)
    impact: dict[str, str] = Field(default_factory=dict)


class DashboardResult(BaseModel):
    """Result from dashboard generation."""
    dashboard_type: DashboardType
    session_id: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    visualizations: list[dict[str, Any]] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    real_time_data: bool = False


# ===== SESSION NOTEBOOK MODELS =====

class SessionSummary(BaseModel):
    """Session summary for notebook generation."""
    session_id: str
    title: str
    summary_markdown: str
    key_changes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_at: str


class NotebookSection(BaseModel):
    """A section of the session notebook."""
    heading: str
    content: str
    level: int = 2  # Markdown heading level (##, ###, etc.)


class SessionNotebook(BaseModel):
    """Complete session notebook/summary document."""
    session_id: str
    title: str
    created_at: str
    project_name: str
    project_path: str
    duration_minutes: float
    sections: list[NotebookSection] = Field(default_factory=list)
    summary_markdown: str
    key_changes: list[str] = Field(default_factory=list)
    agents_used: list[str] = Field(default_factory=list)
    decisions_made: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class NotebookResult(BaseModel):
    """Result from session_create_notebook tool."""
    session_id: str
    status: str
    notebook: SessionNotebook | None = None
    markdown_output: str = ""
    file_path: str | None = None
    search_indexed: bool = False
    message: str = ""


class SearchResult(BaseModel):
    """Result from session search."""
    session_id: str
    title: str | None = None
    snippet: str = ""
    relevance: float = 0.0
    project_name: str | None = None
    project_path: str | None = None
    started_at: str | None = None
    tags: list[str] = Field(default_factory=list)


class SearchResults(BaseModel):
    """Collection of search results."""
    query: str
    total_results: int
    results: list[SearchResult] = Field(default_factory=list)


# ===== KNOWLEDGE SYSTEM MODELS =====


class LearningCategory(str, Enum):
    """Learning category enumeration."""
    ERROR_FIX = "error_fix"
    PATTERN = "pattern"
    PREFERENCE = "preference"
    WORKFLOW = "workflow"


class ErrorCategory(str, Enum):
    """Error category enumeration."""
    COMPILE = "compile"
    RUNTIME = "runtime"
    CONFIG = "config"
    DEPENDENCY = "dependency"
    TEST = "test"
    LINT = "lint"


class ProjectLearning(BaseModel):
    """Project-specific learning record."""
    id: str
    project_path: str
    category: LearningCategory
    trigger_context: str | None = None
    learning_content: str
    source_session_id: str | None = None
    success_count: int = 1
    failure_count: int = 0
    last_used: str | None = None
    promoted_to_universal: bool = False
    created_at: str


class LearningResult(BaseModel):
    """Result of a learning operation."""
    id: str
    status: str
    message: str = ""
    learning: ProjectLearning | None = None


class LearningsQueryResult(BaseModel):
    """Result of querying learnings."""
    project_path: str
    category: str | None = None
    total_count: int
    learnings: list[ProjectLearning] = Field(default_factory=list)


class ErrorSolution(BaseModel):
    """Error→Solution mapping."""
    id: str
    error_pattern: str
    error_hash: str | None = None
    error_category: ErrorCategory | None = None
    solution_steps: list[str] = Field(default_factory=list)
    context_requirements: dict[str, Any] | None = None
    success_rate: float = 1.0
    usage_count: int = 1
    project_path: str | None = None  # None = universal
    source_session_id: str | None = None
    created_at: str
    last_used: str | None = None


class SolutionResult(BaseModel):
    """Result of a solution operation."""
    id: str
    status: str
    error_hash: str | None = None
    message: str = ""
    solution: ErrorSolution | None = None


class SolutionSearchResult(BaseModel):
    """Result of searching for solutions."""
    error_text: str
    total_found: int
    solutions: list[ErrorSolution] = Field(default_factory=list)
    project_specific_count: int = 0
    universal_count: int = 0
