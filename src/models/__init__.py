"""
Session Intelligence Models - Data structures for session management and analytics.

This module contains all the Pydantic models used by the Session Intelligence MCP server
to represent sessions, executions, decisions, patterns, and analytics data.
"""

from .session_models import (
    # Core Session Models
    Session,
    SessionMetadata,
    SessionResult,
    ExecutionStep,
    
    # Agent and Workflow Models  
    AgentExecution,
    AgentContext,
    AgentPerformance,
    AgentError,
    WorkflowState,
    WorkflowResult,
    StateMachine,
    ParallelExecution,
    
    # Decision Models
    Decision,
    DecisionContext,
    DecisionOutcome,
    DecisionResult,
    
    # Analytics and Intelligence Models
    PatternAnalysis,
    SessionIntelligence,
    CommandAnalysis,
    Bottleneck,
    Optimization,
    PredictedIssue,
    LearningInsight,
    Trend,
    Recommendation,
    CommandAlternative,
    
    # Health and Monitoring Models
    HealthStatus,
    PerformanceMetrics,
    SessionHealthResult,
    
    # Dashboard and Reporting Models
    DashboardResult,
    
    # Function Result Models
    ExecutionTrackingResult,
    CoordinationResult,
    PatternAnalysisResult,
    MissingFunctionResult,
    CommandAnalysisResult,
    
    # Enums
    SessionStatus,
    ExecutionStatus,
    ImpactLevel,
    PatternType,
    WorkflowType,
    DashboardType,
    AnalysisScope,
    ExecutionMode,
    OptimizationLevel,
)

__all__ = [
    # Core Session Models
    "Session",
    "SessionMetadata", 
    "SessionResult",
    "ExecutionStep",
    
    # Agent and Workflow Models
    "AgentExecution",
    "AgentContext",
    "AgentPerformance", 
    "AgentError",
    "WorkflowState",
    "WorkflowResult",
    "StateMachine",
    "ParallelExecution",
    
    # Decision Models
    "Decision",
    "DecisionContext",
    "DecisionOutcome",
    "DecisionResult",
    
    # Analytics and Intelligence Models
    "PatternAnalysis",
    "SessionIntelligence",
    "CommandAnalysis", 
    "Bottleneck",
    "Optimization",
    "PredictedIssue",
    "LearningInsight", 
    "Trend",
    "Recommendation",
    "CommandAlternative",
    
    # Health and Monitoring Models
    "HealthStatus",
    "PerformanceMetrics",
    "SessionHealthResult",
    
    # Dashboard and Reporting Models
    "DashboardResult",
    
    # Function Result Models
    "ExecutionTrackingResult",
    "CoordinationResult",
    "PatternAnalysisResult",
    "MissingFunctionResult", 
    "CommandAnalysisResult",
    
    # Enums
    "SessionStatus",
    "ExecutionStatus",
    "ImpactLevel", 
    "PatternType",
    "WorkflowType",
    "DashboardType",
    "AnalysisScope",
    "ExecutionMode",
    "OptimizationLevel",
]