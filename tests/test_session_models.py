"""
Tests for session intelligence data models.

Tests all Pydantic models for proper instantiation, validation, and serialization.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from models.session_models import (
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
    # Core models
    SessionMetadata,
    PerformanceMetrics,
    HealthStatus,
    CommandExecution,
    Pattern,
    Optimization,
    ExecutionStep,
    # Agent models
    AgentContext,
    AgentPerformance,
    AgentError,
    AgentExecution,
    # Workflow models
    StateMachine,
    ParallelExecution,
    WorkflowState,
    # Decision models
    DecisionContext,
    DecisionOutcome,
    Decision,
    # Analytics models
    Recommendation,
    PatternAnalysis,
    Bottleneck,
    PredictedIssue,
    LearningInsight,
    Trend,
    SessionIntelligence,
    CommandAlternative,
    CommandAnalysis,
    # Main session model
    Session,
    # Result models
    SessionResult,
    ExecutionTrackingResult,
    CoordinationResult,
    DecisionResult,
    PatternAnalysisResult,
    SessionHealthResult,
    WorkflowResult,
    CommandAnalysisResult,
    MissingFunctionResult,
    DashboardResult,
)


class TestEnums:
    """Test enum definitions and values."""

    def test_session_status_values(self):
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus.COMPLETED == "completed"
        assert SessionStatus.FAILED == "failed"
        assert SessionStatus.RECOVERED == "recovered"

    def test_execution_status_values(self):
        assert ExecutionStatus.PENDING == "pending"
        assert ExecutionStatus.RUNNING == "running"
        assert ExecutionStatus.SUCCESS == "success"
        assert ExecutionStatus.ERROR == "error"
        assert ExecutionStatus.SKIPPED == "skipped"

    def test_impact_level_values(self):
        assert ImpactLevel.LOW == "low"
        assert ImpactLevel.MEDIUM == "medium"
        assert ImpactLevel.HIGH == "high"
        assert ImpactLevel.CRITICAL == "critical"

    def test_pattern_type_values(self):
        assert PatternType.EXECUTION == "execution"
        assert PatternType.ERROR == "error"
        assert PatternType.PERFORMANCE == "performance"
        assert PatternType.WORKFLOW == "workflow"

    def test_workflow_type_values(self):
        assert WorkflowType.TDD == "tdd"
        assert WorkflowType.ATOMIC == "atomic"
        assert WorkflowType.QUALITY == "quality"
        assert WorkflowType.PRIME == "prime"
        assert WorkflowType.CUSTOM == "custom"

    def test_execution_mode_values(self):
        assert ExecutionMode.SEQUENTIAL == "sequential"
        assert ExecutionMode.PARALLEL == "parallel"
        assert ExecutionMode.ADAPTIVE == "adaptive"


class TestCoreModels:
    """Test core session models."""

    def test_session_metadata_creation(self):
        metadata = SessionMetadata(
            session_type="development",
            environment="local",
            user="test-user",
            git_branch="main",
        )
        assert metadata.session_type == "development"
        assert metadata.environment == "local"
        assert metadata.user == "test-user"
        assert metadata.git_branch == "main"
        assert metadata.tags == []
        assert metadata.custom_attributes == {}

    def test_session_metadata_with_optional_fields(self):
        metadata = SessionMetadata(
            session_type="testing",
            environment="ci",
            user="ci-runner",
            git_commit="abc123",
            parent_session="parent-session-id",
            tags=["automated", "nightly"],
            custom_attributes={"priority": "high"},
        )
        assert metadata.git_commit == "abc123"
        assert metadata.parent_session == "parent-session-id"
        assert "automated" in metadata.tags
        assert metadata.custom_attributes["priority"] == "high"

    def test_performance_metrics_defaults(self):
        metrics = PerformanceMetrics()
        assert metrics.total_execution_time_ms == 0
        assert metrics.agents_executed == 0
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 0
        assert metrics.average_execution_time_ms == 0.0
        assert metrics.efficiency_score == 0.0

    def test_performance_metrics_with_values(self):
        metrics = PerformanceMetrics(
            total_execution_time_ms=5000,
            agents_executed=3,
            successful_executions=2,
            failed_executions=1,
            efficiency_score=75.5,
        )
        assert metrics.total_execution_time_ms == 5000
        assert metrics.agents_executed == 3
        assert metrics.efficiency_score == 75.5

    def test_health_status_defaults(self):
        health = HealthStatus()
        assert health.overall_score == 100.0
        assert health.continuity_valid is True
        assert health.files_valid is True
        assert health.state_consistent is True
        assert health.agents_healthy is True
        assert health.issues == []
        assert health.warnings == []

    def test_health_status_with_issues(self):
        health = HealthStatus(
            overall_score=50.0,
            continuity_valid=False,
            issues=["Session directory missing", "Metadata corrupted"],
            warnings=["Performance degradation detected"],
        )
        assert health.overall_score == 50.0
        assert health.continuity_valid is False
        assert len(health.issues) == 2
        assert len(health.warnings) == 1

    def test_pattern_creation(self):
        pattern = Pattern(
            pattern_id="pattern-001",
            pattern_type=PatternType.ERROR,
            description="Repeated timeout errors",
            frequency=5,
            confidence=0.85,
            impact="negative",
        )
        assert pattern.pattern_id == "pattern-001"
        assert pattern.pattern_type == PatternType.ERROR
        assert pattern.frequency == 5
        assert pattern.confidence == 0.85

    def test_optimization_creation(self):
        optimization = Optimization(
            optimization_id="opt-001",
            description="Batch similar operations",
            potential_impact="30% reduction in execution time",
            effort_level="low",
            confidence=0.9,
            implementation_hints=["Group file operations", "Use parallel processing"],
        )
        assert optimization.optimization_id == "opt-001"
        assert optimization.effort_level == "low"
        assert len(optimization.implementation_hints) == 2


class TestExecutionModels:
    """Test execution tracking models."""

    def test_command_execution_creation(self):
        cmd = CommandExecution(
            command="pytest tests/",
            started=datetime.now(),
            duration_ms=1500,
            exit_code=0,
            stdout="All tests passed",
            success=True,
        )
        assert cmd.command == "pytest tests/"
        assert cmd.duration_ms == 1500
        assert cmd.success is True

    def test_execution_step_creation(self):
        step = ExecutionStep(
            step_id="step-001",
            step_number=1,
            agent="test-runner",
            operation="run_tests",
            description="Execute test suite",
            tools_used=["pytest", "coverage"],
            started=datetime.now(),
            status=ExecutionStatus.RUNNING,
        )
        assert step.step_id == "step-001"
        assert step.step_number == 1
        assert step.agent == "test-runner"
        assert step.status == ExecutionStatus.RUNNING
        assert "pytest" in step.tools_used

    def test_execution_step_with_patterns(self):
        pattern = Pattern(
            pattern_id="p1",
            pattern_type=PatternType.PERFORMANCE,
            description="Slow test",
            frequency=1,
            confidence=0.8,
        )
        step = ExecutionStep(
            step_id="step-002",
            step_number=2,
            agent="analyzer",
            operation="analyze",
            description="Analyze results",
            started=datetime.now(),
            patterns_detected=[pattern],
        )
        assert len(step.patterns_detected) == 1
        assert step.patterns_detected[0].pattern_type == PatternType.PERFORMANCE


class TestAgentModels:
    """Test agent-related models."""

    def test_agent_context_creation(self):
        context = AgentContext(
            session_id="session-001",
            project_path="/home/user/project",
            working_directory="/home/user/project/src",
            environment_variables={"DEBUG": "true"},
            dependencies=["pytest", "ruff"],
        )
        assert context.session_id == "session-001"
        assert context.project_path == "/home/user/project"
        assert context.environment_variables["DEBUG"] == "true"

    def test_agent_performance_defaults(self):
        perf = AgentPerformance()
        assert perf.total_execution_time_ms == 0
        assert perf.successful_steps == 0
        assert perf.failed_steps == 0
        assert perf.efficiency_score == 0.0

    def test_agent_error_creation(self):
        error = AgentError(
            error_id="err-001",
            timestamp=datetime.now(),
            error_type="TimeoutError",
            error_message="Operation timed out after 30s",
            step_id="step-003",
            recovery_attempted=True,
            recovery_successful=False,
        )
        assert error.error_type == "TimeoutError"
        assert error.recovery_attempted is True
        assert error.recovery_successful is False

    def test_agent_execution_creation(self):
        context = AgentContext(
            session_id="session-001",
            project_path="/project",
            working_directory="/project",
        )
        execution = AgentExecution(
            agent_name="quality-resolver",
            agent_type="focused",
            execution_id="exec-001",
            started=datetime.now(),
            status=ExecutionStatus.RUNNING,
            context=context,
        )
        assert execution.agent_name == "quality-resolver"
        assert execution.agent_type == "focused"
        assert execution.status == ExecutionStatus.RUNNING
        assert execution.execution_steps == []


class TestWorkflowModels:
    """Test workflow-related models."""

    def test_state_machine_creation(self):
        sm = StateMachine(
            current_state="testing",
            available_transitions=["deploy", "rollback"],
            state_data={"test_passed": True},
        )
        assert sm.current_state == "testing"
        assert "deploy" in sm.available_transitions
        assert sm.state_data["test_passed"] is True

    def test_workflow_state_creation(self):
        sm = StateMachine(current_state="init", available_transitions=["start"])
        workflow = WorkflowState(
            workflow_type=WorkflowType.TDD,
            current_phase="red",
            phases_completed=["setup"],
            state_machine=sm,
            next_steps=["Write failing test"],
        )
        assert workflow.workflow_type == WorkflowType.TDD
        assert workflow.current_phase == "red"
        assert "setup" in workflow.phases_completed


class TestDecisionModels:
    """Test decision-related models."""

    def test_decision_context_creation(self):
        ctx = DecisionContext(
            session_id="session-001",
            agent_name="architect",
            workflow_phase="planning",
            available_options=["option-a", "option-b"],
            constraints=["time-limit", "budget"],
        )
        assert ctx.session_id == "session-001"
        assert ctx.agent_name == "architect"
        assert len(ctx.available_options) == 2

    def test_decision_creation(self):
        ctx = DecisionContext(
            session_id="session-001",
            project_state={"files_changed": 5},
        )
        decision = Decision(
            decision_id="dec-001",
            timestamp=datetime.now(),
            description="Use pytest instead of unittest",
            rationale="Better async support and fixtures",
            context=ctx,
            impact_level=ImpactLevel.MEDIUM,
            artifacts=["pyproject.toml", "tests/conftest.py"],
        )
        assert decision.decision_id == "dec-001"
        assert decision.impact_level == ImpactLevel.MEDIUM
        assert len(decision.artifacts) == 2


class TestAnalyticsModels:
    """Test analytics and intelligence models."""

    def test_recommendation_creation(self):
        rec = Recommendation(
            recommendation_id="rec-001",
            description="Enable parallel test execution",
            rationale="Tests are independent and can run concurrently",
            impact_estimate="40% faster test runs",
            effort_estimate="low",
            confidence=0.85,
            implementation_steps=["Add pytest-xdist", "Configure workers"],
        )
        assert rec.recommendation_id == "rec-001"
        assert rec.confidence == 0.85
        assert len(rec.implementation_steps) == 2

    def test_pattern_analysis_creation(self):
        analysis = PatternAnalysis(
            pattern_id="pa-001",
            pattern_type=PatternType.EXECUTION,
            frequency=10,
            sessions_affected=["s1", "s2", "s3"],
            agents_involved=["test-runner", "quality-resolver"],
            description="Repeated test failure pattern",
            impact="negative",
            learning_confidence=0.75,
        )
        assert analysis.frequency == 10
        assert len(analysis.sessions_affected) == 3
        assert analysis.learning_confidence == 0.75

    def test_bottleneck_creation(self):
        bottleneck = Bottleneck(
            bottleneck_id="bn-001",
            description="Slow database queries",
            location="data-access-layer",
            impact_score=7.5,
            frequency=15,
            suggested_fixes=["Add indexes", "Use connection pooling"],
        )
        assert bottleneck.impact_score == 7.5
        assert len(bottleneck.suggested_fixes) == 2

    def test_session_intelligence_creation(self):
        intelligence = SessionIntelligence(
            session_id="session-001",
            efficiency_score=85.0,
            patterns_detected=[],
            bottlenecks=[],
            optimization_opportunities=[],
        )
        assert intelligence.session_id == "session-001"
        assert intelligence.efficiency_score == 85.0


class TestSessionModel:
    """Test the main Session model."""

    def test_session_creation(self):
        metadata = SessionMetadata(
            session_type="development",
            environment="local",
            user="developer",
        )
        session = Session(
            id="session-001",
            started=datetime.now(),
            mode="local",
            project_name="test-project",
            project_path="/home/user/test-project",
            metadata=metadata,
        )
        assert session.id == "session-001"
        assert session.status == SessionStatus.ACTIVE
        assert session.project_name == "test-project"
        assert session.agents_executed == []
        assert session.decisions == []

    def test_session_with_all_fields(self):
        metadata = SessionMetadata(
            session_type="integration",
            environment="ci",
            user="ci-bot",
            git_branch="feature/test",
        )
        health = HealthStatus(overall_score=95.0)
        perf = PerformanceMetrics(agents_executed=5, successful_executions=4)

        session = Session(
            id="session-002",
            started=datetime.now(),
            completed=datetime.now(),
            mode="hybrid",
            project_name="integration-project",
            project_path="/ci/workspace",
            status=SessionStatus.COMPLETED,
            metadata=metadata,
            health_status=health,
            performance_metrics=perf,
        )
        assert session.status == SessionStatus.COMPLETED
        assert session.health_status.overall_score == 95.0
        assert session.performance_metrics.agents_executed == 5


class TestResultModels:
    """Test result models for MCP functions."""

    def test_session_result_creation(self):
        result = SessionResult(
            session_id="session-001",
            operation="create",
            status="success",
            message="Session created successfully",
            next_steps=["Initialize tracking", "Set up workflow"],
        )
        assert result.session_id == "session-001"
        assert result.status == "success"
        assert len(result.next_steps) == 2

    def test_execution_tracking_result_creation(self):
        pattern = Pattern(
            pattern_id="p1",
            pattern_type=PatternType.EXECUTION,
            description="Normal execution",
            frequency=1,
            confidence=0.9,
        )
        result = ExecutionTrackingResult(
            step_id="step-001",
            session_id="session-001",
            agent_name="test-runner",
            status="success",
            patterns_detected=[pattern],
            optimizations=[],
        )
        assert result.step_id == "step-001"
        assert len(result.patterns_detected) == 1

    def test_coordination_result_creation(self):
        result = CoordinationResult(
            coordination_id="coord-001",
            session_id="session-001",
            execution_plan={"mode": "parallel", "agents": ["a1", "a2"]},
            timing_estimate=5000,
            parallel_execution_groups=[["a1", "a2"]],
        )
        assert result.coordination_id == "coord-001"
        assert result.timing_estimate == 5000

    def test_session_health_result_creation(self):
        result = SessionHealthResult(
            session_id="session-001",
            health_score=85.0,
            issues=["Minor file sync issue"],
            recovery_actions=["Resync files"],
            diagnostics={"session_age_minutes": 45},
        )
        assert result.health_score == 85.0
        assert len(result.issues) == 1
        assert result.diagnostics["session_age_minutes"] == 45

    def test_dashboard_result_creation(self):
        result = DashboardResult(
            dashboard_type=DashboardType.PERFORMANCE,
            session_id="session-001",
            metrics={"avg_execution_time": 1500, "success_rate": 0.95},
            insights=["Performance improved 15% from last session"],
            real_time_data=True,
        )
        assert result.dashboard_type == DashboardType.PERFORMANCE
        assert result.real_time_data is True
        assert len(result.insights) == 1


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_session_to_dict(self):
        metadata = SessionMetadata(
            session_type="test",
            environment="local",
            user="tester",
        )
        session = Session(
            id="session-001",
            started=datetime.now(),
            mode="local",
            project_name="test",
            project_path="/test",
            metadata=metadata,
        )
        data = session.model_dump()
        assert isinstance(data, dict)
        assert data["id"] == "session-001"
        assert data["project_name"] == "test"
        assert "metadata" in data

    def test_session_from_dict(self):
        data = {
            "id": "session-002",
            "started": datetime.now().isoformat(),
            "mode": "local",
            "project_name": "restored",
            "project_path": "/restored",
            "status": "active",
            "metadata": {
                "session_type": "restored",
                "environment": "local",
                "user": "restorer",
            },
        }
        session = Session.model_validate(data)
        assert session.id == "session-002"
        assert session.project_name == "restored"

    def test_nested_model_serialization(self):
        ctx = AgentContext(
            session_id="s1",
            project_path="/p",
            working_directory="/p/src",
        )
        execution = AgentExecution(
            agent_name="agent-1",
            agent_type="focused",
            execution_id="e1",
            started=datetime.now(),
            context=ctx,
        )
        data = execution.model_dump()
        assert isinstance(data["context"], dict)
        assert data["context"]["session_id"] == "s1"
