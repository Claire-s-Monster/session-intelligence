"""
Tests for SessionIntelligenceEngine decision and learning methods.

Covers:
- session_log_decision() — basic logging, auto-session creation, optional fields
- session_log_learning() — basic logging, category handling
- query_decisions_by_session / query_decisions_by_category — via database backend
- query_project_learnings — via database backend
- update_learning_usage — usage tracking
- session_find_solution — error solution search

asyncio_mode = "auto" (from pyproject.toml) — no @pytest.mark.asyncio needed.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import DecisionResult, LearningResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_session(engine, project_name="test-project"):
    """Create a session and return the SessionResult."""
    return await engine.session_manage_lifecycle(
        operation="create",
        project_name=project_name,
        mode="local",
        metadata={},
    )


# ===========================================================================
# session_log_decision
# ===========================================================================


async def test_log_decision_returns_decision_result(engine):
    """session_log_decision returns a DecisionResult instance."""
    result = await engine.session_log_decision(decision="Use SQLite for testing")

    assert isinstance(result, DecisionResult)


async def test_log_decision_returns_decision_id(engine):
    """Returned DecisionResult has a non-empty decision_id."""
    result = await engine.session_log_decision(decision="Choose pytest over unittest")

    assert result.decision_id
    assert result.decision_id != "error"
    assert result.decision_id.startswith("decision-")


async def test_log_decision_with_active_session(engine):
    """Logging a decision when a session exists attaches it to that session."""
    session_result = await _create_session(engine)
    session_id = session_result.session_id
    engine._current_session_id = session_id

    result = await engine.session_log_decision(
        decision="Adopt hexagonal architecture",
        session_id=session_id,
    )

    assert result.decision_id != "error"
    assert result.session_id == session_id


async def test_log_decision_without_session_auto_creates(engine):
    """session_log_decision succeeds even without a pre-existing session."""
    # Engine starts with no session — should auto-create one
    result = await engine.session_log_decision(
        decision="Auto-create session on first decision"
    )

    assert isinstance(result, DecisionResult)
    assert result.decision_id != "error"


async def test_log_decision_persists_to_database(engine):
    """Decision logged is retrievable from the database backend."""
    session_result = await _create_session(engine)
    session_id = session_result.session_id
    engine._current_session_id = session_id

    dec_result = await engine.session_log_decision(
        decision="Database persistence test decision",
        session_id=session_id,
    )

    rows = await engine.database.query_decisions_by_session(session_id)
    ids = [r["id"] for r in rows]
    assert dec_result.decision_id in ids


async def test_query_decisions_by_session(engine):
    """query_decisions_by_session returns decisions scoped to that session."""
    s1 = await _create_session(engine, project_name="project-alpha")
    s2 = await _create_session(engine, project_name="project-beta")

    # Ensure sessions have distinct IDs (may collide if timestamp-based and fast)
    if s1.session_id == s2.session_id:
        pytest.skip("Sessions got same timestamp-based ID — cannot test isolation")

    engine._current_session_id = s1.session_id
    d1 = await engine.session_log_decision(decision="Alpha decision", session_id=s1.session_id)

    engine._current_session_id = s2.session_id
    d2 = await engine.session_log_decision(decision="Beta decision", session_id=s2.session_id)

    alpha_rows = await engine.database.query_decisions_by_session(s1.session_id)
    beta_rows = await engine.database.query_decisions_by_session(s2.session_id)

    alpha_ids = {r["id"] for r in alpha_rows}
    beta_ids = {r["id"] for r in beta_rows}

    # Each decision appears in its own session's results
    assert d1.decision_id in alpha_ids
    assert d2.decision_id in beta_ids
    # The decisions are not in each other's session
    assert d1.decision_id not in beta_ids
    assert d2.decision_id not in alpha_ids


async def test_query_decisions_by_category(engine):
    """Decisions saved with a category can be queried by that category."""
    session_result = await _create_session(engine)
    session_id = session_result.session_id
    engine._current_session_id = session_id

    # Save decision data directly to db with category set
    await engine.database.save_decision({
        "id": "decision-cat-test-01",
        "session_id": session_id,
        "description": "Architecture decision with category",
        "category": "architecture",
        "impact_level": "high",
        "context": "{}",
        "artifacts": "[]",
    })

    rows = await engine.database.query_decisions_by_category("architecture")
    assert any(r["id"] == "decision-cat-test-01" for r in rows)


async def test_decision_with_all_optional_fields(engine):
    """session_log_decision accepts all optional parameters without error."""
    session_result = await _create_session(engine)
    session_id = session_result.session_id
    engine._current_session_id = session_id

    result = await engine.session_log_decision(
        decision="Full-featured decision",
        session_id=session_id,
        context={"rationale": "performance", "risk": "low"},
        impact_analysis=True,
        link_artifacts=["src/core/engine.py", "tests/conftest.py"],
    )

    assert isinstance(result, DecisionResult)
    assert result.decision_id != "error"
    # impact_analysis requested → result should contain analysis data
    assert isinstance(result.impact_analysis, dict)


# ===========================================================================
# session_log_learning
# ===========================================================================


async def test_log_learning_returns_learning_result(engine):
    """session_log_learning returns a LearningResult instance."""
    result = await engine.session_log_learning(
        category="pattern",
        learning_content="Use fixtures for database isolation",
    )

    assert isinstance(result, LearningResult)


async def test_log_learning_returns_learning_id(engine):
    """Returned LearningResult has a non-empty id starting with 'learn_'."""
    result = await engine.session_log_learning(
        category="error_fix",
        learning_content="Always await async DB calls",
    )

    assert result.id
    assert result.id.startswith("learn_")


async def test_log_learning_with_active_session(engine):
    """Learning logged when a session is active uses that session as source."""
    session_result = await _create_session(engine)
    session_id = session_result.session_id
    engine._current_session_id = session_id

    result = await engine.session_log_learning(
        category="workflow",
        learning_content="Set _current_session_id before logging learnings",
        trigger_context="When engine tests create sessions",
    )

    assert result.id.startswith("learn_")
    # The learning object records the source session
    assert result.learning is not None
    assert result.learning.source_session_id == session_id


async def test_log_learning_status_saved(engine):
    """When database is available, learning status is 'saved'."""
    result = await engine.session_log_learning(
        category="preference",
        learning_content="Prefer compact assertions in tests",
    )

    assert result.status == "saved"


async def test_query_learnings_by_category(engine):
    """Learnings saved under a category are retrievable filtered by that category."""
    await engine.session_log_learning(
        category="error_fix",
        learning_content="Fix: use UTC timestamps in SQLite",
        trigger_context="timestamp mismatch errors",
    )
    await engine.session_log_learning(
        category="pattern",
        learning_content="Pattern: always isolate DB in tests",
    )

    project_path = str(engine.claude_sessions_path.parent)
    error_fix_rows = await engine.database.query_project_learnings(
        project_path=project_path, category="error_fix"
    )
    pattern_rows = await engine.database.query_project_learnings(
        project_path=project_path, category="pattern"
    )

    assert any("UTC timestamps" in r["learning_content"] for r in error_fix_rows)
    assert any("isolate DB" in r["learning_content"] for r in pattern_rows)
    # No cross-contamination
    assert not any("UTC timestamps" in r["learning_content"] for r in pattern_rows)


async def test_update_learning_usage(engine):
    """update_learning_usage increments success/failure counts."""
    learn_result = await engine.session_log_learning(
        category="workflow",
        learning_content="Usage tracking test learning",
    )
    learning_id = learn_result.id

    # Record a successful application
    update = await engine.database.update_learning_usage(learning_id, success=True)
    assert update["updated"] is True

    project_path = str(engine.claude_sessions_path.parent)
    rows = await engine.database.query_project_learnings(project_path=project_path)
    matching = [r for r in rows if r["id"] == learning_id]
    assert matching
    # success_count was 1 on insert, now incremented to 2
    assert matching[0]["success_count"] >= 2


# ===========================================================================
# session_find_solution — save + find error solutions
# ===========================================================================


async def test_save_and_find_error_solutions(engine):
    """Error solutions saved to the DB are returned by session_find_solution."""
    import uuid

    solution_id = f"sol-{uuid.uuid4().hex[:8]}"
    error_pattern = "ModuleNotFoundError: No module named 'core'"

    await engine.database.save_error_solution(
        solution_id=solution_id,
        error_pattern=error_pattern,
        solution_steps=["Add src/ to PYTHONPATH", "Run: PYTHONPATH=src pytest"],
        error_category="dependency",
    )

    result = await engine.session_find_solution(
        error_text=error_pattern,
        error_category="dependency",
    )

    assert result.total_found >= 1
    solution_ids = [s.id for s in result.solutions]
    assert solution_id in solution_ids
