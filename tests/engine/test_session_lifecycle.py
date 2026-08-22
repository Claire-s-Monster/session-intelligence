"""
Tests for SessionIntelligenceEngine session lifecycle methods.

Covers:
- session_manage_lifecycle() — create, finalize, invalid operations
- session_monitor_health() — health scoring
- _get_or_create_current_session_id() — auto-creation
- SessionResult Pydantic model attribute access

asyncio_mode = "auto" (from pyproject.toml) — no @pytest.mark.asyncio needed.
"""

import inspect

import pytest

from core.session_engine import SessionContextRequiredError, SessionIntelligenceEngine
from models.session_models import SessionHealthResult, SessionResult, SessionStatus


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _create_session(engine, project_name="test-project", mode="local", metadata=None):
    """Convenience: create a session and return the SessionResult."""
    return await engine.session_manage_lifecycle(
        operation="create",
        project_name=project_name,
        mode=mode,
        metadata=metadata or {},
    )


# ===========================================================================
# session_manage_lifecycle — create
# ===========================================================================


async def test_create_session_returns_valid_id(engine):
    """create operation returns a SessionResult with a non-empty session_id."""
    result = await _create_session(engine)

    assert isinstance(result, SessionResult)
    assert result.session_id
    assert result.session_id != "error"


async def test_create_session_sets_active_status(engine):
    """Session created via lifecycle management has status='success'."""
    result = await _create_session(engine)

    assert result.status == "success"


async def test_create_session_operation_field(engine):
    """Result.operation is 'create' after a create call."""
    result = await _create_session(engine)

    assert result.operation == "create"


async def test_create_session_stores_metadata(engine):
    """project_name and mode are preserved on the session_data object."""
    result = await _create_session(engine, project_name="my-project", mode="remote")

    assert result.session_data is not None
    assert result.session_data.project_name == "my-project"
    assert result.session_data.mode == "remote"


async def test_create_session_session_data_is_active(engine):
    """Newly created session has SessionStatus.ACTIVE."""
    result = await _create_session(engine)

    assert result.session_data is not None
    assert result.session_data.status == SessionStatus.ACTIVE


async def test_create_session_with_custom_metadata(engine):
    """Custom tags/attributes in metadata dict round-trip through the engine."""
    custom_meta = {
        "tags": ["ci", "nightly"],
        "git_branch": "feature/x",
        "user": "bot",
    }
    result = await _create_session(engine, metadata=custom_meta)

    assert result.status == "success"
    assert result.session_data is not None
    # Tags should be stored on session_metadata
    assert result.session_data.metadata.tags == ["ci", "nightly"]
    assert result.session_data.metadata.git_branch == "feature/x"
    assert result.session_data.metadata.user == "bot"


async def test_multiple_sessions_same_project(engine):
    """Creating two sessions for the same project yields two SessionResult objects,
    both with status='success'. IDs may collide if created within the same second
    (timestamp-based), so we verify both are stored in the cache."""
    result1 = await _create_session(engine, project_name="shared-project")
    result2 = await _create_session(engine, project_name="shared-project")

    assert result1.status == "success"
    assert result2.status == "success"
    # Both session IDs appear in the engine's in-memory cache
    assert result1.session_id in engine.session_cache
    assert result2.session_id in engine.session_cache


# ===========================================================================
# session_manage_lifecycle — finalize
# ===========================================================================


async def test_finalize_session(engine):
    """After creating a session, finalize changes status to 'success' / COMPLETED."""
    create_result = await _create_session(engine)
    session_id = create_result.session_id

    # Pass explicit scope (issue #77): finalize no longer resolves an
    # unscoped call via ambient `_current_session_id`/session_cache state.
    finalize_result = await engine.session_manage_lifecycle(
        operation="finalize", session_id=session_id
    )

    assert finalize_result.status == "success"
    assert finalize_result.operation == "finalize"
    assert finalize_result.session_data is not None
    assert finalize_result.session_data.status == SessionStatus.COMPLETED


async def test_finalize_nonexistent_session_requires_scope(engine):
    """Issue #77: finalizing with no session_id/session_name/project_name and
    no allow_unbound now raises SessionContextRequiredError instead of
    silently falling back to ambient `_get_or_create_current_session_id()`
    state -- on a shared engine that state can belong to a different
    project entirely."""
    with pytest.raises(SessionContextRequiredError):
        await engine.session_manage_lifecycle(operation="finalize")


async def test_finalize_nonexistent_session_allow_unbound_still_completes(engine):
    """The allow_unbound=True escape hatch preserves the pre-#77 ambient
    behavior: it auto-creates/uses the ambient session and finalizes it
    without raising."""
    result = await engine.session_manage_lifecycle(
        operation="finalize", allow_unbound=True
    )

    assert isinstance(result, SessionResult)
    assert result.operation == "finalize"
    assert result.status in {"success", "error"}


# ===========================================================================
# session_manage_lifecycle — invalid operation
# ===========================================================================


async def test_manage_lifecycle_invalid_operation(engine):
    """An unknown operation name returns status='error' gracefully."""
    result = await engine.session_manage_lifecycle(operation="nonexistent_op")

    assert result.status == "error"
    assert result.session_id == "error"


# ===========================================================================
# session_monitor_health
# ===========================================================================


async def test_session_health_check(engine):
    """After creating a session, health monitoring returns a health_score."""
    create_result = await _create_session(engine)
    session_id = create_result.session_id

    health = await engine.session_monitor_health(session_id=session_id)

    assert isinstance(health, SessionHealthResult)
    assert isinstance(health.health_score, float)
    assert 0.0 <= health.health_score <= 100.0


async def test_session_health_no_session_returns_zero(engine):
    """Health monitoring with no active session returns health_score=0.0."""
    health = await engine.session_monitor_health(session_id="nonexistent-id")

    assert health.health_score == 0.0
    assert len(health.issues) > 0


async def test_session_health_includes_diagnostics(engine):
    """With include_diagnostics=True (default), diagnostics dict is populated."""
    create_result = await _create_session(engine)
    session_id = create_result.session_id

    health = await engine.session_monitor_health(
        session_id=session_id, include_diagnostics=True
    )

    assert isinstance(health.diagnostics, dict)
    # At minimum the age should be present
    assert "session_age_minutes" in health.diagnostics


async def test_session_health_custom_checks(engine):
    """Passing a subset of health_checks restricts what is evaluated."""
    create_result = await _create_session(engine)
    session_id = create_result.session_id

    health = await engine.session_monitor_health(
        session_id=session_id,
        health_checks=["state"],  # only state check
    )

    assert isinstance(health, SessionHealthResult)
    # No filesystem checks → no file-related issues expected
    file_issues = [i for i in health.issues if "file" in i.lower()]
    assert file_issues == []


# ===========================================================================
# _get_or_create_current_session_id
# ===========================================================================


async def test_get_or_create_current_session_id_when_no_session(engine):
    """Without any existing session, _get_or_create_current_session_id auto-creates
    one and returns its ID (never returns None — it's 'get or CREATE')."""
    # Engine starts with empty cache and filesystem disabled
    result = engine._get_or_create_current_session_id()

    # The method auto-creates a session and returns its ID
    assert result is not None
    assert isinstance(result, str)
    assert result in engine.session_cache


async def test_get_or_create_returns_cached_id(engine):
    """After creating a session and setting _current_session_id, helper returns it."""
    create_result = await _create_session(engine)
    session_id = create_result.session_id

    engine._current_session_id = session_id
    result = engine._get_or_create_current_session_id()

    assert result == session_id


# ===========================================================================
# Async / coroutine contract
# ===========================================================================


def test_session_manage_lifecycle_is_async():
    """session_manage_lifecycle must be a coroutine function."""
    assert inspect.iscoroutinefunction(
        SessionIntelligenceEngine.session_manage_lifecycle
    )
