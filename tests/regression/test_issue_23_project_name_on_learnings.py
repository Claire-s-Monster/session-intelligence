"""
Regression tests for issue #23: session_log_learning ignores explicit project_name.

Verifies the two-layer fix:
  Layer A — _resolve_session_context returns ResolvedSessionContext with
            project_name / project_path from the resolved session row.
  Layer B — project_learnings.project_name column is persisted and used by
            recall_project for direct filtering (with path-bridge fallback for
            legacy rows where project_name IS NULL).
"""

import uuid
from datetime import UTC, datetime

import pytest

from core.session_engine import ResolvedSessionContext, SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db():
    """In-memory SQLite database, initialized and cleaned up per test."""
    backend = SQLiteBackend(db_path=":memory:")
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
def engine(db, monkeypatch: pytest.MonkeyPatch) -> SessionIntelligenceEngine:
    """Engine wired to in-memory SQLite, no filesystem."""
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENTS_DIR", "/tmp/nonexistent-agents")
    return SessionIntelligenceEngine(
        repository_path=None,
        use_filesystem=False,
        database=db,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _create_and_persist(engine, db, project_name, session_name=None):
    """Create a session in the engine cache and persist it to the DB."""
    result = engine._create_session(
        mode="local",
        project_name=project_name,
        metadata={},
        session_name=session_name,
    )
    assert result.status == "success"
    await db.save_session(result.session_data.model_dump(mode="python"))
    return result.session_id


# ---------------------------------------------------------------------------
# Test 1: explicit project_name round-trips through log → recall
# ---------------------------------------------------------------------------


async def test_log_learning_with_explicit_project_name_recalls_correctly(engine, db):
    """
    session_log_learning(project_name="proj-A", ...) must make the learning
    recallable via session_recall(project_name="proj-A").

    The engine's repository_path is None (cwd-derived path would be wrong).
    The only way the learning shows up in recall is if project_name is persisted
    on the row and recall_project uses it directly.
    """
    learn_result = await engine.session_log_learning(
        category="pattern",
        learning_content="use dataclasses for frozen value objects",
        trigger_context="refactoring session context",
        project_name="proj-A",
    )
    assert learn_result.status == "saved", f"save failed: {learn_result.message}"

    recall = await engine.session_recall(project_name="proj-A")
    learnings = recall.get("learnings", [])
    contents = [lr.get("learning_content", "") for lr in learnings]
    assert any(
        "dataclasses" in c for c in contents
    ), f"learning not found in recall; learnings={learnings}"


# ---------------------------------------------------------------------------
# Test 2: session_id resolution propagates session's project_name
# ---------------------------------------------------------------------------


async def test_log_learning_with_session_id_uses_session_project_name(engine, db):
    """
    session_log_learning(session_id=X) with no explicit project_name must
    persist the session's own project_name on the learning row so that
    session_recall(project_name="proj-B") finds it.
    """
    sid = await _create_and_persist(engine, db, "proj-B")

    learn_result = await engine.session_log_learning(
        category="workflow",
        learning_content="always run tests before commit",
        session_id=sid,
    )
    assert learn_result.status == "saved"

    # Verify the persisted row has project_name="proj-B"
    conn = db._ensure_connected()
    cursor = await conn.execute(
        "SELECT project_name FROM project_learnings WHERE id = ?",
        (learn_result.id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["project_name"] == "proj-B"

    # Recall must surface it
    recall = await engine.session_recall(project_name="proj-B")
    learnings = recall.get("learnings", [])
    contents = [lr.get("learning_content", "") for lr in learnings]
    assert any(
        "commit" in c for c in contents
    ), f"learning not found in recall; learnings={learnings}"


# ---------------------------------------------------------------------------
# Test 3: explicit project_name overrides session's project_name
# ---------------------------------------------------------------------------


async def test_log_learning_explicit_project_name_overrides_session_project_name(
    engine, db
):
    """
    When session_id is bound to proj-A but caller passes project_name="proj-C",
    the persisted row's project_name must be "proj-C" (caller wins).
    """
    sid = await _create_and_persist(engine, db, "proj-A")

    learn_result = await engine.session_log_learning(
        category="preference",
        learning_content="prefer ruff over flake8",
        session_id=sid,
        project_name="proj-C",
    )
    assert learn_result.status == "saved"

    conn = db._ensure_connected()
    cursor = await conn.execute(
        "SELECT project_name FROM project_learnings WHERE id = ?",
        (learn_result.id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["project_name"] == "proj-C", (
        f"expected proj-C but got {row['project_name']}"
    )

    # Recall via proj-C must find it
    recall_c = await engine.session_recall(project_name="proj-C")
    contents_c = [lr.get("learning_content", "") for lr in recall_c.get("learnings", [])]
    assert any("ruff" in c for c in contents_c), f"learning not in proj-C recall: {recall_c}"

    # Recall via proj-A must NOT find it (it was overridden)
    recall_a = await engine.session_recall(project_name="proj-A")
    contents_a = [lr.get("learning_content", "") for lr in recall_a.get("learnings", [])]
    assert not any("ruff" in c for c in contents_a), (
        f"learning should not appear in proj-A recall: {recall_a}"
    )


# ---------------------------------------------------------------------------
# Test 4: legacy rows with project_name=NULL still recalled via path-bridge
# ---------------------------------------------------------------------------


async def test_recall_project_finds_legacy_rows_via_path_bridge(engine, db):
    """
    A learning row written before this fix (project_name=NULL, project_path set)
    must still be surfaced by recall_project via the path-bridge fallback:
      SELECT ... WHERE project_name IS NULL AND project_path = (
          SELECT project_path FROM sessions WHERE project_name=$1 LIMIT 1
      )
    """
    # Create a session with project_name="legacy-proj" and a known project_path
    result = engine._create_session(
        mode="local",
        project_name="legacy-proj",
        metadata={},
    )
    assert result.status == "success"
    # Manually set a predictable project_path
    session_data = result.session_data.model_dump(mode="python")
    session_data["project_path"] = "/old/legacy/path"
    await db.save_session(session_data)

    # Manually insert a legacy learning row (project_name=NULL, project_path set)
    conn = db._ensure_connected()
    legacy_id = f"learn_{uuid.uuid4().hex[:12]}"
    now = datetime.now(UTC).isoformat()
    await conn.execute(
        """
        INSERT INTO project_learnings (
            id, project_path, project_name, category, trigger_context,
            learning_content, source_session_id, success_count, failure_count,
            last_used, promoted_to_universal, created_at
        ) VALUES (?, ?, NULL, ?, ?, ?, ?, 1, 0, ?, FALSE, ?)
        """,
        (
            legacy_id,
            "/old/legacy/path",
            "error_fix",
            "builds used to fail on CI",
            "legacy fix: pin dependency X to 1.2.3",
            result.session_id,
            now,
            now,
        ),
    )
    await conn.commit()

    # recall_project must find the legacy row via the path-bridge fallback
    recall = await engine.session_recall(project_name="legacy-proj")
    learnings = recall.get("learnings", [])
    contents = [lr.get("learning_content", "") for lr in learnings]
    assert any(
        "pin dependency" in c for c in contents
    ), f"legacy learning not found via path-bridge; learnings={learnings}"


# ---------------------------------------------------------------------------
# Test 5: _resolve_session_context returns full ResolvedSessionContext
# ---------------------------------------------------------------------------


async def test_resolved_context_returns_session_metadata(engine, db):
    """
    _resolve_session_context called with project_name="proj-X" must return a
    ResolvedSessionContext with session_id, project_name="proj-X", and
    project_path populated.
    """
    resolved = await engine._resolve_session_context(project_name="proj-X")

    assert isinstance(resolved, ResolvedSessionContext)
    assert resolved.session_id is not None
    assert resolved.session_id != ""
    assert resolved.project_name == "proj-X"
    # project_path may be None or a string — just confirm the attribute exists
    assert hasattr(resolved, "project_path")
