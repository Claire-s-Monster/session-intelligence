"""
Regression tests for issue #158: session_find_solution silently discards
matches (and even previously-found error_solutions) when a project_learnings
row has a NULL trigger_context.

Root cause (core/session_engine.py, session_find_solution):

    matching_count = sum(
        1
        for lr in learnings
        if error_text.lower()
        in (lr.get("learning_content", "") + lr.get("trigger_context", "")).lower()
    )

`trigger_context` is a nullable TEXT column (persistence/sqlite.py,
persistence/postgresql.py) and `ProjectLearning.trigger_context` is
`str | None = None`. When the column is NULL the key is PRESENT in the row
dict with value `None` -- so `.get(key, "")` returns `None`, not the
default `""`. `str + None` raises `TypeError: can only concatenate str
(not "NoneType") to str`.

That TypeError is caught by the broad `except Exception` that wraps the
*entire* try block -- a try block that opens ABOVE the `find_error_solutions`
query too. So the already-built `solutions` list (from the error_solutions
table, which has nothing to do with trigger_context) is discarded as well,
and the whole call quietly degrades to an all-zeros SolutionSearchResult.
"""

from __future__ import annotations

import pytest

from core.session_engine import SessionIntelligenceEngine
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
# Helpers
# ---------------------------------------------------------------------------


async def _seed_project_learning(
    db,
    engine,
    learning_content: str,
    trigger_context: str | None = None,
    category: str = "error_fix",
    learning_id: str | None = None,
):
    """Insert a project_learnings row scoped to the engine's default scope.

    IMPORTANT: trigger_context defaults to None (NULL column), not "".
    Passing "" here would sidestep the exact bug under test.
    """
    scope = str(engine.claude_sessions_path.parent)
    await db.save_project_learning(
        learning_id=learning_id or f"learn_{abs(hash(learning_content))}",
        project_path=scope,
        category=category,
        learning_content=learning_content,
        trigger_context=trigger_context,
    )
    return scope


async def _seed_error_solution(
    db,
    engine,
    error_pattern: str,
    solution_id: str | None = None,
):
    """Insert an error_solutions row scoped to the engine's default scope."""
    scope = str(engine.claude_sessions_path.parent)
    await db.save_error_solution(
        solution_id=solution_id or f"sol_{abs(hash(error_pattern))}",
        error_pattern=error_pattern,
        solution_steps=["do the fix"],
        project_path=scope,
    )
    return scope


# ---------------------------------------------------------------------------
# Test 1: NULL trigger_context must not prevent a learning_content match
# ---------------------------------------------------------------------------


async def test_null_trigger_context_learning_is_still_matched(engine, db):
    """
    Pins: a project_learnings row whose learning_content matches error_text
    but whose trigger_context is NULL must still be counted/found.

    Currently fails: the `lr.get("trigger_context", "")` NULL-column read
    returns None (not the "" default), so `learning_content + None` raises
    TypeError, which is swallowed by the outer except -> total_found == 0.
    """
    error_text = "ISSUE158-ERR-A: ModuleNotFoundError for widget package"
    await _seed_project_learning(
        db,
        engine,
        learning_content=f"Fix: pip install widget ({error_text})",
        trigger_context=None,
    )

    result = await engine.session_find_solution(error_text=error_text)

    assert result.total_found >= 1, (
        f"expected the NULL-trigger_context learning to be matched, "
        f"got total_found={result.total_found}"
    )


# ---------------------------------------------------------------------------
# Test 2: NULL trigger_context must not discard already-found error_solutions
# ---------------------------------------------------------------------------


async def test_null_trigger_context_does_not_discard_error_solutions(engine, db):
    """
    Pins the WIDER blast radius of #158, which is what makes it worse than
    "the learnings-matching branch degrades gracefully to zero": the entire
    try block in session_find_solution wraps BOTH the error_solutions query
    AND the project_learnings matching loop. When the learnings loop raises
    TypeError (NULL trigger_context, see test 1), the broad `except
    Exception` handler discards the *already successfully built* `solutions`
    list from find_error_solutions and returns an all-zeros
    SolutionSearchResult -- even though the error_solutions lookup itself
    had nothing to do with trigger_context and worked fine.

    Currently fails: result.solutions is empty and total_found == 0, even
    though a matching error_solutions row was seeded.
    """
    error_text = "ISSUE158-ERR-B: TimeoutError connecting to gizmo service"
    await _seed_error_solution(db, engine, error_pattern=error_text)
    await _seed_project_learning(
        db,
        engine,
        learning_content="unrelated learning content",
        trigger_context=None,
    )

    result = await engine.session_find_solution(error_text=error_text)

    assert result.solutions, (
        f"expected the seeded error_solutions row to survive, got "
        f"solutions={result.solutions!r} (total_found={result.total_found})"
    )
    assert any(s.error_pattern == error_text for s in result.solutions)


# ---------------------------------------------------------------------------
# Test 3: matching via trigger_context alone must keep working after fix
# ---------------------------------------------------------------------------


async def test_learning_matched_via_trigger_context_only(engine, db):
    """
    Guards against an over-narrow fix that stops consulting trigger_context
    entirely (e.g. by only checking learning_content). A learning whose
    learning_content does NOT contain the error text, but whose
    trigger_context DOES, must still be found both before and after the fix.
    """
    error_text = "ISSUE158-ERR-C: PermissionError writing to /var/lock/gadget"
    await _seed_project_learning(
        db,
        engine,
        learning_content="unrelated content with no error text at all",
        trigger_context=f"seen when: {error_text}",
    )

    result = await engine.session_find_solution(error_text=error_text)

    assert result.total_found >= 1, (
        f"expected trigger_context-only match to be found, got total_found={result.total_found}"
    )


# ---------------------------------------------------------------------------
# Test 4: baseline -- non-NULL trigger_context is unaffected (passes today)
# ---------------------------------------------------------------------------


async def test_non_null_trigger_context_unchanged(engine, db):
    """
    Baseline sanity check that already passes today: a learning with a
    normal, non-empty trigger_context (no NULL involved) is matched via
    learning_content as expected.
    """
    error_text = "ISSUE158-ERR-D: ConnectionRefusedError on port 5432"
    await _seed_project_learning(
        db,
        engine,
        learning_content=f"Fix: start postgres before running tests ({error_text})",
        trigger_context="normal, non-null trigger context",
    )

    result = await engine.session_find_solution(error_text=error_text)

    assert result.total_found >= 1, (
        f"expected non-null trigger_context learning to be matched, "
        f"got total_found={result.total_found}"
    )
