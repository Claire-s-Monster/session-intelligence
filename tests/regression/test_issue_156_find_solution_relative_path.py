"""
Regression tests for issue #156: ``session_find_solution``
(session_engine.py:3619-3703) builds its query scope with

    effective_project = project_path or str(self.claude_sessions_path.parent)

and uses ``effective_project`` unfiltered as the ``project_path`` argument to
both ``self.database.find_error_solutions`` (:3654) and
``self.database.query_project_learnings`` (:3672).

A caller-supplied RELATIVE ``project_path`` resolves against the SERVER's
cwd, not the caller's, so it is meaningless here -- the same reasoning
already applied to write paths by ``usable_project_path()`` (issues
#153/#155) and to session resolution by ``_resolve_session_context``
(session_engine.py:335-346).

Critically, the fix required is NOT "apply ``usable_project_path()`` and let
the existing ``project_path or str(self.claude_sessions_path.parent)``
fallback run". That would turn an unusable ``project_path`` into ``None``
and then silently re-scope the query to the SERVER's OWN directory
(``claude_sessions_path.parent``) -- returning whatever solutions/learnings
happen to be recorded against the *server's* project instead of the
caller's, which is worse than today's harmless empty-ish result. The
required contract mirrors ``session_query_notebooks``
(session_engine.py:3195-3234; read its comment for the exact reasoning):

  - project_path omitted (None)      -> existing default scope, UNCHANGED
  - project_path supplied and usable -> used as the filter, UNCHANGED
  - project_path supplied but UNUSABLE (relative, or the
    UNKNOWN_PROJECT_PATH sentinel)  -> return an EMPTY SolutionSearchResult;
                                        must NOT fall back to
                                        claude_sessions_path.parent and must
                                        NOT query with a different scope.

Uses an in-memory SQLite backend so all tests are fully isolated from real
filesystem state and PostgreSQL.
"""

import uuid

import pytest

from core.session_engine import UNKNOWN_PROJECT_PATH, SessionIntelligenceEngine
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


async def _seed_error_solution(
    db: SQLiteBackend, project_path: str, *, error_pattern: str = "boom: something failed"
) -> str:
    """Insert an error_solutions row scoped to ``project_path``.

    Returns the generated solution_id. Mirrors
    ``SQLiteBackend.save_error_solution``'s signature (solution_id,
    error_pattern, solution_steps, project_path, ...).
    """
    solution_id = f"sol-{uuid.uuid4().hex[:12]}"
    await db.save_error_solution(
        solution_id=solution_id,
        error_pattern=error_pattern,
        solution_steps=["do the fix"],
        project_path=project_path,
    )
    return solution_id


async def _seed_project_learning(
    db: SQLiteBackend, project_path: str, *, learning_content: str = "boom: something failed"
) -> str:
    """Insert a project_learnings row scoped to ``project_path``.

    Returns the generated learning_id. Mirrors
    ``SQLiteBackend.save_project_learning``'s signature (learning_id,
    project_path, category, learning_content, ...).

    Passes ``trigger_context=""`` rather than leaving it at
    ``save_project_learning``'s own ``None`` default: when a row seeded
    here is later matched by ``session_find_solution``'s
    ``project_learnings`` scan (session_engine.py ~3682), that code does
    ``lr.get("learning_content", "") + lr.get("trigger_context", "")`` --
    since the key is present with value ``None`` (not absent), the ``.get``
    default never applies and ``str + None`` raises ``TypeError``. That is
    a separate, pre-existing bug unrelated to issue #156's project_path
    scoping, so seeding a non-None ``trigger_context`` here avoids
    incidentally exercising it and masking the scoping behaviour this
    module is testing.
    """
    learning_id = f"learn-{uuid.uuid4().hex[:12]}"
    await db.save_project_learning(
        learning_id=learning_id,
        project_path=project_path,
        category="pattern",
        learning_content=learning_content,
        trigger_context="",
    )
    return learning_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFindSolutionRelativeProjectPath:
    async def test_relative_project_path_returns_empty_not_other_project_rows(self, engine, db):
        """THE IMPORTANT ONE.

        Seed a solution and a learning that WOULD match if the query were
        (mis)scoped to ``str(engine.claude_sessions_path.parent)`` -- the
        naive-guard fallback target. Calling with a relative project_path
        must return a completely empty SolutionSearchResult.

        If this fails by returning the seeded rows, it proves a naive
        "apply usable_project_path() and let the existing `project_path or
        str(self.claude_sessions_path.parent)` fallback run" fix was used:
        that turns an unusable relative path into None, which then falls
        through to the SERVER's own directory scope and leaks whatever
        solutions/learnings happen to be recorded there -- silent
        cross-project leakage that is strictly worse than an empty result.
        """
        error_text = "boom: something failed in issue 156 relative scope"
        server_scope = str(engine.claude_sessions_path.parent)
        await _seed_error_solution(db, server_scope, error_pattern=error_text)
        await _seed_project_learning(db, server_scope, learning_content=error_text)

        result = await engine.session_find_solution(
            error_text=error_text,
            project_path="relative/caller/path",
        )

        assert result.total_found == 0
        assert result.solutions == []
        assert result.project_specific_count == 0
        assert result.universal_count == 0

    async def test_sentinel_project_path_returns_empty(self, engine, db):
        """THE LIVE LEAK (not merely a trap for a future naive fix, unlike
        test 1 above).

        ``UNKNOWN_PROJECT_PATH`` ("_unknown_") is not a project -- it is the
        marker the engine itself writes when it could not determine which
        project a session/row belongs to (see ``_resolve_session_context``).
        Because it is a non-empty string it is TRUTHY, so the current

            effective_project = project_path or str(self.claude_sessions_path.parent)

        (session_engine.py:3641) does NOT fall back for it: it is used
        VERBATIM as the query filter passed to
        ``database.find_error_solutions`` (:3656) and
        ``database.query_project_learnings`` (:3673). Per issues #151/#154
        the real database already contains many rows whose stored
        ``project_path`` IS the sentinel, and those rows belong to
        ARBITRARY DIFFERENT, UNRELATED projects -- they only share the
        sentinel because none of them could be resolved at write time, not
        because they are the same project. Honouring the sentinel as a
        scope therefore groups every unrelated project's unresolved rows
        into one bucket and hands the whole bucket to ANY caller who passes
        ``project_path="_unknown_"`` today, right now, with no future code
        change required to trigger it.

        This test seeds an error solution and a project learning that are
        stored under the sentinel but are clearly attributable to some
        *other* project (distinctive content mentioning
        "other-project-secret"), then asserts that querying with the
        sentinel returns nothing at all: the sentinel must never be
        honoured as a real scope, matched rows or not.
        """
        error_text = "boom: something failed in issue 156 sentinel scope"
        leaked_pattern = f"{error_text} (other-project-secret)"
        await _seed_error_solution(db, UNKNOWN_PROJECT_PATH, error_pattern=leaked_pattern)
        await _seed_project_learning(db, UNKNOWN_PROJECT_PATH, learning_content=leaked_pattern)
        # A second, differently-worded sentinel-scoped row makes it
        # unambiguous these represent a pool of DIFFERENT unrelated
        # projects' unresolved rows, not one project that happens to be
        # named "_unknown_".
        await _seed_error_solution(
            db, UNKNOWN_PROJECT_PATH, error_pattern=f"{leaked_pattern} (yet-another-project)"
        )

        result = await engine.session_find_solution(
            error_text=error_text,
            project_path=UNKNOWN_PROJECT_PATH,
        )

        assert result.total_found == 0
        assert result.solutions == []
        assert result.project_specific_count == 0
        assert result.universal_count == 0

    async def test_absolute_project_path_still_queries_that_scope(self, engine, db):
        """Guards against over-rejection: a trustworthy, absolute
        caller-supplied project_path must still be used as the query
        filter and find matching rows."""
        error_text = "boom: something failed in issue 156 absolute scope"
        real_scope = "/abs/real/proj-156"
        await _seed_error_solution(db, real_scope, error_pattern=error_text)

        result = await engine.session_find_solution(
            error_text=error_text,
            project_path=real_scope,
        )

        assert result.total_found >= 1
        assert any(s.project_path == real_scope for s in result.solutions)

    async def test_omitted_project_path_keeps_default_scope(self, engine, db):
        """Pins that the default-scope behaviour (no project_path supplied
        at all) is unchanged: it must still fall back to
        ``str(engine.claude_sessions_path.parent)`` and find rows seeded
        there."""
        error_text = "boom: something failed in issue 156 default scope"
        server_scope = str(engine.claude_sessions_path.parent)
        await _seed_error_solution(db, server_scope, error_pattern=error_text)

        result = await engine.session_find_solution(error_text=error_text)

        assert result.total_found >= 1
        assert any(s.project_path == server_scope for s in result.solutions)
