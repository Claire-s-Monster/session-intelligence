"""
Regression tests for issue #62: session_query_notebooks used to forward
project_path into a raw SQL equality match (WHERE s.project_path = ?), so a
legitimate path that merely differed in spelling (a trailing slash, an
unresolved symlink, a subdirectory checkout) returned ZERO rows instead of
the caller's notebooks.

The fix resolves project_path to a project_name via derive_project_name()
and matches on s.project_name (see persistence.sqlite.query_session_summaries
and the guard at the top of core.session_engine.session_query_notebooks,
~line 2583), and also accepts project_name directly as a first-class filter.

These tests seed real session + session_summary rows through the SQLite
backend (there is no shortcut: session_query_notebooks joins session_summaries
against sessions on session_id and filters on s.project_name), then exercise
session_query_notebooks under four scenarios: a differently-spelled path (the
verbatim #62 symptom), the new project_name parameter, an unusable path that
must not silently widen to every project, and the no-filter back-compat path.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from core.project_naming import derive_project_name
from core.session_engine import UNKNOWN_PROJECT_PATH, SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine(tmp_path):
    """SessionIntelligenceEngine backed by a fresh in-process SQLite database."""
    db = SQLiteBackend(str(tmp_path / "test.db"))
    await db.initialize()
    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


async def _seed_notebook(engine: SessionIntelligenceEngine, project_path: str, title: str):
    """Create a session bound to project_path, plus a notebook/summary for it.

    session_query_notebooks filters on s.project_name (joined from sessions),
    so both a sessions row and a session_summaries row are required for
    there to be anything to find.
    """
    session_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    project_name = derive_project_name(project_path)

    await engine.database.save_session(
        {
            "id": session_id,
            "started_at": now,
            "project_path": project_path,
            "project_name": project_name,
        }
    )
    await engine.database.save_session_summary(
        {
            "session_id": session_id,
            "title": title,
            "summary_markdown": f"# {title}",
            "key_changes": [],
            "tags": [],
            "created_at": now,
        }
    )
    return session_id, project_name


# ---------------------------------------------------------------------------
# Test 1: the core regression guard -- differently-spelled path must resolve
# ---------------------------------------------------------------------------


async def test_notebooks_found_when_queried_path_spelling_differs(engine, tmp_path):
    """
    Pre-fix, session_query_notebooks compared project_path with raw SQL
    string equality against s.project_path. A trailing slash on an otherwise
    identical, legitimate path returned zero rows -- this is the verbatim
    symptom reported in #62.
    """
    project_path = str(tmp_path)
    session_id, _project_name = await _seed_notebook(engine, project_path, "Notebook A")

    queried_path = project_path + "/"
    assert queried_path != project_path, "test setup requires a differing spelling"

    results = await engine.session_query_notebooks(project_path=queried_path)

    session_ids = {r["session_id"] for r in results}
    assert session_id in session_ids, (
        f"expected notebook for session {session_id} when querying with "
        f"differently-spelled path {queried_path!r}; got {results!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: the new project_name parameter
# ---------------------------------------------------------------------------


async def test_notebooks_found_when_queried_by_project_name(engine, tmp_path):
    """project_name must work as a first-class filter, independent of project_path."""
    project_path = str(tmp_path)
    session_id, project_name = await _seed_notebook(engine, project_path, "Notebook B")

    results = await engine.session_query_notebooks(project_name=project_name)

    session_ids = {r["session_id"] for r in results}
    assert session_id in session_ids, (
        f"expected notebook for session {session_id} when querying by "
        f"project_name={project_name!r}; got {results!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: an unusable project_path must yield empty, never every project
# ---------------------------------------------------------------------------


async def test_unusable_project_path_returns_empty_not_every_project(engine, tmp_path):
    """
    An unusable project_path filter (a relative path, or the _unknown_
    sentinel) must return an empty list -- never silently widen to an
    unfiltered query returning every project's notebooks. This pins the
    deliberate design choice documented in session_query_notebooks: unlike
    the write-path guards it mirrors, a bad filter here must not fall back
    to "return everything".
    """
    proj_a = tmp_path / "proj_a"
    proj_b = tmp_path / "proj_b"
    proj_a.mkdir()
    proj_b.mkdir()

    session_a, _ = await _seed_notebook(engine, str(proj_a), "Notebook A")
    session_b, _ = await _seed_notebook(engine, str(proj_b), "Notebook B")

    for unusable_path in ("some/relative/dir", UNKNOWN_PROJECT_PATH):
        results = await engine.session_query_notebooks(project_path=unusable_path)

        session_ids = {r["session_id"] for r in results}
        assert results == [], (
            f"expected empty result for unusable project_path {unusable_path!r}, "
            f"got {results!r}"
        )
        assert session_a not in session_ids, (
            f"project_path={unusable_path!r} leaked proj_a's notebook"
        )
        assert session_b not in session_ids, (
            f"project_path={unusable_path!r} leaked proj_b's notebook"
        )


# ---------------------------------------------------------------------------
# Test 4: no filter still returns every project's notebooks (back-compat)
# ---------------------------------------------------------------------------


async def test_no_project_filter_still_returns_all_notebooks(engine, tmp_path):
    """With no project_path and no project_name, the unfiltered path must still work."""
    proj_a = tmp_path / "proj_a"
    proj_b = tmp_path / "proj_b"
    proj_a.mkdir()
    proj_b.mkdir()

    session_a, _ = await _seed_notebook(engine, str(proj_a), "Notebook A")
    session_b, _ = await _seed_notebook(engine, str(proj_b), "Notebook B")

    results = await engine.session_query_notebooks()

    session_ids = {r["session_id"] for r in results}
    assert session_a in session_ids, f"expected proj_a's notebook, got {results!r}"
    assert session_b in session_ids, f"expected proj_b's notebook, got {results!r}"
