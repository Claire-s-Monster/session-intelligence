"""
Regression tests for issue #53: session_log_decision rejects project_path,
diverging from session_log_learning after PR #52.

Verifies session_log_decision now accepts project_path and derives
project_name from it symmetrically with session_log_learning, subject to
the same two guards (absolute path required, derived UNBOUND discarded).
"""

import pytest

from core.project_naming import UNBOUND, derive_project_name
from core.session_engine import SessionContextRequiredError, SessionIntelligenceEngine
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
# Test 1: project_path is accepted without raising
# ---------------------------------------------------------------------------


async def test_log_decision_accepts_project_path(engine, db, tmp_path):
    """
    session_log_decision(project_path=...) must not raise TypeError (the
    parameter did not exist before this fix) and must produce a bound
    session, not the '_unbound_' fallback.
    """
    result = await engine.session_log_decision(
        decision="use dataclasses for frozen value objects",
        project_path=str(tmp_path),
    )

    assert result.decision_id != "error"
    assert result.session_id != "unknown"


# ---------------------------------------------------------------------------
# Test 2: project_path derives the same project_name as session_log_learning
# ---------------------------------------------------------------------------


async def test_decision_project_path_derives_same_name_as_learning(engine, db, tmp_path):
    """
    For the same absolute project_path, session_log_decision must derive
    the same project_name as session_log_learning does.
    """
    expected_name = derive_project_name(str(tmp_path))
    assert expected_name != UNBOUND

    learn_result = await engine.session_log_learning(
        category="pattern",
        learning_content="x",
        project_path=str(tmp_path),
    )
    assert learn_result.status == "saved"

    decision_result = await engine.session_log_decision(
        decision="y",
        project_path=str(tmp_path),
    )

    session = await db.get_session(decision_result.session_id)
    assert session is not None
    assert session["project_name"] == expected_name


# ---------------------------------------------------------------------------
# Test 3: a relative project_path is ignored (guards #48/#49 misattribution)
# ---------------------------------------------------------------------------


async def test_decision_relative_project_path_is_ignored(engine, db):
    """
    A relative project_path must NOT be used to derive a project_name,
    since derive_project_name() would resolve it against the server's cwd
    rather than the caller's — the exact misattribution bug #48/#49 fixed.

    With the relative path correctly ignored and no other session
    identifier supplied, session_log_decision falls through to the same
    "at least one identifier required" guard as session_log_learning and
    raises SessionContextRequiredError rather than silently creating a
    session under the misattributed name.
    """
    relative_path = "some/relative/dir"
    misattributed_name = derive_project_name(relative_path)
    assert misattributed_name != UNBOUND  # sanity: this would be a real (wrong) name

    with pytest.raises(SessionContextRequiredError):
        await engine.session_log_decision(
            decision="z",
            project_path=relative_path,
        )


# ---------------------------------------------------------------------------
# Test 4: explicit project_name wins over project_path
# ---------------------------------------------------------------------------


async def test_decision_explicit_project_name_wins_over_project_path(engine, db, tmp_path):
    """
    When both project_name and project_path are supplied, the explicit
    project_name must win — project_path is only a fallback source.
    """
    decision_result = await engine.session_log_decision(
        decision="w",
        project_name="explicit-proj",
        project_path=str(tmp_path),
    )

    session = await db.get_session(decision_result.session_id)
    assert session is not None
    assert session["project_name"] == "explicit-proj"
