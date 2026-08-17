"""
Regression tests for issue #72: unbound decisions/learnings silently bound to
whichever project most recently created a session in the shared engine.

Bug: session_log_decision used to fall back to the in-process
     "_current_session_id" when no session_id/session_name/project_name was
     supplied. The HTTP transport builds a SINGLE SessionIntelligenceEngine
     shared by every project (see http_server.py lifespan()), so that ambient
     fallback bound to whichever project most recently created a session in
     the process -- not the caller's project. Decisions for
     session-intelligence itself landed under package-incubator.
Fix: session_log_decision now raises SessionContextRequiredError when none of
     session_id/session_name/project_name is supplied (unless
     allow_unbound=True is explicitly passed), matching session_log_learning.
     session_manage_lifecycle(operation="create", ...) also now records an
     absolute caller-supplied project_path on the session instead of always
     stamping the UNKNOWN_PROJECT_PATH sentinel.
"""

import pytest

from core.session_engine import SessionContextRequiredError, SessionIntelligenceEngine, UNKNOWN_PROJECT_PATH
from persistence.sqlite import SQLiteBackend


@pytest.mark.regression
class TestUnboundDecisionIsRejected:

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_unbound_decision_raises_even_after_ambient_session_exists(self, engine):
        """The core #72 regression: an in-process current session must NOT
        silently satisfy an unscoped session_log_decision call."""
        create_result = await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-a"
        )
        assert create_result.status == "success", (
            "Precondition failed: session_manage_lifecycle(create) did not "
            "succeed, so this test cannot exercise the #72 ambient-session "
            "bleed scenario."
        )

        with pytest.raises(SessionContextRequiredError):
            await engine.session_log_decision(decision="Unscoped decision")

    async def test_allow_unbound_escape_hatch_still_works(self, engine):
        await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-a"
        )

        result = await engine.session_log_decision(
            decision="Unscoped but explicitly allowed",
            allow_unbound=True,
        )
        assert result is not None

    async def test_decision_with_project_name_does_not_raise(self, engine):
        await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-a"
        )

        result = await engine.session_log_decision(
            decision="Scoped decision",
            project_name="proj-a",
        )
        assert result is not None


@pytest.mark.regression
class TestCreateRecordsProjectPath:

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_absolute_project_path_is_stored(self, engine, tmp_path):
        abs_path = str(tmp_path / "proj-a")

        result = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-a",
            project_path=abs_path,
        )

        db_session = await engine.database.get_session(result.session_id)
        assert db_session is not None
        assert db_session["project_path"] == abs_path, (
            "An absolute caller-supplied project_path must be recorded "
            "verbatim on the session, not discarded in favor of the "
            f"{UNKNOWN_PROJECT_PATH!r} sentinel."
        )
        assert db_session["project_path"] != UNKNOWN_PROJECT_PATH

    async def test_relative_project_path_is_ignored_and_sentinel_stored(self, engine):
        result = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-a",
            project_path="relative/proj-a",
        )

        db_session = await engine.database.get_session(result.session_id)
        assert db_session is not None
        assert db_session["project_path"] == UNKNOWN_PROJECT_PATH, (
            "A relative project_path resolves against the SERVER's cwd, not "
            "the caller's, so it must be rejected and the "
            f"{UNKNOWN_PROJECT_PATH!r} sentinel stored instead -- otherwise "
            "sessions get misattributed the same way issue #72 did."
        )

    async def test_omitted_project_path_stores_sentinel(self, engine):
        result = await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-a"
        )

        db_session = await engine.database.get_session(result.session_id)
        assert db_session is not None
        assert db_session["project_path"] == UNKNOWN_PROJECT_PATH


@pytest.mark.regression
class TestSharedEngineDoesNotBleedAcrossProjects:
    """Every existing resolver test built a FRESH engine per test, hiding
    #72: the HTTP transport builds ONE engine shared across all projects.
    These tests reuse a single engine for two different projects."""

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_decision_binds_to_named_project_not_most_recently_created(
        self, engine, tmp_path
    ):
        proj_a_path = str(tmp_path / "proj-a")
        proj_b_path = str(tmp_path / "proj-b")

        proj_a_create = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-a",
            project_path=proj_a_path,
        )
        # proj-b is created SECOND, i.e. it is the most-recently-created
        # session in this shared engine when the decision below is logged.
        await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-b",
            project_path=proj_b_path,
        )

        result = await engine.session_log_decision(
            decision="Decision scoped to proj-a",
            project_name="proj-a",
        )

        assert result.session_id == proj_a_create.session_id, (
            "Issue #72: the decision resolved to the wrong session. It must "
            "bind to proj-a's session (the caller's project_name), not to "
            "proj-b's session merely because proj-b's was created most "
            "recently in this shared engine."
        )

        db_session = await engine.database.get_session(result.session_id)
        assert db_session is not None
        assert db_session["project_path"] == proj_a_path, (
            "Issue #72: decision resolved to a session whose project_path "
            f"is {db_session['project_path']!r}, but it should be proj-a's "
            f"path {proj_a_path!r}, not proj-b's."
        )

    async def test_learning_binds_to_named_project_not_most_recently_created(
        self, engine, tmp_path
    ):
        proj_a_path = str(tmp_path / "proj-a")
        proj_b_path = str(tmp_path / "proj-b")

        proj_a_create = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-a",
            project_path=proj_a_path,
        )
        await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-b",
            project_path=proj_b_path,
        )

        result = await engine.session_log_learning(
            category="pattern",
            learning_content="Learning scoped to proj-a",
            trigger_context="Testing issue #72 project binding",
            project_name="proj-a",
        )

        assert result.learning is not None
        assert result.learning.source_session_id == proj_a_create.session_id, (
            "Issue #72: the learning's source_session_id resolved to the "
            "wrong session. It must bind to proj-a's session (the caller's "
            "project_name), not to proj-b's session merely because proj-b's "
            "was created most recently in this shared engine."
        )
        assert result.learning.project_path == proj_a_path, (
            "Issue #72: learning resolved to project_path "
            f"{result.learning.project_path!r}, but it should be proj-a's "
            f"path {proj_a_path!r}, not proj-b's."
        )
