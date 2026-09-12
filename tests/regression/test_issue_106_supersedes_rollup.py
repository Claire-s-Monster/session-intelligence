"""
Regression tests for issue #106: `supersedes` is enforced in
`session_recall` but NOT in the notebook rollup path.

A decision or learning retired by a newer `supersedes` entry still appears
in every generated notebook because the rollup queries --
`query_decisions_by_session` and `query_project_learnings` -- never apply
the `id NOT IN (SELECT supersedes FROM <table> WHERE supersedes IS NOT
NULL)` exclusion that `session_recall` already applies (see
tests/regression/test_issue_87_supersedes.py and the SQL idiom at
src/persistence/sqlite.py:2148-2150 and 2183-2186).

Mirrors the #87 design exactly:
    - `supersedes` lives on the NEW (superseding) row and is insert-only.
    - A dangling pointer (an ID that does not exist) is deliberately
      accepted and retires nothing -- there is no existence validation
      (see issue #96). This is NOT a bug to fix here.
    - No cycle guard is needed: IDs are generated fresh at insert time, so
      a `supersedes` pointer always points backward in time.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


@pytest.mark.regression
class TestSupersedesRollupDecisions:

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_rollup_excludes_superseded_decision_but_keeps_superseder(
        self, engine
    ):
        first = await engine.session_log_decision(
            decision="Old, wrong decision",
            project_name="proj-rollup-a",
        )
        await engine.session_log_decision(
            decision="Corrected decision",
            project_name="proj-rollup-a",
            supersedes=first.decision_id,
        )

        rows = await engine.database.query_decisions_by_session(
            first.session_id, exclude_superseded=True
        )
        descriptions = [row["description"] for row in rows]

        assert "Corrected decision" in descriptions
        assert "Old, wrong decision" not in descriptions

    async def test_chain_of_three_rollup_returns_only_the_latest(
        self, engine
    ):
        a = await engine.session_log_decision(
            decision="A: first attempt",
            project_name="proj-rollup-a",
        )
        b = await engine.session_log_decision(
            decision="B: second attempt",
            project_name="proj-rollup-a",
            supersedes=a.decision_id,
        )
        await engine.session_log_decision(
            decision="C: final answer",
            project_name="proj-rollup-a",
            supersedes=b.decision_id,
        )

        rows = await engine.database.query_decisions_by_session(
            a.session_id, exclude_superseded=True
        )
        descriptions = [row["description"] for row in rows]

        assert "C: final answer" in descriptions
        assert "A: first attempt" not in descriptions
        assert "B: second attempt" not in descriptions

    async def test_default_still_returns_superseded_decision(self, engine):
        """Pins the migration-safety contract: the default must stay OFF.

        migration.py's _migrate_decisions and export_to_json call this
        method with no exclude_superseded argument and must keep seeing
        retired rows, or a migration/export would silently drop them.
        """
        first = await engine.session_log_decision(
            decision="Old, wrong decision",
            project_name="proj-rollup-a",
        )
        await engine.session_log_decision(
            decision="Corrected decision",
            project_name="proj-rollup-a",
            supersedes=first.decision_id,
        )

        rows = await engine.database.query_decisions_by_session(
            first.session_id
        )
        descriptions = [row["description"] for row in rows]

        assert "Corrected decision" in descriptions
        assert "Old, wrong decision" in descriptions

    async def test_dangling_supersedes_is_accepted_and_retires_nothing(
        self, engine
    ):
        unrelated = await engine.session_log_decision(
            decision="Unrelated decision that must survive",
            project_name="proj-rollup-a",
        )
        dangling = await engine.session_log_decision(
            decision="Corrects a decision that never existed",
            project_name="proj-rollup-a",
            supersedes="decision-doesnotexist",
        )
        assert dangling.supersedes == "decision-doesnotexist"

        rows = await engine.database.query_decisions_by_session(
            unrelated.session_id
        )
        descriptions = [row["description"] for row in rows]

        assert "Unrelated decision that must survive" in descriptions
        assert "Corrects a decision that never existed" in descriptions

    async def test_supersedes_defaults_to_none_and_is_still_in_rollup(
        self, engine
    ):
        result = await engine.session_log_decision(
            decision="Plain decision, no correction involved",
            project_name="proj-rollup-a",
        )
        assert result.supersedes is None

        rows = await engine.database.query_decisions_by_session(
            result.session_id
        )
        descriptions = [row["description"] for row in rows]
        assert "Plain decision, no correction involved" in descriptions


@pytest.mark.regression
class TestSupersedesRollupLearnings:

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_rollup_excludes_superseded_learning_but_keeps_superseder(
        self, engine
    ):
        project_path = "/tmp/proj-rollup-b"
        first = await engine.session_log_learning(
            category="error_fix",
            learning_content="Old, wrong fix",
            project_name="proj-rollup-b",
            project_path=project_path,
        )
        await engine.session_log_learning(
            category="error_fix",
            learning_content="Corrected fix",
            project_name="proj-rollup-b",
            project_path=project_path,
            supersedes=first.id,
        )

        rows = await engine.database.query_project_learnings(
            project_path, exclude_superseded=True
        )
        contents = [row["learning_content"] for row in rows]

        assert "Corrected fix" in contents
        assert "Old, wrong fix" not in contents

    async def test_chain_of_three_rollup_returns_only_the_latest(
        self, engine
    ):
        project_path = "/tmp/proj-rollup-b"
        a = await engine.session_log_learning(
            category="error_fix",
            learning_content="A: first attempt",
            project_name="proj-rollup-b",
            project_path=project_path,
        )
        b = await engine.session_log_learning(
            category="error_fix",
            learning_content="B: second attempt",
            project_name="proj-rollup-b",
            project_path=project_path,
            supersedes=a.id,
        )
        await engine.session_log_learning(
            category="error_fix",
            learning_content="C: final answer",
            project_name="proj-rollup-b",
            project_path=project_path,
            supersedes=b.id,
        )

        rows = await engine.database.query_project_learnings(
            project_path, exclude_superseded=True
        )
        contents = [row["learning_content"] for row in rows]

        assert "C: final answer" in contents
        assert "A: first attempt" not in contents
        assert "B: second attempt" not in contents

    async def test_default_still_returns_superseded_learning(self, engine):
        """Pins the migration-safety contract: the default must stay OFF.

        migration.py's export path and other unfiltered callers must keep
        seeing retired learnings when they pass no argument.
        """
        project_path = "/tmp/proj-rollup-b"
        first = await engine.session_log_learning(
            category="error_fix",
            learning_content="Old, wrong fix",
            project_name="proj-rollup-b",
            project_path=project_path,
        )
        await engine.session_log_learning(
            category="error_fix",
            learning_content="Corrected fix",
            project_name="proj-rollup-b",
            project_path=project_path,
            supersedes=first.id,
        )

        rows = await engine.database.query_project_learnings(project_path)
        contents = [row["learning_content"] for row in rows]

        assert "Corrected fix" in contents
        assert "Old, wrong fix" in contents

    async def test_dangling_supersedes_is_accepted_and_retires_nothing(
        self, engine
    ):
        project_path = "/tmp/proj-rollup-b"
        unrelated = await engine.session_log_learning(
            category="pattern",
            learning_content="Unrelated learning that must survive",
            project_name="proj-rollup-b",
            project_path=project_path,
        )
        dangling = await engine.session_log_learning(
            category="error_fix",
            learning_content="Corrects a learning that never existed",
            project_name="proj-rollup-b",
            project_path=project_path,
            supersedes="learn_does_not_exist_00000000",
        )
        assert dangling.learning.supersedes == "learn_does_not_exist_00000000"

        rows = await engine.database.query_project_learnings(project_path)
        contents = [row["learning_content"] for row in rows]

        assert "Unrelated learning that must survive" in contents
        assert "Corrects a learning that never existed" in contents
        assert unrelated.id  # sanity: the earlier entry really was saved

    async def test_supersedes_defaults_to_none_and_is_still_in_rollup(
        self, engine
    ):
        project_path = "/tmp/proj-rollup-b"
        result = await engine.session_log_learning(
            category="pattern",
            learning_content="Plain learning, no correction involved",
            project_name="proj-rollup-b",
            project_path=project_path,
        )
        assert result.learning.supersedes is None

        rows = await engine.database.query_project_learnings(project_path)
        contents = [row["learning_content"] for row in rows]
        assert "Plain learning, no correction involved" in contents


@pytest.mark.regression
class TestSupersedesRollupEndToEndNotebook:
    """End-to-end: go through the real notebook-creation tool path.

    `session_create_notebook` (the MCP tool) dispatches to
    `session_create_notebook_async` (see src/lean_mcp_interface.py:582), so
    that is the method exercised here -- not the two `query_*` methods in
    isolation. This is what actually pins issue #106: it would catch a
    regression where the `exclude_superseded=True` argument is present at
    the query call but never reaches the rendered markdown.
    """

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_notebook_excludes_superseded_decision(self, engine):
        first = await engine.session_log_decision(
            decision="Old, wrong decision",
            project_name="proj-rollup-e2e",
        )
        session_id = first.session_id
        await engine.session_log_decision(
            decision="Corrected decision",
            session_id=session_id,
            project_name="proj-rollup-e2e",
            supersedes=first.decision_id,
        )

        result = await engine.session_create_notebook_async(
            session_id=session_id,
            save_to_file=False,
            save_to_database=False,
        )

        assert result.status == "success"
        assert "Corrected decision" in result.markdown_output
        assert "Old, wrong decision" not in result.markdown_output

    async def test_notebook_excludes_superseded_learning(
        self, engine, tmp_path
    ):
        project_path = str(tmp_path / "e2e-learnings-project")
        first = await engine.session_log_decision(
            decision="Anchor decision to create a session",
            project_name="proj-rollup-e2e-learnings",
            project_path=project_path,
        )
        session_id = first.session_id

        learning_first = await engine.session_log_learning(
            category="error_fix",
            learning_content="Old, wrong fix",
            project_name="proj-rollup-e2e-learnings",
            project_path=project_path,
        )
        await engine.session_log_learning(
            category="error_fix",
            learning_content="Corrected fix",
            project_name="proj-rollup-e2e-learnings",
            project_path=project_path,
            supersedes=learning_first.id,
        )

        result = await engine.session_create_notebook_async(
            session_id=session_id,
            save_to_file=False,
            save_to_database=False,
        )

        assert result.status == "success"
        assert "Corrected fix" in result.markdown_output
        assert "Old, wrong fix" not in result.markdown_output

    async def test_notebook_generation_does_not_mutate_cached_session(
        self, engine
    ):
        """Pins issue #106's follow-up: rendering a notebook must not
        prune superseded decisions off the shared, cached Session object.

        Before the fix, `session_create_notebook_async` rebound
        `session.decisions` in place on whatever `self.session_cache`
        already held, so generating a notebook permanently stripped
        superseded decisions from the in-memory session every other
        caller shares -- a side effect visible for the rest of the
        process's lifetime, not just in the rendered markdown.
        """
        first = await engine.session_log_decision(
            decision="Old, wrong decision",
            project_name="proj-rollup-e2e-mutation",
        )
        session_id = first.session_id
        await engine.session_log_decision(
            decision="Corrected decision",
            session_id=session_id,
            project_name="proj-rollup-e2e-mutation",
            supersedes=first.decision_id,
        )

        cached_session = engine.session_cache[session_id]
        assert len(cached_session.decisions) == 2

        result = await engine.session_create_notebook_async(
            session_id=session_id,
            save_to_file=False,
            save_to_database=False,
        )
        assert result.status == "success"

        # The rendered notebook still excludes the superseded decision...
        assert "Corrected decision" in result.markdown_output
        assert "Old, wrong decision" not in result.markdown_output

        # ...but the cached session object itself must be untouched: both
        # the superseded and superseding decisions are still present, and
        # it must still be the SAME object identity as before.
        assert engine.session_cache[session_id] is cached_session
        assert len(cached_session.decisions) == 2
        descriptions = [d.description for d in cached_session.decisions]
        assert "Old, wrong decision" in descriptions
        assert "Corrected decision" in descriptions
