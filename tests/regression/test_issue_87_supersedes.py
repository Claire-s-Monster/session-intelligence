"""
Regression tests for issue #87: a `supersedes` field so a correction retires
its predecessor.

Design:
    `supersedes` is a nullable TEXT column holding the ID of the PRIOR entry.
    It lives on the NEW (superseding) row. Insert-only; there is no update
    path. A row is "retired" if it is the target of a live `supersedes`
    pointer -- session_recall excludes retired rows but keeps the
    superseding row.

    A dangling pointer (an ID that does not exist) is deliberately accepted
    and simply never retires anything -- there is no existence validation.
    Rejecting on a bad pointer would surface as {"status": "error"} wrapped
    in an HTTP 200 (in-band error), so a false rejection would be silent
    data loss -- see issue #96.

    There is no cycle guard: IDs are generated fresh at insert time, so a
    `supersedes` pointer always points backward in time, making cycles
    structurally impossible.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


@pytest.mark.regression
class TestSupersedesLearnings:
    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_supersedes_persists_on_the_new_row(self, engine):
        first = await engine.session_log_learning(
            category="error_fix",
            learning_content="Old, wrong fix",
            project_name="proj-a",
        )
        second = await engine.session_log_learning(
            category="error_fix",
            learning_content="Corrected fix",
            project_name="proj-a",
            supersedes=first.id,
        )
        assert second.learning.supersedes == first.id

    async def test_recall_excludes_superseded_learning_but_keeps_superseder(self, engine):
        first = await engine.session_log_learning(
            category="error_fix",
            learning_content="Old, wrong fix",
            project_name="proj-a",
        )
        await engine.session_log_learning(
            category="error_fix",
            learning_content="Corrected fix",
            project_name="proj-a",
            supersedes=first.id,
        )

        recall = await engine.session_recall(project_name="proj-a")
        contents = [entry["learning_content"] for entry in recall["learnings"]]

        assert "Corrected fix" in contents
        assert "Old, wrong fix" not in contents

    async def test_chain_of_three_recall_returns_only_the_latest(self, engine):
        a = await engine.session_log_learning(
            category="error_fix",
            learning_content="A: first attempt",
            project_name="proj-a",
        )
        b = await engine.session_log_learning(
            category="error_fix",
            learning_content="B: second attempt",
            project_name="proj-a",
            supersedes=a.id,
        )
        await engine.session_log_learning(
            category="error_fix",
            learning_content="C: final answer",
            project_name="proj-a",
            supersedes=b.id,
        )

        recall = await engine.session_recall(project_name="proj-a")
        contents = [entry["learning_content"] for entry in recall["learnings"]]

        assert "C: final answer" in contents
        assert "A: first attempt" not in contents
        assert "B: second attempt" not in contents

    async def test_dangling_supersedes_is_accepted_and_retires_nothing(self, engine):
        unrelated = await engine.session_log_learning(
            category="pattern",
            learning_content="Unrelated learning that must survive",
            project_name="proj-a",
        )
        dangling = await engine.session_log_learning(
            category="error_fix",
            learning_content="Corrects a learning that never existed",
            project_name="proj-a",
            supersedes="learn_does_not_exist_00000000",
        )
        assert dangling.learning.supersedes == "learn_does_not_exist_00000000"

        recall = await engine.session_recall(project_name="proj-a")
        contents = [entry["learning_content"] for entry in recall["learnings"]]

        assert "Unrelated learning that must survive" in contents
        assert "Corrects a learning that never existed" in contents
        assert unrelated.id  # sanity: the earlier entry really was saved

    async def test_supersedes_defaults_to_none_and_is_still_recalled(self, engine):
        result = await engine.session_log_learning(
            category="pattern",
            learning_content="Plain learning, no correction involved",
            project_name="proj-a",
        )
        assert result.learning.supersedes is None

        recall = await engine.session_recall(project_name="proj-a")
        contents = [entry["learning_content"] for entry in recall["learnings"]]
        assert "Plain learning, no correction involved" in contents


@pytest.mark.regression
class TestSupersedesDecisions:
    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_supersedes_persists_on_the_new_row(self, engine):
        first = await engine.session_log_decision(
            decision="Old, wrong decision",
            project_name="proj-b",
        )
        second = await engine.session_log_decision(
            decision="Corrected decision",
            project_name="proj-b",
            supersedes=first.decision_id,
        )
        assert second.supersedes == first.decision_id

    async def test_recall_excludes_superseded_decision_but_keeps_superseder(self, engine):
        first = await engine.session_log_decision(
            decision="Old, wrong decision",
            project_name="proj-b",
        )
        await engine.session_log_decision(
            decision="Corrected decision",
            project_name="proj-b",
            supersedes=first.decision_id,
        )

        recall = await engine.session_recall(project_name="proj-b")
        descriptions = [entry["description"] for entry in recall["decisions"]]

        assert "Corrected decision" in descriptions
        assert "Old, wrong decision" not in descriptions

    async def test_chain_of_three_recall_returns_only_the_latest(self, engine):
        a = await engine.session_log_decision(
            decision="A: first attempt",
            project_name="proj-b",
        )
        b = await engine.session_log_decision(
            decision="B: second attempt",
            project_name="proj-b",
            supersedes=a.decision_id,
        )
        await engine.session_log_decision(
            decision="C: final answer",
            project_name="proj-b",
            supersedes=b.decision_id,
        )

        recall = await engine.session_recall(project_name="proj-b")
        descriptions = [entry["description"] for entry in recall["decisions"]]

        assert "C: final answer" in descriptions
        assert "A: first attempt" not in descriptions
        assert "B: second attempt" not in descriptions

    async def test_dangling_supersedes_is_accepted_and_retires_nothing(self, engine):
        await engine.session_log_decision(
            decision="Unrelated decision that must survive",
            project_name="proj-b",
        )
        dangling = await engine.session_log_decision(
            decision="Corrects a decision that never existed",
            project_name="proj-b",
            supersedes="decision-doesnotexist",
        )
        assert dangling.supersedes == "decision-doesnotexist"

        recall = await engine.session_recall(project_name="proj-b")
        descriptions = [entry["description"] for entry in recall["decisions"]]

        assert "Unrelated decision that must survive" in descriptions
        assert "Corrects a decision that never existed" in descriptions

    async def test_supersedes_defaults_to_none_and_is_still_recalled(self, engine):
        result = await engine.session_log_decision(
            decision="Plain decision, no correction involved",
            project_name="proj-b",
        )
        assert result.supersedes is None

        recall = await engine.session_recall(project_name="proj-b")
        descriptions = [entry["description"] for entry in recall["decisions"]]
        assert "Plain decision, no correction involved" in descriptions
