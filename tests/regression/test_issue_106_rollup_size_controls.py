"""
Regression tests for issue #106 "PR B": caller-facing rollup size controls.

`session_create_notebook` (via `session_create_notebook_async`, see
src/lean_mcp_interface.py:582) previously hardcoded `exclude_superseded=True`
and had no way to bound the number of decisions or the time window of the
decision/learning rollup, so `summary_markdown` could balloon unboundedly
(observed ~68k chars). This adds three caller-facing controls:

    - `max_decisions`: caps decisions returned by `query_decisions_by_session`.
    - `since_days`: restricts both decisions and learnings to the last N days.
    - `exclude_superseded`: promoted from hardcoded True to caller-controlled.

Mirrors the structure of test_issue_106_supersedes_rollup.py: direct backend
checks plus end-to-end checks through `session_create_notebook_async`, which
is what the MCP tool actually dispatches to.
"""

from datetime import UTC, datetime, timedelta

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend


async def _backdate_decision(
    engine: SessionIntelligenceEngine, decision_id: str, days: int
) -> None:
    conn = engine.database._ensure_connected()
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    await conn.execute("UPDATE decisions SET timestamp = ? WHERE id = ?", (cutoff, decision_id))
    await conn.commit()


async def _backdate_learning(
    engine: SessionIntelligenceEngine, learning_id: str, days: int
) -> None:
    conn = engine.database._ensure_connected()
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    await conn.execute(
        "UPDATE project_learnings SET created_at = ? WHERE id = ?", (cutoff, learning_id)
    )
    await conn.commit()


@pytest.mark.regression
class TestMaxDecisions:
    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_max_decisions_caps_rollup_count(self, engine):
        first = await engine.session_log_decision(
            decision="First decision", project_name="proj-max-decisions"
        )
        session_id = first.session_id
        await engine.session_log_decision(
            decision="Second decision",
            session_id=session_id,
            project_name="proj-max-decisions",
        )
        await engine.session_log_decision(
            decision="Third decision",
            session_id=session_id,
            project_name="proj-max-decisions",
        )

        rows = await engine.database.query_decisions_by_session(session_id)
        assert len(rows) == 3  # sanity: all three land in the same session

        result = await engine.session_create_notebook_async(
            session_id=session_id,
            max_decisions=1,
            save_to_file=False,
            save_to_database=False,
        )

        assert result.status == "success"
        assert len(result.notebook.decisions_made) == 1

    async def test_max_decisions_omitted_keeps_default_of_100(self, engine):
        first = await engine.session_log_decision(
            decision="Only decision", project_name="proj-max-decisions-default"
        )
        session_id = first.session_id

        result = await engine.session_create_notebook_async(
            session_id=session_id,
            save_to_file=False,
            save_to_database=False,
        )

        assert result.status == "success"
        assert len(result.notebook.decisions_made) == 1


@pytest.mark.regression
class TestSinceDays:
    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_since_days_excludes_old_decision_keeps_recent(self, engine):
        old = await engine.session_log_decision(
            decision="Old decision, outside window", project_name="proj-since-days"
        )
        session_id = old.session_id
        await engine.session_log_decision(
            decision="Recent decision, inside window",
            session_id=session_id,
            project_name="proj-since-days",
        )
        await _backdate_decision(engine, old.decision_id, days=10)

        result = await engine.session_create_notebook_async(
            session_id=session_id,
            since_days=1,
            save_to_file=False,
            save_to_database=False,
        )

        assert result.status == "success"
        assert "Recent decision, inside window" in result.markdown_output
        assert "Old decision, outside window" not in result.markdown_output

    async def test_since_days_bounds_learnings_section_too(self, engine, tmp_path):
        project_path = str(tmp_path / "since-days-learnings-project")
        anchor = await engine.session_log_decision(
            decision="Anchor decision to create a session",
            project_name="proj-since-days-learnings",
            project_path=project_path,
        )
        session_id = anchor.session_id

        old_learning = await engine.session_log_learning(
            category="pattern",
            learning_content="Old learning, outside window",
            project_name="proj-since-days-learnings",
            project_path=project_path,
        )
        await engine.session_log_learning(
            category="pattern",
            learning_content="Recent learning, inside window",
            project_name="proj-since-days-learnings",
            project_path=project_path,
        )
        await _backdate_learning(engine, old_learning.id, days=10)

        result = await engine.session_create_notebook_async(
            session_id=session_id,
            since_days=1,
            save_to_file=False,
            save_to_database=False,
        )

        assert result.status == "success"
        assert "Recent learning, inside window" in result.markdown_output
        assert "Old learning, outside window" not in result.markdown_output

    async def test_since_days_validates_minimum(self, engine):
        with pytest.raises(ValueError):
            await engine.database.query_decisions_by_session("any-session", since_days=0)
        with pytest.raises(ValueError):
            await engine.database.query_project_learnings("any-project", since_days=-1)


@pytest.mark.regression
class TestExcludeSupersededIsCallerControlled:
    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_exclude_superseded_false_lets_retired_decision_through(self, engine):
        first = await engine.session_log_decision(
            decision="Old, wrong decision", project_name="proj-exclude-superseded"
        )
        session_id = first.session_id
        await engine.session_log_decision(
            decision="Corrected decision",
            session_id=session_id,
            project_name="proj-exclude-superseded",
            supersedes=first.decision_id,
        )

        result = await engine.session_create_notebook_async(
            session_id=session_id,
            exclude_superseded=False,
            save_to_file=False,
            save_to_database=False,
        )

        assert result.status == "success"
        assert "Corrected decision" in result.markdown_output
        assert "Old, wrong decision" in result.markdown_output

    async def test_exclude_superseded_default_true_still_excludes(self, engine):
        """Pins that the default stayed True after promotion to caller-facing."""
        first = await engine.session_log_decision(
            decision="Old, wrong decision",
            project_name="proj-exclude-superseded-default",
        )
        session_id = first.session_id
        await engine.session_log_decision(
            decision="Corrected decision",
            session_id=session_id,
            project_name="proj-exclude-superseded-default",
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


@pytest.mark.regression
class TestAllControlsOmittedReproducesTodaysOutput:
    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_defaults_match_pre_pr_b_behaviour(self, engine):
        """With max_decisions/since_days omitted and exclude_superseded
        defaulted, the notebook rollup is identical to before this issue:
        unbounded count/time window, superseded rows excluded.
        """
        first = await engine.session_log_decision(
            decision="Old, wrong decision", project_name="proj-defaults"
        )
        session_id = first.session_id
        await engine.session_log_decision(
            decision="Corrected decision",
            session_id=session_id,
            project_name="proj-defaults",
            supersedes=first.decision_id,
        )
        await engine.session_log_decision(
            decision="Unrelated third decision",
            session_id=session_id,
            project_name="proj-defaults",
        )

        result = await engine.session_create_notebook_async(
            session_id=session_id,
            save_to_file=False,
            save_to_database=False,
        )

        assert result.status == "success"
        assert "Corrected decision" in result.markdown_output
        assert "Unrelated third decision" in result.markdown_output
        assert "Old, wrong decision" not in result.markdown_output
        assert len(result.notebook.decisions_made) == 2
