"""
Regression tests for issue #77: session_manage_lifecycle's finalize/resume/
validate operations dropped session_id/session_name/project_name/project_path
at the dispatch layer, so they always fell back to ambient in-process state
(`_get_or_create_current_session_id()` or `list(session_cache.keys())[-1]`)
even when a caller explicitly supplied a scope. Because the HTTP transport
builds a SINGLE SessionIntelligenceEngine shared across every project
(http_server.py lifespan()), that ambient state belongs to whichever project
most recently created/touched a session in the process -- not the caller.
This is how a caller scoped to "session-intelligence" finalized a
"hummingbot" session.

Fix: `_manage_lifecycle_impl` now forwards session_id/session_name/
project_name/project_path/allow_unbound into the resume/finalize/validate
branches, and each of `_resume_session`, `_finalize_session`,
`_validate_session` now requires at least one of session_id/session_name/
project_name (or an absolute project_path a project_name can be derived
from), raising SessionContextRequiredError otherwise, unless
allow_unbound=True opts into the legacy fallback.

Every pre-#72 resolver test built a FRESH engine per test, which is exactly
why this bug class (#72, #74, and now #77) kept shipping. These tests reuse
a SINGLE engine across TWO projects, with the *other* project's session
created SECOND so it is the last key in session_cache -- the value the old
buggy code would have picked.
"""

from pathlib import Path

import pytest

from core.session_engine import SessionContextRequiredError, SessionIntelligenceEngine
from models.session_models import SessionStatus
from persistence.sqlite import SQLiteBackend


@pytest.mark.regression
class TestUnboundLifecycleOpsAreRejected:
    """finalize/resume/validate must reject an unscoped call instead of
    silently falling back to ambient state."""

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path), use_filesystem=False)
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_finalize_without_scope_raises(self, engine):
        await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-a"
        )

        with pytest.raises(SessionContextRequiredError) as exc_info:
            await engine.session_manage_lifecycle(operation="finalize")
        assert "finalize" in str(exc_info.value)

    async def test_resume_without_scope_raises(self, engine):
        await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-a"
        )

        with pytest.raises(SessionContextRequiredError) as exc_info:
            await engine.session_manage_lifecycle(operation="resume")
        assert "resume" in str(exc_info.value)

    async def test_validate_without_scope_raises(self, engine):
        await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-a"
        )

        with pytest.raises(SessionContextRequiredError) as exc_info:
            await engine.session_manage_lifecycle(operation="validate")
        assert "validate" in str(exc_info.value)

    async def test_finalize_allow_unbound_escape_hatch_still_works(self, engine):
        create_result = await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-a"
        )
        engine._current_session_id = create_result.session_id

        result = await engine.session_manage_lifecycle(
            operation="finalize", allow_unbound=True
        )
        assert result.status == "success"

    async def test_resume_allow_unbound_escape_hatch_still_works(self, engine):
        create_result = await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-a"
        )
        engine._current_session_id = create_result.session_id

        result = await engine.session_manage_lifecycle(
            operation="resume", allow_unbound=True
        )
        assert result.status == "success"

    async def test_validate_allow_unbound_escape_hatch_still_works(self, engine):
        create_result = await engine.session_manage_lifecycle(
            operation="create", mode="local", project_name="proj-a"
        )
        engine._current_session_id = create_result.session_id

        result = await engine.session_manage_lifecycle(
            operation="validate", allow_unbound=True
        )
        assert result.status in {"success", "warning"}


@pytest.mark.regression
class TestSharedEngineDoesNotBleedAcrossProjects:
    """A single engine shared by two projects, with the OTHER project's
    session created SECOND so it sits last in session_cache -- the value the
    pre-#77 ambient fallback would have picked."""

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path), use_filesystem=False)
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def _create_two_projects(self, engine, tmp_path: Path):
        proj_a_dir = tmp_path / "proj-a"
        proj_a_dir.mkdir()
        proj_b_dir = tmp_path / "proj-b"
        proj_b_dir.mkdir()

        proj_a_create = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-a",
            project_path=str(proj_a_dir),
        )
        # proj-b is created SECOND: it is the last key in session_cache when
        # the caller's finalize/resume/validate call below is made.
        proj_b_create = await engine.session_manage_lifecycle(
            operation="create",
            mode="local",
            project_name="proj-b",
            project_path=str(proj_b_dir),
        )

        cache_keys = list(engine.session_cache.keys())
        assert cache_keys[-1] == proj_b_create.session_id, (
            "Precondition failed: proj-b's session is not the last key in "
            "session_cache, so this test would not actually exercise the "
            "#77 ambient-fallback bug even if the fix were reverted."
        )
        return proj_a_create, proj_b_create

    async def test_finalize_binds_to_caller_project_not_last_cache_key(
        self, engine, tmp_path
    ):
        proj_a_create, proj_b_create = await self._create_two_projects(engine, tmp_path)

        result = await engine.session_manage_lifecycle(
            operation="finalize", project_name="proj-a"
        )

        assert result.status == "success"
        assert result.session_id == proj_a_create.session_id, (
            "Issue #77: finalize resolved to the wrong session. It must "
            "bind to proj-a's session (the caller's project_name), not fall "
            "back to whichever project last created a session."
        )
        assert result.session_id != proj_b_create.session_id

        # proj-a's session is finalized and dropped from the cache...
        assert proj_a_create.session_id not in engine.session_cache
        # ...while proj-b's untouched session remains active in the cache.
        assert proj_b_create.session_id in engine.session_cache
        assert (
            engine.session_cache[proj_b_create.session_id].status
            == SessionStatus.ACTIVE
        )

    async def test_finalize_binds_by_absolute_project_path(self, engine, tmp_path):
        proj_a_create, proj_b_create = await self._create_two_projects(engine, tmp_path)

        result = await engine.session_manage_lifecycle(
            operation="finalize", project_path=str(tmp_path / "proj-a")
        )

        assert result.status == "success"
        assert result.session_id == proj_a_create.session_id
        assert result.session_id != proj_b_create.session_id

    async def test_finalize_explicit_session_id_wins(self, engine, tmp_path):
        proj_a_create, proj_b_create = await self._create_two_projects(engine, tmp_path)

        result = await engine.session_manage_lifecycle(
            operation="finalize", session_id=proj_a_create.session_id
        )

        assert result.status == "success"
        assert result.session_id == proj_a_create.session_id
        assert result.session_id != proj_b_create.session_id

    async def test_resume_binds_to_caller_project_not_last_cache_key(
        self, engine, tmp_path
    ):
        proj_a_create, proj_b_create = await self._create_two_projects(engine, tmp_path)

        result = await engine.session_manage_lifecycle(
            operation="resume", project_name="proj-a"
        )

        assert result.status == "success"
        assert result.session_id == proj_a_create.session_id, (
            "Issue #77: resume resolved to the wrong session. It must bind "
            "to proj-a's session, not fall back to the last-created session."
        )
        assert result.session_id != proj_b_create.session_id
        assert engine._current_session_id == proj_a_create.session_id

    async def test_validate_binds_to_caller_project_not_last_cache_key(
        self, engine, tmp_path
    ):
        proj_a_create, proj_b_create = await self._create_two_projects(engine, tmp_path)

        result = await engine.session_manage_lifecycle(
            operation="validate", project_name="proj-a"
        )

        assert result.status in {"success", "warning"}
        assert result.session_id == proj_a_create.session_id, (
            "Issue #77: validate resolved to the wrong session. It must "
            "bind to proj-a's session, not fall back to the last-created "
            "session."
        )
        assert result.session_id != proj_b_create.session_id
