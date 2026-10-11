"""
Regression tests for issue #220: reject learnings that are only a tool_use id.

Bug: the agent-stop hook logged "Agent X encountered: toolu_..." as error_fix
     learnings; a bare tool_use id is not knowledge.
Fix: session_log_learning rejects content that fully matches a toolu id,
     optionally behind the hook template. Prose mentioning an id is accepted.
"""

import httpx
import pytest

from core.session_engine import (
    InvalidEntryContentError,
    SessionIntelligenceEngine,
)
from persistence.sqlite import SQLiteBackend

TOOLU = "toolu_01A09q90qw90lq917835lq9"


@pytest.mark.regression
class TestToolUseIdLearningRejected:
    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_rejects_bare_id(self, engine):
        with pytest.raises(InvalidEntryContentError):
            await engine.session_log_learning(
                category="error_fix", learning_content=f"  {TOOLU}\n", project_name="proj-a"
            )

    async def test_rejects_hook_template(self, engine):
        with pytest.raises(InvalidEntryContentError):
            await engine.session_log_learning(
                category="error_fix",
                learning_content=f"Agent focused-code-modifier encountered: {TOOLU}",
                project_name="proj-a",
            )

    async def test_accepts_prose_mentioning_id(self, engine):
        result = await engine.session_log_learning(
            category="error_fix",
            learning_content=f"The failure in {TOOLU} came from a stale worktree; rebase first.",
            project_name="proj-a",
        )
        assert result is not None

    async def test_nothing_persisted_on_rejection(self, engine, monkeypatch):
        saved = []

        async def spy(*args, **kwargs):
            saved.append((args, kwargs))

        monkeypatch.setattr(engine.database, "save_project_learning", spy)
        with pytest.raises(InvalidEntryContentError):
            await engine.session_log_learning(
                category="error_fix", learning_content=TOOLU, project_name="proj-a"
            )
        assert saved == []


@pytest.fixture
async def rest_app(tmp_path):
    from persistence import DatabaseConfig
    from transport.http_server import HTTPSessionIntelligenceServer
    from transport.security import SecurityConfig

    db = SQLiteBackend(str(tmp_path / "rest.db"))
    await db.initialize()
    server = HTTPSessionIntelligenceServer(
        host="127.0.0.1",
        port=4099,
        repository_path=str(tmp_path),
        db_config=DatabaseConfig(),
        security_config=SecurityConfig(
            localhost_only=False, allowed_origins=["*"], require_api_key=False
        ),
    )
    app = server.create_app()
    app.state.database = db
    yield app
    await db.close()


@pytest.mark.regression
class TestRestRouteToolUseIdGuard:
    async def test_rest_rejects_hook_template(self, rest_app, monkeypatch):
        saved = []

        async def spy(*args, **kwargs):
            saved.append(kwargs)

        monkeypatch.setattr(rest_app.state.database, "save_project_learning", spy)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=rest_app), base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/tools/session_log_learning",
                json={
                    "category": "error_fix",
                    "learning_content": f"Agent focused-code-modifier encountered: {TOOLU}",
                    "project_path": "/tmp/proj-a",
                },
            )
        assert resp.status_code == 400
        assert saved == []

    async def test_rest_accepts_normal_content(self, rest_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=rest_app), base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/tools/session_log_learning",
                json={
                    "category": "error_fix",
                    "learning_content": "Rebase the worktree before running the suite.",
                    "project_path": "/tmp/proj-a",
                },
            )
        assert resp.status_code == 200
