"""Tests for AgentValidator wiring inside SessionIntelligenceEngine.

Uses an in-memory SQLite backend and a tmp_path agents directory so tests
are fully isolated from real filesystem state and PostgreSQL.
"""

import logging
from pathlib import Path

import pytest

from core.agent_validator import AgentNotFoundError
from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_agent(base: Path, subdir: str, stem: str, frontmatter: str = "") -> Path:
    """Write a minimal agent .md file under base/subdir/stem.md."""
    target_dir = base / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    content = (
        f"---\n{frontmatter}\n---\n# Agent\nBody.\n"
        if frontmatter
        else "# Agent\nBody.\n"
    )
    path = target_dir / f"{stem}.md"
    path.write_text(content, encoding="utf-8")
    return path


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
def agents_dir(tmp_path: Path) -> Path:
    """Isolated agents root directory."""
    return tmp_path / "agents"


@pytest.fixture
def engine(db, agents_dir, monkeypatch: pytest.MonkeyPatch) -> SessionIntelligenceEngine:
    """Engine wired to in-memory SQLite and isolated agents_dir in strict mode."""
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENTS_DIR", str(agents_dir))
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "strict")
    eng = SessionIntelligenceEngine(
        repository_path=None,
        use_filesystem=False,
        database=db,
    )
    return eng


# ---------------------------------------------------------------------------
# 1. agent_register overrides agent_type from filesystem
# ---------------------------------------------------------------------------


class TestAgentRegisterValidatesAndOverridesType:
    async def test_type_from_frontmatter_wins(
        self,
        engine: SessionIntelligenceEngine,
        agents_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _write_agent(
            agents_dir,
            "meta",
            "test-agent",
            "name: test-agent\ndescription: Test agent for wiring tests",
        )
        # Reinitialise validator so it picks up the newly written file
        from core.agent_validator import AgentValidator

        engine._agent_validator = AgentValidator(agents_root=agents_dir, mode="strict")

        with caplog.at_level(logging.WARNING):
            result = await engine.agent_register(
                agent_name="test-agent",
                agent_type="domain",  # Wrong — file lives under meta/
                description="caller desc",
                display_name=None,
                metadata={},
                capabilities=[],
            )

        assert result.status in ("created", "updated")
        # Verify persisted record has filesystem type
        agent_data = await engine.database.get_agent_by_name("test-agent")
        assert agent_data is not None
        assert agent_data["agent_type"] == "meta"
        # Warning about type override should have been logged
        assert any("meta" in r.message or "domain" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 2. Frontmatter description used when caller omits it
# ---------------------------------------------------------------------------


class TestAgentRegisterUsesFrontmatterDescription:
    async def test_frontmatter_description_as_fallback(
        self,
        engine: SessionIntelligenceEngine,
        agents_dir: Path,
    ) -> None:
        _write_agent(
            agents_dir,
            "domain",
            "desc-agent",
            "name: desc-agent\ndescription: Frontmatter description",
        )
        from core.agent_validator import AgentValidator

        engine._agent_validator = AgentValidator(agents_root=agents_dir, mode="strict")

        result = await engine.agent_register(
            agent_name="desc-agent",
            agent_type="domain",
            description=None,
            display_name=None,
            metadata={},
            capabilities=[],
        )

        assert result.status in ("created", "updated")
        agent_data = await engine.database.get_agent_by_name("desc-agent")
        assert agent_data is not None
        assert agent_data["description"] == "Frontmatter description"


# ---------------------------------------------------------------------------
# 3. Caller description wins over frontmatter when provided
# ---------------------------------------------------------------------------


class TestAgentRegisterKeepsCallerDescription:
    async def test_caller_description_takes_precedence(
        self,
        engine: SessionIntelligenceEngine,
        agents_dir: Path,
    ) -> None:
        _write_agent(
            agents_dir,
            "domain",
            "prio-agent",
            "name: prio-agent\ndescription: Frontmatter description",
        )
        from core.agent_validator import AgentValidator

        engine._agent_validator = AgentValidator(agents_root=agents_dir, mode="strict")

        result = await engine.agent_register(
            agent_name="prio-agent",
            agent_type="domain",
            description="caller's text",
            display_name=None,
            metadata={},
            capabilities=[],
        )

        assert result.status in ("created", "updated")
        agent_data = await engine.database.get_agent_by_name("prio-agent")
        assert agent_data is not None
        assert agent_data["description"] == "caller's text"


# ---------------------------------------------------------------------------
# 4. agent_register raises on unknown agent in strict mode
# ---------------------------------------------------------------------------


class TestAgentRegisterRaisesOnUnknown:
    async def test_raises_agent_not_found(
        self,
        engine: SessionIntelligenceEngine,
        agents_dir: Path,
    ) -> None:
        agents_dir.mkdir(parents=True, exist_ok=True)
        from core.agent_validator import AgentValidator

        engine._agent_validator = AgentValidator(agents_root=agents_dir, mode="strict")

        with pytest.raises(AgentNotFoundError):
            await engine.agent_register(
                agent_name="ghost-agent",
                agent_type="domain",
                description=None,
                display_name=None,
                metadata={},
                capabilities=[],
            )


# ---------------------------------------------------------------------------
# 5. agent_log_decision raises on unknown agent
# ---------------------------------------------------------------------------


class TestAgentLogDecisionRaisesOnUnknown:
    async def test_raises_agent_not_found(
        self,
        engine: SessionIntelligenceEngine,
        agents_dir: Path,
    ) -> None:
        agents_dir.mkdir(parents=True, exist_ok=True)
        from core.agent_validator import AgentValidator

        engine._agent_validator = AgentValidator(agents_root=agents_dir, mode="strict")

        with pytest.raises(AgentNotFoundError):
            await engine.agent_log_decision(
                agent_name="ghost-agent",
                decision_type="implementation",
                context="some context",
                decision="some decision",
            )


# ---------------------------------------------------------------------------
# 6. agent_log_learning raises on unknown agent
# ---------------------------------------------------------------------------


class TestAgentLogLearningRaisesOnUnknown:
    async def test_raises_agent_not_found(
        self,
        engine: SessionIntelligenceEngine,
        agents_dir: Path,
    ) -> None:
        agents_dir.mkdir(parents=True, exist_ok=True)
        from core.agent_validator import AgentValidator

        engine._agent_validator = AgentValidator(agents_root=agents_dir, mode="strict")

        with pytest.raises(AgentNotFoundError):
            await engine.agent_log_learning(
                agent_name="ghost-agent",
                learning_type="pattern",
                title="some learning",
                content="some content",
            )


# ---------------------------------------------------------------------------
# 7. agent_create_notebook raises on unknown agent
# ---------------------------------------------------------------------------


class TestAgentCreateNotebookRaisesOnUnknown:
    async def test_raises_agent_not_found(
        self,
        engine: SessionIntelligenceEngine,
        agents_dir: Path,
    ) -> None:
        agents_dir.mkdir(parents=True, exist_ok=True)
        from core.agent_validator import AgentValidator

        engine._agent_validator = AgentValidator(agents_root=agents_dir, mode="strict")

        with pytest.raises(AgentNotFoundError):
            await engine.agent_create_notebook(
                agent_name="ghost-agent",
                title="My Notebook",
                content="Notebook content.",
            )


# ---------------------------------------------------------------------------
# 8. agent_get_info does NOT validate (read-only path)
# ---------------------------------------------------------------------------


class TestAgentGetInfoDoesNotValidate:
    async def test_returns_none_without_raising(
        self,
        engine: SessionIntelligenceEngine,
        agents_dir: Path,
    ) -> None:
        agents_dir.mkdir(parents=True, exist_ok=True)
        from core.agent_validator import AgentValidator

        engine._agent_validator = AgentValidator(agents_root=agents_dir, mode="strict")

        # Should not raise even though "ghost-agent" doesn't exist anywhere
        result = await engine.agent_get_info("ghost-agent")
        assert result is None


# ---------------------------------------------------------------------------
# 9. agent_query_decisions does NOT validate (read-only path)
# ---------------------------------------------------------------------------


class TestAgentQueryDecisionsDoesNotValidate:
    async def test_returns_empty_without_raising(
        self,
        engine: SessionIntelligenceEngine,
        agents_dir: Path,
    ) -> None:
        agents_dir.mkdir(parents=True, exist_ok=True)
        from core.agent_validator import AgentValidator

        engine._agent_validator = AgentValidator(agents_root=agents_dir, mode="strict")

        result = await engine.agent_query_decisions(agent_name="ghost-agent")
        assert result == []
