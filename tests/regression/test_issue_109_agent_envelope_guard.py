"""
Regression tests for issue #109: extend the tool-result envelope guard from
issue #88 to the agent-tier write tools.

Bug: issue #88 added `_reject_tool_result_envelope()` to session_log_decision
     and session_log_learning, but the agent-tier twins -- agent_log_decision,
     agent_log_learning, and agent_create_notebook -- had no such guard, so a
     captured tool-result envelope like "Tool 'Write' failed: ..." pasted
     verbatim into an agent's decision/learning/notebook body would be
     persisted unmodified.
Fix: all three agent-tier methods now call the same shared guard before any
     persistence or agent-stats update, mirroring the session tier's
     placement (before the try block, so InvalidEntryContentError propagates
     unmodified rather than being converted into a Result(status="error")
     object by the method's own broad except Exception clause).
"""

import pytest

from core.session_engine import (
    InvalidEntryContentError,
    SessionIntelligenceEngine,
)
from persistence.sqlite import SQLiteBackend

POISON = "Tool 'Bash' failed: exit code 1"
LEGITIMATE = "Prefer asyncpg pools sized to worker count"


async def _register(engine, name="test-agent", agent_type="domain"):
    """Register a synthetic agent. Mirrors tests/engine/test_agent_operations.py."""
    return await engine.agent_register(agent_name=name, agent_type=agent_type)


@pytest.mark.regression
class TestAgentEnvelopeGuardRejectsPoison:
    @pytest.fixture
    async def engine(self, tmp_path, monkeypatch):
        """Mirrors tests/engine/conftest.py's `engine` fixture: agent-name
        validation is disabled via env var (so "test-agent" does not need to
        exist under ~/.claude/agents/), filesystem persistence is disabled,
        and DB teardown runs in `finally` so a failing test in this fixture's
        body (including agent_register itself) cannot leak the SQLite
        connection or the process."""
        monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        eng = SessionIntelligenceEngine(
            repository_path=str(tmp_path),
            use_filesystem=False,
            database=db,
        )
        try:
            yield eng
        finally:
            await db.close()

    async def test_agent_log_decision_rejects_tool_result_envelope(self, engine):
        await _register(engine)
        with pytest.raises(InvalidEntryContentError) as excinfo:
            await engine.agent_log_decision(
                agent_name="test-agent",
                decision_type="implementation",
                context="ctx",
                decision=POISON,
            )
        message = str(excinfo.value)
        assert "agent_log_decision" in message
        assert "decision" in message

    async def test_agent_log_decision_rejected_content_not_persisted(self, engine):
        await _register(engine)
        with pytest.raises(InvalidEntryContentError):
            await engine.agent_log_decision(
                agent_name="test-agent",
                decision_type="implementation",
                context="ctx",
                decision=POISON,
            )

        decisions = await engine.agent_query_decisions("test-agent")
        assert not any(d.decision == POISON for d in decisions)

    async def test_agent_log_learning_rejects_tool_result_envelope(self, engine):
        await _register(engine)
        with pytest.raises(InvalidEntryContentError) as excinfo:
            await engine.agent_log_learning(
                agent_name="test-agent",
                learning_type="error_fix",
                title="Bad title",
                content=POISON,
            )
        message = str(excinfo.value)
        assert "agent_log_learning" in message
        assert "content" in message

    async def test_agent_log_learning_rejected_content_not_persisted(self, engine):
        await _register(engine)
        with pytest.raises(InvalidEntryContentError):
            await engine.agent_log_learning(
                agent_name="test-agent",
                learning_type="error_fix",
                title="Bad title",
                content=POISON,
            )

        learnings = await engine.agent_query_learnings("test-agent")
        assert not any(POISON in la.content for la in learnings)

    async def test_agent_create_notebook_rejects_tool_result_envelope(self, engine):
        await _register(engine)
        with pytest.raises(InvalidEntryContentError) as excinfo:
            await engine.agent_create_notebook(
                agent_name="test-agent",
                title="Bad notebook",
                content=POISON,
            )
        message = str(excinfo.value)
        assert "agent_create_notebook" in message
        assert "content" in message

    async def test_agent_create_notebook_rejected_content_not_persisted(self, engine):
        await _register(engine)
        with pytest.raises(InvalidEntryContentError):
            await engine.agent_create_notebook(
                agent_name="test-agent",
                title="Bad notebook",
                content=POISON,
            )

        notebooks = await engine.agent_query_notebooks("test-agent")
        assert not any(POISON in nb.content for nb in notebooks)


@pytest.mark.regression
class TestAgentEnvelopeGuardAllowsLegitimateContent:
    @pytest.fixture
    async def engine(self, tmp_path, monkeypatch):
        """Same pattern as TestAgentEnvelopeGuardRejectsPoison.engine above."""
        monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

        db = SQLiteBackend(str(tmp_path / "test.db"))
        await db.initialize()
        eng = SessionIntelligenceEngine(
            repository_path=str(tmp_path),
            use_filesystem=False,
            database=db,
        )
        try:
            yield eng
        finally:
            await db.close()

    async def test_agent_log_decision_accepts_normal_content(self, engine):
        await _register(engine)
        result = await engine.agent_log_decision(
            agent_name="test-agent",
            decision_type="implementation",
            context="Choosing a pool size",
            decision=LEGITIMATE,
        )
        assert result.status == "success"

        decisions = await engine.agent_query_decisions("test-agent")
        assert any(d.decision == LEGITIMATE for d in decisions)

    async def test_agent_log_learning_accepts_normal_content(self, engine):
        await _register(engine)
        result = await engine.agent_log_learning(
            agent_name="test-agent",
            learning_type="pattern",
            title="DB pooling",
            content=LEGITIMATE,
        )
        assert result.status == "success"

        learnings = await engine.agent_query_learnings("test-agent")
        assert any(LEGITIMATE in la.content for la in learnings)

    async def test_agent_create_notebook_accepts_normal_content(self, engine):
        await _register(engine)
        result = await engine.agent_create_notebook(
            agent_name="test-agent",
            title="Pooling notes",
            content=LEGITIMATE,
        )
        assert result.status == "success"

        notebooks = await engine.agent_query_notebooks("test-agent")
        assert any(LEGITIMATE in nb.content for nb in notebooks)
