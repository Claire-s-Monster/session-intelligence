"""
Regression tests for issue #88: reject tool-result envelopes at insert.

Bug: session_log_decision and session_log_learning stored whatever content
     they were given, including captured tool-result envelopes like
     "Tool 'Write' failed: ..." pasted verbatim from a failed tool call. That
     kind of content is a raw log line, not durable knowledge, and pollutes
     the decisions/learnings tables.
Fix: both session-tier log methods now call a shared guard,
     _reject_tool_result_envelope(), before any persistence. It raises
     InvalidEntryContentError when the content matches the pattern
     "Tool '<name>' failed|error ...", instructing the caller to log what the
     failure taught them instead of the raw envelope.

Scope note: issue #88 also proposed rejecting content that is >80% fenced
code block with no prose. That heuristic is deliberately NOT implemented
here -- it has a real false-positive risk for legitimate learnings that are
mostly a code snippet plus a short explanation -- and is deferred to issue
#89 as a warning, not a hard rejection.
"""

import pytest

from core.session_engine import (
    InvalidEntryContentError,
    SessionIntelligenceEngine,
)
from persistence.sqlite import SQLiteBackend


@pytest.mark.regression
class TestToolResultEnvelopeRejected:

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_session_log_learning_rejects_real_poison_string(self, engine):
        """The actual envelope that motivated issue #88."""
        poison = (
            "Tool 'Write' failed: \"\"\"\n"
            "Regression tests for issue #82: sessions and agent_executions gain a\n"
            "last_seen_at heartbeat col"
        )

        with pytest.raises(InvalidEntryContentError):
            await engine.session_log_learning(
                category="error_fix",
                learning_content=poison,
                project_name="proj-a",
            )

    async def test_session_log_decision_rejects_tool_result_envelope(self, engine):
        with pytest.raises(InvalidEntryContentError):
            await engine.session_log_decision(
                decision="Tool 'Bash' failed: exit code 1",
                project_name="proj-a",
            )

    # Varies the payload after the colon (a shebang line), NOT the prefix, so
    # it does not constitute Bash-prefix coverage -- that is what
    # test_rejects_bash_command_prefix below covers.
    async def test_rejects_tool_prefix_with_shebang_payload(self, engine):
        with pytest.raises(InvalidEntryContentError):
            await engine.session_log_learning(
                category="error_fix",
                learning_content="Tool 'Write' failed: #!/usr/bin/env bash",
                project_name="proj-a",
            )

    async def test_rejects_bash_command_prefix(self, engine):
        with pytest.raises(InvalidEntryContentError):
            await engine.session_log_learning(
                category="error_fix",
                learning_content="Bash command 'pixi' failed: exit 1",
                project_name="proj-a",
            )

    async def test_rejects_mcp_tool_prefix(self, engine):
        with pytest.raises(InvalidEntryContentError):
            await engine.session_log_learning(
                category="error_fix",
                learning_content="MCP tool 'git_status' failed: boom",
                project_name="proj-a",
            )

    async def test_session_log_decision_rejects_bash_command_prefix(self, engine):
        with pytest.raises(InvalidEntryContentError):
            await engine.session_log_decision(
                decision="Bash command 'make' failed: exit 2",
                project_name="proj-a",
            )

    async def test_rejects_new_prefixes_case_insensitive(self, engine):
        with pytest.raises(InvalidEntryContentError):
            await engine.session_log_learning(
                category="error_fix",
                learning_content="bash command 'x' Failed: y",
                project_name="proj-a",
            )

    async def test_rejects_with_leading_whitespace(self, engine):
        with pytest.raises(InvalidEntryContentError):
            await engine.session_log_learning(
                category="error_fix",
                learning_content="   Tool 'Read' failed: file not found",
                project_name="proj-a",
            )

    async def test_rejects_error_variant_case_insensitive(self, engine):
        with pytest.raises(InvalidEntryContentError):
            await engine.session_log_learning(
                category="error_fix",
                learning_content="Tool 'Bash' ERROR: command not found",
                project_name="proj-a",
            )

    async def test_raised_type_and_message_names_the_fix(self, engine):
        with pytest.raises(InvalidEntryContentError) as excinfo:
            await engine.session_log_decision(
                decision="Tool 'Edit' failed: string not found",
                project_name="proj-a",
            )

        message = str(excinfo.value)
        assert "captured tool result" in message
        assert "log what the" in message.lower()
        assert "failure taught" in message.lower()

    async def test_none_learning_content_does_not_raise_type_error(self, engine):
        """The guard must not itself crash on None; whatever downstream
        validation does with a None learning_content is a separate concern
        and is not asserted here."""
        try:
            await engine.session_log_learning(
                category="error_fix",
                learning_content=None,
                project_name="proj-a",
            )
        except InvalidEntryContentError:
            pytest.fail(
                "Guard incorrectly raised InvalidEntryContentError for None content."
            )
        except TypeError as e:
            pytest.fail(f"Guard must not raise TypeError for None content: {e}")
        except Exception:
            pass  # Any other downstream behaviour is acceptable and unasserted.

    async def test_none_decision_does_not_raise_type_error(self, engine):
        """Same as above, for the session_log_decision path."""
        try:
            await engine.session_log_decision(
                decision=None,
                project_name="proj-a",
            )
        except InvalidEntryContentError:
            pytest.fail(
                "Guard incorrectly raised InvalidEntryContentError for None content."
            )
        except TypeError as e:
            pytest.fail(f"Guard must not raise TypeError for None content: {e}")
        except Exception:
            pass  # Any other downstream behaviour is acceptable and unasserted.


@pytest.mark.regression
class TestLegitimateContentIsNotRejected:

    @pytest.fixture
    async def engine(self, tmp_path):
        eng = SessionIntelligenceEngine(repository_path=str(tmp_path))
        eng.database = SQLiteBackend(str(tmp_path / "test.db"))
        await eng.database.initialize()
        yield eng
        await eng.database.close()

    async def test_prose_merely_mentioning_the_word_tool_is_not_rejected(self, engine):
        """The guard matches only content that STARTS with the envelope
        shape. Prose that discusses a tool failure in the middle of a
        sentence must not be rejected."""
        content = (
            "The Tool 'Write' failed intermittently under load in the past; "
            "the fix was to add a retry with exponential backoff around the "
            "write call."
        )

        result = await engine.session_log_learning(
            category="error_fix",
            learning_content=content,
            project_name="proj-a",
        )
        assert result is not None
        assert result.learning is not None

    async def test_code_heavy_learning_is_not_rejected(self, engine):
        """Scope guard: issue #88's fenced-code-block heuristic is
        deliberately NOT implemented as a rejection (deferred to #89 as a
        warning). A legitimate learning that is mostly a code snippet plus
        a one-line explanation must still be accepted."""
        content = (
            "Fix for the heartbeat migration:\n"
            "```python\n"
            "def add_last_seen_at_column(conn):\n"
            "    conn.execute(\n"
            "        'ALTER TABLE sessions ADD COLUMN last_seen_at TIMESTAMPTZ'\n"
            "    )\n"
            "    conn.execute(\n"
            "        'ALTER TABLE agent_executions ADD COLUMN last_seen_at TIMESTAMPTZ'\n"
            "    )\n"
            "```\n"
        )

        result = await engine.session_log_learning(
            category="pattern",
            learning_content=content,
            project_name="proj-a",
        )
        assert result is not None
        assert result.learning is not None

    async def test_lowercase_the_tool_prefix_is_not_rejected(self, engine):
        content = (
            "the tool 'foo' failed because the path was wrong; fix is to "
            "quote it"
        )
        result = await engine.session_log_learning(
            category="error_fix",
            learning_content=content,
            project_name="proj-a",
        )
        assert result is not None
        assert result.learning is not None

    async def test_this_tool_prefix_is_not_rejected(self, engine):
        content = "This tool 'ruff' failed on us until we pinned it"
        result = await engine.session_log_learning(
            category="error_fix",
            learning_content=content,
            project_name="proj-a",
        )
        assert result is not None
        assert result.learning is not None

    async def test_a_command_prefix_is_not_rejected(self, engine):
        content = "a command 'make' failed intermittently -- root cause was a race"
        result = await engine.session_log_learning(
            category="error_fix",
            learning_content=content,
            project_name="proj-a",
        )
        assert result is not None
        assert result.learning is not None
