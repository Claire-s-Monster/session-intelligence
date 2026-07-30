"""
Regression tests for issue #41: Majority of agent_executions have
unresolved agent_type (raw hex ID instead of agent name).

https://github.com/Claire-s-Monster/session-intelligence/issues/41

Root cause: `_track_execution_sync` derived `agent_type` solely from
`step_data.get("agent_type", "unknown")` at execution-creation time. The
SubagentStop hook can report `agent_type` as `""` (a present-but-falsy
key, not a missing one) due to a Claude Code harness quirk, and
`.get(key, default)` does not substitute the default for a present
falsy value. That empty string then persisted verbatim, and the
read-side fallback in `get_agent_stats` (`agent_type or agent_name or
"unknown"`) substituted the raw hex agent_name in its place.

Verifies the fix:
  `SessionIntelligenceEngine` now maintains an in-memory
  `_agent_type_cache: dict[str, str]` keyed by `agent_name`. Whenever a
  call reports a real (non-"unknown", non-falsy) `agent_type`, it is
  cached. Whenever a call reports a falsy/"unknown" `agent_type`, the
  cached value (if any) is preferred over persisting the falsy value.
  On `phase="agent_stop"` calls that reuse an existing RUNNING
  `AgentExecution`, a resolved real value backfills
  `AgentExecution.agent_type` if the existing stored value is currently
  falsy or "unknown" (never overwriting a real value with a worse one).
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from persistence.sqlite import SQLiteBackend

AGENT_NAME = "a9116be028f1d04d3"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    """SessionIntelligenceEngine wired to a fresh in-process SQLite backend.

    Filesystem persistence is OFF - in-memory AgentExecution agent_type
    assertions are the focus of the issue #41 fix verification.
    """
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    db = SQLiteBackend(db_path=str(tmp_path / "issue41.db"))
    await db.initialize()

    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


# ---------------------------------------------------------------------------
# Issue #41: agent_type resolution via in-memory cache
# ---------------------------------------------------------------------------


@pytest.mark.regression
async def test_agent_stop_empty_agent_type_backfilled_from_cache(engine):
    """A Start-phase call reporting a real agent_type, followed by a
    Stop-phase call for the SAME agent_name reporting agent_type="" (the
    harness quirk), must not persist the empty string. The cached real
    value from the Start call must backfill the existing AgentExecution."""
    start_result = engine.session_track_execution(
        session_id=None,
        agent_name=AGENT_NAME,
        step_data={
            "phase": "agent_start",
            "agent_type": "focused-code-modifier",
            "operation": "start",
            "description": "SubagentStart hook",
        },
    )
    assert start_result.status == "success"
    session_id = start_result.session_id

    session = engine.session_cache[session_id]
    agent_execution = next(
        a for a in session.agents_executed if a.agent_name == AGENT_NAME
    )
    assert agent_execution.agent_type == "focused-code-modifier"

    stop_result = engine.session_track_execution(
        session_id=session_id,
        agent_name=AGENT_NAME,
        step_data={
            "phase": "agent_stop",
            "agent_type": "",
            "success": True,
        },
    )
    assert stop_result.status == "success"

    agent_execution = next(
        a for a in session.agents_executed if a.agent_name == AGENT_NAME
    )
    assert agent_execution.agent_type == "focused-code-modifier", (
        "AgentExecution.agent_type was overwritten with an empty string "
        "reported by the SubagentStop hook instead of being backfilled "
        "from the in-memory cache. This is issue #41."
    )


@pytest.mark.regression
async def test_no_prior_cache_falls_back_to_unknown(engine):
    """When no cache entry exists yet for a given agent_name and
    step_data reports an empty/missing agent_type, the result must still
    fall back to "unknown" - no silent substitution of a wrong value,
    and no regression of existing pre-#41 behavior."""
    result = engine.session_track_execution(
        session_id=None,
        agent_name="brand-new-agent-hexid",
        step_data={
            "phase": "agent_start",
            "operation": "start",
            "description": "SubagentStart hook",
        },
    )
    assert result.status == "success"
    session_id = result.session_id

    session = engine.session_cache[session_id]
    agent_execution = next(
        a
        for a in session.agents_executed
        if a.agent_name == "brand-new-agent-hexid"
    )
    assert agent_execution.agent_type == "unknown"


@pytest.mark.regression
async def test_later_real_agent_type_updates_stale_cached_value(engine):
    """If the same agent_name is later reused (a fresh, unrelated
    invocation) with a genuinely different real agent_type on its own
    Start-equivalent call, the cache must update to the newer real
    value - last-real-value-wins, not stuck on the first-seen value."""
    first_start = engine.session_track_execution(
        session_id=None,
        agent_name=AGENT_NAME,
        step_data={
            "phase": "agent_start",
            "agent_type": "focused-code-modifier",
            "operation": "start",
            "description": "first invocation",
        },
    )
    session_id = first_start.session_id
    session = engine.session_cache[session_id]

    first_stop = engine.session_track_execution(
        session_id=session_id,
        agent_name=AGENT_NAME,
        step_data={"phase": "agent_stop", "agent_type": "", "success": True},
    )
    assert first_stop.status == "success"

    # A later, unrelated invocation of the same agent_name (e.g. the
    # subagent was re-dispatched under a different role) reports a
    # genuinely different real agent_type.
    second_start = engine.session_track_execution(
        session_id=session_id,
        agent_name=AGENT_NAME,
        step_data={
            "phase": "agent_start",
            "agent_type": "focused-quality-resolver",
            "operation": "start",
            "description": "second invocation",
        },
    )
    assert second_start.status == "success"

    second_execution = next(
        a
        for a in session.agents_executed
        if a.agent_name == AGENT_NAME
        and a.agent_type == "focused-quality-resolver"
    )
    assert second_execution.agent_type == "focused-quality-resolver"

    second_stop = engine.session_track_execution(
        session_id=session_id,
        agent_name=AGENT_NAME,
        step_data={"phase": "agent_stop", "agent_type": "", "success": True},
    )
    assert second_stop.status == "success"

    assert engine._agent_type_cache[AGENT_NAME] == "focused-quality-resolver"
    assert second_execution.agent_type == "focused-quality-resolver", (
        "Stale first-seen cached agent_type leaked into a later, "
        "unrelated invocation's backfill."
    )
