"""
Regression tests for issue #184: an explicit session_id absent from the
session cache (service restart, or eviction by _finalize_session) must be
hydrated from the database, not replaced by a blank auto-created session.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import ExecutionStatus, SessionStatus
from persistence.sqlite import SQLiteBackend

AGENT_TYPE = "micro-survey"


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(db_path=str(tmp_path / "issue184.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
def make_engine(tmp_path, monkeypatch, db):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")

    def _make() -> SessionIntelligenceEngine:
        return SessionIntelligenceEngine(
            repository_path=str(tmp_path),
            use_filesystem=False,
            database=db,
        )

    return _make


async def _track(engine, session_id, agent_name, phase, success="absent", agent_type=AGENT_TYPE):
    step_data = {"phase": phase, "agent_type": agent_type}
    if success != "absent":
        step_data["success"] = success
    return await engine.session_track_execution(
        session_id=session_id,
        agent_name=agent_name,
        step_data=step_data,
        allow_unbound=True,
    )


async def _persist(engine, db, session_id):
    """Mirror the HTTP post-call sweep: write the session and its executions."""
    session = engine.session_cache[session_id]
    await db.save_session(session.model_dump())
    for agent_exec in session.agents_executed:
        data = agent_exec.model_dump()
        data["session_id"] = session_id
        await db.save_agent_execution(data)


async def _seed_session(make_engine, db, *agent_steps):
    """Create a session with tags via a first engine, persist it, return
    (session_id, original_session) and a fresh 'restarted' engine."""
    first = make_engine()
    sid = (await _track(first, None, "seed-agent", "agent_start")).session_id
    for name, phase, success in agent_steps:
        await _track(first, sid, name, phase, success)
    first.session_cache[sid].metadata.tags = ["original-tag"]
    await _persist(first, db, sid)
    original = first.session_cache[sid]
    return sid, original, make_engine()


@pytest.mark.regression
async def test_start_after_restart_preserves_session(make_engine, db):
    sid, original, restarted = await _seed_session(
        make_engine, db, ("agent-a", "agent_start", "absent"), ("agent-a", "agent_stop", True)
    )
    before = len(original.agents_executed)
    assert sid not in restarted.session_cache

    await _track(restarted, sid, "agent-b", "agent_start")

    session = restarted.session_cache[sid]
    assert session.started == original.started
    assert session.mode == original.mode
    assert session.metadata.tags == ["original-tag"]
    assert len(session.agents_executed) == before + 1


@pytest.mark.regression
async def test_second_stop_after_restart_folds(make_engine, db):
    sid, _, restarted = await _seed_session(
        make_engine, db, ("agent-a", "agent_start", "absent"), ("agent-a", "agent_stop", True)
    )

    await _track(restarted, sid, "agent-a", "agent_stop", True)

    execs = [a for a in restarted.session_cache[sid].agents_executed if a.agent_name == "agent-a"]
    assert len(execs) == 1
    assert execs[0].status == ExecutionStatus.SUCCESS


@pytest.mark.regression
async def test_finalized_session_hydrated_not_reopened(make_engine, db):
    engine = make_engine()
    sid = (await _track(engine, None, "agent-a", "agent_start")).session_id
    original_started = engine.session_cache[sid].started
    await engine.session_manage_lifecycle(operation="finalize", session_id=sid)
    assert sid not in engine.session_cache

    await _track(engine, sid, "agent-b", "agent_start")

    session = engine.session_cache[sid]
    assert session.status == SessionStatus.COMPLETED
    assert session.started == original_started


@pytest.mark.regression
async def test_unknown_session_still_auto_creates(make_engine):
    engine = make_engine()
    sid = "native-session-does-not-exist"

    await _track(engine, sid, "agent-a", "agent_start")

    session = engine.session_cache[sid]
    assert session.mode == "auto"
    assert "hook-bound" in session.metadata.tags


@pytest.mark.regression
async def test_typeless_stop_after_restart_not_ignored(make_engine, db):
    sid, _, restarted = await _seed_session(make_engine, db, ("agent-a", "agent_start", "absent"))
    assert "agent-a" not in restarted._agent_type_cache

    result = await _track(restarted, sid, "agent-a", "agent_stop", True, agent_type="")

    assert result.status != "ignored-no-start"
