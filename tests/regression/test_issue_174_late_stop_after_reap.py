"""
Regression tests for issue #174 (outbox replay): a replayed agent_stop can
arrive AFTER the startup sweep (reap_stale_executions) flipped its execution
to ABANDONED. The late stop must set the real status on the SAME execution,
never create a duplicate, and an indeterminate stop must beat ABANDONED.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import ExecutionStatus
from persistence.sqlite import SQLiteBackend

AGENT_TYPE = "micro-survey"


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(db_path=str(tmp_path / "issue174.db"))
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


async def _track(engine, session_id, agent_name, phase, success="absent"):
    step_data = {"phase": phase, "agent_type": AGENT_TYPE}
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


async def _start_persist_and_reap(make_engine, db):
    """Start agent-a on engine 1, persist it RUNNING, reap it in the DB.

    Returns (session_id, first_engine).
    """
    first = make_engine()
    sid = (await _track(first, None, "agent-a", "agent_start")).session_id
    await _persist(first, db, sid)
    assert await db.reap_stale_executions(older_than_hours=0) == 1
    return sid, first


def _agent_execs(engine, sid):
    return [a for a in engine.session_cache[sid].agents_executed if a.agent_name == "agent-a"]


async def _db_statuses(db, sid):
    session_row = await db.get_session(sid)
    assert session_row is not None
    conn = db._ensure_connected()
    cursor = await conn.execute(
        "SELECT status FROM agent_executions WHERE session_id = ? AND agent_name = 'agent-a'",
        (sid,),
    )
    return [row[0] for row in await cursor.fetchall()]


@pytest.mark.regression
@pytest.mark.parametrize(
    ("success", "expected"),
    [
        (True, ExecutionStatus.SUCCESS),
        (False, ExecutionStatus.ERROR),
        (None, ExecutionStatus.INDETERMINATE),
    ],
)
async def test_late_stop_after_reap_db_only_session(make_engine, db, success, expected):
    """Case A: ABANDONED exists only in the DB; a restarted engine hydrates it."""
    sid, _ = await _start_persist_and_reap(make_engine, db)
    restarted = make_engine()
    assert sid not in restarted.session_cache

    await _track(restarted, sid, "agent-a", "agent_stop", success)

    execs = _agent_execs(restarted, sid)
    assert len(execs) == 1
    assert execs[0].status == expected
    assert execs[0].completed is not None
    assert execs[0].execution_steps[-1].operation == "agent_stop"


@pytest.mark.regression
@pytest.mark.parametrize(
    ("success", "expected"),
    [
        (True, ExecutionStatus.SUCCESS),
        (False, ExecutionStatus.ERROR),
        (None, ExecutionStatus.INDETERMINATE),
    ],
)
async def test_late_stop_after_reap_stale_cache(make_engine, db, success, expected):
    """Case B: reaper flipped the DB row while the cached copy still says RUNNING."""
    sid, first = await _start_persist_and_reap(make_engine, db)
    assert _agent_execs(first, sid)[0].status == ExecutionStatus.RUNNING

    await _track(first, sid, "agent-a", "agent_stop", success)
    await _persist(first, db, sid)

    execs = _agent_execs(first, sid)
    assert len(execs) == 1
    assert execs[0].status == expected
    assert await _db_statuses(db, sid) == [expected.value]


@pytest.mark.regression
async def test_late_stop_in_cache_abandoned_in_memory(make_engine, db):
    """Case C: cached copy itself is ABANDONED (e.g. loaded after the sweep)."""
    sid, first = await _start_persist_and_reap(make_engine, db)
    _agent_execs(first, sid)[0].status = ExecutionStatus.ABANDONED

    await _track(first, sid, "agent-a", "agent_stop", True)

    execs = _agent_execs(first, sid)
    assert len(execs) == 1
    assert execs[0].status == ExecutionStatus.SUCCESS


@pytest.mark.regression
async def test_indeterminate_does_not_overwrite_determinate(make_engine, db):
    """Existing fold rule: INDETERMINATE never overwrites SUCCESS/ERROR."""
    engine = make_engine()
    sid = (await _track(engine, None, "agent-a", "agent_start")).session_id
    await _track(engine, sid, "agent-a", "agent_stop", True)

    await _track(engine, sid, "agent-a", "agent_stop", None)

    execs = _agent_execs(engine, sid)
    assert len(execs) == 1
    assert execs[0].status == ExecutionStatus.SUCCESS
