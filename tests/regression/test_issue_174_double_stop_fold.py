"""
Regression tests for issue #174 (double-stop half): Claude Code can fire
SubagentStop twice for one agent. The second stop must fold into the
already-terminal execution instead of creating a start-less duplicate.
"""

import pytest

from core.session_engine import SessionIntelligenceEngine
from models.session_models import ExecutionStatus
from persistence.sqlite import SQLiteBackend

AGENT_TYPE = "micro-survey"


@pytest.fixture
async def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_INTELLIGENCE_AGENT_VALIDATION", "off")
    db = SQLiteBackend(db_path=str(tmp_path / "issue174.db"))
    await db.initialize()
    eng = SessionIntelligenceEngine(
        repository_path=str(tmp_path),
        use_filesystem=False,
        database=db,
    )
    yield eng
    await db.close()


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


def _execs(engine, session_id, agent_name):
    session = engine.session_cache[session_id]
    return [a for a in session.agents_executed if a.agent_name == agent_name]


@pytest.mark.regression
async def test_double_stop_folds_into_one_execution(engine):
    first = await _track(engine, None, "agent-a", "agent_start")
    sid = first.session_id
    await _track(engine, sid, "agent-a", "agent_stop", True)
    await _track(engine, sid, "agent-a", "agent_stop", True)

    execs = _execs(engine, sid, "agent-a")
    assert len(execs) == 1
    assert execs[0].status == ExecutionStatus.SUCCESS
    assert [s.operation for s in execs[0].execution_steps] == [
        "agent_start",
        "agent_stop",
        "agent_stop",
    ]
    assert execs[0].completed == execs[0].execution_steps[-1].completed


@pytest.mark.regression
async def test_later_determinate_stop_wins(engine):
    sid = (await _track(engine, None, "agent-a", "agent_start")).session_id
    await _track(engine, sid, "agent-a", "agent_stop", False)
    await _track(engine, sid, "agent-a", "agent_stop", True)

    execs = _execs(engine, sid, "agent-a")
    assert len(execs) == 1
    assert execs[0].status == ExecutionStatus.SUCCESS


@pytest.mark.regression
async def test_indeterminate_does_not_overwrite_success(engine):
    sid = (await _track(engine, None, "agent-a", "agent_start")).session_id
    await _track(engine, sid, "agent-a", "agent_stop", True)
    await _track(engine, sid, "agent-a", "agent_stop", None)

    execs = _execs(engine, sid, "agent-a")
    assert len(execs) == 1
    assert execs[0].status == ExecutionStatus.SUCCESS


@pytest.mark.regression
async def test_abandoned_execution_takes_late_stop_status(engine):
    sid = (await _track(engine, None, "agent-a", "agent_start")).session_id
    _execs(engine, sid, "agent-a")[0].status = ExecutionStatus.ABANDONED
    await _track(engine, sid, "agent-a", "agent_stop", True)

    execs = _execs(engine, sid, "agent-a")
    assert len(execs) == 1
    assert execs[0].status == ExecutionStatus.SUCCESS


@pytest.mark.regression
async def test_different_agents_are_not_folded(engine):
    sid = (await _track(engine, None, "agent-a", "agent_start")).session_id
    await _track(engine, sid, "agent-a", "agent_stop", True)
    await _track(engine, sid, "agent-b", "agent_start")
    await _track(engine, sid, "agent-b", "agent_stop", True)

    assert len(_execs(engine, sid, "agent-a")) == 1
    assert len(_execs(engine, sid, "agent-b")) == 1
    assert len(engine.session_cache[sid].agents_executed) == 2


@pytest.mark.regression
async def test_internal_agent_stops_are_never_folded(engine):
    sid = (await _track(engine, None, "bash-executor", "agent_stop", True)).session_id
    await _track(engine, sid, "bash-executor", "agent_stop", True)

    assert len(_execs(engine, sid, "bash-executor")) == 2


@pytest.mark.regression
async def test_resume_start_opens_new_execution(engine):
    sid = (await _track(engine, None, "agent-a", "agent_start")).session_id
    await _track(engine, sid, "agent-a", "agent_stop", True)
    await _track(engine, sid, "agent-a", "agent_start")
    await _track(engine, sid, "agent-a", "agent_stop", True)

    execs = _execs(engine, sid, "agent-a")
    assert len(execs) == 2
    assert all(e.status == ExecutionStatus.SUCCESS for e in execs)
