import uuid

import pytest
import pytest_asyncio

from core.session_engine import SessionIntelligenceEngine as SessionEngine
from persistence.postgresql import PostgreSQLBackend

DSN = "postgresql://localhost/session_intelligence"


@pytest_asyncio.fixture
async def db():
    database = PostgreSQLBackend(DSN)
    await database.initialize()
    yield database
    await database.close()


def _make_engine(database: PostgreSQLBackend) -> SessionEngine:
    return SessionEngine(repository_path=".", database=database, use_filesystem=False)


async def _cleanup(database: PostgreSQLBackend, project_names: list[str]) -> None:
    pool = database._ensure_connected()
    async with pool.acquire() as conn:
        for pn in project_names:
            await conn.execute(
                "DELETE FROM decisions WHERE session_id IN "
                "(SELECT id FROM sessions WHERE project_name = $1)",
                pn,
            )
            await conn.execute("DELETE FROM sessions WHERE project_name = $1", pn)


@pytest.mark.asyncio
async def test_explicit_create_recall_finds_decision(db):
    pn = f"test-proj-{uuid.uuid4().hex[:8]}"
    engine = _make_engine(db)
    try:
        create_result = engine._create_session(
            mode="local", project_name=pn, metadata={}
        )
        assert create_result.status == "success"
        sid = create_result.session_id

        await db.save_session(create_result.session_data.model_dump(mode="python"))

        result = await engine.session_log_decision(
            decision="explicit-create-probe", session_id=sid
        )
        assert result.decision_id != "error"

        pool = db._ensure_connected()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM decisions WHERE session_id = $1", sid
            )
        assert row is not None
    finally:
        await _cleanup(db, [pn])


@pytest.mark.asyncio
async def test_log_without_create_uses_unbound_not_claude(db):
    engine = _make_engine(db)
    try:
        result = await engine.session_log_decision(decision="no-create-probe")
        assert result.decision_id != "error"

        sid = engine._current_session_id
        assert sid is not None

        pool = db._ensure_connected()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT project_name FROM sessions WHERE id = $1", sid
            )
        assert row is not None
        assert row["project_name"] == "_unbound_"
        assert row["project_name"] != ".claude"
    finally:
        await _cleanup(db, ["_unbound_"])


@pytest.mark.asyncio
async def test_create_then_log_uses_current_session(db):
    pn = f"test-proj-{uuid.uuid4().hex[:8]}"
    engine = _make_engine(db)
    try:
        create_result = engine._create_session(
            mode="local", project_name=pn, metadata={}
        )
        assert create_result.status == "success"
        sid = create_result.session_id
        assert engine._current_session_id == sid

        await db.save_session(create_result.session_data.model_dump(mode="python"))

        result = await engine.session_log_decision(decision="create-then-log-probe")
        assert result.decision_id != "error"

        pool = db._ensure_connected()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT session_id FROM decisions WHERE session_id = $1", sid
            )
        assert row is not None
    finally:
        await _cleanup(db, [pn])


@pytest.mark.asyncio
async def test_log_decision_with_project_name_param_creates_correct_session(db):
    pn = f"test-proj-{uuid.uuid4().hex[:8]}"
    engine = _make_engine(db)
    try:
        result = await engine.session_log_decision(
            decision="project-name-param-probe", project_name=pn
        )
        assert result.decision_id != "error"

        pool = db._ensure_connected()
        async with pool.acquire() as conn:
            session_row = await conn.fetchrow(
                "SELECT id FROM sessions WHERE project_name = $1", pn
            )
            assert session_row is not None, f"No session found for project_name={pn!r}"

            decision_row = await conn.fetchrow(
                "SELECT id FROM decisions WHERE session_id = $1", session_row["id"]
            )
            assert decision_row is not None
    finally:
        await _cleanup(db, [pn])
