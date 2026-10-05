"""Tests for the backgrounded initialize save and stale-session pruning (issue #174)."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence import DatabaseConfig
from persistence.sqlite import SQLiteBackend
from transport.http_server import HTTPSessionIntelligenceServer, NotificationManager
from transport.mcp_session_manager import MCPSessionManager
from transport.mcp_session_pruner import MCPSessionPruner
from transport.security import SecurityConfig
from transport.stall_monitor import StallMonitor

SLOW_SAVE_S = 1.0


@pytest.fixture
async def db(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "bg_save.db"))
    await backend.initialize()
    yield backend
    await backend.close()


@pytest.fixture
def monitor() -> StallMonitor:
    return StallMonitor(slow_db_ms=100.0)


@pytest.fixture
async def manager(db, monitor) -> MCPSessionManager:
    return MCPSessionManager(db, db_timer=monitor.timed_db)


@pytest.fixture
async def client(tmp_path, db, manager):
    engine = SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )
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
    app.state.session_engine = engine
    app.state.lean_interface = LeanMCPInterface(engine)
    app.state.mcp_session_manager = manager
    app.state.notification_manager = NotificationManager()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as c:
        yield c


def _body(method: str, params: dict | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}


def _slow_save(db, delay: float = SLOW_SAVE_S):
    real = db.save_mcp_session

    async def slow(data):
        await asyncio.sleep(delay)
        await real(data)

    return slow


async def test_initialize_returns_before_slow_save_completes(client, db, manager, monkeypatch):
    monkeypatch.setattr(db, "save_mcp_session", _slow_save(db))

    start = time.monotonic()
    resp = await client.post("/mcp", json=_body("initialize", {"clientInfo": {"name": "t"}}))
    elapsed = time.monotonic() - start

    assert resp.status_code == 200
    assert elapsed < SLOW_SAVE_S / 2
    session_id = resp.headers["MCP-Session-Id"]
    assert session_id in manager._pending_saves
    assert await db.get_mcp_session(session_id) is None  # not written yet

    await manager.drain_pending_saves()
    assert await db.get_mcp_session(session_id) is not None


async def test_followup_request_succeeds_while_save_pending(client, db, manager, monkeypatch):
    monkeypatch.setattr(db, "save_mcp_session", _slow_save(db))
    resp = await client.post("/mcp", json=_body("initialize"))
    session_id = resp.headers["MCP-Session-Id"]
    assert session_id in manager._pending_saves

    follow = await client.post(
        "/mcp", json=_body("tools/list"), headers={"MCP-Session-Id": session_id}
    )

    assert follow.status_code == 200
    assert "error" not in follow.json()
    assert session_id in manager._pending_saves  # request did not wait for the save

    await manager.drain_pending_saves()
    row = await db.get_mcp_session(session_id)
    assert row is not None  # single insert, no conflict


async def test_link_before_insert_completes_is_not_lost(db, manager, monkeypatch):
    monkeypatch.setattr(db, "save_mcp_session", _slow_save(db, 0.2))
    session_id = await manager.create_mcp_session()
    session = {"id": "eng-1"}
    await db.save_session(
        {
            "id": session["id"],
            "project_name": "p",
            "branch": "main",
            "started_at": datetime.now(UTC).isoformat(),
            "status": "active",
        }
    )

    await manager.link_engine_session(session_id, "eng-1")

    row = await db.get_mcp_session(session_id)
    assert row is not None
    assert row["engine_session_id"] == "eng-1"


async def test_save_failure_is_logged_and_not_fatal(db, manager, monkeypatch, caplog):
    async def boom(_data):
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "save_mcp_session", boom)
    with caplog.at_level(logging.WARNING, logger="transport.mcp_session_manager"):
        session_id = await manager.create_mcp_session()
        await manager.drain_pending_saves()

    assert any(session_id in r.getMessage() and "db down" in r.getMessage() for r in caplog.records)
    assert await manager.validate_session(session_id)  # still served from memory
    assert not manager._pending_saves


async def test_save_duration_reported_under_initialize_label(db, manager, monitor, monkeypatch):
    monkeypatch.setattr(db, "save_mcp_session", _slow_save(db, 0.2))
    await manager.create_mcp_session()
    await manager.drain_pending_saves()

    assert monitor.snapshot()["maxima"]["slowest_db"]["label"] == "initialize.save_mcp_session"


async def test_shutdown_waits_for_pending_saves(db, manager, monkeypatch):
    monkeypatch.setattr(db, "save_mcp_session", _slow_save(db, 0.3))
    session_id = await manager.create_mcp_session()

    await manager.drain_pending_saves(timeout=5.0)

    assert await db.get_mcp_session(session_id) is not None


async def test_shutdown_wait_is_bounded(db, manager, monkeypatch):
    monkeypatch.setattr(db, "save_mcp_session", _slow_save(db, 30.0))
    await manager.create_mcp_session()

    start = time.monotonic()
    await manager.drain_pending_saves(timeout=0.2)

    assert time.monotonic() - start < 2.0
    assert not manager._pending_saves


async def test_prune_once_deletes_stale_and_evicts_memory(db, manager):
    old_ts = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
    await db.save_mcp_session(
        {"mcp_session_id": "mcp-old", "created_at": old_ts, "last_activity": old_ts}
    )
    manager._active_sessions["mcp-old"] = {"mcp_session_id": "mcp-old", "last_activity": old_ts}
    fresh = await manager.create_mcp_session()
    await manager.drain_pending_saves()

    pruner = MCPSessionPruner(db, manager, retention=timedelta(hours=24))
    assert await pruner.prune_once() == 1

    assert await db.get_mcp_session("mcp-old") is None
    assert "mcp-old" not in manager._active_sessions
    assert await db.get_mcp_session(fresh) is not None


async def test_prune_once_never_raises(db, caplog):
    async def boom(*_a, **_k):
        raise RuntimeError("prune boom")

    db.delete_stale_mcp_sessions = boom  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="transport.mcp_session_pruner"):
        assert await MCPSessionPruner(db).prune_once() == 0
    assert any("Failed to prune" in r.getMessage() for r in caplog.records)


async def test_pruner_runs_periodically_and_stops_cleanly(db):
    old_ts = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
    await db.save_mcp_session(
        {"mcp_session_id": "mcp-old", "created_at": old_ts, "last_activity": old_ts}
    )
    pruner = MCPSessionPruner(db, retention=timedelta(hours=24), interval_s=0.05)

    pruner.start()
    await asyncio.sleep(0.3)
    task = pruner._task
    await pruner.stop()

    assert await db.get_mcp_session("mcp-old") is None
    assert task is not None and task.done()
    assert pruner._task is None
    await pruner.stop()  # idempotent


def test_pruner_env_configuration(monkeypatch, db):
    monkeypatch.setenv("SESSION_MCP_SESSION_RETENTION_HOURS", "6")
    monkeypatch.setenv("SESSION_MCP_SESSION_PRUNE_INTERVAL_S", "120")
    pruner = MCPSessionPruner(db)
    assert pruner.retention == timedelta(hours=6)
    assert pruner.interval_s == 120.0


def test_pruner_env_defaults(monkeypatch, db):
    monkeypatch.delenv("SESSION_MCP_SESSION_RETENTION_HOURS", raising=False)
    monkeypatch.delenv("SESSION_MCP_SESSION_PRUNE_INTERVAL_S", raising=False)
    pruner = MCPSessionPruner(db)
    assert pruner.retention == timedelta(hours=24)
    assert pruner.interval_s == 3600.0
