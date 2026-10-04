"""Tests for StallMonitor (issue #174) and its server_info wiring."""

from __future__ import annotations

import asyncio
import json
import time
import warnings

import httpx
import pytest

from core.session_engine import SessionIntelligenceEngine
from lean_mcp_interface import LeanMCPInterface
from persistence import DatabaseConfig
from persistence.sqlite import SQLiteBackend
from transport.http_server import HTTPSessionIntelligenceServer, NotificationManager
from transport.mcp_session_manager import MCPSessionManager
from transport.security import SecurityConfig
from transport.stall_monitor import StallMonitor


class FakePool:
    def __init__(self, size: int, idle: int, max_size: int) -> None:
        self._v = (size, idle, max_size)

    def get_size(self) -> int:
        return self._v[0]

    def get_idle_size(self) -> int:
        return self._v[1]

    def get_max_size(self) -> int:
        return self._v[2]


def _types(monitor: StallMonitor) -> list[str]:
    return [e["type"] for e in monitor.snapshot(events=500)["events"]]


async def test_blocking_call_produces_loop_lag_naming_tool():
    mon = StallMonitor(loop_lag_warn_ms=100, interval_ms=20)
    mon.start()
    await asyncio.sleep(0.05)
    with mon.track_request("tools/call", "slow_tool"):
        time.sleep(0.3)  # blocks the loop on purpose
        await asyncio.sleep(0.05)
    await mon.stop()
    lag = [e for e in mon.snapshot()["events"] if e["type"] == "loop_lag"]
    assert lag
    assert lag[0]["lag_ms"] >= 100
    assert lag[0]["in_flight"][0]["tool"] == "slow_tool"
    assert mon.snapshot()["maxima"]["loop_lag_ms"] >= 100


def test_sync_tool_recording_and_aggregate():
    mon = StallMonitor(sync_tool_warn_ms=50)
    mon.record_sync_tool("fast", 10)
    mon.record_sync_tool("slow", 80)
    mon.record_sync_tool("slow", 120)
    snap = mon.snapshot()
    assert snap["counters"]["sync_tool"] == 2
    assert snap["top_sync_tools"] == {"slow": {"count": 2, "total_ms": 200.0, "max_ms": 120.0}}
    assert snap["maxima"]["slowest_sync_tool"] == {"name": "slow", "ms": 120.0}


def test_time_sync_tool_context_manager():
    mon = StallMonitor(sync_tool_warn_ms=20)
    with mon.time_sync_tool("blocker"):
        time.sleep(0.05)
    assert "blocker" in mon.snapshot()["top_sync_tools"]


def test_slow_request_records_in_flight_counts():
    mon = StallMonitor(slow_request_ms=20)
    with mon.track_request("initialize"):
        assert len(mon.snapshot()["in_flight"]) == 1
    with mon.track_request("tools/call", "t"):
        with mon.track_request("tools/call", "other"):
            time.sleep(0.04)
    snap = mon.snapshot()
    assert snap["in_flight"] == []
    slow = [e for e in snap["events"] if e["type"] == "slow_request"]
    assert {e["tool"] for e in slow} == {"t", "other"}
    outer = next(e for e in slow if e["tool"] == "other")
    assert outer["in_flight_at_start"] == 1
    assert outer["in_flight_at_end"] == 1
    assert snap["maxima"]["slowest_request"]["ms"] >= 40


async def test_slow_db_recording():
    mon = StallMonitor(slow_db_ms=20)
    async with mon.timed_db("quick"):
        pass
    async with mon.timed_db("persist"):
        await asyncio.sleep(0.04)
    snap = mon.snapshot()
    assert snap["counters"]["slow_db"] == 1
    assert snap["maxima"]["slowest_db"]["label"] == "persist"


def test_ring_buffer_bound_and_counters():
    mon = StallMonitor(sync_tool_warn_ms=0, max_events=200)
    for i in range(250):
        mon.record_sync_tool(f"t{i}", 1 + i)
    snap = mon.snapshot(events=1000)
    assert len(snap["events"]) == 200
    assert snap["counters"]["sync_tool"] == 250
    assert snap["events"][-1]["tool"] == "t249"
    assert len(mon.snapshot()["events"]) == 50
    assert snap["events"][0]["ts"].endswith("+00:00")


def test_pool_exhausted_and_rate_limit():
    pool = FakePool(10, 0, 10)
    mon = StallMonitor(loop_lag_warn_ms=1000, pool_getter=lambda: pool)
    mon.check_tick(0)
    mon.check_tick(0)
    assert _types(mon) == ["pool_exhausted"]
    assert mon.snapshot()["pool"] == {"size": 10, "idle": 0, "max": 10}
    mon._last_pool_event -= 2.0
    mon.check_tick(0)
    assert _types(mon) == ["pool_exhausted", "pool_exhausted"]


def test_no_idle_but_pool_can_still_grow_is_not_exhausted():
    mon = StallMonitor(pool_getter=lambda: FakePool(2, 0, 10))
    mon.check_tick(0)
    assert _types(mon) == []


def test_pool_none_and_idle_connections_are_quiet():
    mon = StallMonitor(pool_getter=lambda: None)
    mon.check_tick(0)
    assert mon.snapshot()["pool"] is None
    mon2 = StallMonitor(pool_getter=lambda: FakePool(5, 3, 10))
    mon2.check_tick(0)
    assert _types(mon2) == []


async def test_stop_cancels_cleanly():
    before = len(asyncio.all_tasks())
    mon = StallMonitor(interval_ms=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mon.start()
        await asyncio.sleep(0.03)
        await mon.stop()
        await mon.stop()  # idempotent
    assert len(asyncio.all_tasks()) == before


def test_env_thresholds(monkeypatch):
    monkeypatch.setenv("SESSION_SLOW_DB_MS", "123")
    monkeypatch.setenv("SESSION_LOOP_LAG_WARN_MS", "bogus")
    mon = StallMonitor()
    assert mon.slow_db_ms == 123.0
    assert mon.loop_lag_warn_ms == 250.0


@pytest.fixture
async def client(tmp_path):
    db = SQLiteBackend(str(tmp_path / "stall.db"))
    await db.initialize()
    engine = SessionIntelligenceEngine(
        repository_path=str(tmp_path), use_filesystem=False, database=db
    )
    server = HTTPSessionIntelligenceServer(
        host="127.0.0.1",
        port=4098,
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
    app.state.mcp_session_manager = MCPSessionManager(db)
    app.state.notification_manager = NotificationManager()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as c:
        yield c
    await db.close()


async def test_server_info_includes_stall_diagnostics(client):
    init = await client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    sid = init.headers["MCP-Session-Id"]
    resp = await client.post(
        "/mcp",
        headers={"MCP-Session-Id": sid},
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "server_info", "arguments": {"events": 5}},
        },
    )
    payload = json.loads(resp.json()["result"]["content"][0]["text"])
    diag = payload["stall_diagnostics"]
    assert set(diag) == {
        "thresholds",
        "counters",
        "maxima",
        "top_sync_tools",
        "in_flight",
        "pool",
        "events",
    }
    # the server_info request itself is in flight while the snapshot is taken
    assert diag["in_flight"][0]["tool"] == "server_info"
    assert diag["pool"] is None


async def test_health_stalls_route(client):
    resp = await client.get("/health/stalls")
    assert resp.status_code == 200
    assert "counters" in resp.json()
