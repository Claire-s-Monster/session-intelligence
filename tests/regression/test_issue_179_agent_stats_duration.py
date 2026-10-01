"""
Regression tests for issue #179: get_agent_stats' avg_duration_ms was None for
every agent because it only read duration keys from the `performance` blob,
which nothing populates.

https://github.com/Claire-s-Monster/session-intelligence/issues/179

Fix: per row, duration is a positive blob value (duration_ms,
total_duration_ms, total_execution_time_ms), else completed_at - started_at,
else None. Both backends share persistence.base.execution_duration_ms.

SQLite is exercised against a real temp database. PostgreSQL is exercised
through a fake asyncpg pool (no live DB; never the production database).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from persistence.base import execution_duration_ms
from persistence.postgresql import PostgreSQLBackend
from persistence.sqlite import SQLiteBackend

AGENT_TYPE = "focused"
AGENT_NAME = "focused-code-modifier"


def _times(seconds: float | None) -> tuple[datetime, datetime | None]:
    start = datetime.now(UTC) - timedelta(minutes=10)
    return start, (start + timedelta(seconds=seconds) if seconds is not None else None)


# Each case: (status, performance, seconds-between-timestamps or None)
Case = tuple[str, dict[str, Any], float | None]


async def _sqlite_avg(tmp_path, cases: list[Case]) -> float | None:
    db = SQLiteBackend(str(tmp_path / "test.db"))
    await db.initialize()
    try:
        sid = f"session-{uuid.uuid4().hex[:8]}"
        await db.save_session(
            {
                "id": sid,
                "started_at": datetime.now(UTC).isoformat(),
                "ended_at": None,
                "project_path": "/tmp/proj",
                "project_name": "proj",
                "mode": "local",
                "status": "active",
                "metadata": {},
                "performance_metrics": {},
                "health_status": {},
            }
        )
        for status, perf, secs in cases:
            start, end = _times(secs)
            await db.save_agent_execution(
                {
                    "id": f"exec-{uuid.uuid4().hex[:8]}",
                    "session_id": sid,
                    "agent_name": AGENT_NAME,
                    "agent_type": AGENT_TYPE,
                    "started_at": start.isoformat(),
                    "completed_at": end.isoformat() if end else None,
                    "status": status,
                    "execution_steps": [],
                    "performance": perf,
                    "errors": [],
                }
            )
        stats = await db.get_agent_stats(time_window_hours=168)
        entry = next(a for a in stats["agents"] if a["agent_type"] == AGENT_TYPE)
        return entry["avg_duration_ms"]
    finally:
        await db.close()


class _FakeConn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def fetch(self, *_a: Any) -> list[dict[str, Any]]:
        return self._rows

    async def fetchrow(self, *_a: Any) -> tuple[int]:
        return (0,)


class _FakeAcquire:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *_a: Any) -> None:
        return None


class _FakePool:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._conn = _FakeConn(rows)

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self._conn)


async def _pg_avg(cases: list[Case]) -> float | None:
    rows = []
    for status, perf, secs in cases:
        start, end = _times(secs)
        rows.append(
            {
                "agent_type": AGENT_TYPE,
                "agent_name": AGENT_NAME,
                "status": status,
                "performance": perf,  # asyncpg may hand back dict or JSON string
                "started_at": start,
                "completed_at": end,
            }
        )
    backend = PostgreSQLBackend(dsn="postgresql://localhost/fake_issue_179")
    backend._pool = _FakePool(rows)  # type: ignore[assignment]
    stats = await backend.get_agent_stats(time_window_hours=168)
    entry = next(a for a in stats["agents"] if a["agent_type"] == AGENT_TYPE)
    return entry["avg_duration_ms"]


@pytest.fixture(params=["sqlite", "postgresql"])
def avg_for(request, tmp_path):
    async def run(cases: list[Case]) -> float | None:
        if request.param == "sqlite":
            return await _sqlite_avg(tmp_path, cases)
        return await _pg_avg(cases)

    return run


@pytest.mark.regression
class TestExecutionDurationNaive:
    def test_mixed_naive_start_aware_end(self):
        n = datetime(2026, 3, 1, 12, 0, 0)
        end = n.astimezone().astimezone(UTC) + timedelta(seconds=164)
        row = {"started_at": n, "completed_at": end}
        assert execution_duration_ms(row) == 164000.0

    def test_two_naive_iso_strings(self):
        row = {
            "started_at": "2026-03-01T12:00:00",
            "completed_at": "2026-03-01T12:02:44",
        }
        assert execution_duration_ms(row) == 164000.0


@pytest.mark.regression
class TestAgentStatsDuration:
    async def test_timestamps_used_when_blob_empty(self, avg_for):
        assert await avg_for([("completed", {}, 164.0)]) == pytest.approx(164000.0, abs=0.1)

    async def test_positive_blob_beats_timestamps(self, avg_for):
        assert await avg_for([("completed", {"duration_ms": 5000}, 164.0)]) == 5000.0

    async def test_blob_key_order(self, avg_for):
        perf = {"total_duration_ms": 7000, "total_execution_time_ms": 9000}
        assert await avg_for([("completed", perf, 164.0)]) == 7000.0

    async def test_total_execution_time_ms_key_used(self, avg_for):
        assert await avg_for([("completed", {"total_execution_time_ms": 9000}, 1.0)]) == 9000.0

    async def test_running_execution_contributes_nothing(self, avg_for):
        assert await avg_for([("running", {}, None)]) is None

    async def test_running_row_does_not_dilute_average(self, avg_for):
        got = await avg_for([("completed", {}, 100.0), ("running", {}, None)])
        assert got == pytest.approx(100000.0, abs=0.1)

    async def test_indeterminate_excluded_from_average(self, avg_for):
        got = await avg_for([("completed", {}, 10.0), ("indeterminate", {}, 500.0)])
        assert got == pytest.approx(10000.0, abs=0.1)

    async def test_zero_blob_value_falls_through_to_timestamps(self, avg_for):
        got = await avg_for([("completed", {"total_execution_time_ms": 0}, 164.0)])
        assert got == pytest.approx(164000.0, abs=0.1)
