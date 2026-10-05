"""Stall instrumentation for the HTTP MCP server (issue #174).

Separates two suspects behind intermittent multi-second unresponsiveness:
event-loop blocking (sync tool bodies, subprocess calls) versus DB pool
exhaustion / long DB waits. Records bounded, JSON-serialisable evidence that
can be read over MCP (``server_info`` -> ``stall_diagnostics``). It changes no
behaviour of the code it observes.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import os
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

EVENT_TYPES = ("loop_lag", "sync_tool", "slow_request", "slow_db", "pool_exhausted")
_POOL_EVENT_MIN_GAP_S = 1.0


def _env_ms(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class StallMonitor:
    """Collects loop-lag, sync-tool, slow-request, slow-DB and pool events."""

    def __init__(
        self,
        *,
        loop_lag_warn_ms: float | None = None,
        slow_request_ms: float | None = None,
        sync_tool_warn_ms: float | None = None,
        slow_db_ms: float | None = None,
        interval_ms: float | None = None,
        pool_getter: Callable[[], Any | None] | None = None,
        extra_stats: Callable[[], dict[str, Any]] | None = None,
        max_events: int = 200,
    ) -> None:
        self.loop_lag_warn_ms = (
            loop_lag_warn_ms
            if loop_lag_warn_ms is not None
            else _env_ms("SESSION_LOOP_LAG_WARN_MS", 250.0)
        )
        self.slow_request_ms = (
            slow_request_ms
            if slow_request_ms is not None
            else _env_ms("SESSION_SLOW_REQUEST_MS", 1000.0)
        )
        self.sync_tool_warn_ms = (
            sync_tool_warn_ms
            if sync_tool_warn_ms is not None
            else _env_ms("SESSION_SYNC_TOOL_WARN_MS", 200.0)
        )
        self.slow_db_ms = (
            slow_db_ms if slow_db_ms is not None else _env_ms("SESSION_SLOW_DB_MS", 500.0)
        )
        self.interval_ms = (
            interval_ms
            if interval_ms is not None
            else _env_ms("SESSION_LOOP_LAG_INTERVAL_MS", 200.0)
        )
        self._pool_getter = pool_getter
        self._extra_stats = extra_stats
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._counters: dict[str, int] = dict.fromkeys(EVENT_TYPES, 0)
        self._max_loop_lag_ms = 0.0
        self._slowest_request: dict[str, Any] | None = None
        self._slowest_db: dict[str, Any] | None = None
        self._sync_tools: dict[str, dict[str, float]] = {}
        self._in_flight: dict[int, dict[str, Any]] = {}
        self._ids = itertools.count(1)
        self._last_pool_event = float("-inf")
        self._task: asyncio.Task[None] | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the loop-lag task (idempotent). Requires a running loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(
                self._run(), name="stall-monitor-lag"
            )

    async def stop(self) -> None:
        """Cancel the loop-lag task and wait for it to finish."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        interval_s = self.interval_ms / 1000.0
        while True:
            before = time.monotonic()
            await asyncio.sleep(interval_s)
            self.check_tick((time.monotonic() - before - interval_s) * 1000.0)

    def check_tick(self, lag_ms: float) -> None:
        """Process one lag measurement (also callable directly)."""
        if lag_ms >= self.loop_lag_warn_ms:
            self._max_loop_lag_ms = max(self._max_loop_lag_ms, lag_ms)
            pool = self.pool_stats()
            self._record(
                "loop_lag",
                lag_ms=round(lag_ms, 1),
                in_flight=self.in_flight_snapshot(),
                pool=pool,
            )
            logger.warning(
                "Event loop lag %.0fms; in_flight=%s pool=%s", lag_ms, self._in_flight_brief(), pool
            )
        self._check_pool()

    def _check_pool(self) -> None:
        pool = self.pool_stats()
        # idle counts only created connections; the pool is exhausted only at max size.
        if pool is None or pool["idle"] != 0 or pool["size"] < pool["max"]:
            return
        now = time.monotonic()
        if now - self._last_pool_event < _POOL_EVENT_MIN_GAP_S:
            return
        self._last_pool_event = now
        self._record("pool_exhausted", pool=pool, in_flight_count=len(self._in_flight))
        logger.warning("DB pool has no idle connections: %s", pool)

    # -- recording ---------------------------------------------------------

    def _record(self, event_type: str, **fields: Any) -> None:
        self._counters[event_type] += 1
        self._events.append({"ts": datetime.now(UTC).isoformat(), "type": event_type, **fields})

    @contextmanager
    def track_request(self, kind: str, tool: str | None = None) -> Iterator[None]:
        """Register a request as in-flight; record slow_request on exit."""
        rid = next(self._ids)
        start = time.monotonic()
        at_start = len(self._in_flight)
        self._in_flight[rid] = {"kind": kind, "tool": tool, "start": start}
        try:
            yield
        finally:
            self._in_flight.pop(rid, None)
            duration_ms = (time.monotonic() - start) * 1000.0
            if duration_ms >= self.slow_request_ms:
                if self._slowest_request is None or duration_ms > self._slowest_request["ms"]:
                    self._slowest_request = {
                        "kind": kind,
                        "tool": tool,
                        "ms": round(duration_ms, 1),
                    }
                self._record(
                    "slow_request",
                    kind=kind,
                    tool=tool,
                    duration_ms=round(duration_ms, 1),
                    in_flight_at_start=at_start,
                    in_flight_at_end=len(self._in_flight),
                )

    def record_sync_tool(self, name: str, duration_ms: float) -> None:
        """Record a sync tool body that blocked the loop for duration_ms."""
        if duration_ms < self.sync_tool_warn_ms:
            return
        agg = self._sync_tools.setdefault(name, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
        agg["count"] += 1
        agg["total_ms"] += duration_ms
        agg["max_ms"] = max(agg["max_ms"], duration_ms)
        self._record("sync_tool", tool=name, duration_ms=round(duration_ms, 1))
        logger.warning("Sync tool %s blocked the event loop for %.0fms", name, duration_ms)

    @contextmanager
    def time_sync_tool(self, name: str) -> Iterator[None]:
        start = time.monotonic()
        try:
            yield
        finally:
            self.record_sync_tool(name, (time.monotonic() - start) * 1000.0)

    @asynccontextmanager
    async def timed_db(
        self, label: str, extra: Mapping[str, Any] | None = None
    ) -> AsyncIterator[None]:
        """Time an awaited DB operation; record slow_db at/above threshold.

        ``extra`` may be a dict the body fills in while it runs; its contents at
        exit are attached to the slow_db event (e.g. persist counts, issue #190).
        """
        start = time.monotonic()
        try:
            yield
        finally:
            duration_ms = (time.monotonic() - start) * 1000.0
            if duration_ms >= self.slow_db_ms:
                if self._slowest_db is None or duration_ms > self._slowest_db["ms"]:
                    self._slowest_db = {"label": label, "ms": round(duration_ms, 1)}
                fields = dict(extra) if extra else {}
                self._record(
                    "slow_db", **{**fields, "label": label, "duration_ms": round(duration_ms, 1)}
                )
                logger.warning("Slow DB operation %s: %.0fms", label, duration_ms)

    # -- reading -----------------------------------------------------------

    def pool_stats(self) -> dict[str, int] | None:
        pool = self._pool_getter() if self._pool_getter else None
        if pool is None:
            return None
        try:
            return {
                "size": int(pool.get_size()),
                "idle": int(pool.get_idle_size()),
                "max": int(pool.get_max_size()),
            }
        except Exception:
            logger.debug("Could not read pool stats", exc_info=True)
            return None

    def _read_extra_stats(self) -> dict[str, Any]:
        if self._extra_stats is None:
            return {}
        try:
            return dict(self._extra_stats())
        except Exception:
            logger.debug("Could not read extra stall stats", exc_info=True)
            return {}

    def in_flight_snapshot(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        return [
            {"kind": r["kind"], "tool": r["tool"], "age_ms": round((now - r["start"]) * 1000.0, 1)}
            for r in self._in_flight.values()
        ]

    def _in_flight_brief(self) -> list[str]:
        return [f"{r['kind']}:{r['tool']}" for r in self.in_flight_snapshot()]

    def snapshot(self, events: int = 50) -> dict[str, Any]:
        """JSON-serialisable evidence; newest event last."""
        recent = list(self._events)[-events:] if events > 0 else []
        slowest_sync = max(self._sync_tools.items(), key=lambda kv: kv[1]["max_ms"], default=None)
        return {
            **self._read_extra_stats(),
            "thresholds": {
                "loop_lag_warn_ms": self.loop_lag_warn_ms,
                "slow_request_ms": self.slow_request_ms,
                "sync_tool_warn_ms": self.sync_tool_warn_ms,
                "slow_db_ms": self.slow_db_ms,
                "interval_ms": self.interval_ms,
            },
            "counters": dict(self._counters),
            "maxima": {
                "loop_lag_ms": round(self._max_loop_lag_ms, 1),
                "slowest_sync_tool": (
                    {"name": slowest_sync[0], "ms": round(slowest_sync[1]["max_ms"], 1)}
                    if slowest_sync
                    else None
                ),
                "slowest_request": self._slowest_request,
                "slowest_db": self._slowest_db,
            },
            "top_sync_tools": {
                name: {
                    "count": int(a["count"]),
                    "total_ms": round(a["total_ms"], 1),
                    "max_ms": round(a["max_ms"], 1),
                }
                for name, a in self._sync_tools.items()
            },
            "in_flight": self.in_flight_snapshot(),
            "pool": self.pool_stats(),
            "events": recent,
        }
