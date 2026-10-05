"""Periodic pruning of stale mcp_sessions rows (issue #174).

Every hook invocation runs the MCP ``initialize`` handshake, which inserts an
``mcp_sessions`` row; nothing else removes them. ``last_activity`` is refreshed on
every later request carrying the session id (``MCPSessionManager.update_activity``),
so a row idle past the retention window belongs to a client that has gone away.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from persistence.base import DatabaseBackend as Database
    from transport.mcp_session_manager import MCPSessionManager

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_HOURS = 24.0
DEFAULT_PRUNE_INTERVAL_S = 3600.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


class MCPSessionPruner:
    """Deletes stale mcp_sessions rows once on demand and then periodically."""

    def __init__(
        self,
        database: Database,
        manager: MCPSessionManager | None = None,
        retention: timedelta | None = None,
        interval_s: float | None = None,
    ) -> None:
        self._database = database
        self._manager = manager
        self.retention = retention or timedelta(
            hours=_env_float("SESSION_MCP_SESSION_RETENTION_HOURS", DEFAULT_RETENTION_HOURS)
        )
        self.interval_s = (
            interval_s
            if interval_s is not None
            else _env_float("SESSION_MCP_SESSION_PRUNE_INTERVAL_S", DEFAULT_PRUNE_INTERVAL_S)
        )
        self._task: asyncio.Task[None] | None = None

    async def prune_once(self) -> int:
        """Prune DB rows and matching in-memory entries. Never raises."""
        try:
            deleted = await self._database.delete_stale_mcp_sessions(self.retention)
            if self._manager is not None:
                await self._manager.cleanup_inactive_sessions(int(self.retention.total_seconds()))
        except Exception:
            logger.warning("Failed to prune stale MCP sessions", exc_info=True)
            return 0
        hours = self.retention.total_seconds() / 3600
        logger.info(f"Pruned {deleted} stale MCP session(s) idle > {hours:g}h")
        return deleted

    def start(self) -> None:
        """Start the periodic task (idempotent). Requires a running loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(
                self._run(), name="mcp-session-pruner"
            )

    async def stop(self) -> None:
        """Cancel the periodic task and wait for it to finish."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            await self.prune_once()
