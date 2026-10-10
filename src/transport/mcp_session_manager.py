"""
MCP Session Manager - Maps MCP session IDs to engine sessions.

The MCP protocol uses MCP-Session-Id headers for stateful communication.
This manager:
1. Tracks MCP session IDs from the initialize handshake
2. Maps them to internal SessionIntelligenceEngine sessions
3. Handles session lifecycle (creation, resume, cleanup)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from persistence.base import _as_aware_utc

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from persistence.base import DatabaseBackend as Database

logger = logging.getLogger(__name__)

# Format of the ids create_mcp_session mints; get_or_adopt_session (issue #205)
# derives its accept-pattern from the same parts so the two cannot drift.
MCP_SESSION_ID_PREFIX = "mcp-"
MCP_SESSION_ID_HEX_LEN = 16
MCP_SESSION_ID_PATTERN = re.compile(
    rf"{re.escape(MCP_SESSION_ID_PREFIX)}[0-9a-f]{{{MCP_SESSION_ID_HEX_LEN}}}"
)


class MCPSessionManager:
    """Manages MCP session IDs and their mapping to engine sessions."""

    def __init__(
        self,
        database: Database | None = None,
        db_timer: Callable[[str], AbstractAsyncContextManager[None]] | None = None,
    ) -> None:
        self.database = database
        self._db_timer = db_timer
        self._active_sessions: dict[str, dict[str, Any]] = {}
        # Issue #174: the initialize DB write runs off the response path. Tasks
        # are referenced here (not just by the loop, which holds them weakly).
        self._pending_saves: dict[str, asyncio.Task[None]] = {}

    async def create_mcp_session(self, client_info: dict[str, Any] | None = None) -> str:
        """Create a new MCP session.

        The session is registered in memory synchronously and the id returned at
        once; the DB row is written by a background task (issue #174).
        """
        mcp_session_id = f"{MCP_SESSION_ID_PREFIX}{uuid.uuid4().hex[:MCP_SESSION_ID_HEX_LEN]}"
        self._register_session(mcp_session_id, client_info)
        logger.info(f"Created MCP session: {mcp_session_id}")
        return mcp_session_id

    def _register_session(
        self, mcp_session_id: str, client_info: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Register an entry in memory and schedule its background DB write."""
        now = datetime.now(UTC).isoformat()

        session_data = {
            "mcp_session_id": mcp_session_id,
            "engine_session_id": None,
            "created_at": now,
            "last_activity": now,
            "client_info": client_info or {},
        }

        self._active_sessions[mcp_session_id] = session_data

        if self.database:
            task = asyncio.get_running_loop().create_task(
                self._persist_new_session(session_data),
                name=f"mcp-session-save-{mcp_session_id}",
            )
            self._pending_saves[mcp_session_id] = task
            task.add_done_callback(functools.partial(self._drop_pending, mcp_session_id))

        return session_data

    async def get_or_adopt_session(self, mcp_session_id: str) -> bool:
        """Validate a session, re-adopting an unknown id this server could have minted.

        Issue #205: after a restart (or a prune) clients still hold ids the server no
        longer knows; answering 404 forces every client to reconnect. An unknown id
        in our own format is adopted instead. Anything else returns False (404).
        """
        if await self.validate_session(mcp_session_id):
            return True

        if not MCP_SESSION_ID_PATTERN.fullmatch(mcp_session_id):
            return False

        # validate_session awaited the DB; a concurrent request may have adopted the
        # id meanwhile. No await between this check and the insert, so only one wins.
        if mcp_session_id in self._active_sessions:
            return True

        self._register_session(mcp_session_id, {"adopted": True})
        logger.info(f"Re-adopted unknown MCP session id {mcp_session_id} after restart")
        return True

    def _drop_pending(self, mcp_session_id: str, _task: asyncio.Task[None]) -> None:
        self._pending_saves.pop(mcp_session_id, None)

    async def _persist_new_session(self, session_data: dict[str, Any]) -> None:
        """Background DB write for a new session; never raises."""
        mcp_session_id = session_data["mcp_session_id"]
        database = self.database
        if database is None:
            return
        try:
            if self._db_timer is not None:
                async with self._db_timer("initialize.save_mcp_session"):
                    await database.save_mcp_session(session_data)
            else:
                await database.save_mcp_session(session_data)
        except Exception as e:
            logger.warning(f"Failed to persist MCP session {mcp_session_id}: {e}")

    async def _await_pending_save(self, mcp_session_id: str) -> None:
        """Wait for the session's initial insert, if still in flight."""
        task = self._pending_saves.get(mcp_session_id)
        if task is not None:
            await asyncio.shield(task)

    async def drain_pending_saves(self, timeout: float = 5.0) -> None:
        """Give in-flight initial saves a bounded time to finish (shutdown)."""
        pending = list(self._pending_saves.values())
        if not pending:
            return
        _, still_pending = await asyncio.wait(pending, timeout=timeout)
        if still_pending:
            logger.warning(f"{len(still_pending)} MCP session save(s) unfinished at shutdown")
            for task in still_pending:
                task.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)

    async def get_engine_session_id(self, mcp_session_id: str) -> str | None:
        """Get the engine session ID for an MCP session."""
        if mcp_session_id in self._active_sessions:
            return self._active_sessions[mcp_session_id].get("engine_session_id")

        if self.database:
            await self._await_pending_save(mcp_session_id)
            try:
                mcp_session = await self.database.get_mcp_session(mcp_session_id)
                if mcp_session:
                    self._active_sessions[mcp_session_id] = mcp_session
                    return mcp_session.get("engine_session_id")
            except Exception as e:
                logger.warning(f"Failed to load MCP session from database: {e}")

        return None

    async def link_engine_session(self, mcp_session_id: str, engine_session_id: str) -> None:
        """Link an MCP session to an engine session."""
        if mcp_session_id in self._active_sessions:
            self._active_sessions[mcp_session_id]["engine_session_id"] = engine_session_id

        if self.database:
            try:
                # The link is a bare UPDATE: before the initial INSERT lands it would
                # match 0 rows and be lost, so wait for the insert first.
                await self._await_pending_save(mcp_session_id)
                await self.database.link_mcp_to_engine_session(mcp_session_id, engine_session_id)
            except Exception as e:
                logger.warning(f"Failed to persist engine session link: {e}")

        logger.info(f"Linked MCP session {mcp_session_id} to engine {engine_session_id}")

    async def update_activity(self, mcp_session_id: str) -> None:
        """Update last activity timestamp for session keepalive."""
        now = datetime.now(UTC).isoformat()

        if mcp_session_id in self._active_sessions:
            self._active_sessions[mcp_session_id]["last_activity"] = now

        # While the initial INSERT is pending, an UPDATE would match 0 rows. Skip it:
        # the INSERT reads last_activity from the in-memory dict (just refreshed above)
        # and, being the session's first write, is at most a few ms stale.
        if self.database and mcp_session_id not in self._pending_saves:
            try:
                await self.database.update_mcp_session_activity(mcp_session_id)
            except Exception as e:
                logger.warning(f"Failed to update MCP session activity: {e}")

    async def validate_session(self, mcp_session_id: str) -> bool:
        """Validate that an MCP session exists and is active."""
        if mcp_session_id in self._active_sessions:
            return True

        if self.database:
            try:
                await self._await_pending_save(mcp_session_id)
                mcp_session = await self.database.get_mcp_session(mcp_session_id)
                if mcp_session:
                    self._active_sessions[mcp_session_id] = mcp_session
                    return True
            except Exception as e:
                logger.warning(f"Failed to validate MCP session: {e}")

        return False

    async def get_session_info(self, mcp_session_id: str) -> dict[str, Any] | None:
        """Get full session information."""
        if mcp_session_id in self._active_sessions:
            return self._active_sessions[mcp_session_id].copy()

        if self.database:
            try:
                return await self.database.get_mcp_session(mcp_session_id)
            except Exception as e:
                logger.warning(f"Failed to get MCP session info: {e}")

        return None

    def get_active_session_count(self) -> int:
        """Get count of active sessions in memory."""
        return len(self._active_sessions)

    async def cleanup_inactive_sessions(self, max_age_seconds: int = 3600) -> int:
        """Clean up inactive sessions from memory cache."""
        now = datetime.now(UTC)
        to_remove = []

        for session_id, session_data in self._active_sessions.items():
            raw = session_data["last_activity"]
            # Cached DB rows carry a datetime on PostgreSQL and a (possibly naive,
            # pre-#112) ISO string on SQLite; sessions created here carry an aware
            # ISO string. Normalize all of them before subtracting from aware `now`.
            last_activity = _as_aware_utc(
                datetime.fromisoformat(raw) if isinstance(raw, str) else raw
            )
            age = (now - last_activity).total_seconds()
            if age > max_age_seconds:
                to_remove.append(session_id)

        for session_id in to_remove:
            del self._active_sessions[session_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} inactive MCP sessions")

        return len(to_remove)
