"""
MCP Session Manager - Maps MCP session IDs to engine sessions.

The MCP protocol uses MCP-Session-Id headers for stateful communication.
This manager:
1. Tracks MCP session IDs from the initialize handshake
2. Maps them to internal SessionIntelligenceEngine sessions
3. Handles session lifecycle (creation, resume, cleanup)
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from persistence.base import DatabaseBackend as Database

logger = logging.getLogger(__name__)


class MCPSessionManager:
    """Manages MCP session IDs and their mapping to engine sessions."""

    def __init__(self, database: Database | None = None) -> None:
        self.database = database
        self._active_sessions: dict[str, dict[str, Any]] = {}

    async def create_mcp_session(self, client_info: dict[str, Any] | None = None) -> str:
        """Create a new MCP session."""
        mcp_session_id = f"mcp-{uuid.uuid4().hex[:16]}"
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
            try:
                await self.database.save_mcp_session(session_data)
            except Exception as e:
                logger.warning(f"Failed to persist MCP session: {e}")

        logger.info(f"Created MCP session: {mcp_session_id}")
        return mcp_session_id

    async def get_engine_session_id(self, mcp_session_id: str) -> str | None:
        """Get the engine session ID for an MCP session."""
        if mcp_session_id in self._active_sessions:
            return self._active_sessions[mcp_session_id].get("engine_session_id")

        if self.database:
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
                await self.database.link_mcp_to_engine_session(mcp_session_id, engine_session_id)
            except Exception as e:
                logger.warning(f"Failed to persist engine session link: {e}")

        logger.info(f"Linked MCP session {mcp_session_id} to engine {engine_session_id}")

    async def update_activity(self, mcp_session_id: str) -> None:
        """Update last activity timestamp for session keepalive."""
        now = datetime.now(UTC).isoformat()

        if mcp_session_id in self._active_sessions:
            self._active_sessions[mcp_session_id]["last_activity"] = now

        if self.database:
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
            last_activity = datetime.fromisoformat(session_data["last_activity"])
            age = (now - last_activity).total_seconds()
            if age > max_age_seconds:
                to_remove.append(session_id)

        for session_id in to_remove:
            del self._active_sessions[session_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} inactive MCP sessions")

        return len(to_remove)
