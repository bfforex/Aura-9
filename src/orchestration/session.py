"""Session Manager — creates and manages agent sessions."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from loguru import logger

SESSION_TTL_HOURS = 24


class SessionManager:
    """Creates and tracks agent sessions."""

    def __init__(self, l1=None) -> None:
        self._l1 = l1

    def create_session(self) -> str:
        """Create a new session ID. Format: sess-{uuid4}-{unix_timestamp}."""
        ts = int(time.time())
        session_id = f"sess-{uuid.uuid4()}-{ts}"
        return session_id

    async def get_session(self, session_id: str) -> dict | None:
        """Retrieve session metadata."""
        if self._l1:
            return await self._l1.get_metadata(session_id)
        return None

    async def update_session(self, session_id: str) -> None:
        """Update session last_active timestamp."""
        if self._l1:
            await self._l1.update_metadata(session_id)

    async def end_session(self, session_id: str) -> None:
        """Mark session as ended."""
        if self._l1:
            meta = await self._l1.get_metadata(session_id)
            if meta:
                logger.info(f"Session ended: {session_id}")

    def is_stale(self, session_id: str, last_active_iso: str | None = None) -> bool:
        """Return True if session is stale (last_active > 24h ago)."""
        if not last_active_iso:
            return False
        try:
            last = datetime.fromisoformat(last_active_iso)
            now = datetime.now(UTC)
            if last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            delta = now - last
            return delta.total_seconds() > SESSION_TTL_HOURS * 3600
        except Exception:
            return False
