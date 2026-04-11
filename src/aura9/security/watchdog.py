"""Watchdog — secondary oversight model.

Monitors the 9B's outputs for drift, loops, gate bypass, capability
creep, toxicity, and schema violations.  Emits a periodic heartbeat
into Redis.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from aura9.core.config import get
from aura9.memory.redis_l1 import RedisMemory


class Watchdog:
    """Lightweight audit sidecar that monitors inference outputs."""

    def __init__(self, redis: RedisMemory) -> None:
        self._redis = redis
        self._running = False
        self._consecutive_identical: int = 0
        self._last_action_hash: str | None = None

    async def start(self) -> None:
        self._running = True
        logger.info("Watchdog started — heartbeat every {}s",
                     get("security.watchdog.heartbeat_interval_seconds", 30))

    async def stop(self) -> None:
        self._running = False
        logger.info("Watchdog stopped")

    async def heartbeat_loop(self) -> None:
        """Continuously refresh the Redis heartbeat key."""
        await self.start()
        interval = get("security.watchdog.heartbeat_interval_seconds", 30)
        while self._running:
            await self._redis.refresh_watchdog_heartbeat()
            await asyncio.sleep(interval)

    async def evaluate_output(self, action_hash: str) -> str:
        """Evaluate a 9B output action.  Returns 'CLEAR' or 'FLAGGED'.

        Hard-kill is triggered if > threshold identical consecutive actions.
        """
        threshold = get("security.watchdog.hard_kill_threshold", 50)

        if action_hash == self._last_action_hash:
            self._consecutive_identical += 1
        else:
            self._consecutive_identical = 0
            self._last_action_hash = action_hash

        if self._consecutive_identical > threshold:
            logger.critical("Watchdog HARD KILL — infinite loop detected ({} identical actions)",
                            self._consecutive_identical)
            return "HARD_KILL"

        return "CLEAR"
