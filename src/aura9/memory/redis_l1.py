"""L1 — Episodic Memory + ASD State (Redis).

Manages session-scoped working context, the Aura State Daemon live
state tree, and ASD checkpoints.
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis
from loguru import logger

from aura9.core.config import get


class RedisMemory:
    """Async Redis wrapper for L1 episodic memory and ASD state."""

    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        url = get("memory.redis.url", "redis://localhost:6379/0")
        self._client = aioredis.from_url(url, decode_responses=True)
        logger.info("L1 Redis connected — {}", url)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> aioredis.Redis:
        assert self._client is not None, "RedisMemory not connected"  # noqa: S101
        return self._client

    # -- ASD state -------------------------------------------------------------

    async def read_asd_state(self) -> dict[str, Any] | None:
        raw = await self.client.get("asd:state")
        return json.loads(raw) if raw else None

    async def write_asd_state(self, state: dict[str, Any]) -> None:
        await self.client.set("asd:state", json.dumps(state))
        logger.debug("ASD state written to Redis")

    async def write_asd_checkpoint(self, task_id: str, state: dict[str, Any]) -> None:
        ttl = get("memory.redis.asd_checkpoint_ttl_days", 7) * 86400
        key = f"asd:checkpoint:{task_id}"
        await self.client.setex(key, ttl, json.dumps(state))
        logger.debug("ASD checkpoint written — {}", key)

    # -- session data ----------------------------------------------------------

    async def append_turn(self, session_id: str, turn: dict[str, Any]) -> None:
        key = f"sess:{session_id}:turns"
        await self.client.rpush(key, json.dumps(turn))
        ttl = get("memory.redis.session_turn_ttl_seconds", 1800)
        await self.client.expire(key, ttl)

    async def store_tool_result(self, session_id: str, result: dict[str, Any]) -> None:
        key = f"sess:{session_id}:tool_results"
        await self.client.rpush(key, json.dumps(result))
        ttl = get("memory.redis.tool_result_ttl_seconds", 300)
        await self.client.expire(key, ttl)

    # -- watchdog heartbeat ----------------------------------------------------

    async def refresh_watchdog_heartbeat(self) -> None:
        ttl = get("security.watchdog.heartbeat_ttl_seconds", 90)
        await self.client.setex("watchdog:heartbeat", ttl, "alive")

    async def watchdog_heartbeat_age(self) -> int | None:
        """Return remaining TTL on the heartbeat key (None if expired)."""
        ttl = await self.client.ttl("watchdog:heartbeat")
        return ttl if ttl > 0 else None

    # -- health ----------------------------------------------------------------

    async def ping(self) -> bool:
        try:
            return await self.client.ping()
        except Exception:
            return False
