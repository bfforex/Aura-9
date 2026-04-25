"""L1 Redis Memory — session-scoped ephemeral storage."""

from __future__ import annotations

from datetime import UTC, datetime

from loguru import logger

# TTLs (seconds)
SESSION_TTL = 86400       # 24 hours
SUSPENDED_TTL = 604800    # 7 days
CHECKPOINT_TTL = 2592000  # 30 days
MCP_CALLS_TTL = 172800    # 48 hours
WATCHDOG_HEARTBEAT_TTL = 90
WATCHDOG_BUFFER_TTL = 300
RETRY_QUEUE_TTL = 3600    # 1 hour per item


class L1RedisMemory:
    """L1 Redis memory with session-scoped key namespaces."""

    def __init__(self, redis_client) -> None:
        self._r = redis_client

    # -------------------------------------------------------------------------
    # Turns (Redis Stream)
    # -------------------------------------------------------------------------

    async def add_turn(self, session_id: str, role: str, content: str) -> str:
        """Append a turn to the session stream. Returns the stream entry ID."""
        key = f"sess:{session_id}:turns"
        entry_id = await self._r.xadd(key, {"role": role, "content": content})
        await self._r.expire(key, SESSION_TTL)
        return entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)

    async def get_turns(self, session_id: str, count: int = 50) -> list[dict]:
        """Retrieve recent turns from the session stream."""
        key = f"sess:{session_id}:turns"
        entries = await self._r.xrevrange(key, count=count)
        turns = []
        for entry_id, fields in reversed(entries):
            turn = {"id": entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)}
            for k, v in fields.items():
                turn[k.decode() if isinstance(k, bytes) else k] = (
                    v.decode() if isinstance(v, bytes) else v
                )
            turns.append(turn)
        return turns

    # -------------------------------------------------------------------------
    # Scratchpad
    # -------------------------------------------------------------------------

    async def set_scratchpad(self, session_id: str, content: str) -> None:
        key = f"sess:{session_id}:scratchpad"
        await self._r.set(key, content, ex=SESSION_TTL)

    async def get_scratchpad(self, session_id: str) -> str | None:
        key = f"sess:{session_id}:scratchpad"
        val = await self._r.get(key)
        return val.decode() if isinstance(val, bytes) else val

    # -------------------------------------------------------------------------
    # Tool results
    # -------------------------------------------------------------------------

    async def set_tool_result(self, session_id: str, call_id: str, result: str) -> None:
        key = f"sess:{session_id}:tool_results:{call_id}"
        await self._r.set(key, result, ex=SESSION_TTL)

    async def get_tool_result(self, session_id: str, call_id: str) -> str | None:
        key = f"sess:{session_id}:tool_results:{call_id}"
        val = await self._r.get(key)
        return val.decode() if isinstance(val, bytes) else val

    # -------------------------------------------------------------------------
    # Session metadata
    # -------------------------------------------------------------------------

    async def update_metadata(self, session_id: str) -> None:
        key = f"sess:{session_id}:metadata"
        now = datetime.now(UTC).isoformat()
        await self._r.hset(key, mapping={"last_active": now, "session_id": session_id})
        # Only set created_at if not already present
        if not await self._r.hexists(key, "created_at"):
            await self._r.hset(key, "created_at", now)
        await self._r.expire(key, SESSION_TTL)

    async def get_metadata(self, session_id: str) -> dict | None:
        key = f"sess:{session_id}:metadata"
        data = await self._r.hgetall(key)
        if not data:
            return None
        return {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in data.items()
        }

    # -------------------------------------------------------------------------
    # ASD state
    # -------------------------------------------------------------------------

    async def set_asd_state(self, state_json: str) -> None:
        await self._r.set("asd:state", state_json)  # No TTL

    async def get_asd_state(self) -> str | None:
        val = await self._r.get("asd:state")
        return val.decode() if isinstance(val, bytes) else val

    # -------------------------------------------------------------------------
    # Checkpoints
    # -------------------------------------------------------------------------

    async def save_checkpoint(self, task_id: str, ckpt_id: str, checkpoint_json: str) -> None:
        key = f"asd:checkpoint:{task_id}:{ckpt_id}"
        await self._r.set(key, checkpoint_json, ex=CHECKPOINT_TTL)

    async def get_checkpoint(self, task_id: str, ckpt_id: str) -> str | None:
        key = f"asd:checkpoint:{task_id}:{ckpt_id}"
        val = await self._r.get(key)
        return val.decode() if isinstance(val, bytes) else val

    # -------------------------------------------------------------------------
    # MCP call counters
    # -------------------------------------------------------------------------

    async def increment_mcp_calls(self, server_id: str) -> int:
        from datetime import date  # noqa: PLC0415
        today = date.today().isoformat()
        key = f"mcp:calls:{server_id}:{today}"
        count = await self._r.incr(key)
        await self._r.expire(key, MCP_CALLS_TTL)
        return int(count)

    # -------------------------------------------------------------------------
    # Watchdog
    # -------------------------------------------------------------------------

    async def refresh_watchdog_heartbeat(self) -> None:
        await self._r.set("watchdog:heartbeat", "1", ex=WATCHDOG_HEARTBEAT_TTL)

    async def append_watchdog_buffer(self, content: str) -> None:
        key = "watchdog:buffer"
        await self._r.append(key, content)
        await self._r.expire(key, WATCHDOG_BUFFER_TTL)

    # -------------------------------------------------------------------------
    # ISEC progress
    # -------------------------------------------------------------------------

    async def set_isec_progress(self, progress_json: str) -> None:
        await self._r.set("isec:progress", progress_json)  # No TTL

    async def get_isec_progress(self) -> str | None:
        val = await self._r.get("isec:progress")
        return val.decode() if isinstance(val, bytes) else val

    # -------------------------------------------------------------------------
    # Session suspension
    # -------------------------------------------------------------------------

    async def suspend_session(self, session_id: str) -> None:
        """Extend all session keys to SUSPENDED_TTL (7 days)."""
        patterns = [
            f"sess:{session_id}:turns",
            f"sess:{session_id}:scratchpad",
            f"sess:{session_id}:metadata",
        ]
        for key in patterns:
            try:
                await self._r.expire(key, SUSPENDED_TTL)
            except Exception as exc:
                logger.warning(f"L1: suspend_session key={key} failed: {exc}")

        # Also extend tool_results keys (scan pattern)
        try:
            tool_pattern = f"sess:{session_id}:tool_results:*"
            cursor = 0
            while True:
                cursor, keys = await self._r.scan(cursor, match=tool_pattern, count=100)
                for key in keys:
                    await self._r.expire(key, SUSPENDED_TTL)
                if cursor == 0:
                    break
        except Exception as exc:
            logger.warning(f"L1: suspend_session tool_results scan failed: {exc}")

    # -------------------------------------------------------------------------
    # FalkorDB retry queue
    # -------------------------------------------------------------------------

    async def push_falkordb_retry(self, item_json: str) -> None:
        key = "falkordb:retry_queue"
        await self._r.rpush(key, item_json)

    async def pop_falkordb_retry(self) -> str | None:
        key = "falkordb:retry_queue"
        val = await self._r.lpop(key)
        return val.decode() if isinstance(val, bytes) else val

    async def falkordb_retry_len(self) -> int:
        key = "falkordb:retry_queue"
        return int(await self._r.llen(key))

    async def set_expiry(self, key: str, seconds: int) -> None:
        """Set TTL on a Redis key."""
        await self._r.expire(key, seconds)

    # -------------------------------------------------------------------------
    # Continuation
    # -------------------------------------------------------------------------

    async def set_continuation(self, session_id: str, turn_id: str, content: str) -> None:
        key = f"sess:{session_id}:continuation:{turn_id}"
        await self._r.set(key, content, ex=SESSION_TTL)

    async def get_continuation(self, session_id: str, turn_id: str) -> str | None:
        key = f"sess:{session_id}:continuation:{turn_id}"
        val = await self._r.get(key)
        return val.decode() if isinstance(val, bytes) else val
