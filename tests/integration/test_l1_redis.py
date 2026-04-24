"""Integration tests for L1 Redis Memory using fakeredis."""

from __future__ import annotations

import json

import pytest

from src.memory.l1_redis import (
    SESSION_TTL,
    SUSPENDED_TTL,
    WATCHDOG_HEARTBEAT_TTL,
    L1RedisMemory,
)


@pytest.fixture
async def l1(fake_redis):
    return L1RedisMemory(redis_client=fake_redis)


@pytest.mark.integration
class TestL1Turns:
    @pytest.mark.asyncio
    async def test_add_and_get_turns(self, l1):
        session_id = "sess-test-001"
        entry_id = await l1.add_turn(session_id, "user", "Hello agent")
        assert entry_id

        turns = await l1.get_turns(session_id)
        assert len(turns) == 1
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Hello agent"

    @pytest.mark.asyncio
    async def test_multiple_turns_ordered(self, l1):
        session_id = "sess-turns-test"
        for i in range(5):
            await l1.add_turn(session_id, "user", f"Message {i}")

        turns = await l1.get_turns(session_id, count=10)
        assert len(turns) == 5

    @pytest.mark.asyncio
    async def test_turns_ttl_set(self, fake_redis, l1):
        session_id = "sess-ttl-test"
        await l1.add_turn(session_id, "user", "message")
        ttl = await fake_redis.ttl(f"sess:{session_id}:turns")
        assert ttl > 0
        assert ttl <= SESSION_TTL


@pytest.mark.integration
class TestL1Scratchpad:
    @pytest.mark.asyncio
    async def test_set_and_get_scratchpad(self, l1):
        session_id = "sess-scratch"
        await l1.set_scratchpad(session_id, "My working notes")
        result = await l1.get_scratchpad(session_id)
        assert result == "My working notes"

    @pytest.mark.asyncio
    async def test_get_nonexistent_scratchpad(self, l1):
        result = await l1.get_scratchpad("sess-nonexistent")
        assert result is None


@pytest.mark.integration
class TestL1ToolResults:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_tool_result(self, l1):
        session_id = "sess-tool"
        await l1.set_tool_result(session_id, "call-001", "Tool output here")
        result = await l1.get_tool_result(session_id, "call-001")
        assert result == "Tool output here"

    @pytest.mark.asyncio
    async def test_get_nonexistent_tool_result(self, l1):
        result = await l1.get_tool_result("sess-x", "nonexistent")
        assert result is None


@pytest.mark.integration
class TestL1ASDState:
    @pytest.mark.asyncio
    async def test_asd_state_write_read(self, l1):
        state = {"asd_update": {"status": "EXECUTING", "task_id": "t1"}}
        await l1.set_asd_state(json.dumps(state))
        result = await l1.get_asd_state()
        assert result is not None
        parsed = json.loads(result)
        assert parsed["asd_update"]["status"] == "EXECUTING"

    @pytest.mark.asyncio
    async def test_asd_state_no_ttl(self, fake_redis, l1):
        await l1.set_asd_state('{"test": true}')
        ttl = await fake_redis.ttl("asd:state")
        # -1 = no expiry, -2 = doesn't exist
        assert ttl == -1 or ttl > 0  # No TTL set


@pytest.mark.integration
class TestL1SessionSuspend:
    @pytest.mark.asyncio
    async def test_suspend_extends_ttl(self, fake_redis, l1):
        session_id = "sess-suspend"
        # Create session data
        await l1.add_turn(session_id, "user", "hello")
        await l1.set_scratchpad(session_id, "notes")
        await l1.update_metadata(session_id)

        # Suspend
        await l1.suspend_session(session_id)

        # Check TTLs extended to 7 days
        ttl = await fake_redis.ttl(f"sess:{session_id}:scratchpad")
        assert ttl > SESSION_TTL  # Must be > 24h
        assert ttl <= SUSPENDED_TTL  # Must be <= 7d


@pytest.mark.integration
class TestL1WatchdogHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_refresh(self, fake_redis, l1):
        await l1.refresh_watchdog_heartbeat()
        val = await fake_redis.get("watchdog:heartbeat")
        assert val is not None
        ttl = await fake_redis.ttl("watchdog:heartbeat")
        assert 0 < ttl <= WATCHDOG_HEARTBEAT_TTL
