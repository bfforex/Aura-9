"""Aura State Daemon (ASD).

Persistent JSON state tree maintained in Redis (primary) with a
FalkorDB shadow copy.  Updated exclusively via Precision Planner Mode.
"""

from __future__ import annotations

import enum
from typing import Any

from loguru import logger

from aura9.memory.redis_l1 import RedisMemory


class ASDStatus(enum.StrEnum):
    EXECUTING = "EXECUTING"
    PAUSED = "PAUSED"
    BLOCKED = "BLOCKED"
    CORRECTING = "CORRECTING"
    SUSPENDED = "SUSPENDED"


def default_state() -> dict[str, Any]:
    """Return a blank ASD state tree conforming to the Precision Planner schema."""
    return {
        "asd_update": {
            "current_objective": "",
            "status": ASDStatus.PAUSED.value,
            "active_subtasks": [],
            "completed_subtasks": [],
            "blocked_by": None,
            "confidence": 0.0,
            "next_action": "",
            "failure_class": None,
            "checkpoint_required": False,
            "tais_status": "NORMAL",
        }
    }


SCHEMA_KEYS = {
    "current_objective",
    "status",
    "active_subtasks",
    "completed_subtasks",
    "blocked_by",
    "confidence",
    "next_action",
    "failure_class",
    "checkpoint_required",
    "tais_status",
}


def validate_state(state: dict[str, Any]) -> bool:
    """Validate that a state dict conforms to the immutable Precision Planner schema."""
    update = state.get("asd_update")
    if not isinstance(update, dict):
        return False
    return set(update.keys()) == SCHEMA_KEYS


class AuraStateDaemon:
    """Manages ASD state reads/writes against Redis."""

    def __init__(self, redis: RedisMemory) -> None:
        self._redis = redis

    async def read(self) -> dict[str, Any]:
        state = await self._redis.read_asd_state()
        return state if state else default_state()

    async def write(self, state: dict[str, Any]) -> None:
        if not validate_state(state):
            logger.error("ASD state validation failed — STATE_FAIL")
            raise ValueError("ASD state does not conform to Precision Planner schema")
        await self._redis.write_asd_state(state)
        logger.info("ASD state updated — status={}", state["asd_update"]["status"])

    async def checkpoint(self, task_id: str) -> None:
        state = await self.read()
        await self._redis.write_asd_checkpoint(task_id, state)
        logger.info("ASD checkpoint saved — task_id={}", task_id)
