"""ASD — Aura State Daemon: manages task lifecycle state."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum

from loguru import logger

# State-change coalescing window (seconds)
COALESCE_WINDOW = 5.0

_VALID_STATUSES = {
    "CREATED", "PLANNED", "EXECUTING", "CORRECTING", "VERIFYING",
    "DELIVERED", "PAUSED", "BLOCKED", "SUSPENDED", "ESCALATED", "FAILED", "IDLE",
}
_VALID_TAIS_STATUSES = {"NORMAL", "THROTTLE", "COOLDOWN", "EMERGENCY", "SENSOR_FAIL"}
_TERMINAL_STATES = {"DELIVERED", "ESCALATED", "FAILED"}


class ASDStatus(StrEnum):
    CREATED = "CREATED"
    PLANNED = "PLANNED"
    EXECUTING = "EXECUTING"
    CORRECTING = "CORRECTING"
    VERIFYING = "VERIFYING"
    DELIVERED = "DELIVERED"
    PAUSED = "PAUSED"
    BLOCKED = "BLOCKED"
    SUSPENDED = "SUSPENDED"
    ESCALATED = "ESCALATED"
    FAILED = "FAILED"
    IDLE = "IDLE"


class PrecisionPlannerValidator:
    """Validates ASD state updates against the Precision Planner schema."""

    REQUIRED_FIELDS = {
        "task_id", "session_id", "current_objective", "status",
        "active_subtasks", "completed_subtasks", "blocked_by",
        "confidence", "next_action", "failure_class", "checkpoint_required",
        "tais_status", "tais_halt_reason",
    }

    def validate(self, data: dict) -> bool:
        """Strict schema validation. Returns True if valid."""
        if not isinstance(data, dict):
            return False

        # Must have all required fields
        if not self.REQUIRED_FIELDS.issubset(data.keys()):
            missing = self.REQUIRED_FIELDS - data.keys()
            logger.debug(f"PrecisionPlanner: missing fields: {missing}")
            return False

        if data.get("status") not in _VALID_STATUSES:
            logger.debug(f"PrecisionPlanner: invalid status: {data.get('status')}")
            return False

        if data.get("tais_status") not in _VALID_TAIS_STATUSES:
            logger.debug(f"PrecisionPlanner: invalid tais_status: {data.get('tais_status')}")
            return False

        if not isinstance(data.get("active_subtasks"), list):
            return False
        if not isinstance(data.get("completed_subtasks"), list):
            return False
        if not isinstance(data.get("checkpoint_required"), bool):
            return False

        conf = data.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            return False

        return True


class AuraStateDaemon:
    """Manages ASD state with coalescing, checkpointing, and terminal-state persistence."""

    def __init__(self, l1=None, l3=None, config=None) -> None:
        self._l1 = l1
        self._l3 = l3
        self._config = config
        self._validator = PrecisionPlannerValidator()
        self._pending_update: dict | None = None
        self._coalesce_task: asyncio.Task | None = None
        self._schema_fail_count = 0

    async def update_state(self, update: dict) -> None:
        """Validate and write ASD state with 5-second coalescing window."""
        asd_data = update.get("asd_update", update)

        if not self._validator.validate(asd_data):
            self._schema_fail_count += 1
            if self._schema_fail_count >= 2:
                logger.error("ASD: STATE_FAIL — second schema validation failure")
                self._schema_fail_count = 0
            logger.warning("ASD: invalid state update, dropping")
            return

        self._schema_fail_count = 0
        self._pending_update = update

        # Coalesce: reset the window if already pending
        if self._coalesce_task and not self._coalesce_task.done():
            self._coalesce_task.cancel()

        self._coalesce_task = asyncio.create_task(self._flush_after_window())

    async def _flush_after_window(self) -> None:
        await asyncio.sleep(COALESCE_WINDOW)
        if self._pending_update:
            await self._flush()

    async def _flush(self) -> None:
        if not self._pending_update:
            return

        state = self._pending_update
        self._pending_update = None

        state_json = json.dumps(state)

        if self._l1:
            await self._l1.set_asd_state(state_json)

        asd_data = state.get("asd_update", state)
        status = asd_data.get("status", "")

        # Create checkpoint on each flush
        task_id = asd_data.get("task_id", "unknown")
        if self._l1:
            ckpt_id = await self.create_checkpoint(task_id)
            await self._l1.save_checkpoint(task_id, ckpt_id, state_json)

        # Terminal state → persist to L3 asynchronously
        if status in _TERMINAL_STATES and self._l3:
            asyncio.create_task(self._persist_terminal(asd_data))

        logger.debug(f"ASD: flushed state task_id={task_id} status={status}")

    async def _persist_terminal(self, asd_data: dict) -> None:
        try:
            if self._l3:
                await self._l3.create_entity(
                    "TaskCompletion",
                    {
                        "task_id": asd_data.get("task_id", ""),
                        "status": asd_data.get("status", ""),
                        "confidence": asd_data.get("confidence", 0.0),
                        "failure_class": asd_data.get("failure_class"),
                    },
                    asd_data.get("session_id", ""),
                )
            # Expire the asd:state key after checkpoint_ttl_days
            if self._l1:
                ttl_days = 30
                if self._config is not None:
                    continuity = getattr(self._config, "continuity", None)
                    if continuity is not None:
                        ttl_days = getattr(continuity, "checkpoint_ttl_days", 30)
                await self.expire_terminal_state(ttl_days)
        except Exception as exc:
            logger.warning(f"ASD: terminal persist to L3 failed: {exc}")

    async def expire_terminal_state(self, days: int) -> None:
        """Set TTL on the asd:state key after a terminal state is reached."""
        if not self._l1:
            return
        ttl_seconds = days * 86400
        try:
            await self._l1.set_expiry("asd:state", ttl_seconds)
            logger.debug(f"ASD: asd:state TTL set to {days} days")
        except Exception as exc:
            logger.warning(f"ASD: expire_terminal_state failed: {exc}")

    async def get_state(self) -> dict | None:
        if not self._l1:
            return None
        raw = await self._l1.get_asd_state()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    async def create_checkpoint(self, task_id: str) -> str:
        """Create and store a checkpoint. Returns ckpt_id."""
        ckpt_id = str(uuid.uuid4())[:8]
        ts = datetime.now(UTC).isoformat()

        # Snapshot _pending_update at the start to avoid race with _flush
        state = self._pending_update

        if self._l1 and state:
            ckpt_data = {
                "ckpt_id": ckpt_id,
                "task_id": task_id,
                "created_at": ts,
                "state": state,
            }
            await self._l1.save_checkpoint(task_id, ckpt_id, json.dumps(ckpt_data))

        return ckpt_id

    async def force_flush(self) -> None:
        """Immediately flush any pending state (bypass coalesce window)."""
        if self._coalesce_task and not self._coalesce_task.done():
            self._coalesce_task.cancel()
        await self._flush()
