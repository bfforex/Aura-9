"""Human Gate — escalation points requiring user confirmation."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from loguru import logger

COALESCE_WINDOW = 60     # seconds
RENOTIFY_MINUTES = 30
SUSPEND_MINUTES = 120
MAX_GATES_PER_MISSION = 5


@dataclass
class GateResponse:
    approved: bool
    gate_id: str
    response_text: str = ""
    timed_out: bool = False


class HumanGate:
    """Manages human escalation gates with coalescing and timeout ladder."""

    def __init__(self, redis_client=None, gate_count: int = 0) -> None:
        self._redis = redis_client
        self._gate_count = gate_count
        self._pending_gates: dict[str, asyncio.Future] = {}

    async def request(
        self,
        question: str,
        context: str,
        task_id: str,
        session_id: str,
    ) -> GateResponse:
        """Request human confirmation at a decision gate."""
        if self._gate_count >= MAX_GATES_PER_MISSION:
            logger.warning(f"HumanGate: max gates ({MAX_GATES_PER_MISSION}) reached — escalating")
            return GateResponse(
                approved=False,
                gate_id="ESCALATED",
                response_text="Maximum gate count reached",
                timed_out=False,
            )

        gate_id = str(uuid.uuid4())[:8]
        self._gate_count += 1

        payload = {
            "gate_id": gate_id,
            "task_id": task_id,
            "session_id": session_id,
            "question": question,
            "context": context,
            "created_at": datetime.now(UTC).isoformat(),
        }

        await self._emit_gate_request(payload)
        logger.info(f"HumanGate: gate {gate_id} requested for task {task_id}")

        # Wait for response with timeout ladder
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_gates[gate_id] = future

        try:
            # First timeout: re-notify at 30 min
            response = await asyncio.wait_for(
                asyncio.shield(future),
                timeout=RENOTIFY_MINUTES * 60,
            )
            return GateResponse(approved=response.get("approved", False), gate_id=gate_id)
        except TimeoutError:
            logger.warning(
                f"HumanGate: gate {gate_id} timed out at {RENOTIFY_MINUTES}min — re-notifying"
            )
            await self._emit_gate_request({**payload, "renotify": True})

            try:
                response = await asyncio.wait_for(
                    future,
                    timeout=(SUSPEND_MINUTES - RENOTIFY_MINUTES) * 60,
                )
                return GateResponse(approved=response.get("approved", False), gate_id=gate_id)
            except TimeoutError:
                logger.warning(f"HumanGate: gate {gate_id} suspended at {SUSPEND_MINUTES}min")
                return GateResponse(
                    approved=False,
                    gate_id=gate_id,
                    timed_out=True,
                    response_text="Gate timed out — session suspended",
                )
        finally:
            self._pending_gates.pop(gate_id, None)

    def respond(self, gate_id: str, approved: bool, response_text: str = "") -> None:
        """Provide a response to a pending gate (called from IPC subscriber)."""
        future = self._pending_gates.get(gate_id)
        if future and not future.done():
            future.set_result({"approved": approved, "response_text": response_text})

    async def _emit_gate_request(self, payload: dict) -> None:
        if self._redis:
            try:
                from src.ipc.channels import GATE_REQUEST  # noqa: PLC0415
                from src.ipc.publisher import publish  # noqa: PLC0415
                await publish(GATE_REQUEST, payload, self._redis)
            except Exception as exc:
                logger.warning(f"HumanGate: emit failed: {exc}")
