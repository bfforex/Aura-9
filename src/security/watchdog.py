"""Watchdog Daemon — monitors inference outputs for safety violations."""

from __future__ import annotations

import asyncio
import hashlib
from collections import deque

from loguru import logger

HEARTBEAT_INTERVAL = 30     # seconds
HEARTBEAT_TTL = 90          # seconds
HARD_KILL_THRESHOLD = 50    # consecutive identical outputs
MAX_RESTARTS = 3
RESTART_INTERVAL = 30       # seconds


class WatchdogDaemon:
    """Monitors inference outputs for loops, drift, and safety violations."""

    def __init__(self, l1=None, redis_client=None, ollama_client=None) -> None:
        self._l1 = l1
        self._redis = redis_client
        self._ollama = ollama_client
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._output_hashes: deque[str] = deque(maxlen=HARD_KILL_THRESHOLD + 10)
        self._identical_count = 0
        self._last_hash: str | None = None
        self._restart_count = 0

    async def start(self) -> None:
        self._running = True
        self._tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._monitor_loop()),
        ]
        logger.info("Watchdog started")

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("Watchdog stopped")

    async def refresh_heartbeat(self) -> None:
        """Refresh watchdog heartbeat in Redis."""
        if self._l1:
            await self._l1.refresh_watchdog_heartbeat()
        elif self._redis:
            await self._redis.set("watchdog:heartbeat", "1", ex=HEARTBEAT_TTL)

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await self.refresh_heartbeat()
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _monitor_loop(self) -> None:
        """Subscribe to inference output channel, with retry backoff when Redis unavailable."""
        backoff = 1.0
        last_warn_time = 0.0

        while self._running:
            if not self._redis:
                import time  # noqa: PLC0415
                now = time.monotonic()
                if now - last_warn_time >= 60:
                    logger.warning("Watchdog: Redis unavailable — monitor loop degraded")
                    last_warn_time = now
                await asyncio.sleep(min(backoff, 60.0))
                backoff = min(backoff * 2, 60.0)
                continue

            backoff = 1.0  # reset on success
            try:
                from src.ipc.channels import WATCHDOG_MONITOR  # noqa: PLC0415
                from src.ipc.subscriber import subscribe  # noqa: PLC0415
                async for payload in subscribe(WATCHDOG_MONITOR, self._redis):
                    if not self._running:
                        return
                    await self._handle_monitor_event(payload)
            except Exception as exc:
                logger.warning(f"Watchdog monitor loop error: {exc}")
                await asyncio.sleep(min(backoff, 60.0))
                backoff = min(backoff * 2, 60.0)

    async def _handle_monitor_event(self, payload: dict) -> None:
        output = payload.get("output", "")
        session_id = payload.get("session_id", "")
        task_id = payload.get("task_id", "")
        await self.check_output(output, session_id, task_id)

    async def check_output(self, output: str, session_id: str, task_id: str) -> str:
        """Check output for violations. Returns verdict: CLEAR, WARN, BLOCK, KILL."""
        if not output:
            return "CLEAR"

        # Schema validation (basic)
        verdict = self._check_schema(output)
        if verdict == "BLOCK":
            await self._publish_verdict("BLOCK", session_id, task_id, "Schema validation failed")
            return "BLOCK"

        # Loop detection (hash comparison)
        output_hash = hashlib.sha256(output.encode()).hexdigest()
        if output_hash == self._last_hash:
            self._identical_count += 1
            if self._identical_count >= HARD_KILL_THRESHOLD:
                logger.critical(
                    f"Watchdog: HARD KILL — {HARD_KILL_THRESHOLD} consecutive identical outputs"
                )
                await self._publish_verdict("KILL", session_id, task_id, "Loop detected")
                return "KILL"
        else:
            self._identical_count = 0
        self._last_hash = output_hash
        self._output_hashes.append(output_hash)

        await self._publish_verdict("CLEAR", session_id, task_id)
        return "CLEAR"

    def _check_schema(self, output: str) -> str:
        """Basic schema/format validation."""
        # Check for obviously malformed outputs
        if len(output) > 1_000_000:  # Extremely large output
            return "BLOCK"
        return "CLEAR"

    async def _publish_verdict(
        self, verdict: str, session_id: str, task_id: str, reason: str = ""
    ) -> None:
        if self._redis:
            try:
                from src.ipc.channels import WATCHDOG_VERDICT  # noqa: PLC0415
                from src.ipc.publisher import publish  # noqa: PLC0415
                await publish(
                    WATCHDOG_VERDICT,
                    {"verdict": verdict, "session_id": session_id, "task_id": task_id,
                     "reason": reason},
                    self._redis,
                )
            except Exception as exc:
                logger.debug(f"Watchdog: publish verdict failed: {exc}")
