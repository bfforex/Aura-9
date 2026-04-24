"""Continuity Engine — checkpointing, health checks, and snapshot management."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from loguru import logger

CHECKPOINT_INTERVAL = 15 * 60    # 15 minutes
HEALTH_CHECK_INTERVAL = 5 * 60   # 5 minutes
SNAPSHOT_INTERVAL = 24 * 3600    # 24 hours
STALE_HOURS = 24


class ContinuityEngine:
    """Manages checkpoints, health, and crash recovery."""

    def __init__(self, l1=None, l2=None, l3=None, asd=None) -> None:
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3
        self._asd = asd
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._running = True
        self._tasks = [
            asyncio.create_task(self._checkpoint_loop()),
            asyncio.create_task(self._health_loop()),
            asyncio.create_task(self._snapshot_loop()),
        ]
        logger.info("ContinuityEngine started")

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("ContinuityEngine stopped")

    async def _checkpoint_loop(self) -> None:
        while self._running:
            await asyncio.sleep(CHECKPOINT_INTERVAL)
            try:
                await self._create_checkpoint()
            except Exception as exc:
                logger.error(f"Continuity: checkpoint failed: {exc}")

    async def _health_loop(self) -> None:
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            try:
                await self._stale_session_cleanup()
            except Exception as exc:
                logger.error(f"Continuity: health check failed: {exc}")

    async def _snapshot_loop(self) -> None:
        while self._running:
            await asyncio.sleep(SNAPSHOT_INTERVAL)
            try:
                await self._take_snapshots()
            except Exception as exc:
                logger.error(f"Continuity: snapshot failed: {exc}")

    async def _create_checkpoint(self) -> None:
        if not self._asd or not self._l1:
            return
        state = await self._asd.get_state()
        if state:
            asd_data = state.get("asd_update", state)
            task_id = asd_data.get("task_id", "unknown")
            await self._asd.create_checkpoint(task_id)
            logger.debug(f"Continuity: checkpoint created for task {task_id}")

    async def _stale_session_cleanup(self) -> None:
        """Clean up or suspend sessions inactive for > 24 hours."""
        if not self._l1:
            return
        # In production, would scan session metadata keys and check last_active
        logger.debug("Continuity: stale session health check")

    async def _take_snapshots(self) -> None:
        """Trigger FalkorDB and Qdrant snapshots."""
        logger.info("Continuity: taking storage snapshots")
        # Would trigger actual snapshot APIs

    async def cold_start_resume(self) -> dict | None:
        """Restore from latest checkpoint on cold start."""
        if not self._l1:
            return None
        raw = await self._l1.get_asd_state()
        if raw:
            try:
                state = json.loads(raw)
                logger.info("Continuity: resumed from cold-start checkpoint")
                return state
            except Exception as exc:
                logger.warning(f"Continuity: cold start parse failed: {exc}")
        return None

    @staticmethod
    def is_stale(last_active_iso: str) -> bool:
        """Return True if last_active is more than STALE_HOURS ago."""
        try:
            last = datetime.fromisoformat(last_active_iso)
            now = datetime.now(UTC)
            if last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            delta = now - last
            return delta.total_seconds() > STALE_HOURS * 3600
        except Exception:
            return False
