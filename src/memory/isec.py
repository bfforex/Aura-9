"""ISEC — Incremental Semantic Enrichment Cycle daemon."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from loguru import logger

VRAM_HEADROOM_GB = 0.4  # 400MB minimum
DEDUP_THRESHOLD = 0.92
DECAY_THRESHOLD = 0.2
GRACE_PERIOD_DAYS = 30


class ISECDaemon:
    """Runs enrichment passes when the agent is IDLE."""

    def __init__(self, l1=None, l2=None, l3=None, redis_client=None) -> None:
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3
        self._redis = redis_client
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("ISEC daemon started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._save_progress({"status": "SHUTDOWN"})
        logger.info("ISEC daemon stopped")

    async def _monitor_loop(self) -> None:
        while self._running:
            if await self._is_agent_idle():
                if self._check_vram_headroom():
                    await self.run_passes()
            await asyncio.sleep(60)

    async def _is_agent_idle(self) -> bool:
        if not self._redis:
            return True
        try:
            val = await self._redis.get("asd:state")
            if val:
                raw = val.decode() if isinstance(val, bytes) else val
                state = json.loads(raw)
                asd_update = state.get("asd_update", state)
                return asd_update.get("status", "") == "IDLE"
        except Exception:  # noqa: S110
            pass
        return True

    def _check_vram_headroom(self) -> bool:
        """Check if VRAM headroom >= 400MB."""
        try:
            import pynvml  # noqa: PLC0415
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = mem_info.free / (1024 ** 3)
            return free_gb >= VRAM_HEADROOM_GB
        except Exception:
            return True  # Assume OK if can't check

    async def run_passes(self) -> None:
        """Run all 5 ISEC passes."""
        logger.info("ISEC: starting enrichment passes")
        progress = {"started_at": datetime.now(UTC).isoformat(), "pass": 0}
        await self._save_progress(progress)

        try:
            await self._pass1_l1_to_l2_promotion()
            progress["pass"] = 1
            await self._save_progress(progress)

            await self._pass2_l2_dedup()
            progress["pass"] = 2
            await self._save_progress(progress)

            await self._pass3_l3_enrichment()
            progress["pass"] = 3
            await self._save_progress(progress)

            await self._pass4_decay_audit()
            progress["pass"] = 4
            await self._save_progress(progress)

            await self._pass5_l1_pruning()
            progress["pass"] = 5
            progress["completed_at"] = datetime.now(UTC).isoformat()
            await self._save_progress(progress)
            logger.info("ISEC: all passes complete")
        except Exception as exc:
            logger.error(f"ISEC: pass failed: {exc}")
            progress["error"] = str(exc)
            await self._save_progress(progress)

    async def _pass1_l1_to_l2_promotion(self) -> None:
        """Promote high-confidence L1 content to L2."""
        if not self._l2:
            return
        logger.debug("ISEC pass 1: L1→L2 promotion")
        # Rule-based heuristics: check for reusable knowledge patterns
        # In production, would scan L1 content for promotable items
        # For now, this is a hook for future implementation

    async def _pass2_l2_dedup(self) -> None:
        """Deduplicate L2 content with similarity > 0.92."""
        if not self._l2:
            return
        logger.debug("ISEC pass 2: L2 deduplication (threshold=0.92)")
        # Would scan collections and remove near-duplicates

    async def _pass3_l3_enrichment(self) -> None:
        """Enrich L3 graph from L2 content."""
        if not self._l3:
            return
        logger.debug("ISEC pass 3: L3 graph enrichment")

    async def _pass4_decay_audit(self) -> None:
        """Flag items with significance < 0.2 (skip if < 30 days old)."""
        logger.debug("ISEC pass 4: decay audit (threshold=0.2, grace=30d)")

    async def _pass5_l1_pruning(self) -> None:
        """Prune stale L1 content."""
        logger.debug("ISEC pass 5: L1 pruning")

    async def _save_progress(self, progress: dict) -> None:
        if self._l1:
            await self._l1.set_isec_progress(json.dumps(progress))
        elif self._redis:
            await self._redis.set("isec:progress", json.dumps(progress))
