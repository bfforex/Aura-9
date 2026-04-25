"""Diagnostics — system status and telemetry."""

from __future__ import annotations

from datetime import UTC, datetime

from loguru import logger


async def get_system_status(
    tais_daemon=None, asd_daemon=None, l1=None
) -> dict:
    """Return a high-level system status summary."""
    status: dict = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tais": await get_tais_status(tais_daemon),
        "memory": await get_memory_stats(l1),
    }

    if asd_daemon:
        state = await asd_daemon.get_state()
        if state:
            asd_data = state.get("asd_update", state)
            status["asd"] = {
                "status": asd_data.get("status", "UNKNOWN"),
                "task_id": asd_data.get("task_id"),
                "confidence": asd_data.get("confidence"),
            }

    return status


async def get_tais_status(tais_daemon=None) -> dict:
    """Return TAIS status."""
    if not tais_daemon:
        return {"status": "UNKNOWN"}
    return {
        "status": tais_daemon.get_status().value,
        "temp_celsius": tais_daemon.get_temp(),
    }


async def get_memory_stats(l1=None) -> dict:
    """Return memory layer statistics."""
    stats: dict = {}
    if l1:
        try:
            isec = await l1.get_isec_progress()
            stats["isec_progress"] = isec
        except Exception as exc:
            logger.debug(f"Diagnostics: failed to fetch ISEC progress: {exc}")
    return stats
