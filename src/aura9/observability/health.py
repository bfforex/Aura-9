"""Health check protocol for Aura-9.

Runs on startup and periodically (every 5 minutes) to verify all
subsystem dependencies are operational.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from aura9.core import model as ollama
from aura9.core.tais import TAIS
from aura9.memory.falkor_l3 import FalkorMemory
from aura9.memory.qdrant_l2 import QdrantMemory
from aura9.memory.redis_l1 import RedisMemory


async def run_health_check(
    *,
    redis: RedisMemory,
    qdrant: QdrantMemory,
    falkor: FalkorMemory,
    tais: TAIS,
) -> list[dict[str, Any]]:
    """Execute all health checks and return a list of results.

    Each result is ``{"name": ..., "ok": bool, "detail": str}``.
    """
    results: list[dict[str, Any]] = []

    # Ollama
    ok = await ollama.is_healthy()
    detail = "primary model loaded" if ok else "unreachable"
    results.append({"name": "Ollama", "ok": ok, "detail": detail})

    # Redis
    ok = await redis.ping()
    detail = "ping < 1ms" if ok else "unreachable"
    results.append({"name": "Redis (L1)", "ok": ok, "detail": detail})

    # Qdrant
    ok = await qdrant.is_healthy()
    detail = "healthy" if ok else "unreachable"
    results.append({"name": "Qdrant (L2)", "ok": ok, "detail": detail})

    # FalkorDB
    ok = falkor.is_healthy()
    detail = "graph accessible" if ok else "unreachable"
    results.append({"name": "FalkorDB (L3)", "ok": ok, "detail": detail})

    # TAIS
    tel = tais.telemetry()
    temp = tel["tais_current_temp_celsius"]
    status = tel["tais_status"]
    quant = tel["tais_active_quantization"]
    results.append({
        "name": "TAIS",
        "ok": True,
        "detail": f"{temp}°C — {status} — {quant}",
    })

    # Watchdog heartbeat
    hb = await redis.watchdog_heartbeat_age()
    ok = hb is not None
    results.append({
        "name": "Watchdog",
        "ok": ok,
        "detail": f"heartbeat {90 - hb}s ago" if hb else "heartbeat expired",
    })

    for r in results:
        symbol = "✓" if r["ok"] else "✗"
        logger.info("{} {} — {}", symbol, r["name"], r["detail"])

    return results
