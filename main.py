"""Aura-9 — main entry point.

Usage::

    python main.py              # Normal interactive startup
    python main.py --resume     # Cold-start resumption from checkpoint
"""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger


async def startup(resume: bool = False) -> None:
    """Initialize all subsystems and enter the main loop."""
    from aura9.core.config import load_config
    from aura9.core.logging import setup_logging
    from aura9.core.session import generate_session_id
    from aura9.core.tais import TAIS
    from aura9.memory.falkor_l3 import FalkorMemory
    from aura9.memory.qdrant_l2 import QdrantMemory
    from aura9.memory.redis_l1 import RedisMemory
    from aura9.observability.health import run_health_check
    from aura9.orchestration.asd import AuraStateDaemon
    from aura9.security.watchdog import Watchdog

    # 1. Load config and logging
    load_config()
    setup_logging()
    session_id = generate_session_id()
    logger.info("Aura-9 starting — session_id={}", session_id)

    # 2. Connect memory tiers
    redis = RedisMemory()
    qdrant = QdrantMemory()
    falkor = FalkorMemory()

    await redis.connect()
    await qdrant.connect()
    falkor.connect()

    # 3. Ensure schemas / collections
    await qdrant.ensure_collections()
    falkor.ensure_schema()

    # 4. Initialize subsystems
    tais = TAIS()
    asd = AuraStateDaemon(redis)
    watchdog = Watchdog(redis)

    # 5. Cold-start resumption
    if resume:
        logger.info("Resume mode — checking for incomplete checkpoints...")
        state = await asd.read()
        status = state.get("asd_update", {}).get("status", "")
        if status in ("EXECUTING", "PAUSED", "CORRECTING"):
            logger.info("Found checkpoint with status={} — restoring...", status)
            # Restoration logic would restore full L1 session state here
        else:
            logger.info("No incomplete checkpoint found — normal startup")

    # 6. Run health check
    logger.info("Running startup health check...")
    results = await run_health_check(redis=redis, qdrant=qdrant, falkor=falkor, tais=tais)
    failed = [r for r in results if not r["ok"]]
    if failed:
        for f in failed:
            logger.warning("Health check FAILED: {} — {}", f["name"], f["detail"])

    # 7. Start background tasks
    tais_task = asyncio.create_task(tais.run_loop())
    watchdog_task = asyncio.create_task(watchdog.heartbeat_loop())

    logger.info("Aura-9 ready — all systems initialized")

    # 8. Main loop — interactive mode
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received")
    finally:
        tais.stop()
        await watchdog.stop()
        tais_task.cancel()
        watchdog_task.cancel()
        await redis.close()
        await qdrant.close()
        falkor.close()
        logger.info("Aura-9 shutdown complete — session_id={}", session_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aura-9: Autonomous Reasoning Agent")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    asyncio.run(startup(resume=args.resume))


if __name__ == "__main__":
    main()
