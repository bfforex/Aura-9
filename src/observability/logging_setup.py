"""Logging setup for Aura-9 using loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(config=None) -> None:
    """Configure loguru with rotation, retention, and compression."""
    log_dir = "./logs"
    rotation = "10 MB"
    retention = "7 days"
    compression = "gz"

    if config and hasattr(config, "observability"):
        obs = config.observability
        log_dir = getattr(obs, "log_dir", log_dir)
        rotation = f"{getattr(obs, 'log_rotation_mb', 10)} MB"
        retention = f"{getattr(obs, 'log_retention_days', 7)} days"
        compression = getattr(obs, "log_compression", "gz")

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )

    # Main log file
    logger.add(
        f"{log_dir}/aura9.log",
        rotation=rotation,
        retention=retention,
        compression=compression,
        enqueue=True,
        level="DEBUG",
    )

    # TAIS-specific log
    logger.add(
        f"{log_dir}/tais.log",
        rotation=rotation,
        retention=retention,
        compression=compression,
        enqueue=True,
        level="DEBUG",
        filter=lambda record: "TAIS" in record["message"],
    )

    # Watchdog-specific log
    logger.add(
        f"{log_dir}/watchdog.log",
        rotation=rotation,
        retention=retention,
        compression=compression,
        enqueue=True,
        level="DEBUG",
        filter=lambda record: (
            "Watchdog" in record["message"]
            or "watchdog" in record["message"].lower()
        ),
    )

    # ISEC-specific log
    logger.add(
        f"{log_dir}/isec.log",
        rotation=rotation,
        retention=retention,
        compression=compression,
        enqueue=True,
        level="DEBUG",
        filter=lambda record: "ISEC" in record["message"],
    )

    logger.info("Logging initialized")
