"""Logging setup for Aura-9 using loguru.

Configures the five log channels defined in the spec:
  - aura9.log   — main inference and orchestration
  - tais.log    — thermal events
  - watchdog.log — watchdog alerts and heartbeats
  - audit.log   — immutable audit trail (no rotation)
  - isec.log    — consolidation pass reports
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from aura9.core.config import get


def setup_logging() -> None:
    """Configure loguru sinks from config/settings.yaml."""
    # Remove default stderr sink; we add our own below.
    logger.remove()

    # Console sink (always present during dev)
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

    rotation = get("logging.rotation", "10 MB")
    retention = get("logging.retention", "7 days")
    compression = get("logging.compression", "gz")

    log_specs: list[tuple[str, str, bool]] = [
        (get("logging.main_log", "./logs/aura9.log"), "DEBUG", True),
        (get("logging.tais_log", "./logs/tais.log"), "DEBUG", True),
        (get("logging.watchdog_log", "./logs/watchdog.log"), "DEBUG", True),
        (get("logging.isec_log", "./logs/isec.log"), "DEBUG", True),
    ]

    for log_path, level, rotate in log_specs:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            level=level,
            rotation=rotation if rotate else None,
            retention=retention if rotate else None,
            compression=compression if rotate else None,
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

    # Audit log — immutable, append-only, NO rotation
    audit_path = get("logging.audit_log", "./logs/audit.log")
    Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        audit_path,
        level="DEBUG",
        rotation=None,
        retention=None,
        compression=None,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
