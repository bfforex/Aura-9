"""TAIS — Thermal-Aware Inference Scheduling.

Monitors GPU junction temperature via pynvml and dynamically adjusts
the inference workload to protect local hardware during sustained
autonomous operation.
"""

from __future__ import annotations

import asyncio
import enum
from typing import Any

from loguru import logger

from aura9.core.config import get

# pynvml is optional — graceful degradation on systems without an NVIDIA GPU.
try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


class TAISStatus(enum.StrEnum):
    NORMAL = "NORMAL"
    THROTTLE = "THROTTLE"
    COOLDOWN = "COOLDOWN"
    EMERGENCY = "EMERGENCY"


class TAIS:
    """Thermal-Aware Inference Scheduler.

    Runs as a background asyncio task, polling GPU temperature and
    exposing the current status for the inference engine to query.
    """

    def __init__(self) -> None:
        self.status: TAISStatus = TAISStatus.NORMAL
        self.current_temp: float = 0.0
        self.active_quantization: str = get(
            "model.primary.default_quantization", "Q5_K_M"
        )
        self.throttle_events: int = 0
        self.emergency_halts: int = 0
        self._running = False
        self._handle: pynvml.c_nvmlDevice_t | None = None  # type: ignore[name-defined]

    # -- thresholds from config ------------------------------------------------

    @property
    def _normal_max(self) -> float:
        return get("tais.thresholds.normal_max_celsius", 75)

    @property
    def _throttle_max(self) -> float:
        return get("tais.thresholds.throttle_max_celsius", 80)

    @property
    def _cooldown_max(self) -> float:
        return get("tais.thresholds.cooldown_max_celsius", 83)

    @property
    def _recovery(self) -> float:
        return get("tais.thresholds.recovery_celsius", 72)

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Initialise pynvml and mark the scheduler as running."""
        if not _PYNVML_AVAILABLE:
            logger.warning("pynvml not available — TAIS running in stub mode (no GPU monitoring)")
            self._running = True
            return

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._running = True
        logger.info("TAIS started — GPU monitoring active")

    def stop(self) -> None:
        self._running = False
        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                logger.debug("pynvml shutdown warning (non-critical)")
        logger.info("TAIS stopped")

    async def run_loop(self) -> None:
        """Main polling loop — call as an asyncio task."""
        self.start()
        try:
            while self._running:
                self._poll()
                interval = (
                    get("tais.polling_interval_active_seconds", 5)
                    if self.status == TAISStatus.NORMAL
                    else get("tais.polling_interval_idle_seconds", 30)
                )
                await asyncio.sleep(interval)
        finally:
            self.stop()

    # -- internal --------------------------------------------------------------

    def _poll(self) -> None:
        if not _PYNVML_AVAILABLE or self._handle is None:
            return

        try:
            self.current_temp = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            logger.exception("TAIS: failed to read GPU temperature")
            return

        previous = self.status
        if self.current_temp > self._cooldown_max:
            self.status = TAISStatus.EMERGENCY
            self.emergency_halts += 1
            logger.critical("TAIS EMERGENCY — {}°C — halting inference", self.current_temp)
        elif self.current_temp > self._throttle_max:
            self.status = TAISStatus.COOLDOWN
            logger.warning("TAIS COOLDOWN — {}°C — pausing inference queue", self.current_temp)
        elif self.current_temp > self._normal_max:
            if self.status != TAISStatus.THROTTLE:
                self.throttle_events += 1
            self.status = TAISStatus.THROTTLE
            self.active_quantization = get(
                "model.primary.throttle_quantization", "Q4_K_M"
            )
            logger.info(
                "TAIS THROTTLE — {}°C — shifted to {}",
                self.current_temp, self.active_quantization,
            )
        elif self.current_temp <= self._recovery:
            if previous in (TAISStatus.COOLDOWN, TAISStatus.THROTTLE):
                self.active_quantization = get(
                    "model.primary.default_quantization", "Q5_K_M"
                )
                logger.info(
                    "TAIS recovered — {}°C — restored {}",
                    self.current_temp, self.active_quantization,
                )
            self.status = TAISStatus.NORMAL

    def telemetry(self) -> dict[str, Any]:
        """Return a snapshot of TAIS metrics."""
        return {
            "tais_current_temp_celsius": self.current_temp,
            "tais_status": self.status.value,
            "tais_throttle_events_total": self.throttle_events,
            "tais_emergency_halts_total": self.emergency_halts,
            "tais_active_quantization": self.active_quantization,
        }
