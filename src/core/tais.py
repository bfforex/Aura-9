"""TAIS — Thermal-Aware Inference Scheduling daemon."""

from __future__ import annotations

import asyncio
from enum import StrEnum

from loguru import logger

# Temperature thresholds (°C)
TEMP_NORMAL_MAX = 74
TEMP_THROTTLE = 75
TEMP_COOLDOWN = 80
TEMP_EMERGENCY = 83
TEMP_RESUME = 72

# Polling intervals (seconds)
POLL_INFERENCE = 5
POLL_IDLE = 30


class TAISStatus(StrEnum):
    NORMAL = "NORMAL"
    THROTTLE = "THROTTLE"
    COOLDOWN = "COOLDOWN"
    EMERGENCY = "EMERGENCY"
    SENSOR_FAIL = "SENSOR_FAIL"


class TAISHaltException(Exception):  # noqa: N818
    """Raised when TAIS transitions to EMERGENCY and inference must stop."""


class TAISDaemon:
    """Thermal-Aware Inference Scheduling daemon.

    Monitors GPU temperature and enforces compute throttling/halting.
    """

    def __init__(self, ollama_client=None, redis_client=None, config=None) -> None:
        self._ollama_client = ollama_client
        self._redis_client = redis_client
        self._config = config
        self._status: TAISStatus = TAISStatus.NORMAL
        self._temp: float | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._is_inferring = False

        # pynvml state — initialised once in start()
        self._nvml_available = False
        self._nvml_handle = None

        # Prometheus metrics (lazy import to avoid circular deps)
        self._metrics_initialized = False

        # Throttle/emergency counters
        self._throttle_events = 0
        self._emergency_halts = 0
        self._sensor_fail_events = 0

    def _init_metrics(self) -> None:
        if self._metrics_initialized:
            return
        try:
            from src.observability.metrics import (  # noqa: PLC0415
                TAIS_CURRENT_TEMP,
                TAIS_EMERGENCY_HALTS,
                TAIS_SENSOR_FAIL_EVENTS,
                TAIS_THROTTLE_EVENTS,
            )

            self._metric_temp = TAIS_CURRENT_TEMP
            self._metric_throttle = TAIS_THROTTLE_EVENTS
            self._metric_emergency = TAIS_EMERGENCY_HALTS
            self._metric_sensor_fail = TAIS_SENSOR_FAIL_EVENTS
            self._metrics_initialized = True
        except Exception:
            self._metrics_initialized = False

    async def start(self) -> None:
        """Start the TAIS polling loop."""
        self._running = True
        # Initialise pynvml once
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_available = True
            logger.info("TAIS: pynvml initialised successfully")
        except Exception as exc:
            self._nvml_available = False
            logger.warning(f"TAIS: pynvml init failed: {exc} — GPU monitoring disabled")
        self._task = asyncio.create_task(self._polling_loop())
        logger.info("TAIS daemon started")

    async def stop(self) -> None:
        """Stop the TAIS polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._nvml_available:
            try:
                import pynvml  # noqa: PLC0415

                pynvml.nvmlShutdown()
            except Exception as exc:
                logger.debug(f"TAIS: nvmlShutdown failed: {exc}")
        logger.info("TAIS daemon stopped")

    def get_status(self) -> TAISStatus:
        return self._status

    def get_temp(self) -> float | None:
        return self._temp

    def set_inferring(self, *, active: bool) -> None:
        """Signal whether inference is currently active (affects poll interval)."""
        self._is_inferring = active

    async def _polling_loop(self) -> None:
        while self._running:
            interval = POLL_INFERENCE if self._is_inferring else POLL_IDLE
            await self._check_temperature()
            await asyncio.sleep(interval)

    async def _check_temperature(self) -> None:
        temp = self._read_gpu_temp()

        if temp is None:
            if self._status != TAISStatus.SENSOR_FAIL:
                self._sensor_fail_events += 1
                logger.warning("TAIS: sensor failure — assuming THROTTLE")
                self._update_metrics_sensor_fail()
            self._temp = None
            await self._transition(TAISStatus.SENSOR_FAIL)
            return

        self._temp = temp
        self._update_metrics_temp(temp)

        prev_status = self._status

        if temp >= TEMP_EMERGENCY:
            if prev_status not in (TAISStatus.EMERGENCY,):
                if self._is_inferring:
                    logger.critical(f"TAIS EMERGENCY: temp={temp}°C — killing active inference")
                await self._transition(TAISStatus.EMERGENCY)
        elif temp >= TEMP_COOLDOWN:
            await self._transition(TAISStatus.COOLDOWN)
        elif temp >= TEMP_THROTTLE:
            await self._transition(TAISStatus.THROTTLE)
        elif temp < TEMP_RESUME:
            if prev_status in (TAISStatus.THROTTLE, TAISStatus.COOLDOWN, TAISStatus.EMERGENCY):
                logger.info(f"TAIS: temp={temp}°C — resuming NORMAL")
            await self._transition(TAISStatus.NORMAL)
        # between TEMP_RESUME and TEMP_THROTTLE: maintain current status

    def _read_gpu_temp(self) -> float | None:
        """Read GPU temperature via pynvml. Returns None on failure."""
        if not self._nvml_available or self._nvml_handle is None:
            return None
        try:
            import pynvml  # noqa: PLC0415

            return float(
                pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                )
            )
        except Exception as exc:
            logger.debug(f"TAIS: pynvml read failed: {exc}")
            return None

    async def _transition(self, new_status: TAISStatus) -> None:
        """Handle status transition, including model swaps and pub/sub."""
        prev = self._status

        if new_status == prev and new_status != TAISStatus.EMERGENCY:
            return

        self._status = new_status
        logger.info(f"TAIS transition: {prev} → {new_status}")

        if new_status == TAISStatus.THROTTLE and prev == TAISStatus.NORMAL:
            self._throttle_events += 1
            self._update_metrics_throttle()
            await self._switch_to_q4()

        elif new_status == TAISStatus.NORMAL and prev in (
            TAISStatus.THROTTLE,
            TAISStatus.COOLDOWN,
            TAISStatus.SENSOR_FAIL,
        ):
            await self._switch_to_q5()

        elif new_status == TAISStatus.EMERGENCY:
            self._emergency_halts += 1
            self._update_metrics_emergency()
            await self._handle_emergency()

        await self._publish_status()

    async def _switch_to_q4(self) -> None:
        if self._ollama_client:
            try:
                import time  # noqa: PLC0415

                t0 = time.monotonic()
                model_q5 = (
                    self._config.model.primary
                    if self._config else "qwen3.5:9b-instruct-q5_k_m"
                )
                model_q4 = (
                    self._config.model.primary_fallback
                    if self._config else "qwen3.5:9b-instruct-q4_k_m"
                )
                await self._ollama_client.unload_model(model_q5)
                await self._ollama_client.load_model(model_q4)
                elapsed = time.monotonic() - t0
                logger.info(f"TAIS: switched to Q4_K_M in {elapsed:.2f}s")
            except Exception as exc:
                logger.error(f"TAIS: model switch to Q4 failed: {exc}")

    async def _switch_to_q5(self) -> None:
        if self._ollama_client:
            try:
                model_q5 = (
                    self._config.model.primary
                    if self._config else "qwen3.5:9b-instruct-q5_k_m"
                )
                model_q4 = (
                    self._config.model.primary_fallback
                    if self._config else "qwen3.5:9b-instruct-q4_k_m"
                )
                await self._ollama_client.unload_model(model_q4)
                await self._ollama_client.load_model(model_q5)
                logger.info("TAIS: switched back to Q5_K_M")
            except Exception as exc:
                logger.error(f"TAIS: model switch to Q5 failed: {exc}")

    async def _handle_emergency(self) -> None:
        """Kill active inference and wait for user confirmation."""
        logger.critical("TAIS EMERGENCY HALT — autonomous execution suspended")
        # The TAISHaltException is raised by OllamaClient when it checks status

    async def _publish_status(self) -> None:
        if self._redis_client:
            try:
                from src.ipc.channels import TAIS_STATUS  # noqa: PLC0415
                from src.ipc.publisher import publish  # noqa: PLC0415

                await publish(
                    TAIS_STATUS,
                    {"status": self._status.value, "temp": self._temp},
                    self._redis_client,
                )
            except Exception as exc:
                logger.debug(f"TAIS: publish failed: {exc}")

    def _update_metrics_temp(self, temp: float) -> None:
        self._init_metrics()
        if self._metrics_initialized:
            try:
                self._metric_temp.set(temp)
            except Exception as exc:
                logger.debug(f"TAIS: failed to update temperature metric: {exc}")

    def _update_metrics_throttle(self) -> None:
        self._init_metrics()
        if self._metrics_initialized:
            try:
                self._metric_throttle.inc()
            except Exception as exc:
                logger.debug(f"TAIS: failed to update throttle metric: {exc}")

    def _update_metrics_emergency(self) -> None:
        self._init_metrics()
        if self._metrics_initialized:
            try:
                self._metric_emergency.inc()
            except Exception as exc:
                logger.debug(f"TAIS: failed to update emergency metric: {exc}")

    def _update_metrics_sensor_fail(self) -> None:
        self._init_metrics()
        if self._metrics_initialized:
            try:
                self._metric_sensor_fail.inc()
            except Exception as exc:
                logger.debug(f"TAIS: failed to update sensor-fail metric: {exc}")
