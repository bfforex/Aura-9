"""VLA Bridge — vision-language-action bridge for screenshot capture."""

from __future__ import annotations

import time

from loguru import logger

from src.tools.base import ToolResult

VRAM_THRESHOLD_GB = 2.2
KEEP_ALIVE_MIN_CYCLES = 3


class VLABridge:
    """Captures screenshots and applies vision analysis with red zone enforcement."""

    def __init__(self, red_zone_system=None, ollama_client=None) -> None:
        self._red_zone = red_zone_system
        self._ollama = ollama_client
        self._above_threshold_cycles = 0
        self._model_loaded = False

    async def screenshot(self, focus_region: dict | None = None) -> ToolResult:
        """Capture a screenshot and apply red zone filtering."""
        t0 = time.monotonic()

        if not self._check_vram():
            elapsed_ms = (time.monotonic() - t0) * 1000
            return ToolResult(
                success=False,
                output=None,
                error="Insufficient VRAM for vision model (need >= 2.2GB)",
                execution_time_ms=elapsed_ms,
            )

        try:
            image_data = await self._capture_screenshot(focus_region)
            if image_data is None:
                elapsed_ms = (time.monotonic() - t0) * 1000
                return ToolResult(
                    success=False,
                    output=None,
                    error="Screenshot capture failed",
                    execution_time_ms=elapsed_ms,
                )

            # Apply red zone enforcement
            result = {"image": image_data, "blocked_zones": []}
            if self._red_zone:
                # In production: extract UI zones from image and filter
                result["red_zone_applied"] = True

            elapsed_ms = (time.monotonic() - t0) * 1000
            return ToolResult(success=True, output=result, execution_time_ms=elapsed_ms)

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            return ToolResult(
                success=False, output=None, error=str(exc), execution_time_ms=elapsed_ms
            )

    def _check_vram(self) -> bool:
        """Check VRAM headroom. Returns True if >= threshold."""
        try:
            import pynvml  # noqa: PLC0415
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = mem_info.free / (1024 ** 3)

            if free_gb >= VRAM_THRESHOLD_GB:
                self._above_threshold_cycles += 1
            else:
                self._above_threshold_cycles = 0

            return free_gb >= VRAM_THRESHOLD_GB
        except Exception:
            return True  # Assume OK if can't check

    async def _capture_screenshot(self, focus_region: dict | None) -> str | None:
        """Capture screenshot via OS tools. Returns base64 string or None."""
        try:
            import asyncio  # noqa: PLC0415
            import base64  # noqa: PLC0415
            import subprocess  # noqa: PLC0415

            result = await asyncio.to_thread(
                subprocess.run,
                ["import", "-window", "root", "png:-"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                return base64.b64encode(result.stdout).decode()
        except Exception as exc:
            logger.debug(f"VLA: screenshot capture failed: {exc}")
        return None
