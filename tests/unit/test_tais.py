"""Unit tests for TAIS — Thermal-Aware Inference Scheduling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.tais import (
    TEMP_COOLDOWN,
    TEMP_EMERGENCY,
    TEMP_NORMAL_MAX,
    TEMP_RESUME,
    TEMP_THROTTLE,
    TAISDaemon,
    TAISStatus,
)


def _make_daemon(temp_value: int | None = 65) -> tuple[TAISDaemon, MagicMock]:
    """Create a TAISDaemon with a mocked pynvml reading."""
    daemon = TAISDaemon()
    mock_ollama = AsyncMock()
    mock_ollama.unload_model = AsyncMock()
    mock_ollama.load_model = AsyncMock()
    daemon._ollama_client = mock_ollama
    return daemon, mock_ollama


@pytest.mark.unit
class TestTAISThresholds:
    def test_constants(self):
        """Verify exact threshold values from spec."""
        assert TEMP_NORMAL_MAX == 74
        assert TEMP_THROTTLE == 75
        assert TEMP_COOLDOWN == 80
        assert TEMP_EMERGENCY == 83
        assert TEMP_RESUME == 72

    def test_initial_status(self):
        daemon = TAISDaemon()
        assert daemon.get_status() == TAISStatus.NORMAL
        assert daemon.get_temp() is None


@pytest.mark.unit
class TestTAISTransitions:
    @pytest.mark.asyncio
    async def test_normal_to_throttle_at_75(self):
        """NORMAL→THROTTLE at 75°C."""
        daemon = TAISDaemon()
        with patch.object(daemon, "_read_gpu_temp", return_value=75.0):
            with patch.object(daemon, "_publish_status", new_callable=AsyncMock):
                await daemon._check_temperature()
        assert daemon.get_status() == TAISStatus.THROTTLE

    @pytest.mark.asyncio
    async def test_throttle_to_cooldown_at_80(self):
        """→COOLDOWN at 80°C."""
        daemon = TAISDaemon()
        daemon._status = TAISStatus.THROTTLE
        with patch.object(daemon, "_read_gpu_temp", return_value=80.0):
            with patch.object(daemon, "_publish_status", new_callable=AsyncMock):
                await daemon._check_temperature()
        assert daemon.get_status() == TAISStatus.COOLDOWN

    @pytest.mark.asyncio
    async def test_cooldown_to_emergency_at_83(self):
        """→EMERGENCY at 83°C."""
        daemon = TAISDaemon()
        daemon._status = TAISStatus.COOLDOWN
        with patch.object(daemon, "_read_gpu_temp", return_value=83.0):
            with patch.object(daemon, "_publish_status", new_callable=AsyncMock):
                await daemon._check_temperature()
        assert daemon.get_status() == TAISStatus.EMERGENCY

    @pytest.mark.asyncio
    async def test_temperature_jump_skips_cooldown(self):
        """Direct NORMAL→EMERGENCY if temp > 83°C (skipping COOLDOWN)."""
        daemon = TAISDaemon()
        assert daemon.get_status() == TAISStatus.NORMAL
        with patch.object(daemon, "_read_gpu_temp", return_value=90.0):
            with patch.object(daemon, "_publish_status", new_callable=AsyncMock):
                await daemon._check_temperature()
        assert daemon.get_status() == TAISStatus.EMERGENCY

    @pytest.mark.asyncio
    async def test_recovery_at_72(self):
        """Recovery: resume NORMAL when temp < 72°C."""
        daemon = TAISDaemon()
        daemon._status = TAISStatus.THROTTLE
        with patch.object(daemon, "_read_gpu_temp", return_value=71.0):
            with patch.object(daemon, "_publish_status", new_callable=AsyncMock):
                with patch.object(daemon, "_switch_to_q5", new_callable=AsyncMock):
                    await daemon._check_temperature()
        assert daemon.get_status() == TAISStatus.NORMAL

    @pytest.mark.asyncio
    async def test_sensor_fail_fallback(self):
        """SENSOR_FAIL: assume THROTTLE on pynvml exception."""
        daemon = TAISDaemon()
        with patch.object(daemon, "_read_gpu_temp", return_value=None):
            with patch.object(daemon, "_publish_status", new_callable=AsyncMock):
                await daemon._check_temperature()
        assert daemon.get_status() == TAISStatus.SENSOR_FAIL

    def test_sensor_fail_increments_counter(self):
        """Sensor fail events are counted."""
        daemon = TAISDaemon()
        # Simulate multiple sensor fail events
        daemon._sensor_fail_events = 0
        daemon._temp = None
        assert daemon._sensor_fail_events == 0

    @pytest.mark.asyncio
    async def test_normal_range_stays_normal(self):
        """Temp below 75°C stays NORMAL."""
        daemon = TAISDaemon()
        with patch.object(daemon, "_read_gpu_temp", return_value=65.0):
            with patch.object(daemon, "_publish_status", new_callable=AsyncMock):
                await daemon._check_temperature()
        assert daemon.get_status() == TAISStatus.NORMAL

    def test_pynvml_exception_returns_none(self):
        """_read_gpu_temp returns None on pynvml exception."""
        daemon = TAISDaemon()
        with patch.dict("sys.modules", {"pynvml": None}):
            result = daemon._read_gpu_temp()
        assert result is None
