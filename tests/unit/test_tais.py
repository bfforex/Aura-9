"""Tests for TAIS."""

from aura9.core.tais import TAIS, TAISStatus


def test_initial_state() -> None:
    tais = TAIS()
    assert tais.status == TAISStatus.NORMAL
    assert tais.current_temp == 0.0
    assert tais.throttle_events == 0
    assert tais.emergency_halts == 0


def test_telemetry_snapshot() -> None:
    tais = TAIS()
    tel = tais.telemetry()
    assert "tais_current_temp_celsius" in tel
    assert "tais_status" in tel
    assert "tais_active_quantization" in tel
    assert tel["tais_status"] == "NORMAL"
