"""Tests for configuration loader."""

from pathlib import Path

from aura9.core.config import get, load_config

# Resolve to the repo root: tests/unit/test_config.py → repo/config/settings.yaml
_SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"


def test_load_config_returns_dict() -> None:
    cfg = load_config(_SETTINGS_PATH)
    assert isinstance(cfg, dict)
    assert "identity" in cfg
    assert cfg["identity"]["name"] == "Aura-9"


def test_get_dot_notation() -> None:
    load_config(_SETTINGS_PATH)
    assert get("model.primary.context_window") == 32768
    assert get("tais.thresholds.normal_max_celsius") == 75


def test_get_returns_default_for_missing_key() -> None:
    load_config(_SETTINGS_PATH)
    assert get("nonexistent.key", "fallback") == "fallback"
