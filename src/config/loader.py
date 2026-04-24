"""Configuration loader for Aura-9."""

from __future__ import annotations

import yaml

from src.config.schema import Aura9Config


def load_config(path: str = "config/aura9.config.yaml") -> Aura9Config:
    """Load and validate configuration from YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Aura9Config.model_validate(raw or {})
