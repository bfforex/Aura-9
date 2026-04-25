"""Configuration loader for Aura-9."""

from __future__ import annotations

import os

import yaml
from loguru import logger

from src.config.schema import Aura9Config

_DEFAULT_CONFIG_PATH = "config/aura9.config.yaml"


def load_config(path: str | None = None) -> Aura9Config:
    """Load and validate configuration from YAML file.

    The config path is resolved in order:
    1. ``path`` argument (if provided)
    2. ``AURA9_CONFIG`` environment variable
    3. Default ``config/aura9.config.yaml``

    If the file is absent a warning is logged and all-defaults config is returned.
    """
    resolved = path or os.environ.get("AURA9_CONFIG", _DEFAULT_CONFIG_PATH)
    try:
        with open(resolved) as f:
            raw = yaml.safe_load(f)
        return Aura9Config.model_validate(raw or {})
    except FileNotFoundError:
        logger.warning(
            f"Config file not found at '{resolved}' — using all defaults"
        )
        return Aura9Config()
    except Exception as exc:
        logger.error(f"Failed to load config from '{resolved}': {exc} — using all defaults")
        return Aura9Config()
