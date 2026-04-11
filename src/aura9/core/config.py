"""Configuration loader for Aura-9."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"

_config_cache: dict[str, Any] | None = None


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load and cache the YAML configuration.

    Parameters
    ----------
    path:
        Explicit path to settings.yaml.  Falls back to the repo default.
        Passing *path* forces a reload even if the cache is populated.
    """
    global _config_cache  # noqa: PLW0603
    if _config_cache is not None and path is None:
        return _config_cache

    config_path = path or _DEFAULT_CONFIG_PATH
    with open(config_path) as fh:
        _config_cache = yaml.safe_load(fh)
    return _config_cache


def get(key: str, default: Any = None) -> Any:
    """Dot-notation lookup into the config tree.

    Example::

        get("tais.thresholds.normal_max_celsius")  # -> 75
    """
    cfg = load_config()
    parts = key.split(".")
    node: Any = cfg
    for part in parts:
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return default
        if node is None:
            return default
    return node
