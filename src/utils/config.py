"""Configuration loading utilities.

Loads the central YAML configuration and exposes it as a typed, dot-accessible
object so the rest of the codebase never hardcodes paths or hyperparameters.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ConfigBox(dict):
    """A dict that also supports dot-notation attribute access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, ConfigBox):
            value = ConfigBox(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> ConfigBox:
    """Load and cache the project configuration.

    Args:
        config_path: Optional override path to a YAML config file. If not
            provided, the ``HEALTHCARE_MLOPS_CONFIG`` environment variable is
            checked, falling back to the default ``configs/config.yaml``.

    Returns:
        A ConfigBox instance exposing config values via attribute access.
    """
    path = Path(config_path or os.environ.get("HEALTHCARE_MLOPS_CONFIG", "") or DEFAULT_CONFIG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return ConfigBox(raw)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root into an absolute Path."""
    return PROJECT_ROOT / relative_path
