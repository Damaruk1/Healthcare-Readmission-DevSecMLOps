"""Centralized structured logging using Loguru.

All modules import ``get_logger`` from here instead of using ``print`` or
instantiating their own loggers, ensuring consistent structured JSON logs
across the entire application.
"""

from __future__ import annotations

import sys

from loguru import logger as _logger

from src.utils.config import load_config, resolve_path

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = load_config()
    log_dir = resolve_path(cfg.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    _logger.remove()

    # Human-readable console sink
    _logger.add(
        sys.stdout,
        level=cfg.logging.level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>"
        ),
    )

    # Structured JSON sink (rotating) for machine parsing / log aggregation
    if cfg.logging.json_format:
        _logger.add(
            str(log_dir / "app.jsonl"),
            level=cfg.logging.level,
            serialize=True,
            rotation="10 MB",
            retention="14 days",
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )

    _CONFIGURED = True


def get_logger(name: str):
    """Return a Loguru logger bound with a module name for structured logs."""
    _configure()
    return _logger.bind(module=name)
