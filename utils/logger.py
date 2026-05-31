"""Logging configuration for local debugging."""

from __future__ import annotations

import logging
import os
import sys

DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
NOISY_LOGGERS = ("httpx", "httpcore", "openai")


def _coerce_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level

    level_name = (level or os.getenv("LOG_LEVEL") or DEFAULT_LOG_LEVEL).upper()
    numeric_level = logging.getLevelName(level_name)
    if isinstance(numeric_level, int):
        return numeric_level

    return logging.INFO


def configure_logging(level: str | int | None = None) -> None:
    """Configure process-wide logging for agent and tool debugging."""

    logging.basicConfig(
        level=_coerce_level(level),
        format=LOG_FORMAT,
        stream=sys.stderr,
        force=True,
    )

    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
