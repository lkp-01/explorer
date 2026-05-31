"""日志配置：本地调试与事后复盘（第 8 步）。

第 8 步的体会是：等主流程跑通后，你会发现不知道 agent 到底在干什么、调了哪个工具、
出了什么错——这时才自然想去补日志。loop.py 里已经埋了 model_turn / tool_call /
tool_result 三类日志，这里负责把它们打到控制台，并可选地同时写入文件，方便回看。
"""

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


def configure_logging(
    level: str | int | None = None,
    log_file: str | None = None,
) -> None:
    """配置进程级日志。给了 log_file 就同时把日志写入该文件。"""

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        try:
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        except OSError:
            # 文件不可写时退回"仅控制台"，不让日志配置本身拖垮程序
            pass

    logging.basicConfig(
        level=_coerce_level(level),
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )

    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
