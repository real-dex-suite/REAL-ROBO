"""Utility helpers for configuring REAL-ROBO logging."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from typing import Optional, Union

_LOGGER_NAME = "real_robo"
_DEFAULT_CONSOLE_LEVEL = logging.INFO
_DEBUG_LOG_NAME = "real_robo_debug.log"
_ERROR_LOG_NAME = "real_robo_error.log"

_COLOR_MAP = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}
_RESET_COLOR = "\033[0m"

_LOGGER_INITIALIZED = False


def _determine_log_dir(override: Optional[Union[str, Path]]) -> Path:
    if override is not None:
        return Path(override).expanduser()

    env_override = os.getenv("REAL_ROBO_LOG_DIR")
    if env_override:
        return Path(env_override).expanduser()

    # Default to the package root's logs directory: real_robo/logs
    return Path(__file__).resolve().parent.parent / "logs"


def _stream_supports_color(stream) -> bool:
    if os.getenv("REAL_ROBO_LOG_NO_COLOR"):
        return False
    if sys.platform == "win32":
        return False
    return hasattr(stream, "isatty") and stream.isatty()


class _ConsoleFormatter(logging.Formatter):
    """Formatter that prepends REAL-ROBO prefixes and optional colors."""

    def __init__(self, use_color: bool) -> None:
        super().__init__()
        self.use_color = use_color

    def _colorize(self, text: str, level: int) -> str:
        if not self.use_color:
            return text
        try:
            color = _COLOR_MAP[level]
        except KeyError:
            return text
        return f"{color}{text}{_RESET_COLOR}"

    def format(self, record: logging.LogRecord) -> str:
        prefix = self._colorize(f"[REAL-ROBO {record.levelname}]", record.levelno)
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        return f"{prefix} {message} ({record.filename}:{record.lineno})"


def _create_console_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(_ConsoleFormatter(_stream_supports_color(handler.stream)))
    return handler


def _create_file_handler(path: Path, level: int) -> logging.Handler:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[REAL-ROBO %(levelname)s] %(asctime)s - %(message)s (%(filename)s:%(lineno)d)"
    )
    handler.setFormatter(formatter)
    return handler


def _create_error_handler(path: Path, level: int) -> logging.Handler:
    handler = RotatingFileHandler(path, maxBytes=10_485_760, backupCount=5, encoding="utf-8")
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[REAL-ROBO %(levelname)s] %(asctime)s - %(message)s (%(filename)s:%(lineno)d)"
    )
    handler.setFormatter(formatter)
    return handler


def _configure_logging(
    *,
    console_level: Optional[int] = None,
    log_dir: Optional[Union[str, Path]] = None,
    enable_file_logging: Optional[bool] = None,
) -> None:
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        if console_level is not None:
            _update_console_level(console_level)
        return

    # Decide whether to attach file handlers. Default: disabled.
    if enable_file_logging is None:
        # Enable files only when explicitly requested via env var.
        env_on = os.getenv("REAL_ROBO_LOG_TO_FILE", "0") not in ("", "0", "false", "False")
        enable_file_logging = bool(env_on)

    base_logger = logging.getLogger(_LOGGER_NAME)
    base_logger.setLevel(logging.DEBUG)
    base_logger.propagate = False

    console_handler = _create_console_handler(
        console_level if console_level is not None else _DEFAULT_CONSOLE_LEVEL
    )
    base_logger.addHandler(console_handler)
    if enable_file_logging:
        resolved_log_dir = _determine_log_dir(log_dir)
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        debug_handler = _create_file_handler(
            resolved_log_dir / _DEBUG_LOG_NAME, logging.DEBUG
        )
        error_handler = _create_error_handler(
            resolved_log_dir / _ERROR_LOG_NAME, logging.ERROR
        )
        base_logger.addHandler(debug_handler)
        base_logger.addHandler(error_handler)

    _LOGGER_INITIALIZED = True


def _update_console_level(level: int) -> None:
    base_logger = logging.getLogger(_LOGGER_NAME)
    for handler in base_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(level)


def get_real_robo_logger(
    name: Optional[str] = None,
    *,
    console_level: Optional[int] = None,
    log_dir: Optional[Union[str, Path]] = None,
    enable_file_logging: Optional[bool] = None,
) -> logging.Logger:
    """Return a configured logger for REAL-ROBO components.

    Args:
        name: Logical name for the logger. Child loggers receive the prefix
            ``real_robo`` automatically. Using ``None`` (default) returns the
            project root logger.
        console_level: Optional override for console handler severity.
        log_dir: Optional path or string pointing to the directory where log
            files should be written. If omitted, ``REAL_ROBO_LOG_DIR`` is
            honored, falling back to ``real_robo/logs``.
    """

    _configure_logging(
        console_level=console_level,
        log_dir=log_dir,
        enable_file_logging=enable_file_logging,
    )

    if not name:
        return logging.getLogger(_LOGGER_NAME)

    if name.startswith(_LOGGER_NAME):
        return logging.getLogger(name)

    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


__all__ = ["get_real_robo_logger"]
