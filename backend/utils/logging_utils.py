"""
Logging utilities for the gymnastics photo sorter.
Sets up structured, coloured logging to both console and rotating file handlers.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


# ANSI colour codes for console output
_COLOURS = {
    "DEBUG": "\033[36m",      # Cyan
    "INFO": "\033[32m",       # Green
    "WARNING": "\033[33m",    # Yellow
    "ERROR": "\033[31m",      # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET": "\033[0m",
}


class ColouredFormatter(logging.Formatter):
    """Formatter that adds ANSI colour codes to log level names."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        reset = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname:<8}{reset}"
        return super().format(record)


class QueueHandler(logging.Handler):
    """
    Logging handler that pushes log records into an asyncio-compatible queue
    so the Gradio UI can display live logs.
    """

    def __init__(self, queue: "asyncio.Queue") -> None:  # noqa: F821
        super().__init__()
        self._queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._queue.put_nowait(msg)
        except Exception:
            self.handleError(record)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_to_file: bool = True,
    ui_queue: Optional["asyncio.Queue"] = None,  # noqa: F821
) -> None:
    """
    Configure the root logger with:
    - Coloured console output
    - Rotating file handler (optional)
    - UI queue handler (optional, for live Gradio log feed)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = ColouredFormatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    handlers: list[logging.Handler] = [console_handler]

    # File handler
    if log_to_file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / "image_sorter.log",
            maxBytes=10 * 1024 * 1024,   # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # UI queue handler
    if ui_queue is not None:
        queue_handler = QueueHandler(ui_queue)
        queue_handler.setLevel(numeric_level)
        plain_formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )
        queue_handler.setFormatter(plain_formatter)
        handlers.append(queue_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # Remove any existing handlers to avoid duplicate output
    root_logger.handlers.clear()
    for h in handlers:
        root_logger.addHandler(h)

    # Silence overly verbose third-party loggers
    for noisy in ("PIL", "urllib3", "httpx", "httpcore", "gradio", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper for getting a named logger."""
    return logging.getLogger(name)
