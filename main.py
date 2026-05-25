"""
Main entry point for the Gymnastics Photo Sorter.

Usage:
    python main.py                    # launch Gradio UI with defaults
    python main.py --port 7861        # custom port
    python main.py --share            # create a public Gradio link
    python main.py --device cpu       # force CPU inference
    python main.py --log-level DEBUG  # verbose logging
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when run as a script
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.utils.config import get_config, set_config
from backend.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gymnastics Photo Sorter – AI-powered competition photo organiser."
    )
    parser.add_argument("--port", type=int, default=None, help="Gradio server port")
    parser.add_argument("--host", type=str, default=None, help="Gradio server host")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default=None,
        help="Compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging verbosity",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom config.json file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration (from disk if available, else defaults)
    cfg = get_config() if args.config is None else __import__(
        "backend.utils.config", fromlist=["AppConfig"]
    ).AppConfig.load(args.config)

    # Apply CLI overrides
    if args.port is not None:
        cfg.ui.port = args.port
    if args.host is not None:
        cfg.ui.host = args.host
    if args.share:
        cfg.ui.share = True
    if args.device is not None:
        cfg.device = args.device
    if args.log_level is not None:
        cfg.log_level = args.log_level

    # Initialise logging
    cfg.ensure_dirs()
    setup_logging(
        level=cfg.log_level,
        log_dir=cfg.log_dir,
        log_to_file=cfg.log_to_file,
    )

    set_config(cfg)

    import logging
    logger = logging.getLogger("main")
    logger.info("Gymnastics Photo Sorter v%s starting…", cfg.app_version)
    logger.info("Device: %s", cfg.resolve_device())
    logger.info("UI: http://%s:%d", cfg.ui.host, cfg.ui.port)

    # Launch the Gradio UI
    from backend.ui.app import launch
    launch(cfg)


if __name__ == "__main__":
    main()
