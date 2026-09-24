"""Shared logging configuration: console + a rotating file per pipeline step."""

import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logging(step_name: str) -> logging.Logger:
    """Configure root logging to write to logs/<step_name>.log and stdout."""
    log_dir = PROJECT_ROOT / os.getenv("LOG_DIR", "logs/")
    log_dir.mkdir(parents=True, exist_ok=True)

    level = os.getenv("LOG_LEVEL", "INFO")
    log_path = log_dir / f"{step_name}.log"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    return logging.getLogger(step_name)
