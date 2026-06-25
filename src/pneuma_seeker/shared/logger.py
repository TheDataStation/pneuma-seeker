# src/pneuma_seeker/shared/logger.py
import logging
import os
import sys

from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
    name: str = "pneuma_seeker_logger",
    log_path: Optional[str] = None,
    level: int = logging.INFO,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Sets up a centralized logger. If `log_path` is specified, logs will be written
    to a rotating file. Otherwise, logs will be printed to stdout.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:  # Prevent duplicate handlers on re-import
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler (stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_path:
            os.makedirs(log_path, exist_ok=True)
            file_handler = RotatingFileHandler(
                os.path.join(log_path, "pneuma_seeker.log"),
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def formatted_log(logger: logging.Logger, component_name: str, text: str):
    """
    Logs a message with a specific format, including the component name.
    """
    logger.info(f"[{component_name}] {text}")
