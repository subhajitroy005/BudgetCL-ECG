"""Structured logging shared by the package and the experiment runners."""

from __future__ import annotations

import logging
import sys

__all__ = ["get_logger"]

_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger, attaching a stdout handler only once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
