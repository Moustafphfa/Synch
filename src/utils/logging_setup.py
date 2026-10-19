"""
utils/logging_setup.py — Shared logging factory for the Synch pipeline.

Every module calls get_logger(__name__) instead of calling
logging.basicConfig() individually. basicConfig() is a no-op after the
first call, so only one module's format would win. This ensures a
consistent format across the entire pipeline.
"""

import logging
import sys

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once. Safe to call from any module."""
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Ensures 3rd party imports don't hijack the root logger
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring the root handler is configured."""
    setup_logging()
    return logging.getLogger(name)