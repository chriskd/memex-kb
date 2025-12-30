"""Logging configuration for memex.

This module provides consistent logging across the codebase.

Usage in other modules:
    import logging
    log = logging.getLogger(__name__)

    log.debug("Detailed info for debugging")
    log.info("General operational info")
    log.warning("Unexpected but handled situation")
    log.error("Error that prevented operation")
    log.exception("Error with full traceback")

The log level can be configured via the MEMEX_LOG_LEVEL environment variable:
    - DEBUG: Detailed debugging information
    - INFO: General operational messages (default)
    - WARNING: Unexpected situations that were handled
    - ERROR: Errors that prevented an operation
"""

import logging
import os
import sys


def configure_logging() -> None:
    """Configure logging for the memex package.

    Call this once at application startup (e.g., in cli.py or server.py).
    Subsequent calls are no-ops.
    """
    # Get the package root logger
    root_logger = logging.getLogger("memex")

    # Skip if already configured (has handlers)
    if root_logger.handlers:
        return

    # Determine log level from environment
    level_name = os.environ.get("MEMEX_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # Create console handler with formatting
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    # Use a clean format: [level] logger: message
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Configure the package logger
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate messages)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    This is a convenience wrapper that ensures the logger is in the
    memex namespace.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
