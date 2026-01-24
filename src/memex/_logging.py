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


def configure_logging(quiet: bool = False) -> None:
    """Configure logging for the memex package.

    Call this once at application startup (e.g., in cli.py or server.py).
    Subsequent calls are no-ops unless reconfigure() is called.

    Args:
        quiet: If True, suppress warnings and show only errors.
               Can also be set via MEMEX_QUIET=1 environment variable.
    """
    # Get the package root logger
    root_logger = logging.getLogger("memex")

    # Skip if already configured (has handlers)
    if root_logger.handlers:
        return

    # Check for quiet mode from environment or argument
    quiet = quiet or os.environ.get("MEMEX_QUIET", "").lower() in ("1", "true", "yes")

    # Determine log level from environment (or override with ERROR if quiet)
    if quiet:
        level = logging.ERROR
    else:
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


def set_quiet_mode(quiet: bool = True) -> None:
    """Set quiet mode after logging has been configured.

    Use this to enable/disable quiet mode dynamically (e.g., from --quiet flag).

    Args:
        quiet: If True, suppress warnings and show only errors.
    """
    root_logger = logging.getLogger("memex")
    level = logging.ERROR if quiet else logging.INFO
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


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
