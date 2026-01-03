"""Tests for memex._logging module."""

import logging
import os
import sys
from unittest.mock import patch

import pytest

from memex._logging import configure_logging, get_logger


class TestConfigureLogging:
    """Tests for configure_logging function."""

    @pytest.fixture(autouse=True)
    def reset_logging(self):
        """Reset memex logger before each test."""
        # Remove all handlers from memex logger
        logger = logging.getLogger("memex")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
        yield

    def test_default_log_level_is_info(self):
        """Default log level is INFO when MEMEX_LOG_LEVEL not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove MEMEX_LOG_LEVEL if present
            os.environ.pop("MEMEX_LOG_LEVEL", None)
            configure_logging()

        logger = logging.getLogger("memex")
        assert logger.level == logging.INFO

    def test_respects_log_level_env_var(self):
        """Log level can be set via MEMEX_LOG_LEVEL environment variable."""
        with patch.dict(os.environ, {"MEMEX_LOG_LEVEL": "DEBUG"}):
            configure_logging()

        logger = logging.getLogger("memex")
        assert logger.level == logging.DEBUG

    def test_warning_level_from_env(self):
        """WARNING level is correctly parsed from environment."""
        with patch.dict(os.environ, {"MEMEX_LOG_LEVEL": "WARNING"}):
            configure_logging()

        logger = logging.getLogger("memex")
        assert logger.level == logging.WARNING

    def test_case_insensitive_level(self):
        """Log level parsing is case-insensitive."""
        with patch.dict(os.environ, {"MEMEX_LOG_LEVEL": "error"}):
            configure_logging()

        logger = logging.getLogger("memex")
        assert logger.level == logging.ERROR

    def test_invalid_level_defaults_to_info(self):
        """Invalid log level falls back to INFO."""
        with patch.dict(os.environ, {"MEMEX_LOG_LEVEL": "INVALID"}):
            configure_logging()

        logger = logging.getLogger("memex")
        assert logger.level == logging.INFO

    def test_creates_handler(self):
        """Creates a StreamHandler pointing to stderr."""
        configure_logging()

        logger = logging.getLogger("memex")
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream == sys.stderr

    def test_second_call_is_noop(self):
        """Calling configure_logging twice doesn't add duplicate handlers."""
        configure_logging()
        configure_logging()

        logger = logging.getLogger("memex")
        assert len(logger.handlers) == 1

    def test_propagate_disabled(self):
        """Logger propagation is disabled to prevent duplicate messages."""
        configure_logging()

        logger = logging.getLogger("memex")
        assert logger.propagate is False

    def test_formatter_format(self):
        """Handler uses expected format pattern."""
        configure_logging()

        logger = logging.getLogger("memex")
        handler = logger.handlers[0]
        # Check format string contains expected parts
        assert handler.formatter._fmt == "[%(levelname)s] %(name)s: %(message)s"


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self):
        """Returns a Logger instance."""
        logger = get_logger("memex.test")
        assert isinstance(logger, logging.Logger)

    def test_returns_correct_name(self):
        """Returns logger with the requested name."""
        logger = get_logger("memex.core")
        assert logger.name == "memex.core"

    def test_child_logger_inherits_from_memex(self):
        """Child loggers inherit from the memex package logger."""
        configure_logging()
        parent = logging.getLogger("memex")
        child = get_logger("memex.child")

        # Child should be under parent hierarchy
        assert child.parent == parent

    def test_can_log_messages(self):
        """Logger can emit log messages."""
        logger = get_logger("memex.test_module")
        # Should not raise
        logger.info("Test message")
        logger.debug("Debug message")
        logger.warning("Warning message")
