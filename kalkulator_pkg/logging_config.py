"""Structured logging configuration for Kalkulator."""

import logging
import sys
from datetime import datetime
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured log entries with timestamp, module, level, and message."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        return f"{timestamp} [{record.levelname}] {record.name}: {record.getMessage()}"


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """Set up structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs (if None, logs to stderr)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("kalkulator")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(StructuredFormatter())
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "kalkulator") -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Logger name (typically module name)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"kalkulator.{name}")


def safe_log(
    module_name: str, level: str, message: str, *args, exc_info: bool = False, **kwargs
) -> None:
    """Safely log a message, handling ImportError if logging is unavailable.

    This utility function eliminates the need for try/except ImportError blocks
    around logging calls throughout the codebase.

    Args:
        module_name: Module name for the logger
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message: Log message format string
        *args: Arguments for message formatting
        exc_info: If True, include exception traceback
        **kwargs: Additional keyword arguments for logging
    """
    try:
        logger = get_logger(module_name)
        log_func = getattr(logger, level.lower(), logger.info)
        if exc_info:
            log_func(message, *args, exc_info=True, **kwargs)
        else:
            log_func(message, *args, **kwargs)
    except ImportError:
        # Logging module not available - silently skip
        pass
    except Exception:
        # Unexpected error in logging - don't fail the application
        pass
