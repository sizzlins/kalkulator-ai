"""Structured logging configuration for Kalkulator."""

import atexit
import logging
import logging.handlers
import queue
import sys
from datetime import datetime
from typing import Optional

# Global queue listener for cleanup
_queue_listener: Optional[logging.handlers.QueueListener] = None


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured log entries with timestamp, module, level, and message."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        return f"{timestamp} [{record.levelname}] {record.name}: {record.getMessage()}"


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None, async_mode: bool = True
) -> logging.Logger:
    """Set up structured logging for the application.
    
    Supports async mode (default) to prevent I/O blocking in performance-critical code.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs (if None, logs to stderr)
        async_mode: If True, use QueueHandler for non-blocking logging

    Returns:
        Configured logger instance
    """
    global _queue_listener
    
    logger = logging.getLogger("kalkulator")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    # Create actual handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(StructuredFormatter())
    handlers.append(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        handlers.append(file_handler)

    if async_mode:
        # Use QueueHandler for non-blocking logging
        log_queue = queue.Queue(-1)  # No limit
        queue_handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(queue_handler)
        
        # Start QueueListener in background thread
        _queue_listener = logging.handlers.QueueListener(
            log_queue, *handlers, respect_handler_level=True
        )
        _queue_listener.start()
        
        # Ensure cleanup on exit
        atexit.register(_shutdown_logging)
    else:
        # Direct handlers (synchronous)
        for handler in handlers:
            logger.addHandler(handler)

    return logger


def _shutdown_logging():
    """Shutdown the async logging listener."""
    global _queue_listener
    if _queue_listener is not None:
        _queue_listener.stop()
        _queue_listener = None


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
