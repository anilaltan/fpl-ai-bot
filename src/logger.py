"""
Centralized Logging Module for FPL SaaS Application

This module provides centralized logging configuration and a decorator for
function execution tracking.
"""

import logging
import logging.handlers
import time
import traceback
import functools
from typing import Any, Callable
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: str = "app.log") -> logging.Logger:
    """
    Configure centralized logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file

    Returns:
        Root logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / log_file

    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatters
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return root_logger


def log_execution(func: Callable) -> Callable:
    """
    Decorator to log function execution details.

    Logs function name, arguments, execution time, and results/errors.

    Args:
        func: Function to be decorated

    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)

        # Log function call with arguments
        func_name = func.__name__
        args_repr = _format_args(args, kwargs)

        logger.info(f"EXECUTING: {func_name}({args_repr})")

        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            logger.info(f"SUCCESS: {func_name} completed in {execution_time:.2f}ms")
            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_type = type(e).__name__

            logger.error(
                f"ERROR: {func_name} failed after {execution_time:.2f}ms - "
                f"{error_type}: {str(e)}"
            )

            # Log full traceback
            logger.error(f"TRACEBACK: {traceback.format_exc()}")

            # Re-raise the exception
            raise

    return wrapper


def _format_args(args: tuple, kwargs: dict) -> str:
    """
    Format function arguments for logging.

    Args:
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Formatted string representation of arguments
    """
    formatted_args = []

    # Add positional args (skip 'self' for methods)
    for i, arg in enumerate(args):
        if i == 0 and hasattr(arg, '__class__'):
            # Likely 'self' - show class name instead
            formatted_args.append(f"<{arg.__class__.__name__}>")
        else:
            formatted_args.append(repr(arg))

    # Add keyword args
    for key, value in kwargs.items():
        formatted_args.append(f"{key}={repr(value)}")

    return ", ".join(formatted_args)


# Initialize default logger
logger = logging.getLogger(__name__)
