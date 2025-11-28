"""
Centralized logging configuration for the entire project.

This module provides a consistent logging setup across all modules.
Import and use the get_logger function in any module that needs logging.

Example:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting preprocessing")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None,
) -> None:
    """
    Configure global logging settings.

    Call this once at application startup to configure logging.

    Args:
        log_file: Path to log file. If None, logs only to console.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string. If None, uses default format.
    """
    # Default format: timestamp - module - level - message
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.

    Args:
        name: Module name (use __name__ to get current module)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    return logging.getLogger(name)


# Initialize default logging configuration
# This can be overridden by calling setup_logging() in main script
setup_logging(
    log_file="logs/training.log",
    level="INFO",
)