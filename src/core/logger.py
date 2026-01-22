"""
Logging Setup
Structured logging for the multi-omics integration pipeline
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "omics_pipeline",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup structured logger with file and console handlers

    Args:
        name: Logger name
        log_file: Path to log file (None for no file logging)
        level: Logging level (default: INFO)
        console: Enable console output (default: True)

    Returns:
        Configured logger

    Usage:
        logger = setup_logger("data_prep", log_file=Path("outputs/logs/prep.log"))
        logger.info("Processing started")
        logger.debug("Detailed debug information")
        logger.warning("Warning message")
        logger.error("Error occurred")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Don't propagate to root logger

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger by name

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerContext:
    """
    Context manager for temporary logger configuration

    Usage:
        with LoggerContext("mylogger", level=logging.DEBUG):
            logger.debug("This will be logged")
        # Logger reverts to previous configuration
    """

    def __init__(self, logger_name: str, level: Optional[int] = None):
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.new_level = level

    def __enter__(self):
        if self.new_level is not None:
            self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls

    Args:
        logger: Logger to use

    Usage:
        @log_function_call(logger)
        def my_function(arg1, arg2):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__}(args={args}, kwargs={kwargs})")
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed")
            return result
        return wrapper
    return decorator


def create_run_id() -> str:
    """
    Create unique run identifier based on timestamp

    Returns:
        Run ID string (format: YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
