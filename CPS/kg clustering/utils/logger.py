"""Logging utilities for AdaptKG experiments."""

import os
import logging
from typing import Optional


def setup_logger(
    name: str = 'adaptkg',
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger with console and optional file handlers.

    Args:
        name: Logger name.
        log_dir: Directory for log files. If None, only console logging.
        log_level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'experiment.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def log_hyperparams(logger: logging.Logger, config: dict, prefix: str = 'Config'):
    """Log all hyperparameters."""
    logger.info(f"\n{'='*50}")
    logger.info(f"  {prefix}")
    logger.info(f"{'='*50}")
    for key, value in config.items():
        if isinstance(value, (list, tuple)):
            logger.info(f"  {key:25s}: {list(value)}")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                logger.info(f"  {key}.{sub_key:18s}: {sub_value}")
        else:
            logger.info(f"  {key:25s}: {value}")
    logger.info(f"{'='*50}\n")
