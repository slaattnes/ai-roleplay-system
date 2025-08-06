"""Logging utilities for the AI agent role-playing system."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

from src.utils.config import load_config

# Load main configuration
main_config = load_config("main")
log_level_str = main_config["system"]["log_level"]
log_dir = Path(main_config["system"]["log_directory"])

# Ensure log directory exists
log_dir.mkdir(parents=True, exist_ok=True)

# Map string log level to logging constant
LOG_LEVELS: Dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

log_level = LOG_LEVELS.get(log_level_str, logging.INFO)

def setup_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """Set up a logger with the specified name."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        current_date = datetime.now().strftime("%Y-%m-%d")
        file_handler = logging.FileHandler(
            log_dir / f"{current_date}_{name}.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create a default system logger
system_logger = setup_logger("system")