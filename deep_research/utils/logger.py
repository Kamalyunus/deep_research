"""
Centralized logging configuration for the Deep Research Agent.

This module provides a consistent logging setup across the application,
with options for different log levels, formatting, and output destinations.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any, Union

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default log directory
DEFAULT_LOG_DIR = "logs"

# Log levels mapping for easier configuration
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def setup_logger(
    name: Optional[str] = None,
    level: Union[str, int] = "info",
    format_string: Optional[str] = None,
    log_to_file: bool = False,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 3
) -> logging.Logger:
    """
    Configure and return a logger with the specified settings.
    
    Args:
        name: Logger name (uses root logger if None)
        level: Log level (debug, info, warning, error, critical or logging constants)
        format_string: Custom log format string
        log_to_file: Whether to log to a file
        log_dir: Directory for log files (defaults to 'logs')
        log_file: Log file name (defaults to 'deep_research_{date}.log')
        max_bytes: Maximum bytes per log file for rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Get logger
    logger = logging.getLogger(name)
    
    # Set log level
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.lower(), logging.INFO)
    logger.setLevel(level)
    
    # Use default format if none provided
    if format_string is None:
        format_string = DEFAULT_LOG_FORMAT
    
    formatter = logging.Formatter(format_string)
    
    # Clear existing handlers (to avoid duplicates when reconfiguring)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        if log_dir is None:
            log_dir = DEFAULT_LOG_DIR
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate default log filename if none provided
        if log_file is None:
            date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_file = f"deep_research_{date_str}.log"
        
        # Create full path
        log_path = os.path.join(log_dir, log_file)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name, or create it with default settings if it doesn't exist.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If this logger hasn't been configured yet, set basic configuration
    if not logger.handlers and not logging.getLogger().handlers:
        return setup_logger(name)
    
    return logger

def configure_root_logger(
    level: Union[str, int] = "info",
    log_to_file: bool = True,
    log_dir: Optional[str] = None
) -> None:
    """
    Configure the root logger for the application.
    
    Args:
        level: Log level
        log_to_file: Whether to log to a file
        log_dir: Directory for log files
    """
    setup_logger(None, level, log_to_file=log_to_file, log_dir=log_dir)

def set_log_level(level: Union[str, int], logger_name: Optional[str] = None) -> None:
    """
    Change the log level of an existing logger.
    
    Args:
        level: New log level
        logger_name: Logger to modify (root logger if None)
    """
    logger = logging.getLogger(logger_name)
    
    # Convert string level to constant if needed
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.lower(), logging.INFO)
    
    logger.setLevel(level)
    
    # Also update all handlers to this level
    for handler in logger.handlers:
        handler.setLevel(level)