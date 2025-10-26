#!/usr/bin/env python3

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
import glob

def setup_logging(log_level: str = None, max_log_files: int = 10):
    """Setup comprehensive logging configuration.
    
    Args:
        log_level: Log level from environment (DEBUG, INFO, WARNING, ERROR)
        max_log_files: Maximum number of log files to keep
    """
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Clean up old log files
    _cleanup_old_logs(logs_dir, max_log_files)
    
    # Get log level from environment or default to INFO
    level = getattr(logging, (log_level or os.getenv('LOG_LEVEL', 'INFO')).upper(), logging.INFO)
    
    # Create a timestamp for log files  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler with detailed formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(max(level, logging.INFO))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / f"attierly_{timestamp}.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler for errors only
    error_handler = logging.handlers.RotatingFileHandler(
        logs_dir / f"errors_{timestamp}.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n'
        'Exception: %(exc_info)s\n'
        'Stack trace: %(stack_info)s\n'
        '-' * 80 + '\n'
    )
    error_handler.setFormatter(error_formatter)
    root_logger.addHandler(error_handler)
    
    # Set specific logger levels based on main level
    app_level = level if level <= logging.DEBUG else logging.INFO
    logging.getLogger('services.ai_orchestrator').setLevel(app_level)
    logging.getLogger('services.user_service').setLevel(app_level)
    logging.getLogger('services.ecommerce_service').setLevel(app_level)
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    
    # Log startup message
    logging.info(f"Logging configured - Level: {logging.getLevelName(level)} - Log files: {logs_dir}/attierly_{timestamp}.log, {logs_dir}/errors_{timestamp}.log")


def _cleanup_old_logs(logs_dir: Path, max_files: int = 10):
    """Remove old log files, keeping only the most recent ones."""
    try:
        # Get all log files sorted by modification time (newest first)
        log_files = sorted(
            logs_dir.glob("*.log"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        # Remove old files beyond the limit
        for old_file in log_files[max_files:]:
            old_file.unlink(missing_ok=True)
            
    except Exception as e:
        print(f"Warning: Could not clean up old log files: {e}")


def get_logger(name):
    """Get a logger with the given name."""
    return logging.getLogger(name) 