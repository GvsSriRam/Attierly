#!/usr/bin/env python3

import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging():
    """Setup comprehensive logging configuration for debugging."""
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create a timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler with detailed formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        f"{logs_dir}/attierly_{timestamp}.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler for errors only
    error_handler = logging.handlers.RotatingFileHandler(
        f"{logs_dir}/errors_{timestamp}.log",
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
    
    # Set specific logger levels
    logging.getLogger('services.ai_orchestrator').setLevel(logging.DEBUG)
    logging.getLogger('services.user_service').setLevel(logging.DEBUG)
    logging.getLogger('services.ecommerce_service').setLevel(logging.DEBUG)
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    
    # Log startup message
    logging.info(f"Logging configured - Log files: {logs_dir}/attierly_{timestamp}.log, {logs_dir}/errors_{timestamp}.log")

def get_logger(name):
    """Get a logger with the given name."""
    return logging.getLogger(name) 