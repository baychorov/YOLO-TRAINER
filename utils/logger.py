"""This module provides a custom Logger class using Loguru and a notification decorator."""

import sys
import os
from dataclasses import dataclass
from typing import Optional, Any
from loguru import logger as builtin_logger


def filtered_logging_decorator(func):
    """
    A decorator that intercepts log messages.
    If the log level meets or exceeds the threshold and the notifier is active, 
    it forwards the message to the notification service (e.g., Email).
    """
    def inner_function(*args, **kwargs):
        # Execute the original logging function first
        func(*args, **kwargs)

        try:
            # args[0] is 'self' (the Logger instance), args[1] is the 'message'
            logger_instance = args[0]
            message = args[1]
            
            # Convert string log levels (e.g., 'INFO') to their numerical values for comparison
            notifier_log_level = logger_instance.logger.level(logger_instance.notifier_log_level).no
            current_log_level = logger_instance.logger.level(func.__name__.upper()).no

            # Trigger the notification if criteria are met
            if logger_instance.is_notifier_active and current_log_level >= notifier_log_level:
                if logger_instance.notifier:
                    logger_instance.notifier.send_message(str(message))

        except Exception as e:
            logger_instance.logger.error(f"Failed to send log message to notifier: {e}")

    return inner_function


@dataclass
class Logger:
    """
    Custom logger class utilizing Loguru for terminal and file logging.
    Incorporates an automated notification trigger for critical logs.
    """
    filepath: str
    rotation: str
    retention: int
    file_log_level: str
    notifier_log_level: str
    is_notifier_active: bool
    notifier: Optional[Any] = None

    def __post_init__(self) -> None:
        """
        Executes immediately after dataclass initialization to set up Loguru handlers.
        """
        self.logger = builtin_logger
        
        # Remove default Loguru handlers to prevent duplicate prints
        self.logger.remove()
        
        # Ensure the target directory for the log file exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        # Define formats for terminal (colored) and file (plain text)
        term_format = "<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> | <level>{level: <8}</level> | <cyan>[{module}:{function}:{line}]</cyan> | <level>{message}</level>"
        file_format = "[{time:YYYY-MM-DD HH:mm:ss}] | {level: <8} | [{module}:{function}:{line}] | {message}"
        
        # 1. File Sink
        self.logger.add(
            sink=self.filepath,
            rotation=self.rotation,
            retention=self.retention,
            level=self.file_log_level,
            encoding="utf-8",
            format=file_format,
        )
        
        # 2. Console Sink (Outputs to terminal)
        self.logger.add(
            sink=sys.stdout,
            level="DEBUG",  # Terminal will show all logs
            format=term_format,
            colorize=True
        )
        
        # Set depth to 2 so Loguru points to the file that actually called the logger
        self.nested_function_depth = 2

        # Disable notifier safely if the flag is False
        if not self.is_notifier_active:
            self.notifier = None

    @filtered_logging_decorator
    def debug(self, message: str) -> None:
        self.logger.opt(depth=self.nested_function_depth).debug(message)

    @filtered_logging_decorator
    def info(self, message: str) -> None:
        self.logger.opt(depth=self.nested_function_depth).info(message)

    @filtered_logging_decorator
    def warning(self, message: str) -> None:
        self.logger.opt(depth=self.nested_function_depth).warning(message)

    @filtered_logging_decorator
    def error(self, message: str) -> None:
        self.logger.opt(depth=self.nested_function_depth).error(message)

    @filtered_logging_decorator
    def critical(self, message: str) -> None:
        self.logger.opt(depth=self.nested_function_depth).critical(message)