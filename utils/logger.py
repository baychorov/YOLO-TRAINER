"""This module provides a custom Logger class using Loguru and a notification decorator."""

import sys
import os
from dataclasses import dataclass
from typing import Optional, Any
from loguru import logger as builtin_logger


def filtered_logging_decorator(func):
    def inner_function(*args, **kwargs):
        func(*args, **kwargs)

        try:
            logger_instance = args[0]
            message = args[1]

            notifier_log_level = logger_instance.logger.level(logger_instance.notifier_log_level).no
            current_log_level = logger_instance.logger.level(func.__name__.upper()).no

            if logger_instance.is_notifier_active and current_log_level >= notifier_log_level:
                if logger_instance.notifier:
                    logger_instance.notifier.send_message(str(message))

        except Exception as e:
            logger_instance.logger.error(f"Failed to send log message to notifier: {e}")

    return inner_function


@dataclass
class Logger:
    filepath: str
    rotation: str
    retention: int
    file_log_level: str
    notifier_log_level: str
    is_notifier_active: bool
    notifier: Optional[Any] = None

    def __post_init__(self) -> None:
        self.logger = builtin_logger
        self.logger.remove()

        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)

        msg_format = "[{time:YYYY-MM-DD HH:mm:ss}] | {level: <8} | [{module}:{function}:{line}] | {message}"

        self.logger.add(
            sink=self.filepath,
            rotation=self.rotation,
            retention=self.retention,
            level=self.file_log_level,
            encoding="utf-8",
            format=msg_format,
        )

        self.logger.add(
            sink=sys.stdout,
            level="DEBUG",
            format="<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> | <level>{level: <8}</level> | <cyan>[{module}:{function}:{line}]</cyan> | <level>{message}</level>",
            colorize=True,
        )

        self.nested_function_depth = 2

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