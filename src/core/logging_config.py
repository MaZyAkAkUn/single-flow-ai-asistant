import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields that were passed to the logger
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message',
                          'asctime', 'formatted_message'):
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_file: str = "logs/personal_assistant.log",
    log_to_console: bool = True,
) -> None:
    """Set up logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_file: Path to log file
        log_to_console: Whether to log to console
    """
    # Create main logger
    logger = logging.getLogger("personal_assistant")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # JSON formatter
    formatter = JSONFormatter()

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler for main log
    if log_to_file:
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Set up tools logger
    tools_logger = logging.getLogger("personal_assistant.tools")
    tools_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers from tools logger
    for handler in tools_logger.handlers[:]:
        tools_logger.removeHandler(handler)

    # Tools logger inherits from main logger but also gets its own file handler
    if log_to_file:
        tools_log_file = "logs/tools.log"
        os.makedirs(os.path.dirname(tools_log_file), exist_ok=True)
        tools_file_handler = logging.FileHandler(tools_log_file, encoding="utf-8")
        tools_file_handler.setFormatter(formatter)
        tools_logger.addHandler(tools_file_handler)
        # Also add console handler for tools logger
        if log_to_console:
            tools_console_handler = logging.StreamHandler(sys.stdout)
            tools_console_handler.setFormatter(formatter)
            tools_logger.addHandler(tools_console_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: Logger name (will be prefixed with 'personal_assistant.')

    Returns:
        Logger instance
    """
    return logging.getLogger(f"personal_assistant.{name}")


def get_tools_logger() -> logging.Logger:
    """Get the tools logger instance.

    Returns:
        Tools logger instance
    """
    return logging.getLogger("personal_assistant.tools")
