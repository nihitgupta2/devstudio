"""
DevStudio MCP Structured Logging
Copyright (C) 2024 Nihit Gupta
Licensed under AGPL-3.0-or-later
For commercial licensing: nihitgupta.ng@outlook.com
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

from ..config import config


def setup_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Set up structured logging with rich console output and file logging.

    Args:
        name: Logger name, defaults to module name

    Returns:
        Configured structured logger
    """
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.logging.level),
    )

    # Set up file handler if enabled
    handlers = []

    if config.logging.file_enabled:
        # Ensure log directory exists
        log_path = Path(config.logging.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=config.logging.file_path,
            maxBytes=config.logging.max_file_size,
            backupCount=config.logging.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, config.logging.level))
        handlers.append(file_handler)

    # Rich console handler for development
    if config.server.debug:
        console = Console(stderr=True)
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
        )
        rich_handler.setLevel(logging.DEBUG)
        handlers.append(rich_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    for handler in handlers:
        root_logger.addHandler(handler)

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="ISO"),
    ]

    if config.logging.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=config.server.debug),
        ])

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger(name)


def log_function_call(
    logger: structlog.stdlib.BoundLogger,
    function_name: str,
    args: Optional[Dict[str, Any]] = None,
    result: Optional[Any] = None,
    error: Optional[Exception] = None,
    duration_ms: Optional[float] = None,
) -> None:
    """
    Log function call with structured data.

    Args:
        logger: Structured logger instance
        function_name: Name of the function being called
        args: Function arguments (sanitized)
        result: Function result (sanitized)
        error: Exception if function failed
        duration_ms: Execution duration in milliseconds
    """
    log_data = {
        "function": function_name,
        "duration_ms": duration_ms,
    }

    if args:
        log_data["args"] = _sanitize_log_data(args)

    if error:
        log_data["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        logger.error("Function call failed", **log_data)
    else:
        if result is not None:
            log_data["result"] = _sanitize_log_data(result)
        logger.info("Function call completed", **log_data)


def log_api_request(
    logger: structlog.stdlib.BoundLogger,
    method: str,
    url: str,
    status_code: Optional[int] = None,
    response_time_ms: Optional[float] = None,
    error: Optional[Exception] = None,
) -> None:
    """
    Log API request with structured data.

    Args:
        logger: Structured logger instance
        method: HTTP method
        url: Request URL (sanitized)
        status_code: HTTP status code
        response_time_ms: Response time in milliseconds
        error: Exception if request failed
    """
    log_data = {
        "method": method,
        "url": _sanitize_url(url),
        "status_code": status_code,
        "response_time_ms": response_time_ms,
    }

    if error:
        log_data["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        logger.error("API request failed", **log_data)
    else:
        logger.info("API request completed", **log_data)


def log_tool_execution(
    logger: structlog.stdlib.BoundLogger,
    tool_name: str,
    session_id: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    result: Optional[Any] = None,
    error: Optional[Exception] = None,
    duration_ms: Optional[float] = None,
) -> None:
    """
    Log MCP tool execution with structured data.

    Args:
        logger: Structured logger instance
        tool_name: Name of the MCP tool
        session_id: Session identifier
        parameters: Tool parameters (sanitized)
        result: Tool execution result (sanitized)
        error: Exception if tool execution failed
        duration_ms: Execution duration in milliseconds
    """
    log_data = {
        "tool": tool_name,
        "session_id": session_id,
        "duration_ms": duration_ms,
    }

    if parameters:
        log_data["parameters"] = _sanitize_log_data(parameters)

    if error:
        log_data["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        logger.error("Tool execution failed", **log_data)
    else:
        if result is not None:
            log_data["result"] = _sanitize_log_data(result)
        logger.info("Tool execution completed", **log_data)


def _sanitize_log_data(data: Any) -> Any:
    """
    Sanitize data for logging by removing sensitive information.

    Args:
        data: Data to sanitize

    Returns:
        Sanitized data
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if _is_sensitive_key(key):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = _sanitize_log_data(value)
        return sanitized
    elif isinstance(data, (list, tuple)):
        return [_sanitize_log_data(item) for item in data]
    elif isinstance(data, str) and len(data) > 1000:
        # Truncate very long strings
        return data[:997] + "..."
    else:
        return data


def _sanitize_url(url: str) -> str:
    """
    Sanitize URL by removing sensitive information.

    Args:
        url: URL to sanitize

    Returns:
        Sanitized URL
    """
    # Remove query parameters that might contain sensitive data
    if "?" in url:
        base_url = url.split("?")[0]
        return f"{base_url}?[QUERY_PARAMS_REDACTED]"
    return url


def _is_sensitive_key(key: str) -> bool:
    """
    Check if a key contains sensitive information.

    Args:
        key: Key to check

    Returns:
        True if key is sensitive
    """
    sensitive_patterns = [
        "password", "passwd", "pwd",
        "secret", "key", "token",
        "auth", "credential", "api_key",
        "private", "confidential",
    ]

    key_lower = key.lower()
    return any(pattern in key_lower for pattern in sensitive_patterns)


# Create default logger
logger = setup_logger(__name__)