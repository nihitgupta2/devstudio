"""
DevStudio MCP Custom Exceptions
Copyright (C) 2024 Nihit Gupta

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

For commercial licensing options, contact: nihitgupta.ng@outlook.com

---

Custom exceptions for DevStudio MCP server following mcpcat.io best practices.

Implements three-tier error model: Transport, Protocol, and Application level errors
with proper JSON-RPC error codes and human-friendly messages for AI clients.
"""

import functools
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """
    Base MCP-compliant exception with JSON-RPC error codes.

    Follows mcpcat.io error handling guidelines for structured error responses.
    """

    def __init__(
        self,
        message: str,
        code: int = -32603,  # Internal Error (default)
        data: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data or {}
        self.original_error = original_error
        # User-friendly message for AI clients (sanitized)
        self.user_message = user_message or self._sanitize_message(message)

    def _sanitize_message(self, message: str) -> str:
        """Sanitize internal error messages for external consumption."""
        # Remove sensitive information and provide actionable feedback
        if "api_key" in message.lower() or "token" in message.lower():
            return "Authentication configuration error"
        elif "file not found" in message.lower():
            return "Required file or resource not found"
        elif "permission" in message.lower():
            return "Insufficient permissions to perform operation"
        elif "network" in message.lower() or "connection" in message.lower():
            return "Network connectivity issue"
        else:
            return "Operation failed - please check inputs and try again"

    def to_mcp_response(self) -> Dict[str, Any]:
        """Convert to MCP-compliant error response."""
        return {
            "isError": True,
            "content": [{"type": "text", "text": self.user_message}],
            "error": {
                "code": self.code,
                "message": self.user_message,
                "data": self.data
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        return {
            "error": True,
            "error_code": self.code,
            "message": self.user_message,
            "details": self.data,
            "original_error": str(self.original_error) if self.original_error else None,
        }


class ValidationError(MCPError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(
            message=message,
            code=-32602,  # Invalid Parameters
            data={"field": field, "value": str(value) if value is not None else None},
            user_message=f"Invalid parameter: {message}"
        )


class ConfigurationError(MCPError):
    """Raised when there's a configuration issue."""

    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            code=-32603,  # Internal Error
            data={"config_key": config_key} if config_key else {},
            user_message="Server configuration error - please check settings"
        )


class RecordingError(MCPError):
    """Raised when there's an error during recording operations."""

    def __init__(self, message: str, operation: Optional[str] = None, session_id: Optional[str] = None):
        super().__init__(
            message=message,
            code=-32603,
            data={"operation": operation, "session_id": session_id},
            user_message=f"Recording operation failed: {self._get_recording_message(message)}"
        )

    def _get_recording_message(self, message: str) -> str:
        """Get user-friendly recording error message."""
        if "permission" in message.lower():
            return "Screen recording permission required"
        elif "audio" in message.lower():
            return "Audio device not available"
        elif "display" in message.lower():
            return "Display capture not available"
        else:
            return "Recording hardware or software issue"


class TranscriptionError(MCPError):
    """Raised when there's an error during transcription processing."""

    def __init__(self, message: str, provider: Optional[str] = None, file_path: Optional[str] = None):
        super().__init__(
            message=message,
            code=-32603,
            data={"provider": provider, "file_path": file_path},
            user_message=f"Transcription failed: {self._get_transcription_message(message)}"
        )

    def _get_transcription_message(self, message: str) -> str:
        """Get user-friendly transcription error message."""
        if "api" in message.lower():
            return "AI transcription service temporarily unavailable"
        elif "format" in message.lower():
            return "Unsupported audio format"
        elif "length" in message.lower():
            return "Audio file too long for processing"
        else:
            return "Transcription service error"


class ContentGenerationError(MCPError):
    """Raised when there's an error during content generation."""

    def __init__(self, message: str, content_type: Optional[str] = None, provider: Optional[str] = None):
        super().__init__(
            message=message,
            code=-32603,
            data={"content_type": content_type, "provider": provider},
            user_message=f"Content generation failed: {self._get_generation_message(message)}"
        )

    def _get_generation_message(self, message: str) -> str:
        """Get user-friendly generation error message."""
        if "template" in message.lower():
            return "Content template not found"
        elif "token" in message.lower() or "limit" in message.lower():
            return "Content too large for processing"
        elif "model" in message.lower():
            return "AI model temporarily unavailable"
        else:
            return "Content generation service error"


class FileOperationError(MCPError):
    """Raised when there's an error during file operations."""

    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None):
        super().__init__(
            message=message,
            code=-32603,
            data={"file_path": file_path, "operation": operation},
            user_message="File operation failed - check file path and permissions"
        )


class AuthenticationError(MCPError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed", provider: Optional[str] = None):
        super().__init__(
            message=message,
            code=-32603,
            data={"provider": provider},
            user_message="Authentication failed - please check API credentials"
        )


class RateLimitError(MCPError):
    """Raised when rate limits are exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", provider: Optional[str] = None, retry_after: Optional[int] = None):
        super().__init__(
            message=message,
            code=-32603,
            data={"provider": provider, "retry_after": retry_after},
            user_message=f"Rate limit exceeded - please wait {retry_after or 60} seconds"
        )


class ResourceNotFoundError(MCPError):
    """Raised when a requested resource is not found."""

    def __init__(self, message: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None):
        super().__init__(
            message=message,
            code=-32603,
            data={"resource_type": resource_type, "resource_id": resource_id},
            user_message="Resource not found - check resource path or identifier"
        )


class ServerError(MCPError):
    """Raised for internal server errors."""

    def __init__(self, message: str = "Internal server error", component: Optional[str] = None):
        super().__init__(
            message=message,
            code=-32603,
            data={"component": component},
            user_message="Internal server error - please try again"
        )


def handle_mcp_error(func):
    """
    Decorator for MCP tool functions to handle errors gracefully.

    Implements mcpcat.io error handling patterns with proper logging
    and structured error responses.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # Log the operation attempt
            logger.info(f"Executing MCP tool: {func.__name__} with args={args}, kwargs={kwargs}")
            result = await func(*args, **kwargs)
            return result

        except MCPError as e:
            # Known MCP errors - log and return structured response
            logger.warning(f"MCP error in {func.__name__}: {e.message}")
            return e.to_mcp_response()

        except ValueError as e:
            # Validation errors
            logger.warning(f"Validation error in {func.__name__}: {e}")
            error = ValidationError(str(e))
            return error.to_mcp_response()

        except FileNotFoundError as e:
            # File/resource errors
            logger.error(f"Resource error in {func.__name__}: {e}")
            error = ResourceNotFoundError(str(e))
            return error.to_mcp_response()

        except PermissionError as e:
            # Permission errors
            logger.error(f"Permission error in {func.__name__}: {e}")
            error = MCPError(
                message=str(e),
                user_message="Insufficient permissions to perform operation"
            )
            return error.to_mcp_response()

        except Exception as e:
            # Unexpected errors - log full details, return safe message
            logger.exception(f"Unexpected error in {func.__name__}: {e}")
            error = MCPError(
                message=str(e),
                user_message="Unexpected error occurred - please try again"
            )
            return error.to_mcp_response()

    return wrapper
