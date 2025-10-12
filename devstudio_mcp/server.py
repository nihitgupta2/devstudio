"""
DevStudio MCP Server - Production-Grade Screen Recording Server
Copyright (C) 2024 Nihit Gupta

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

For commercial licensing options, contact: nihitgupta.ng@outlook.com

---

Production-grade MCP server following 2024 best practices with registry pattern,
proper capability negotiation, and error handling.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .config import Settings
from .registry import MCPRegistry
from .utils.logger import setup_logger
from .utils.exceptions import MCPError, ServerError


class ServerInfo(BaseModel):
    """Server information model for capability negotiation."""
    name: str = Field(default="DevStudio MCP", description="Server name")
    version: str = Field(default="1.0.0", description="Server version")
    description: str = Field(
        default="AI-powered content creation MCP server",
        description="Server description"
    )
    protocol_version: str = Field(default="1.0.0", description="MCP protocol version")


class DevStudioMCP:
    """
    Production-grade DevStudio MCP server.

    Follows 2024 MCP best practices:
    - Registry pattern for tool/resource/prompt management
    - Proper capability negotiation
    - Self-contained tool calls
    - Human-friendly error messages
    - Workspace isolation
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the DevStudio MCP server."""
        self.settings = settings or Settings()
        self.logger = setup_logger(self.settings.logging.level)

        # Initialize FastMCP with server info
        self.mcp = FastMCP(
            name="DevStudio MCP",
            version="1.0.0"
        )

        # Initialize registry for clean separation of concerns
        self.registry = MCPRegistry(self.mcp, self.settings)

        # Server state
        self._is_running = False
        self._capabilities: Dict[str, Any] = {}

        self.logger.info("DevStudio MCP server initialized")

    async def initialize(self) -> None:
        """Initialize server capabilities and register components."""
        try:
            # Register all tools, resources, and prompts via registry
            self.registry.register_all()

            # Set server capabilities for MCP negotiation
            self._capabilities = self.registry.get_capabilities()

            self.logger.info(
                f"Server initialized with "
                f"{self.registry.get_tool_count()} tools, "
                f"{self.registry.get_resource_count()} resources, "
                f"{self.registry.get_prompt_count()} prompts"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize server: {e}")
            raise ServerError(f"Server initialization failed: {e}")

    async def run(self) -> None:
        """
        Run the MCP server with proper error handling.

        Follows best practice of creating connections per tool call
        rather than maintaining persistent connections.
        """
        try:
            await self.initialize()

            self.logger.info("Starting DevStudio MCP server on stdio transport...")
            self._is_running = True

            # Run with stdio transport (standard for MCP)
            # Simplified call without nested asyncio
            self.mcp.run()

        except KeyboardInterrupt:
            self.logger.info("Server stopped by user")
            self._is_running = False
        except Exception as e:
            self.logger.error(f"Server runtime error: {e}")
            self._is_running = False
            raise ServerError(f"Server runtime failed: {e}")

    def get_server_info(self) -> ServerInfo:
        """Get server information for client negotiation."""
        return ServerInfo()

    def get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities for MCP protocol negotiation."""
        return self._capabilities.copy()

    def is_running(self) -> bool:
        """Check if server is currently running."""
        return self._is_running

    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        self.logger.info("Shutting down DevStudio MCP server...")
        self._is_running = False


def main() -> None:
    """
    Main entry point for the DevStudio MCP server.

    Handles startup, error recovery, and graceful shutdown
    following production best practices.
    """
    try:
        # Load settings with environment variable support
        settings = Settings()

        # Create server
        server = DevStudioMCP(settings)

        # Initialize server (async part)
        import asyncio
        asyncio.run(server.initialize())

        # Run server (sync part)
        server.mcp.run()

    except KeyboardInterrupt:
        logging.info("Server stopped by user interrupt")
    except Exception as e:
        logging.error(f"Server failed: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Application terminated by user")
    except Exception as e:
        logging.error(f"Application failed: {e}")
        exit(1)