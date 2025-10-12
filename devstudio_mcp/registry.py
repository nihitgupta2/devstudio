"""
DevStudio MCP Registry - Centralized registration for tools, resources, and prompts
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

This module follows current MCP best practices by providing a clean separation
of concerns and centralized capability registration.
"""

import logging
from typing import Dict, List, Any
from fastmcp import FastMCP

from .tools import recording
from .resources import media_manager
from .config import Settings


class MCPRegistry:
    """Registry for managing MCP tools, resources, and prompts."""

    def __init__(self, mcp: FastMCP, settings: Settings) -> None:
        """Initialize registry with MCP instance and settings."""
        self.mcp = mcp
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self._registered_tools: Dict[str, Any] = {}
        self._registered_resources: Dict[str, Any] = {}
        self._registered_prompts: Dict[str, Any] = {}

    def register_all(self) -> None:
        """Register all tools, resources, and prompts."""
        self.register_tools()
        self.register_resources()
        self.register_prompts()

    def register_tools(self) -> None:
        """Register Phase 1 recording tools (Phase 2/3 archived for future release)."""
        try:
            # Phase 1: Recording tools (ACTIVE)
            self.logger.info("Registering Phase 1 recording tools...")
            recording_tools = recording.get_tools(self.settings)
            for tool_name, tool_func in recording_tools.items():
                # Use FastMCP tool decorator with explicit name to avoid duplicates
                decorated_tool = self.mcp.tool(name=tool_name)(tool_func)
                self._registered_tools[tool_name] = decorated_tool
                self.logger.info(f"✓ Registered recording tool: {tool_name}")

            self.logger.info(f"✅ Successfully registered {len(self._registered_tools)} Phase 1 tools")

            # Phase 2/3: AI Processing, Content Generation, Monetization (ARCHIVED)
            # These features are preserved in git branch: archive/phase-2-3-ai-features
            # Will be reactivated when backend AI infrastructure is ready

        except ImportError as e:
            self.logger.error(f"Failed to import tool modules: {e}")
            raise RuntimeError(f"Tool module import failed: {e}")
        except Exception as e:
            self.logger.error(f"Error registering tools: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Tool registration failed: {e}")

    def register_resources(self) -> None:
        """Register Phase 1 resources (media management only)."""
        # Phase 1: Media management resources (ACTIVE)
        media_resources = media_manager.get_resources(self.settings)
        for resource_name, resource_func in media_resources.items():
            self.mcp.add_resource(resource_func)
            self._registered_resources[resource_name] = resource_func

        # Phase 2/3: Session data resources (ARCHIVED)
        # Preserved in git branch: archive/phase-2-3-ai-features

    def register_prompts(self) -> None:
        """Register prompts (Phase 1 has no prompts, Phase 2/3 archived)."""
        # Phase 1: No prompts needed for recording-only functionality

        # Phase 2/3: Content generation and analysis prompts (ARCHIVED)
        # Preserved in git branch: archive/phase-2-3-ai-features
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities for MCP negotiation (Phase 1: Recording Only)."""
        return {
            "tools": list(self._registered_tools.keys()),
            "resources": list(self._registered_resources.keys()),
            "prompts": list(self._registered_prompts.keys()),
            "features": {
                "screen_recording": True,
                "audio_recording": True,
                "multi_monitor_support": True,
                "audio_video_muxing": True,
                "screenshot_capture": True
            }
        }

    def get_tool_count(self) -> int:
        """Get number of registered tools."""
        return len(self._registered_tools)

    def get_resource_count(self) -> int:
        """Get number of registered resources."""
        return len(self._registered_resources)

    def get_prompt_count(self) -> int:
        """Get number of registered prompts."""
        return len(self._registered_prompts)
