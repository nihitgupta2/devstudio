"""
MCP Registry - Centralized registration for tools, resources, and prompts.

This module follows current MCP best practices by providing a clean separation
of concerns and centralized capability registration.
"""

import logging
from typing import Dict, List, Any
from fastmcp import FastMCP

from .tools import recording, processing, generation
from .resources import media_manager, session_data
from .prompts import content_prompts, analysis_prompts
from .monetization import get_monetization_tools
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
        """Register all available tools with proper error handling and validation."""
        try:
            # Import and register recording tools using proper FastMCP decorators
            self.logger.info("Registering recording tools...")
            recording_tools = recording.get_tools(self.settings)
            for tool_name, tool_func in recording_tools.items():
                # Use FastMCP tool decorator with explicit name to avoid duplicates
                decorated_tool = self.mcp.tool(name=tool_name)(tool_func)
                self._registered_tools[tool_name] = decorated_tool
                self.logger.info(f"✓ Registered recording tool: {tool_name}")

            # Import and register processing tools
            self.logger.info("Registering processing tools...")
            processing_tools = processing.get_tools(self.settings)
            for tool_name, tool_func in processing_tools.items():
                decorated_tool = self.mcp.tool(name=tool_name)(tool_func)
                self._registered_tools[tool_name] = decorated_tool
                self.logger.info(f"✓ Registered processing tool: {tool_name}")

            # Import and register generation tools
            self.logger.info("Registering generation tools...")
            generation_tools = generation.get_tools(self.settings)
            for tool_name, tool_func in generation_tools.items():
                decorated_tool = self.mcp.tool(name=tool_name)(tool_func)
                self._registered_tools[tool_name] = decorated_tool
                self.logger.info(f"✓ Registered generation tool: {tool_name}")

            # Import and register monetization tools
            self.logger.info("Registering monetization tools...")
            monetization_tools = get_monetization_tools(self.settings)
            for tool_name, tool_func in monetization_tools.items():
                decorated_tool = self.mcp.tool(name=tool_name)(tool_func)
                self._registered_tools[tool_name] = decorated_tool
                self.logger.info(f"✓ Registered monetization tool: {tool_name}")

            self.logger.info(f"✅ Successfully registered {len(self._registered_tools)} tools total")

        except ImportError as e:
            self.logger.error(f"Failed to import tool modules: {e}")
            raise RuntimeError(f"Tool module import failed: {e}")
        except Exception as e:
            self.logger.error(f"Error registering tools: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Tool registration failed: {e}")

    def register_resources(self) -> None:
        """Register all available resources."""
        # Media management resources
        media_resources = media_manager.get_resources(self.settings)
        for resource_name, resource_func in media_resources.items():
            self.mcp.add_resource(resource_func)
            self._registered_resources[resource_name] = resource_func

        # Session data resources
        session_resources = session_data.get_resources(self.settings)
        for resource_name, resource_func in session_resources.items():
            self.mcp.add_resource(resource_func)
            self._registered_resources[resource_name] = resource_func

    def register_prompts(self) -> None:
        """Register all available prompts."""
        # Content generation prompts
        content_prompt_templates = content_prompts.get_prompts(self.settings)
        for prompt_name, prompt_func in content_prompt_templates.items():
            self.mcp.add_prompt(prompt_func)
            self._registered_prompts[prompt_name] = prompt_func

        # Analysis prompts
        analysis_prompt_templates = analysis_prompts.get_prompts(self.settings)
        for prompt_name, prompt_func in analysis_prompt_templates.items():
            self.mcp.add_prompt(prompt_func)
            self._registered_prompts[prompt_name] = prompt_func

    def get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities for MCP negotiation."""
        return {
            "tools": list(self._registered_tools.keys()),
            "resources": list(self._registered_resources.keys()),
            "prompts": list(self._registered_prompts.keys()),
            "features": {
                "screen_recording": True,
                "audio_transcription": True,
                "content_generation": True,
                "multi_format_output": True,
                "ai_processing": True
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
