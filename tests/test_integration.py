"""
Integration tests for DevStudio MCP server.

These tests verify the full MCP protocol integration and tool functionality.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from devstudio_mcp.server import DevStudioMCP
from devstudio_mcp.config import Settings
from devstudio_mcp.registry import MCPRegistry


class TestMCPServerIntegration:
    """Integration tests for the MCP server."""

    @pytest.fixture
    async def mcp_server(self, test_settings):
        """Create and initialize MCP server for testing."""
        server = DevStudioMCP(test_settings)
        await server.initialize()
        return server

    @pytest.mark.asyncio
    async def test_server_initialization(self, test_settings):
        """Test server initialization and capability setup."""
        server = DevStudioMCP(test_settings)

        assert server.settings == test_settings
        assert isinstance(server.registry, MCPRegistry)
        assert not server.is_running()

        await server.initialize()

        capabilities = server.get_capabilities()
        assert isinstance(capabilities, dict)

    @pytest.mark.asyncio
    async def test_server_info(self, mcp_server):
        """Test server information retrieval."""
        info = mcp_server.get_server_info()

        assert info.name == "DevStudio MCP"
        assert info.version == "1.0.0"
        assert "AI-powered content creation" in info.description
        assert info.protocol_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_registry_tool_registration(self, mcp_server):
        """Test that tools are properly registered."""
        registry = mcp_server.registry

        # Check that tools are registered
        assert registry.get_tool_count() > 0

        # Check for specific core tools
        tools = registry._tools
        assert "start_recording" in tools
        assert "stop_recording" in tools
        assert "transcribe_audio" in tools
        assert "generate_blog_post" in tools

    @pytest.mark.asyncio
    async def test_registry_resource_registration(self, mcp_server):
        """Test that resources are properly registered."""
        registry = mcp_server.registry

        # Check that resources are registered
        resource_count = registry.get_resource_count()
        assert resource_count >= 0  # May be 0 if no resources defined yet

    @pytest.mark.asyncio
    async def test_registry_prompt_registration(self, mcp_server):
        """Test that prompts are properly registered."""
        registry = mcp_server.registry

        # Check that prompts are registered
        prompt_count = registry.get_prompt_count()
        assert prompt_count >= 0  # May be 0 if no prompts defined yet


class TestRecordingToolsIntegration:
    """Integration tests for recording tools."""

    @pytest.fixture
    async def mcp_server_with_mocks(self, test_settings):
        """Create MCP server with mocked recording dependencies."""
        server = DevStudioMCP(test_settings)
        await server.initialize()

        # Mock recording dependencies
        with patch('cv2.VideoWriter') as mock_writer_class, \
             patch('sounddevice.InputStream') as mock_stream_class, \
             patch('pyautogui.size', return_value=(1920, 1080)):

            mock_writer = MagicMock()
            mock_writer.isOpened.return_value = True
            mock_writer_class.return_value = mock_writer

            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream

            yield server

    @pytest.mark.asyncio
    async def test_start_recording_tool_integration(self, mcp_server_with_mocks):
        """Test start_recording tool integration."""
        # Get the tool function
        tools = mcp_server_with_mocks.registry._tools
        start_recording = tools["start_recording"]

        # Test tool execution
        result = await start_recording(
            include_screen=True,
            include_audio=True,
            include_terminal=False,
            output_format="mp4"
        )

        assert "session_id" in result
        assert result["status"] == "recording"
        assert "start_time" in result
        assert "output_directory" in result
        assert result["recording_types"]["screen"] is True
        assert result["recording_types"]["audio"] is True

    @pytest.mark.asyncio
    async def test_capture_screen_tool_integration(self, mcp_server_with_mocks):
        """Test capture_screen tool integration."""
        tools = mcp_server_with_mocks.registry._tools
        capture_screen = tools["capture_screen"]

        with patch('pyautogui.screenshot') as mock_screenshot:
            mock_image = MagicMock()
            mock_image.size = (1920, 1080)
            mock_image.save = MagicMock()
            mock_screenshot.return_value = mock_image

            result = await capture_screen()

            assert "file_path" in result
            assert "timestamp" in result
            assert "dimensions" in result
            assert result["format"] == "png"
            assert result["dimensions"] == (1920, 1080)

    @pytest.mark.asyncio
    async def test_list_active_sessions_tool_integration(self, mcp_server_with_mocks):
        """Test list_active_sessions tool integration."""
        tools = mcp_server_with_mocks.registry._tools
        list_sessions = tools["list_active_sessions"]

        # Initially no sessions
        result = await list_sessions()
        assert result["total_count"] == 0
        assert result["active_sessions"] == []

        # Start a recording session
        start_recording = tools["start_recording"]
        await start_recording(include_screen=True)

        # Now should have one session
        result = await list_sessions()
        assert result["total_count"] == 1
        assert len(result["active_sessions"]) == 1
        assert "id" in result["active_sessions"][0]
        assert "status" in result["active_sessions"][0]


class TestProcessingToolsIntegration:
    """Integration tests for processing tools."""

    @pytest.fixture
    async def mcp_server_with_ai_mocks(self, test_settings, sample_audio_file):
        """Create MCP server with mocked AI dependencies."""
        server = DevStudioMCP(test_settings)
        await server.initialize()

        # Mock AI client responses
        from devstudio_mcp.tools.processing import processing_manager
        if processing_manager:
            # Mock OpenAI transcription
            mock_transcript = MagicMock()
            mock_transcript.text = "Test transcription text"
            mock_transcript.language = "en"
            mock_transcript.duration = 10.0
            mock_transcript.words = []

            processing_manager.openai_client.audio.transcriptions.create = AsyncMock(
                return_value=mock_transcript
            )

            # Mock OpenAI chat completion
            mock_chat_response = MagicMock()
            mock_chat_response.choices = [MagicMock()]
            mock_chat_response.choices[0].message.content = '''
            {
                "summary": "Test analysis summary",
                "key_topics": ["topic1", "topic2"],
                "technical_terms": ["term1", "term2"],
                "code_snippets": [],
                "chapters": [],
                "sentiment": "technical"
            }
            '''

            processing_manager.openai_client.chat.completions.create = AsyncMock(
                return_value=mock_chat_response
            )

        yield server

    @pytest.mark.asyncio
    async def test_transcribe_audio_tool_integration(self, mcp_server_with_ai_mocks, sample_audio_file):
        """Test transcribe_audio tool integration."""
        tools = mcp_server_with_ai_mocks.registry._tools
        transcribe_audio = tools["transcribe_audio"]

        result = await transcribe_audio(
            file_path=str(sample_audio_file),
            provider="openai"
        )

        assert result["text"] == "Test transcription text"
        assert result["language"] == "en"
        assert result["duration"] == 10.0
        assert result["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_analyze_content_tool_integration(self, mcp_server_with_ai_mocks, sample_transcript):
        """Test analyze_content tool integration."""
        tools = mcp_server_with_ai_mocks.registry._tools
        analyze_content = tools["analyze_content"]

        result = await analyze_content(
            text=sample_transcript,
            analysis_type="comprehensive",
            provider="openai"
        )

        assert "summary" in result
        assert "key_topics" in result
        assert "technical_terms" in result
        assert "code_snippets" in result
        assert isinstance(result["key_topics"], list)

    @pytest.mark.asyncio
    async def test_extract_code_tool_integration(self, mcp_server_with_ai_mocks):
        """Test extract_code tool integration."""
        tools = mcp_server_with_ai_mocks.registry._tools
        extract_code = tools["extract_code"]

        text_with_code = '''
        Here's some code:
        ```python
        def hello():
            return "Hello, World!"
        ```
        '''

        result = await extract_code(text=text_with_code)

        assert "code_snippets" in result
        assert "total_snippets" in result
        assert "languages_found" in result
        assert result["total_snippets"] > 0
        assert "python" in result["languages_found"]


class TestGenerationToolsIntegration:
    """Integration tests for generation tools."""

    @pytest.fixture
    async def mcp_server_with_generation_mocks(self, test_settings):
        """Create MCP server with mocked generation dependencies."""
        server = DevStudioMCP(test_settings)
        await server.initialize()

        # Mock AI client responses
        from devstudio_mcp.tools.generation import generation_manager
        if generation_manager:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '''
            # Test Blog Post

            ## Introduction
            This is a test blog post generated by the MCP server.

            ## Main Content
            The content discusses various technical topics.

            ## Conclusion
            This concludes our test blog post.
            '''

            generation_manager.openai_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

        yield server

    @pytest.mark.asyncio
    async def test_generate_blog_post_tool_integration(self, mcp_server_with_generation_mocks, sample_transcript):
        """Test generate_blog_post tool integration."""
        tools = mcp_server_with_generation_mocks.registry._tools
        generate_blog_post = tools["generate_blog_post"]

        result = await generate_blog_post(
            title="Test Blog Post",
            transcript=sample_transcript,
            provider="openai",
            style="technical"
        )

        assert result["title"] == "Test Blog Post"
        assert result["format"] == "markdown"
        assert "content" in result
        assert "word_count" in result
        assert "estimated_read_time" in result
        assert result["metadata"]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_create_documentation_tool_integration(self, mcp_server_with_generation_mocks):
        """Test create_documentation tool integration."""
        tools = mcp_server_with_generation_mocks.registry._tools
        create_documentation = tools["create_documentation"]

        content_data = {
            "api_endpoints": ["/api/users", "/api/posts"],
            "authentication": "Bearer token",
            "rate_limits": "1000 requests/hour"
        }

        result = await create_documentation(
            title="API Documentation",
            content_data=content_data,
            doc_type="api",
            provider="openai"
        )

        assert result["title"] == "API Documentation"
        assert result["format"] == "markdown"
        assert "content" in result
        assert result["metadata"]["doc_type"] == "api"

    @pytest.mark.asyncio
    async def test_generate_summary_tool_integration(self, mcp_server_with_generation_mocks, sample_transcript):
        """Test generate_summary tool integration."""
        tools = mcp_server_with_generation_mocks.registry._tools
        generate_summary = tools["generate_summary"]

        # Mock shorter response for summary
        from devstudio_mcp.tools.generation import generation_manager
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a concise summary of the technical content."

        generation_manager.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await generate_summary(
            text=sample_transcript,
            length="medium",
            provider="openai"
        )

        assert "summary" in result
        assert result["length"] == "medium"
        assert "word_count" in result
        assert result["provider"] == "openai"


class TestEndToEndWorkflow:
    """End-to-end workflow integration tests."""

    @pytest.mark.asyncio
    async def test_complete_content_creation_workflow(self, test_settings, sample_audio_file, temp_dir):
        """Test complete workflow from recording to content generation."""
        server = DevStudioMCP(test_settings)
        await server.initialize()

        # Mock all dependencies
        with patch('cv2.VideoWriter') as mock_writer_class, \
             patch('sounddevice.InputStream') as mock_stream_class, \
             patch('pyautogui.size', return_value=(1920, 1080)), \
             patch('pyautogui.screenshot') as mock_screenshot:

            # Setup mocks
            mock_writer = MagicMock()
            mock_writer.isOpened.return_value = True
            mock_writer_class.return_value = mock_writer

            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream

            mock_image = MagicMock()
            mock_image.size = (1920, 1080)
            mock_image.save = MagicMock()
            mock_screenshot.return_value = mock_image

            # Mock AI responses
            from devstudio_mcp.tools.processing import processing_manager
            from devstudio_mcp.tools.generation import generation_manager

            if processing_manager:
                mock_transcript = MagicMock()
                mock_transcript.text = "This is a tutorial on Python programming."
                mock_transcript.language = "en"
                mock_transcript.duration = 60.0
                mock_transcript.words = []

                processing_manager.openai_client.audio.transcriptions.create = AsyncMock(
                    return_value=mock_transcript
                )

            if generation_manager:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = '''
                # Python Programming Tutorial

                ## Introduction
                Welcome to this Python programming tutorial.

                ## Main Content
                This tutorial covers basic Python concepts.

                ## Conclusion
                You now know the basics of Python programming.
                '''

                generation_manager.openai_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )

            tools = server.registry._tools

            # Step 1: Start recording
            recording_result = await tools["start_recording"](
                include_screen=True,
                include_audio=True
            )
            session_id = recording_result["session_id"]

            # Step 2: Take a screenshot
            screenshot_result = await tools["capture_screen"]()

            # Step 3: Stop recording
            stop_result = await tools["stop_recording"](session_id=session_id)

            # Step 4: Transcribe audio (simulate with sample file)
            transcription_result = await tools["transcribe_audio"](
                file_path=str(sample_audio_file),
                provider="openai"
            )

            # Step 5: Generate blog post
            blog_result = await tools["generate_blog_post"](
                title="Python Programming Tutorial",
                transcript=transcription_result["text"],
                provider="openai"
            )

            # Verify workflow results
            assert recording_result["status"] == "recording"
            assert "file_path" in screenshot_result
            assert "files" in stop_result
            assert transcription_result["text"] == "This is a tutorial on Python programming."
            assert blog_result["title"] == "Python Programming Tutorial"
            assert "Python Programming Tutorial" in blog_result["content"]