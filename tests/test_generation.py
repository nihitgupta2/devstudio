"""
Unit tests for content generation functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from devstudio_mcp.tools.generation import GenerationManager, GeneratedContent, ContentTemplate
from devstudio_mcp.utils.exceptions import ContentGenerationError, ValidationError, AuthenticationError


class TestGenerationManager:
    """Test cases for GenerationManager."""

    def test_initialization(self, generation_manager):
        """Test GenerationManager initialization."""
        assert generation_manager.openai_client is not None
        assert generation_manager.anthropic_client is not None
        assert generation_manager.gemini_client is not None
        assert len(generation_manager.templates) > 0

    def test_templates_loaded(self, generation_manager):
        """Test that content templates are properly loaded."""
        templates = generation_manager.templates

        assert "blog_post" in templates
        assert "documentation" in templates
        assert "course_outline" in templates
        assert "youtube_description" in templates

        # Check blog post template
        blog_template = templates["blog_post"]
        assert isinstance(blog_template, ContentTemplate)
        assert blog_template.name == "Technical Blog Post"
        assert "title" in blog_template.variables
        assert "markdown" in blog_template.output_format.lower()

    @pytest.mark.asyncio
    async def test_generate_blog_post_openai_success(self, generation_manager, sample_transcript):
        """Test successful blog post generation with OpenAI."""
        # Mock the OpenAI chat response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        # Python Functions Tutorial

        ## Introduction
        Welcome to this comprehensive guide on Python functions.

        ## Main Content
        Functions are reusable blocks of code that perform specific tasks.

        ## Code Examples
        ```python
        def greet(name):
            return f"Hello, {name}!"
        ```

        ## Conclusion
        Functions make your code more organized and reusable.
        '''

        generation_manager.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await generation_manager.generate_blog_post(
            title="Python Functions Tutorial",
            transcript=sample_transcript,
            provider="openai"
        )

        assert isinstance(result, GeneratedContent)
        assert result.title == "Python Functions Tutorial"
        assert result.format == "markdown"
        assert "Python Functions Tutorial" in result.content
        assert result.word_count > 0
        assert result.estimated_read_time > 0
        assert result.metadata["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_generate_blog_post_with_code_snippets(self, generation_manager, sample_transcript):
        """Test blog post generation with code snippets."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "# Test Blog Post\n\nContent with code examples."

        generation_manager.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        code_snippets = [
            {"language": "python", "code": "print('hello')", "description": "Print statement"},
            {"language": "javascript", "code": "console.log('hello')", "description": "Console log"}
        ]

        result = await generation_manager.generate_blog_post(
            title="Test Blog Post",
            transcript=sample_transcript,
            code_snippets=code_snippets,
            provider="openai"
        )

        assert result.metadata["has_code"] is True

    @pytest.mark.asyncio
    async def test_generate_blog_post_no_openai_key(self, generation_manager, sample_transcript):
        """Test blog post generation without OpenAI API key."""
        generation_manager.openai_client = None

        with pytest.raises(AuthenticationError) as exc_info:
            await generation_manager.generate_blog_post(
                title="Test",
                transcript=sample_transcript,
                provider="openai"
            )
        assert "OpenAI API key not configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_blog_post_unsupported_provider(self, generation_manager, sample_transcript):
        """Test blog post generation with unsupported provider."""
        with pytest.raises(ValidationError) as exc_info:
            await generation_manager.generate_blog_post(
                title="Test",
                transcript=sample_transcript,
                provider="unsupported"
            )
        assert "Unsupported generation provider" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_documentation_success(self, generation_manager):
        """Test successful documentation generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        # API Documentation

        ## Overview
        This API provides comprehensive functionality.

        ## Usage
        Use the API endpoints to interact with the system.
        '''

        generation_manager.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        content_data = {
            "endpoints": ["/api/users", "/api/posts"],
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "authentication": "Bearer token required"
        }

        result = await generation_manager.generate_documentation(
            title="API Documentation",
            content_data=content_data,
            doc_type="api",
            provider="openai"
        )

        assert isinstance(result, GeneratedContent)
        assert result.title == "API Documentation"
        assert result.format == "markdown"
        assert "API Documentation" in result.content
        assert result.metadata["doc_type"] == "api"

    @pytest.mark.asyncio
    async def test_generate_course_outline_success(self, generation_manager):
        """Test successful course outline generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        # Python Programming Course

        ## Course Overview
        Comprehensive Python programming course for beginners.

        ## Learning Objectives
        - Understand Python syntax
        - Build real-world applications

        ## Course Structure
        Module 1: Python Basics
        Module 2: Data Structures
        Module 3: Object-Oriented Programming
        '''

        generation_manager.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        learning_objectives = [
            "Understand Python syntax and semantics",
            "Build practical applications",
            "Master object-oriented programming"
        ]

        result = await generation_manager.generate_course_outline(
            course_title="Python Programming Course",
            learning_objectives=learning_objectives,
            duration="8 weeks",
            skill_level="beginner",
            provider="openai"
        )

        assert isinstance(result, GeneratedContent)
        assert result.title == "Python Programming Course"
        assert result.format == "markdown"
        assert "Python Programming Course" in result.content
        assert result.metadata["skill_level"] == "beginner"
        assert result.metadata["duration"] == "8 weeks"

    @pytest.mark.asyncio
    async def test_generate_summary_success(self, generation_manager, sample_transcript):
        """Test successful summary generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This tutorial covers Python functions and their usage."

        generation_manager.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await generation_manager.generate_summary(
            text=sample_transcript,
            length="short",
            provider="openai"
        )

        assert "Python functions" in result

    def test_build_blog_prompt(self, generation_manager):
        """Test blog post prompt building."""
        code_snippets = [
            {"language": "python", "code": "print('hello')", "description": "Print statement"}
        ]

        prompt = generation_manager._build_blog_prompt(
            title="Test Blog",
            transcript="Test transcript",
            code_snippets=code_snippets,
            style="technical"
        )

        assert "Test Blog" in prompt
        assert "Test transcript" in prompt
        assert "python" in prompt
        assert "technical" in prompt

    def test_build_documentation_prompt(self, generation_manager):
        """Test documentation prompt building."""
        content_data = {"endpoints": ["/api/users"], "methods": ["GET"]}

        prompt = generation_manager._build_documentation_prompt(
            title="API Docs",
            content_data=content_data,
            doc_type="api"
        )

        assert "API Docs" in prompt
        assert "api" in prompt
        assert str(content_data) in prompt

    def test_build_course_prompt(self, generation_manager):
        """Test course outline prompt building."""
        objectives = ["Learn Python", "Build apps"]

        prompt = generation_manager._build_course_prompt(
            title="Python Course",
            objectives=objectives,
            duration="4 weeks",
            skill_level="beginner"
        )

        assert "Python Course" in prompt
        assert "4 weeks" in prompt
        assert "beginner" in prompt
        assert "Learn Python" in prompt

    def test_format_blog_content(self, generation_manager):
        """Test blog content formatting."""
        content = "# Test Blog\n\nThis is test content."
        title = "Test Blog Post"

        formatted = generation_manager._format_blog_content(content, title, [])

        assert "---" in formatted  # Frontmatter
        assert f'title: "{title}"' in formatted
        assert "DevStudio MCP" in formatted
        assert content in formatted

    def test_format_documentation_content(self, generation_manager):
        """Test documentation content formatting."""
        content = "# API Docs\n\nDocumentation content."
        title = "API Documentation"

        formatted = generation_manager._format_documentation_content(
            content, title, "api"
        )

        assert title in formatted
        assert "Api Documentation" in formatted
        assert "DevStudio MCP" in formatted
        assert content in formatted

    def test_format_course_content(self, generation_manager):
        """Test course content formatting."""
        content = "# Course\n\nCourse content."
        title = "Python Course"

        formatted = generation_manager._format_course_content(content, title)

        assert title in formatted
        assert "Course Outline" in formatted
        assert "DevStudio MCP" in formatted
        assert content in formatted


class TestGeneratedContent:
    """Test cases for GeneratedContent model."""

    def test_generated_content_creation(self):
        """Test GeneratedContent model creation."""
        content = GeneratedContent(
            title="Test Content",
            content="# Test\n\nContent here.",
            format="markdown",
            metadata={"provider": "openai"},
            word_count=10,
            estimated_read_time=1
        )

        assert content.title == "Test Content"
        assert content.format == "markdown"
        assert content.word_count == 10
        assert content.estimated_read_time == 1
        assert content.metadata["provider"] == "openai"


class TestContentTemplate:
    """Test cases for ContentTemplate model."""

    def test_content_template_creation(self):
        """Test ContentTemplate model creation."""
        template = ContentTemplate(
            name="Test Template",
            description="A test template",
            template="# {title}\n\n{content}",
            variables=["title", "content"],
            output_format="markdown"
        )

        assert template.name == "Test Template"
        assert template.description == "A test template"
        assert "{title}" in template.template
        assert "title" in template.variables
        assert template.output_format == "markdown"