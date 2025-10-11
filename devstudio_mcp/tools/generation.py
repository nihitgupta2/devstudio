"""
Content generation tools for DevStudio MCP server.

Implements markdown blog posts, documentation, course outlines,
and multi-format content generation with AI assistance.
"""

import asyncio
import datetime
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openai
from anthropic import Anthropic
import google.generativeai as genai
from pydantic import BaseModel, Field

from ..config import Settings
from ..utils.exceptions import ContentGenerationError, ValidationError, AuthenticationError, handle_mcp_error
from ..utils.logger import setup_logger

logger = setup_logger()


class ContentTemplate(BaseModel):
    """Model for content templates."""
    name: str
    description: str
    template: str
    variables: List[str]
    output_format: str


class GeneratedContent(BaseModel):
    """Model for generated content results."""
    title: str
    content: str
    format: str
    metadata: Dict[str, Any]
    word_count: int
    estimated_read_time: int


class GenerationManager:
    """Manages content generation operations."""

    def __init__(self, settings: Settings):
        """Initialize generation manager with AI providers."""
        self.settings = settings
        self.logger = logger

        # Initialize AI clients (reuse from processing module)
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_client = None

        self._setup_ai_clients()
        self._load_templates()

    def _setup_ai_clients(self) -> None:
        """Setup AI provider clients."""
        try:
            if self.settings.ai.openai_api_key:
                self.openai_client = openai.OpenAI(api_key=self.settings.ai.openai_api_key)

            if self.settings.ai.anthropic_api_key:
                self.anthropic_client = Anthropic(api_key=self.settings.ai.anthropic_api_key)

            if self.settings.ai.google_api_key:
                genai.configure(api_key=self.settings.ai.google_api_key)
                self.gemini_client = genai.GenerativeModel('gemini-pro')

        except Exception as e:
            self.logger.warning(f"AI client setup warning: {e}")

    def _load_templates(self) -> None:
        """Load content generation templates."""
        self.templates = {
            "blog_post": ContentTemplate(
                name="Technical Blog Post",
                description="Technical blog post with code examples",
                template="""
# {title}

## Introduction
{introduction}

## Main Content
{main_content}

## Code Examples
{code_examples}

## Conclusion
{conclusion}

---
*Generated on {date} with DevStudio MCP*
                """.strip(),
                variables=["title", "introduction", "main_content", "code_examples", "conclusion", "date"],
                output_format="markdown"
            ),
            "documentation": ContentTemplate(
                name="Technical Documentation",
                description="API or feature documentation",
                template="""
# {title}

## Overview
{overview}

## Prerequisites
{prerequisites}

## Usage
{usage}

## Examples
{examples}

## API Reference
{api_reference}

## Troubleshooting
{troubleshooting}
                """.strip(),
                variables=["title", "overview", "prerequisites", "usage", "examples", "api_reference", "troubleshooting"],
                output_format="markdown"
            ),
            "course_outline": ContentTemplate(
                name="Course Outline",
                description="Educational course structure",
                template="""
# {course_title}

## Course Overview
{overview}

## Learning Objectives
{objectives}

## Course Structure
{structure}

## Prerequisites
{prerequisites}

## Resources
{resources}
                """.strip(),
                variables=["course_title", "overview", "objectives", "structure", "prerequisites", "resources"],
                output_format="markdown"
            ),
            "youtube_description": ContentTemplate(
                name="YouTube Video Description",
                description="YouTube video description with timestamps",
                template="""
{video_description}

## Timestamps
{timestamps}

## Resources Mentioned
{resources}

## Connect with me
{social_links}

#programming #tutorial #coding
                """.strip(),
                variables=["video_description", "timestamps", "resources", "social_links"],
                output_format="text"
            )
        }

    async def generate_blog_post(
        self,
        title: str,
        transcript: str,
        code_snippets: Optional[List[Dict[str, str]]] = None,
        provider: str = "openai",
        style: str = "technical"
    ) -> GeneratedContent:
        """Generate a technical blog post from transcript and code."""
        try:
            # Build generation prompt
            prompt = self._build_blog_prompt(title, transcript, code_snippets or [], style)

            # Generate content with specified provider
            if provider.lower() == "openai":
                content = await self._generate_with_openai(prompt, "blog_post")
            elif provider.lower() == "anthropic":
                content = await self._generate_with_anthropic(prompt, "blog_post")
            elif provider.lower() == "google":
                content = await self._generate_with_gemini(prompt, "blog_post")
            else:
                raise ValidationError(f"Unsupported generation provider: {provider}")

            # Process and format the content
            formatted_content = self._format_blog_content(content, title, code_snippets or [])

            return GeneratedContent(
                title=title,
                content=formatted_content,
                format="markdown",
                metadata={
                    "provider": provider,
                    "style": style,
                    "generated_at": datetime.datetime.now().isoformat(),
                    "has_code": bool(code_snippets)
                },
                word_count=len(formatted_content.split()),
                estimated_read_time=max(1, len(formatted_content.split()) // 200)
            )

        except Exception as e:
            raise ContentGenerationError(f"Blog post generation failed: {e}", content_type="blog_post", provider=provider)

    async def generate_documentation(
        self,
        title: str,
        content_data: Dict[str, Any],
        doc_type: str = "api",
        provider: str = "openai"
    ) -> GeneratedContent:
        """Generate technical documentation."""
        try:
            prompt = self._build_documentation_prompt(title, content_data, doc_type)

            if provider.lower() == "openai":
                content = await self._generate_with_openai(prompt, "documentation")
            elif provider.lower() == "anthropic":
                content = await self._generate_with_anthropic(prompt, "documentation")
            elif provider.lower() == "google":
                content = await self._generate_with_gemini(prompt, "documentation")
            else:
                raise ValidationError(f"Unsupported generation provider: {provider}")

            formatted_content = self._format_documentation_content(content, title, doc_type)

            return GeneratedContent(
                title=title,
                content=formatted_content,
                format="markdown",
                metadata={
                    "provider": provider,
                    "doc_type": doc_type,
                    "generated_at": datetime.datetime.now().isoformat()
                },
                word_count=len(formatted_content.split()),
                estimated_read_time=max(1, len(formatted_content.split()) // 200)
            )

        except Exception as e:
            raise ContentGenerationError(f"Documentation generation failed: {e}", content_type="documentation", provider=provider)

    async def generate_course_outline(
        self,
        course_title: str,
        learning_objectives: List[str],
        duration: str,
        skill_level: str = "intermediate",
        provider: str = "openai"
    ) -> GeneratedContent:
        """Generate course outline and structure."""
        try:
            prompt = self._build_course_prompt(course_title, learning_objectives, duration, skill_level)

            if provider.lower() == "openai":
                content = await self._generate_with_openai(prompt, "course_outline")
            elif provider.lower() == "anthropic":
                content = await self._generate_with_anthropic(prompt, "course_outline")
            elif provider.lower() == "google":
                content = await self._generate_with_gemini(prompt, "course_outline")
            else:
                raise ValidationError(f"Unsupported generation provider: {provider}")

            formatted_content = self._format_course_content(content, course_title)

            return GeneratedContent(
                title=course_title,
                content=formatted_content,
                format="markdown",
                metadata={
                    "provider": provider,
                    "skill_level": skill_level,
                    "duration": duration,
                    "generated_at": datetime.datetime.now().isoformat()
                },
                word_count=len(formatted_content.split()),
                estimated_read_time=max(1, len(formatted_content.split()) // 200)
            )

        except Exception as e:
            raise ContentGenerationError(f"Course outline generation failed: {e}", content_type="course_outline", provider=provider)

    async def _generate_with_openai(self, prompt: str, content_type: str) -> str:
        """Generate content using OpenAI."""
        if not self.openai_client:
            raise AuthenticationError("OpenAI API key not configured", provider="openai")

        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are an expert technical content creator specializing in {content_type}."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )

            return response.choices[0].message.content

        except Exception as e:
            raise ContentGenerationError(f"OpenAI generation failed: {e}", provider="openai")

    async def _generate_with_anthropic(self, prompt: str, content_type: str) -> str:
        """Generate content using Anthropic Claude."""
        if not self.anthropic_client:
            raise AuthenticationError("Anthropic API key not configured", provider="anthropic")

        try:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-sonnet-20240229",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text

        except Exception as e:
            raise ContentGenerationError(f"Anthropic generation failed: {e}", provider="anthropic")

    async def _generate_with_gemini(self, prompt: str, content_type: str) -> str:
        """Generate content using Google Gemini."""
        if not self.gemini_client:
            raise AuthenticationError("Google API key not configured", provider="google")

        try:
            response = await asyncio.to_thread(
                self.gemini_client.generate_content,
                prompt
            )

            return response.text

        except Exception as e:
            raise ContentGenerationError(f"Gemini generation failed: {e}", provider="google")

    def _build_blog_prompt(self, title: str, transcript: str, code_snippets: List[Dict[str, str]], style: str) -> str:
        """Build blog post generation prompt."""
        code_context = ""
        if code_snippets:
            code_context = "\n\nCode snippets to include:\n"
            for i, snippet in enumerate(code_snippets, 1):
                code_context += f"{i}. {snippet.get('language', 'Code')}: {snippet.get('description', 'Code snippet')}\n"
                code_context += f"```{snippet.get('language', '')}\n{snippet.get('code', '')}\n```\n\n"

        return f"""
        Create a technical blog post with the following requirements:

        Title: {title}
        Style: {style}

        Source transcript:
        {transcript}
        {code_context}

        Please create a well-structured blog post that:
        1. Has a compelling introduction
        2. Is organized with clear sections and headers
        3. Includes the provided code snippets naturally
        4. Has a strong conclusion
        5. Uses markdown formatting
        6. Is engaging for developers and technical readers

        Make it informative, practical, and easy to follow.
        """

    def _build_documentation_prompt(self, title: str, content_data: Dict[str, Any], doc_type: str) -> str:
        """Build documentation generation prompt."""
        return f"""
        Create comprehensive technical documentation for: {title}

        Type: {doc_type}
        Content data: {content_data}

        Please create documentation that includes:
        1. Clear overview and purpose
        2. Prerequisites and requirements
        3. Step-by-step usage instructions
        4. Code examples with explanations
        5. API reference if applicable
        6. Common troubleshooting issues
        7. Best practices and tips

        Use markdown formatting and make it developer-friendly.
        """

    def _build_course_prompt(self, title: str, objectives: List[str], duration: str, skill_level: str) -> str:
        """Build course outline generation prompt."""
        objectives_str = "\n".join(f"- {obj}" for obj in objectives)

        return f"""
        Create a comprehensive course outline for: {title}

        Duration: {duration}
        Skill Level: {skill_level}
        Learning Objectives:
        {objectives_str}

        Please create a course outline that includes:
        1. Course overview and description
        2. Detailed learning objectives
        3. Module/lesson breakdown with timing
        4. Prerequisites and requirements
        5. Projects and assignments
        6. Resources and materials needed
        7. Assessment methods

        Structure it as a professional course curriculum.
        """

    def _format_blog_content(self, content: str, title: str, code_snippets: List[Dict[str, str]]) -> str:
        """Format and enhance blog post content."""
        # Add frontmatter
        frontmatter = f"""---
title: "{title}"
date: {datetime.datetime.now().strftime("%Y-%m-%d")}
tags: ["programming", "tutorial", "devstudio"]
author: "DevStudio MCP"
---

"""

        # Clean and format content
        formatted = frontmatter + content.strip()

        # Add footer
        formatted += f"\n\n---\n*This post was generated using DevStudio MCP on {datetime.datetime.now().strftime('%Y-%m-%d')}*"

        return formatted

    def _format_documentation_content(self, content: str, title: str, doc_type: str) -> str:
        """Format documentation content."""
        header = f"# {title}\n\n*{doc_type.title()} Documentation*\n\n"
        footer = f"\n\n---\n*Documentation generated with DevStudio MCP*"

        return header + content.strip() + footer

    def _format_course_content(self, content: str, title: str) -> str:
        """Format course outline content."""
        header = f"# {title}\n\n*Course Outline*\n\n"
        footer = f"\n\n---\n*Course outline generated with DevStudio MCP*"

        return header + content.strip() + footer

    async def generate_summary(self, text: str, length: str = "medium", provider: str = "openai") -> str:
        """Generate content summary."""
        try:
            length_guide = {
                "short": "2-3 sentences",
                "medium": "1-2 paragraphs",
                "long": "3-4 paragraphs"
            }

            prompt = f"""
            Create a {length} summary ({length_guide.get(length, "1-2 paragraphs")}) of the following content:

            {text}

            Make it informative and capture the key points.
            """

            if provider.lower() == "openai":
                content = await self._generate_with_openai(prompt, "summary")
            elif provider.lower() == "anthropic":
                content = await self._generate_with_anthropic(prompt, "summary")
            elif provider.lower() == "google":
                content = await self._generate_with_gemini(prompt, "summary")
            else:
                raise ValidationError(f"Unsupported generation provider: {provider}")

            return content.strip()

        except Exception as e:
            raise ContentGenerationError(f"Summary generation failed: {e}", content_type="summary", provider=provider)


# Initialize global generation manager
generation_manager = None


def get_tools(settings: Settings) -> Dict[str, Any]:
    """Get generation tools for MCP registration."""
    global generation_manager
    generation_manager = GenerationManager(settings)

    @handle_mcp_error
    async def generate_blog_post(
        title: str,
        transcript: str,
        code_snippets: Optional[List[Dict[str, str]]] = None,
        provider: str = "openai",
        style: str = "technical"
    ) -> Dict[str, Any]:
        """
        Generate a technical blog post from transcript and code snippets.

        Args:
            title: Blog post title
            transcript: Source transcript text
            code_snippets: List of code snippets to include
            provider: AI provider to use
            style: Writing style (technical, tutorial, casual)

        Returns:
            Generated blog post in markdown format
        """
        result = await generation_manager.generate_blog_post(
            title=title,
            transcript=transcript,
            code_snippets=code_snippets,
            provider=provider,
            style=style
        )

        return {
            "title": result.title,
            "content": result.content,
            "format": result.format,
            "word_count": result.word_count,
            "estimated_read_time": result.estimated_read_time,
            "metadata": result.metadata
        }

    @handle_mcp_error
    async def create_documentation(
        title: str,
        content_data: Dict[str, Any],
        doc_type: str = "api",
        provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        Generate technical documentation from content data.

        Args:
            title: Documentation title
            content_data: Source data and information
            doc_type: Type of documentation (api, feature, guide)
            provider: AI provider to use

        Returns:
            Generated documentation in markdown format
        """
        result = await generation_manager.generate_documentation(
            title=title,
            content_data=content_data,
            doc_type=doc_type,
            provider=provider
        )

        return {
            "title": result.title,
            "content": result.content,
            "format": result.format,
            "word_count": result.word_count,
            "estimated_read_time": result.estimated_read_time,
            "metadata": result.metadata
        }

    @handle_mcp_error
    async def generate_summary(
        text: str,
        length: str = "medium",
        provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        Generate a summary of the provided text.

        Args:
            text: Text to summarize
            length: Summary length (short, medium, long)
            provider: AI provider to use

        Returns:
            Generated summary text
        """
        summary = await generation_manager.generate_summary(
            text=text,
            length=length,
            provider=provider
        )

        return {
            "summary": summary,
            "length": length,
            "word_count": len(summary.split()),
            "provider": provider
        }

    @handle_mcp_error
    async def create_course_outline(
        course_title: str,
        learning_objectives: List[str],
        duration: str,
        skill_level: str = "intermediate",
        provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        Generate a course outline and structure.

        Args:
            course_title: Title of the course
            learning_objectives: List of learning objectives
            duration: Course duration (e.g., "4 weeks", "20 hours")
            skill_level: Target skill level (beginner, intermediate, advanced)
            provider: AI provider to use

        Returns:
            Generated course outline in markdown format
        """
        result = await generation_manager.generate_course_outline(
            course_title=course_title,
            learning_objectives=learning_objectives,
            duration=duration,
            skill_level=skill_level,
            provider=provider
        )

        return {
            "title": result.title,
            "content": result.content,
            "format": result.format,
            "word_count": result.word_count,
            "estimated_read_time": result.estimated_read_time,
            "metadata": result.metadata
        }

    return {
        "generate_blog_post": generate_blog_post,
        "create_documentation": create_documentation,
        "generate_summary": generate_summary,
        "create_course_outline": create_course_outline
    }