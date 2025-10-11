"""
Processing tools for DevStudio MCP server.

Implements AI transcription, content analysis, and code extraction
with support for multiple AI providers (OpenAI, Anthropic, Google).
"""

import asyncio
import json
import mimetypes
import re
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Union

import openai
from anthropic import Anthropic
import google.generativeai as genai
from pydantic import BaseModel, Field

from ..config import Settings
from ..utils.exceptions import (
    TranscriptionError,
    ValidationError,
    AuthenticationError,
    handle_mcp_error
)
from ..utils.logger import setup_logger

logger = setup_logger()


class TranscriptionResult(BaseModel):
    """Model for transcription results."""
    text: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    timestamps: Optional[List[Dict[str, Any]]] = None
    provider: str
    model: str


class ContentAnalysis(BaseModel):
    """Model for content analysis results."""
    summary: str
    key_topics: List[str]
    technical_terms: List[str]
    code_snippets: List[Dict[str, str]]
    chapters: List[Dict[str, Any]]
    sentiment: Optional[str] = None


class ProcessingManager:
    """Manages AI processing operations."""

    def __init__(self, settings: Settings):
        """Initialize processing manager with AI providers."""
        self.settings = settings
        self.logger = logger

        # Initialize AI clients
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_client = None

        self._setup_ai_clients()

    def _setup_ai_clients(self) -> None:
        """Setup AI provider clients."""
        try:
            # OpenAI
            if self.settings.ai.openai_api_key:
                self.openai_client = openai.OpenAI(api_key=self.settings.ai.openai_api_key)
                self.logger.info("OpenAI client initialized")

            # Anthropic
            if self.settings.ai.anthropic_api_key:
                self.anthropic_client = Anthropic(api_key=self.settings.ai.anthropic_api_key)
                self.logger.info("Anthropic client initialized")

            # Google Gemini
            if self.settings.ai.google_api_key:
                genai.configure(api_key=self.settings.ai.google_api_key)
                self.gemini_client = genai.GenerativeModel('gemini-pro')
                self.logger.info("Gemini client initialized")

        except Exception as e:
            self.logger.warning(f"AI client setup warning: {e}")

    async def transcribe_audio_file(
        self,
        file_path: Union[str, Path],
        provider: str = "openai",
        model: Optional[str] = None,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe audio file using specified AI provider."""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise ValidationError(f"Audio file not found: {file_path}")

            # Validate file format
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type or not mime_type.startswith('audio/'):
                raise ValidationError(f"Invalid audio file format: {file_path.suffix}")

            # Route to appropriate provider
            if provider.lower() == "openai":
                return await self._transcribe_with_openai(file_path, model, language)
            elif provider.lower() == "anthropic":
                return await self._transcribe_with_anthropic(file_path, model, language)
            elif provider.lower() == "google":
                return await self._transcribe_with_gemini(file_path, model, language)
            else:
                raise ValidationError(f"Unsupported transcription provider: {provider}")

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}", provider=provider, file_path=str(file_path))

    async def _transcribe_with_openai(
        self,
        file_path: Path,
        model: Optional[str],
        language: Optional[str]
    ) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper."""
        if not self.openai_client:
            raise AuthenticationError("OpenAI API key not configured", provider="openai")

        try:
            model = model or "whisper-1"

            with open(file_path, "rb") as audio_file:
                transcript = await asyncio.to_thread(
                    self.openai_client.audio.transcriptions.create,
                    model=model,
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )

            return TranscriptionResult(
                text=transcript.text,
                language=transcript.language,
                duration=transcript.duration,
                provider="openai",
                model=model,
                timestamps=getattr(transcript, 'words', None)
            )

        except Exception as e:
            raise TranscriptionError(f"OpenAI transcription failed: {e}", provider="openai")

    async def _transcribe_with_anthropic(
        self,
        file_path: Path,
        model: Optional[str],
        language: Optional[str]
    ) -> TranscriptionResult:
        """Transcribe using Anthropic (placeholder - they don't have audio transcription yet)."""
        raise TranscriptionError("Anthropic does not currently support audio transcription", provider="anthropic")

    async def _transcribe_with_gemini(
        self,
        file_path: Path,
        model: Optional[str],
        language: Optional[str]
    ) -> TranscriptionResult:
        """Transcribe using Google Gemini (placeholder - need to implement with Cloud Speech-to-Text)."""
        if not self.gemini_client:
            raise AuthenticationError("Google API key not configured", provider="google")

        # Note: This would need Google Cloud Speech-to-Text API
        raise TranscriptionError("Gemini audio transcription not yet implemented", provider="google")

    async def analyze_content(
        self,
        text: str,
        analysis_type: str = "comprehensive",
        provider: str = "openai"
    ) -> ContentAnalysis:
        """Analyze content for topics, technical terms, and structure."""
        try:
            if not text.strip():
                raise ValidationError("Empty text provided for analysis")

            # Route to appropriate provider
            if provider.lower() == "openai":
                return await self._analyze_with_openai(text, analysis_type)
            elif provider.lower() == "anthropic":
                return await self._analyze_with_anthropic(text, analysis_type)
            elif provider.lower() == "google":
                return await self._analyze_with_gemini(text, analysis_type)
            else:
                raise ValidationError(f"Unsupported analysis provider: {provider}")

        except Exception as e:
            raise TranscriptionError(f"Content analysis failed: {e}", provider=provider)

    async def _analyze_with_openai(self, text: str, analysis_type: str) -> ContentAnalysis:
        """Analyze content using OpenAI."""
        if not self.openai_client:
            raise AuthenticationError("OpenAI API key not configured", provider="openai")

        try:
            prompt = self._build_analysis_prompt(text, analysis_type)

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert content analyzer for technical content creators."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            # Parse the structured response
            analysis_text = response.choices[0].message.content
            return self._parse_analysis_response(analysis_text)

        except Exception as e:
            raise TranscriptionError(f"OpenAI analysis failed: {e}", provider="openai")

    async def _analyze_with_anthropic(self, text: str, analysis_type: str) -> ContentAnalysis:
        """Analyze content using Anthropic Claude."""
        if not self.anthropic_client:
            raise AuthenticationError("Anthropic API key not configured", provider="anthropic")

        try:
            prompt = self._build_analysis_prompt(text, analysis_type)

            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            analysis_text = response.content[0].text
            return self._parse_analysis_response(analysis_text)

        except Exception as e:
            raise TranscriptionError(f"Anthropic analysis failed: {e}", provider="anthropic")

    async def _analyze_with_gemini(self, text: str, analysis_type: str) -> ContentAnalysis:
        """Analyze content using Google Gemini."""
        if not self.gemini_client:
            raise AuthenticationError("Google API key not configured", provider="google")

        try:
            prompt = self._build_analysis_prompt(text, analysis_type)

            response = await asyncio.to_thread(
                self.gemini_client.generate_content,
                prompt
            )

            analysis_text = response.text
            return self._parse_analysis_response(analysis_text)

        except Exception as e:
            raise TranscriptionError(f"Gemini analysis failed: {e}", provider="google")

    def _build_analysis_prompt(self, text: str, analysis_type: str) -> str:
        """Build analysis prompt based on type."""
        base_prompt = f"""
        Analyze the following technical content and provide a structured analysis:

        Content:
        {text}

        Please provide:
        1. A concise summary (2-3 sentences)
        2. Key topics covered (list of 3-7 main topics)
        3. Technical terms mentioned (programming languages, frameworks, tools, etc.)
        4. Any code snippets found (language and brief description)
        5. Suggested chapter/section breaks with timestamps if available
        6. Overall sentiment (educational, tutorial, demo, etc.)

        Format your response as JSON with the following structure:
        {{
            "summary": "...",
            "key_topics": ["topic1", "topic2", ...],
            "technical_terms": ["term1", "term2", ...],
            "code_snippets": [{{"language": "python", "description": "function definition"}}, ...],
            "chapters": [{{"title": "Introduction", "start_time": "00:00", "description": "..."}}],
            "sentiment": "educational"
        }}
        """

        return base_prompt

    def _parse_analysis_response(self, response_text: str) -> ContentAnalysis:
        """Parse AI response into ContentAnalysis model."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Fallback parsing
                data = self._fallback_parse(response_text)

            return ContentAnalysis(
                summary=data.get("summary", "Content analysis completed"),
                key_topics=data.get("key_topics", []),
                technical_terms=data.get("technical_terms", []),
                code_snippets=data.get("code_snippets", []),
                chapters=data.get("chapters", []),
                sentiment=data.get("sentiment", "technical")
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse analysis response: {e}")
            return ContentAnalysis(
                summary="Analysis completed but parsing failed",
                key_topics=[],
                technical_terms=[],
                code_snippets=[],
                chapters=[]
            )

    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails."""
        return {
            "summary": "Content analyzed successfully",
            "key_topics": [],
            "technical_terms": [],
            "code_snippets": [],
            "chapters": [],
            "sentiment": "technical"
        }

    async def extract_code_snippets(self, text: str) -> List[Dict[str, str]]:
        """Extract and categorize code snippets from text."""
        try:
            code_blocks = []

            # Pattern for code blocks with language specification
            code_pattern = r'```(\w+)?\n(.*?)\n```'
            matches = re.findall(code_pattern, text, re.DOTALL)

            for language, code in matches:
                code_blocks.append({
                    "language": language or "unknown",
                    "code": code.strip(),
                    "description": f"{language or 'Code'} snippet"
                })

            # Also look for inline code
            inline_pattern = r'`([^`]+)`'
            inline_matches = re.findall(inline_pattern, text)

            for inline_code in inline_matches:
                if len(inline_code) > 10:  # Only longer inline code
                    code_blocks.append({
                        "language": "inline",
                        "code": inline_code,
                        "description": "Inline code snippet"
                    })

            return code_blocks

        except Exception as e:
            self.logger.warning(f"Code extraction failed: {e}")
            return []


# Initialize global processing manager
processing_manager = None


def get_tools(settings: Settings) -> Dict[str, Any]:
    """Get processing tools for MCP registration."""
    global processing_manager
    processing_manager = ProcessingManager(settings)

    @handle_mcp_error
    async def transcribe_audio(
        file_path: str,
        provider: str = "openai",
        model: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text using AI providers.

        Args:
            file_path: Path to audio file
            provider: AI provider (openai, anthropic, google)
            model: Specific model to use (optional)
            language: Source language (optional)

        Returns:
            Transcription result with text, confidence, and metadata
        """
        result = await processing_manager.transcribe_audio_file(
            file_path=file_path,
            provider=provider,
            model=model,
            language=language
        )

        return {
            "text": result.text,
            "confidence": result.confidence,
            "language": result.language,
            "duration": result.duration,
            "provider": result.provider,
            "model": result.model,
            "has_timestamps": bool(result.timestamps)
        }

    @handle_mcp_error
    async def analyze_content(
        text: str,
        analysis_type: str = "comprehensive",
        provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        Analyze content for topics, technical terms, and structure.

        Args:
            text: Text content to analyze
            analysis_type: Type of analysis (comprehensive, summary, technical)
            provider: AI provider to use

        Returns:
            Content analysis with topics, terms, and structure
        """
        result = await processing_manager.analyze_content(
            text=text,
            analysis_type=analysis_type,
            provider=provider
        )

        return {
            "summary": result.summary,
            "key_topics": result.key_topics,
            "technical_terms": result.technical_terms,
            "code_snippets": result.code_snippets,
            "chapters": result.chapters,
            "sentiment": result.sentiment
        }

    @handle_mcp_error
    async def extract_code(text: str) -> Dict[str, Any]:
        """
        Extract code snippets from text content.

        Args:
            text: Text content containing code

        Returns:
            Extracted code snippets with language detection
        """
        code_snippets = await processing_manager.extract_code_snippets(text)

        return {
            "code_snippets": code_snippets,
            "total_snippets": len(code_snippets),
            "languages_found": list(set(snippet["language"] for snippet in code_snippets))
        }

    return {
        "transcribe_audio": transcribe_audio,
        "analyze_content": analyze_content,
        "extract_code": extract_code
    }