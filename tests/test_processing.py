"""
Unit tests for processing functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from devstudio_mcp.tools.processing import ProcessingManager, TranscriptionResult, ContentAnalysis
from devstudio_mcp.utils.exceptions import TranscriptionError, ValidationError, AuthenticationError


class TestProcessingManager:
    """Test cases for ProcessingManager."""

    def test_initialization(self, processing_manager):
        """Test ProcessingManager initialization."""
        assert processing_manager.openai_client is not None
        assert processing_manager.anthropic_client is not None
        assert processing_manager.gemini_client is not None

    def test_initialization_without_api_keys(self):
        """Test ProcessingManager initialization without API keys."""
        from devstudio_mcp.config import Settings
        settings = Settings()  # No API keys
        manager = ProcessingManager(settings)

        assert manager.openai_client is None
        assert manager.anthropic_client is None
        assert manager.gemini_client is None

    @pytest.mark.asyncio
    async def test_transcribe_audio_file_openai_success(self, processing_manager, sample_audio_file):
        """Test successful audio transcription with OpenAI."""
        # Mock the OpenAI transcription response
        mock_transcript = MagicMock()
        mock_transcript.text = "Hello, this is a test transcription."
        mock_transcript.language = "en"
        mock_transcript.duration = 10.5
        mock_transcript.words = [{"word": "Hello", "start": 0.0, "end": 1.0}]

        processing_manager.openai_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcript)

        result = await processing_manager.transcribe_audio_file(
            file_path=sample_audio_file,
            provider="openai"
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, this is a test transcription."
        assert result.language == "en"
        assert result.duration == 10.5
        assert result.provider == "openai"

    @pytest.mark.asyncio
    async def test_transcribe_audio_file_not_found(self, processing_manager):
        """Test transcription with non-existent file."""
        with pytest.raises(ValidationError) as exc_info:
            await processing_manager.transcribe_audio_file(
                file_path="nonexistent.wav",
                provider="openai"
            )
        assert "Audio file not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_transcribe_audio_file_invalid_format(self, processing_manager, temp_dir):
        """Test transcription with invalid file format."""
        invalid_file = temp_dir / "test.txt"
        invalid_file.write_text("This is not an audio file")

        with pytest.raises(ValidationError) as exc_info:
            await processing_manager.transcribe_audio_file(
                file_path=invalid_file,
                provider="openai"
            )
        assert "Invalid audio file format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_transcribe_audio_file_no_openai_key(self, processing_manager, sample_audio_file):
        """Test transcription without OpenAI API key."""
        processing_manager.openai_client = None

        with pytest.raises(AuthenticationError) as exc_info:
            await processing_manager.transcribe_audio_file(
                file_path=sample_audio_file,
                provider="openai"
            )
        assert "OpenAI API key not configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_transcribe_audio_file_unsupported_provider(self, processing_manager, sample_audio_file):
        """Test transcription with unsupported provider."""
        with pytest.raises(ValidationError) as exc_info:
            await processing_manager.transcribe_audio_file(
                file_path=sample_audio_file,
                provider="unsupported"
            )
        assert "Unsupported transcription provider" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analyze_content_openai_success(self, processing_manager, sample_transcript):
        """Test successful content analysis with OpenAI."""
        # Mock the OpenAI chat response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "summary": "This is a Python tutorial about functions",
            "key_topics": ["Python", "functions", "programming"],
            "technical_terms": ["def", "parameter", "return"],
            "code_snippets": [{"language": "python", "description": "function definition"}],
            "chapters": [{"title": "Introduction", "start_time": "00:00"}],
            "sentiment": "educational"
        }
        '''

        processing_manager.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await processing_manager.analyze_content(
            text=sample_transcript,
            provider="openai"
        )

        assert isinstance(result, ContentAnalysis)
        assert "Python tutorial" in result.summary
        assert "Python" in result.key_topics
        assert "def" in result.technical_terms

    @pytest.mark.asyncio
    async def test_analyze_content_empty_text(self, processing_manager):
        """Test content analysis with empty text."""
        with pytest.raises(ValidationError) as exc_info:
            await processing_manager.analyze_content(
                text="",
                provider="openai"
            )
        assert "Empty text provided for analysis" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_code_snippets(self, processing_manager):
        """Test code snippet extraction."""
        text_with_code = '''
        Here's a Python function:

        ```python
        def hello(name):
            return f"Hello, {name}!"
        ```

        And some inline code: `print("Hello")`

        Another example:
        ```javascript
        function greet(name) {
            return `Hello, ${name}!`;
        }
        ```
        '''

        snippets = await processing_manager.extract_code_snippets(text_with_code)

        assert len(snippets) >= 2  # At least Python and JavaScript snippets

        # Check for Python snippet
        python_snippet = next((s for s in snippets if s["language"] == "python"), None)
        assert python_snippet is not None
        assert "def hello" in python_snippet["code"]

        # Check for JavaScript snippet
        js_snippet = next((s for s in snippets if s["language"] == "javascript"), None)
        assert js_snippet is not None
        assert "function greet" in js_snippet["code"]

    @pytest.mark.asyncio
    async def test_parse_analysis_response_valid_json(self, processing_manager):
        """Test parsing valid JSON analysis response."""
        json_response = '''
        {
            "summary": "Test summary",
            "key_topics": ["topic1", "topic2"],
            "technical_terms": ["term1", "term2"],
            "code_snippets": [{"language": "python", "description": "test"}],
            "chapters": [{"title": "Chapter 1", "start_time": "00:00"}],
            "sentiment": "educational"
        }
        '''

        result = processing_manager._parse_analysis_response(json_response)

        assert isinstance(result, ContentAnalysis)
        assert result.summary == "Test summary"
        assert result.key_topics == ["topic1", "topic2"]
        assert result.technical_terms == ["term1", "term2"]
        assert len(result.code_snippets) == 1
        assert len(result.chapters) == 1

    @pytest.mark.asyncio
    async def test_parse_analysis_response_invalid_json(self, processing_manager):
        """Test parsing invalid JSON analysis response."""
        invalid_response = "This is not valid JSON"

        result = processing_manager._parse_analysis_response(invalid_response)

        assert isinstance(result, ContentAnalysis)
        assert "Analysis completed but parsing failed" in result.summary
        assert result.key_topics == []
        assert result.technical_terms == []


class TestTranscriptionResult:
    """Test cases for TranscriptionResult model."""

    def test_transcription_result_creation(self):
        """Test TranscriptionResult model creation."""
        result = TranscriptionResult(
            text="Test transcription",
            provider="openai",
            model="whisper-1"
        )

        assert result.text == "Test transcription"
        assert result.provider == "openai"
        assert result.model == "whisper-1"
        assert result.confidence is None
        assert result.language is None


class TestContentAnalysis:
    """Test cases for ContentAnalysis model."""

    def test_content_analysis_creation(self):
        """Test ContentAnalysis model creation."""
        analysis = ContentAnalysis(
            summary="Test summary",
            key_topics=["topic1", "topic2"],
            technical_terms=["term1", "term2"],
            code_snippets=[{"language": "python", "code": "print('hello')"}],
            chapters=[{"title": "Chapter 1", "start_time": "00:00"}]
        )

        assert analysis.summary == "Test summary"
        assert len(analysis.key_topics) == 2
        assert len(analysis.technical_terms) == 2
        assert len(analysis.code_snippets) == 1
        assert len(analysis.chapters) == 1
        assert analysis.sentiment is None