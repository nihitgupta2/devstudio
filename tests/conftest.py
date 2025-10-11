"""
Pytest configuration and fixtures for DevStudio MCP tests.
"""

import asyncio
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from devstudio_mcp.config import Settings
from devstudio_mcp.tools.recording import RecordingManager
from devstudio_mcp.tools.processing import ProcessingManager
from devstudio_mcp.tools.generation import GenerationManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture
def test_settings(temp_dir):
    """Create test settings with temporary directories."""
    return Settings(
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        google_api_key="test-google-key",
        log_level="DEBUG"
    )


@pytest.fixture
def recording_manager(test_settings):
    """Create a RecordingManager instance for testing."""
    return RecordingManager(test_settings)


@pytest.fixture
def processing_manager(test_settings):
    """Create a ProcessingManager instance for testing."""
    manager = ProcessingManager(test_settings)
    # Mock the AI clients to avoid actual API calls
    manager.openai_client = AsyncMock()
    manager.anthropic_client = AsyncMock()
    manager.gemini_client = AsyncMock()
    return manager


@pytest.fixture
def generation_manager(test_settings):
    """Create a GenerationManager instance for testing."""
    manager = GenerationManager(test_settings)
    # Mock the AI clients to avoid actual API calls
    manager.openai_client = AsyncMock()
    manager.anthropic_client = AsyncMock()
    manager.gemini_client = AsyncMock()
    return manager


@pytest.fixture
def sample_transcript():
    """Sample transcript text for testing."""
    return """
    Hello everyone, welcome to this tutorial on Python programming.
    Today we're going to learn about functions and how to create them.
    Let me show you a simple example:

    def greet(name):
        return f"Hello, {name}!"

    This function takes a name parameter and returns a greeting.
    We can call it like this: greet("World") which returns "Hello, World!"
    """


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing."""
    import soundfile as sf
    import numpy as np

    # Create a simple sine wave
    sample_rate = 44100
    duration = 1  # 1 second
    frequency = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

    audio_file = temp_dir / "test_audio.wav"
    sf.write(str(audio_file), audio_data, sample_rate)

    return audio_file


@pytest.fixture
def mock_video_writer():
    """Mock OpenCV VideoWriter for testing."""
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True
    mock_writer.write = MagicMock()
    mock_writer.release = MagicMock()
    return mock_writer


@pytest.fixture
def mock_audio_stream():
    """Mock sounddevice InputStream for testing."""
    mock_stream = MagicMock()
    mock_stream.start = MagicMock()
    mock_stream.stop = MagicMock()
    mock_stream.close = MagicMock()
    return mock_stream