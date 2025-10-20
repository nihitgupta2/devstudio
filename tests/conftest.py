"""
Pytest configuration and fixtures for DevStudio MCP tests.
"""

import asyncio
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from devstudio_mcp.config import Settings
from devstudio_mcp.tools.recording import RecordingManager


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
    return Settings(log_level="DEBUG")


@pytest.fixture
def recording_manager(test_settings):
    """Create a RecordingManager instance for testing."""
    return RecordingManager(test_settings)


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
