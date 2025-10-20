"""
Unit tests for recording functionality.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from devstudio_mcp.tools.recording import RecordingManager, RecordingSession
from devstudio_mcp.utils.exceptions import RecordingError, ValidationError


class TestRecordingManager:
    """Test cases for RecordingManager."""

    def test_initialization(self, recording_manager):
        """Test RecordingManager initialization."""
        assert recording_manager.active_sessions == {}
        assert recording_manager.output_dir.exists()

    @pytest.mark.asyncio
    async def test_start_recording_session_screen_only(self, recording_manager):
        """Test starting a screen-only recording session."""
        with patch('cv2.VideoWriter') as mock_writer_class:
            mock_writer = MagicMock()
            mock_writer.isOpened.return_value = True
            mock_writer_class.return_value = mock_writer

            with patch('pyautogui.size', return_value=(1920, 1080)):
                session = await recording_manager.start_recording_session(
                    include_screen=True,
                    include_audio=False,
                    include_terminal=False
                )

                assert session.id in recording_manager.active_sessions
                assert session.status == "recording"
                assert session.screen_recording is True
                assert session.audio_recording is False
                assert session.screen_file is not None

    @pytest.mark.asyncio
    async def test_start_recording_session_audio_only(self, recording_manager):
        """Test starting an audio-only recording session."""
        with patch('sounddevice.InputStream') as mock_stream_class:
            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream

            session = await recording_manager.start_recording_session(
                include_screen=False,
                include_audio=True,
                include_terminal=False
            )

            assert session.id in recording_manager.active_sessions
            assert session.status == "recording"
            assert session.screen_recording is False
            assert session.audio_recording is True
            assert session.audio_file is not None

    @pytest.mark.asyncio
    async def test_start_recording_session_full(self, recording_manager):
        """Test starting a full recording session with all features."""
        with patch('cv2.VideoWriter') as mock_writer_class, \
             patch('sounddevice.InputStream') as mock_stream_class, \
             patch('pyautogui.size', return_value=(1920, 1080)):

            mock_writer = MagicMock()
            mock_writer.isOpened.return_value = True
            mock_writer_class.return_value = mock_writer

            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream

            session = await recording_manager.start_recording_session(
                include_screen=True,
                include_audio=True,
                include_terminal=True
            )

            assert session.id in recording_manager.active_sessions
            assert session.status == "recording"
            assert session.screen_recording is True
            assert session.audio_recording is True
            assert session.screen_file is not None
            assert session.audio_file is not None
            assert session.terminal_log is not None

    @pytest.mark.asyncio
    async def test_invalid_output_format(self, recording_manager):
        """Test that invalid output format raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            await recording_manager.start_recording_session(
                output_format="invalid"
            )
        assert "Invalid output format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stop_recording_session(self, recording_manager):
        """Test stopping a recording session."""
        # Start a session first
        with patch('cv2.VideoWriter') as mock_writer_class, \
             patch('pyautogui.size', return_value=(1920, 1080)):

            mock_writer = MagicMock()
            mock_writer.isOpened.return_value = True
            mock_writer_class.return_value = mock_writer

            session = await recording_manager.start_recording_session(
                include_screen=True,
                include_audio=False
            )

            # Stop the session
            results = await recording_manager.stop_recording_session(session.id)

            assert results["session_id"] == session.id
            assert "files" in results
            assert "duration" in results
            assert results["duration"] > 0
            assert session.id not in recording_manager.active_sessions

    @pytest.mark.asyncio
    async def test_stop_nonexistent_session(self, recording_manager):
        """Test stopping a non-existent session raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            await recording_manager.stop_recording_session("nonexistent")
        assert "Recording session nonexistent not found" in str(exc_info.value)

    def test_get_active_sessions_empty(self, recording_manager):
        """Test getting active sessions when none exist."""
        sessions = recording_manager.get_active_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_get_active_sessions_with_data(self, recording_manager):
        """Test getting active sessions with data."""
        with patch('cv2.VideoWriter') as mock_writer_class, \
             patch('pyautogui.size', return_value=(1920, 1080)):

            mock_writer = MagicMock()
            mock_writer.isOpened.return_value = True
            mock_writer_class.return_value = mock_writer

            session = await recording_manager.start_recording_session()
            sessions = recording_manager.get_active_sessions()

            assert len(sessions) == 1
            assert sessions[0]["id"] == session.id
            assert sessions[0]["status"] == "recording"
            assert "start_time" in sessions[0]


class TestRecordingSession:
    """Test cases for RecordingSession model."""

    def test_recording_session_creation(self):
        """Test RecordingSession model creation."""
        session = RecordingSession()

        assert session.id is not None
        assert session.status == "inactive"
        assert session.screen_recording is False
        assert session.audio_recording is False
        assert session.screen_file is None
        assert session.audio_file is None
        assert session.terminal_log is None

    def test_recording_session_with_custom_values(self, temp_dir):
        """Test RecordingSession with custom values."""
        session = RecordingSession(
            status="recording",
            output_dir=temp_dir,
            screen_recording=True,
            audio_recording=True
        )

        assert session.status == "recording"
        assert session.output_dir == temp_dir
        assert session.screen_recording is True
        assert session.audio_recording is True