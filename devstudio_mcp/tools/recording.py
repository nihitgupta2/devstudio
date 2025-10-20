"""
DevStudio MCP Recording Tools - Screen Recording, Audio Capture, Terminal Monitoring
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

Implements screen recording, audio capture, and terminal monitoring
with production-grade error handling and Windows compatibility.
"""

import asyncio
import datetime
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import av
import mss
import numpy as np
import sounddevice as sd
import soundfile as sf
from fastmcp import FastMCP
from PIL import Image
from pydantic import BaseModel, Field
from screeninfo import get_monitors

from ..config import Settings
from ..utils.exceptions import RecordingError, ValidationError, handle_mcp_error
from ..utils.logger import setup_logger

logger = setup_logger()


class RecordingSession(BaseModel):
    """Model for recording session data."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime.datetime = Field(default_factory=datetime.datetime.now)
    status: str = Field(default="inactive")
    output_dir: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()) / "devstudio_recordings")
    screen_recording: bool = Field(default=False)
    audio_recording: bool = Field(default=False)
    screen_file: Optional[Path] = None
    audio_file: Optional[Path] = None
    terminal_log: Optional[Path] = None
    auto_mux: bool = Field(default=True)
    cleanup_source_files: bool = Field(default=False)
    muxed_file: Optional[Path] = None


class RecordingManager:
    """Manages recording sessions and operations."""

    def __init__(self, settings: Settings):
        """Initialize recording manager with configuration settings."""
        self.settings = settings
        self.active_sessions: Dict[str, RecordingSession] = {}
        self.logger = logger

        # Use configured output directory from settings
        self.output_dir = Path(settings.recording.output_dir)
        self.logger.info(f"RecordingManager initialized with output directory: {self.output_dir}")

        # Create recordings directory if it doesn't exist
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Recording output directory created/verified: {self.output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            raise RecordingError(f"Cannot create output directory: {e}", operation="initialization")

    async def start_recording_session(
        self,
        include_screen: bool = True,
        include_audio: bool = True,
        include_terminal: bool = False,
        output_format: str = "mp4",
        screen_id: Optional[int] = None,
        auto_mux: bool = True,
        cleanup_source_files: bool = False
    ) -> RecordingSession:
        """Start a new recording session."""
        try:
            session = RecordingSession(
                output_dir=self.output_dir,
                screen_recording=include_screen,
                audio_recording=include_audio,
                auto_mux=auto_mux,
                cleanup_source_files=cleanup_source_files
            )

            # Validate format
            if output_format not in ["mp4", "avi", "mov"]:
                raise ValidationError("Invalid output format", field="output_format")

            session.status = "starting"
            self.active_sessions[session.id] = session

            # Start screen recording if requested
            if include_screen:
                session.screen_file = await self._start_screen_recording(session, output_format, screen_id)

            # Start audio recording if requested
            if include_audio:
                session.audio_file = await self._start_audio_recording(session)

            # Start terminal monitoring if requested
            if include_terminal:
                session.terminal_log = await self._start_terminal_monitoring(session)

            session.status = "recording"
            self.logger.info(f"Started recording session {session.id}")

            return session

        except ValidationError:
            # Re-raise validation errors without wrapping
            raise
        except Exception as e:
            if session.id in self.active_sessions:
                del self.active_sessions[session.id]
            raise RecordingError(f"Failed to start recording session: {e}", operation="start_session")

    async def _start_screen_recording(self, session: RecordingSession, format: str, screen_id: Optional[int] = None) -> Path:
        """Start screen recording using PyAV with bundled FFmpeg."""
        try:
            # Get screen dimensions and position
            screen_region = None
            if screen_id is not None:
                # Use specific screen
                screens = self.get_available_screens()
                if screen_id >= len(screens):
                    raise RecordingError(f"Invalid screen_id {screen_id}", operation="screen_recording")
                screen = screens[screen_id]
                screen_width = screen['width']
                screen_height = screen['height']
                # Store region for screenshot capture (x, y, width, height)
                screen_region = (screen['x'], screen['y'], screen['width'], screen['height'])
            else:
                # Use primary screen
                import pyautogui  # Lazy import to avoid display requirement on headless systems
                screen_width, screen_height = pyautogui.size()
                screen_region = None  # None = full screen (primary)

            output_path = session.output_dir / f"screen_{session.id}.{format}"

            # Create PyAV output container
            container = av.open(str(output_path), 'w')

            # Add video stream with H.264 codec (industry standard)
            stream = container.add_stream('h264', rate=30)
            stream.width = screen_width
            stream.height = screen_height
            stream.pix_fmt = 'yuv420p'  # Universal compatibility

            # Set encoding options for quality
            stream.options = {
                'preset': 'fast',  # Balance between speed and compression
                'crf': '23',       # Constant Rate Factor (18-28, lower = better quality)
            }

            # Store recording objects in session for cleanup
            if not hasattr(session, '_recording_objects'):
                session._recording_objects = {}

            session._recording_objects['av_container'] = container
            session._recording_objects['av_stream'] = stream
            session._recording_objects['recording_active'] = True
            session._recording_objects['screen_region'] = screen_region
            session._recording_objects['screen_id'] = screen_id  # Store for MSS monitor selection

            # Start recording in background thread
            recording_task = asyncio.create_task(
                self._record_screen_frames(session, container, stream, screen_width, screen_height, screen_region)
            )
            session._recording_objects['recording_task'] = recording_task

            self.logger.info(f"Screen recording started for {screen_width}x{screen_height} (region={screen_region}) to {output_path}")

            return output_path

        except Exception as e:
            raise RecordingError(f"Failed to start screen recording: {e}", operation="screen_recording")

    async def _record_screen_frames(self, session: RecordingSession, container, stream, width, height, screen_region=None):
        """
        Record screen frames using PyAV with timestamp-based frame duration.

        This implements the OBS/Loom approach: capture frames as fast as possible,
        then use PTS (Presentation Timestamp) to make them play smoothly at 30 FPS.
        Slow captures automatically result in longer frame display times.
        """
        try:
            import time

            target_fps = 30
            frame_count = 0
            start_time = time.time()

            # Determine monitor index for MSS
            # MSS uses 1-indexed monitors (0 = all monitors combined)
            monitor_index = None
            if hasattr(session, '_recording_objects') and 'screen_id' in session._recording_objects:
                # screen_id is 0-indexed, MSS monitors are 1-indexed
                monitor_index = session._recording_objects['screen_id'] + 1

            self.logger.info(f"Starting CFR recording at {target_fps} FPS using PTS-based frame timing")

            with mss.mss() as sct:
                while (hasattr(session, '_recording_objects') and
                       session._recording_objects.get('recording_active', False)):

                    capture_start = time.time()

                    # Capture screenshot using MSS for proper multi-monitor support
                    if monitor_index is not None:
                        # Capture specific monitor
                        screenshot_data = sct.grab(sct.monitors[monitor_index])
                    else:
                        # Capture primary monitor (monitor 1 in MSS)
                        screenshot_data = sct.grab(sct.monitors[1])

                    # Convert MSS screenshot to PIL Image, then to numpy array
                    screenshot = Image.frombytes('RGB', screenshot_data.size, screenshot_data.bgra, 'raw', 'BGRX')
                    frame_array = np.array(screenshot)

                    # Create PyAV VideoFrame from numpy array
                    frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')

                    # CRITICAL: Set frame PTS based on actual elapsed time
                    # This is how OBS/Loom achieve smooth playback with variable capture rates
                    #
                    # stream.time_base is 1/30 for 30fps stream
                    # PTS units = elapsed_seconds / time_base
                    #           = elapsed_seconds / (1/30)
                    #           = elapsed_seconds * 30
                    #
                    # Example: Frame captured at 0.090s gets PTS=2.7
                    # Video player will display this frame until next frame arrives
                    # Result: Frame shown for ~3 frames at 30fps = smooth playback
                    elapsed = capture_start - start_time
                    frame.pts = int(elapsed * target_fps)

                    # Encode frame to video stream
                    for packet in stream.encode(frame):
                        container.mux(packet)

                    frame_count += 1

                    # Log progress every 2 seconds with actual vs target FPS
                    if frame_count % 60 == 0:
                        elapsed_total = time.time() - start_time
                        actual_capture_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
                        self.logger.info(
                            f"Recording: {frame_count} frames in {elapsed_total:.1f}s | "
                            f"Capture FPS: {actual_capture_fps:.1f} | "
                            f"Output: {target_fps} FPS CFR"
                        )

                    # Small yield to prevent blocking event loop
                    # Don't try to "maintain FPS" - capture as fast as possible
                    # PTS handles the timing for playback
                    await asyncio.sleep(0)

        except Exception as e:
            self.logger.error(f"Screen recording error: {e}")
        finally:
            # CRITICAL: Flush remaining frames and close container properly
            try:
                self.logger.info(f"Finalizing video (wrote {frame_count} frames, {frame_count / 30:.1f}s)")

                # Flush any remaining packets
                if stream:
                    for packet in stream.encode():
                        container.mux(packet)

                # Close container to write file footer
                if container:
                    container.close()
                    self.logger.info("Video container closed successfully")

            except Exception as finalize_error:
                self.logger.error(f"Failed to finalize video: {finalize_error}")

            self.logger.info("Screen recording thread completed")

    async def _start_audio_recording(self, session: RecordingSession) -> Path:
        """Start audio recording using sounddevice with background thread."""
        try:
            output_path = session.output_dir / f"audio_{session.id}.wav"

            # Configure audio recording
            sample_rate = 44100
            channels = 2
            chunk_size = 1024

            # Initialize recording objects if not exists
            if not hasattr(session, '_recording_objects'):
                session._recording_objects = {}

            # Set up audio recording state
            session._recording_objects['audio_data'] = []
            session._recording_objects['audio_sample_rate'] = sample_rate
            session._recording_objects['audio_channels'] = channels

            # Start audio recording task
            audio_task = asyncio.create_task(
                self._record_audio_stream(session, sample_rate, channels, chunk_size, output_path)
            )
            session._recording_objects['audio_task'] = audio_task

            self.logger.info(f"Audio recording started for {sample_rate}Hz, {channels} channels to {output_path}")

            return output_path

        except Exception as e:
            raise RecordingError(f"Failed to start audio recording: {e}", operation="audio_recording")

    async def _record_audio_stream(self, session: RecordingSession, sample_rate: int, channels: int, chunk_size: int, output_path: Path):
        """Record audio stream in background thread."""
        try:
            # Create audio stream
            audio_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                blocksize=chunk_size,
                callback=lambda indata, frames, time, status: self._audio_callback(session, indata, frames, time, status)
            )

            session._recording_objects['audio_stream'] = audio_stream

            # Start the audio stream
            audio_stream.start()

            # Keep recording while active
            while (hasattr(session, '_recording_objects') and
                   session._recording_objects.get('recording_active', False)):
                await asyncio.sleep(0.1)  # Check every 100ms

        except Exception as e:
            self.logger.error(f"Audio recording error: {e}")
        finally:
            # Save recorded audio data
            if hasattr(session, '_recording_objects') and session._recording_objects.get('audio_data'):
                try:
                    # Concatenate all audio chunks
                    audio_data = np.concatenate(session._recording_objects['audio_data'], axis=0)

                    # Save to WAV file
                    sf.write(
                        str(output_path),
                        audio_data,
                        session._recording_objects['audio_sample_rate']
                    )
                    self.logger.info(f"Audio saved to {output_path}")
                except Exception as save_error:
                    self.logger.error(f"Failed to save audio: {save_error}")

            # Clean up stream
            if 'audio_stream' in session._recording_objects:
                try:
                    session._recording_objects['audio_stream'].stop()
                    session._recording_objects['audio_stream'].close()
                except:
                    pass

            self.logger.info("Audio recording thread completed")

    def _audio_callback(self, session: RecordingSession, indata, frames, time, status):
        """Callback function for audio stream data."""
        if status:
            self.logger.warning(f"Audio callback status: {status}")

        # Store audio data if recording is active
        if (hasattr(session, '_recording_objects') and
            session._recording_objects.get('recording_active', False)):
            session._recording_objects['audio_data'].append(indata.copy())

    async def _start_terminal_monitoring(self, session: RecordingSession) -> Path:
        """Start terminal command monitoring with background thread."""
        try:
            output_path = session.output_dir / f"terminal_{session.id}.log"

            # Initialize recording objects if not exists
            if not hasattr(session, '_recording_objects'):
                session._recording_objects = {}

            # Create log file and setup monitoring
            output_path.touch()
            session._recording_objects['terminal_log'] = output_path

            # Start terminal monitoring task
            terminal_task = asyncio.create_task(
                self._monitor_terminal_activity(session, output_path)
            )
            session._recording_objects['terminal_task'] = terminal_task

            self.logger.info(f"Terminal monitoring started to {output_path}")

            return output_path

        except Exception as e:
            raise RecordingError(f"Failed to start terminal monitoring: {e}", operation="terminal_monitoring")

    async def _monitor_terminal_activity(self, session: RecordingSession, output_path: Path):
        """Monitor terminal activity in background thread."""
        try:
            import psutil
            import subprocess
            from datetime import datetime

            # Get current process and its children
            current_process = psutil.Process()

            with open(output_path, 'w') as log_file:
                log_file.write(f"Terminal monitoring started at {datetime.now().isoformat()}\n")
                log_file.write(f"Process ID: {current_process.pid}\n")
                log_file.write("-" * 50 + "\n")

                last_processes = set()

                while (hasattr(session, '_recording_objects') and
                       session._recording_objects.get('recording_active', False)):

                    try:
                        # Get all running processes
                        current_processes = set()

                        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                            try:
                                proc_info = proc.info
                                if proc_info['cmdline']:
                                    cmd_line = ' '.join(proc_info['cmdline'])
                                    process_signature = f"{proc_info['name']}:{cmd_line}"
                                    current_processes.add((proc_info['pid'], process_signature, proc_info['create_time']))
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue

                        # Find new processes
                        new_processes = current_processes - last_processes

                        for pid, signature, create_time in new_processes:
                            # Filter for terminal-related processes
                            process_name = signature.split(':')[0].lower()
                            if any(term in process_name for term in ['python', 'node', 'npm', 'pip', 'git', 'code', 'bash', 'cmd', 'powershell']):
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                log_entry = f"[{timestamp}] PID:{pid} - {signature}\n"
                                log_file.write(log_entry)
                                log_file.flush()

                        last_processes = current_processes

                    except Exception as monitor_error:
                        self.logger.warning(f"Terminal monitoring iteration error: {monitor_error}")

                    # Check every 2 seconds
                    await asyncio.sleep(2.0)

        except Exception as e:
            self.logger.error(f"Terminal monitoring error: {e}")
        finally:
            try:
                with open(output_path, 'a') as log_file:
                    log_file.write(f"\nTerminal monitoring ended at {datetime.now().isoformat()}\n")
            except:
                pass
            self.logger.info("Terminal monitoring thread completed")

    async def stop_recording_session(self, session_id: str) -> Dict[str, Any]:
        """Stop a recording session and return file paths."""
        try:
            if session_id not in self.active_sessions:
                raise ValidationError(f"Recording session {session_id} not found", field="session_id")

            session = self.active_sessions[session_id]
            session.status = "stopping"
            self.logger.info(f"Stopping recording session {session_id}")

            # Stop background recording threads
            if hasattr(session, '_recording_objects'):
                # Signal all recordings to stop
                session._recording_objects['recording_active'] = False
                self.logger.info("Signaled recording threads to stop")

                # Wait for screen recording task to complete
                if 'recording_task' in session._recording_objects:
                    try:
                        self.logger.info("Waiting for screen recording task to complete...")
                        await asyncio.wait_for(session._recording_objects['recording_task'], timeout=10.0)
                        self.logger.info("Screen recording task completed")
                    except asyncio.TimeoutError:
                        self.logger.error(f"Screen recording task timeout for session {session_id}")
                    except Exception as task_error:
                        self.logger.error(f"Screen recording task error: {task_error}")

                # Wait for audio recording task to complete
                if 'audio_task' in session._recording_objects:
                    try:
                        self.logger.info("Waiting for audio recording task to complete...")
                        await asyncio.wait_for(session._recording_objects['audio_task'], timeout=10.0)
                        self.logger.info("Audio recording task completed")
                    except asyncio.TimeoutError:
                        self.logger.error(f"Audio recording task timeout for session {session_id}")
                    except Exception as task_error:
                        self.logger.error(f"Audio recording task error: {task_error}")

                # Wait for terminal monitoring task to complete
                if 'terminal_task' in session._recording_objects:
                    try:
                        self.logger.info("Waiting for terminal monitoring task to complete...")
                        await asyncio.wait_for(session._recording_objects['terminal_task'], timeout=10.0)
                        self.logger.info("Terminal monitoring task completed")
                    except asyncio.TimeoutError:
                        self.logger.error(f"Terminal monitoring task timeout for session {session_id}")
                    except Exception as task_error:
                        self.logger.error(f"Terminal monitoring task error: {task_error}")

                # Clean up recording objects (defensive - should be done in finally blocks)
                if 'av_container' in session._recording_objects:
                    try:
                        self.logger.info("Closing video container (cleanup)")
                        session._recording_objects['av_container'].close()
                        self.logger.info("Video container closed in cleanup")
                    except Exception as close_error:
                        self.logger.error(f"Error closing video container in cleanup: {close_error}")

                if 'audio_stream' in session._recording_objects:
                    try:
                        self.logger.info("Stopping audio stream (cleanup)")
                        session._recording_objects['audio_stream'].stop()
                        self.logger.info("Audio stream stopped in cleanup")
                    except Exception as stream_error:
                        self.logger.error(f"Error stopping audio stream in cleanup: {stream_error}")

            # Compile results
            results = {
                "session_id": session_id,
                "files": {},
                "duration": (datetime.datetime.now() - session.start_time).total_seconds()
            }

            if session.screen_file:
                results["files"]["screen"] = str(session.screen_file)
                self.logger.info(f"Screen recording saved to: {session.screen_file}")

            if session.audio_file:
                results["files"]["audio"] = str(session.audio_file)
                self.logger.info(f"Audio recording saved to: {session.audio_file}")

            if session.terminal_log:
                results["files"]["terminal"] = str(session.terminal_log)
                self.logger.info(f"Terminal log saved to: {session.terminal_log}")

            # Auto-mux audio and video if both exist and auto_mux is enabled
            if (session.auto_mux and
                session.screen_file and
                session.audio_file and
                session.screen_file.exists() and
                session.audio_file.exists()):

                self.logger.info("Auto-muxing audio and video...")

                # Generate muxed file path
                muxed_path = session.output_dir / f"combined_{session.id}.mp4"

                # Perform muxing
                try:
                    muxed_result = self.mux_audio_video(
                        session.screen_file,
                        session.audio_file,
                        muxed_path
                    )

                    results["files"]["combined"] = str(muxed_result)
                    self.logger.info(f"Auto-muxed file created: {muxed_result}")

                    # Cleanup source files if requested
                    if session.cleanup_source_files:
                        self.logger.info("Cleaning up source files...")
                        try:
                            session.screen_file.unlink()
                            session.audio_file.unlink()
                            # Remove from results
                            del results["files"]["screen"]
                            del results["files"]["audio"]
                            self.logger.info("Source files cleaned up successfully")
                        except Exception as cleanup_error:
                            self.logger.error(f"Failed to cleanup source files: {cleanup_error}")
                            results["cleanup_error"] = str(cleanup_error)

                except Exception as mux_error:
                    self.logger.error(f"Auto-muxing failed: {mux_error}")
                    # Don't fail the entire operation, just skip muxing
                    results["mux_error"] = str(mux_error)

            session.status = "completed"
            del self.active_sessions[session_id]

            self.logger.info(f"Successfully stopped recording session {session_id}")

            return results

        except ValidationError:
            raise  # Re-raise validation errors as-is
        except Exception as e:
            self.logger.error(f"Failed to stop recording session {session_id}: {e}", exc_info=True)
            raise RecordingError(f"Failed to stop recording session: {e}", operation="stop_session")

    def get_available_screens(self) -> List[Dict[str, Any]]:
        """Get information about all available screens/monitors."""
        try:
            monitors = get_monitors()
            screens = []

            for i, monitor in enumerate(monitors):
                screen_info = {
                    "id": i,
                    "name": f"Screen {i + 1}",
                    "x": monitor.x,
                    "y": monitor.y,
                    "width": monitor.width,
                    "height": monitor.height,
                    "is_primary": getattr(monitor, 'is_primary', i == 0),
                    "scale": getattr(monitor, 'scale', 1.0)
                }
                screens.append(screen_info)

            self.logger.info(f"Detected {len(screens)} screens")
            return screens

        except Exception as e:
            self.logger.error(f"Failed to detect screens: {e}")
            # Fallback to pyautogui size if screeninfo fails
            try:
                import pyautogui  # Lazy import to avoid display requirement on headless systems
                width, height = pyautogui.size()
                return [{
                    "id": 0,
                    "name": "Primary Screen",
                    "x": 0,
                    "y": 0,
                    "width": width,
                    "height": height,
                    "is_primary": True,
                    "scale": 1.0
                }]
            except Exception as fallback_error:
                self.logger.error(f"Fallback screen detection failed: {fallback_error}")
                raise RecordingError(f"Cannot detect screens: {e}", operation="screen_detection")

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active recording sessions."""
        return [
            {
                "id": session.id,
                "start_time": session.start_time.isoformat(),
                "status": session.status,
                "screen_recording": session.screen_recording,
                "audio_recording": session.audio_recording
            }
            for session in self.active_sessions.values()
        ]

    def mux_audio_video(self, video_path: Path, audio_path: Path, output_path: Path) -> Path:
        """
        Mux separate audio and video files into a single MP4 using PyAV.

        Uses decode + re-encode approach (like combine_av.py) to handle
        format incompatibilities between H.264/MP4 video and WAV audio.

        Args:
            video_path: Path to video file (MP4)
            audio_path: Path to audio file (WAV)
            output_path: Path for output MP4

        Returns:
            Path to the muxed output file
        """
        try:
            self.logger.info(f"Muxing audio and video: {video_path} + {audio_path} -> {output_path}")

            # Open input containers
            video_container = av.open(str(video_path))
            audio_container = av.open(str(audio_path))

            # Get stream information
            video_stream_in = video_container.streams.video[0]
            audio_stream_in = audio_container.streams.audio[0]

            self.logger.info(f"Video: {video_stream_in.width}x{video_stream_in.height} @ {video_stream_in.average_rate}fps")
            self.logger.info(f"Audio: {audio_stream_in.rate}Hz, {audio_stream_in.channels} channels")

            # Create output container (MP4 format)
            output_container = av.open(str(output_path), 'w', format='mp4')

            # Add video stream (H.264 encoder)
            output_video = output_container.add_stream('libx264', rate=video_stream_in.average_rate)
            output_video.width = video_stream_in.width
            output_video.height = video_stream_in.height
            output_video.pix_fmt = video_stream_in.pix_fmt

            # Add audio stream (AAC encoder for MP4 compatibility)
            output_audio = output_container.add_stream('aac', rate=audio_stream_in.rate)

            video_frames = 0
            audio_frames = 0

            # Process video frames (decode + re-encode)
            for frame in video_container.decode(video=0):
                video_frames += 1
                if video_frames % 30 == 0:
                    self.logger.debug(f"Muxing: {video_frames} video frames processed")

                # Encode and mux video frame
                for packet in output_video.encode(frame):
                    if packet:
                        output_container.mux(packet)

            # Process audio frames (decode + re-encode WAV → AAC)
            for frame in audio_container.decode(audio=0):
                audio_frames += 1

                # Encode and mux audio frame
                for packet in output_audio.encode(frame):
                    if packet:
                        output_container.mux(packet)

            # Flush remaining packets
            self.logger.info("Flushing remaining packets...")
            for packet in output_video.encode(None):
                if packet:
                    output_container.mux(packet)

            for packet in output_audio.encode(None):
                if packet:
                    output_container.mux(packet)

            # Close all containers
            output_container.close()
            video_container.close()
            audio_container.close()

            file_size = output_path.stat().st_size
            self.logger.info(f"Muxing complete: {video_frames} video frames, {audio_frames} audio frames")
            self.logger.info(f"Output: {output_path} ({file_size / (1024*1024):.2f} MB)")

            return output_path

        except Exception as e:
            self.logger.error(f"Failed to mux audio and video: {e}", exc_info=True)
            raise RecordingError(f"Audio/video muxing failed: {e}", operation="muxing")


# Initialize global recording manager
recording_manager = None


def get_tools(settings: Settings) -> Dict[str, Any]:
    """Get recording tools for MCP registration."""
    global recording_manager
    recording_manager = RecordingManager(settings)

    @handle_mcp_error
    async def start_recording(
        include_screen: Annotated[bool, "Whether to record screen activity"] = True,
        include_audio: Annotated[bool, "Whether to record audio"] = True,
        include_terminal: Annotated[bool, "Whether to monitor terminal commands"] = False,
        output_format: Annotated[str, "Video output format (mp4, avi, mov)"] = "mp4",
        screen_id: Annotated[Optional[int], "Specific screen to record (None for primary)"] = None,
        auto_mux: Annotated[bool, "Automatically combine audio+video into single MP4"] = True,
        cleanup_source_files: Annotated[bool, "Delete source files after muxing"] = False
    ) -> Dict[str, Any]:
        """
        Start a new recording session with screen, audio, and/or terminal capture.

        Args:
            include_screen: Whether to record screen activity
            include_audio: Whether to record audio
            include_terminal: Whether to monitor terminal commands
            output_format: Video output format (mp4, avi, mov)
            screen_id: Specific screen to record (None for primary/all screens)
            auto_mux: Automatically combine audio+video into single MP4 (default: True)
            cleanup_source_files: Delete source files after muxing (default: False)

        Returns:
            Recording session information with session ID and status
        """
        # Validate screen_id if specified
        if screen_id is not None:
            available_screens = recording_manager.get_available_screens()
            if screen_id < 0 or screen_id >= len(available_screens):
                raise ValidationError(f"Invalid screen_id {screen_id}. Available screens: 0-{len(available_screens)-1}", field="screen_id")

        session = await recording_manager.start_recording_session(
            include_screen=include_screen,
            include_audio=include_audio,
            include_terminal=include_terminal,
            output_format=output_format,
            screen_id=screen_id,
            auto_mux=auto_mux,
            cleanup_source_files=cleanup_source_files
        )

        return {
            "session_id": session.id,
            "status": session.status,
            "start_time": session.start_time.isoformat(),
            "output_directory": str(session.output_dir),
            "recording_types": {
                "screen": session.screen_recording,
                "audio": session.audio_recording,
                "terminal": bool(session.terminal_log)
            },
            "screen_id": screen_id
        }

    @handle_mcp_error
    async def stop_recording(
        session_id: Annotated[str, "ID of the recording session to stop"]
    ) -> Dict[str, Any]:
        """
        Stop an active recording session and get output file paths.

        Args:
            session_id: ID of the recording session to stop

        Returns:
            Session results with file paths and duration
        """
        return await recording_manager.stop_recording_session(session_id)

    @handle_mcp_error
    async def capture_screen(
        screen_id: Annotated[Optional[int], "Specific screen to capture (None for primary)"] = None
    ) -> Dict[str, Any]:
        """
        Take a single screenshot of the current screen.

        Args:
            screen_id: Specific screen to capture (None for primary screen)

        Returns:
            Screenshot file path and metadata
        """
        try:
            # Ensure output directory exists and is writable
            recording_manager.output_dir.mkdir(parents=True, exist_ok=True)

            # Verify directory is writable by attempting to create a test file
            test_file = recording_manager.output_dir / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()  # Clean up test file
            except Exception as write_error:
                raise RecordingError(
                    f"Output directory is not writable: {recording_manager.output_dir}. Error: {write_error}",
                    operation="screenshot"
                )

            # Validate screen_id if specified
            if screen_id is not None:
                screens = recording_manager.get_available_screens()
                if screen_id < 0 or screen_id >= len(screens):
                    raise ValidationError(f"Invalid screen_id {screen_id}. Available screens: 0-{len(screens)-1}", field="screen_id")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = recording_manager.output_dir / f"screenshot_{timestamp}.png"

            # Take screenshot using MSS for proper multi-monitor support
            try:
                with mss.mss() as sct:
                    if screen_id is not None:
                        # Capture specific monitor (MSS uses 1-indexed monitors)
                        monitor_index = screen_id + 1
                        screenshot_data = sct.grab(sct.monitors[monitor_index])
                    else:
                        # Capture primary monitor (monitor 1 in MSS)
                        screenshot_data = sct.grab(sct.monitors[1])

                    # Convert MSS screenshot to PIL Image
                    screenshot = Image.frombytes('RGB', screenshot_data.size, screenshot_data.bgra, 'raw', 'BGRX')

                logger.info(f"Screenshot captured successfully, size: {screenshot.size}")
            except Exception as screenshot_error:
                raise RecordingError(
                    f"Failed to capture screenshot: {screenshot_error}. "
                    "Please ensure the display is accessible and pyautogui is properly installed.",
                    operation="screenshot"
                )

            # Save screenshot with error handling
            try:
                screenshot.save(output_path)
                logger.info(f"Screenshot saved successfully to {output_path}")
            except Exception as save_error:
                raise RecordingError(
                    f"Failed to save screenshot to {output_path}: {save_error}",
                    operation="screenshot"
                )

            # Verify file was created and get file info
            if not output_path.exists():
                raise RecordingError(
                    f"Screenshot file was not created at {output_path}",
                    operation="screenshot"
                )

            file_size = output_path.stat().st_size
            logger.info(f"Screenshot file verified: {output_path} ({file_size} bytes)")

            return {
                "file_path": str(output_path),
                "timestamp": timestamp,
                "dimensions": screenshot.size,
                "format": "png",
                "size_bytes": file_size,
                "output_directory": str(recording_manager.output_dir)
            }

        except RecordingError:
            raise  # Re-raise RecordingError as-is
        except Exception as e:
            raise RecordingError(f"Unexpected error during screenshot capture: {e}", operation="screenshot")

    @handle_mcp_error
    async def list_active_sessions() -> Dict[str, Any]:
        """
        List all active recording sessions.

        Returns:
            List of active recording sessions with their status
        """
        return {
            "active_sessions": recording_manager.get_active_sessions(),
            "total_count": len(recording_manager.active_sessions)
        }

    @handle_mcp_error
    async def mux_audio_video_files(
        video_path: Annotated[str, "Path to video file (MP4)"],
        audio_path: Annotated[str, "Path to audio file (WAV)"],
        output_path: Annotated[str, "Path for combined output file"]
    ) -> Dict[str, Any]:
        """
        Combine separate audio and video files into a single MP4 file.

        Args:
            video_path: Path to video file (MP4)
            audio_path: Path to audio file (WAV)
            output_path: Path for combined output file

        Returns:
            Information about the muxed file
        """
        try:
            video_p = Path(video_path)
            audio_p = Path(audio_path)
            output_p = Path(output_path)

            # Validate input files exist
            if not video_p.exists():
                raise ValidationError(f"Video file not found: {video_path}", field="video_path")
            if not audio_p.exists():
                raise ValidationError(f"Audio file not found: {audio_path}", field="audio_path")

            # Perform muxing
            result_path = recording_manager.mux_audio_video(video_p, audio_p, output_p)

            return {
                "output_file": str(result_path),
                "video_input": str(video_p),
                "audio_input": str(audio_p),
                "size_bytes": result_path.stat().st_size
            }

        except (ValidationError, RecordingError):
            raise
        except Exception as e:
            raise RecordingError(f"Failed to mux files: {e}", operation="muxing")

    @handle_mcp_error
    async def get_available_screens() -> Dict[str, Any]:
        """
        Get information about all available screens/monitors.

        Returns:
            List of available screens with their properties
        """
        return {
            "screens": recording_manager.get_available_screens(),
            "total_count": len(recording_manager.get_available_screens())
        }

    return {
        "start_recording": start_recording,
        "stop_recording": stop_recording,
        "capture_screen": capture_screen,
        "list_active_sessions": list_active_sessions,
        "mux_audio_video": mux_audio_video_files,
        "get_available_screens": get_available_screens
    }
