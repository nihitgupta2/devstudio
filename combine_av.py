#!/usr/bin/env python3
"""
Audio/Video Muxing Script using PyAV

Combines separate audio and video files into a single MP4 file.
Uses PyAV (Python FFmpeg bindings) for reliable muxing with proper synchronization.

Usage:
    python combine_av.py

Files:
    Input:  screen_*.mp4 and audio_*.wav in the recordings directory
    Output: combined_*.mp4 in the recordings directory
"""

import av
import os
import sys
from pathlib import Path


def find_recording_files(recordings_dir: str | None = None) -> tuple[Path, Path]:
    """Find the most recent screen and audio recording files."""
    if recordings_dir is None:
        recordings_dir = str(Path.home() / "devstudio" / "recordings")
    recordings_path = Path(recordings_dir)

    if not recordings_path.exists():
        raise FileNotFoundError(f"Recordings directory not found: {recordings_dir}")

    # Find video files
    video_files = list(recordings_path.glob("screen_*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No screen recording files found in {recordings_dir}")

    # Find audio files
    audio_files = list(recordings_path.glob("audio_*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No audio recording files found in {recordings_dir}")

    # Use the most recent files (assuming they have the same timestamp)
    video_file = max(video_files, key=lambda p: p.stat().st_mtime)
    audio_file = max(audio_files, key=lambda p: p.stat().st_mtime)

    print(f"Found video file: {video_file}")
    print(f"Found audio file: {audio_file}")

    return video_file, audio_file


def mux_audio_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """
    Combine audio and video files using PyAV.

    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        output_path: Path for combined output
    """
    print("Starting audio/video muxing process...")
    print(f"Video: {video_path}")
    print(f"Audio: {audio_path}")
    print(f"Output: {output_path}")

    try:
        # Open input containers
        print("Opening video file...")
        video_container = av.open(str(video_path))
        print("Opening audio file...")
        audio_container = av.open(str(audio_path))

        # Get stream information
        video_stream = video_container.streams.video[0]
        audio_stream = audio_container.streams.audio[0]

        print(f"Video info: {video_stream.width}x{video_stream.height} @ {video_stream.average_rate}fps")
        print(f"Audio info: {audio_stream.rate}Hz, {audio_stream.channels} channels")

        # Create output container
        print("Creating output MP4 container...")
        output_container = av.open(str(output_path), 'w', format='mp4')

        # Add video stream (copy existing codec)
        output_video = output_container.add_stream('libx264', rate=video_stream.average_rate)
        output_video.width = video_stream.width
        output_video.height = video_stream.height
        output_video.pix_fmt = video_stream.pix_fmt

        # Add audio stream (AAC for MP4 compatibility)
        output_audio = output_container.add_stream('aac', rate=audio_stream.rate)
        # Note: channels, sample_rate, and other properties are set automatically from input

        print("Muxing streams...")
        video_frames = 0
        audio_frames = 0

        # Process video frames
        for frame in video_container.decode(video=0):
            video_frames += 1
            if video_frames % 30 == 0:  # Progress indicator every second
                print(f"Processed {video_frames} video frames...")

            # Encode and mux video frame
            packets = output_video.encode(frame)
            for packet in packets:
                if packet:
                    output_container.mux(packet)

        # Process audio frames
        for frame in audio_container.decode(audio=0):
            audio_frames += 1
            if audio_frames % 100 == 0:  # Progress indicator
                print(f"Processed {audio_frames} audio frames...")

            # Encode and mux audio frame
            packets = output_audio.encode(frame)
            for packet in packets:
                if packet:
                    output_container.mux(packet)

        # Flush remaining packets
        print("Flushing remaining packets...")
        video_packets = output_video.encode(None)
        for packet in video_packets:
            if packet:
                output_container.mux(packet)

        audio_packets = output_audio.encode(None)
        for packet in audio_packets:
            if packet:
                output_container.mux(packet)

        # Close containers
        output_container.close()
        video_container.close()
        audio_container.close()

        # Verify output
        file_size = output_path.stat().st_size
        print("Muxing completed successfully!")
        print(f"Output file: {output_path}")
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        print(f"Processed {video_frames} video frames and {audio_frames} audio frames")

    except Exception as e:
        print(f"Error during muxing: {e}")
        raise


def main():
    """Main function to combine audio and video files."""
    try:
        # Find recording files
        video_file, audio_file = find_recording_files()

        # Create output filename based on input files
        video_stem = video_file.stem  # Gets filename without extension
        output_file = video_file.parent / f"combined_{video_stem}.mp4"

        # Perform muxing
        mux_audio_video(video_file, audio_file, output_file)

        print("\n" + "="*50)
        print("SUCCESS: Audio and video files combined successfully!")
        print(f"Output file: {output_file}")
        print("="*50)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
