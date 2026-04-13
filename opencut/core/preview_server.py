"""
Mini Player / Preview Backend.

Extract preview frames, generate preview clips, and serve
frame-accurate thumbnails for the editing UI.
"""

import logging
import os
import subprocess
import tempfile
from typing import Callable, Dict, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def extract_preview_frame(
    video_path: str,
    timestamp: float,
    output_path: Optional[str] = None,
    width: int = 0,
    height: int = 0,
    on_progress: Optional[Callable] = None,
) -> str:
    """Extract a single frame from a video at a given timestamp.

    Args:
        video_path: Path to the source video.
        timestamp: Time position in seconds.
        output_path: Output image path (auto-generated if None).
        width: Optional resize width (0 = original).
        height: Optional resize height (0 = original).

    Returns:
        Path to the extracted frame image.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if timestamp < 0:
        raise ValueError("Timestamp must be non-negative")

    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"preview_frame_{os.getpid()}_{int(timestamp*1000)}.jpg"
        )

    if on_progress:
        on_progress(20, f"Seeking to {timestamp:.2f}s")

    builder = (
        FFmpegCmd()
        .pre_input("-ss", str(timestamp))
        .input(video_path)
    )

    if width > 0 or height > 0:
        w = width if width > 0 else -1
        h = height if height > 0 else -1
        builder = builder.video_filter(f"scale={w}:{h}")

    builder = builder.frames(1).option("q:v", "2")

    cmd = builder.output(output_path).build()
    run_ffmpeg(cmd, timeout=15)

    if on_progress:
        on_progress(100, "Frame extracted")

    return output_path


def generate_preview_clip(
    video_path: str,
    start: float,
    end: float,
    output_path: Optional[str] = None,
    max_width: int = 854,
    max_height: int = 480,
    target_fps: int = 0,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Generate a lightweight preview clip for in-browser playback.

    Args:
        video_path: Path to the source video.
        start: Start time in seconds.
        end: End time in seconds.
        output_path: Output video path (auto-generated if None).
        max_width: Maximum width for preview (0 = no limit).
        max_height: Maximum height for preview (0 = no limit).
        target_fps: Target frame rate (0 = original).

    Returns:
        Dict with output_path, duration, file_size.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if end <= start:
        raise ValueError("End time must be greater than start time")

    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"preview_clip_{os.getpid()}.mp4"
        )

    if on_progress:
        on_progress(10, "Preparing preview clip")

    info = get_video_info(video_path)
    src_w = info.get("width", 1920)
    src_h = info.get("height", 1080)

    # Determine scale filter
    vf_parts = []
    if max_width > 0 and max_height > 0 and (src_w > max_width or src_h > max_height):
        vf_parts.append(
            f"scale='min({max_width},iw)':min'({max_height},ih)'"
            f":force_original_aspect_ratio=decrease"
        )
        # Use simpler scale that works reliably
        vf_parts = [f"scale={max_width}:-2"]

    if target_fps > 0:
        vf_parts.append(f"fps={target_fps}")

    if on_progress:
        on_progress(30, "Encoding preview")

    builder = (
        FFmpegCmd()
        .pre_input("-ss", str(start))
        .input(video_path)
        .seek(end=str(end - start))
        .video_codec("libx264", crf=28, preset="ultrafast")
        .audio_codec("aac", bitrate="96k")
        .faststart()
    )

    if vf_parts:
        builder = builder.video_filter(",".join(vf_parts))

    cmd = builder.output(output_path).build()
    run_ffmpeg(cmd, timeout=120)

    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    duration = end - start

    if on_progress:
        on_progress(100, "Preview clip ready")

    logger.info("Generated preview clip: %.1fs, %d bytes", duration, file_size)
    return {
        "output_path": output_path,
        "duration": round(duration, 3),
        "file_size": file_size,
        "width": min(src_w, max_width) if max_width > 0 else src_w,
        "height": min(src_h, max_height) if max_height > 0 else src_h,
    }


def get_frame_at_position(
    video_path: str,
    timestamp: float,
    on_progress: Optional[Callable] = None,
) -> bytes:
    """Get a JPEG-encoded frame at a precise timestamp.

    Returns raw JPEG bytes suitable for streaming directly to the client.

    Args:
        video_path: Path to the source video.
        timestamp: Time position in seconds.

    Returns:
        JPEG-encoded frame bytes.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(20, "Extracting frame")

    cmd = [
        get_ffmpeg_path(),
        "-hide_banner", "-loglevel", "error",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-f", "image2",
        "-c:v", "mjpeg",
        "-q:v", "3",
        "pipe:1",
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(
            f"Frame extraction failed: {result.stderr[:200].decode()}"
        )

    if on_progress:
        on_progress(100, "Frame ready")

    return result.stdout


def generate_thumbnail_strip(
    video_path: str,
    count: int = 10,
    width: int = 160,
    height: int = 90,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Generate a strip of evenly-spaced thumbnails for scrubbing.

    Args:
        video_path: Path to the source video.
        count: Number of thumbnails to generate.
        width: Width of each thumbnail.
        height: Height of each thumbnail.
        output_path: Output directory for thumbnails.

    Returns:
        Dict with thumbnail paths and timestamps.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    count = max(1, min(count, 100))

    info = get_video_info(video_path)
    duration = info.get("duration", 0)
    if duration <= 0:
        raise ValueError("Could not determine video duration")

    if output_path is None:
        output_path = tempfile.mkdtemp(prefix="opencut_thumbs_")

    os.makedirs(output_path, exist_ok=True)
    interval = duration / count
    thumbnails = []

    for i in range(count):
        ts = interval * (i + 0.5)
        thumb_file = os.path.join(output_path, f"thumb_{i:04d}.jpg")

        cmd = (
            FFmpegCmd()
            .pre_input("-ss", str(ts))
            .input(video_path)
            .video_filter(f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
            .frames(1)
            .option("q:v", "4")
            .output(thumb_file)
            .build()
        )

        try:
            run_ffmpeg(cmd, timeout=10)
            thumbnails.append({
                "path": thumb_file,
                "timestamp": round(ts, 3),
                "index": i,
            })
        except RuntimeError as exc:
            logger.warning("Failed to generate thumbnail %d: %s", i, exc)

        if on_progress:
            on_progress(int(100 * (i + 1) / count), f"Thumbnail {i+1}/{count}")

    return {
        "thumbnails": thumbnails,
        "count": len(thumbnails),
        "interval": round(interval, 3),
        "duration": round(duration, 3),
        "output_dir": output_path,
    }
