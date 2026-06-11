"""
Media file probing utilities using ffprobe.

Extracts video/audio metadata needed for timeline calculations and XML export.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

from opencut.helpers import get_ffprobe_path

logger = logging.getLogger("opencut")


@dataclass
class VideoStream:
    """Video stream metadata."""
    width: int = 1920
    height: int = 1080
    fps: float = 29.97
    fps_fraction: Fraction = field(default_factory=lambda: Fraction(30000, 1001))
    codec: str = "h264"
    duration: float = 0.0
    pixel_aspect_ratio: str = "1:1"
    field_order: str = "progressive"

    @property
    def timebase(self) -> int:
        """FCP XML timebase (rounded frame rate). Use effective_timebase for NTSC-aware values."""
        return round(self.fps)

    @property
    def ntsc(self) -> bool:
        """Whether this is an NTSC frame rate (e.g., 29.97, 23.976, 59.94)."""
        # NTSC rates are approximately N * 1000/1001
        for base in [24, 30, 60, 120]:
            ntsc_rate = base * 1000 / 1001
            if abs(self.fps - ntsc_rate) < 0.05:
                return True
        return False

    @property
    def effective_timebase(self) -> int:
        """The timebase value for FCP XML."""
        if self.ntsc:
            # NTSC: timebase is the nearest integer above the actual rate
            for base in [24, 30, 60, 120]:
                if abs(self.fps - base * 1000 / 1001) < 0.05:
                    return base
        return round(self.fps)

    def seconds_to_frames(self, seconds: float) -> int:
        """Convert seconds to frame count at this stream's frame rate."""
        return round(seconds * self.fps)


@dataclass
class AudioStream:
    """Audio stream metadata."""
    sample_rate: int = 48000
    channels: int = 2
    bit_depth: int = 16
    codec: str = "aac"
    duration: float = 0.0


@dataclass
class MediaInfo:
    """Complete media file information."""
    path: str = ""
    filename: str = ""
    duration: float = 0.0
    format_name: str = ""
    video: Optional[VideoStream] = None
    audio: Optional[AudioStream] = None
    video_is_attached_pic: bool = False

    @property
    def has_video(self) -> bool:
        return self.video is not None

    @property
    def has_real_video(self) -> bool:
        return self.video is not None and not self.video_is_attached_pic

    @property
    def attached_pic(self) -> bool:
        return self.video_is_attached_pic

    @property
    def has_audio(self) -> bool:
        return self.audio is not None

    @property
    def pathurl(self) -> str:
        """
        File URL for use in FCP XML.

        Matches the format Premiere Pro uses in its own XML exports:
          Windows: file://localhost/C%3A/Users/name/video.mp4
          Mac:     file://localhost/Volumes/HD/video.mp4
          Linux:   file://localhost/home/user/video.mp4

        All special characters (including : after drive letters, spaces,
        #, etc.) are percent-encoded except forward slashes.
        """
        from urllib.parse import quote

        abs_path = os.path.abspath(self.path)

        if os.name == "nt":
            # Windows: normalize to forward slashes first
            url_path = abs_path.replace("\\", "/")

            # Handle UNC paths: \\server\share -> file://server/share
            if url_path.startswith("//"):
                encoded = quote(url_path[2:], safe="/")
                return f"file://{encoded}"
            else:
                # Drive letter path: C:/path -> encode everything except /
                encoded = quote(url_path, safe="/")
                return f"file://localhost/{encoded}"
        else:
            # Unix/Mac: encode everything except /
            encoded = quote(abs_path, safe="/")
            return f"file://localhost{encoded}"


def probe(filepath: str) -> MediaInfo:
    """
    Probe a media file and extract metadata using ffprobe.

    Args:
        filepath: Path to the media file.

    Returns:
        MediaInfo with video/audio stream details.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        RuntimeError: If ffprobe fails.
    """
    filepath = str(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Media file not found: {filepath}")

    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        filepath,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    except FileNotFoundError as err:
        raise RuntimeError("ffprobe not found. Install FFmpeg: https://ffmpeg.org/download.html") from err
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed on '{filepath}': {e.stderr}") from e
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(f"ffprobe timed out on '{filepath}'") from err

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe returned invalid JSON for '{filepath}': {e}") from e
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    def _safe_float(value, default=0.0):
        """Coerce ffprobe numeric fields — they're sometimes ``"N/A"`` on
        live streams, fragmented sources, or files with missing metadata."""
        if value is None:
            return default
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        # Reject NaN / inf — they corrupt downstream timeline math silently.
        if result != result or result in (float("inf"), float("-inf")):
            return default
        return result

    def _safe_int(value, default=0):
        try:
            return int(float(value)) if value is not None else default
        except (TypeError, ValueError):
            return default

    info = MediaInfo(
        path=filepath,
        filename=os.path.basename(filepath),
        duration=_safe_float(fmt.get("duration")),
        format_name=fmt.get("format_name", ""),
    )

    def _is_attached_pic(stream) -> bool:
        """True if the stream is an embedded thumbnail / album art.
        ffprobe exposes this as disposition.attached_pic == 1."""
        disp = stream.get("disposition") or {}
        return bool(disp.get("attached_pic"))

    # Parse video stream — prefer the first non-attached-pic stream so files
    # with embedded thumbnails (MP3 album art, MP4 poster frames) report
    # the real video dimensions rather than the thumbnail's.
    for stream in streams:
        if stream.get("codec_type") != "video" or _is_attached_pic(stream):
            continue
        fps_str = stream.get("r_frame_rate", "30/1")
        try:
            fps_frac = Fraction(fps_str)
            fps = float(fps_frac)
        except (ValueError, ZeroDivisionError):
            fps_frac = Fraction(30000, 1001)
            fps = 29.97

        # Get pixel aspect ratio
        sar = stream.get("sample_aspect_ratio", "1:1")
        if sar == "0:1" or sar == "N/A":
            sar = "1:1"

        field_order = stream.get("field_order", "progressive")

        info.video = VideoStream(
            width=_safe_int(stream.get("width"), 1920),
            height=_safe_int(stream.get("height"), 1080),
            fps=fps,
            fps_fraction=fps_frac,
            codec=stream.get("codec_name", "h264"),
            duration=_safe_float(stream.get("duration"), _safe_float(fmt.get("duration"))),
            pixel_aspect_ratio=sar,
            field_order=field_order,
        )
        break
    # Second pass: if every video stream is attached-pic, fall back to the
    # first one so we emit *some* dimensions rather than None.
    if info.video is None:
        for stream in streams:
            if stream.get("codec_type") == "video":
                info.video_is_attached_pic = _is_attached_pic(stream)
                info.video = VideoStream(
                    width=_safe_int(stream.get("width"), 1920),
                    height=_safe_int(stream.get("height"), 1080),
                    fps=29.97,
                    fps_fraction=Fraction(30000, 1001),
                    codec=stream.get("codec_name", "h264"),
                    duration=_safe_float(stream.get("duration"), _safe_float(fmt.get("duration"))),
                    pixel_aspect_ratio="1:1",
                    field_order="progressive",
                )
                break

    # Parse audio stream
    for stream in streams:
        if stream.get("codec_type") == "audio":
            # Determine bit depth
            bit_depth = 16
            bits_raw = stream.get("bits_per_raw_sample") or stream.get("bits_per_sample")
            if bits_raw:
                coerced = _safe_int(bits_raw, 0)
                if coerced > 0:
                    bit_depth = coerced
                else:
                    logger.warning("Could not parse bit depth '%s', using default 16", bits_raw)

            info.audio = AudioStream(
                sample_rate=_safe_int(stream.get("sample_rate"), 48000),
                channels=_safe_int(stream.get("channels"), 2),
                bit_depth=bit_depth,
                codec=stream.get("codec_name", "aac"),
                duration=_safe_float(stream.get("duration"), _safe_float(fmt.get("duration"))),
            )
            break

    return info


def get_audio_duration(filepath: str) -> float:
    """Get the duration of an audio/video file in seconds."""
    info = probe(filepath)
    return info.duration
