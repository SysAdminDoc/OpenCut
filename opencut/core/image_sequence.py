"""
OpenCut Image Sequence Import & Assembly

Detects numbered image sequences in a folder and assembles them into
video files using FFmpeg's image2 demuxer.
"""

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg

logger = logging.getLogger("opencut")

# Supported image extensions for sequence detection
_IMAGE_EXTENSIONS = frozenset({
    ".tif", ".tiff", ".exr", ".dpx", ".jpg", ".jpeg",
    ".png", ".raw", ".bmp", ".tga", ".sgi",
})

# Pattern to extract numeric portion from filenames
_NUMERIC_RE = re.compile(r"^(.*?)(\d+)(\.[^.]+)$")


@dataclass
class SequenceInfo:
    """Information about a detected image sequence."""
    pattern: str  # FFmpeg-compatible pattern, e.g. "img_%04d.png"
    first_frame: int = 0
    last_frame: int = 0
    total_frames: int = 0
    extension: str = ""
    detected_pattern_str: str = ""  # Human-readable, e.g. "img_0001.png .. img_0240.png"
    gaps: List[int] = field(default_factory=list)  # Missing frame numbers
    folder: str = ""


def detect_image_sequence(folder_path: str) -> SequenceInfo:
    """
    Detect a numbered image sequence in a folder.

    Scans for files matching patterns like img_0001.tif, frame001.png,
    DSC_0234.jpg, etc.  Groups by prefix+extension and picks the largest
    group.

    Args:
        folder_path: Directory containing the image files.

    Returns:
        SequenceInfo with pattern, frame range, and gap info.

    Raises:
        FileNotFoundError: If folder doesn't exist.
        ValueError: If no image sequence detected.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Scan all files and group by (prefix, extension, digit_width)
    groups: Dict[Tuple[str, str, int], List[int]] = {}

    for fname in os.listdir(folder_path):
        m = _NUMERIC_RE.match(fname)
        if not m:
            continue
        prefix, digits, ext = m.group(1), m.group(2), m.group(3)
        if ext.lower() not in _IMAGE_EXTENSIONS:
            continue
        digit_width = len(digits)
        frame_num = int(digits)
        key = (prefix, ext.lower(), digit_width)
        groups.setdefault(key, []).append(frame_num)

    if not groups:
        raise ValueError(f"No image sequence detected in {folder_path}")

    # Pick the largest group
    best_key = max(groups, key=lambda k: len(groups[k]))
    prefix, ext, digit_width = best_key
    frame_numbers = sorted(groups[best_key])

    first_frame = frame_numbers[0]
    last_frame = frame_numbers[-1]
    total_frames = len(frame_numbers)

    # Detect gaps
    expected = set(range(first_frame, last_frame + 1))
    actual = set(frame_numbers)
    gaps = sorted(expected - actual)

    # Build FFmpeg pattern
    ffmpeg_pattern = f"{prefix}%0{digit_width}d{ext}"

    # Human-readable range
    first_name = f"{prefix}{str(first_frame).zfill(digit_width)}{ext}"
    last_name = f"{prefix}{str(last_frame).zfill(digit_width)}{ext}"
    detected_str = f"{first_name} .. {last_name}"

    return SequenceInfo(
        pattern=ffmpeg_pattern,
        first_frame=first_frame,
        last_frame=last_frame,
        total_frames=total_frames,
        extension=ext.lstrip("."),
        detected_pattern_str=detected_str,
        gaps=gaps[:50],  # Limit reported gaps
        folder=folder_path,
    )


# Codec configuration presets
_CODEC_OPTIONS = {
    "h264": {
        "codec": "libx264",
        "quality": {"high": "18", "medium": "23", "low": "28"},
        "extra": ["-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    },
    "h265": {
        "codec": "libx265",
        "quality": {"high": "20", "medium": "25", "low": "30"},
        "extra": ["-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                  "-tag:v", "hvc1"],
    },
    "prores_ks": {
        "codec": "prores_ks",
        "quality": {"high": "0", "medium": "2", "low": "3"},  # profile values
        "extra": ["-pix_fmt", "yuva444p10le"],
        "quality_flag": "-profile:v",
    },
    "ffv1": {
        "codec": "ffv1",
        "quality": {"high": "1", "medium": "1", "low": "1"},  # level
        "extra": ["-level", "3", "-slicecrc", "1"],
        "quality_flag": "-level",
    },
}

# Extension map for codecs
_CODEC_EXT = {
    "h264": ".mp4",
    "h265": ".mp4",
    "prores_ks": ".mov",
    "ffv1": ".mkv",
}


def assemble_image_sequence(
    folder_path: str,
    output_path_str: Optional[str] = None,
    fps: int = 24,
    pattern: Optional[str] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    codec: str = "prores_ks",
    quality: str = "high",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Assemble an image sequence into a video file.

    Args:
        folder_path: Directory containing image files.
        output_path_str: Destination video path. Auto-generated if None.
        fps: Output framerate.
        pattern: FFmpeg pattern (e.g. "img_%04d.png"). Auto-detected if None.
        start_frame: First frame number. Auto-detected if None.
        end_frame: Last frame number. Auto-detected if None.
        codec: Video codec: "h264", "h265", "prores_ks", "ffv1".
        quality: Quality preset: "high", "medium", "low".
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, total_frames, duration, codec.
    """
    if on_progress:
        on_progress(5, "Detecting image sequence...")

    # Detect sequence if pattern not provided
    if pattern is None:
        seq_info = detect_image_sequence(folder_path)
        pattern = seq_info.pattern
        if start_frame is None:
            start_frame = seq_info.first_frame
        if end_frame is None:
            end_frame = seq_info.last_frame
    else:
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = start_frame + 999  # Reasonable default

    # Validate codec
    codec = codec.lower()
    if codec not in _CODEC_OPTIONS:
        logger.warning("Unknown codec %s, falling back to prores_ks", codec)
        codec = "prores_ks"

    quality = quality.lower()
    if quality not in ("high", "medium", "low"):
        quality = "high"

    codec_cfg = _CODEC_OPTIONS[codec]
    ext = _CODEC_EXT[codec]

    # Generate output path
    if output_path_str is None:
        base_name = os.path.basename(folder_path.rstrip("/\\")) or "sequence"
        output_path_str = os.path.join(folder_path, f"{base_name}_assembled{ext}")

    if on_progress:
        on_progress(15, f"Assembling with {codec} codec...")

    # Calculate expected frames
    total_frames = end_frame - start_frame + 1
    expected_duration = total_frames / max(1, fps)

    # Build FFmpeg command
    ffmpeg = get_ffmpeg_path()
    input_pattern = os.path.join(folder_path, pattern)

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-framerate", str(fps),
        "-start_number", str(start_frame),
        "-i", input_pattern,
    ]

    # Frame limit if end_frame specified
    if end_frame is not None:
        frames_count = end_frame - start_frame + 1
        cmd.extend(["-frames:v", str(frames_count)])

    # Codec settings
    cmd.extend(["-c:v", codec_cfg["codec"]])

    # Quality setting
    quality_val = codec_cfg["quality"].get(quality, codec_cfg["quality"]["high"])
    quality_flag = codec_cfg.get("quality_flag", "-crf")
    cmd.extend([quality_flag, quality_val])

    # Extra codec options
    cmd.extend(codec_cfg["extra"])

    cmd.append(output_path_str)

    if on_progress:
        on_progress(30, f"Encoding {total_frames} frames...")

    run_ffmpeg(cmd, timeout=max(600, total_frames))

    if on_progress:
        on_progress(90, "Verifying output...")

    # Get actual output duration
    actual_duration = expected_duration
    try:
        probe_cmd = [
            get_ffprobe_path(), "-v", "quiet",
            "-print_format", "json", "-show_format",
            output_path_str,
        ]
        import json
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if probe_result.returncode == 0:
            probe_data = json.loads(probe_result.stdout)
            actual_duration = float(probe_data.get("format", {}).get("duration", expected_duration))
    except Exception:
        pass

    if on_progress:
        on_progress(100, "Image sequence assembled")

    return {
        "output_path": output_path_str,
        "total_frames": total_frames,
        "duration": round(actual_duration, 3),
        "fps": fps,
        "codec": codec,
        "quality": quality,
        "pattern": pattern,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }
