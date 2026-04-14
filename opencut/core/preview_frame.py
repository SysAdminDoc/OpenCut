"""
OpenCut Preview Frame - Before/After Operation Preview

Extract a single frame from video, apply a processing operation via FFmpeg,
and return both original and processed frames as base64 for before/after
comparison in the UI.
"""

import base64
import json
import logging
import os
import subprocess as _sp
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg

logger = logging.getLogger("opencut")


@dataclass
class PreviewResult:
    """Result of a preview operation with before/after frames."""
    original_b64: str
    processed_b64: str
    width: int
    height: int
    timestamp: float


# ---------------------------------------------------------------------------
# Operation filter map -- maps operation names to FFmpeg filter chains
# ---------------------------------------------------------------------------
_OPERATION_FILTERS: Dict[str, callable] = {}


def _denoise_filter(params: dict) -> str:
    """Build FFmpeg denoise filter string."""
    strength = params.get("strength", "moderate")
    if strength == "light":
        return "hqdn3d=2:2:3:3"
    elif strength == "heavy":
        return "hqdn3d=8:6:10:8"
    return "hqdn3d=4:3:6:4.5"


def _upscale_filter(params: dict) -> str:
    """Build FFmpeg upscale filter string."""
    factor = int(params.get("factor", 2))
    factor = max(1, min(factor, 4))
    return f"scale=iw*{factor}:ih*{factor}:flags=lanczos"


def _color_correct_filter(params: dict) -> str:
    """Build FFmpeg color correction filter string."""
    brightness = float(params.get("brightness", 0.0))
    contrast = float(params.get("contrast", 1.0))
    saturation = float(params.get("saturation", 1.0))
    gamma = float(params.get("gamma", 1.0))
    return f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma}"


def _stabilize_filter(params: dict) -> str:
    """Build FFmpeg stabilization simulation filter for a single frame.

    Single-frame stabilization is limited; apply a slight unsharp to
    simulate the perceived sharpness gain from stabilization.
    """
    return "unsharp=5:5:0.8:5:5:0.0"


def _lut_apply_filter(params: dict) -> str:
    """Build FFmpeg LUT application filter string."""
    lut_path = params.get("lut_path", "")
    if lut_path and os.path.isfile(lut_path):
        # Normalize path separators for FFmpeg
        safe_path = lut_path.replace("\\", "/").replace(":", "\\:")
        return f"lut3d=file='{safe_path}'"
    # Fallback: warm color shift as demo LUT
    return "colorbalance=rs=0.1:gs=0.05:bs=-0.05"


def _brightness_filter(params: dict) -> str:
    """Build FFmpeg brightness adjustment filter."""
    value = float(params.get("value", 0.1))
    value = max(-1.0, min(1.0, value))
    return f"eq=brightness={value}"


def _contrast_filter(params: dict) -> str:
    """Build FFmpeg contrast adjustment filter."""
    value = float(params.get("value", 1.2))
    value = max(0.0, min(3.0, value))
    return f"eq=contrast={value}"


def _saturation_filter(params: dict) -> str:
    """Build FFmpeg saturation adjustment filter."""
    value = float(params.get("value", 1.3))
    value = max(0.0, min(3.0, value))
    return f"eq=saturation={value}"


_OPERATION_FILTERS = {
    "denoise": _denoise_filter,
    "upscale": _upscale_filter,
    "color_correct": _color_correct_filter,
    "stabilize": _stabilize_filter,
    "lut_apply": _lut_apply_filter,
    "brightness": _brightness_filter,
    "contrast": _contrast_filter,
    "saturation": _saturation_filter,
}

SUPPORTED_OPERATIONS = list(_OPERATION_FILTERS.keys())


def _probe_dimensions(video_path: str) -> tuple:
    """Probe video dimensions using ffprobe. Returns (width, height)."""
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        video_path,
    ]
    try:
        result = _sp.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0:
            info = json.loads(result.stdout.decode(errors="replace"))
            streams = info.get("streams", [])
            if streams:
                w = int(streams[0].get("width", 0))
                h = int(streams[0].get("height", 0))
                return (w, h)
    except Exception as exc:
        logger.debug("Failed to probe dimensions: %s", exc)
    return (0, 0)


def extract_frame(
    video_path: str,
    timestamp: float = 0.0,
    on_progress: Optional[Callable] = None,
) -> bytes:
    """Extract a single frame from video as PNG bytes.

    Args:
        video_path: Path to the source video file.
        timestamp: Time in seconds to extract frame from.
        on_progress: Progress callback(pct, msg).

    Returns:
        PNG image bytes of the extracted frame.

    Raises:
        RuntimeError: If FFmpeg frame extraction fails.
        FileNotFoundError: If video_path does not exist.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(10, "Extracting frame...")

    timestamp = max(0.0, float(timestamp))

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="opencut_frame_")
    os.close(tmp_fd)

    try:
        cmd = [
            get_ffmpeg_path(),
            "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-frames:v", "1",
            "-f", "image2",
            tmp_path,
        ]
        run_ffmpeg(cmd, timeout=60)

        if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise RuntimeError("FFmpeg produced empty frame output")

        if on_progress:
            on_progress(90, "Reading frame data...")

        with open(tmp_path, "rb") as f:
            frame_data = f.read()

        if on_progress:
            on_progress(100, "Frame extracted")

        return frame_data
    finally:
        try:
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


def preview_operation(
    video_path: str,
    operation: str,
    params: Optional[dict] = None,
    timestamp: float = 0.0,
    on_progress: Optional[Callable] = None,
) -> PreviewResult:
    """Extract a frame, apply an operation, return before/after as base64.

    Args:
        video_path: Path to the source video file.
        operation: Operation name (one of SUPPORTED_OPERATIONS).
        params: Operation-specific parameters dict.
        timestamp: Time in seconds to extract the frame from.
        on_progress: Progress callback(pct, msg).

    Returns:
        PreviewResult with original and processed frames as base64 strings.

    Raises:
        ValueError: If operation is not supported.
        RuntimeError: If FFmpeg processing fails.
        FileNotFoundError: If video_path does not exist.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if operation not in _OPERATION_FILTERS:
        raise ValueError(
            f"Unsupported operation: {operation}. "
            f"Supported: {', '.join(SUPPORTED_OPERATIONS)}"
        )

    if params is None:
        params = {}

    timestamp = max(0.0, float(timestamp))

    if on_progress:
        on_progress(5, "Probing video dimensions...")

    width, height = _probe_dimensions(video_path)

    if on_progress:
        on_progress(10, "Extracting original frame...")

    # Extract original frame
    original_bytes = extract_frame(video_path, timestamp)

    if on_progress:
        on_progress(40, f"Applying {operation}...")

    # Build filter for the operation
    filter_fn = _OPERATION_FILTERS[operation]
    filter_str = filter_fn(params)

    # Apply operation to extract a processed frame directly from video
    tmp_fd, tmp_processed = tempfile.mkstemp(
        suffix=".png", prefix="opencut_preview_"
    )
    os.close(tmp_fd)

    try:
        cmd = [
            get_ffmpeg_path(),
            "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-frames:v", "1",
            "-vf", filter_str,
            "-f", "image2",
            tmp_processed,
        ]
        run_ffmpeg(cmd, timeout=60)

        if not os.path.isfile(tmp_processed) or os.path.getsize(tmp_processed) == 0:
            raise RuntimeError(f"FFmpeg produced empty output for operation: {operation}")

        if on_progress:
            on_progress(80, "Encoding results...")

        with open(tmp_processed, "rb") as f:
            processed_bytes = f.read()

        original_b64 = base64.b64encode(original_bytes).decode("ascii")
        processed_b64 = base64.b64encode(processed_bytes).decode("ascii")

        if on_progress:
            on_progress(100, "Preview complete")

        return PreviewResult(
            original_b64=original_b64,
            processed_b64=processed_b64,
            width=width,
            height=height,
            timestamp=timestamp,
        )
    finally:
        try:
            if os.path.isfile(tmp_processed):
                os.unlink(tmp_processed)
        except OSError:
            pass
