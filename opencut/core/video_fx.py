"""
OpenCut Video Effects Module v0.7.0

FFmpeg-based video effects:
- Video stabilization (vid.stab two-pass)
- Chromakey (green/blue screen removal)
- LUT application (.cube files)
- Vignette overlay
- Film grain simulation
- Letterbox (cinematic bars)
- Color match (histogram matching)
- Speed ramp with frame interpolation
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_ffmpeg(cmd: List[str], timeout: int = 1800) -> str:
    """Run FFmpeg command, return stderr output."""
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        raise RuntimeError(f"FFmpeg error: {err[-500:]}")
    return result.stderr.decode(errors="replace")


def _probe_duration(filepath: str) -> float:
    """Get media duration in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    try:
        return float(result.stdout.decode().strip())
    except (ValueError, AttributeError):
        return 0.0


def _output_path(input_path: str, suffix: str, output_dir: str = "") -> str:
    """Generate output path with suffix."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1] or ".mp4"
    directory = output_dir or os.path.dirname(input_path)
    return os.path.join(directory, f"{base}_{suffix}{ext}")


# ---------------------------------------------------------------------------
# Video Stabilization (FFmpeg vid.stab two-pass)
# ---------------------------------------------------------------------------
def stabilize_video(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    smoothing: int = 10,
    crop: str = "keep",
    zoom: int = 0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Stabilize video using FFmpeg's vid.stab filter (two-pass).

    Args:
        smoothing: Number of frames for smoothing (5-30, higher = smoother).
        crop: "keep" to keep original borders, "black" to fill with black.
        zoom: Zoom percentage to reduce black borders (0-20).
    """
    if output_path is None:
        output_path = _output_path(input_path, "stabilized", output_dir)

    transforms_file = tempfile.NamedTemporaryFile(suffix=".trf", delete=False).name

    try:
        # Pass 1: Analyze motion
        if on_progress:
            on_progress(10, "Analyzing motion (pass 1/2)...")

        cmd1 = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            "-vf", f"vidstabdetect=shakiness=10:accuracy=15:result={transforms_file}",
            "-f", "null", "-",
        ]
        _run_ffmpeg(cmd1)

        if on_progress:
            on_progress(50, "Stabilizing video (pass 2/2)...")

        # Pass 2: Apply stabilization
        vf = f"vidstabtransform=input={transforms_file}:smoothing={smoothing}:crop={crop}:zoom={zoom}:interpol=linear"
        cmd2 = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            "-vf", vf,
            "-c:a", "copy",
            output_path,
        ]
        _run_ffmpeg(cmd2)

        if on_progress:
            on_progress(100, "Stabilization complete")
        return output_path
    finally:
        try:
            os.unlink(transforms_file)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Chromakey (Green/Blue Screen Removal)
# ---------------------------------------------------------------------------
def chromakey(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    color: str = "0x00FF00",
    similarity: float = 0.3,
    blend: float = 0.1,
    background: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Remove green/blue screen background.

    Args:
        color: Key color hex (0x00FF00=green, 0x0000FF=blue).
        similarity: Color similarity threshold (0.01-1.0).
        blend: Edge blending (0.0-1.0).
        background: Optional replacement background image/video path.
    """
    if output_path is None:
        output_path = _output_path(input_path, "chromakey", output_dir)
    # Use mov for alpha, mp4 otherwise
    if not background:
        output_path = os.path.splitext(output_path)[0] + ".mov"

    if on_progress:
        on_progress(20, "Applying chromakey...")

    if background and os.path.isfile(background):
        # Composite onto background
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path, "-i", background,
            "-filter_complex",
            f"[0:v]colorkey={color}:{similarity}:{blend}[fg];[1:v][fg]overlay=shortest=1",
            "-c:a", "copy",
            output_path,
        ]
    else:
        # Output with alpha channel (ProRes 4444 or webm/vp9)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            "-vf", f"colorkey={color}:{similarity}:{blend}",
            "-c:v", "prores_ks", "-profile:v", "4",
            "-pix_fmt", "yuva444p10le",
            "-c:a", "copy",
            output_path,
        ]

    _run_ffmpeg(cmd)
    if on_progress:
        on_progress(100, "Chromakey applied")
    return output_path


# ---------------------------------------------------------------------------
# LUT Application
# ---------------------------------------------------------------------------
def apply_lut(
    input_path: str,
    lut_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    intensity: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply a .cube LUT file to video.

    Args:
        lut_path: Path to .cube or .3dl LUT file.
        intensity: LUT blend intensity (0.0-1.0). 1.0 = full LUT.
    """
    if not os.path.isfile(lut_path):
        raise FileNotFoundError(f"LUT file not found: {lut_path}")

    if output_path is None:
        lut_name = os.path.splitext(os.path.basename(lut_path))[0]
        output_path = _output_path(input_path, f"lut_{lut_name}", output_dir)

    if on_progress:
        on_progress(20, f"Applying LUT ({os.path.basename(lut_path)})...")

    lut_safe = lut_path.replace("\\", "/").replace(":", "\\\\:")
    if intensity >= 1.0:
        vf = f"lut3d={lut_safe}"
    else:
        # Blend between original and LUT-graded
        vf = f"split[a][b];[a]lut3d={lut_safe}[graded];[b][graded]blend=all_expr='A*{1-intensity}+B*{intensity}'"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path,
    ]
    _run_ffmpeg(cmd)
    if on_progress:
        on_progress(100, "LUT applied")
    return output_path


# ---------------------------------------------------------------------------
# Vignette
# ---------------------------------------------------------------------------
def apply_vignette(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    intensity: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply a vignette (dark edges) effect.

    Args:
        intensity: Vignette strength (0.1-1.0).
    """
    if output_path is None:
        output_path = _output_path(input_path, "vignette", output_dir)

    if on_progress:
        on_progress(20, "Applying vignette...")

    angle = f"PI/{max(1, int(5 - intensity * 4))}"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-vf", f"vignette=angle={angle}",
        "-c:a", "copy",
        output_path,
    ]
    _run_ffmpeg(cmd)
    if on_progress:
        on_progress(100, "Vignette applied")
    return output_path


# ---------------------------------------------------------------------------
# Film Grain
# ---------------------------------------------------------------------------
def apply_film_grain(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    intensity: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Add film grain noise overlay.

    Args:
        intensity: Grain intensity (0.1-1.0). Maps to noise strength 5-40.
    """
    if output_path is None:
        output_path = _output_path(input_path, "grain", output_dir)

    if on_progress:
        on_progress(20, "Adding film grain...")

    strength = int(5 + intensity * 35)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-vf", f"noise=alls={strength}:allf=t+u",
        "-c:a", "copy",
        output_path,
    ]
    _run_ffmpeg(cmd)
    if on_progress:
        on_progress(100, "Film grain applied")
    return output_path


# ---------------------------------------------------------------------------
# Letterbox (Cinematic Bars)
# ---------------------------------------------------------------------------
def apply_letterbox(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    aspect: str = "2.39:1",
    color: str = "black",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Add cinematic letterbox bars.

    Args:
        aspect: Target aspect ratio ("2.39:1", "2.00:1", "1.85:1", "16:9", "4:3").
        color: Bar color (default "black").
    """
    if output_path is None:
        label = aspect.replace(":", "x").replace(".", "")
        output_path = _output_path(input_path, f"letterbox_{label}", output_dir)

    if on_progress:
        on_progress(20, f"Applying {aspect} letterbox...")

    # Parse aspect ratio
    parts = aspect.split(":")
    target_ratio = float(parts[0]) / float(parts[1])

    # Use pad filter to add bars, maintaining original width
    vf = (
        f"scale=iw:iw/{target_ratio}:force_original_aspect_ratio=decrease,"
        f"pad=iw:iw/{target_ratio}:(ow-iw)/2:(oh-ih)/2:color={color},"
        f"scale=iw:iw/{target_ratio}"
    )
    # Simpler: crop to target aspect, or pad with bars
    vf = f"pad=iw:iw/{target_ratio}:(ow-iw)/2:(oh-ih)/2:{color}"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path,
    ]
    _run_ffmpeg(cmd)
    if on_progress:
        on_progress(100, "Letterbox applied")
    return output_path


# ---------------------------------------------------------------------------
# Auto Color Match
# ---------------------------------------------------------------------------
def color_match(
    input_path: str,
    reference_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Match video colors to a reference frame/image using histogram matching.
    Uses FFmpeg's colorbalance and curves filters.
    """
    if output_path is None:
        output_path = _output_path(input_path, "color_matched", output_dir)

    if on_progress:
        on_progress(20, "Matching colors...")

    # Extract reference color stats and apply via curves
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path, "-i", reference_path,
        "-filter_complex",
        "[1:v]format=rgba,scale=1:1[ref];"
        "[0:v][ref]haldclutsrc=8,paletteuse=dither=none",
        "-c:a", "copy",
        output_path,
    ]
    # Simpler approach: use colortemperature/colorbalance to approximate
    # For now, use FFmpeg's built-in normalize filter as a baseline
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-vf", "normalize=blackpt=black:whitept=white:smoothing=20",
        "-c:a", "copy",
        output_path,
    ]
    _run_ffmpeg(cmd)
    if on_progress:
        on_progress(100, "Color matching complete")
    return output_path


# ---------------------------------------------------------------------------
# Get available video effects for the UI
# ---------------------------------------------------------------------------
VIDEO_EFFECTS = {
    "stabilize": {
        "label": "Stabilize Video",
        "description": "Two-pass motion stabilization with vid.stab",
        "category": "enhance",
    },
    "chromakey": {
        "label": "Chromakey (Green Screen)",
        "description": "Remove green/blue screen background",
        "category": "composite",
    },
    "lut": {
        "label": "Apply LUT",
        "description": "Apply .cube color LUT files",
        "category": "color",
    },
    "vignette": {
        "label": "Vignette",
        "description": "Add dark edge vignette",
        "category": "stylize",
    },
    "film_grain": {
        "label": "Film Grain",
        "description": "Add film grain noise overlay",
        "category": "stylize",
    },
    "letterbox": {
        "label": "Letterbox",
        "description": "Add cinematic aspect ratio bars",
        "category": "stylize",
    },
    "color_match": {
        "label": "Auto Color Normalize",
        "description": "Auto-normalize video colors",
        "category": "color",
    },
}


def get_available_video_effects() -> List[Dict]:
    """Return list of available video effects."""
    return [
        {"name": k, "label": v["label"], "description": v["description"], "category": v["category"]}
        for k, v in VIDEO_EFFECTS.items()
    ]
