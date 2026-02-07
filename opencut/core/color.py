"""
OpenCut Color Grading

Applies color grading to video via LUT files and built-in FFmpeg filters.

Supports:
  - .cube LUT files (industry standard)
  - .3dl LUT files
  - Built-in color presets using FFmpeg filters (no external LUT needed)

All processing done through FFmpeg - no additional Python deps required.
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ColorResult:
    """Result of a color grading operation."""
    output_path: str = ""
    preset_used: str = ""
    lut_file: str = ""
    original_duration: float = 0.0


# ---------------------------------------------------------------------------
# Built-in color presets (FFmpeg filter chains, no LUT file needed)
# ---------------------------------------------------------------------------
BUILTIN_PRESETS = {
    "warm": {
        "label": "Warm",
        "description": "Warm, golden tones",
        "filter": "colorbalance=rs=0.1:gs=0.0:bs=-0.1:rh=0.1:gh=0.05:bh=-0.05",
    },
    "cool": {
        "label": "Cool",
        "description": "Cool blue undertones",
        "filter": "colorbalance=rs=-0.1:gs=0.0:bs=0.15:rh=-0.05:gh=0.0:bh=0.1",
    },
    "vintage": {
        "label": "Vintage",
        "description": "Faded retro film look",
        "filter": "curves=vintage,colorbalance=rs=0.05:gs=-0.03:bs=-0.08,eq=contrast=0.9:saturation=0.7",
    },
    "cinematic": {
        "label": "Cinematic",
        "description": "Teal-orange movie look",
        "filter": "colorbalance=rs=0.12:gs=-0.04:bs=-0.08:rm=-0.08:gm=0.02:bm=0.1:rh=0.05:gh=-0.02:bh=0.08,eq=contrast=1.1:saturation=0.85",
    },
    "bw_classic": {
        "label": "Black & White",
        "description": "Classic monochrome",
        "filter": "hue=s=0,eq=contrast=1.15:brightness=0.02",
    },
    "bw_high_contrast": {
        "label": "B&W High Contrast",
        "description": "Punchy black and white",
        "filter": "hue=s=0,eq=contrast=1.4:brightness=-0.02:gamma=0.9",
    },
    "matte": {
        "label": "Matte",
        "description": "Lifted blacks, soft matte finish",
        "filter": "curves=all='0/0.06 0.25/0.28 0.5/0.5 0.75/0.72 1/0.94',eq=saturation=0.85",
    },
    "vivid": {
        "label": "Vivid",
        "description": "Punchy, saturated colors",
        "filter": "eq=saturation=1.4:contrast=1.1:brightness=0.03",
    },
    "desaturated": {
        "label": "Desaturated",
        "description": "Muted, low saturation",
        "filter": "eq=saturation=0.5:contrast=1.05",
    },
    "sepia": {
        "label": "Sepia",
        "description": "Classic sepia tone",
        "filter": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
    },
    "cross_process": {
        "label": "Cross Process",
        "description": "Experimental color shift",
        "filter": "curves=cross_process,eq=saturation=1.1",
    },
    "negative": {
        "label": "Negative",
        "description": "Inverted colors",
        "filter": "negate",
    },
    "bleach_bypass": {
        "label": "Bleach Bypass",
        "description": "Desaturated high-contrast film look",
        "filter": "eq=saturation=0.4:contrast=1.5:brightness=-0.05",
    },
    "golden_hour": {
        "label": "Golden Hour",
        "description": "Warm sunset lighting",
        "filter": "colorbalance=rs=0.15:gs=0.08:bs=-0.1:rh=0.12:gh=0.06:bh=-0.06,eq=brightness=0.04:saturation=1.15",
    },
    "moonlight": {
        "label": "Moonlight",
        "description": "Cold blue nighttime look",
        "filter": "colorbalance=rs=-0.15:gs=-0.05:bs=0.2:rh=-0.1:gh=-0.02:bh=0.15,eq=brightness=-0.08:contrast=1.1",
    },
}


def get_color_presets() -> List[Dict]:
    """Return all available built-in color presets."""
    return [
        {"name": name, "label": data["label"], "description": data["description"]}
        for name, data in BUILTIN_PRESETS.items()
    ]


# ---------------------------------------------------------------------------
# LUT file validation
# ---------------------------------------------------------------------------
def validate_lut_file(lut_path: str) -> Dict:
    """
    Validate a LUT file and return its metadata.

    Returns:
        {"valid": bool, "format": str, "size": int, "error": str}
    """
    result = {"valid": False, "format": "", "size": 0, "error": ""}

    if not os.path.isfile(lut_path):
        result["error"] = f"File not found: {lut_path}"
        return result

    ext = os.path.splitext(lut_path)[1].lower()

    if ext == ".cube":
        result["format"] = "cube"
    elif ext == ".3dl":
        result["format"] = "3dl"
    else:
        result["error"] = f"Unsupported LUT format: {ext}. Use .cube or .3dl"
        return result

    size = os.path.getsize(lut_path)
    result["size"] = size

    if size == 0:
        result["error"] = "LUT file is empty"
        return result

    if size > 50 * 1024 * 1024:  # 50 MB
        result["error"] = "LUT file too large (>50 MB)"
        return result

    # Quick sanity check - read first few lines
    try:
        with open(lut_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.read(2048)

        if ext == ".cube":
            # .cube files should have LUT_3D_SIZE or LUT_1D_SIZE
            if "LUT_3D_SIZE" not in header and "LUT_1D_SIZE" not in header:
                result["error"] = "Invalid .cube file: missing LUT size declaration"
                return result

        result["valid"] = True

    except Exception as e:
        result["error"] = f"Cannot read LUT file: {e}"

    return result


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------
def _probe_duration(filepath: str) -> float:
    """Get media duration in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0.0))
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Color grading application
# ---------------------------------------------------------------------------
def apply_color_grade(
    input_path: str,
    preset: str = "",
    lut_path: str = "",
    intensity: float = 1.0,
    output_dir: str = "",
    quality: str = "medium",
    on_progress: Optional[Callable] = None,
) -> ColorResult:
    """
    Apply color grading to a video.

    Either a built-in preset OR a custom LUT file can be used.

    Args:
        input_path:  Source video file.
        preset:      Built-in preset name (ignored if lut_path given).
        lut_path:    Path to .cube or .3dl LUT file.
        intensity:   Blend intensity 0.0-1.0 (1.0 = full effect).
        output_dir:  Output directory.
        quality:     Encoding quality.
        on_progress: Callback(pct, msg).

    Returns:
        ColorResult with output path and metadata.
    """
    if on_progress:
        on_progress(5, "Preparing color grade...")

    if not lut_path and not preset:
        raise ValueError("Must specify either a preset or a LUT file")

    duration = _probe_duration(input_path)
    intensity = max(0.0, min(1.0, intensity))

    # Build video filter
    if lut_path:
        # External LUT file
        lut_info = validate_lut_file(lut_path)
        if not lut_info["valid"]:
            raise ValueError(f"Invalid LUT file: {lut_info['error']}")

        # FFmpeg lut3d filter
        lut_escaped = lut_path.replace("\\", "/").replace(":", "\\:")
        vf_filter = f"lut3d='{lut_escaped}'"
        preset_label = os.path.splitext(os.path.basename(lut_path))[0]

        if on_progress:
            on_progress(20, f"Applying LUT: {os.path.basename(lut_path)}")

    else:
        # Built-in preset
        preset_data = BUILTIN_PRESETS.get(preset)
        if not preset_data:
            raise ValueError(f"Unknown color preset: {preset}")

        vf_filter = preset_data["filter"]
        preset_label = preset

        if on_progress:
            on_progress(20, f"Applying preset: {preset_data['label']}")

    # Apply intensity blending (mix original with graded)
    if intensity < 1.0:
        # Use split + merge to blend
        # Original gets (1 - intensity), graded gets (intensity)
        blend_filter = (
            f"split[original][tograde];"
            f"[tograde]{vf_filter}[graded];"
            f"[original][graded]blend=all_mode=normal:all_opacity={intensity:.3f}"
        )
        vf_filter = blend_filter

    # Output path
    if not output_dir:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base}_color_{preset_label}.mp4")

    # Quality settings
    crf_map = {"low": "28", "medium": "23", "high": "18", "lossless": "0"}
    crf = crf_map.get(quality, "23")

    if on_progress:
        on_progress(40, "Encoding color-graded video...")

    # Determine if we need filter_complex or vf
    if "split[" in vf_filter:
        # Complex filter graph
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-y",
            "-i", input_path,
            "-filter_complex", vf_filter,
            "-c:v", "libx264", "-crf", crf, "-preset", "fast",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-y",
            "-i", input_path,
            "-vf", vf_filter,
            "-c:v", "libx264", "-crf", crf, "-preset", "fast",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path,
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"FFmpeg color grade error: {result.stderr[-1000:]}")
            raise RuntimeError(f"FFmpeg encoding failed: {result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Color grade encoding timed out (>60 minutes)")

    if on_progress:
        on_progress(100, "Color grade complete")

    return ColorResult(
        output_path=output_path,
        preset_used=preset_label,
        lut_file=lut_path or "",
        original_duration=duration,
    )
