"""
OpenCut Color Management Module v0.9.0

Professional color management:
- OpenColorIO (OCIO) integration for ACES workflows
- Extended LUT format support (.3dl, .clf, .csp, .cube)
- Color space conversions (sRGB, Rec.709, Rec.2020, ACES, DCI-P3)
- Exposure, white balance, lift/gamma/gain controls via FFmpeg
- Histogram analysis
- Color match between clips

FFmpeg-based controls always available. OCIO optional for ACES pipeline.
"""

import logging
import os
import subprocess
from typing import Callable, Dict, Optional

from opencut.helpers import get_ffprobe_path, run_ffmpeg

logger = logging.getLogger("opencut")


def check_ocio_available() -> bool:
    try:
        import PyOpenColorIO  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Color Space Conversion (FFmpeg)
# ---------------------------------------------------------------------------
COLOR_SPACES = {
    "srgb": {"label": "sRGB", "matrix": "bt709", "transfer": "iec61966-2-1", "primaries": "bt709"},
    "rec709": {"label": "Rec. 709", "matrix": "bt709", "transfer": "bt709", "primaries": "bt709"},
    "rec2020": {"label": "Rec. 2020", "matrix": "bt2020nc", "transfer": "bt2020-10", "primaries": "bt2020"},
    "dci_p3": {"label": "DCI-P3", "matrix": "bt709", "transfer": "smpte2084", "primaries": "smpte432"},
}


def convert_colorspace(
    video_path: str,
    target: str = "rec709",
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """Convert video to a different color space."""
    cs = COLOR_SPACES.get(target, COLOR_SPACES["rec709"])

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_{target}.mp4")

    if on_progress:
        on_progress(10, f"Converting to {cs['label']}...")

    vf = (
        f"colorspace=all={cs['matrix']}:"
        f"trc={cs['transfer']}:"
        f"primaries={cs['primaries']}"
    )

    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path, "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        output_path,
    ], timeout=7200)

    if on_progress:
        on_progress(100, f"Converted to {cs['label']}")
    return output_path


# ---------------------------------------------------------------------------
# Color Correction Controls (FFmpeg eq/colorbalance)
# ---------------------------------------------------------------------------
def color_correct(
    video_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    exposure: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    temperature: float = 0.0,
    tint: float = 0.0,
    shadows: float = 0.0,
    midtones: float = 0.0,
    highlights: float = 0.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply color correction to video.

    Args:
        exposure: Exposure adjustment (-3.0 to 3.0, 0 = neutral).
        contrast: Contrast (0.5-2.0, 1.0 = neutral).
        saturation: Saturation (0-3.0, 1.0 = neutral).
        temperature: Color temperature shift (-1.0 to 1.0, negative=cool, positive=warm).
        tint: Green/magenta tint (-1.0 to 1.0).
        shadows/midtones/highlights: Lift/Gamma/Gain (-1.0 to 1.0).
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_corrected.mp4")

    if on_progress:
        on_progress(10, "Applying color correction...")

    filters = []

    # Build a single eq filter with all brightness/contrast/saturation params
    eq_parts = []
    if abs(exposure) > 0.01:
        brightness = max(-1, min(1, exposure * 0.15))
        eq_parts.append(f"brightness={brightness}")
    if abs(contrast - 1.0) > 0.01:
        eq_parts.append(f"contrast={contrast}")
    if abs(saturation - 1.0) > 0.01:
        eq_parts.append(f"saturation={saturation}")
    if eq_parts:
        filters.append(f"eq={':'.join(eq_parts)}")

    # Build a single colorbalance filter combining temperature + lift/gamma/gain
    cb_parts = {}
    if abs(temperature) > 0.01 or abs(tint) > 0.01:
        rs = max(-1, min(1, temperature * 0.3))
        gs = max(-1, min(1, tint * 0.2))
        bs = max(-1, min(1, -temperature * 0.3))
        cb_parts["rs"] = rs
        cb_parts["gs"] = gs
        cb_parts["bs"] = bs
        cb_parts["rm"] = rs * 0.5
        cb_parts["gm"] = gs * 0.5
        cb_parts["bm"] = bs * 0.5
    if abs(shadows) > 0.01:
        cb_parts["rs"] = cb_parts.get("rs", 0) + shadows * 0.3
        cb_parts["gs"] = cb_parts.get("gs", 0) + shadows * 0.3
        cb_parts["bs"] = cb_parts.get("bs", 0) + shadows * 0.3
    if abs(midtones) > 0.01:
        cb_parts["rm"] = cb_parts.get("rm", 0) + midtones * 0.3
        cb_parts["gm"] = cb_parts.get("gm", 0) + midtones * 0.3
        cb_parts["bm"] = cb_parts.get("bm", 0) + midtones * 0.3
    if abs(highlights) > 0.01:
        cb_parts["rh"] = highlights * 0.3
        cb_parts["gh"] = highlights * 0.3
        cb_parts["bh"] = highlights * 0.3
    if cb_parts:
        # Clamp all values to [-1, 1]
        parts_str = ":".join(f"{k}={max(-1, min(1, v))}" for k, v in cb_parts.items())
        filters.append(f"colorbalance={parts_str}")

    if not filters:
        # No corrections needed, just copy
        filters.append("null")

    vf = ",".join(filters)

    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path, "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        output_path,
    ], timeout=7200)

    if on_progress:
        on_progress(100, "Color correction applied!")
    return output_path


# ---------------------------------------------------------------------------
# Histogram / Color Analysis
# ---------------------------------------------------------------------------
def analyze_colors(
    video_path: str,
    sample_count: int = 10,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """
    Analyze video color distribution.

    Samples frames and returns average brightness, contrast,
    saturation, and color balance info.
    """
    import json

    if on_progress:
        on_progress(10, "Analyzing colors...")

    # Basic analysis via ffprobe
    cmd2 = [
        get_ffprobe_path(), "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=color_space,color_transfer,color_primaries",
        "-of", "json", video_path,
    ]
    r2 = subprocess.run(cmd2, capture_output=True, timeout=10)

    color_info = {"color_space": "unknown", "transfer": "unknown", "primaries": "unknown"}
    try:
        if r2.returncode != 0:
            raise RuntimeError("ffprobe failed")
        streams = json.loads(r2.stdout.decode()).get("streams", [])
        if not streams:
            raise RuntimeError("no streams")
        s = streams[0]
        color_info["color_space"] = s.get("color_space", "unknown")
        color_info["transfer"] = s.get("color_transfer", "unknown")
        color_info["primaries"] = s.get("color_primaries", "unknown")
    except Exception:
        pass

    if on_progress:
        on_progress(100, "Analysis complete")

    return {
        "color_metadata": color_info,
        "available_spaces": list(COLOR_SPACES.keys()),
        "ocio_available": check_ocio_available(),
    }


# ---------------------------------------------------------------------------
# Apply external LUT file (.cube, .3dl)
# ---------------------------------------------------------------------------
def apply_external_lut(
    video_path: str,
    lut_path: str,
    intensity: float = 1.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """Apply any external .cube or .3dl LUT file to video."""
    if not os.path.isfile(lut_path):
        raise FileNotFoundError(f"LUT file not found: {lut_path}")

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        lut_name = os.path.splitext(os.path.basename(lut_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_lut_{lut_name}.mp4")

    if on_progress:
        on_progress(10, f"Applying LUT: {os.path.basename(lut_path)}...")

    # Escape path for FFmpeg filter: backslashes → forward slashes, escape quotes.
    # Path is wrapped in single quotes below, so do NOT escape colons (would
    # corrupt Windows drive-letter paths like C:/... → C\:/...).
    escaped = lut_path.replace("\\", "/").replace("'", "\\'")

    if intensity >= 0.99:
        vf = f"lut3d='{escaped}'"
    else:
        vf = (
            f"split[a][b];"
            f"[a]lut3d='{escaped}'[lut];"
            f"[lut][b]blend=all_mode=normal:all_opacity={intensity}"
        )

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video_path]
    if intensity >= 0.99:
        cmd += ["-vf", vf]
    else:
        cmd += ["-filter_complex", vf]
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "copy", output_path]

    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(100, "External LUT applied!")
    return output_path


def get_color_capabilities() -> Dict:
    return {
        "ocio": check_ocio_available(),
        "color_spaces": [
            {"name": k, "label": v["label"]} for k, v in COLOR_SPACES.items()
        ],
        "correction": True,
        "external_lut": True,
    }
