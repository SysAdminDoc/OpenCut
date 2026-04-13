"""
OpenCut Color Space Auto-Detection & Conversion Module

Detect and convert video color spaces (primaries, transfer characteristics,
matrix coefficients) via FFmpeg probing and the colorspace/zscale filters.
"""

import json
import logging
import os
import subprocess as _sp
from dataclasses import dataclass
from typing import Callable, List, Optional

from opencut.helpers import get_ffprobe_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Known color space profiles
# ---------------------------------------------------------------------------

KNOWN_PROFILES = {
    ("bt709", "bt709", "bt709"): "bt709_sdr",
    ("bt2020nc", "arib-std-b67", "bt2020"): "bt2020_hlg",
    ("bt2020nc", "smpte2084", "bt2020"): "bt2020_pq",
    ("bt709", "iec61966-2-1", "bt709"): "srgb",
    ("bt709", "smpte428", "bt709"): "dci_p3",
}


# ---------------------------------------------------------------------------
# ColorSpaceInfo dataclass
# ---------------------------------------------------------------------------

@dataclass
class ColorSpaceInfo:
    """Describes the color space characteristics of a video file."""
    primaries: str = "unknown"
    transfer: str = "unknown"
    matrix: str = "unknown"
    bit_depth: int = 8
    is_hdr: bool = False
    is_wide_gamut: bool = False
    profile_name: str = "unknown"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_colorspace(input_path: str) -> ColorSpaceInfo:
    """
    Probe a video file for color space metadata.

    Reads color_primaries, color_trc, colorspace, and bits_per_raw_sample
    from FFmpeg and maps them to a known profile name.

    Args:
        input_path: Path to the video file.

    Returns:
        ColorSpaceInfo dataclass.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=color_primaries,color_transfer,color_space,bits_per_raw_sample,pix_fmt",
        "-of", "json",
        input_path,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    info = ColorSpaceInfo()

    if result.returncode != 0:
        logger.warning("ffprobe colorspace detection failed for %s", input_path)
        return info

    try:
        data = json.loads(result.stdout.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return info

    streams = data.get("streams", [])
    if not streams:
        return info

    stream = streams[0]

    info.primaries = stream.get("color_primaries", "unknown") or "unknown"
    info.transfer = stream.get("color_transfer", "unknown") or "unknown"
    info.matrix = stream.get("color_space", "unknown") or "unknown"

    # Bit depth from bits_per_raw_sample or pixel format
    bits_raw = stream.get("bits_per_raw_sample")
    if bits_raw and str(bits_raw).isdigit():
        info.bit_depth = int(bits_raw)
    else:
        pix_fmt = stream.get("pix_fmt", "")
        if "10" in pix_fmt or "10le" in pix_fmt or "10be" in pix_fmt:
            info.bit_depth = 10
        elif "12" in pix_fmt:
            info.bit_depth = 12
        else:
            info.bit_depth = 8

    # HDR detection
    transfer_lower = info.transfer.lower()
    if "smpte2084" in transfer_lower or "st2084" in transfer_lower:
        info.is_hdr = True
    elif "arib-std-b67" in transfer_lower or "hlg" in transfer_lower:
        info.is_hdr = True

    # Wide gamut detection
    primaries_lower = info.primaries.lower()
    if "bt2020" in primaries_lower or "2020" in primaries_lower:
        info.is_wide_gamut = True
    elif "p3" in primaries_lower or "dci" in primaries_lower:
        info.is_wide_gamut = True

    # Profile name matching
    key = (info.matrix.lower(), info.transfer.lower(), info.primaries.lower())
    info.profile_name = KNOWN_PROFILES.get(key, "unknown")

    # Fallback profile detection for common cases
    if info.profile_name == "unknown":
        if info.is_hdr and "bt2020" in primaries_lower:
            if "smpte2084" in transfer_lower or "st2084" in transfer_lower:
                info.profile_name = "bt2020_pq"
            elif "arib-std-b67" in transfer_lower or "hlg" in transfer_lower:
                info.profile_name = "bt2020_hlg"
        elif "bt709" in primaries_lower and not info.is_hdr:
            info.profile_name = "bt709_sdr"

    return info


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_colorspace(
    input_path: str,
    target_primaries: str = "bt709",
    target_transfer: str = "bt709",
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Convert a video's color space to the specified target.

    Uses FFmpeg's colorspace filter for matrix/primaries/transfer conversion,
    with zscale fallback for HDR-to-SDR scenarios.

    Args:
        input_path: Path to the input video.
        target_primaries: Target color primaries (e.g. "bt709", "bt2020").
        target_transfer: Target transfer characteristics (e.g. "bt709", "smpte2084").
        output_path_override: Custom output path (auto-generated if None).
        on_progress: Progress callback (percent, message).

    Returns:
        dict with output_path and source/target color space info.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if on_progress:
        on_progress(5, "Detecting source color space...")

    source_info = detect_colorspace(input_path)
    out = output_path_override or output_path(input_path, f"cs_{target_primaries}")

    if on_progress:
        on_progress(15, f"Converting from {source_info.profile_name} to {target_primaries}/{target_transfer}...")

    # Determine if we need zscale (for HDR/wide-gamut) or colorspace filter
    source_is_hdr = source_info.is_hdr
    target_is_hdr = "smpte2084" in target_transfer or "hlg" in target_transfer or "2084" in target_transfer

    if source_is_hdr and not target_is_hdr:
        # HDR to SDR: use zscale + tonemap
        vf = (
            f"zscale=t=linear:npl=100,"
            f"tonemap=hable,"
            f"zscale=t={target_transfer}:m={target_primaries}:p={target_primaries}:r=tv,"
            f"format=yuv420p"
        )
    elif not source_is_hdr and target_is_hdr:
        # SDR to HDR: use zscale
        vf = (
            f"zscale=t={target_transfer}:m={target_primaries}:p={target_primaries},"
            f"format=yuv420p10le"
        )
    else:
        # Same domain conversion (SDR-to-SDR or HDR-to-HDR)
        vf = (
            f"colorspace=all={target_primaries}"
            f":trc={target_transfer}"
            f":primaries={target_primaries},"
            f"format=yuv420p"
        )

    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "copy",
        out,
    ], timeout=7200)

    if on_progress:
        on_progress(100, "Color space conversion complete!")

    return {
        "output_path": out,
        "source_primaries": source_info.primaries,
        "source_transfer": source_info.transfer,
        "source_profile": source_info.profile_name,
        "target_primaries": target_primaries,
        "target_transfer": target_transfer,
    }


# ---------------------------------------------------------------------------
# Batch Detection
# ---------------------------------------------------------------------------

def batch_detect_colorspace(
    file_paths: list,
    on_progress: Optional[Callable] = None,
) -> List[ColorSpaceInfo]:
    """
    Detect color space for multiple files.

    Args:
        file_paths: List of video file paths.
        on_progress: Progress callback (percent, message).

    Returns:
        List of ColorSpaceInfo, one per file (in order).
    """
    if not file_paths:
        return []

    results = []
    total = len(file_paths)

    for i, fp in enumerate(file_paths):
        if on_progress:
            pct = int((i / total) * 100)
            on_progress(pct, f"Detecting colorspace {i + 1}/{total}...")

        try:
            info = detect_colorspace(fp)
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning("Colorspace detection failed for %s: %s", fp, e)
            info = ColorSpaceInfo()
            info.profile_name = "error"

        results.append(info)

    if on_progress:
        on_progress(100, f"Detected colorspace for {total} file(s)")

    return results
