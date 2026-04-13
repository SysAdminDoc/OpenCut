"""
OpenCut HDR/SDR Tone Mapping Module

Detect HDR content and convert to SDR using industry-standard tone mapping
algorithms (Hable, Reinhard, Mobius, Linear) via FFmpeg zscale + tonemap filters.
"""

import json
import logging
import os
import subprocess as _sp
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import get_ffprobe_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# HDR Info dataclass
# ---------------------------------------------------------------------------

VALID_TONEMAP_ALGORITHMS = ("hable", "reinhard", "mobius", "linear")


@dataclass
class HDRInfo:
    """Describes the HDR characteristics of a video file."""
    is_hdr: bool = False
    transfer: str = "sdr"        # "pq", "hlg", "sdr"
    primaries: str = "bt709"     # "bt2020", "bt709"
    max_cll: int = 0
    max_fall: int = 0


# ---------------------------------------------------------------------------
# HDR Detection
# ---------------------------------------------------------------------------

def detect_hdr(input_path: str) -> HDRInfo:
    """
    Probe a video file for HDR metadata.

    Returns an HDRInfo dataclass with is_hdr, transfer function,
    color primaries, MaxCLL and MaxFALL values.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=color_transfer,color_primaries",
        "-show_entries",
        "stream_side_data=max_content,max_average",
        "-of", "json",
        input_path,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    info = HDRInfo()

    if result.returncode != 0:
        logger.warning("ffprobe HDR detection failed for %s", input_path)
        return info

    try:
        data = json.loads(result.stdout.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return info

    streams = data.get("streams", [])
    if not streams:
        return info

    stream = streams[0]
    color_transfer = stream.get("color_transfer", "").lower()
    color_primaries = stream.get("color_primaries", "").lower()

    # Determine transfer function
    if "smpte2084" in color_transfer or "st2084" in color_transfer:
        info.transfer = "pq"
        info.is_hdr = True
    elif "arib-std-b67" in color_transfer or "hlg" in color_transfer:
        info.transfer = "hlg"
        info.is_hdr = True
    else:
        info.transfer = "sdr"

    # Determine primaries
    if "bt2020" in color_primaries or "2020" in color_primaries:
        info.primaries = "bt2020"
        # BT.2020 primaries with non-SDR transfer is definitely HDR
        if info.transfer != "sdr":
            info.is_hdr = True
    else:
        info.primaries = "bt709"

    # Extract MaxCLL / MaxFALL from side data
    side_data_list = stream.get("side_data_list", [])
    for sd in side_data_list:
        if "max_content" in sd:
            try:
                info.max_cll = int(sd["max_content"])
            except (ValueError, TypeError):
                pass
        if "max_average" in sd:
            try:
                info.max_fall = int(sd["max_average"])
            except (ValueError, TypeError):
                pass

    return info


# ---------------------------------------------------------------------------
# HDR to SDR Tone Mapping
# ---------------------------------------------------------------------------

def tonemap_hdr_to_sdr(
    input_path: str,
    algorithm: str = "hable",
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Convert an HDR video to SDR using the specified tone mapping algorithm.

    Args:
        input_path: Path to the HDR video file.
        algorithm: Tone mapping algorithm - "hable", "reinhard", "mobius", or "linear".
        output_path_override: Custom output path (auto-generated if None).
        on_progress: Progress callback (percent, message).

    Returns:
        dict with output_path, algorithm, and HDR info.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if algorithm not in VALID_TONEMAP_ALGORITHMS:
        raise ValueError(
            f"Invalid algorithm '{algorithm}'. Must be one of: {', '.join(VALID_TONEMAP_ALGORITHMS)}"
        )

    if on_progress:
        on_progress(5, "Detecting HDR metadata...")

    hdr_info = detect_hdr(input_path)

    out = output_path_override or output_path(input_path, f"tonemapped_{algorithm}")

    if on_progress:
        on_progress(10, f"Tone mapping with {algorithm}...")

    # Build the zscale + tonemap filter chain
    vf = (
        f"zscale=t=linear:npl=100,"
        f"tonemap={algorithm},"
        f"zscale=t=bt709:m=bt709:r=tv,"
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
        on_progress(100, "Tone mapping complete!")

    return {
        "output_path": out,
        "algorithm": algorithm,
        "source_is_hdr": hdr_info.is_hdr,
        "source_transfer": hdr_info.transfer,
        "source_primaries": hdr_info.primaries,
    }


# ---------------------------------------------------------------------------
# Detect and Suggest
# ---------------------------------------------------------------------------

def detect_and_suggest(input_path: str) -> dict:
    """
    Detect HDR status and suggest an appropriate action.

    Returns dict with is_hdr, transfer, primaries, and suggested_action.
    """
    info = detect_hdr(input_path)

    if info.is_hdr:
        suggested_action = (
            "tonemap_to_sdr"
            if info.transfer in ("pq", "hlg")
            else "review_manually"
        )
    else:
        suggested_action = "none"

    return {
        "is_hdr": info.is_hdr,
        "transfer": info.transfer,
        "primaries": info.primaries,
        "max_cll": info.max_cll,
        "max_fall": info.max_fall,
        "suggested_action": suggested_action,
    }
