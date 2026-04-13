"""
OpenCut DNxHR/DNxHD Export Module v1.0.0

Provides Avid DNxHR encoding for professional post-production workflows.
Supports all DNxHR profiles (LB through 444) with MOV and MXF container
options.

Profiles:
  - dnxhr_lb:   Low Bandwidth — offline editing
  - dnxhr_sq:   Standard Quality — general editing
  - dnxhr_hq:   High Quality — broadcast finishing
  - dnxhr_hqx:  High Quality 12-bit — HDR/grading
  - dnxhr_444:  4:4:4 — highest quality, full chroma
"""

import logging
import os
import subprocess as _sp
import time
from typing import Callable, Dict, Optional

from opencut.helpers import get_ffmpeg_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

DNXHR_PROFILES: Dict[str, Dict] = {
    "dnxhr_lb": {
        "label": "DNxHR LB",
        "profile_value": "dnxhr_lb",
        "pix_fmt": "yuv422p",
        "description": "Low Bandwidth — lightweight offline/proxy editing",
    },
    "dnxhr_sq": {
        "label": "DNxHR SQ",
        "profile_value": "dnxhr_sq",
        "pix_fmt": "yuv422p",
        "description": "Standard Quality — general purpose editing",
    },
    "dnxhr_hq": {
        "label": "DNxHR HQ",
        "profile_value": "dnxhr_hq",
        "pix_fmt": "yuv422p",
        "description": "High Quality — broadcast and finishing",
    },
    "dnxhr_hqx": {
        "label": "DNxHR HQX",
        "profile_value": "dnxhr_hqx",
        "pix_fmt": "yuv422p10le",
        "description": "High Quality 12-bit — HDR grading workflows",
    },
    "dnxhr_444": {
        "label": "DNxHR 444",
        "profile_value": "dnxhr_444",
        "pix_fmt": "yuv444p10le",
        "description": "4:4:4 full chroma — highest quality mastering",
    },
}

VALID_CONTAINERS = {"mov", "mxf"}


# ---------------------------------------------------------------------------
# Encoder detection
# ---------------------------------------------------------------------------

_encoder_available: Optional[bool] = None


def detect_dnxhd_encoder() -> bool:
    """Check whether the ``dnxhd`` encoder is available in FFmpeg.

    The result is cached after the first call.
    """
    global _encoder_available
    if _encoder_available is not None:
        return _encoder_available

    ffmpeg = get_ffmpeg_path()
    try:
        result = _sp.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        _encoder_available = "dnxhd" in result.stdout
    except (OSError, _sp.TimeoutExpired):
        _encoder_available = False

    return _encoder_available


def get_dnxhr_profiles() -> list:
    """Return available DNxHR profiles for UI display."""
    available = detect_dnxhd_encoder()
    profiles = []
    for key, info in DNXHR_PROFILES.items():
        profiles.append({
            "name": key,
            "label": info["label"],
            "description": info["description"],
            "pix_fmt": info["pix_fmt"],
            "containers": sorted(VALID_CONTAINERS),
            "encoder_available": available,
        })
    return profiles


# ---------------------------------------------------------------------------
# Export function
# ---------------------------------------------------------------------------

def export_dnxhr(
    input_path: str,
    profile: str = "dnxhr_hq",
    container: str = "mov",
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Export a video file in Avid DNxHR format.

    Args:
        input_path: Path to the source video file.
        profile: DNxHR profile — one of ``dnxhr_lb``, ``dnxhr_sq``,
            ``dnxhr_hq``, ``dnxhr_hqx``, ``dnxhr_444``.
        container: Container format — ``mov`` (default) or ``mxf``.
        output_path_override: Explicit output path.  If *None*, auto-generated
            from *input_path*.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        Dict with ``output_path``, ``profile``, ``container``, ``encoder``,
        ``encode_time_seconds``, ``file_size_mb``.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        ValueError: If *profile* or *container* is invalid.
        RuntimeError: If the ``dnxhd`` encoder is not available or
            FFmpeg fails.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    profile = profile.lower().strip()
    if profile not in DNXHR_PROFILES:
        raise ValueError(
            f"Unknown DNxHR profile: {profile}. "
            f"Valid profiles: {', '.join(DNXHR_PROFILES.keys())}"
        )

    container = container.lower().strip()
    if container not in VALID_CONTAINERS:
        raise ValueError(
            f"Unsupported container: {container}. "
            f"Valid containers: {', '.join(sorted(VALID_CONTAINERS))}"
        )

    if not detect_dnxhd_encoder():
        raise RuntimeError(
            "DNxHD/DNxHR encoder is not available in your FFmpeg build."
        )

    prof_info = DNXHR_PROFILES[profile]

    if on_progress:
        on_progress(5, f"Preparing {prof_info['label']} export...")

    # Determine output path
    ext = f".{container}"
    if output_path_override:
        out = output_path_override
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = os.path.dirname(input_path)
        out = os.path.join(directory, f"{base}_{profile}{ext}")

    if on_progress:
        on_progress(10, f"Encoding {prof_info['label']} ({container.upper()})...")

    # Build FFmpeg command
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-c:v", "dnxhd",
        "-profile:v", prof_info["profile_value"],
        "-pix_fmt", prof_info["pix_fmt"],
        "-c:a", "pcm_s16le",
        out,
    ]

    start_time = time.time()
    run_ffmpeg(cmd, timeout=7200)
    encode_time = time.time() - start_time

    file_size_bytes = os.path.getsize(out) if os.path.exists(out) else 0
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    if on_progress:
        on_progress(100, f"{prof_info['label']} export complete")

    return {
        "output_path": out,
        "profile": profile,
        "container": container,
        "encoder": "dnxhd",
        "encode_time_seconds": round(encode_time, 2),
        "file_size_mb": file_size_mb,
    }
