"""
OpenCut ProRes Export Module v1.0.0

Provides ProRes encoding on Windows (and other platforms) using
the ``prores_ks`` FFmpeg encoder.  Supports all ProRes profiles
from Proxy through 4444 XQ, with automatic alpha channel handling
for 4444/4444 XQ profiles.

Profiles:
  - proxy    (profile 0): Lightweight editing proxy
  - lt       (profile 1): Light, reduced data rate
  - 422      (profile 2): Standard production codec
  - 422hq    (profile 3): High quality, default choice
  - 4444     (profile 4): Mastering with alpha support
  - 4444xq   (profile 5): Highest quality, alpha + extended range
"""

import logging
import os
import subprocess as _sp
import time
from typing import Callable, Dict, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

PRORES_PROFILES: Dict[str, Dict] = {
    "proxy": {
        "label": "ProRes Proxy",
        "profile_id": 0,
        "pix_fmt": "yuv422p10le",
        "description": "Lightweight offline/proxy editing",
    },
    "lt": {
        "label": "ProRes LT",
        "profile_id": 1,
        "pix_fmt": "yuv422p10le",
        "description": "Light — reduced data rate for editing",
    },
    "422": {
        "label": "ProRes 422",
        "profile_id": 2,
        "pix_fmt": "yuv422p10le",
        "description": "Standard production codec",
    },
    "422hq": {
        "label": "ProRes 422 HQ",
        "profile_id": 3,
        "pix_fmt": "yuv422p10le",
        "description": "High quality for finishing and grading",
    },
    "4444": {
        "label": "ProRes 4444",
        "profile_id": 4,
        "pix_fmt": "yuva444p10le",
        "description": "Mastering quality with alpha channel support",
    },
    "4444xq": {
        "label": "ProRes 4444 XQ",
        "profile_id": 5,
        "pix_fmt": "yuva444p10le",
        "description": "Highest quality — extended dynamic range with alpha",
    },
}


# ---------------------------------------------------------------------------
# Encoder availability detection
# ---------------------------------------------------------------------------

_encoder_available: Optional[bool] = None


def detect_prores_encoder() -> bool:
    """Check whether the ``prores_ks`` encoder is available in FFmpeg.

    The result is cached after the first call.

    Returns:
        True if ``prores_ks`` is available.
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
        _encoder_available = "prores_ks" in result.stdout
    except (OSError, _sp.TimeoutExpired):
        _encoder_available = False

    return _encoder_available


def get_prores_profiles() -> list:
    """Return available ProRes profiles for UI display."""
    available = detect_prores_encoder()
    profiles = []
    for key, info in PRORES_PROFILES.items():
        profiles.append({
            "name": key,
            "label": info["label"],
            "profile_id": info["profile_id"],
            "description": info["description"],
            "supports_alpha": key in ("4444", "4444xq"),
            "encoder_available": available,
        })
    return profiles


# ---------------------------------------------------------------------------
# Alpha channel detection
# ---------------------------------------------------------------------------

def _has_alpha_channel(input_path: str) -> bool:
    """Detect whether the input file has an alpha channel via ffprobe."""
    ffprobe = get_ffprobe_path()
    try:
        result = _sp.run(
            [ffprobe, "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt",
             "-of", "csv=p=0", input_path],
            capture_output=True, text=True, timeout=10,
        )
        pix_fmt = result.stdout.strip()
        return "a" in pix_fmt and ("rgba" in pix_fmt or "yuva" in pix_fmt
                                    or "gbrap" in pix_fmt or "argb" in pix_fmt
                                    or "bgra" in pix_fmt or "abgr" in pix_fmt)
    except (OSError, _sp.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Export function
# ---------------------------------------------------------------------------

def export_prores(
    input_path: str,
    profile: str = "422hq",
    output_path_override: Optional[str] = None,
    include_alpha: bool = False,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Export a video file in Apple ProRes format.

    Args:
        input_path: Path to the source video file.
        profile: ProRes profile name — one of ``proxy``, ``lt``, ``422``,
            ``422hq``, ``4444``, ``4444xq``.  Defaults to ``422hq``.
        output_path_override: Explicit output path.  If *None*, auto-generated
            from *input_path* with a ``_prores_{profile}`` suffix.
        include_alpha: For 4444/4444 XQ profiles, include alpha channel if
            present in the source.  Ignored for 422-family profiles.
        on_progress: Optional callback ``(percent, message)`` for progress.

    Returns:
        Dict with ``output_path``, ``profile``, ``profile_id``, ``encoder``,
        ``encode_time_seconds``, ``file_size_mb``, ``has_alpha``.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        ValueError: If *profile* is not recognized.
        RuntimeError: If the ``prores_ks`` encoder is not available or
            FFmpeg fails.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    profile = profile.lower().strip()
    if profile not in PRORES_PROFILES:
        raise ValueError(
            f"Unknown ProRes profile: {profile}. "
            f"Valid profiles: {', '.join(PRORES_PROFILES.keys())}"
        )

    if not detect_prores_encoder():
        raise RuntimeError(
            "ProRes encoder (prores_ks) is not available in your FFmpeg build."
        )

    prof_info = PRORES_PROFILES[profile]
    profile_id = prof_info["profile_id"]
    supports_alpha = profile in ("4444", "4444xq")

    if on_progress:
        on_progress(5, f"Preparing ProRes {prof_info['label']} export...")

    # Determine alpha handling
    has_alpha = False
    if supports_alpha and include_alpha:
        if on_progress:
            on_progress(8, "Checking source for alpha channel...")
        has_alpha = _has_alpha_channel(input_path)

    # Determine pixel format
    if supports_alpha and has_alpha:
        pix_fmt = "yuva444p10le"
    elif supports_alpha:
        # 4444 without alpha still uses yuv444p10le
        pix_fmt = "yuv444p10le"
    else:
        pix_fmt = prof_info["pix_fmt"]

    # Determine output path
    if output_path_override:
        out = output_path_override
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = os.path.dirname(input_path)
        out = os.path.join(directory, f"{base}_prores_{profile}.mov")

    if on_progress:
        on_progress(10, f"Encoding ProRes {prof_info['label']}...")

    # Build FFmpeg command
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-c:v", "prores_ks",
        "-profile:v", str(profile_id),
        "-pix_fmt", pix_fmt,
        "-c:a", "pcm_s16le",
        out,
    ]

    start_time = time.time()
    run_ffmpeg(cmd, timeout=7200)
    encode_time = time.time() - start_time

    file_size_bytes = os.path.getsize(out) if os.path.exists(out) else 0
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    if on_progress:
        on_progress(100, f"ProRes {prof_info['label']} export complete")

    return {
        "output_path": out,
        "profile": profile,
        "profile_id": profile_id,
        "encoder": "prores_ks",
        "encode_time_seconds": round(encode_time, 2),
        "file_size_mb": file_size_mb,
        "has_alpha": has_alpha,
    }
