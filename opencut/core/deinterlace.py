"""
OpenCut AI Deinterlacing

Detects interlaced content and applies high-quality deinterlacing
using FFmpeg's yadif, bwdif, or nnedi filters.

Uses FFmpeg only — no additional dependencies required.
"""

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class InterlaceInfo:
    """Interlace detection results."""
    is_interlaced: bool = False
    field_order: str = "unknown"   # "tff", "bff", or "unknown"
    detection_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Interlace detection
# ---------------------------------------------------------------------------
def detect_interlaced(input_path: str) -> InterlaceInfo:
    """
    Detect whether a video file contains interlaced content.

    Uses FFmpeg's idet (interlace detection) filter to analyse a sample
    of frames and determine field order and interlace ratio.

    Args:
        input_path: Source video file.

    Returns:
        InterlaceInfo with detection results.
    """
    # Run idet filter on the first ~500 frames for a quick analysis
    cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-i", input_path,
        "-vf", "idet",
        "-frames:v", "500",
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        logger.warning("Interlace detection timed out for %s", input_path)
        return InterlaceInfo()

    stderr = result.stderr

    # Parse idet output:
    # [Parsed_idet_0 ...] Repeated Fields: Neither: 500 Top:    0 Bottom:    0
    # [Parsed_idet_0 ...] Single frame detection: TFF:  240 BFF:    5 Progressive:  255 Undetermined:    0
    # [Parsed_idet_0 ...] Multi frame detection:  TFF:  238 BFF:    3 Progressive:  259 Undetermined:    0

    tff_total = 0
    bff_total = 0
    progressive_total = 0
    undetermined_total = 0

    # Use multi-frame detection as it's more reliable
    multi_match = re.search(
        r"Multi frame detection:\s+"
        r"TFF:\s*(\d+)\s+BFF:\s*(\d+)\s+"
        r"Progressive:\s*(\d+)\s+Undetermined:\s*(\d+)",
        stderr,
    )
    if multi_match:
        tff_total = int(multi_match.group(1))
        bff_total = int(multi_match.group(2))
        progressive_total = int(multi_match.group(3))
        undetermined_total = int(multi_match.group(4))
    else:
        # Fall back to single-frame detection
        single_match = re.search(
            r"Single frame detection:\s+"
            r"TFF:\s*(\d+)\s+BFF:\s*(\d+)\s+"
            r"Progressive:\s*(\d+)\s+Undetermined:\s*(\d+)",
            stderr,
        )
        if single_match:
            tff_total = int(single_match.group(1))
            bff_total = int(single_match.group(2))
            progressive_total = int(single_match.group(3))
            undetermined_total = int(single_match.group(4))

    total_frames = tff_total + bff_total + progressive_total + undetermined_total
    if total_frames == 0:
        return InterlaceInfo()

    interlaced_frames = tff_total + bff_total
    interlaced_ratio = interlaced_frames / total_frames

    # Determine field order
    if tff_total > bff_total:
        field_order = "tff"
    elif bff_total > tff_total:
        field_order = "bff"
    else:
        field_order = "unknown"

    # Consider interlaced if more than 30% of frames are detected as such
    is_interlaced = interlaced_ratio > 0.30

    # Confidence: how clear-cut the detection is
    # High confidence when interlaced ratio is near 0 or near 1
    if is_interlaced:
        confidence = min(1.0, interlaced_ratio / 0.9)
    else:
        confidence = min(1.0, (1.0 - interlaced_ratio) / 0.7)

    return InterlaceInfo(
        is_interlaced=is_interlaced,
        field_order=field_order if is_interlaced else "unknown",
        detection_confidence=round(confidence, 3),
    )


# ---------------------------------------------------------------------------
# Deinterlace
# ---------------------------------------------------------------------------
_VALID_METHODS = {"yadif", "bwdif", "nnedi"}


def deinterlace(
    input_path: str,
    output_path_override: str = None,
    method: str = "bwdif",
    field_order: str = "auto",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Deinterlace a video file.

    Args:
        input_path: Source video file.
        output_path_override: Optional output file path. If None, generates
            one alongside the input file.
        method: Deinterlacing method — "yadif" (fast), "bwdif" (better
            quality, default), or "nnedi" (best quality if available).
        field_order: Field order — "tff" (top field first), "bff" (bottom
            field first), or "auto" (auto-detect).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path of the deinterlaced file.
    """
    if method not in _VALID_METHODS:
        method = "bwdif"

    if on_progress:
        on_progress(5, f"Preparing deinterlace ({method})...")

    # Auto-detect field order if requested
    if field_order == "auto":
        if on_progress:
            on_progress(10, "Auto-detecting field order...")
        info = detect_interlaced(input_path)
        if info.field_order in ("tff", "bff"):
            field_order = info.field_order
        else:
            field_order = "tff"  # safe default
        if on_progress:
            on_progress(20, f"Detected field order: {field_order}")

    # Build the video filter
    if method == "yadif":
        # yadif: mode=0 (send frame), parity based on field order
        parity = 0 if field_order == "tff" else 1
        vf = f"yadif=mode=0:parity={parity}:deint=0"
    elif method == "bwdif":
        # bwdif: mode=send_frame, parity based on field order
        parity = 0 if field_order == "tff" else 1
        vf = f"bwdif=mode=send_frame:parity={parity}:deint=all"
    elif method == "nnedi":
        # nnedi: field_order, deint=all, nsize=s32x6, nns=n64
        field_val = 0 if field_order == "tff" else 1
        vf = f"nnedi=weights=nnedi3_weights.bin:deint=all:field={field_val}:nsize=s32x6:nns=n64"
    else:
        vf = "bwdif"

    out = output_path_override or output_path(input_path, f"deinterlaced_{method}")

    if on_progress:
        on_progress(30, f"Deinterlacing with {method}...")

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(out)
        .build()
    )

    try:
        run_ffmpeg(cmd, timeout=3600)
    except RuntimeError as e:
        # If nnedi fails (missing weights), fall back to bwdif
        if method == "nnedi" and "nnedi3_weights" in str(e):
            logger.warning("nnedi weights not found, falling back to bwdif")
            if on_progress:
                on_progress(35, "nnedi unavailable, falling back to bwdif...")
            parity = 0 if field_order == "tff" else 1
            vf_fallback = f"bwdif=mode=send_frame:parity={parity}:deint=all"
            cmd = (
                FFmpegCmd()
                .input(input_path)
                .video_filter(vf_fallback)
                .video_codec("libx264", crf=18, preset="medium")
                .audio_codec("aac", bitrate="192k")
                .faststart()
                .output(out)
                .build()
            )
            run_ffmpeg(cmd, timeout=3600)
        else:
            raise

    if on_progress:
        on_progress(100, "Deinterlacing complete")

    return {"output_path": out}
