"""
OpenCut AI Deinterlacing

Detects interlaced content and applies high-quality deinterlacing
using FFmpeg's yadif, bwdif, or nnedi filters.

Uses FFmpeg only — no additional dependencies required.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_ffprobe_path,
    get_video_info,
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


# ---------------------------------------------------------------------------
# Auto-Detect Interlacing (enhanced via ffprobe + idet)
# ---------------------------------------------------------------------------

@dataclass
class DetailedInterlaceInfo:
    """Extended interlace detection results."""
    is_interlaced: bool = False
    field_order: str = "unknown"       # "tff", "bff", "progressive", "unknown"
    detection_confidence: float = 0.0
    probe_field_order: str = "unknown"  # Raw ffprobe field_order value
    idet_tff: int = 0
    idet_bff: int = 0
    idet_progressive: int = 0
    idet_undetermined: int = 0
    recommended_method: str = "bwdif"


def auto_detect_interlacing(video_path: str) -> DetailedInterlaceInfo:
    """
    Detect interlacing using both ffprobe ``field_order`` metadata and
    the ``idet`` analysis filter for robust, two-phase detection.

    Phase 1 reads the container-level ``field_order`` tag (fast, metadata
    only).  Phase 2 runs the ``idet`` filter on the first 500 frames for
    statistical frame-level analysis.  The two results are combined for
    a high-confidence determination.

    Args:
        video_path: Source video file.

    Returns:
        DetailedInterlaceInfo with combined results and recommended
        deinterlacing method.
    """
    result = DetailedInterlaceInfo()

    # Phase 1: ffprobe field_order metadata
    probe_cmd = [
        get_ffprobe_path(), "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=field_order",
        "-of", "json",
        video_path,
    ]
    try:
        probe_out = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if probe_out.returncode == 0:
            probe_data = json.loads(probe_out.stdout)
            streams = probe_data.get("streams", [])
            if streams:
                fo = streams[0].get("field_order", "unknown")
                result.probe_field_order = fo
                if fo in ("tt", "tb"):
                    result.field_order = "tff"
                elif fo in ("bb", "bt"):
                    result.field_order = "bff"
                elif fo == "progressive":
                    result.field_order = "progressive"
    except Exception:
        pass

    # Phase 2: idet filter analysis
    idet_info = detect_interlaced(video_path)
    result.idet_tff = 0
    result.idet_bff = 0
    result.idet_progressive = 0
    result.idet_undetermined = 0

    # Re-run idet to capture raw numbers
    idet_cmd = [
        get_ffmpeg_path(), "-hide_banner",
        "-i", video_path,
        "-vf", "idet",
        "-frames:v", "500",
        "-f", "null", "-",
    ]
    try:
        idet_out = subprocess.run(idet_cmd, capture_output=True, text=True, timeout=120)
        multi_match = re.search(
            r"Multi frame detection:\s+"
            r"TFF:\s*(\d+)\s+BFF:\s*(\d+)\s+"
            r"Progressive:\s*(\d+)\s+Undetermined:\s*(\d+)",
            idet_out.stderr,
        )
        if multi_match:
            result.idet_tff = int(multi_match.group(1))
            result.idet_bff = int(multi_match.group(2))
            result.idet_progressive = int(multi_match.group(3))
            result.idet_undetermined = int(multi_match.group(4))
    except subprocess.TimeoutExpired:
        pass

    # Combine results
    result.is_interlaced = idet_info.is_interlaced
    result.detection_confidence = idet_info.detection_confidence

    # If ffprobe says progressive and idet agrees, boost confidence
    if result.probe_field_order == "progressive" and not idet_info.is_interlaced:
        result.detection_confidence = max(result.detection_confidence, 0.95)
        result.is_interlaced = False
        result.field_order = "progressive"
    elif idet_info.is_interlaced:
        if idet_info.field_order in ("tff", "bff"):
            result.field_order = idet_info.field_order

    # Recommend method based on content
    total = result.idet_tff + result.idet_bff + result.idet_progressive + result.idet_undetermined
    if total > 0:
        interlaced_ratio = (result.idet_tff + result.idet_bff) / total
        if interlaced_ratio > 0.7:
            result.recommended_method = "bwdif"  # Heavy interlacing
        elif interlaced_ratio > 0.3:
            result.recommended_method = "yadif"  # Mixed content
        else:
            result.recommended_method = "yadif"  # Light interlacing

    return result


# ---------------------------------------------------------------------------
# Adaptive Deinterlace
# ---------------------------------------------------------------------------

_ADAPTIVE_METHODS = {"auto", "yadif", "bwdif", "nnedi"}


def adaptive_deinterlace(
    video_path: str,
    method: str = "auto",
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Adaptively deinterlace a video, auto-detecting interlacing type and
    choosing the best method.

    When *method* is ``"auto"`` the function runs
    :func:`auto_detect_interlacing` first and selects yadif (fast) or
    bwdif (quality) based on interlacing density.  If the video is
    already progressive it returns early without re-encoding.

    Args:
        video_path:  Source video file.
        method:  ``"auto"``, ``"yadif"``, ``"bwdif"``, or ``"nnedi"``.
        output_path_override:  Optional output file path.
        on_progress:  Callback ``(pct, msg)``.

    Returns:
        dict with *output_path*, *method_used*, *field_order*,
        *was_interlaced*, and *detection_confidence*.
    """
    if method not in _ADAPTIVE_METHODS:
        method = "auto"

    if on_progress:
        on_progress(5, "Detecting interlacing...")

    detection = auto_detect_interlacing(video_path)

    if method == "auto":
        if not detection.is_interlaced:
            if on_progress:
                on_progress(100, "Video is progressive, no deinterlacing needed")
            return {
                "output_path": video_path,
                "method_used": "none",
                "field_order": detection.field_order,
                "was_interlaced": False,
                "detection_confidence": detection.detection_confidence,
            }
        method = detection.recommended_method

    if on_progress:
        on_progress(15, f"Deinterlacing with {method} (field order: {detection.field_order})...")

    field_order = detection.field_order if detection.field_order in ("tff", "bff") else "tff"

    result = deinterlace(
        video_path,
        output_path_override=output_path_override,
        method=method,
        field_order=field_order,
        on_progress=on_progress,
    )

    return {
        "output_path": result["output_path"],
        "method_used": method,
        "field_order": detection.field_order,
        "was_interlaced": detection.is_interlaced,
        "detection_confidence": detection.detection_confidence,
    }
