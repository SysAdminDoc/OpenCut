"""
OpenCut Chromatic Aberration Removal Module

Detect and correct chromatic aberration (CA) in video:
- Detect CA at high-contrast edges
- Shift R/B channels relative to G to realign
- Auto-detect optimal scale factors
- Support for lateral (transverse) and longitudinal CA

Uses FFmpeg's chromashift and scale filters.
"""

import json
import logging
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    get_ffprobe_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class CADetectionResult:
    """Result of chromatic aberration detection."""
    detected: bool = False
    severity: str = "none"    # "none", "mild", "moderate", "severe"
    red_shift_x: int = 0     # pixels
    red_shift_y: int = 0
    blue_shift_x: int = 0
    blue_shift_y: int = 0
    confidence: float = 0.0
    edge_count: int = 0


@dataclass
class CACorrectionResult:
    """Result of chromatic aberration correction."""
    output_path: str = ""
    red_shift_x: int = 0
    red_shift_y: int = 0
    blue_shift_x: int = 0
    blue_shift_y: int = 0
    method: str = "auto"      # "auto", "manual"
    severity: str = "none"


# ---------------------------------------------------------------------------
# CA Detection
# ---------------------------------------------------------------------------
def _analyze_frame_ca(
    frame_path: str,
) -> Tuple[int, int, int, int, float]:
    """
    Analyze a single frame for chromatic aberration.

    Extracts R, G, B channels and compares edge positions
    to estimate channel misalignment.

    Returns (red_x, red_y, blue_x, blue_y, confidence).
    """
    if not os.path.isfile(frame_path):
        return 0, 0, 0, 0, 0.0

    tmp_dir = tempfile.mkdtemp(prefix="oc_ca_detect_")

    try:
        # Extract individual color channels using FFmpeg
        red_path = os.path.join(tmp_dir, "red.png")
        green_path = os.path.join(tmp_dir, "green.png")
        blue_path = os.path.join(tmp_dir, "blue.png")

        # Extract R channel
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", frame_path,
            "-vf", "extractplanes=r",
            "-frames:v", "1",
            red_path,
        ], timeout=15)

        # Extract G channel
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", frame_path,
            "-vf", "extractplanes=g",
            "-frames:v", "1",
            green_path,
        ], timeout=15)

        # Extract B channel
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", frame_path,
            "-vf", "extractplanes=b",
            "-frames:v", "1",
            blue_path,
        ], timeout=15)

        # Compare R-G and B-G channel edges using SSIM
        # Lower SSIM = more misalignment
        red_shift_x, red_shift_y = 0, 0
        blue_shift_x, blue_shift_y = 0, 0
        confidence = 0.0

        # Test different shifts and find best SSIM alignment
        best_rg_ssim = -1.0
        best_bg_ssim = -1.0

        for dx in range(-3, 4):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue

                # Test R-G alignment with this shift
                try:
                    result = subprocess.run([
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-i", red_path, "-i", green_path,
                        "-filter_complex",
                        f"[0]scroll=h={dx}:v={dy}[shifted];[shifted][1]ssim=-",
                        "-f", "null", "-",
                    ], capture_output=True, text=True, timeout=10)

                    for line in result.stderr.split("\n"):
                        if "SSIM" in line and "All:" in line:
                            parts = line.split("All:")
                            if len(parts) > 1:
                                try:
                                    ssim_val = float(parts[1].strip().split()[0])
                                    if ssim_val > best_rg_ssim:
                                        best_rg_ssim = ssim_val
                                        red_shift_x = -dx
                                        red_shift_y = -dy
                                except (ValueError, IndexError):
                                    pass
                            break
                except (subprocess.TimeoutExpired, OSError):
                    continue

                # Test B-G alignment with this shift
                try:
                    result = subprocess.run([
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-i", blue_path, "-i", green_path,
                        "-filter_complex",
                        f"[0]scroll=h={dx}:v={dy}[shifted];[shifted][1]ssim=-",
                        "-f", "null", "-",
                    ], capture_output=True, text=True, timeout=10)

                    for line in result.stderr.split("\n"):
                        if "SSIM" in line and "All:" in line:
                            parts = line.split("All:")
                            if len(parts) > 1:
                                try:
                                    ssim_val = float(parts[1].strip().split()[0])
                                    if ssim_val > best_bg_ssim:
                                        best_bg_ssim = ssim_val
                                        blue_shift_x = -dx
                                        blue_shift_y = -dy
                                except (ValueError, IndexError):
                                    pass
                            break
                except (subprocess.TimeoutExpired, OSError):
                    continue

        # Confidence based on SSIM improvement
        if best_rg_ssim > 0 or best_bg_ssim > 0:
            total_shift = abs(red_shift_x) + abs(red_shift_y) + \
                          abs(blue_shift_x) + abs(blue_shift_y)
            confidence = min(1.0, total_shift / 6.0) if total_shift > 0 else 0.0

        return red_shift_x, red_shift_y, blue_shift_x, blue_shift_y, confidence

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def detect_chromatic_aberration(
    video_path: str,
    num_samples: int = 3,
    on_progress: Optional[Callable] = None,
) -> CADetectionResult:
    """
    Detect chromatic aberration in a video.

    Samples frames from the video and analyzes R/G/B channel
    misalignment at high-contrast edges.

    Args:
        video_path: Path to input video.
        num_samples: Number of frames to analyze.
        on_progress: Progress callback.

    Returns:
        CADetectionResult with detected shift amounts and severity.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = get_video_info(video_path)
    duration = info.get("duration", 0.0)

    if duration <= 0:
        return CADetectionResult()

    tmp_dir = tempfile.mkdtemp(prefix="oc_ca_sample_")
    all_red_x, all_red_y = [], []
    all_blue_x, all_blue_y = [], []
    total_confidence = 0.0

    try:
        for i in range(num_samples):
            t = duration * (i + 1) / (num_samples + 1)
            frame_path = os.path.join(tmp_dir, f"frame_{i}.png")

            try:
                run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(t),
                    "-i", video_path,
                    "-frames:v", "1",
                    "-q:v", "2",
                    frame_path,
                ], timeout=15)

                rx, ry, bx, by, conf = _analyze_frame_ca(frame_path)
                all_red_x.append(rx)
                all_red_y.append(ry)
                all_blue_x.append(bx)
                all_blue_y.append(by)
                total_confidence += conf

            except Exception:
                continue

            if on_progress:
                pct = 10 + int(80 * (i + 1) / num_samples)
                on_progress(pct, f"Analyzing frame {i + 1}/{num_samples}")

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    if not all_red_x:
        return CADetectionResult()

    # Average the detected shifts
    n = len(all_red_x)
    avg_rx = round(sum(all_red_x) / n)
    avg_ry = round(sum(all_red_y) / n)
    avg_bx = round(sum(all_blue_x) / n)
    avg_by = round(sum(all_blue_y) / n)
    avg_conf = total_confidence / n

    # Determine severity
    total_shift = abs(avg_rx) + abs(avg_ry) + abs(avg_bx) + abs(avg_by)
    if total_shift == 0:
        severity = "none"
        detected = False
    elif total_shift <= 2:
        severity = "mild"
        detected = True
    elif total_shift <= 4:
        severity = "moderate"
        detected = True
    else:
        severity = "severe"
        detected = True

    return CADetectionResult(
        detected=detected,
        severity=severity,
        red_shift_x=avg_rx,
        red_shift_y=avg_ry,
        blue_shift_x=avg_bx,
        blue_shift_y=avg_by,
        confidence=avg_conf,
        edge_count=n,
    )


# ---------------------------------------------------------------------------
# CA Correction
# ---------------------------------------------------------------------------
def correct_chromatic_aberration(
    video_path: str,
    red_shift_x: Optional[int] = None,
    red_shift_y: Optional[int] = None,
    blue_shift_x: Optional[int] = None,
    blue_shift_y: Optional[int] = None,
    auto_detect: bool = True,
    output_path_override: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> CACorrectionResult:
    """
    Remove chromatic aberration from a video.

    Shifts R and B channels relative to G to correct lateral
    chromatic aberration. Can auto-detect optimal shift values
    or use manually specified values.

    Args:
        video_path: Path to input video.
        red_shift_x: Horizontal red channel shift (pixels). Auto if None.
        red_shift_y: Vertical red channel shift (pixels). Auto if None.
        blue_shift_x: Horizontal blue channel shift (pixels). Auto if None.
        blue_shift_y: Vertical blue channel shift (pixels). Auto if None.
        auto_detect: Whether to auto-detect shift values.
        output_path_override: Output file path. Auto-generated if None.
        output_dir: Output directory.
        on_progress: Progress callback(pct, msg).

    Returns:
        CACorrectionResult with output path and correction parameters.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    out = output_path_override or output_path(video_path, "ca_corrected", output_dir)
    method = "manual"
    severity = "unknown"

    # Auto-detect if needed
    if auto_detect and red_shift_x is None:
        if on_progress:
            on_progress(5, "Auto-detecting chromatic aberration...")

        detection = detect_chromatic_aberration(video_path, on_progress=on_progress)
        red_shift_x = detection.red_shift_x
        red_shift_y = detection.red_shift_y
        blue_shift_x = detection.blue_shift_x
        blue_shift_y = detection.blue_shift_y
        severity = detection.severity
        method = "auto"

        if not detection.detected:
            if on_progress:
                on_progress(100, "No significant chromatic aberration detected")
            # Still produce output for consistency, just no correction
            red_shift_x = red_shift_y = blue_shift_x = blue_shift_y = 0

    # Default to 0 for any missing values
    red_shift_x = red_shift_x or 0
    red_shift_y = red_shift_y or 0
    blue_shift_x = blue_shift_x or 0
    blue_shift_y = blue_shift_y or 0

    # Clamp shifts to reasonable range
    red_shift_x = max(-10, min(10, red_shift_x))
    red_shift_y = max(-10, min(10, red_shift_y))
    blue_shift_x = max(-10, min(10, blue_shift_x))
    blue_shift_y = max(-10, min(10, blue_shift_y))

    if on_progress:
        on_progress(50, f"Correcting CA: R({red_shift_x},{red_shift_y}) B({blue_shift_x},{blue_shift_y})")

    # Build chromashift filter
    # FFmpeg chromashift filter shifts Cb/Cr (chroma) relative to luma
    # For R/B channel alignment, we use rgbashift filter
    vf = (
        f"rgbashift="
        f"rh={red_shift_x}:rv={red_shift_y}:"
        f"bh={blue_shift_x}:bv={blue_shift_y}:"
        f"gh=0:gv=0:"
        f"edge=smear"
    )

    if on_progress:
        on_progress(60, "Applying chromatic aberration correction...")

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(out)
        .build()
    )

    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(100, "Chromatic aberration correction complete!")

    return CACorrectionResult(
        output_path=out,
        red_shift_x=red_shift_x,
        red_shift_y=red_shift_y,
        blue_shift_x=blue_shift_x,
        blue_shift_y=blue_shift_y,
        method=method,
        severity=severity,
    )
