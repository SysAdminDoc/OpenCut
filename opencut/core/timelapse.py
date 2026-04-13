"""
OpenCut Timelapse Deflicker

Analyzes and corrects brightness flicker in timelapse footage using
FFmpeg signalstats for luminance analysis and deflicker/eq filters
for correction.
"""

import json
import logging
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


@dataclass
class FlickerAnalysis:
    """Results from flicker analysis."""
    per_frame_luminance: List[float] = field(default_factory=list)
    flicker_score: float = 0.0  # 0.0 = no flicker, 1.0 = severe
    needs_deflicker: bool = False
    frame_count: int = 0
    avg_luminance: float = 0.0
    min_luminance: float = 0.0
    max_luminance: float = 0.0
    std_dev: float = 0.0


def analyze_flicker(
    input_path: str,
    on_progress: Optional[Callable] = None,
) -> FlickerAnalysis:
    """
    Analyze per-frame brightness variation to detect flicker.

    Uses FFmpeg signalstats filter to extract YAVG (average luma) per frame.

    Args:
        input_path: Source video file.
        on_progress: Progress callback(pct, msg).

    Returns:
        FlickerAnalysis with per-frame luminance and flicker score.
    """
    if on_progress:
        on_progress(5, "Analyzing video luminance...")

    ffmpeg = get_ffmpeg_path()

    # Extract per-frame average luminance via signalstats
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "info",
        "-i", input_path,
        "-vf", "signalstats=stat=tout+vrep+brng",
        "-f", "null", "-",
    ]

    if on_progress:
        on_progress(10, "Running signalstats analysis...")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if on_progress:
        on_progress(70, "Parsing luminance data...")

    # Parse YAVG values from stderr
    yavg_re = re.compile(r"YAVG:\s*(\d+(?:\.\d+)?)")
    luminance_values: List[float] = []

    for line in result.stderr.splitlines():
        match = yavg_re.search(line)
        if match:
            luminance_values.append(float(match.group(1)))

    if not luminance_values:
        # Fallback: try lavfi.signalstats.YAVG metadata
        yavg_meta_re = re.compile(r"lavfi\.signalstats\.YAVG=(\d+(?:\.\d+)?)")
        for line in result.stderr.splitlines():
            match = yavg_meta_re.search(line)
            if match:
                luminance_values.append(float(match.group(1)))

    if not luminance_values:
        logger.warning("No luminance data extracted from %s", input_path)
        return FlickerAnalysis(
            per_frame_luminance=[],
            flicker_score=0.0,
            needs_deflicker=False,
            frame_count=0,
        )

    if on_progress:
        on_progress(85, "Computing flicker metrics...")

    # Compute statistics
    n = len(luminance_values)
    avg_lum = sum(luminance_values) / n
    min_lum = min(luminance_values)
    max_lum = max(luminance_values)

    # Standard deviation
    variance = sum((v - avg_lum) ** 2 for v in luminance_values) / max(1, n)
    std_dev = math.sqrt(variance)

    # Flicker score: normalized standard deviation
    # Typical YAVG range is 0-255. We normalize std_dev against the mean.
    # A coefficient of variation > 0.1 is noticeable flicker
    if avg_lum > 0:
        cv = std_dev / avg_lum  # coefficient of variation
    else:
        cv = 0.0

    # Map CV to 0-1 score (CV of 0.05 = 0.25 score, CV of 0.2 = 1.0)
    flicker_score = min(1.0, max(0.0, cv / 0.2))
    needs_deflicker = flicker_score > 0.15

    if on_progress:
        on_progress(100, f"Flicker score: {flicker_score:.2f}")

    return FlickerAnalysis(
        per_frame_luminance=luminance_values,
        flicker_score=round(flicker_score, 4),
        needs_deflicker=needs_deflicker,
        frame_count=n,
        avg_luminance=round(avg_lum, 2),
        min_luminance=round(min_lum, 2),
        max_luminance=round(max_lum, 2),
        std_dev=round(std_dev, 4),
    )


def _rolling_average(values: List[float], window: int) -> List[float]:
    """Compute a centered rolling average of a list of floats."""
    n = len(values)
    result = []
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_vals = values[start:end]
        result.append(sum(window_vals) / len(window_vals))
    return result


def deflicker(
    input_path: str,
    output_path_str: Optional[str] = None,
    window_size: int = 15,
    strength: float = 0.8,
    method: str = "auto",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Remove flicker from timelapse footage.

    Args:
        input_path: Source video file.
        output_path_str: Destination path. Auto-generated if None.
        window_size: Rolling average window for luminance smoothing.
        strength: Correction strength 0.0-1.0.
        method: "simple" (FFmpeg deflicker), "smooth" (per-frame brightness),
                or "auto" (choose based on flicker severity).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, method used, and correction stats.
    """
    if on_progress:
        on_progress(5, "Analyzing flicker...")

    strength = max(0.0, min(1.0, float(strength)))
    window_size = max(3, min(99, int(window_size)))
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window

    # Auto-detect method
    if method == "auto":
        analysis = analyze_flicker(input_path, on_progress=None)
        if analysis.flicker_score < 0.3:
            method = "simple"
        else:
            method = "smooth"
        if on_progress:
            on_progress(15, f"Auto-selected method: {method} (score={analysis.flicker_score:.2f})")
    elif method not in ("simple", "smooth"):
        method = "simple"

    # Generate output path
    if output_path_str is None:
        output_path_str = output_path(input_path, "deflickered")

    ffmpeg = get_ffmpeg_path()

    if method == "simple":
        return _deflicker_simple(input_path, output_path_str, window_size,
                                 strength, ffmpeg, on_progress)
    else:
        return _deflicker_smooth(input_path, output_path_str, window_size,
                                 strength, ffmpeg, on_progress)


def _deflicker_simple(
    input_path: str,
    output_path_str: str,
    window_size: int,
    strength: float,
    ffmpeg: str,
    on_progress: Optional[Callable],
) -> dict:
    """Apply FFmpeg's built-in deflicker filter."""
    if on_progress:
        on_progress(20, "Applying FFmpeg deflicker filter...")

    # FFmpeg deflicker filter with configurable window size
    # mode=am (arithmetic mean) is the most balanced
    vf = f"deflicker=size={window_size}:mode=am"

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path_str,
    ]

    if on_progress:
        on_progress(40, "Encoding deflickered video...")

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Deflicker complete (simple)")

    return {
        "output_path": output_path_str,
        "method": "simple",
        "window_size": window_size,
        "strength": strength,
    }


def _deflicker_smooth(
    input_path: str,
    output_path_str: str,
    window_size: int,
    strength: float,
    ffmpeg: str,
    on_progress: Optional[Callable],
) -> dict:
    """
    Per-frame brightness correction using rolling average of luminance curve.

    1. Analyze per-frame luminance
    2. Compute rolling average (target)
    3. Calculate per-frame brightness adjustment
    4. Apply via FFmpeg eq filter with per-frame sendcmd
    """
    if on_progress:
        on_progress(10, "Analyzing per-frame luminance...")

    # Step 1: Get luminance data
    analysis = analyze_flicker(input_path, on_progress=None)
    lum_values = analysis.per_frame_luminance

    if not lum_values:
        logger.warning("No luminance data; falling back to simple deflicker")
        return _deflicker_simple(input_path, output_path_str, window_size,
                                 strength, ffmpeg, on_progress)

    if on_progress:
        on_progress(30, "Computing brightness corrections...")

    # Step 2: Rolling average as target luminance
    target_lum = _rolling_average(lum_values, window_size)

    # Step 3: Calculate per-frame brightness adjustment
    # eq filter brightness range is -1.0 to 1.0
    # We calculate the ratio between target and actual, then convert to brightness offset
    adjustments = []
    for actual, target in zip(lum_values, target_lum):
        if actual > 0:
            ratio = target / actual
            # Convert ratio to brightness adjustment (-1.0 to 1.0)
            # brightness=0 means no change, positive = brighter
            adj = (ratio - 1.0) * strength
            # Clamp to safe range
            adj = max(-0.5, min(0.5, adj))
        else:
            adj = 0.0
        adjustments.append(adj)

    if on_progress:
        on_progress(50, "Generating correction script...")

    # Step 4: Write sendcmd script for per-frame adjustment
    # Get video FPS for timestamp calculation
    fps = 24.0
    try:
        probe_cmd = [
            get_ffprobe_path(), "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-select_streams", "v:0",
            input_path,
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if probe_result.returncode == 0:
            probe_data = json.loads(probe_result.stdout)
            streams = probe_data.get("streams", [])
            if streams:
                fps_str = streams[0].get("r_frame_rate", "24/1")
                parts = fps_str.split("/")
                if len(parts) == 2 and float(parts[1]) > 0:
                    fps = float(parts[0]) / float(parts[1])
    except Exception:
        pass

    # Write sendcmd file
    tmp_dir = tempfile.gettempdir()
    cmd_path = os.path.join(tmp_dir, f"opencut_deflicker_{os.getpid()}.cmd")
    with open(cmd_path, "w") as f:
        for i, adj in enumerate(adjustments):
            timestamp = i / fps
            f.write(f"{timestamp:.6f} [enter] eq brightness {adj:.6f};\n")

    if on_progress:
        on_progress(60, "Applying per-frame corrections...")

    # Apply via FFmpeg with sendcmd + eq filter
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-i", input_path,
        "-vf", f"sendcmd=f='{cmd_path}',eq",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path_str,
    ]

    if on_progress:
        on_progress(70, "Encoding with per-frame brightness correction...")

    run_ffmpeg(cmd, timeout=3600)

    # Cleanup
    try:
        os.unlink(cmd_path)
    except OSError:
        pass

    # Compute correction stats
    abs_adj = [abs(a) for a in adjustments]
    avg_correction = sum(abs_adj) / max(1, len(abs_adj))
    max_correction = max(abs_adj) if abs_adj else 0.0

    if on_progress:
        on_progress(100, "Deflicker complete (smooth)")

    return {
        "output_path": output_path_str,
        "method": "smooth",
        "window_size": window_size,
        "strength": strength,
        "frames_corrected": len(adjustments),
        "avg_correction": round(avg_correction, 4),
        "max_correction": round(max_correction, 4),
        "original_flicker_score": analysis.flicker_score,
    }
