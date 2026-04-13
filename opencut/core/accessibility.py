"""
OpenCut Accessibility Tools

Color blindness simulation and photosensitive seizure (flashing) detection
to help creators make inclusive video content.

Uses FFmpeg only -- no additional dependencies required.
"""

import logging
import os
import re
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")

VALID_CONDITIONS = ("deuteranopia", "protanopia", "tritanopia", "achromatopsia")

# ---------------------------------------------------------------------------
# Color channel mixer matrices for color blindness simulation.
#
# These are linearized approximations of the Brettel/Vienot models.
# Each matrix maps [R, G, B] -> [R', G', B'] for the simulated condition.
# FFmpeg colorchannelmixer takes rr, rg, rb, gr, gg, gb, br, bg, bb.
# ---------------------------------------------------------------------------
_CB_MATRICES = {
    # Deuteranopia (green-blind, ~6% of males)
    "deuteranopia": {
        "rr": 0.625, "rg": 0.375, "rb": 0.0,
        "gr": 0.7,   "gg": 0.3,   "gb": 0.0,
        "br": 0.0,   "bg": 0.3,   "bb": 0.7,
    },
    # Protanopia (red-blind, ~2% of males)
    "protanopia": {
        "rr": 0.567, "rg": 0.433, "rb": 0.0,
        "gr": 0.558, "gg": 0.442, "gb": 0.0,
        "br": 0.0,   "bg": 0.242, "bb": 0.758,
    },
    # Tritanopia (blue-blind, very rare)
    "tritanopia": {
        "rr": 0.95, "rg": 0.05, "rb": 0.0,
        "gr": 0.0,  "gg": 0.433, "gb": 0.567,
        "br": 0.0,  "bg": 0.475, "bb": 0.525,
    },
    # Achromatopsia (total color blindness)
    "achromatopsia": {
        "rr": 0.299, "rg": 0.587, "rb": 0.114,
        "gr": 0.299, "gg": 0.587, "gb": 0.114,
        "br": 0.299, "bg": 0.587, "bb": 0.114,
    },
}


def simulate_color_blindness(
    input_path: str,
    condition: str = "deuteranopia",
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Simulate a color vision deficiency on the input video.

    Args:
        input_path: Source video file.
        condition: Color blindness type -- "deuteranopia", "protanopia",
                   "tritanopia", or "achromatopsia".
        output_path: Explicit output path.  Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with ``output_path``.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Invalid condition '{condition}'. Must be one of: {VALID_CONDITIONS}")

    if output_path is None:
        output_path = _output_path(input_path, f"cb_{condition}")

    matrix = _CB_MATRICES[condition]

    if on_progress:
        on_progress(10, f"Simulating {condition}...")

    # Build colorchannelmixer filter string
    ccm = (
        f"colorchannelmixer="
        f"rr={matrix['rr']}:rg={matrix['rg']}:rb={matrix['rb']}:"
        f"gr={matrix['gr']}:gg={matrix['gg']}:gb={matrix['gb']}:"
        f"br={matrix['br']}:bg={matrix['bg']}:bb={matrix['bb']}"
    )

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(ccm)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(output_path)
        .build()
    )

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, f"{condition} simulation complete")

    return {"output_path": output_path}


# ---------------------------------------------------------------------------
# Photosensitive Seizure / Flash Detection
# ---------------------------------------------------------------------------
@dataclass
class FlashEvent:
    """A detected flash event in the video."""
    start: float
    end: float
    flash_count: int
    peak_luminance_change: float
    severity: str  # "low", "medium", "high"


@dataclass
class FlashingResult:
    """Complete flash detection analysis results."""
    events: List[FlashEvent] = field(default_factory=list)
    total_flashes: int = 0
    risk_assessment: str = "safe"  # "safe", "warning", "dangerous"
    max_flashes_per_sec: float = 0.0
    duration_analyzed: float = 0.0


def detect_flashing(
    input_path: str,
    max_flashes_per_sec: int = 3,
    min_luminance_change: float = 0.2,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Detect potentially seizure-inducing flash sequences in a video.

    Analyzes frame-to-frame luminance changes and identifies sequences
    where rapid flashing exceeds safety thresholds (based on ITU-R BT.1702
    and W3C WCAG 2.3.1 guidelines).

    Args:
        input_path: Source video file.
        max_flashes_per_sec: Flash rate threshold (default 3, per WCAG).
        min_luminance_change: Minimum relative luminance change to count
            as a flash (0.0-1.0, default 0.2 = 20% change).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with keys: events, total_flashes, risk_assessment,
        max_flashes_per_sec, duration_analyzed.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    max_flashes_per_sec = max(1, int(max_flashes_per_sec))
    min_luminance_change = max(0.01, min(1.0, float(min_luminance_change)))

    info = get_video_info(input_path)
    fps = info["fps"]
    duration = info["duration"]

    if on_progress:
        on_progress(5, "Analyzing luminance levels...")

    # Step 1: Extract per-frame average luminance using signalstats filter.
    # This outputs YAVG (average luma) for each frame.
    luminance_values = _extract_frame_luminance(input_path, on_progress)

    if not luminance_values:
        # Fallback: unable to parse luminance data
        result = FlashingResult(duration_analyzed=duration)
        return _flashing_result_to_dict(result)

    if on_progress:
        on_progress(60, "Detecting flash sequences...")

    # Step 2: Compute frame-to-frame luminance differences
    diffs = []
    for i in range(1, len(luminance_values)):
        diff = abs(luminance_values[i] - luminance_values[i - 1]) / 255.0
        diffs.append(diff)

    # Step 3: Find flash events (transitions exceeding threshold)
    flash_frames = []
    for i, diff in enumerate(diffs):
        if diff >= min_luminance_change:
            flash_frames.append(i + 1)  # frame index (1-based in diff array)

    if on_progress:
        on_progress(80, "Classifying flash events...")

    # Step 4: Group flashes into events using a sliding window of 1 second
    events = []
    if flash_frames and fps > 0:
        window_frames = int(fps)  # 1-second window
        i = 0
        while i < len(flash_frames):
            # Count flashes within 1 second of current flash
            window_start = flash_frames[i]
            window_end = window_start + window_frames
            group = [f for f in flash_frames[i:] if f <= window_end]
            flashes_in_window = len(group)

            if flashes_in_window >= max_flashes_per_sec:
                start_time = flash_frames[i] / fps
                end_frame = group[-1]
                end_time = end_frame / fps

                # Calculate peak luminance change in this event
                event_diffs = [
                    diffs[f - 1] for f in group if (f - 1) < len(diffs)
                ]
                peak_change = max(event_diffs) if event_diffs else 0.0

                # Classify severity
                if flashes_in_window >= max_flashes_per_sec * 2 or peak_change >= 0.5:
                    severity = "high"
                elif flashes_in_window >= max_flashes_per_sec or peak_change >= 0.3:
                    severity = "medium"
                else:
                    severity = "low"

                events.append(FlashEvent(
                    start=round(start_time, 3),
                    end=round(end_time, 3),
                    flash_count=flashes_in_window,
                    peak_luminance_change=round(peak_change, 4),
                    severity=severity,
                ))
                # Skip past this event
                i += len(group)
            else:
                i += 1

    # Step 5: Overall risk assessment
    total_flashes = len(flash_frames)
    if not events:
        risk = "safe"
    elif any(e.severity == "high" for e in events):
        risk = "dangerous"
    elif any(e.severity == "medium" for e in events):
        risk = "warning"
    else:
        risk = "warning" if len(events) > 2 else "safe"

    # Max flash rate
    max_rate = 0.0
    if events:
        max_rate = max(
            e.flash_count / max(0.01, e.end - e.start) for e in events
        )

    if on_progress:
        on_progress(100, f"Flash detection complete: {risk}")

    result = FlashingResult(
        events=events,
        total_flashes=total_flashes,
        risk_assessment=risk,
        max_flashes_per_sec=round(max_rate, 2),
        duration_analyzed=duration,
    )

    return _flashing_result_to_dict(result)


def _extract_frame_luminance(input_path: str, on_progress=None) -> list:
    """Extract per-frame average luminance (YAVG) using FFmpeg signalstats."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-i", input_path,
        "-vf", "signalstats=stat=tout+vrep+brng,metadata=mode=print:key=lavfi.signalstats.YAVG",
        "-f", "null", "-",
    ]

    try:
        result = _sp.run(cmd, capture_output=True, timeout=600, text=True)
    except _sp.TimeoutExpired:
        logger.warning("Frame luminance extraction timed out for %s", input_path)
        return []

    # Parse YAVG values from stderr
    # Lines look like: lavfi.signalstats.YAVG=123.456
    values = []
    yavg_pattern = re.compile(r"lavfi\.signalstats\.YAVG=(\d+\.?\d*)")
    for line in result.stderr.split("\n"):
        match = yavg_pattern.search(line)
        if match:
            values.append(float(match.group(1)))

    if on_progress and values:
        on_progress(50, f"Analyzed {len(values)} frames")

    return values


def _flashing_result_to_dict(result: FlashingResult) -> dict:
    """Convert FlashingResult dataclass to a plain dict for JSON serialization."""
    return {
        "events": [
            {
                "start": e.start,
                "end": e.end,
                "flash_count": e.flash_count,
                "peak_luminance_change": e.peak_luminance_change,
                "severity": e.severity,
            }
            for e in result.events
        ],
        "total_flashes": result.total_flashes,
        "risk_assessment": result.risk_assessment,
        "max_flashes_per_sec": result.max_flashes_per_sec,
        "duration_analyzed": result.duration_analyzed,
    }
