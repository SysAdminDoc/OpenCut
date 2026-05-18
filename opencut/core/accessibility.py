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
from typing import Callable, List, Optional, Tuple

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
    flash_type: str = "general"  # "general", "red", "mixed"
    standard: str = "legacy"
    min_gap_ms: float = 0.0
    unsafe_gap_count: int = 0
    area_ratio: float = 1.0


@dataclass
class FrameFlashMetrics:
    """Per-frame global colour/luminance metrics used by the PSE checker."""
    index: int
    yavg: float
    relative_luminance: float
    r: float
    g: float
    b: float


@dataclass
class FlashCandidate:
    """One flash, represented by a pair of opposing frame transitions."""
    frame: int
    time: float
    flash_type: str
    magnitude: float


@dataclass
class FlashingResult:
    """Complete flash detection analysis results."""
    events: List[FlashEvent] = field(default_factory=list)
    total_flashes: int = 0
    risk_assessment: str = "safe"  # "safe", "warning", "dangerous"
    max_flashes_per_sec: float = 0.0
    duration_analyzed: float = 0.0
    general_flash_count: int = 0
    red_flash_count: int = 0
    standard_profile: str = "bt1702-3"
    frame_rate_profile: str = "60hz"
    thresholds: dict = field(default_factory=dict)


def detect_flashing(
    input_path: str,
    max_flashes_per_sec: int = 3,
    min_luminance_change: float = 0.1,
    standard_profile: str = "bt1702-3",
    screen_area_ratio: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Detect potentially seizure-inducing flash sequences in a video.

    Analyzes frame-to-frame luminance changes and identifies sequences
    where rapid flashing exceeds safety thresholds (based on ITU-R BT.1702-3
    and W3C WCAG 2.3.1 guidelines). The F238 path counts pairs of opposing
    transitions as flashes, applies the BT.1702 360 ms / 334 ms safe-gap rule,
    and separately tracks saturated-red flash pairs using the PEAT/WCAG working
    red definition plus Japan-style "isolated red" caution.

    Args:
        input_path: Source video file.
        max_flashes_per_sec: Flash rate threshold (default 3, per WCAG).
        min_luminance_change: Minimum relative luminance change to count
            as a flash (0.0-1.0, default 0.1 = WCAG/BT.1702 10% threshold).
        standard_profile: "bt1702-3", "wcag22", or "japan-animation".
        screen_area_ratio: Estimated concurrent flashing area. Values at or
            below 0.25 are below the BT.1702/WCAG area threshold.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with keys: events, total_flashes, risk_assessment,
        max_flashes_per_sec, duration_analyzed.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    max_flashes_per_sec = max(1, int(max_flashes_per_sec))
    min_luminance_change = max(0.01, min(1.0, float(min_luminance_change)))
    screen_area_ratio = max(0.0, min(1.0, float(screen_area_ratio)))
    standard_profile = _normalize_flash_profile(standard_profile)

    info = get_video_info(input_path)
    fps = info["fps"]
    duration = info["duration"]
    frame_rate_profile = "60hz" if fps >= 55 else "50hz"
    safe_gap_ms = 334.0 if frame_rate_profile == "60hz" else 360.0

    if on_progress:
        on_progress(5, "Analyzing luminance and red-flash levels...")

    # Step 1: Extract per-frame average luminance and best-effort global RGB.
    # The RGB path is intentionally a coarse preflight, not a certified FPA.
    luminance_values = _extract_frame_luminance(input_path, on_progress)
    rgb_values = _extract_frame_rgb_average(input_path)
    frame_metrics = _build_frame_metrics(luminance_values, rgb_values)

    if not frame_metrics:
        # Fallback: unable to parse luminance data
        result = FlashingResult(
            duration_analyzed=duration,
            standard_profile=standard_profile,
            frame_rate_profile=frame_rate_profile,
            thresholds=_flash_thresholds_dict(
                min_luminance_change=min_luminance_change,
                max_flashes=max_flashes_per_sec,
                safe_gap_ms=safe_gap_ms,
                screen_area_ratio=screen_area_ratio,
            ),
        )
        return _flashing_result_to_dict(result)

    if on_progress:
        on_progress(60, "Detecting BT.1702 flash pairs...")

    general_flashes = _find_general_flash_candidates(
        frame_metrics,
        fps=fps,
        min_luminance_change=min_luminance_change,
    )
    red_flashes = _find_red_flash_candidates(frame_metrics, fps=fps)

    if on_progress:
        on_progress(80, "Classifying PSE risk windows...")

    # Step 4: Group flash pairs into one-second standard windows.
    area_over_threshold = screen_area_ratio > 0.25
    events = []
    if area_over_threshold:
        events.extend(_classify_flash_windows(
            general_flashes,
            flash_type="general",
            max_flashes=max_flashes_per_sec,
            safe_gap_ms=safe_gap_ms,
            standard=standard_profile,
            area_ratio=screen_area_ratio,
        ))
        events.extend(_classify_flash_windows(
            red_flashes,
            flash_type="red",
            max_flashes=max_flashes_per_sec,
            safe_gap_ms=min(333.0, safe_gap_ms),
            standard=standard_profile,
            area_ratio=screen_area_ratio,
        ))

    # Step 5: Overall risk assessment
    total_flashes = len(general_flashes) + len(red_flashes)
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
        max_rate = max(e.flash_count / max(0.01, e.end - e.start) for e in events)

    if on_progress:
        on_progress(100, f"Flash detection complete: {risk}")

    result = FlashingResult(
        events=events,
        total_flashes=total_flashes,
        risk_assessment=risk,
        max_flashes_per_sec=round(max_rate, 2),
        duration_analyzed=duration,
        general_flash_count=len(general_flashes),
        red_flash_count=len(red_flashes),
        standard_profile=standard_profile,
        frame_rate_profile=frame_rate_profile,
        thresholds=_flash_thresholds_dict(
            min_luminance_change=min_luminance_change,
            max_flashes=max_flashes_per_sec,
            safe_gap_ms=safe_gap_ms,
            screen_area_ratio=screen_area_ratio,
        ),
    )

    return _flashing_result_to_dict(result)


def _normalize_flash_profile(profile: str) -> str:
    normalized = str(profile or "bt1702-3").strip().lower().replace("_", "-")
    aliases = {
        "bt1702": "bt1702-3",
        "itu-r-bt1702": "bt1702-3",
        "itu-r-bt.1702": "bt1702-3",
        "wcag": "wcag22",
        "wcag-2.2": "wcag22",
        "wcag2.2": "wcag22",
        "japan": "japan-animation",
        "japan-red": "japan-animation",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"bt1702-3", "wcag22", "japan-animation"}:
        raise ValueError("standard_profile must be bt1702-3, wcag22, or japan-animation")
    return normalized


def _flash_thresholds_dict(
    *,
    min_luminance_change: float,
    max_flashes: int,
    safe_gap_ms: float,
    screen_area_ratio: float,
) -> dict:
    return {
        "standard": "ITU-R BT.1702-3 / WCAG 2.3.1 preflight",
        "general_luminance_delta": round(min_luminance_change, 4),
        "darker_luminance_below": 0.8,
        "max_flashes_per_second": max_flashes,
        "safe_gap_50hz_ms": 360,
        "safe_gap_60hz_ms": 334,
        "applied_safe_gap_ms": safe_gap_ms,
        "area_ratio": round(screen_area_ratio, 4),
        "area_threshold_ratio": 0.25,
        "red_ratio_threshold": 0.8,
        "red_signal_delta_threshold": 20,
        "certification": "preflight only; use Harding/PEAT-equivalent tools for formal delivery sign-off",
    }


def _srgb_to_linear(value: float) -> float:
    value = max(0.0, min(1.0, value))
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _relative_luminance(r: float, g: float, b: float) -> float:
    return 0.2126 * _srgb_to_linear(r) + 0.7152 * _srgb_to_linear(g) + 0.0722 * _srgb_to_linear(b)


def _build_frame_metrics(
    luminance_values: List[float],
    rgb_values: List[Tuple[float, float, float]],
) -> List[FrameFlashMetrics]:
    metrics = []
    for index, yavg in enumerate(luminance_values):
        y_norm = max(0.0, min(1.0, float(yavg) / 255.0))
        if index < len(rgb_values):
            r, g, b = rgb_values[index]
            rel_lum = _relative_luminance(r, g, b)
        else:
            r = g = b = y_norm
            rel_lum = y_norm
        metrics.append(FrameFlashMetrics(index=index, yavg=float(yavg), relative_luminance=rel_lum, r=r, g=g, b=b))
    return metrics


def _transition_sign(delta: float) -> int:
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def _find_general_flash_candidates(
    metrics: List[FrameFlashMetrics],
    *,
    fps: float,
    min_luminance_change: float,
) -> List[FlashCandidate]:
    transitions = []
    for i in range(1, len(metrics)):
        previous = metrics[i - 1]
        current = metrics[i]
        delta = current.relative_luminance - previous.relative_luminance
        darker = min(previous.relative_luminance, current.relative_luminance)
        magnitude = abs(delta)
        if darker < 0.8 and magnitude >= min_luminance_change:
            transitions.append((i, _transition_sign(delta), magnitude))

    flashes = []
    i = 1
    while i < len(transitions):
        _prev_frame, prev_sign, prev_mag = transitions[i - 1]
        frame, sign, mag = transitions[i]
        if prev_sign and sign and prev_sign != sign:
            flashes.append(FlashCandidate(
                frame=frame,
                time=frame / fps if fps > 0 else 0.0,
                flash_type="general",
                magnitude=min(prev_mag, mag),
            ))
            i += 2
        else:
            i += 1
    return flashes


def _red_signal(metric: FrameFlashMetrics) -> float:
    return max(0.0, (metric.r - metric.g - metric.b) * 320.0)


def _is_saturated_red(metric: FrameFlashMetrics) -> bool:
    total = metric.r + metric.g + metric.b
    return total > 0 and (metric.r / total) >= 0.8 and _red_signal(metric) > 20.0


def _find_red_flash_candidates(metrics: List[FrameFlashMetrics], *, fps: float) -> List[FlashCandidate]:
    transitions = []
    for i in range(1, len(metrics)):
        previous = metrics[i - 1]
        current = metrics[i]
        previous_signal = _red_signal(previous)
        current_signal = _red_signal(current)
        delta = current_signal - previous_signal
        if abs(delta) > 20.0 and (_is_saturated_red(previous) or _is_saturated_red(current)):
            transitions.append((i, _transition_sign(delta), abs(delta) / 320.0))

    flashes = []
    i = 1
    while i < len(transitions):
        _prev_frame, prev_sign, prev_mag = transitions[i - 1]
        frame, sign, mag = transitions[i]
        if prev_sign and sign and prev_sign != sign:
            flashes.append(FlashCandidate(
                frame=frame,
                time=frame / fps if fps > 0 else 0.0,
                flash_type="red",
                magnitude=min(prev_mag, mag),
            ))
            i += 2
        else:
            i += 1
    return flashes


def _classify_flash_windows(
    candidates: List[FlashCandidate],
    *,
    flash_type: str,
    max_flashes: int,
    safe_gap_ms: float,
    standard: str,
    area_ratio: float,
) -> List[FlashEvent]:
    events = []
    i = 0
    while i < len(candidates):
        start = candidates[i]
        group = [item for item in candidates[i:] if item.time - start.time <= 1.0]
        count = len(group)
        if count > max_flashes:
            gaps = [
                (group[j].time - group[j - 1].time) * 1000.0
                for j in range(1, len(group))
            ]
            min_gap = min(gaps) if gaps else 0.0
            unsafe_gap_count = sum(1 for gap in gaps if gap < safe_gap_ms)
            peak = max(item.magnitude for item in group)
            severity = "high" if count >= max_flashes * 2 or peak >= 0.5 else "medium"
            events.append(FlashEvent(
                start=round(group[0].time, 3),
                end=round(group[-1].time, 3),
                flash_count=count,
                peak_luminance_change=round(peak, 4),
                severity=severity,
                flash_type=flash_type,
                standard=standard,
                min_gap_ms=round(min_gap, 1),
                unsafe_gap_count=unsafe_gap_count,
                area_ratio=round(area_ratio, 4),
            ))
            i += count
        else:
            i += 1
    return events


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


def _extract_frame_rgb_average(input_path: str) -> List[Tuple[float, float, float]]:
    """Extract a best-effort per-frame 1x1 RGB average using FFmpeg."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-i", input_path,
        "-vf", "scale=1:1:flags=area,format=rgb24",
        "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    try:
        result = _sp.run(cmd, capture_output=True, timeout=600)
    except _sp.TimeoutExpired:
        logger.warning("Frame RGB extraction timed out for %s", input_path)
        return []
    if result.returncode != 0 or not result.stdout:
        return []
    data = result.stdout
    values: List[Tuple[float, float, float]] = []
    for i in range(0, len(data) - 2, 3):
        values.append((data[i] / 255.0, data[i + 1] / 255.0, data[i + 2] / 255.0))
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
                "flash_type": e.flash_type,
                "standard": e.standard,
                "min_gap_ms": e.min_gap_ms,
                "unsafe_gap_count": e.unsafe_gap_count,
                "area_ratio": e.area_ratio,
            }
            for e in result.events
        ],
        "total_flashes": result.total_flashes,
        "general_flash_count": result.general_flash_count,
        "red_flash_count": result.red_flash_count,
        "risk_assessment": result.risk_assessment,
        "max_flashes_per_sec": result.max_flashes_per_sec,
        "duration_analyzed": result.duration_analyzed,
        "standard_profile": result.standard_profile,
        "frame_rate_profile": result.frame_rate_profile,
        "thresholds": result.thresholds,
    }
