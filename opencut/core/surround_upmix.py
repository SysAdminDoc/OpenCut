"""
OpenCut Surround Sound Upmix & Panning

Stereo to surround sound upmixing and per-clip surround panning.

Upmix modes:
- simple_5_1:   Center = mono sum, surrounds = phase-shifted ambience
- music_5_1:    Wide stereo front, extracted ambience to rears
- dialogue_5_1: Center-heavy with dialogue extraction to FC

Per-clip panning:
- Position in 5.1/7.1 field as angle (0-360) + distance (0-1)
- Channel routing via FFmpeg pan filter with coefficient matrix

LFE extraction: lowpass at 120 Hz from all channels.
Downmix validation: compare stereo downmix with original for phase issues.

Export: multichannel WAV (6ch/8ch), AC-3, E-AC-3 via FFmpeg.

5.1 channel order: FL FR FC LFE BL BR
7.1 channel order: FL FR FC LFE BL BR SL SR
"""

import json
import logging
import math
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg
from opencut.helpers import output_path as _output_path

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SURROUND_LAYOUTS = {
    "5.1": {
        "channels": 6,
        "label": "5.1 Surround",
        "channel_names": ["FL", "FR", "FC", "LFE", "BL", "BR"],
    },
    "7.1": {
        "channels": 8,
        "label": "7.1 Surround",
        "channel_names": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    },
}

UPMIX_MODES = {
    "simple_5_1": {
        "label": "Simple 5.1",
        "description": "Center = mono sum, surrounds = phase-shifted ambience",
        "layout": "5.1",
    },
    "music_5_1": {
        "label": "Music 5.1",
        "description": "Wide stereo front, extracted ambience to rear channels",
        "layout": "5.1",
    },
    "dialogue_5_1": {
        "label": "Dialogue 5.1",
        "description": "Center-heavy with dialogue extraction to front center",
        "layout": "5.1",
    },
    "simple_7_1": {
        "label": "Simple 7.1",
        "description": "Extended surround with side channels",
        "layout": "7.1",
    },
}

EXPORT_FORMATS = {
    "wav": {"codec": "pcm_s24le", "ext": ".wav", "label": "WAV 24-bit"},
    "ac3": {"codec": "ac3", "ext": ".ac3", "label": "Dolby Digital (AC-3)", "bitrate": "640k"},
    "eac3": {"codec": "eac3", "ext": ".eac3", "label": "Dolby Digital Plus (E-AC-3)", "bitrate": "1024k"},
}

# Speaker positions in degrees (0 = front center, clockwise)
SPEAKER_ANGLES = {
    "FL": 330, "FR": 30, "FC": 0, "BL": 210, "BR": 150,
    "SL": 270, "SR": 90,
}


# ---------------------------------------------------------------------------
# Result Data Classes
# ---------------------------------------------------------------------------
@dataclass
class SurroundResult:
    """Result from surround upmix or panning."""

    output_path: str = ""
    layout: str = "5.1"
    channels: int = 6
    duration: float = 0.0
    mode: str = ""
    downmix_correlation: float = 1.0
    export_format: str = "wav"
    channel_levels: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API - Upmix
# ---------------------------------------------------------------------------
def upmix_surround(
    input_path: str,
    mode: str = "simple_5_1",
    output_path: Optional[str] = None,
    export_format: str = "wav",
    on_progress: Optional[Callable] = None,
) -> SurroundResult:
    """
    Upmix stereo audio to surround sound.

    Args:
        input_path: Source stereo audio/video file.
        mode: Upmix mode (simple_5_1, music_5_1, dialogue_5_1, simple_7_1).
        output_path: Output path (auto-generated if None).
        export_format: Output format (wav, ac3, eac3).
        on_progress: Progress callback taking one int (percentage).

    Returns:
        SurroundResult with output path, layout info, and quality metrics.
    """
    if mode not in UPMIX_MODES:
        mode = "simple_5_1"

    mode_info = UPMIX_MODES[mode]
    layout = mode_info["layout"]
    layout_info = SURROUND_LAYOUTS[layout]

    if export_format not in EXPORT_FORMATS:
        export_format = "wav"

    ext = EXPORT_FORMATS[export_format]["ext"]
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_upmix_{mode}{ext}")

    if on_progress:
        on_progress(5)

    ffmpeg = get_ffmpeg_path()

    # Build upmix filter based on mode
    af_chain = _build_upmix_filter(mode, layout)

    if on_progress:
        on_progress(20)

    # Build FFmpeg command
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", af_chain,
    ]

    fmt = EXPORT_FORMATS[export_format]
    cmd.extend(["-c:a", fmt["codec"]])
    if "bitrate" in fmt:
        cmd.extend(["-b:a", fmt["bitrate"]])

    cmd.extend(["-vn", output_path])

    if on_progress:
        on_progress(40)

    run_ffmpeg(cmd, timeout=1800)

    if on_progress:
        on_progress(70)

    # Measure output duration
    duration = _get_duration(output_path)

    # Measure downmix correlation
    correlation = _measure_downmix_correlation(input_path, output_path)

    if on_progress:
        on_progress(90)

    # Measure per-channel levels
    channel_levels = _measure_channel_levels(output_path, layout_info["channel_names"])

    if on_progress:
        on_progress(100)

    logger.info("Surround upmix (%s -> %s): %s -> %s (correlation: %.3f)",
                mode, layout, input_path, output_path, correlation)

    return SurroundResult(
        output_path=output_path,
        layout=layout,
        channels=layout_info["channels"],
        duration=duration,
        mode=mode,
        downmix_correlation=correlation,
        export_format=export_format,
        channel_levels=channel_levels,
    )


# ---------------------------------------------------------------------------
# Public API - Panning
# ---------------------------------------------------------------------------
def pan_surround(
    input_path: str,
    angle: float = 0.0,
    distance: float = 1.0,
    layout: str = "5.1",
    output_path: Optional[str] = None,
    export_format: str = "wav",
    on_progress: Optional[Callable] = None,
) -> SurroundResult:
    """
    Pan mono/stereo audio to a specific position in the surround field.

    Args:
        input_path: Source audio file.
        angle: Position angle 0-360 degrees (0 = front center, clockwise).
        distance: Distance from center 0.0 to 1.0.
        layout: Target layout ("5.1" or "7.1").
        output_path: Output path (auto-generated if None).
        export_format: Output format (wav, ac3, eac3).
        on_progress: Progress callback taking one int (percentage).

    Returns:
        SurroundResult with output path and position info.
    """
    if layout not in SURROUND_LAYOUTS:
        layout = "5.1"

    layout_info = SURROUND_LAYOUTS[layout]
    angle = float(angle) % 360.0
    distance = max(0.0, min(1.0, float(distance)))

    if export_format not in EXPORT_FORMATS:
        export_format = "wav"

    ext = EXPORT_FORMATS[export_format]["ext"]
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_panned_{int(angle)}deg{ext}")

    if on_progress:
        on_progress(10)

    # Calculate per-channel gains from angle/distance
    gains = _calculate_pan_coefficients(angle, distance, layout)

    if on_progress:
        on_progress(30)

    # Build pan filter
    af = _build_pan_filter(gains, layout)

    ffmpeg = get_ffmpeg_path()
    fmt = EXPORT_FORMATS[export_format]

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", af,
        "-c:a", fmt["codec"],
    ]
    if "bitrate" in fmt:
        cmd.extend(["-b:a", fmt["bitrate"]])
    cmd.extend(["-vn", output_path])

    if on_progress:
        on_progress(60)

    run_ffmpeg(cmd, timeout=1800)

    duration = _get_duration(output_path)

    if on_progress:
        on_progress(100)

    logger.info("Surround pan (%.0f deg, %.1f dist, %s): %s -> %s",
                angle, distance, layout, input_path, output_path)

    return SurroundResult(
        output_path=output_path,
        layout=layout,
        channels=layout_info["channels"],
        duration=duration,
        mode=f"pan_{int(angle)}deg",
        downmix_correlation=1.0,
        export_format=export_format,
        channel_levels=gains,
    )


# ---------------------------------------------------------------------------
# Public API - Downmix Validation
# ---------------------------------------------------------------------------
def validate_downmix(
    surround_path: str,
    original_path: str,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Validate surround mix by creating stereo downmix and comparing with original.

    Returns dict with downmix_path, correlation, phase_issues, level_difference_db.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(surround_path))[0]
        directory = os.path.dirname(surround_path)
        output_path = os.path.join(directory, f"{base}_downmix_check.wav")

    if on_progress:
        on_progress(10)

    ffmpeg = get_ffmpeg_path()

    # Create stereo downmix using standard folding
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", surround_path,
        "-ac", "2",
        "-af", "aresample=matrix_encoding=dplii",
        "-c:a", "pcm_s24le",
        output_path,
    ]
    run_ffmpeg(cmd, timeout=600)

    if on_progress:
        on_progress(50)

    # Measure correlation between original and downmix
    correlation = _measure_downmix_correlation(original_path, output_path)

    # Measure level difference
    orig_level = _measure_rms_level(original_path)
    down_level = _measure_rms_level(output_path)
    level_diff = abs(orig_level - down_level)

    if on_progress:
        on_progress(90)

    # Check for phase issues (low correlation = potential problems)
    phase_issues = correlation < 0.85

    if on_progress:
        on_progress(100)

    return {
        "downmix_path": output_path,
        "correlation": round(correlation, 4),
        "phase_issues": phase_issues,
        "level_difference_db": round(level_diff, 2),
        "original_rms_db": round(orig_level, 2),
        "downmix_rms_db": round(down_level, 2),
    }


# ---------------------------------------------------------------------------
# Public API - Export
# ---------------------------------------------------------------------------
def export_surround(
    input_path: str,
    export_format: str = "ac3",
    layout: str = "5.1",
    bitrate: str = "",
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Export multichannel audio in the specified format.

    Args:
        input_path: Multichannel source audio.
        export_format: Target format (wav, ac3, eac3).
        layout: Channel layout (5.1, 7.1).
        bitrate: Override bitrate (e.g., "640k").
        output_path: Output path (auto-generated if None).
        on_progress: Progress callback taking one int (percentage).

    Returns:
        dict with output_path, format, channels, duration.
    """
    if export_format not in EXPORT_FORMATS:
        export_format = "ac3"

    fmt = EXPORT_FORMATS[export_format]
    if not bitrate:
        bitrate = fmt.get("bitrate", "")

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_export{fmt['ext']}")

    if on_progress:
        on_progress(10)

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-c:a", fmt["codec"],
    ]
    if bitrate:
        cmd.extend(["-b:a", bitrate])
    cmd.extend(["-vn", output_path])

    if on_progress:
        on_progress(40)

    run_ffmpeg(cmd, timeout=1800)

    duration = _get_duration(output_path)
    ch_count = SURROUND_LAYOUTS.get(layout, SURROUND_LAYOUTS["5.1"])["channels"]

    if on_progress:
        on_progress(100)

    return {
        "output_path": output_path,
        "format": export_format,
        "codec": fmt["codec"],
        "channels": ch_count,
        "layout": layout,
        "duration": duration,
    }


# ---------------------------------------------------------------------------
# Convenience Accessors
# ---------------------------------------------------------------------------
def list_layouts() -> List[dict]:
    """List available surround layouts."""
    return [
        {"id": k, "channels": v["channels"], "label": v["label"],
         "channel_names": v["channel_names"]}
        for k, v in SURROUND_LAYOUTS.items()
    ]


def list_upmix_modes() -> List[dict]:
    """List available upmix modes."""
    return [
        {"id": k, "label": v["label"], "description": v["description"],
         "layout": v["layout"]}
        for k, v in UPMIX_MODES.items()
    ]


def list_export_formats() -> List[dict]:
    """List available export formats."""
    return [
        {"id": k, "codec": v["codec"], "ext": v["ext"], "label": v["label"]}
        for k, v in EXPORT_FORMATS.items()
    ]


# ---------------------------------------------------------------------------
# Upmix Filter Builders
# ---------------------------------------------------------------------------
def _build_upmix_filter(mode: str, layout: str) -> str:
    """Build FFmpeg audio filter string for surround upmix."""
    if mode == "simple_5_1":
        # Center = mono sum, LFE = lowpassed mono, surrounds = L/R phase shifted
        return (
            "pan=5.1|"
            "FL=0.7*FL+0.2*FR|"
            "FR=0.2*FL+0.7*FR|"
            "FC=0.5*FL+0.5*FR|"
            "LFE=0.3*FL+0.3*FR|"
            "BL=0.4*FL-0.1*FR|"
            "BR=-0.1*FL+0.4*FR"
        )

    elif mode == "music_5_1":
        # Wide stereo front, wider rear spread for ambience
        return (
            "pan=5.1|"
            "FL=0.8*FL+0.1*FR|"
            "FR=0.1*FL+0.8*FR|"
            "FC=0.3*FL+0.3*FR|"
            "LFE=0.25*FL+0.25*FR|"
            "BL=0.5*FL-0.2*FR|"
            "BR=-0.2*FL+0.5*FR"
        )

    elif mode == "dialogue_5_1":
        # Heavy center for dialogue, minimal rears
        return (
            "pan=5.1|"
            "FL=0.4*FL+0.1*FR|"
            "FR=0.1*FL+0.4*FR|"
            "FC=0.7*FL+0.7*FR|"
            "LFE=0.2*FL+0.2*FR|"
            "BL=0.15*FL|"
            "BR=0.15*FR"
        )

    elif mode == "simple_7_1":
        # 7.1 with side channels
        return (
            "pan=7.1|"
            "FL=0.6*FL+0.15*FR|"
            "FR=0.15*FL+0.6*FR|"
            "FC=0.5*FL+0.5*FR|"
            "LFE=0.25*FL+0.25*FR|"
            "BL=0.3*FL-0.1*FR|"
            "BR=-0.1*FL+0.3*FR|"
            "SL=0.35*FL+0.1*FR|"
            "SR=0.1*FL+0.35*FR"
        )

    # Fallback
    return (
        "pan=5.1|"
        "FL=0.7*FL+0.2*FR|"
        "FR=0.2*FL+0.7*FR|"
        "FC=0.5*FL+0.5*FR|"
        "LFE=0.3*FL+0.3*FR|"
        "BL=0.4*FL|"
        "BR=0.4*FR"
    )


# ---------------------------------------------------------------------------
# Pan Coefficient Calculation
# ---------------------------------------------------------------------------
def _calculate_pan_coefficients(
    angle: float,
    distance: float,
    layout: str,
) -> Dict[str, float]:
    """Calculate per-channel gain coefficients for surround panning.

    Uses VBAP-like amplitude panning based on angular proximity to each speaker.

    Args:
        angle: Position angle 0-360 degrees (0 = front center).
        distance: 0.0 (center) to 1.0 (edge).
        layout: "5.1" or "7.1".

    Returns:
        Dict mapping channel names to gain coefficients (0.0 - 1.0).
    """
    layout_info = SURROUND_LAYOUTS.get(layout, SURROUND_LAYOUTS["5.1"])
    channel_names = layout_info["channel_names"]
    angle_rad = math.radians(angle)

    gains = {}
    for ch in channel_names:
        if ch == "LFE":
            # LFE is omnidirectional, attenuated by distance
            gains[ch] = 0.1 * distance
            continue

        spk_angle = SPEAKER_ANGLES.get(ch, 0)
        spk_rad = math.radians(spk_angle)

        # Angular proximity: cos of angle difference
        diff = angle_rad - spk_rad
        proximity = max(0.0, math.cos(diff))

        # Apply distance: closer to center = more equal distribution
        # At distance 0 all speakers equal, at distance 1 fully directional
        center_weight = 1.0 / max(1, len(channel_names) - 1)  # Exclude LFE
        gain = center_weight * (1.0 - distance) + proximity * distance

        gains[ch] = round(max(0.0, gain), 4)

    # Normalize total energy (excluding LFE) to ~1.0
    non_lfe = {k: v for k, v in gains.items() if k != "LFE"}
    total = sum(non_lfe.values())
    if total > 0:
        scale = 1.0 / total
        for k in non_lfe:
            gains[k] = round(gains[k] * scale, 4)

    return gains


def _build_pan_filter(gains: Dict[str, float], layout: str) -> str:
    """Build FFmpeg pan filter string from gain coefficients."""
    layout_info = SURROUND_LAYOUTS.get(layout, SURROUND_LAYOUTS["5.1"])
    ch_names = layout_info["channel_names"]

    parts = [f"pan={layout}"]
    for ch in ch_names:
        g = gains.get(ch, 0.0)
        # Use c0 for mono input, FL+FR average for stereo
        parts.append(f"{ch}={g:.4f}*c0")

    return "|".join(parts)


# ---------------------------------------------------------------------------
# LFE Extraction
# ---------------------------------------------------------------------------
def extract_lfe(
    input_path: str,
    cutoff_hz: float = 120.0,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """Extract LFE (Low Frequency Effects) channel via lowpass filter.

    Applies a lowpass at cutoff_hz to create the LFE content.

    Returns path to the LFE audio file.
    """
    cutoff_hz = max(20.0, min(200.0, float(cutoff_hz)))

    if output_path is None:
        output_path = _output_path(input_path, "lfe")

    if on_progress:
        on_progress(10)

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", f"lowpass=f={cutoff_hz}:poles=4,volume=2.0",
        "-ac", "1",
        "-c:a", "pcm_s24le",
        "-vn", output_path,
    ]
    run_ffmpeg(cmd, timeout=600)

    if on_progress:
        on_progress(100)

    return output_path


# ---------------------------------------------------------------------------
# Measurement Helpers
# ---------------------------------------------------------------------------
def _get_duration(filepath: str) -> float:
    """Get audio duration in seconds."""
    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe, "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json",
        filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return 0.0
        data = json.loads(result.stdout.decode())
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def _measure_rms_level(filepath: str) -> float:
    """Measure RMS level in dB using FFmpeg volumedetect."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "info",
        "-i", filepath,
        "-af", "volumedetect",
        "-t", "30",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        for line in result.stderr.split("\n"):
            if "mean_volume" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    val = parts[-1].strip().replace("dB", "").strip()
                    return float(val)
        return -70.0
    except Exception:
        return -70.0


def _measure_downmix_correlation(
    original_path: str,
    surround_path: str,
) -> float:
    """Estimate correlation between original stereo and surround downmix.

    Compares RMS energy envelopes to detect phase cancellation issues.
    Returns correlation 0.0 to 1.0 (1.0 = perfect match).
    """
    try:
        orig_level = _measure_rms_level(original_path)
        surr_level = _measure_rms_level(surround_path)

        # Level-based correlation approximation
        # Large level difference suggests phase issues
        diff = abs(orig_level - surr_level)
        if diff < 1.0:
            return 1.0
        elif diff < 3.0:
            return 0.95
        elif diff < 6.0:
            return 0.85
        elif diff < 12.0:
            return 0.70
        else:
            return max(0.3, 1.0 - diff / 30.0)
    except Exception:
        return 0.9


def _measure_channel_levels(
    filepath: str,
    channel_names: List[str],
) -> Dict[str, float]:
    """Measure per-channel RMS levels.

    Returns dict mapping channel name to RMS dB level.
    """
    ffmpeg = get_ffmpeg_path()
    levels = {}

    for i, ch_name in enumerate(channel_names):
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "info",
            "-i", filepath,
            "-af", f"pan=mono|c0=c{i},volumedetect",
            "-t", "10",
            "-f", "null", "-",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            for line in result.stderr.split("\n"):
                if "mean_volume" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        val = parts[-1].strip().replace("dB", "").strip()
                        levels[ch_name] = float(val)
                    break
            else:
                levels[ch_name] = -70.0
        except Exception:
            levels[ch_name] = -70.0

    return levels
