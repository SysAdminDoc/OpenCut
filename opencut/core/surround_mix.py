"""
OpenCut Surround Sound Panning & Upmix v1.0.0

Surround sound processing:
- Stereo to 5.1/7.1 upmix
- Per-clip panning in surround field
- Stereo downmix preview
- Multichannel export (WAV, FLAC, AC3, EAC3)

All processing uses FFmpeg — no additional model downloads required.
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from opencut.helpers import get_ffmpeg_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHANNEL_LAYOUTS = {
    "5.1": {"channels": 6, "layout": "5.1", "label": "5.1 Surround"},
    "7.1": {"channels": 8, "layout": "7.1", "label": "7.1 Surround"},
    "stereo": {"channels": 2, "layout": "stereo", "label": "Stereo"},
    "mono": {"channels": 1, "layout": "mono", "label": "Mono"},
}

# 5.1 channel order: FL FR FC LFE BL BR
# 7.1 channel order: FL FR FC LFE BL BR SL SR
SURROUND_POSITIONS = {
    "front_left": {"angle": -30, "channel_51": "FL"},
    "front_right": {"angle": 30, "channel_51": "FR"},
    "center": {"angle": 0, "channel_51": "FC"},
    "lfe": {"angle": 0, "channel_51": "LFE"},
    "back_left": {"angle": -110, "channel_51": "BL"},
    "back_right": {"angle": 110, "channel_51": "BR"},
    "side_left": {"angle": -90, "channel_71": "SL"},
    "side_right": {"angle": 90, "channel_71": "SR"},
}

EXPORT_FORMATS = {
    "wav": {"codec": "pcm_s24le", "ext": ".wav", "label": "WAV 24-bit"},
    "flac": {"codec": "flac", "ext": ".flac", "label": "FLAC Lossless"},
    "ac3": {"codec": "ac3", "ext": ".ac3", "label": "Dolby Digital (AC3)"},
    "eac3": {"codec": "eac3", "ext": ".eac3", "label": "Dolby Digital Plus (E-AC3)"},
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class SurroundPosition:
    """Position in the surround field."""
    angle: float = 0.0        # -180 to 180 degrees (0 = center front)
    elevation: float = 0.0    # -90 to 90 degrees
    distance: float = 1.0     # 0.0 to 1.0 (relative distance)
    lfe_amount: float = 0.0   # 0.0 to 1.0 (LFE channel contribution)


@dataclass
class UpmixResult:
    """Result from surround upmix."""
    output_path: str = ""
    source_channels: int = 0
    target_channels: int = 0
    target_layout: str = ""
    duration: float = 0.0


@dataclass
class PanResult:
    """Result from surround panning."""
    output_path: str = ""
    position: Optional[SurroundPosition] = None
    channels: int = 6
    layout: str = "5.1"


# ---------------------------------------------------------------------------
# Upmix
# ---------------------------------------------------------------------------
def upmix_to_surround(
    audio_path: str,
    channels: str = "5.1",
    output_path_val: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> UpmixResult:
    """
    Upmix stereo audio to 5.1 or 7.1 surround sound.

    Uses FFmpeg's pan filter to distribute stereo content across surround
    channels with appropriate weighting for a natural surround image.

    Args:
        audio_path: Source audio/video file (stereo).
        channels: Target layout - "5.1" or "7.1".
        output_path_val: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        UpmixResult with output path and channel info.
    """
    if channels not in ("5.1", "7.1"):
        raise ValueError(f"Unsupported channel layout: {channels}. Use '5.1' or '7.1'.")

    ffmpeg = get_ffmpeg_path()

    if output_path_val is None:
        output_path_val = output_path(audio_path, f"_upmix_{channels.replace('.', '')}")

    if on_progress:
        on_progress(5, f"Upmixing to {channels}...")

    layout_info = CHANNEL_LAYOUTS[channels]
    target_ch = layout_info["channels"]

    # Build upmix filter
    if channels == "5.1":
        # Distribute stereo to 5.1:
        # FL = 0.7*L + 0.2*R
        # FR = 0.2*L + 0.7*R
        # FC = 0.5*L + 0.5*R (center from phantom image)
        # LFE = 0.3*L + 0.3*R (filtered low frequencies)
        # BL = 0.4*L (rear left from left)
        # BR = 0.4*R (rear right from right)
        af = (
            "pan=5.1|"
            "FL=0.7*FL+0.2*FR|"
            "FR=0.2*FL+0.7*FR|"
            "FC=0.5*FL+0.5*FR|"
            "LFE=0.3*FL+0.3*FR|"
            "BL=0.4*FL|"
            "BR=0.4*FR"
        )
    else:  # 7.1
        af = (
            "pan=7.1|"
            "FL=0.6*FL+0.15*FR|"
            "FR=0.15*FL+0.6*FR|"
            "FC=0.5*FL+0.5*FR|"
            "LFE=0.25*FL+0.25*FR|"
            "BL=0.3*FL|"
            "BR=0.3*FR|"
            "SL=0.35*FL+0.1*FR|"
            "SR=0.1*FL+0.35*FR"
        )

    if on_progress:
        on_progress(20, "Applying surround upmix filter...")

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-y", "-i", audio_path,
        "-af", af,
        "-c:a", "pcm_s24le",
        output_path_val,
    ]
    run_ffmpeg(cmd, timeout=1800)

    if on_progress:
        on_progress(100, f"Upmix to {channels} complete")

    return UpmixResult(
        output_path=output_path_val,
        source_channels=2,
        target_channels=target_ch,
        target_layout=channels,
    )


# ---------------------------------------------------------------------------
# Surround Panning
# ---------------------------------------------------------------------------
def pan_in_surround(
    audio_path: str,
    position: SurroundPosition,
    channels: str = "5.1",
    output_path_val: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> PanResult:
    """
    Pan mono/stereo audio to a specific position in the surround field.

    Distributes the input across surround channels based on the specified
    angle and distance, creating a positioned sound source.

    Args:
        audio_path: Source audio file.
        position: SurroundPosition with angle, elevation, distance.
        channels: Target layout - "5.1" or "7.1".
        output_path_val: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        PanResult with output path and position info.
    """
    if channels not in ("5.1", "7.1"):
        raise ValueError(f"Unsupported layout: {channels}")

    ffmpeg = get_ffmpeg_path()

    if output_path_val is None:
        output_path_val = output_path(audio_path, f"_panned_{channels.replace('.', '')}")

    if on_progress:
        on_progress(10, f"Panning to angle {position.angle}...")

    # Calculate per-channel gains from position
    gains = _calculate_surround_gains(position, channels)

    # Build pan filter
    if channels == "5.1":
        af = (
            f"pan=5.1|"
            f"FL={gains['FL']:.3f}*c0|"
            f"FR={gains['FR']:.3f}*c0|"
            f"FC={gains['FC']:.3f}*c0|"
            f"LFE={gains['LFE']:.3f}*c0|"
            f"BL={gains['BL']:.3f}*c0|"
            f"BR={gains['BR']:.3f}*c0"
        )
    else:
        af = (
            f"pan=7.1|"
            f"FL={gains['FL']:.3f}*c0|"
            f"FR={gains['FR']:.3f}*c0|"
            f"FC={gains['FC']:.3f}*c0|"
            f"LFE={gains['LFE']:.3f}*c0|"
            f"BL={gains['BL']:.3f}*c0|"
            f"BR={gains['BR']:.3f}*c0|"
            f"SL={gains.get('SL', 0):.3f}*c0|"
            f"SR={gains.get('SR', 0):.3f}*c0"
        )

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-y", "-i", audio_path,
        "-af", af,
        "-c:a", "pcm_s24le",
        output_path_val,
    ]
    run_ffmpeg(cmd, timeout=1800)

    if on_progress:
        on_progress(100, "Surround panning complete")

    return PanResult(
        output_path=output_path_val,
        position=position,
        channels=CHANNEL_LAYOUTS[channels]["channels"],
        layout=channels,
    )


def _calculate_surround_gains(
    position: SurroundPosition,
    layout: str,
) -> Dict[str, float]:
    """
    Calculate per-channel gains from a surround position using VBAP-like panning.
    """
    import math

    angle = position.angle  # -180 to 180
    dist = max(0.01, min(1.0, position.distance))

    # Normalize angle to radians
    angle_rad = math.radians(angle)

    # Simple VBAP-like amplitude panning
    # Map angle to channel gains
    gains = {}

    # Front left (-30 degrees)
    gains["FL"] = max(0, math.cos(angle_rad - math.radians(-30))) * dist
    # Front right (+30 degrees)
    gains["FR"] = max(0, math.cos(angle_rad - math.radians(30))) * dist
    # Center (0 degrees)
    gains["FC"] = max(0, math.cos(angle_rad)) * dist * 0.7
    # LFE (omnidirectional, distance-based)
    gains["LFE"] = position.lfe_amount * dist
    # Back left (-110 degrees)
    gains["BL"] = max(0, math.cos(angle_rad - math.radians(-110))) * dist
    # Back right (+110 degrees)
    gains["BR"] = max(0, math.cos(angle_rad - math.radians(110))) * dist

    if layout == "7.1":
        gains["SL"] = max(0, math.cos(angle_rad - math.radians(-90))) * dist
        gains["SR"] = max(0, math.cos(angle_rad - math.radians(90))) * dist

    # Normalize so total energy is approximately 1.0
    total = sum(v for k, v in gains.items() if k != "LFE")
    if total > 0:
        scale = 1.0 / total
        for k in gains:
            if k != "LFE":
                gains[k] *= scale

    return gains


# ---------------------------------------------------------------------------
# Downmix Preview
# ---------------------------------------------------------------------------
def downmix_preview(
    surround_path: str,
    output_path_val: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Create a stereo downmix preview of surround audio.

    Folds surround channels back to stereo using ITU-R BS.775 coefficients.

    Args:
        surround_path: Surround audio file (5.1 or 7.1).
        output_path_val: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to the stereo downmix file.
    """
    ffmpeg = get_ffmpeg_path()

    if output_path_val is None:
        output_path_val = output_path(surround_path, "_stereo_downmix")

    if on_progress:
        on_progress(10, "Creating stereo downmix preview...")

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-y", "-i", surround_path,
        "-ac", "2",
        "-af", "aresample=matrix_encoding=dplii",
        output_path_val,
    ]
    run_ffmpeg(cmd, timeout=1800)

    if on_progress:
        on_progress(100, "Stereo downmix complete")

    return output_path_val


# ---------------------------------------------------------------------------
# Multichannel Export
# ---------------------------------------------------------------------------
def export_multichannel(
    audio_path: str,
    format: str = "wav",
    output_path_val: Optional[str] = None,
    bitrate: str = "640k",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Export multichannel audio in the specified format.

    Supports WAV, FLAC, AC3 (Dolby Digital), and E-AC3 (Dolby Digital Plus).

    Args:
        audio_path: Source multichannel audio file.
        format: Export format - "wav", "flac", "ac3", or "eac3".
        output_path_val: Output path (auto-generated if None).
        bitrate: Bitrate for lossy formats (AC3/E-AC3).
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to the exported file.
    """
    if format not in EXPORT_FORMATS:
        raise ValueError(
            f"Unsupported format: {format}. "
            f"Use one of: {', '.join(EXPORT_FORMATS.keys())}"
        )

    ffmpeg = get_ffmpeg_path()
    fmt_info = EXPORT_FORMATS[format]

    if output_path_val is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        directory = os.path.dirname(audio_path)
        output_path_val = os.path.join(directory, f"{base}_export{fmt_info['ext']}")

    if on_progress:
        on_progress(10, f"Exporting as {fmt_info['label']}...")

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-y", "-i", audio_path,
        "-c:a", fmt_info["codec"],
    ]

    # Add bitrate for lossy codecs
    if format in ("ac3", "eac3"):
        cmd.extend(["-b:a", bitrate])

    cmd.append(output_path_val)

    run_ffmpeg(cmd, timeout=1800)

    if on_progress:
        on_progress(100, f"Multichannel export complete ({fmt_info['label']})")

    return output_path_val
