"""
OpenCut Spatial Audio Module (Feature 2.5)

Convert stereo audio to binaural (via FFmpeg sofalizer filter),
upmix to 5.1/7.1 surround (via FFmpeg pan filter), and detect
the channel layout of an audio file.

Functions:
    to_binaural          - Convert stereo to binaural via sofalizer
    to_surround          - Upmix stereo to 5.1 or 7.1
    detect_channel_layout - Detect the channel layout of an audio file
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffprobe_path,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# Default SOFA file path (user can override)
DEFAULT_SOFA_PATH = os.path.join(
    os.path.expanduser("~"), ".opencut", "sofa", "ClubFritz6.sofa"
)


@dataclass
class SpatialResult:
    """Result of a spatial audio conversion."""
    output_path: str
    input_layout: str
    output_layout: str
    channels: int
    method: str


def detect_channel_layout(
    audio_path: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Detect the channel layout and channel count of an audio file.

    Args:
        audio_path: Path to the audio or video file.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        Dict with 'channel_layout', 'channels', 'sample_rate', 'codec'.
    """
    if on_progress:
        on_progress(10, "Probing audio channels...")

    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels,channel_layout,sample_rate,codec_name",
        "-of", "json",
        audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=30, check=False)
    defaults = {
        "channel_layout": "stereo",
        "channels": 2,
        "sample_rate": 44100,
        "codec": "unknown",
    }

    if result.returncode != 0:
        logger.warning("ffprobe failed for channel detection on %s", audio_path)
        return defaults

    try:
        data = json.loads(result.stdout.decode())
        streams = data.get("streams", [])
        if not streams:
            return defaults
        s = streams[0]
        return {
            "channel_layout": s.get("channel_layout", "stereo"),
            "channels": int(s.get("channels", 2)),
            "sample_rate": int(s.get("sample_rate", 44100)),
            "codec": s.get("codec_name", "unknown"),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Failed to parse channel info for %s: %s", audio_path, e)
        return defaults


def to_binaural(
    audio_path: str,
    output: Optional[str] = None,
    sofa_path: Optional[str] = None,
    gain: float = 0.0,
    on_progress: Optional[Callable] = None,
) -> SpatialResult:
    """Convert stereo/surround audio to binaural using FFmpeg sofalizer.

    If no SOFA file is available, falls back to a crossfeed-based
    pseudo-binaural effect using FFmpeg's built-in filters.

    Args:
        audio_path: Path to the input audio or video file.
        output: Output file path. Auto-generated if None.
        sofa_path: Path to a SOFA HRTF file. Uses default if None.
        gain: Gain adjustment in dB (default 0.0).
        on_progress: Optional progress callback(pct, msg).

    Returns:
        SpatialResult with output path and conversion details.
    """
    if on_progress:
        on_progress(5, "Detecting input channel layout...")

    layout_info = detect_channel_layout(audio_path)
    input_layout = layout_info["channel_layout"]

    if output is None:
        output = output_path(audio_path, "binaural")

    # Ensure output has audio extension
    _, ext = os.path.splitext(output)
    if not ext:
        output += ".wav"

    sofa = sofa_path or DEFAULT_SOFA_PATH
    method = "sofalizer"

    if on_progress:
        on_progress(20, "Building binaural filter...")

    if os.path.isfile(sofa):
        # Use sofalizer with SOFA HRTF file
        af = f"sofalizer=sofa={sofa}:gain={gain}"
    else:
        # Fallback: crossfeed-based pseudo-binaural
        logger.info("SOFA file not found at %s, using crossfeed fallback", sofa)
        method = "crossfeed_fallback"
        af = (
            "crossfeed=strength=0.7:range=700:slope=0.5,"
            "aecho=0.8:0.88:6:0.4,"
            "equalizer=f=1000:t=h:w=200:g=-2"
        )

    if on_progress:
        on_progress(30, f"Converting to binaural ({method})...")

    cmd = (
        FFmpegCmd()
        .input(audio_path)
        .audio_filter(af)
        .audio_codec("pcm_s16le")
        .option("ac", "2")
        .output(output)
        .build()
    )

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(95, "Binaural conversion complete")

    return SpatialResult(
        output_path=output,
        input_layout=input_layout,
        output_layout="binaural",
        channels=2,
        method=method,
    )


# ---------------------------------------------------------------------------
# Surround upmix pan filter definitions
# ---------------------------------------------------------------------------
# 5.1 (FL FR FC LFE BL BR) from stereo
_PAN_5_1 = (
    "pan=5.1|"
    "FL=0.6*FL+0.1*FR|"
    "FR=0.1*FL+0.6*FR|"
    "FC=0.35*FL+0.35*FR|"
    "LFE=0.25*FL+0.25*FR|"
    "BL=0.4*FL+0.15*FR|"
    "BR=0.15*FL+0.4*FR"
)

# 7.1 (FL FR FC LFE BL BR SL SR) from stereo
_PAN_7_1 = (
    "pan=7.1|"
    "FL=0.5*FL+0.1*FR|"
    "FR=0.1*FL+0.5*FR|"
    "FC=0.3*FL+0.3*FR|"
    "LFE=0.2*FL+0.2*FR|"
    "BL=0.35*FL+0.1*FR|"
    "BR=0.1*FL+0.35*FR|"
    "SL=0.3*FL+0.15*FR|"
    "SR=0.15*FL+0.3*FR"
)


def to_surround(
    audio_path: str,
    channels: int = 6,
    output: Optional[str] = None,
    lfe_cutoff: int = 120,
    on_progress: Optional[Callable] = None,
) -> SpatialResult:
    """Upmix stereo audio to 5.1 or 7.1 surround using FFmpeg pan filter.

    Args:
        audio_path: Path to the input audio or video file.
        channels: Target channel count: 6 for 5.1, 8 for 7.1.
        output: Output file path. Auto-generated if None.
        lfe_cutoff: LFE low-pass cutoff frequency in Hz.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        SpatialResult with output path and conversion details.
    """
    if channels not in (6, 8):
        raise ValueError(f"Unsupported channel count: {channels}. Use 6 (5.1) or 8 (7.1).")

    if on_progress:
        on_progress(5, "Detecting input channel layout...")

    layout_info = detect_channel_layout(audio_path)
    input_layout = layout_info["channel_layout"]

    layout_name = "5.1" if channels == 6 else "7.1"
    suffix = layout_name.replace(".", "")

    if output is None:
        output = output_path(audio_path, f"surround_{suffix}")

    _, ext = os.path.splitext(output)
    if not ext:
        output += ".wav"

    if on_progress:
        on_progress(20, f"Upmixing to {layout_name}...")

    pan_filter = _PAN_5_1 if channels == 6 else _PAN_7_1

    # Add LFE lowpass filter
    af = f"{pan_filter},lowpass=f={lfe_cutoff}:c=LFE"

    cmd = (
        FFmpegCmd()
        .input(audio_path)
        .audio_filter(af)
        .audio_codec("pcm_s16le")
        .option("ac", str(channels))
        .output(output)
        .build()
    )

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(95, f"{layout_name} upmix complete")

    return SpatialResult(
        output_path=output,
        input_layout=input_layout,
        output_layout=layout_name,
        channels=channels,
        method="pan_upmix",
    )
