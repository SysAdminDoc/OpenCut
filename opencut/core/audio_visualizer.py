"""
OpenCut Audio-Reactive Visualizers v1.0.0

Generate audio-reactive video visualizations:
- Waveform bars
- Circular spectrum
- Frequency waterfall
- Render as overlay on video or standalone

All processing uses FFmpeg — no additional model downloads required.
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import get_ffmpeg_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VISUALIZER_STYLES = {
    "waveform_bars": {
        "label": "Waveform Bars",
        "description": "Vertical bars reacting to audio amplitude",
        "filter": "showwaves",
    },
    "spectrum_circle": {
        "label": "Circular Spectrum",
        "description": "Circular frequency spectrum display",
        "filter": "showfreqs",
    },
    "waterfall": {
        "label": "Frequency Waterfall",
        "description": "Scrolling spectrogram / waterfall display",
        "filter": "showspectrum",
    },
    "waveform_line": {
        "label": "Waveform Line",
        "description": "Classic oscilloscope waveform",
        "filter": "showwaves",
    },
    "volume_meter": {
        "label": "Volume Meter",
        "description": "VU meter style visualization",
        "filter": "showvolume",
    },
}

DEFAULT_COLORS = {
    "primary": "0x00AAFF",
    "secondary": "0xFF6600",
    "background": "0x1A1A2E",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class VisualizerConfig:
    """Configuration for audio visualizer generation."""
    style: str = "waveform_bars"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    color_primary: str = "0x00AAFF"
    color_secondary: str = "0xFF6600"
    color_background: str = "0x1A1A2E"
    bar_count: int = 64
    opacity: float = 0.8
    as_overlay: bool = False
    overlay_position: str = "bottom"  # top, bottom, left, right, center
    overlay_height_ratio: float = 0.2  # fraction of video height for overlay


@dataclass
class VisualizerResult:
    """Result from visualizer generation."""
    output_path: str = ""
    style: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0
    fps: int = 30


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def generate_visualizer(
    audio_path: str,
    style: str = "waveform_bars",
    output_path_val: Optional[str] = None,
    config: Optional[VisualizerConfig] = None,
    video_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> VisualizerResult:
    """
    Generate an audio-reactive visualizer video.

    Args:
        audio_path: Source audio file.
        style: Visualizer style (see VISUALIZER_STYLES).
        output_path_val: Output video path (auto-generated if None).
        config: VisualizerConfig with rendering settings.
        video_path: Optional video to overlay the visualizer on.
        on_progress: Progress callback(pct, msg).

    Returns:
        VisualizerResult with output path and metadata.
    """
    config = config or VisualizerConfig(style=style)
    config.style = style

    if style not in VISUALIZER_STYLES:
        raise ValueError(
            f"Unknown visualizer style: {style}. "
            f"Available: {', '.join(VISUALIZER_STYLES.keys())}"
        )

    if on_progress:
        on_progress(5, f"Generating {VISUALIZER_STYLES[style]['label']}...")

    if output_path_val is None:
        output_path_val = output_path(audio_path, f"_viz_{style}")
        # Force .mp4 extension for video output
        base, _ = os.path.splitext(output_path_val)
        output_path_val = base + ".mp4"

    if style == "waveform_bars":
        result = create_waveform_bars(audio_path, output_path_val, config, on_progress)
    elif style == "spectrum_circle":
        result = create_spectrum_circle(audio_path, output_path_val, config, on_progress)
    elif style == "waterfall":
        result = _create_waterfall(audio_path, output_path_val, config, on_progress)
    elif style == "waveform_line":
        result = _create_waveform_line(audio_path, output_path_val, config, on_progress)
    elif style == "volume_meter":
        result = _create_volume_meter(audio_path, output_path_val, config, on_progress)
    else:
        result = create_waveform_bars(audio_path, output_path_val, config, on_progress)

    # Overlay on video if requested
    if video_path and config.as_overlay:
        if on_progress:
            on_progress(80, "Overlaying visualizer on video...")
        result = _overlay_on_video(
            video_path, result.output_path, output_path_val, config, on_progress,
        )

    return result


# ---------------------------------------------------------------------------
# Waveform Bars
# ---------------------------------------------------------------------------
def create_waveform_bars(
    audio_path: str,
    output_path_val: Optional[str] = None,
    config: Optional[VisualizerConfig] = None,
    on_progress: Optional[Callable] = None,
) -> VisualizerResult:
    """
    Create a waveform bars visualization.

    Vertical bars that react to audio amplitude, similar to a bar-graph
    equalizer display.

    Args:
        audio_path: Source audio file.
        output_path_val: Output path (auto-generated if None).
        config: VisualizerConfig with rendering settings.
        on_progress: Progress callback(pct, msg).

    Returns:
        VisualizerResult with output path and metadata.
    """
    config = config or VisualizerConfig()
    ffmpeg = get_ffmpeg_path()

    if output_path_val is None:
        output_path_val = output_path(audio_path, "_waveform_bars")
        base, _ = os.path.splitext(output_path_val)
        output_path_val = base + ".mp4"

    if on_progress:
        on_progress(10, "Rendering waveform bars...")

    # FFmpeg showwaves with bars mode
    filter_str = (
        f"showwaves=s={config.width}x{config.height}"
        f":mode=cline:rate={config.fps}"
        f":colors={config.color_primary},"
        f"format=yuv420p"
    )

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-y", "-i", audio_path,
        "-filter_complex", filter_str,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path_val,
    ]
    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Waveform bars complete")

    return VisualizerResult(
        output_path=output_path_val,
        style="waveform_bars",
        width=config.width,
        height=config.height,
        fps=config.fps,
    )


# ---------------------------------------------------------------------------
# Spectrum Circle
# ---------------------------------------------------------------------------
def create_spectrum_circle(
    audio_path: str,
    output_path_val: Optional[str] = None,
    config: Optional[VisualizerConfig] = None,
    on_progress: Optional[Callable] = None,
) -> VisualizerResult:
    """
    Create a circular spectrum visualization.

    Frequency bands displayed in a circular arrangement, with magnitude
    shown as radial bars emanating from center.

    Args:
        audio_path: Source audio file.
        output_path_val: Output path (auto-generated if None).
        config: VisualizerConfig with rendering settings.
        on_progress: Progress callback(pct, msg).

    Returns:
        VisualizerResult with output path and metadata.
    """
    config = config or VisualizerConfig()
    ffmpeg = get_ffmpeg_path()

    if output_path_val is None:
        output_path_val = output_path(audio_path, "_spectrum_circle")
        base, _ = os.path.splitext(output_path_val)
        output_path_val = base + ".mp4"

    if on_progress:
        on_progress(10, "Rendering circular spectrum...")

    # Use showfreqs for circular-like frequency display
    filter_str = (
        f"showfreqs=s={config.width}x{config.height}"
        f":mode=bar:fscale=log"
        f":colors={config.color_primary}|{config.color_secondary},"
        f"format=yuv420p"
    )

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-y", "-i", audio_path,
        "-filter_complex", filter_str,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path_val,
    ]
    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Circular spectrum complete")

    return VisualizerResult(
        output_path=output_path_val,
        style="spectrum_circle",
        width=config.width,
        height=config.height,
        fps=config.fps,
    )


# ---------------------------------------------------------------------------
# Internal Styles
# ---------------------------------------------------------------------------
def _create_waterfall(
    audio_path: str,
    output_path_val: str,
    config: VisualizerConfig,
    on_progress: Optional[Callable] = None,
) -> VisualizerResult:
    """Create a scrolling spectrogram / frequency waterfall."""
    ffmpeg = get_ffmpeg_path()

    if on_progress:
        on_progress(10, "Rendering frequency waterfall...")

    filter_str = (
        f"showspectrum=s={config.width}x{config.height}"
        f":mode=combined:slide=scroll"
        f":color=intensity:scale=log,"
        f"format=yuv420p"
    )

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-y", "-i", audio_path,
        "-filter_complex", filter_str,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path_val,
    ]
    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Frequency waterfall complete")

    return VisualizerResult(
        output_path=output_path_val,
        style="waterfall",
        width=config.width,
        height=config.height,
        fps=config.fps,
    )


def _create_waveform_line(
    audio_path: str,
    output_path_val: str,
    config: VisualizerConfig,
    on_progress: Optional[Callable] = None,
) -> VisualizerResult:
    """Create a classic oscilloscope waveform line."""
    ffmpeg = get_ffmpeg_path()

    if on_progress:
        on_progress(10, "Rendering waveform line...")

    filter_str = (
        f"showwaves=s={config.width}x{config.height}"
        f":mode=line:rate={config.fps}"
        f":colors={config.color_primary},"
        f"format=yuv420p"
    )

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-y", "-i", audio_path,
        "-filter_complex", filter_str,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path_val,
    ]
    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Waveform line complete")

    return VisualizerResult(
        output_path=output_path_val,
        style="waveform_line",
        width=config.width,
        height=config.height,
        fps=config.fps,
    )


def _create_volume_meter(
    audio_path: str,
    output_path_val: str,
    config: VisualizerConfig,
    on_progress: Optional[Callable] = None,
) -> VisualizerResult:
    """Create a VU meter style volume visualization."""
    ffmpeg = get_ffmpeg_path()

    if on_progress:
        on_progress(10, "Rendering volume meter...")

    filter_str = (
        f"showvolume=w={config.width}:h=80:f=0.5"
        f":c={config.color_primary}:b=4,"
        f"scale={config.width}:{config.height},"
        f"format=yuv420p"
    )

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-y", "-i", audio_path,
        "-filter_complex", filter_str,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path_val,
    ]
    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Volume meter complete")

    return VisualizerResult(
        output_path=output_path_val,
        style="volume_meter",
        width=config.width,
        height=config.height,
        fps=config.fps,
    )


# ---------------------------------------------------------------------------
# Video Overlay
# ---------------------------------------------------------------------------
def _overlay_on_video(
    video_path: str,
    viz_path: str,
    output_path_val: str,
    config: VisualizerConfig,
    on_progress: Optional[Callable] = None,
) -> VisualizerResult:
    """Overlay a visualizer video on top of a source video."""
    ffmpeg = get_ffmpeg_path()

    # Calculate overlay dimensions and position
    overlay_h = int(config.height * config.overlay_height_ratio)

    position_map = {
        "bottom": f"0:H-{overlay_h}",
        "top": "0:0",
        "center": "(W-w)/2:(H-h)/2",
        "left": f"0:(H-{overlay_h})/2",
        "right": f"W-w:(H-{overlay_h})/2",
    }
    pos = position_map.get(config.overlay_position, f"0:H-{overlay_h}")

    temp_out = output_path_val + ".tmp.mp4"

    filter_complex = (
        f"[1:v]scale={config.width}:{overlay_h},"
        f"colorchannelmixer=aa={config.opacity}[viz];"
        f"[0:v][viz]overlay={pos}:shortest=1"
    )

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-y", "-i", video_path, "-i", viz_path,
        "-filter_complex", filter_complex,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        temp_out,
    ]
    run_ffmpeg(cmd, timeout=3600)

    # Move temp to final output
    if os.path.isfile(temp_out):
        os.replace(temp_out, output_path_val)

    # Clean up intermediate viz file if it differs from output
    if viz_path != output_path_val and os.path.isfile(viz_path):
        try:
            os.unlink(viz_path)
        except OSError:
            pass

    if on_progress:
        on_progress(100, "Overlay complete")

    return VisualizerResult(
        output_path=output_path_val,
        style=config.style,
        width=config.width,
        height=config.height,
        fps=config.fps,
    )
