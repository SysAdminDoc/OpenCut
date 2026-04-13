"""
OpenCut Audiogram Generator

Creates animated audio visualizations (audiograms) for podcasts and
music clips using FFmpeg showwaves/showspectrum filters with optional
artwork, title, and caption overlays.
"""

import json
import logging
import os
import re
import subprocess
from typing import Callable, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg

logger = logging.getLogger("opencut")

# Supported visualization styles
AUDIOGRAM_STYLES = ("bars", "wave", "circular")

# Aspect ratio presets
_ASPECT_PRESETS = {
    "square": (1080, 1080),
    "vertical": (1080, 1920),
    "landscape": (1920, 1080),
}


def _get_audio_duration(filepath: str) -> float:
    """Get audio file duration in seconds via ffprobe."""
    try:
        cmd = [
            get_ffprobe_path(), "-v", "quiet",
            "-print_format", "json", "-show_format",
            filepath,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0.0))
    except Exception:
        pass
    return 0.0


def _escape_ffmpeg_text(text: str) -> str:
    """Escape special characters for FFmpeg drawtext filter."""
    # FFmpeg drawtext requires escaping these characters
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("%", "%%")
    return text


def generate_audiogram(
    audio_path: str,
    output_path_str: Optional[str] = None,
    style: str = "bars",
    width: int = 1080,
    height: int = 1080,
    bg_color: str = "#1a1a2e",
    wave_color: str = "#e94560",
    duration: Optional[float] = None,
    artwork_path: Optional[str] = None,
    title_text: Optional[str] = None,
    caption_srt: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Generate an audiogram video from an audio file.

    Composites layers: background color -> artwork (centered) -> waveform
    animation -> title text -> optional SRT captions.

    Args:
        audio_path: Source audio file (MP3, WAV, FLAC, etc.).
        output_path_str: Destination video path. Auto-generated if None.
        style: Visualization style: "bars", "wave", or "circular".
        width: Video width.
        height: Video height.
        bg_color: Background color (hex or name).
        wave_color: Waveform/bars color (hex or name).
        duration: Limit duration in seconds. None = full audio length.
        artwork_path: Optional album art / logo image to center.
        title_text: Optional title text overlaid at top.
        caption_srt: Optional path to SRT subtitle file for captions.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, duration, style, dimensions.
    """
    if on_progress:
        on_progress(5, "Preparing audiogram...")

    if style not in AUDIOGRAM_STYLES:
        logger.warning("Unknown audiogram style %s, defaulting to bars", style)
        style = "bars"

    # Get audio duration
    audio_duration = _get_audio_duration(audio_path)
    if duration is not None:
        audio_duration = min(audio_duration, float(duration))
    if audio_duration <= 0:
        raise ValueError("Could not determine audio duration or audio is empty")

    # Generate output path
    if output_path_str is None:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_path_str = os.path.join(
            os.path.dirname(audio_path),
            f"{base}_audiogram.mp4",
        )

    ffmpeg = get_ffmpeg_path()

    if on_progress:
        on_progress(15, f"Building {style} visualization...")

    # --- Build filter complex ---
    inputs = []
    input_idx = 0

    # Input 0: solid color background
    bg_color.lstrip("#") if bg_color.startswith("#") else bg_color
    inputs.extend([
        "-f", "lavfi", "-i",
        f"color=c={bg_color}:s={width}x{height}:d={audio_duration:.3f}:r=30",
    ])
    bg_input = input_idx
    input_idx += 1

    # Input 1: audio file
    inputs.extend(["-i", audio_path])
    audio_input = input_idx
    input_idx += 1

    # Optional artwork input
    artwork_input = None
    if artwork_path and os.path.isfile(artwork_path):
        inputs.extend(["-i", artwork_path])
        artwork_input = input_idx
        input_idx += 1

    # Waveform visualization dimensions
    # Place waveform in lower portion of frame
    wave_h = height // 4
    wave_w = width

    # Clean wave color for FFmpeg
    wave_color_clean = wave_color.lstrip("#") if wave_color.startswith("#") else wave_color
    if re.match(r'^[0-9a-fA-F]{6}$', wave_color_clean):
        wave_color_ffmpeg = f"0x{wave_color_clean}"
    else:
        wave_color_ffmpeg = wave_color

    # Build the filter chain
    current_label = f"[{bg_input}:v]"
    fc_parts = []

    # Artwork overlay (if provided)
    if artwork_input is not None:
        art_size = min(width, height) // 2
        art_y = height // 6
        art_x = (width - art_size) // 2
        fc_parts.append(
            f"[{artwork_input}:v]scale={art_size}:{art_size}:force_original_aspect_ratio=decrease,"
            f"pad={art_size}:{art_size}:(ow-iw)/2:(oh-ih)/2:color=black@0.0,format=rgba[art];"
            f"{current_label}[art]overlay=x={art_x}:y={art_y}[bg_art]"
        )
        current_label = "[bg_art]"

    # Waveform generation based on style
    wave_y = height - wave_h - (height // 10)

    if style == "bars":
        # Frequency bars via showspectrum
        fc_parts.append(
            f"[{audio_input}:a]showspectrum=s={wave_w}x{wave_h}:mode=combined"
            f":color=fire:scale=cbrt:overlap=1:fscale=log[spectrum];"
            f"{current_label}[spectrum]overlay=x=0:y={wave_y}:shortest=1[waved]"
        )
        current_label = "[waved]"

    elif style == "wave":
        # Waveform line via showwaves
        fc_parts.append(
            f"[{audio_input}:a]showwaves=s={wave_w}x{wave_h}:mode=cline"
            f":rate=30:colors={wave_color_ffmpeg}[waves];"
            f"{current_label}[waves]overlay=x=0:y={wave_y}:shortest=1[waved]"
        )
        current_label = "[waved]"

    elif style == "circular":
        # Circular spectrum visualization via showspectrum in radial mode
        circ_size = min(width, height) // 2
        circ_x = (width - circ_size) // 2
        circ_y = (height - circ_size) // 2
        fc_parts.append(
            f"[{audio_input}:a]showspectrum=s={circ_size}x{circ_size}:mode=combined"
            f":color=cool:scale=sqrt:overlap=1:fscale=log[circ];"
            f"{current_label}[circ]overlay=x={circ_x}:y={circ_y}:shortest=1[waved]"
        )
        current_label = "[waved]"

    # Title text overlay
    if title_text:
        escaped = _escape_ffmpeg_text(title_text[:200])
        font_size = max(24, width // 25)
        title_y = height // 20
        fc_parts.append(
            f"{current_label}drawtext=text='{escaped}':"
            f"fontsize={font_size}:fontcolor=white:x=(w-text_w)/2:y={title_y}:"
            f"shadowcolor=black@0.6:shadowx=2:shadowy=2[titled]"
        )
        current_label = "[titled]"

    # SRT caption overlay
    if caption_srt and os.path.isfile(caption_srt):
        # Escape path for FFmpeg filter (backslashes and colons)
        srt_escaped = caption_srt.replace("\\", "/").replace(":", "\\:")
        caption_font_size = max(20, width // 30)
        fc_parts.append(
            f"{current_label}subtitles='{srt_escaped}':"
            f"force_style='FontSize={caption_font_size},Alignment=2,"
            f"MarginV={height // 8}'[captioned]"
        )
        current_label = "[captioned]"

    # Join filter complex
    filter_complex = ";".join(fc_parts)

    # Strip brackets from final label for -map
    final_label = current_label.strip("[]")

    if on_progress:
        on_progress(30, "Encoding audiogram video...")

    # Build full command
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
    ]
    cmd.extend(inputs)

    if duration is not None:
        cmd.extend(["-t", f"{audio_duration:.3f}"])

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", f"[{final_label}]",
        "-map", f"{audio_input}:a",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        output_path_str,
    ])

    if on_progress:
        on_progress(40, f"Rendering {style} audiogram ({audio_duration:.1f}s)...")

    run_ffmpeg(cmd, timeout=max(600, int(audio_duration * 10)))

    if on_progress:
        on_progress(100, "Audiogram complete")

    return {
        "output_path": output_path_str,
        "duration": round(audio_duration, 2),
        "style": style,
        "width": width,
        "height": height,
        "has_artwork": artwork_path is not None and os.path.isfile(artwork_path or ""),
        "has_title": bool(title_text),
        "has_captions": caption_srt is not None and os.path.isfile(caption_srt or ""),
    }
