"""
OpenCut Reaction Video Template

Reaction-specific presets (corner PiP, side panel, bottom bar).
Auto-sync reaction audio with content via cross-correlation.
Audio ducking for main content when reaction audio is active.
"""

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional

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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ReactionPreset:
    """Defines a reaction video layout preset."""
    name: str = ""
    description: str = ""
    # Content video placement (percentages)
    content_x: float = 0.0
    content_y: float = 0.0
    content_w: float = 100.0
    content_h: float = 100.0
    # Reaction video placement (percentages)
    reaction_x: float = 70.0
    reaction_y: float = 70.0
    reaction_w: float = 25.0
    reaction_h: float = 25.0


@dataclass
class ReactionResult:
    """Result from reaction video creation."""
    output_path: str = ""
    preset_name: str = ""
    duration: float = 0.0
    audio_offset: float = 0.0
    width: int = 0
    height: int = 0


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------
_REACTION_PRESETS: Dict[str, ReactionPreset] = {
    "corner_pip": ReactionPreset(
        name="corner_pip",
        description="Small reaction in bottom-right corner over content",
        content_x=0, content_y=0, content_w=100, content_h=100,
        reaction_x=70, reaction_y=70, reaction_w=25, reaction_h=25,
    ),
    "corner_pip_top": ReactionPreset(
        name="corner_pip_top",
        description="Small reaction in top-right corner over content",
        content_x=0, content_y=0, content_w=100, content_h=100,
        reaction_x=70, reaction_y=5, reaction_w=25, reaction_h=25,
    ),
    "side_panel": ReactionPreset(
        name="side_panel",
        description="Content left (65%), reaction right (35%)",
        content_x=0, content_y=0, content_w=65, content_h=100,
        reaction_x=65, reaction_y=0, reaction_w=35, reaction_h=100,
    ),
    "bottom_bar": ReactionPreset(
        name="bottom_bar",
        description="Content on top (70%), reaction bar at bottom (30%)",
        content_x=0, content_y=0, content_w=100, content_h=70,
        reaction_x=0, reaction_y=70, reaction_w=100, reaction_h=30,
    ),
    "top_bar": ReactionPreset(
        name="top_bar",
        description="Reaction bar on top (30%), content below (70%)",
        content_x=0, content_y=30, content_w=100, content_h=70,
        reaction_x=0, reaction_y=0, reaction_w=100, reaction_h=30,
    ),
    "large_pip": ReactionPreset(
        name="large_pip",
        description="Larger PiP reaction in bottom-right",
        content_x=0, content_y=0, content_w=100, content_h=100,
        reaction_x=55, reaction_y=55, reaction_w=40, reaction_h=40,
    ),
}


# ---------------------------------------------------------------------------
# Audio sync via cross-correlation
# ---------------------------------------------------------------------------
def _extract_audio_pcm(video_path: str, output_path: str, duration: float = 30.0) -> str:
    """Extract first N seconds of audio as raw PCM for correlation."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-t", str(duration),
        "-ac", "1", "-ar", "16000", "-f", "s16le",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)
    return output_path


def estimate_audio_offset(
    content_path: str,
    reaction_path: str,
    max_offset: float = 10.0,
) -> float:
    """
    Estimate time offset between content and reaction audio
    using cross-correlation of the first 30s.

    Returns offset in seconds (positive means reaction is ahead).
    """
    try:
        import struct

        temp_dir = tempfile.mkdtemp(prefix="opencut_sync_")
        pcm_a = os.path.join(temp_dir, "content.pcm")
        pcm_b = os.path.join(temp_dir, "reaction.pcm")

        _extract_audio_pcm(content_path, pcm_a)
        _extract_audio_pcm(reaction_path, pcm_b)

        # Read PCM samples
        def _read_samples(path):
            with open(path, "rb") as f:
                raw = f.read()
            count = len(raw) // 2
            return list(struct.unpack(f"<{count}h", raw[:count * 2]))

        samples_a = _read_samples(pcm_a)
        samples_b = _read_samples(pcm_b)

        if not samples_a or not samples_b:
            return 0.0

        sample_rate = 16000
        max_lag = int(max_offset * sample_rate)

        # Simple cross-correlation
        best_lag = 0
        best_corr = 0.0
        step = max(1, max_lag // 500)  # Coarse search

        min_len = min(len(samples_a), len(samples_b))
        for lag in range(-max_lag, max_lag + 1, step):
            corr = 0.0
            count = 0
            for j in range(0, min_len - abs(lag), 8):
                idx_a = j
                idx_b = j + lag
                if 0 <= idx_b < len(samples_b):
                    corr += samples_a[idx_a] * samples_b[idx_b]
                    count += 1
            if count > 0:
                corr /= count
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        offset = best_lag / sample_rate
        logger.info("Estimated audio offset: %.3fs (correlation: %.0f)", offset, best_corr)
        return offset

    except Exception as exc:
        logger.warning("Audio sync estimation failed: %s", exc)
        return 0.0
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_reaction_presets() -> Dict[str, dict]:
    """Return all reaction preset definitions as serializable dicts."""
    result = {}
    for name, preset in _REACTION_PRESETS.items():
        result[name] = {
            "name": preset.name,
            "description": preset.description,
            "content": {"x": preset.content_x, "y": preset.content_y,
                        "w": preset.content_w, "h": preset.content_h},
            "reaction": {"x": preset.reaction_x, "y": preset.reaction_y,
                         "w": preset.reaction_w, "h": preset.reaction_h},
        }
    return result


def create_reaction_video(
    content_path: str,
    reaction_path: str,
    preset_name: str = "corner_pip",
    output_width: int = 1920,
    output_height: int = 1080,
    auto_sync: bool = False,
    audio_offset: float = 0.0,
    duck_level: float = 0.3,
    border_width: int = 2,
    border_color: str = "white",
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> ReactionResult:
    """
    Create a reaction video composite.

    Args:
        content_path: Path to the content video.
        reaction_path: Path to the reaction video.
        preset_name: Reaction layout preset name.
        output_width: Canvas width.
        output_height: Canvas height.
        auto_sync: Auto-detect audio offset via cross-correlation.
        audio_offset: Manual audio offset in seconds.
        duck_level: Audio ducking level for content (0.0-1.0).
        border_width: Border width around reaction video.
        border_color: Border color.
        output_path_str: Explicit output path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        ReactionResult with output_path and metadata.
    """
    if not os.path.isfile(content_path):
        raise FileNotFoundError(f"Content video not found: {content_path}")
    if not os.path.isfile(reaction_path):
        raise FileNotFoundError(f"Reaction video not found: {reaction_path}")

    preset = _REACTION_PRESETS.get(preset_name)
    if not preset:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(_REACTION_PRESETS.keys())}"
        )

    if on_progress:
        on_progress(5, f"Preparing {preset_name} reaction video...")

    W, H = output_width, output_height

    # Auto-sync audio
    if auto_sync:
        if on_progress:
            on_progress(10, "Analyzing audio for sync...")
        audio_offset = estimate_audio_offset(content_path, reaction_path)

    if output_path_str is None:
        output_path_str = _output_path(content_path, f"reaction_{preset_name}")

    info = get_video_info(content_path)
    duration = info.get("duration", 0)

    if on_progress:
        on_progress(20, "Building reaction composite...")

    # Calculate pixel positions
    cx = int(preset.content_x / 100.0 * W)
    cy = int(preset.content_y / 100.0 * H)
    cw = int(preset.content_w / 100.0 * W)
    ch = int(preset.content_h / 100.0 * H)
    cw = cw - (cw % 2)
    ch = ch - (ch % 2)

    rx = int(preset.reaction_x / 100.0 * W)
    ry = int(preset.reaction_y / 100.0 * H)
    rw = int(preset.reaction_w / 100.0 * W)
    rh = int(preset.reaction_h / 100.0 * H)
    rw = rw - (rw % 2)
    rh = rh - (rh % 2)

    # Build filter_complex
    fc_parts = []

    # Background canvas
    fc_parts.append(f"color=c=black:s={W}x{H}:d={duration:.3f}:r=30[canvas]")

    # Scale content
    fc_parts.append(
        f"[0:v]scale={cw}:{ch}:force_original_aspect_ratio=decrease,"
        f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2[content]"
    )

    # Scale reaction with optional border
    if border_width > 0:
        inner_w = rw - 2 * border_width
        inner_h = rh - 2 * border_width
        inner_w = max(inner_w, 16) - (max(inner_w, 16) % 2)
        inner_h = max(inner_h, 16) - (max(inner_h, 16) % 2)
        fc_parts.append(
            f"[1:v]scale={inner_w}:{inner_h}:force_original_aspect_ratio=decrease,"
            f"pad={rw}:{rh}:(ow-iw)/2:(oh-ih)/2:color={border_color}[reaction]"
        )
    else:
        fc_parts.append(
            f"[1:v]scale={rw}:{rh}:force_original_aspect_ratio=decrease,"
            f"pad={rw}:{rh}:(ow-iw)/2:(oh-ih)/2[reaction]"
        )

    # Overlay content on canvas
    fc_parts.append(f"[canvas][content]overlay=x={cx}:y={cy}[comp1]")
    # Overlay reaction on top
    fc_parts.append(f"[comp1][reaction]overlay=x={rx}:y={ry}:shortest=1[outv]")

    # Audio mixing with ducking
    duck = max(0.0, min(1.0, duck_level))
    fc_parts.append(
        f"[0:a]volume={duck}[ca];"
        f"[1:a]adelay=0|0[ra];"
        f"[ca][ra]amix=inputs=2:duration=shortest[outa]"
    )

    filter_complex = ";".join(fc_parts)

    if on_progress:
        on_progress(40, "Encoding reaction video...")

    cmd = FFmpegCmd()
    cmd.input(content_path)
    if audio_offset > 0:
        cmd.input(reaction_path, itsoffset=str(audio_offset))
    elif audio_offset < 0:
        cmd.input(reaction_path, itsoffset=str(audio_offset))
    else:
        cmd.input(reaction_path)
    cmd.filter_complex(filter_complex, maps=["[outv]", "[outa]"])
    cmd.video_codec("libx264", crf=18, preset="fast")
    cmd.audio_codec("aac", bitrate="192k")
    cmd.option("shortest")
    cmd.faststart()
    cmd.output(output_path_str)

    run_ffmpeg(cmd.build(), timeout=3600)

    if on_progress:
        on_progress(100, "Reaction video complete")

    return ReactionResult(
        output_path=output_path_str,
        preset_name=preset_name,
        duration=duration,
        audio_offset=audio_offset,
        width=W,
        height=H,
    )
