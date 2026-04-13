"""
OpenCut Adjustment Layers Module

Apply saved correction stacks as non-destructive overlays spanning
timeline ranges.  Generates intermediate overlay clips via FFmpeg
filters, then composites them onto the source video for the
specified in/out range.

Correction types supported:
  - brightness, contrast, saturation (``eq`` filter)
  - hue_shift (``hue`` filter)
  - blur (``boxblur`` filter)
  - sharpen (``unsharp`` filter)
  - gamma (``eq`` gamma parameter)
  - temperature (``colorbalance`` filter)

Presets are stored as JSON files under ``~/.opencut/adjustment_presets/``.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_PRESETS_DIR = os.path.join(_OPENCUT_DIR, "adjustment_presets")


@dataclass
class Correction:
    """A single adjustment parameter."""
    type: str = ""
    value: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdjustmentLayerResult:
    """Result of adjustment layer creation / application."""
    output_path: str = ""
    corrections_applied: int = 0
    duration: float = 0.0
    filter_chain: str = ""


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------

def _build_filter_chain(corrections: List[Dict[str, Any]]) -> str:
    """Convert a list of correction dicts into an FFmpeg filter chain string."""
    filters: List[str] = []

    for c in corrections:
        ctype = str(c.get("type", "")).lower()
        value = float(c.get("value", 0))

        if ctype == "brightness":
            filters.append(f"eq=brightness={value:.3f}")
        elif ctype == "contrast":
            filters.append(f"eq=contrast={max(0.0, value):.3f}")
        elif ctype == "saturation":
            filters.append(f"eq=saturation={max(0.0, value):.3f}")
        elif ctype == "gamma":
            filters.append(f"eq=gamma={max(0.01, value):.3f}")
        elif ctype == "hue_shift":
            filters.append(f"hue=h={value:.1f}")
        elif ctype == "blur":
            radius = max(1, int(value))
            filters.append(f"boxblur={radius}:{radius}")
        elif ctype == "sharpen":
            amount = max(0.0, value)
            filters.append(f"unsharp=5:5:{amount:.2f}")
        elif ctype == "temperature":
            # Warm = positive, cool = negative
            rs = f"{max(-1.0, min(1.0, value * 0.3)):.2f}"
            bs = f"{max(-1.0, min(1.0, -value * 0.3)):.2f}"
            filters.append(f"colorbalance=rs={rs}:bs={bs}")
        else:
            logger.warning("Unknown correction type '%s', skipping", ctype)

    return ",".join(filters) if filters else "null"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_adjustment_layer(
    corrections: List[Dict[str, Any]],
    duration: float,
    out_path: str,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    on_progress: Optional[Callable] = None,
) -> AdjustmentLayerResult:
    """Create a transparent adjustment-layer clip with corrections baked in.

    Generates a solid-color overlay video with the requested filter
    corrections applied.  This clip can later be composited onto the
    timeline.

    Args:
        corrections: List of correction dicts, each with ``type`` and ``value``.
        duration: Duration in seconds.
        out_path: Output file path for the generated clip.
        width: Frame width.
        height: Frame height.
        fps: Frame rate.
        on_progress: Optional ``(pct, msg)`` callback.

    Returns:
        AdjustmentLayerResult with the output path and metadata.
    """
    if on_progress:
        on_progress(10, "Building adjustment filter chain...")

    filter_chain = _build_filter_chain(corrections)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    cmd = [
        get_ffmpeg_path(), "-y",
        "-f", "lavfi",
        "-i", f"color=c=gray:s={width}x{height}:r={fps}:d={duration}",
        "-vf", filter_chain,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        out_path,
    ]

    if on_progress:
        on_progress(40, "Rendering adjustment layer...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Adjustment layer created")

    logger.info("Created adjustment layer: %s (%d corrections, %.1fs)",
                out_path, len(corrections), duration)

    return AdjustmentLayerResult(
        output_path=out_path,
        corrections_applied=len(corrections),
        duration=duration,
        filter_chain=filter_chain,
    )


def apply_adjustment_to_range(
    video_path: str,
    layer_path: str,
    start: float,
    end: float,
    out_path: str = "",
    opacity: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> AdjustmentLayerResult:
    """Composite an adjustment layer onto a video for a time range.

    Args:
        video_path: Path to the source video.
        layer_path: Path to the adjustment layer clip.
        start: Start time in seconds.
        end: End time in seconds.
        out_path: Output file path (auto-generated if empty).
        opacity: Overlay opacity 0.0 - 1.0.
        on_progress: Optional progress callback.

    Returns:
        AdjustmentLayerResult with the composited output.
    """
    if on_progress:
        on_progress(10, "Preparing overlay composite...")

    if not out_path:
        out_path = output_path(video_path, "adjusted")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    opacity = max(0.0, min(1.0, opacity))
    enable_expr = f"between(t,{start},{end})"

    filter_complex = (
        f"[1:v]format=yuva420p,colorchannelmixer=aa={opacity:.2f}[adj];"
        f"[0:v][adj]overlay=0:0:enable='{enable_expr}'[out]"
    )

    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", video_path,
        "-i", layer_path,
        "-filter_complex", filter_complex,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        out_path,
    ]

    if on_progress:
        on_progress(40, "Compositing adjustment layer...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Adjustment applied")

    duration = end - start
    logger.info("Applied adjustment layer to %s (%.1f-%.1fs)", video_path, start, end)

    return AdjustmentLayerResult(
        output_path=out_path,
        corrections_applied=1,
        duration=duration,
    )


def save_adjustment_preset(
    corrections: List[Dict[str, Any]],
    name: str,
    description: str = "",
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Save a set of corrections as a reusable preset.

    Args:
        corrections: List of correction dicts.
        name: Preset name (used as filename).
        description: Optional human-readable description.
        on_progress: Optional progress callback.

    Returns:
        Dict with preset metadata and save path.
    """
    if on_progress:
        on_progress(30, "Saving adjustment preset...")

    os.makedirs(_PRESETS_DIR, exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    preset_path = os.path.join(_PRESETS_DIR, f"{safe_name}.json")

    preset_data = {
        "name": name,
        "description": description,
        "corrections": corrections,
        "version": 1,
    }

    with open(preset_path, "w", encoding="utf-8") as f:
        json.dump(preset_data, f, indent=2)

    if on_progress:
        on_progress(100, "Preset saved")

    logger.info("Saved adjustment preset '%s' to %s", name, preset_path)

    return {
        "name": name,
        "path": preset_path,
        "corrections_count": len(corrections),
        "description": description,
    }


def load_adjustment_preset(
    name: str,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Load an adjustment preset by name.

    Args:
        name: Preset name.
        on_progress: Optional progress callback.

    Returns:
        Preset data dict with ``corrections`` list.
    """
    if on_progress:
        on_progress(50, "Loading preset...")

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    preset_path = os.path.join(_PRESETS_DIR, f"{safe_name}.json")

    if not os.path.isfile(preset_path):
        raise FileNotFoundError(f"Preset not found: {name}")

    with open(preset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if on_progress:
        on_progress(100, "Preset loaded")

    return data


def list_adjustment_presets(
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """List all saved adjustment presets.

    Returns:
        List of preset summary dicts.
    """
    if on_progress:
        on_progress(50, "Listing presets...")

    if not os.path.isdir(_PRESETS_DIR):
        return []

    presets = []
    for fname in sorted(os.listdir(_PRESETS_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            fpath = os.path.join(_PRESETS_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            presets.append({
                "name": data.get("name", fname),
                "description": data.get("description", ""),
                "corrections_count": len(data.get("corrections", [])),
                "path": fpath,
            })
        except (json.JSONDecodeError, OSError):
            continue

    if on_progress:
        on_progress(100, "Done")

    return presets
