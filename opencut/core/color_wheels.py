"""
OpenCut Three-Way Color Wheels (13.2)

Lift/Gamma/Gain color grading via FFmpeg colorbalance and eq filters:
- Per-zone (shadows/midtones/highlights) RGB adjustments
- Global offset control
- Single-frame preview mode
- Full video rendering with progress

Maps traditional color wheel UI values to FFmpeg colorbalance parameters.
"""

import logging
import os
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional, Tuple

from opencut.helpers import FFmpegCmd, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ColorWheelSettings:
    """Settings for three-way color wheel grading.

    Each value is (red, green, blue) in range -1.0 to 1.0.
    Offset is a global (red, green, blue) shift applied after grading.
    """
    lift: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gamma: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gain: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    saturation: float = 1.0
    contrast: float = 1.0
    brightness: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def is_neutral(self) -> bool:
        """Return True if all settings are at default (no grading)."""
        zero = (0.0, 0.0, 0.0)
        return (
            self.lift == zero
            and self.gamma == zero
            and self.gain == zero
            and self.offset == zero
            and self.saturation == 1.0
            and self.contrast == 1.0
            and self.brightness == 0.0
        )


@dataclass
class ColorWheelResult:
    """Result of a color wheel grading operation."""
    output_path: str = ""
    settings: Optional[Dict] = None
    preview: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _build_colorbalance_filter(settings: ColorWheelSettings) -> str:
    """Build FFmpeg colorbalance filter string from ColorWheelSettings."""
    lr, lg, lb = [_clamp(v) for v in settings.lift]
    gr, gg, gb = [_clamp(v) for v in settings.gamma]
    gnr, gng, gnb = [_clamp(v) for v in settings.gain]

    parts = []
    # Shadows (lift)
    parts.append(f"rs={lr}:gs={lg}:bs={lb}")
    # Midtones (gamma)
    parts.append(f"rm={gr}:gm={gg}:bm={gb}")
    # Highlights (gain)
    parts.append(f"rh={gnr}:gh={gng}:bh={gnb}")

    return "colorbalance=" + ":".join(parts)


def _build_eq_filter(settings: ColorWheelSettings) -> Optional[str]:
    """Build FFmpeg eq filter for saturation/contrast/brightness."""
    parts = []
    if settings.saturation != 1.0:
        parts.append(f"saturation={_clamp(settings.saturation, 0.0, 3.0)}")
    if settings.contrast != 1.0:
        parts.append(f"contrast={_clamp(settings.contrast, 0.0, 3.0)}")
    if settings.brightness != 0.0:
        parts.append(f"brightness={_clamp(settings.brightness, -1.0, 1.0)}")
    if parts:
        return "eq=" + ":".join(parts)
    return None


def _build_offset_filter(settings: ColorWheelSettings) -> Optional[str]:
    """Build FFmpeg colorchannelmixer for global offset."""
    r, g, b = settings.offset
    if r == 0.0 and g == 0.0 and b == 0.0:
        return None
    # Add offset to each channel's own contribution
    rr = _clamp(1.0 + r, 0.0, 2.0)
    gg = _clamp(1.0 + g, 0.0, 2.0)
    bb = _clamp(1.0 + b, 0.0, 2.0)
    return f"colorchannelmixer=rr={rr}:gg={gg}:bb={bb}"


def _build_filter_chain(settings: ColorWheelSettings) -> str:
    """Combine all color wheel filters into a single chain."""
    filters = []
    filters.append(_build_colorbalance_filter(settings))
    eq = _build_eq_filter(settings)
    if eq:
        filters.append(eq)
    offset = _build_offset_filter(settings)
    if offset:
        filters.append(offset)
    return ",".join(filters)


def _parse_settings(lift=None, gamma=None, gain=None, offset=None,
                    settings=None, **kwargs) -> ColorWheelSettings:
    """Parse color wheel settings from either a dataclass or raw tuples."""
    if isinstance(settings, ColorWheelSettings):
        return settings
    if isinstance(settings, dict):
        return ColorWheelSettings(
            lift=tuple(settings.get("lift", (0, 0, 0))),
            gamma=tuple(settings.get("gamma", (0, 0, 0))),
            gain=tuple(settings.get("gain", (0, 0, 0))),
            offset=tuple(settings.get("offset", (0, 0, 0))),
            saturation=float(settings.get("saturation", 1.0)),
            contrast=float(settings.get("contrast", 1.0)),
            brightness=float(settings.get("brightness", 0.0)),
        )
    return ColorWheelSettings(
        lift=tuple(lift or (0, 0, 0)),
        gamma=tuple(gamma or (0, 0, 0)),
        gain=tuple(gain or (0, 0, 0)),
        offset=tuple(offset or (0, 0, 0)),
        saturation=float(kwargs.get("saturation", 1.0)),
        contrast=float(kwargs.get("contrast", 1.0)),
        brightness=float(kwargs.get("brightness", 0.0)),
    )


# ---------------------------------------------------------------------------
# Apply Color Wheels (full video)
# ---------------------------------------------------------------------------
def apply_color_wheels(
    video_path: str,
    lift: Optional[Tuple[float, float, float]] = None,
    gamma: Optional[Tuple[float, float, float]] = None,
    gain: Optional[Tuple[float, float, float]] = None,
    offset: Optional[Tuple[float, float, float]] = None,
    output_path: Optional[str] = None,
    output_dir: str = "",
    settings: Optional[ColorWheelSettings] = None,
    on_progress: Optional[Callable] = None,
    **kwargs,
) -> ColorWheelResult:
    """
    Apply three-way color wheel grading to a full video.

    Args:
        video_path: Source video file.
        lift: (R, G, B) shadow adjustment, each -1.0 to 1.0.
        gamma: (R, G, B) midtone adjustment.
        gain: (R, G, B) highlight adjustment.
        offset: (R, G, B) global offset.
        output_path: Explicit output file path.
        output_dir: Output directory (auto-named).
        settings: ColorWheelSettings dataclass (overrides individual args).
        on_progress: Callback(percent, message).

    Returns:
        ColorWheelResult with output path and applied settings.
    """
    cw = _parse_settings(lift, gamma, gain, offset, settings, **kwargs)

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_color_graded.mp4")

    if on_progress:
        on_progress(10, "Applying color wheels...")

    vf = _build_filter_chain(cw)

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Color wheels applied.")

    return ColorWheelResult(
        output_path=output_path,
        settings=cw.to_dict(),
        preview=False,
    )


# ---------------------------------------------------------------------------
# Preview Color Wheels (single frame)
# ---------------------------------------------------------------------------
def preview_color_wheels(
    video_path: str,
    timestamp: float = 0.0,
    settings: Optional[ColorWheelSettings] = None,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
    **kwargs,
) -> ColorWheelResult:
    """
    Generate a single-frame preview of color wheel settings.

    Args:
        video_path: Source video file.
        timestamp: Frame time in seconds.
        settings: ColorWheelSettings (or dict).
        output_path: Explicit output file path.
        on_progress: Callback(percent, message).

    Returns:
        ColorWheelResult with preview image path.
    """
    cw = _parse_settings(settings=settings, **kwargs)

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_cw_preview.png")

    if on_progress:
        on_progress(10, "Generating color wheel preview...")

    vf = _build_filter_chain(cw)

    cmd = (
        FFmpegCmd()
        .pre_input("-ss", str(timestamp))
        .input(video_path)
        .video_filter(vf)
        .frames(1)
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Color wheel preview generated.")

    return ColorWheelResult(
        output_path=output_path,
        settings=cw.to_dict(),
        preview=True,
    )
