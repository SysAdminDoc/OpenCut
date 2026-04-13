"""
OpenCut Real-Time Color Scopes (13.1)

Generate professional color analysis scopes from video frames:
- Waveform monitor: luminance/RGB waveform display
- Vectorscope: chrominance distribution plot
- RGB parade: per-channel waveform side-by-side
- Histogram: luminance/RGB histogram chart

All scopes rendered via FFmpeg video filters to PNG output.
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Optional

from opencut.helpers import FFmpegCmd, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ScopeResult:
    """Result from a single scope generation."""
    scope_type: str = ""
    output_path: str = ""
    width: int = 0
    height: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AllScopesResult:
    """Result from generating all scopes at once."""
    scopes: Dict[str, ScopeResult] = field(default_factory=dict)
    timestamp: float = 0.0
    output_dir: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scopes"] = {k: v.to_dict() if hasattr(v, "to_dict") else v
                       for k, v in self.scopes.items()}
        return d


# ---------------------------------------------------------------------------
# Scope dimensions
# ---------------------------------------------------------------------------
SCOPE_WIDTH = 720
SCOPE_HEIGHT = 480


# ---------------------------------------------------------------------------
# Waveform Monitor
# ---------------------------------------------------------------------------
def generate_waveform(
    video_path: str,
    timestamp: float = 0.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    scope_width: int = SCOPE_WIDTH,
    scope_height: int = SCOPE_HEIGHT,
    mode: str = "lowpass",
    on_progress: Optional[Callable] = None,
) -> ScopeResult:
    """
    Generate a waveform monitor image from a video frame.

    Args:
        video_path: Source video file.
        timestamp: Time position in seconds to sample.
        output_path: Explicit output file path.
        scope_width: Width of output scope image.
        scope_height: Height of output scope image.
        mode: Waveform filter mode (lowpass, flat, aflat, chroma, color,
              acolor, xflat, yflat).
        on_progress: Callback(percent, message).

    Returns:
        ScopeResult with output path and dimensions.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_waveform.png")

    if on_progress:
        on_progress(10, "Generating waveform monitor...")

    valid_modes = ("lowpass", "flat", "aflat", "chroma", "color",
                   "acolor", "xflat", "yflat")
    if mode not in valid_modes:
        mode = "lowpass"

    vf = f"waveform=mode={mode}:filter=lowpass:scale=digital,scale={scope_width}:{scope_height}"

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
        on_progress(100, "Waveform monitor generated.")

    return ScopeResult(
        scope_type="waveform",
        output_path=output_path,
        width=scope_width,
        height=scope_height,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Vectorscope
# ---------------------------------------------------------------------------
def generate_vectorscope(
    video_path: str,
    timestamp: float = 0.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    scope_size: int = SCOPE_HEIGHT,
    color_mode: str = "color3",
    on_progress: Optional[Callable] = None,
) -> ScopeResult:
    """
    Generate a vectorscope image from a video frame.

    Args:
        video_path: Source video file.
        timestamp: Time position in seconds to sample.
        output_path: Explicit output file path.
        scope_size: Size of the vectorscope (square output).
        color_mode: Vectorscope color mode (color, color2, color3, color4,
                    color5).
        on_progress: Callback(percent, message).

    Returns:
        ScopeResult with output path and dimensions.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_vectorscope.png")

    if on_progress:
        on_progress(10, "Generating vectorscope...")

    valid_modes = ("color", "color2", "color3", "color4", "color5")
    if color_mode not in valid_modes:
        color_mode = "color3"

    vf = f"vectorscope=mode=color3:envelope=instant:graticule=green,scale={scope_size}:{scope_size}"

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
        on_progress(100, "Vectorscope generated.")

    return ScopeResult(
        scope_type="vectorscope",
        output_path=output_path,
        width=scope_size,
        height=scope_size,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# RGB Parade
# ---------------------------------------------------------------------------
def generate_rgb_parade(
    video_path: str,
    timestamp: float = 0.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    scope_width: int = SCOPE_WIDTH,
    scope_height: int = SCOPE_HEIGHT,
    on_progress: Optional[Callable] = None,
) -> ScopeResult:
    """
    Generate an RGB parade (per-channel waveform) image from a video frame.

    Args:
        video_path: Source video file.
        timestamp: Time position in seconds to sample.
        output_path: Explicit output file path.
        scope_width: Width of output scope image.
        scope_height: Height of output scope image.
        on_progress: Callback(percent, message).

    Returns:
        ScopeResult with output path and dimensions.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_rgb_parade.png")

    if on_progress:
        on_progress(10, "Generating RGB parade...")

    # Split into RGB channels displayed side by side
    vf = (
        f"waveform=mode=column:filter=lowpass:components=7"
        f":display=parade,scale={scope_width}:{scope_height}"
    )

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
        on_progress(100, "RGB parade generated.")

    return ScopeResult(
        scope_type="rgb_parade",
        output_path=output_path,
        width=scope_width,
        height=scope_height,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
def generate_histogram(
    video_path: str,
    timestamp: float = 0.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    scope_width: int = SCOPE_WIDTH,
    scope_height: int = SCOPE_HEIGHT,
    display_mode: str = "stack",
    on_progress: Optional[Callable] = None,
) -> ScopeResult:
    """
    Generate a histogram image from a video frame.

    Args:
        video_path: Source video file.
        timestamp: Time position in seconds to sample.
        output_path: Explicit output file path.
        scope_width: Width of output histogram image.
        scope_height: Height of output histogram image.
        display_mode: Histogram display mode (stack, parade, overlay).
        on_progress: Callback(percent, message).

    Returns:
        ScopeResult with output path and dimensions.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_histogram.png")

    if on_progress:
        on_progress(10, "Generating histogram...")

    valid_modes = ("stack", "parade", "overlay")
    if display_mode not in valid_modes:
        display_mode = "stack"

    # levels_mode 0 = linear, display_mode selects layout
    dm = {"stack": 0, "parade": 1, "overlay": 2}.get(display_mode, 0)
    vf = (
        f"histogram=display_mode={dm}:level_height={scope_height}"
        f",scale={scope_width}:{scope_height}"
    )

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
        on_progress(100, "Histogram generated.")

    return ScopeResult(
        scope_type="histogram",
        output_path=output_path,
        width=scope_width,
        height=scope_height,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# All Scopes at Once
# ---------------------------------------------------------------------------
def generate_all_scopes(
    video_path: str,
    timestamp: float = 0.0,
    output_dir: str = "",
    scope_width: int = SCOPE_WIDTH,
    scope_height: int = SCOPE_HEIGHT,
    on_progress: Optional[Callable] = None,
) -> AllScopesResult:
    """
    Generate all four scope types (waveform, vectorscope, RGB parade,
    histogram) from a single video frame.

    Args:
        video_path: Source video file.
        timestamp: Time position in seconds to sample.
        output_dir: Directory for output files.
        scope_width: Width of scope images.
        scope_height: Height of scope images.
        on_progress: Callback(percent, message).

    Returns:
        AllScopesResult containing all four ScopeResult entries.
    """
    if not output_dir:
        output_dir = os.path.dirname(video_path)
    os.makedirs(output_dir, exist_ok=True)

    result = AllScopesResult(timestamp=timestamp, output_dir=output_dir)
    generators = [
        ("waveform", generate_waveform),
        ("vectorscope", generate_vectorscope),
        ("rgb_parade", generate_rgb_parade),
        ("histogram", generate_histogram),
    ]
    total = len(generators)

    for idx, (name, gen_func) in enumerate(generators):
        if on_progress:
            pct = int((idx / total) * 80) + 10
            on_progress(pct, f"Generating {name}...")

        kwargs = dict(
            video_path=video_path,
            timestamp=timestamp,
            output_dir=output_dir,
        )
        if name != "vectorscope":
            kwargs["scope_width"] = scope_width
            kwargs["scope_height"] = scope_height
        else:
            kwargs["scope_size"] = scope_height

        scope = gen_func(**kwargs)
        result.scopes[name] = scope

    if on_progress:
        on_progress(100, "All scopes generated.")

    return result
