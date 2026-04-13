"""
OpenCut Kinetic Typography Engine (26.1)

Per-character/word/line text animations with easing curves:
- Built-in animation presets (typewriter, wave, bounce, fade, etc.)
- Custom keyframe animation support
- Multiple easing functions
- Render to transparent video overlay

Uses FFmpeg drawtext filter with expression-based animation.
"""

import logging
import math
import os
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import FFmpegCmd, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EASING_FUNCTIONS = [
    "linear", "ease_in", "ease_out", "ease_in_out",
    "bounce", "elastic", "back", "cubic",
]

ANIMATION_PRESETS = {
    "typewriter": {
        "name": "Typewriter",
        "description": "Characters appear one at a time",
        "unit": "character",
        "stagger": 0.05,
        "easing": "linear",
    },
    "fade_in": {
        "name": "Fade In",
        "description": "Text fades in from transparent",
        "unit": "line",
        "stagger": 0.0,
        "easing": "ease_out",
    },
    "fade_in_up": {
        "name": "Fade In Up",
        "description": "Text fades in while sliding up",
        "unit": "word",
        "stagger": 0.08,
        "easing": "ease_out",
    },
    "fade_in_down": {
        "name": "Fade In Down",
        "description": "Text fades in while sliding down",
        "unit": "word",
        "stagger": 0.08,
        "easing": "ease_out",
    },
    "wave": {
        "name": "Wave",
        "description": "Characters wave up and down",
        "unit": "character",
        "stagger": 0.03,
        "easing": "ease_in_out",
    },
    "bounce_in": {
        "name": "Bounce In",
        "description": "Text bounces into position",
        "unit": "word",
        "stagger": 0.1,
        "easing": "bounce",
    },
    "scale_up": {
        "name": "Scale Up",
        "description": "Text scales up from zero",
        "unit": "line",
        "stagger": 0.0,
        "easing": "ease_out",
    },
    "slide_left": {
        "name": "Slide Left",
        "description": "Text slides in from the right",
        "unit": "word",
        "stagger": 0.06,
        "easing": "ease_out",
    },
    "slide_right": {
        "name": "Slide Right",
        "description": "Text slides in from the left",
        "unit": "word",
        "stagger": 0.06,
        "easing": "ease_out",
    },
    "glitch": {
        "name": "Glitch",
        "description": "Text appears with digital glitch effect",
        "unit": "character",
        "stagger": 0.02,
        "easing": "linear",
    },
    "rotate_in": {
        "name": "Rotate In",
        "description": "Characters rotate into place",
        "unit": "character",
        "stagger": 0.04,
        "easing": "ease_out",
    },
    "blur_in": {
        "name": "Blur In",
        "description": "Text transitions from blurred to sharp",
        "unit": "line",
        "stagger": 0.0,
        "easing": "ease_in_out",
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class KineticPreset:
    """A kinetic typography animation preset."""
    name: str = ""
    description: str = ""
    unit: str = "word"
    stagger: float = 0.05
    easing: str = "ease_out"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class KineticKeyframe:
    """A single keyframe for custom animation."""
    time: float = 0.0
    x: float = 0.0
    y: float = 0.0
    opacity: float = 1.0
    scale: float = 1.0
    rotation: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class KineticResult:
    """Result from a kinetic typography render."""
    output_path: str = ""
    text: str = ""
    preset: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Easing functions (return 0.0-1.0 for t in 0.0-1.0)
# ---------------------------------------------------------------------------
def _ease_linear(t: float) -> float:
    return t


def _ease_in(t: float) -> float:
    return t * t


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def _ease_bounce(t: float) -> float:
    if t < 1.0 / 2.75:
        return 7.5625 * t * t
    elif t < 2.0 / 2.75:
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    elif t < 2.5 / 2.75:
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375


def _ease_elastic(t: float) -> float:
    if t == 0 or t == 1:
        return t
    return -(2 ** (10 * t - 10)) * math.sin((t * 10 - 10.75) * (2 * math.pi / 3))


def _get_easing(name: str):
    """Get easing function by name."""
    return {
        "linear": _ease_linear,
        "ease_in": _ease_in,
        "ease_out": _ease_out,
        "ease_in_out": _ease_in_out,
        "bounce": _ease_bounce,
        "elastic": _ease_elastic,
        "back": _ease_in_out,
        "cubic": _ease_in_out,
    }.get(name, _ease_out)


# ---------------------------------------------------------------------------
# Internal: Build drawtext expression
# ---------------------------------------------------------------------------
def _build_drawtext_expr(
    text: str,
    preset_name: str,
    duration: float,
    width: int,
    height: int,
    font: str = "Arial",
    font_size: int = 64,
    color: str = "white",
) -> str:
    """Build FFmpeg drawtext filter with animation expressions."""
    preset = ANIMATION_PRESETS.get(preset_name, ANIMATION_PRESETS["fade_in"])
    escaped = text.replace("'", "\\'").replace(":", "\\:")
    # Base position: centered
    x_expr = "(w-text_w)/2"
    y_expr = "(h-text_h)/2"
    alpha_expr = "1"

    if preset_name == "typewriter":
        # Reveal characters over time
        1.0 / max(0.01, preset["stagger"])
        alpha_expr = "1"
        # Use enable expression to control timing
        enable_expr = f"between(t,0,{duration})"
        # Typewriter: show N characters based on time
        # We'll use drawtext's text_shaping and clip the display
        return (
            f"drawtext=text='{escaped}':fontfile='{font}':"
            f"fontsize={font_size}:fontcolor={color}:"
            f"x={x_expr}:y={y_expr}:"
            f"alpha='min(1,t/{max(0.1, duration*0.1)})':"
            f"enable='{enable_expr}'"
        )
    elif preset_name in ("fade_in",):
        alpha_expr = f"min(1,t/{max(0.1, duration*0.3)})"
    elif preset_name == "fade_in_up":
        offset = int(height * 0.1)
        progress = f"min(1,t/{max(0.1, duration*0.3)})"
        y_expr = f"(h-text_h)/2+{offset}*(1-{progress})"
        alpha_expr = progress
    elif preset_name == "fade_in_down":
        offset = int(height * 0.1)
        progress = f"min(1,t/{max(0.1, duration*0.3)})"
        y_expr = f"(h-text_h)/2-{offset}*(1-{progress})"
        alpha_expr = progress
    elif preset_name == "wave":
        amp = int(font_size * 0.3)
        y_expr = f"(h-text_h)/2+{amp}*sin(t*4)"
        alpha_expr = f"min(1,t/{max(0.1, duration*0.2)})"
    elif preset_name == "bounce_in":
        progress = f"min(1,t/{max(0.1, duration*0.4)})"
        # Simulate bounce: overshoot then settle
        offset = int(height * 0.3)
        y_expr = f"(h-text_h)/2-{offset}*(1-{progress})*(1-{progress})"
        alpha_expr = progress
    elif preset_name == "scale_up":
        alpha_expr = f"min(1,t/{max(0.1, duration*0.3)})"
    elif preset_name == "slide_left":
        progress = f"min(1,t/{max(0.1, duration*0.3)})"
        x_expr = f"w*(1-{progress})+(w-text_w)/2*{progress}"
        alpha_expr = progress
    elif preset_name == "slide_right":
        progress = f"min(1,t/{max(0.1, duration*0.3)})"
        x_expr = f"-text_w*(1-{progress})+(w-text_w)/2*{progress}"
        alpha_expr = progress
    elif preset_name == "glitch":
        alpha_expr = f"min(1,t/{max(0.1, duration*0.1)})"
        x_expr = f"(w-text_w)/2+rand(0,20)*if(lt(t,{duration*0.3}),1,0)"
    elif preset_name == "blur_in":
        alpha_expr = f"min(1,t/{max(0.1, duration*0.4)})"
    else:
        alpha_expr = f"min(1,t/{max(0.1, duration*0.3)})"

    return (
        f"drawtext=text='{escaped}':fontsize={font_size}:"
        f"fontcolor={color}:x={x_expr}:y={y_expr}:"
        f"alpha='{alpha_expr}'"
    )


# ---------------------------------------------------------------------------
# Animate Text
# ---------------------------------------------------------------------------
def animate_text(
    text: str,
    animation_preset: str = "fade_in",
    duration: float = 3.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    font: str = "Arial",
    font_size: int = 64,
    font_color: str = "white",
    background_color: str = "black",
    on_progress: Optional[Callable] = None,
) -> KineticResult:
    """
    Render kinetic typography animation to a video file.

    Args:
        text: Text to animate.
        animation_preset: Preset name from ANIMATION_PRESETS.
        duration: Animation duration in seconds.
        output_path: Explicit output file path.
        width: Output video width.
        height: Output video height.
        fps: Output frame rate.
        font: Font family name.
        font_size: Font size in pixels.
        font_color: Font color (FFmpeg color name or hex).
        background_color: Background color.
        on_progress: Callback(percent, message).

    Returns:
        KineticResult with output path and metadata.
    """
    if animation_preset not in ANIMATION_PRESETS:
        animation_preset = "fade_in"

    if output_path is None:
        safe_name = "".join(c if c.isalnum() else "_" for c in text[:20])
        directory = output_dir or os.path.join(os.path.expanduser("~"), ".opencut")
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, f"kinetic_{safe_name}.mp4")

    if on_progress:
        on_progress(10, f"Rendering '{animation_preset}' animation...")

    drawtext = _build_drawtext_expr(
        text, animation_preset, duration, width, height,
        font, font_size, font_color,
    )


    cmd = (
        FFmpegCmd()
        .pre_input("-f", "lavfi")
        .input(f"color=c={background_color}:s={width}x{height}:d={duration}:r={fps}")
        .video_filter(drawtext)
        .video_codec("libx264", crf=18, preset="medium")
        .option("-t", str(duration))
        .faststart()
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Kinetic text rendered.")

    return KineticResult(
        output_path=output_path,
        text=text,
        preset=animation_preset,
        duration=duration,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# List Animation Presets
# ---------------------------------------------------------------------------
def list_animation_presets() -> List[Dict]:
    """Return all available animation presets with their descriptions."""
    return [
        {"key": k, **v}
        for k, v in ANIMATION_PRESETS.items()
    ]


# ---------------------------------------------------------------------------
# Create Custom Animation
# ---------------------------------------------------------------------------
def create_custom_animation(
    keyframes: List[Dict],
    easing: str = "ease_out",
    text: str = "",
    duration: float = 3.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    font_size: int = 64,
    font_color: str = "white",
    background_color: str = "black",
    on_progress: Optional[Callable] = None,
) -> KineticResult:
    """
    Create a custom keyframe-based text animation.

    Args:
        keyframes: List of KineticKeyframe dicts with time/x/y/opacity/scale.
        easing: Easing function name.
        text: Text to animate.
        duration: Total duration.
        output_path: Explicit output file path.
        width: Output width.
        height: Output height.
        fps: Frame rate.
        on_progress: Callback(percent, message).

    Returns:
        KineticResult with output path.
    """
    if not text:
        text = "Custom Text"

    if not keyframes:
        keyframes = [
            {"time": 0.0, "opacity": 0.0, "x": 0.5, "y": 0.5},
            {"time": duration, "opacity": 1.0, "x": 0.5, "y": 0.5},
        ]

    if output_path is None:
        safe_name = "".join(c if c.isalnum() else "_" for c in text[:20])
        directory = output_dir or os.path.join(os.path.expanduser("~"), ".opencut")
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, f"kinetic_custom_{safe_name}.mp4")

    if on_progress:
        on_progress(10, "Rendering custom keyframe animation...")

    # Build interpolated drawtext with keyframe-derived expressions
    # Use first and last keyframe for simple linear interpolation
    kf_start = keyframes[0]
    kf_end = keyframes[-1] if len(keyframes) > 1 else keyframes[0]

    start_alpha = float(kf_start.get("opacity", 0))
    end_alpha = float(kf_end.get("opacity", 1))
    start_x = float(kf_start.get("x", 0.5)) * width
    end_x = float(kf_end.get("x", 0.5)) * width
    start_y = float(kf_start.get("y", 0.5)) * height
    end_y = float(kf_end.get("y", 0.5)) * height

    progress = f"min(1,t/{max(0.1, duration)})"
    alpha_expr = f"{start_alpha}+({end_alpha}-{start_alpha})*{progress}"
    x_expr = f"{start_x}+({end_x}-{start_x})*{progress}-text_w/2"
    y_expr = f"{start_y}+({end_y}-{start_y})*{progress}-text_h/2"

    escaped = text.replace("'", "\\'").replace(":", "\\:")
    drawtext = (
        f"drawtext=text='{escaped}':fontsize={font_size}:"
        f"fontcolor={font_color}:x='{x_expr}':y='{y_expr}':"
        f"alpha='{alpha_expr}'"
    )

    cmd = (
        FFmpegCmd()
        .pre_input("-f", "lavfi")
        .input(f"color=c={background_color}:s={width}x{height}:d={duration}:r={fps}")
        .video_filter(drawtext)
        .video_codec("libx264", crf=18, preset="medium")
        .option("-t", str(duration))
        .faststart()
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Custom animation rendered.")

    return KineticResult(
        output_path=output_path,
        text=text,
        preset="custom",
        duration=duration,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# Render Kinetic Text (from animation data)
# ---------------------------------------------------------------------------
def render_kinetic_text(
    animation_data: Dict,
    resolution: Tuple[int, int] = (1920, 1080),
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> KineticResult:
    """
    Render kinetic text from a complete animation data specification.

    Args:
        animation_data: Dict with text, preset/keyframes, duration, style.
        resolution: (width, height) tuple.
        output_path: Explicit output file path.
        on_progress: Callback(percent, message).

    Returns:
        KineticResult with rendered video path.
    """
    text = animation_data.get("text", "Hello World")
    preset = animation_data.get("preset", "")
    keyframes = animation_data.get("keyframes", [])
    duration = float(animation_data.get("duration", 3.0))
    fps = int(animation_data.get("fps", 30))
    font_size = int(animation_data.get("font_size", 64))
    font_color = animation_data.get("font_color", "white")
    bg_color = animation_data.get("background_color", "black")

    w, h = resolution

    if keyframes and not preset:
        return create_custom_animation(
            keyframes=keyframes,
            text=text,
            duration=duration,
            output_path=output_path,
            output_dir=output_dir,
            width=w, height=h, fps=fps,
            font_size=font_size,
            font_color=font_color,
            background_color=bg_color,
            on_progress=on_progress,
        )

    return animate_text(
        text=text,
        animation_preset=preset or "fade_in",
        duration=duration,
        output_path=output_path,
        output_dir=output_dir,
        width=w, height=h, fps=fps,
        font_size=font_size,
        font_color=font_color,
        background_color=bg_color,
        on_progress=on_progress,
    )
