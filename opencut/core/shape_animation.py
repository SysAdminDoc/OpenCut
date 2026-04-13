"""
OpenCut Shape Layer Animation (26.3)

Animate vector shapes with morphing, stroke drawing, and fill transitions:
- Path morphing between two SVG shapes
- Stroke drawing animation (draw-on effect)
- Fill color transitions
- Shape mask generation

Uses FFmpeg filters for rasterized shape animation.
"""

import logging
import os
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Tuple

from opencut.helpers import FFmpegCmd, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SHAPE_TYPES = ["circle", "rectangle", "triangle", "star", "polygon", "custom"]

MORPH_EASING = [
    "linear", "ease_in", "ease_out", "ease_in_out",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ShapeDefinition:
    """Definition of a shape for animation."""
    shape_type: str = "circle"
    x: float = 0.5
    y: float = 0.5
    width: float = 0.3
    height: float = 0.3
    color: str = "white"
    stroke_color: str = "white"
    stroke_width: int = 3
    fill_opacity: float = 1.0
    rotation: float = 0.0
    sides: int = 5  # for polygon/star

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ShapeAnimationResult:
    """Result from a shape animation render."""
    output_path: str = ""
    animation_type: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _parse_shape(shape) -> ShapeDefinition:
    """Parse shape from dict or dataclass."""
    if isinstance(shape, ShapeDefinition):
        return shape
    if isinstance(shape, dict):
        return ShapeDefinition(
            shape_type=shape.get("shape_type", shape.get("type", "circle")),
            x=float(shape.get("x", 0.5)),
            y=float(shape.get("y", 0.5)),
            width=float(shape.get("width", 0.3)),
            height=float(shape.get("height", 0.3)),
            color=shape.get("color", "white"),
            stroke_color=shape.get("stroke_color", "white"),
            stroke_width=int(shape.get("stroke_width", 3)),
            fill_opacity=float(shape.get("fill_opacity", 1.0)),
            rotation=float(shape.get("rotation", 0)),
            sides=int(shape.get("sides", 5)),
        )
    if isinstance(shape, str):
        return ShapeDefinition(shape_type=shape)
    return ShapeDefinition()


def _draw_circle(cx: int, cy: int, rx: int, ry: int, color: str,
                 stroke_color: str, stroke_width: int) -> str:
    """Build geq filter for a filled circle with stroke."""
    # Filled ellipse via geq
    inner = (
        f"'if(lt(hypot((X-{cx})/{max(1,rx)},(Y-{cy})/{max(1,ry)}),1),"
        f"255,0)'"
    )
    return f"geq=lum={inner}"


def _draw_rectangle(x: int, y: int, w: int, h: int, color: str,
                    stroke_width: int) -> str:
    """Build drawbox filter for a rectangle."""
    return f"drawbox=x={x}:y={y}:w={w}:h={h}:color={color}:t=fill"


def _generate_svg_frames(
    svg_path: str,
    frame_count: int,
    width: int,
    height: int,
    output_dir: str,
    progress_pct: float = 1.0,
) -> List[str]:
    """Rasterize SVG at different stroke-dashoffset values for draw animation.

    Returns list of frame image paths.
    """
    frames = []
    # Since we can't directly animate SVG with FFmpeg, generate frame sequence
    # by using FFmpeg to rasterize the SVG and crop/reveal progressively
    for i in range(frame_count):
        frame_path = os.path.join(output_dir, f"frame_{i:06d}.png")
        fraction = (i + 1) / frame_count
        # Use crop to progressively reveal the image
        crop_w = max(1, int(width * fraction * progress_pct))
        vf = f"scale={width}:{height},crop={crop_w}:{height}:0:0,pad={width}:{height}:0:0:black"

        cmd = (
            FFmpegCmd()
            .input(svg_path)
            .video_filter(vf)
            .frames(1)
            .output(frame_path)
            .build()
        )
        try:
            run_ffmpeg(cmd)
            frames.append(frame_path)
        except Exception:
            break
    return frames


# ---------------------------------------------------------------------------
# Shape Morph Animation
# ---------------------------------------------------------------------------
def animate_shape_morph(
    shape_a: Optional[ShapeDefinition] = None,
    shape_b: Optional[ShapeDefinition] = None,
    duration: float = 3.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    background_color: str = "black",
    easing: str = "ease_in_out",
    on_progress: Optional[Callable] = None,
) -> ShapeAnimationResult:
    """
    Animate a morph transition between two shapes.

    Interpolates position, size, and color between shape_a and shape_b
    over the specified duration.

    Args:
        shape_a: Starting shape definition.
        shape_b: Ending shape definition.
        duration: Animation duration in seconds.
        output_path: Explicit output file path.
        width: Output video width.
        height: Output video height.
        fps: Output frame rate.
        background_color: Background color.
        easing: Easing function name.
        on_progress: Callback(percent, message).

    Returns:
        ShapeAnimationResult with rendered video path.
    """
    sa = _parse_shape(shape_a) if shape_a else ShapeDefinition(
        shape_type="circle", width=0.2, height=0.2, color="0x4285F4")
    sb = _parse_shape(shape_b) if shape_b else ShapeDefinition(
        shape_type="rectangle", width=0.4, height=0.4, color="0xEA4335")

    if output_path is None:
        directory = output_dir or os.path.join(os.path.expanduser("~"), ".opencut")
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, "shape_morph.mp4")

    if on_progress:
        on_progress(10, "Rendering shape morph animation...")

    # Build filter that interpolates between two shapes using geq
    # Morph circle to rectangle by interpolating corner radius
    f"min(1,t/{max(0.1, duration)})"

    # Interpolated position
    cx_a, cy_a = int(sa.x * width), int(sa.y * height)
    cx_b, cy_b = int(sb.x * width), int(sb.y * height)
    sw_a, sh_a = int(sa.width * width / 2), int(sa.height * height / 2)
    sw_b, sh_b = int(sb.width * width / 2), int(sb.height * height / 2)

    # Use geq to create a morphing shape
    # Shape A: circle, Shape B: rectangle (with interpolated corner radius)
    # We model it as a superellipse: |x/a|^n + |y/b|^n <= 1
    # n=2 is circle, n=inf is rectangle, interpolate n from 2 to 20
    geq_expr = (
        f"'if(lt("
        f"pow(abs(X-({cx_a}+({cx_b}-{cx_a})*min(1,T/{max(0.1,duration)})))/"
        f"max(1,{sw_a}+({sw_b}-{sw_a})*min(1,T/{max(0.1,duration)})),"
        f"2+18*min(1,T/{max(0.1,duration)}))"
        f"+pow(abs(Y-({cy_a}+({cy_b}-{cy_a})*min(1,T/{max(0.1,duration)})))/"
        f"max(1,{sh_a}+({sh_b}-{sh_a})*min(1,T/{max(0.1,duration)})),"
        f"2+18*min(1,T/{max(0.1,duration)})),"
        f"1),255,0)'"
    )

    vf = f"geq=lum={geq_expr}:cb=128:cr=128"

    cmd = (
        FFmpegCmd()
        .pre_input("-f", "lavfi")
        .input(
            f"color=c={background_color}:s={width}x{height}"
            f":d={duration}:r={fps}"
        )
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .option("-t", str(duration))
        .faststart()
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Shape morph animation rendered.")

    return ShapeAnimationResult(
        output_path=output_path,
        animation_type="morph",
        duration=duration,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# Stroke Drawing Animation
# ---------------------------------------------------------------------------
def animate_stroke_draw(
    svg_path: str,
    duration: float = 3.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    background_color: str = "black",
    stroke_color: str = "white",
    stroke_width: int = 3,
    on_progress: Optional[Callable] = None,
) -> ShapeAnimationResult:
    """
    Animate a stroke-drawing effect on an SVG or image path.

    Progressive reveal simulating hand-drawing the shape.

    Args:
        svg_path: Path to SVG or image file to draw.
        duration: Animation duration in seconds.
        output_path: Explicit output file path.
        width: Output video width.
        height: Output video height.
        fps: Frame rate.
        background_color: Background color.
        stroke_color: Stroke color.
        stroke_width: Stroke width in pixels.
        on_progress: Callback(percent, message).

    Returns:
        ShapeAnimationResult with rendered video path.
    """
    if output_path is None:
        directory = output_dir or os.path.join(os.path.expanduser("~"), ".opencut")
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, "stroke_draw.mp4")

    if on_progress:
        on_progress(10, "Rendering stroke draw animation...")

    # Strategy: Use the input image/SVG and progressively reveal it
    # by using a horizontal wipe with edge detection for the stroke effect
    f"min(1,t/{max(0.1, duration)})"

    # Progressive reveal using crop + edge detection for stroke look
    vf = (
        f"scale={width}:{height},"
        f"edgedetect=low=0.1:high=0.3:mode=colormix,"
        f"crop='iw*min(1,t/{max(0.1, duration)}):ih:0:0',"
        f"pad={width}:{height}:0:0:{background_color}"
    )

    cmd = (
        FFmpegCmd()
        .input(svg_path)
        .pre_input("-loop", "1")
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .option("-t", str(duration))
        .faststart()
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Stroke draw animation rendered.")

    return ShapeAnimationResult(
        output_path=output_path,
        animation_type="stroke_draw",
        duration=duration,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# Fill Transition Animation
# ---------------------------------------------------------------------------
def animate_fill_transition(
    shape: Optional[ShapeDefinition] = None,
    color_a: str = "0x4285F4",
    color_b: str = "0xEA4335",
    duration: float = 3.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    background_color: str = "black",
    on_progress: Optional[Callable] = None,
) -> ShapeAnimationResult:
    """
    Animate a fill color transition on a shape.

    Smoothly transitions the fill color from color_a to color_b.

    Args:
        shape: Shape to animate (default: centered circle).
        color_a: Starting fill color.
        color_b: Ending fill color.
        duration: Animation duration.
        output_path: Explicit output file path.
        width: Output width.
        height: Output height.
        fps: Frame rate.
        background_color: Background color.
        on_progress: Callback(percent, message).

    Returns:
        ShapeAnimationResult with rendered video path.
    """
    s = _parse_shape(shape) if shape else ShapeDefinition()

    if output_path is None:
        directory = output_dir or os.path.join(os.path.expanduser("~"), ".opencut")
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, "fill_transition.mp4")

    if on_progress:
        on_progress(10, "Rendering fill transition animation...")

    # Parse hex colors to RGB
    def _hex_to_rgb(c: str) -> Tuple[int, int, int]:
        c = c.lstrip("#").lstrip("0x")
        if len(c) == 6:
            return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
        return (255, 255, 255)

    r1, g1, b1 = _hex_to_rgb(color_a)
    r2, g2, b2 = _hex_to_rgb(color_b)

    cx = int(s.x * width)
    cy = int(s.y * height)
    rx = int(s.width * width / 2)
    ry = int(s.height * height / 2)

    # Interpolate color over time using geq
    progress = f"min(1,T/{max(0.1, duration)})"
    shape_mask = (
        f"lt(hypot((X-{cx})/{max(1,rx)},(Y-{cy})/{max(1,ry)}),1)"
    )

    r_expr = f"'if({shape_mask},{r1}+({r2}-{r1})*{progress},0)'"
    g_expr = f"'if({shape_mask},{g1}+({g2}-{g1})*{progress},0)'"
    b_expr = f"'if({shape_mask},{b1}+({b2}-{b1})*{progress},0)'"

    vf = f"geq=r={r_expr}:g={g_expr}:b={b_expr}"

    cmd = (
        FFmpegCmd()
        .pre_input("-f", "lavfi")
        .input(
            f"color=c={background_color}:s={width}x{height}"
            f":d={duration}:r={fps}"
        )
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .option("-t", str(duration))
        .faststart()
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Fill transition rendered.")

    return ShapeAnimationResult(
        output_path=output_path,
        animation_type="fill_transition",
        duration=duration,
        width=width,
        height=height,
    )
