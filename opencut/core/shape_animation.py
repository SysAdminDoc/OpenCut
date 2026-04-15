"""
OpenCut Shape Animation Engine (Category 79 - Motion Design)

Animate vector shapes: path morphing, stroke drawing, fill transitions.
Shapes defined as point lists with SVG path string support. Render via
Pillow ImageDraw with anti-aliasing and export as PNG sequence + FFmpeg.

Functions:
    render_shape_animation  - Render shape animation to video
    morph_shapes            - Interpolate between two shapes
    list_shape_types        - Return supported shape types
    list_animation_types    - Return supported animation types
    parse_svg_path          - Parse SVG path string to point list
"""

import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ShapeAnimResult:
    """Result of a shape animation render."""

    output_path: str = ""
    frames: int = 0
    shapes_count: int = 0
    duration: float = 0.0
    fps: int = 30

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "frames": self.frames,
            "shapes_count": self.shapes_count,
            "duration": self.duration,
            "fps": self.fps,
        }


# ---------------------------------------------------------------------------
# Shape Types
# ---------------------------------------------------------------------------

SHAPE_TYPES = [
    "circle", "rectangle", "rounded_rect", "star",
    "polygon", "line", "arc", "custom_path",
]

ANIMATION_TYPES = [
    "morph", "draw_stroke", "fill_fade", "scale_rotate",
]


def list_shape_types() -> List[dict]:
    """Return list of supported shape types."""
    descriptions = {
        "circle": "Circle defined by center and radius",
        "rectangle": "Rectangle defined by corner and dimensions",
        "rounded_rect": "Rectangle with rounded corners",
        "star": "Star with configurable points and inner/outer radius",
        "polygon": "Regular polygon with N sides",
        "line": "Line segment between two points",
        "arc": "Circular arc defined by center, radius, and angles",
        "custom_path": "Custom shape from SVG path string or point list",
    }
    return [
        {"type": t, "description": descriptions.get(t, "")}
        for t in SHAPE_TYPES
    ]


def list_animation_types() -> List[dict]:
    """Return list of supported animation types."""
    descriptions = {
        "morph": "Interpolate between two shapes by matching point count",
        "draw_stroke": "Animate dash offset to simulate drawing the shape",
        "fill_fade": "Animate fill color and opacity over time",
        "scale_rotate": "Transform shape with scale and rotation easing",
    }
    return [
        {"type": t, "description": descriptions.get(t, "")}
        for t in ANIMATION_TYPES
    ]


# ---------------------------------------------------------------------------
# Easing Functions
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


EASING = {
    "linear": _ease_linear,
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "ease_in_out": _ease_in_out,
}


def _get_easing(name: str) -> Callable:
    return EASING.get(name, _ease_out)


# ---------------------------------------------------------------------------
# Shape Generators — return list of (x, y) points
# ---------------------------------------------------------------------------


def generate_circle(cx: float, cy: float, radius: float,
                    num_points: int = 64) -> List[Tuple[float, float]]:
    """Generate circle as point list."""
    points = []
    for i in range(num_points):
        angle = 2.0 * math.pi * i / num_points
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    return points


def generate_rectangle(x: float, y: float, w: float, h: float,
                       num_points: int = 64) -> List[Tuple[float, float]]:
    """Generate rectangle as point list with evenly distributed points."""
    perimeter = 2 * (w + h)
    points = []
    for i in range(num_points):
        d = (i / num_points) * perimeter
        if d < w:
            points.append((x + d, y))
        elif d < w + h:
            points.append((x + w, y + (d - w)))
        elif d < 2 * w + h:
            points.append((x + w - (d - w - h), y + h))
        else:
            points.append((x, y + h - (d - 2 * w - h)))
    return points


def generate_rounded_rect(x: float, y: float, w: float, h: float,
                          radius: float = 10.0,
                          num_points: int = 64) -> List[Tuple[float, float]]:
    """Generate rounded rectangle as point list."""
    r = min(radius, w / 2, h / 2)
    points = []
    pts_per_corner = max(1, num_points // 4)
    corners = [
        (x + w - r, y + r, -math.pi / 2, 0),           # top-right
        (x + w - r, y + h - r, 0, math.pi / 2),          # bottom-right
        (x + r, y + h - r, math.pi / 2, math.pi),        # bottom-left
        (x + r, y + r, math.pi, 3 * math.pi / 2),        # top-left
    ]
    for cx, cy, start_a, end_a in corners:
        for j in range(pts_per_corner):
            t = j / max(pts_per_corner - 1, 1)
            angle = start_a + t * (end_a - start_a)
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            points.append((px, py))
    return points


def generate_star(cx: float, cy: float, outer_r: float, inner_r: float,
                  num_spikes: int = 5,
                  num_points: int = 64) -> List[Tuple[float, float]]:
    """Generate star shape as point list."""
    points = []
    total_vertices = num_spikes * 2
    pts_per_segment = max(1, num_points // total_vertices)

    vertices = []
    for i in range(total_vertices):
        angle = math.pi * 2 * i / total_vertices - math.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        vertices.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))

    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        for j in range(pts_per_segment):
            t = j / max(pts_per_segment, 1)
            px = p1[0] + (p2[0] - p1[0]) * t
            py = p1[1] + (p2[1] - p1[1]) * t
            points.append((px, py))
    return points


def generate_polygon(cx: float, cy: float, radius: float,
                     sides: int = 6,
                     num_points: int = 64) -> List[Tuple[float, float]]:
    """Generate regular polygon as point list."""
    vertices = []
    for i in range(sides):
        angle = 2.0 * math.pi * i / sides - math.pi / 2
        vertices.append((cx + radius * math.cos(angle),
                         cy + radius * math.sin(angle)))

    points = []
    pts_per_side = max(1, num_points // sides)
    for i in range(sides):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % sides]
        for j in range(pts_per_side):
            t = j / max(pts_per_side, 1)
            points.append((p1[0] + (p2[0] - p1[0]) * t,
                           p1[1] + (p2[1] - p1[1]) * t))
    return points


def generate_line(x1: float, y1: float, x2: float, y2: float,
                  num_points: int = 64) -> List[Tuple[float, float]]:
    """Generate line as point list."""
    points = []
    for i in range(num_points):
        t = i / max(num_points - 1, 1)
        points.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    return points


def generate_arc(cx: float, cy: float, radius: float,
                 start_angle: float = 0.0, end_angle: float = 180.0,
                 num_points: int = 64) -> List[Tuple[float, float]]:
    """Generate arc as point list. Angles in degrees."""
    points = []
    sa = math.radians(start_angle)
    ea = math.radians(end_angle)
    for i in range(num_points):
        t = i / max(num_points - 1, 1)
        angle = sa + (ea - sa) * t
        points.append((cx + radius * math.cos(angle),
                       cy + radius * math.sin(angle)))
    return points


# ---------------------------------------------------------------------------
# SVG Path Parser (minimal)
# ---------------------------------------------------------------------------


def parse_svg_path(path_str: str,
                   num_points: int = 64) -> List[Tuple[float, float]]:
    """Parse a minimal SVG path string (M, L, C, Z commands) to points.

    Supports absolute Move, Line, and Cubic Bezier commands.
    Returns a point list resampled to num_points.
    """
    raw_points = []
    tokens = re.findall(r'[MLCZmlcz]|[-+]?\d*\.?\d+', path_str)
    idx = 0
    cx, cy = 0.0, 0.0
    start_x, start_y = 0.0, 0.0

    while idx < len(tokens):
        cmd = tokens[idx]
        if cmd in ("M", "m"):
            idx += 1
            x, y = float(tokens[idx]), float(tokens[idx + 1])
            idx += 2
            if cmd == "m":
                x += cx
                y += cy
            cx, cy = x, y
            start_x, start_y = x, y
            raw_points.append((cx, cy))
        elif cmd in ("L", "l"):
            idx += 1
            x, y = float(tokens[idx]), float(tokens[idx + 1])
            idx += 2
            if cmd == "l":
                x += cx
                y += cy
            cx, cy = x, y
            raw_points.append((cx, cy))
        elif cmd in ("C", "c"):
            idx += 1
            x1 = float(tokens[idx])
            y1 = float(tokens[idx + 1])
            x2 = float(tokens[idx + 2])
            y2 = float(tokens[idx + 3])
            x = float(tokens[idx + 4])
            y = float(tokens[idx + 5])
            idx += 6
            if cmd == "c":
                x1 += cx
                y1 += cy
                x2 += cx
                y2 += cy
                x += cx
                y += cy
            # Sample cubic bezier
            for i in range(10):
                t = i / 9
                it = 1 - t
                bx = it ** 3 * cx + 3 * it ** 2 * t * x1 + 3 * it * t ** 2 * x2 + t ** 3 * x
                by = it ** 3 * cy + 3 * it ** 2 * t * y1 + 3 * it * t ** 2 * y2 + t ** 3 * y
                raw_points.append((bx, by))
            cx, cy = x, y
        elif cmd in ("Z", "z"):
            idx += 1
            raw_points.append((start_x, start_y))
            cx, cy = start_x, start_y
        else:
            idx += 1

    return _resample_points(raw_points, num_points)


# ---------------------------------------------------------------------------
# Point Resampling & Morphing
# ---------------------------------------------------------------------------


def _path_length(points: List[Tuple[float, float]]) -> float:
    """Compute total path length."""
    length = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        length += math.sqrt(dx * dx + dy * dy)
    return length


def _resample_points(points: List[Tuple[float, float]],
                     count: int) -> List[Tuple[float, float]]:
    """Resample a point list to exactly 'count' evenly spaced points."""
    if len(points) < 2:
        if points:
            return [points[0]] * count
        return [(0.0, 0.0)] * count

    total = _path_length(points)
    if total < 0.001:
        return [points[0]] * count

    step = total / max(count - 1, 1)
    result = [points[0]]
    accumulated = 0.0
    pi = 1
    target = step

    while len(result) < count and pi < len(points):
        dx = points[pi][0] - points[pi - 1][0]
        dy = points[pi][1] - points[pi - 1][1]
        seg_len = math.sqrt(dx * dx + dy * dy)

        if accumulated + seg_len >= target:
            overshoot = target - accumulated
            t = overshoot / max(seg_len, 0.001)
            new_x = points[pi - 1][0] + dx * t
            new_y = points[pi - 1][1] + dy * t
            result.append((new_x, new_y))
            target += step
            # Don't advance pi; we might place multiple points on same segment
        else:
            accumulated += seg_len
            pi += 1

    # Fill remaining if rounding errors
    while len(result) < count:
        result.append(points[-1])
    return result[:count]


def morph_shapes(shape1: List[Tuple[float, float]],
                 shape2: List[Tuple[float, float]],
                 t: float,
                 num_points: int = 64) -> List[Tuple[float, float]]:
    """Morph between two shapes by interpolating matched points.

    Both shapes are resampled to num_points for consistent matching.
    """
    pts1 = _resample_points(shape1, num_points)
    pts2 = _resample_points(shape2, num_points)
    result = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        result.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    return result


# ---------------------------------------------------------------------------
# Shape Definition Helpers
# ---------------------------------------------------------------------------


def build_shape(shape_def: dict,
                num_points: int = 64) -> List[Tuple[float, float]]:
    """Build a point list from a shape definition dict.

    Args:
        shape_def: Dict with 'type' and type-specific parameters.
        num_points: Number of points to generate.
    """
    stype = shape_def.get("type", "circle")
    cx = float(shape_def.get("cx", 400))
    cy = float(shape_def.get("cy", 300))

    if stype == "circle":
        radius = float(shape_def.get("radius", 100))
        return generate_circle(cx, cy, radius, num_points)
    elif stype == "rectangle":
        w = float(shape_def.get("width", 200))
        h = float(shape_def.get("height", 150))
        return generate_rectangle(cx - w / 2, cy - h / 2, w, h, num_points)
    elif stype == "rounded_rect":
        w = float(shape_def.get("width", 200))
        h = float(shape_def.get("height", 150))
        r = float(shape_def.get("corner_radius", 15))
        return generate_rounded_rect(cx - w / 2, cy - h / 2, w, h, r, num_points)
    elif stype == "star":
        outer_r = float(shape_def.get("outer_radius", 100))
        inner_r = float(shape_def.get("inner_radius", 40))
        spikes = int(shape_def.get("spikes", 5))
        return generate_star(cx, cy, outer_r, inner_r, spikes, num_points)
    elif stype == "polygon":
        radius = float(shape_def.get("radius", 100))
        sides = int(shape_def.get("sides", 6))
        return generate_polygon(cx, cy, radius, sides, num_points)
    elif stype == "line":
        x2 = float(shape_def.get("x2", cx + 200))
        y2 = float(shape_def.get("y2", cy))
        return generate_line(cx, cy, x2, y2, num_points)
    elif stype == "arc":
        radius = float(shape_def.get("radius", 100))
        sa = float(shape_def.get("start_angle", 0))
        ea = float(shape_def.get("end_angle", 180))
        return generate_arc(cx, cy, radius, sa, ea, num_points)
    elif stype == "custom_path":
        path_str = shape_def.get("path", "")
        if path_str:
            return parse_svg_path(path_str, num_points)
        points = shape_def.get("points", [])
        if points:
            parsed = [(float(p[0]), float(p[1])) for p in points]
            return _resample_points(parsed, num_points)
        return generate_circle(cx, cy, 100, num_points)
    else:
        return generate_circle(cx, cy, 100, num_points)


# ---------------------------------------------------------------------------
# Color Helpers
# ---------------------------------------------------------------------------


def _parse_color(color_str: str) -> Tuple[int, int, int]:
    """Parse hex color to RGB."""
    c = color_str.lstrip("#")
    if len(c) == 3:
        c = c[0] * 2 + c[1] * 2 + c[2] * 2
    if len(c) >= 6:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (255, 255, 255)


def _lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int],
                t: float) -> Tuple[int, int, int]:
    """Linearly interpolate between two RGB colors."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


# ---------------------------------------------------------------------------
# Frame Rendering
# ---------------------------------------------------------------------------


def _draw_shape_outline(draw, points: List[Tuple[float, float]],
                        color: Tuple[int, int, int, int],
                        width: int = 2,
                        progress: float = 1.0):
    """Draw shape outline (stroke) with optional progress for draw animation."""
    if len(points) < 2:
        return

    total = len(points)
    visible = max(2, int(total * progress))

    for i in range(visible - 1):
        p1 = (int(points[i][0]), int(points[i][1]))
        p2 = (int(points[i + 1][0]), int(points[i + 1][1]))
        draw.line([p1, p2], fill=color, width=width)


def _draw_shape_filled(draw, points: List[Tuple[float, float]],
                       fill: Tuple[int, int, int, int],
                       outline: Optional[Tuple[int, int, int, int]] = None,
                       outline_width: int = 0):
    """Draw filled shape polygon."""
    if len(points) < 3:
        return
    int_points = [(int(p[0]), int(p[1])) for p in points]
    draw.polygon(int_points, fill=fill, outline=outline)


def _apply_transform(points: List[Tuple[float, float]],
                     scale: float = 1.0,
                     rotation: float = 0.0,
                     translate: Tuple[float, float] = (0, 0),
                     center: Optional[Tuple[float, float]] = None
                     ) -> List[Tuple[float, float]]:
    """Apply scale, rotation, and translation to point list."""
    if center is None:
        if points:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            center = (cx, cy)
        else:
            center = (0, 0)

    rad = math.radians(rotation)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    result = []
    for px, py in points:
        # Translate to origin
        dx = px - center[0]
        dy = py - center[1]
        # Scale
        dx *= scale
        dy *= scale
        # Rotate
        rx = dx * cos_r - dy * sin_r
        ry = dx * sin_r + dy * cos_r
        # Translate back + offset
        result.append((rx + center[0] + translate[0],
                       ry + center[1] + translate[1]))
    return result


def _render_shape_frame(shapes: List[dict],
                        resolution: Tuple[int, int],
                        bg_color: str = "#00000000"):
    """Render a single frame with all shapes.

    Each shape dict should have:
        points: List of (x,y) tuples
        fill_color: hex string or None
        fill_opacity: 0.0-1.0
        stroke_color: hex string or None
        stroke_width: int
        stroke_progress: 0.0-1.0 (for draw animation)
    """
    from PIL import Image, ImageDraw  # noqa: F821

    if bg_color and bg_color != "#00000000":
        r, g, b = _parse_color(bg_color)
        img = Image.new("RGBA", resolution, (r, g, b, 255))
    else:
        img = Image.new("RGBA", resolution, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for shape in shapes:
        points = shape.get("points", [])
        if not points:
            continue

        fill_color = shape.get("fill_color")
        fill_opacity = float(shape.get("fill_opacity", 1.0))
        stroke_color = shape.get("stroke_color", "#FFFFFF")
        stroke_width = int(shape.get("stroke_width", 2))
        stroke_progress = float(shape.get("stroke_progress", 1.0))

        if fill_color and fill_opacity > 0.01:
            r, g, b = _parse_color(fill_color)
            alpha = int(fill_opacity * 255)
            _draw_shape_filled(draw, points, (r, g, b, alpha))

        if stroke_color and stroke_width > 0:
            r, g, b = _parse_color(stroke_color)
            _draw_shape_outline(draw, points, (r, g, b, 255),
                                stroke_width, stroke_progress)

    return img


# ---------------------------------------------------------------------------
# FFmpeg Encode
# ---------------------------------------------------------------------------


def _encode_frames(frame_dir: str, output_path: str,
                   fps: int, resolution: Tuple[int, int]) -> str:
    """Encode PNG frame sequence to video."""
    from opencut.helpers import get_ffmpeg_path, run_ffmpeg

    pattern = os.path.join(frame_dir, "frame_%06d.png")
    cmd = [
        get_ffmpeg_path(), "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuva420p",
        "-crf", "18", "-preset", "fast",
        output_path,
    ]
    run_ffmpeg(cmd)
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_shape_animation(
    shapes: List[dict],
    animation: str = "morph",
    duration: float = 3.0,
    fps: int = 30,
    resolution: Tuple[int, int] = (1920, 1080),
    easing: str = "ease_out",
    stroke_color: str = "#FFFFFF",
    stroke_width: int = 2,
    fill_color: Optional[str] = None,
    fill_color_end: Optional[str] = None,
    num_points: int = 64,
    bg_color: str = "#00000000",
    scale_start: float = 1.0,
    scale_end: float = 1.0,
    rotation_start: float = 0.0,
    rotation_end: float = 0.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ShapeAnimResult:
    """Render shape animation to video.

    Args:
        shapes: List of shape definition dicts. For 'morph' animation,
            exactly 2 shapes define start and end. For others, animate
            each shape independently.
        animation: Animation type (morph, draw_stroke, fill_fade, scale_rotate).
        duration: Animation duration in seconds.
        fps: Frames per second.
        resolution: Output resolution.
        easing: Easing function name.
        stroke_color: Stroke color hex.
        stroke_width: Stroke width pixels.
        fill_color: Fill color hex (start for fill_fade).
        fill_color_end: End fill color for fill_fade animation.
        num_points: Point count for shape generation/resampling.
        bg_color: Background color (transparent by default).
        scale_start: Starting scale for scale_rotate animation.
        scale_end: Ending scale for scale_rotate animation.
        rotation_start: Starting rotation degrees for scale_rotate.
        rotation_end: Ending rotation degrees for scale_rotate.
        output_path: Explicit output path.
        output_dir: Output directory.
        on_progress: Progress callback.

    Returns:
        ShapeAnimResult with output path and metadata.
    """
    if not shapes:
        raise ValueError("At least one shape definition is required")

    if animation not in ANIMATION_TYPES:
        raise ValueError(
            f"Unknown animation: {animation}. "
            f"Available: {ANIMATION_TYPES}"
        )

    if animation == "morph" and len(shapes) < 2:
        raise ValueError("Morph animation requires at least 2 shapes")

    total_frames = max(1, int(duration * fps))
    ease_fn = _get_easing(easing)

    # Build point lists from shape definitions
    built_shapes = [build_shape(s, num_points) for s in shapes]

    effective_dir = output_dir or tempfile.gettempdir()
    frame_dir = tempfile.mkdtemp(prefix="opencut_shapeanim_", dir=effective_dir)

    for frame_idx in range(total_frames):
        raw_t = frame_idx / max(total_frames - 1, 1)
        t = ease_fn(raw_t)

        frame_shapes = []

        if animation == "morph":
            # Morph between first and last shape
            morphed = morph_shapes(built_shapes[0], built_shapes[-1],
                                   t, num_points)
            frame_shapes.append({
                "points": morphed,
                "stroke_color": stroke_color,
                "stroke_width": stroke_width,
                "fill_color": fill_color,
                "fill_opacity": 1.0,
                "stroke_progress": 1.0,
            })

        elif animation == "draw_stroke":
            for pts in built_shapes:
                frame_shapes.append({
                    "points": pts,
                    "stroke_color": stroke_color,
                    "stroke_width": stroke_width,
                    "fill_color": None,
                    "fill_opacity": 0.0,
                    "stroke_progress": t,
                })

        elif animation == "fill_fade":
            start_c = _parse_color(fill_color or "#FFFFFF")
            end_c = _parse_color(fill_color_end or fill_color or "#FFFFFF")
            interp_c = _lerp_color(start_c, end_c, t)
            interp_hex = f"#{interp_c[0]:02x}{interp_c[1]:02x}{interp_c[2]:02x}"

            for pts in built_shapes:
                frame_shapes.append({
                    "points": pts,
                    "stroke_color": stroke_color,
                    "stroke_width": stroke_width,
                    "fill_color": interp_hex,
                    "fill_opacity": t,
                    "stroke_progress": 1.0,
                })

        elif animation == "scale_rotate":
            cur_scale = scale_start + (scale_end - scale_start) * t
            cur_rot = rotation_start + (rotation_end - rotation_start) * t

            for pts in built_shapes:
                transformed = _apply_transform(pts, cur_scale, cur_rot)
                frame_shapes.append({
                    "points": transformed,
                    "stroke_color": stroke_color,
                    "stroke_width": stroke_width,
                    "fill_color": fill_color,
                    "fill_opacity": 1.0 if fill_color else 0.0,
                    "stroke_progress": 1.0,
                })

        frame_img = _render_shape_frame(frame_shapes, resolution, bg_color)
        frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
        frame_img.save(frame_path, "PNG")

        if on_progress and total_frames > 1:
            pct = int((frame_idx + 1) / total_frames * 90)
            on_progress(pct)

    # Encode
    if not output_path:
        output_path = os.path.join(effective_dir,
                                   f"shape_{animation}.mp4")

    if on_progress:
        on_progress(92)

    _encode_frames(frame_dir, output_path, fps, resolution)

    if on_progress:
        on_progress(100)

    return ShapeAnimResult(
        output_path=output_path,
        frames=total_frames,
        shapes_count=len(shapes),
        duration=duration,
        fps=fps,
    )
