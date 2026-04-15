"""
OpenCut Kinetic Typography Engine (Category 79 - Motion Design)

Animate text per-character, per-word, or per-line with preset easing curves.
Render via Pillow frame-by-frame with FFmpeg encode for video output.

Functions:
    render_kinetic_text  - Render animated text to video
    preview_frame        - Render a single preview frame
    list_presets          - Return available animation presets
"""

import logging
import math
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------


@dataclass
class KineticResult:
    """Result of a kinetic typography render."""

    output_path: str = ""
    frames_rendered: int = 0
    duration: float = 0.0
    preset_used: str = ""
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "frames_rendered": self.frames_rendered,
            "duration": self.duration,
            "preset_used": self.preset_used,
            "resolution": list(self.resolution),
            "fps": self.fps,
        }


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
    if t == 0.0 or t == 1.0:
        return t
    p = 0.3
    s = p / 4.0
    return math.pow(2.0, -10.0 * t) * math.sin((t - s) * (2.0 * math.pi) / p) + 1.0


def _ease_cubic_bezier(t: float, x1: float = 0.25, y1: float = 0.1,
                       x2: float = 0.25, y2: float = 1.0) -> float:
    """Approximate cubic bezier easing via binary search."""
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        bx = 3.0 * x1 * mid * (1 - mid) ** 2 + 3.0 * x2 * mid ** 2 * (1 - mid) + mid ** 3
        if bx < t:
            lo = mid
        else:
            hi = mid
    mid = (lo + hi) / 2.0
    by = 3.0 * y1 * mid * (1 - mid) ** 2 + 3.0 * y2 * mid ** 2 * (1 - mid) + mid ** 3
    return by


EASING_FUNCTIONS = {
    "linear": _ease_linear,
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "ease_in_out": _ease_in_out,
    "bounce": _ease_bounce,
    "elastic": _ease_elastic,
    "cubic_bezier": _ease_cubic_bezier,
}


def get_easing(name: str) -> Callable:
    """Return easing function by name, defaulting to ease_out."""
    return EASING_FUNCTIONS.get(name, _ease_out)


# ---------------------------------------------------------------------------
# Text Segmentation
# ---------------------------------------------------------------------------


def _segment_text(text: str, mode: str) -> List[str]:
    """Split text into animatable elements based on mode."""
    if mode == "char" or mode == "character":
        return list(text)
    elif mode == "word":
        return text.split()
    elif mode == "line":
        return text.split("\n") if "\n" in text else [text]
    return [text]


# ---------------------------------------------------------------------------
# Animation Presets
# ---------------------------------------------------------------------------


def _keyframes_bounce(element_idx: int, total: int, duration: float) -> dict:
    """Bounce in from below."""
    delay = element_idx * 0.08
    return {
        "easing": "bounce",
        "delay": delay,
        "anim_duration": min(0.6, duration - delay),
        "start": {"y_offset": 100, "opacity": 0.0, "scale": 1.0, "rotation": 0.0},
        "end": {"y_offset": 0, "opacity": 1.0, "scale": 1.0, "rotation": 0.0},
    }


def _keyframes_elastic(element_idx: int, total: int, duration: float) -> dict:
    delay = element_idx * 0.06
    return {
        "easing": "elastic",
        "delay": delay,
        "anim_duration": min(0.8, duration - delay),
        "start": {"y_offset": -80, "opacity": 0.0, "scale": 0.3, "rotation": 0.0},
        "end": {"y_offset": 0, "opacity": 1.0, "scale": 1.0, "rotation": 0.0},
    }


def _keyframes_typewriter(element_idx: int, total: int, duration: float) -> dict:
    delay = element_idx * (duration * 0.7 / max(total, 1))
    return {
        "easing": "linear",
        "delay": delay,
        "anim_duration": 0.01,
        "start": {"y_offset": 0, "opacity": 0.0, "scale": 1.0, "rotation": 0.0},
        "end": {"y_offset": 0, "opacity": 1.0, "scale": 1.0, "rotation": 0.0},
    }


def _keyframes_wave(element_idx: int, total: int, duration: float) -> dict:
    delay = element_idx * 0.05
    return {
        "easing": "ease_in_out",
        "delay": delay,
        "anim_duration": min(0.5, duration - delay),
        "start": {"y_offset": 30 * math.sin(element_idx * 0.8), "opacity": 0.0,
                  "scale": 1.0, "rotation": 0.0},
        "end": {"y_offset": 0, "opacity": 1.0, "scale": 1.0, "rotation": 0.0},
    }


def _keyframes_cascade(element_idx: int, total: int, duration: float) -> dict:
    delay = element_idx * 0.12
    return {
        "easing": "ease_out",
        "delay": delay,
        "anim_duration": min(0.4, duration - delay),
        "start": {"y_offset": -60, "opacity": 0.0, "scale": 0.8, "rotation": -15.0},
        "end": {"y_offset": 0, "opacity": 1.0, "scale": 1.0, "rotation": 0.0},
    }


def _keyframes_spiral(element_idx: int, total: int, duration: float) -> dict:
    angle = element_idx * (360.0 / max(total, 1))
    delay = element_idx * 0.07
    return {
        "easing": "ease_out",
        "delay": delay,
        "anim_duration": min(0.6, duration - delay),
        "start": {"y_offset": 0, "opacity": 0.0, "scale": 0.1,
                  "rotation": angle},
        "end": {"y_offset": 0, "opacity": 1.0, "scale": 1.0, "rotation": 0.0},
    }


def _keyframes_explode(element_idx: int, total: int, duration: float) -> dict:
    rng = random.Random(element_idx * 42)
    delay = 0.0
    return {
        "easing": "ease_out",
        "delay": delay,
        "anim_duration": min(0.7, duration),
        "start": {"y_offset": rng.uniform(-150, 150), "opacity": 0.0,
                  "scale": 2.0, "rotation": rng.uniform(-90, 90)},
        "end": {"y_offset": 0, "opacity": 1.0, "scale": 1.0, "rotation": 0.0},
    }


def _keyframes_assemble(element_idx: int, total: int, duration: float) -> dict:
    rng = random.Random(element_idx * 17)
    delay = rng.uniform(0.0, duration * 0.3)
    return {
        "easing": "ease_in_out",
        "delay": delay,
        "anim_duration": min(0.5, duration - delay),
        "start": {"y_offset": rng.uniform(-200, 200), "opacity": 0.0,
                  "scale": rng.uniform(0.5, 1.5), "rotation": rng.uniform(-180, 180)},
        "end": {"y_offset": 0, "opacity": 1.0, "scale": 1.0, "rotation": 0.0},
    }


def _keyframes_fade_in(element_idx: int, total: int, duration: float) -> dict:
    delay = element_idx * 0.05
    return {
        "easing": "ease_in_out",
        "delay": delay,
        "anim_duration": min(0.4, duration - delay),
        "start": {"y_offset": 0, "opacity": 0.0, "scale": 1.0, "rotation": 0.0},
        "end": {"y_offset": 0, "opacity": 1.0, "scale": 1.0, "rotation": 0.0},
    }


def _keyframes_slide_up(element_idx: int, total: int, duration: float) -> dict:
    delay = element_idx * 0.06
    return {
        "easing": "ease_out",
        "delay": delay,
        "anim_duration": min(0.4, duration - delay),
        "start": {"y_offset": 80, "opacity": 0.0, "scale": 1.0, "rotation": 0.0},
        "end": {"y_offset": 0, "opacity": 1.0, "scale": 1.0, "rotation": 0.0},
    }


def _keyframes_slide_left(element_idx: int, total: int, duration: float) -> dict:
    delay = element_idx * 0.06
    return {
        "easing": "ease_out",
        "delay": delay,
        "anim_duration": min(0.4, duration - delay),
        "start": {"y_offset": 0, "opacity": 0.0, "scale": 1.0, "rotation": 0.0,
                  "x_offset": 120},
        "end": {"y_offset": 0, "opacity": 1.0, "scale": 1.0, "rotation": 0.0,
                "x_offset": 0},
    }


def _keyframes_scale_pop(element_idx: int, total: int, duration: float) -> dict:
    delay = element_idx * 0.05
    return {
        "easing": "elastic",
        "delay": delay,
        "anim_duration": min(0.6, duration - delay),
        "start": {"y_offset": 0, "opacity": 0.0, "scale": 0.0, "rotation": 0.0},
        "end": {"y_offset": 0, "opacity": 1.0, "scale": 1.0, "rotation": 0.0},
    }


ANIMATION_PRESETS = {
    "bounce": _keyframes_bounce,
    "elastic": _keyframes_elastic,
    "typewriter": _keyframes_typewriter,
    "wave": _keyframes_wave,
    "cascade": _keyframes_cascade,
    "spiral": _keyframes_spiral,
    "explode": _keyframes_explode,
    "assemble": _keyframes_assemble,
    "fade_in": _keyframes_fade_in,
    "slide_up": _keyframes_slide_up,
    "slide_left": _keyframes_slide_left,
    "scale_pop": _keyframes_scale_pop,
}


def list_presets() -> List[dict]:
    """Return list of available animation presets with descriptions."""
    descriptions = {
        "bounce": "Elements bounce in from below with physics-based easing",
        "elastic": "Springy overshoot entrance from above",
        "typewriter": "Characters appear one at a time like typing",
        "wave": "Sinusoidal wave motion entrance",
        "cascade": "Staggered waterfall from above with slight rotation",
        "spiral": "Elements spiral in from center with rotation",
        "explode": "Random positions converge to final layout",
        "assemble": "Scattered pieces assemble into text",
        "fade_in": "Simple staggered opacity fade",
        "slide_up": "Slide up from below with fade",
        "slide_left": "Slide in from right with fade",
        "scale_pop": "Pop in from zero scale with elastic overshoot",
    }
    return [
        {"name": name, "description": descriptions.get(name, "")}
        for name in ANIMATION_PRESETS
    ]


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def _interpolate(start: float, end: float, t: float) -> float:
    """Linear interpolation between start and end by t (0-1)."""
    return start + (end - start) * t


def _get_element_transform(kf: dict, time_s: float, easing_fn: Callable) -> dict:
    """Compute interpolated transform for an element at a given time."""
    delay = kf.get("delay", 0.0)
    anim_dur = kf.get("anim_duration", 0.5)
    start = kf["start"]
    end = kf["end"]

    if time_s < delay:
        return dict(start)

    local_t = (time_s - delay) / max(anim_dur, 0.001)
    local_t = max(0.0, min(1.0, local_t))
    eased_t = easing_fn(local_t)

    result = {}
    for key in end:
        s_val = start.get(key, 0.0)
        e_val = end[key]
        result[key] = _interpolate(s_val, e_val, eased_t)
    # Fill any keys only in start
    for key in start:
        if key not in result:
            result[key] = start[key]
    return result


# ---------------------------------------------------------------------------
# Font Helpers
# ---------------------------------------------------------------------------


def _load_font(font_name: Optional[str], font_size: int):
    """Load a PIL ImageFont, falling back to default."""
    from PIL import ImageFont  # noqa: F821

    if font_name:
        for candidate in [font_name, f"{font_name}.ttf", f"{font_name}.otf"]:
            try:
                return ImageFont.truetype(candidate, font_size)
            except (OSError, IOError):
                continue
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _measure_text(draw, text: str, font) -> Tuple[int, int]:
    """Measure text size using PIL draw context."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ---------------------------------------------------------------------------
# Frame Rendering
# ---------------------------------------------------------------------------


def _compute_element_positions(elements: List[str], draw, font,
                               resolution: Tuple[int, int],
                               line_spacing: int = 10) -> List[Tuple[int, int]]:
    """Compute centered base positions for each text element."""
    sizes = [_measure_text(draw, e, font) for e in elements]
    total_width = sum(s[0] for s in sizes) + max(0, len(sizes) - 1) * 5
    max_height = max(s[1] for s in sizes) if sizes else 20

    cx = resolution[0] // 2
    cy = resolution[1] // 2

    positions = []
    x = cx - total_width // 2
    for i, (w, h) in enumerate(sizes):
        positions.append((x, cy - max_height // 2))
        x += w + 5
    return positions


def _render_frame(elements: List[str], transforms: List[dict],
                  positions: List[Tuple[int, int]], font,
                  resolution: Tuple[int, int],
                  color: str = "#FFFFFF",
                  outline_color: Optional[str] = None,
                  outline_width: int = 0,
                  shadow_color: Optional[str] = None,
                  shadow_offset: Tuple[int, int] = (2, 2),
                  bg_color: str = "#00000000"):
    """Render a single frame with all text elements at their transforms."""
    from PIL import Image, ImageDraw  # noqa: F821

    # Transparent background by default
    if bg_color and bg_color != "#00000000":
        img = Image.new("RGBA", resolution, bg_color)
    else:
        img = Image.new("RGBA", resolution, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for i, (elem, tf, base_pos) in enumerate(zip(elements, transforms, positions)):
        opacity = tf.get("opacity", 1.0)
        if opacity <= 0.01:
            continue

        x_off = tf.get("x_offset", 0.0)
        y_off = tf.get("y_offset", 0.0)
        scale = tf.get("scale", 1.0)
        rotation = tf.get("rotation", 0.0)

        x = base_pos[0] + x_off
        y = base_pos[1] + y_off

        # For scale/rotation, render element to temp image and composite
        if abs(scale - 1.0) > 0.01 or abs(rotation) > 0.5:
            elem_img = _render_element_transformed(
                elem, font, color, outline_color, outline_width,
                shadow_color, shadow_offset, scale, rotation, opacity
            )
            if elem_img:
                paste_x = int(x - elem_img.width // 2 + _measure_text(draw, elem, font)[0] // 2)
                paste_y = int(y - elem_img.height // 2 + _measure_text(draw, elem, font)[1] // 2)
                img.alpha_composite(elem_img, (paste_x, paste_y))
        else:
            alpha = int(opacity * 255)
            r, g, b = _parse_color(color)
            fill = (r, g, b, alpha)

            if shadow_color and shadow_offset:
                sr, sg, sb = _parse_color(shadow_color)
                draw.text((x + shadow_offset[0], y + shadow_offset[1]), elem,
                          font=font, fill=(sr, sg, sb, alpha))

            if outline_color and outline_width > 0:
                _draw_text_outline(draw, x, y, elem, font, outline_color,
                                   outline_width, alpha)

            draw.text((x, y), elem, font=font, fill=fill)

    return img


def _render_element_transformed(text: str, font, color: str,
                                outline_color: Optional[str],
                                outline_width: int,
                                shadow_color: Optional[str],
                                shadow_offset: Tuple[int, int],
                                scale: float, rotation: float,
                                opacity: float):
    """Render a single text element with scale and rotation."""
    from PIL import Image, ImageDraw  # noqa: F821

    # Render at base size first
    tmp = Image.new("RGBA", (400, 200), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 20
    h = bbox[3] - bbox[1] + 20

    elem_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    elem_draw = ImageDraw.Draw(elem_img)

    alpha = int(opacity * 255)
    r, g, b = _parse_color(color)

    if shadow_color and shadow_offset:
        sr, sg, sb = _parse_color(shadow_color)
        elem_draw.text((10 + shadow_offset[0], 10 + shadow_offset[1]), text,
                       font=font, fill=(sr, sg, sb, alpha))

    if outline_color and outline_width > 0:
        _draw_text_outline(elem_draw, 10, 10, text, font, outline_color,
                           outline_width, alpha)

    elem_draw.text((10, 10), text, font=font, fill=(r, g, b, alpha))

    # Apply scale
    if abs(scale - 1.0) > 0.01:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        elem_img = elem_img.resize((new_w, new_h), Image.LANCZOS)

    # Apply rotation
    if abs(rotation) > 0.5:
        elem_img = elem_img.rotate(-rotation, expand=True, resample=Image.BICUBIC)

    return elem_img


def _draw_text_outline(draw, x: float, y: float, text: str, font,
                       outline_color: str, width: int, alpha: int):
    """Draw text outline by rendering text at offsets."""
    or_, og, ob = _parse_color(outline_color)
    ofill = (or_, og, ob, alpha)
    for dx in range(-width, width + 1):
        for dy in range(-width, width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=ofill)


def _parse_color(color_str: str) -> Tuple[int, int, int]:
    """Parse hex color string to RGB tuple."""
    c = color_str.lstrip("#")
    if len(c) == 3:
        c = c[0] * 2 + c[1] * 2 + c[2] * 2
    if len(c) >= 6:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (255, 255, 255)


# ---------------------------------------------------------------------------
# FFmpeg Encode
# ---------------------------------------------------------------------------


def _encode_frames_to_video(frame_dir: str, output_path: str,
                            fps: int, resolution: Tuple[int, int]) -> str:
    """Encode PNG frame sequence to MP4 via FFmpeg."""
    from opencut.helpers import get_ffmpeg_path, run_ffmpeg

    pattern = os.path.join(frame_dir, "frame_%06d.png")
    cmd = [
        get_ffmpeg_path(), "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuva420p",
        "-crf", "18", "-preset", "fast",
        "-vf", f"scale={resolution[0]}:{resolution[1]}",
        output_path,
    ]
    run_ffmpeg(cmd)
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_kinetic_text(
    text: str,
    preset: str = "bounce",
    mode: str = "char",
    duration: float = 3.0,
    fps: int = 30,
    resolution: Tuple[int, int] = (1920, 1080),
    font_name: Optional[str] = None,
    font_size: int = 72,
    color: str = "#FFFFFF",
    outline_color: Optional[str] = None,
    outline_width: int = 0,
    shadow_color: Optional[str] = None,
    shadow_offset: Tuple[int, int] = (2, 2),
    background_color: str = "#00000000",
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> KineticResult:
    """Render kinetic typography animation to video.

    Args:
        text: Text to animate.
        preset: Animation preset name (bounce, elastic, typewriter, etc.).
        mode: Segmentation mode - 'char', 'word', or 'line'.
        duration: Animation duration in seconds.
        fps: Frames per second.
        resolution: Output resolution (width, height).
        font_name: Font file name/path (falls back to system default).
        font_size: Font size in pixels.
        color: Text color as hex string.
        outline_color: Optional outline color.
        outline_width: Outline width in pixels.
        shadow_color: Optional drop shadow color.
        shadow_offset: Shadow offset (dx, dy).
        background_color: Background color (transparent by default).
        output_path: Explicit output path. Auto-generated if None.
        output_dir: Directory for output. Temp dir if empty.
        on_progress: Progress callback taking int percentage.

    Returns:
        KineticResult with output path and render metadata.
    """
    from PIL import Image, ImageDraw  # noqa: F821

    if not text:
        raise ValueError("Text cannot be empty")

    preset_fn = ANIMATION_PRESETS.get(preset)
    if preset_fn is None:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(ANIMATION_PRESETS.keys())}")

    elements = _segment_text(text, mode)
    if not elements:
        raise ValueError("Text segmentation produced no elements")

    total_frames = max(1, int(duration * fps))
    font = _load_font(font_name, font_size)

    # Compute base positions using a temp draw context
    tmp_img = Image.new("RGBA", resolution, (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp_img)
    positions = _compute_element_positions(elements, tmp_draw, font, resolution)
    del tmp_img, tmp_draw

    # Generate keyframes for each element
    keyframe_list = []
    for i, elem in enumerate(elements):
        kf = preset_fn(i, len(elements), duration)
        keyframe_list.append(kf)

    # Render frames
    effective_dir = output_dir or tempfile.gettempdir()
    frame_dir = tempfile.mkdtemp(prefix="opencut_kinetic_", dir=effective_dir)

    for frame_idx in range(total_frames):
        time_s = frame_idx / fps
        transforms = []
        for kf in keyframe_list:
            easing_fn = get_easing(kf.get("easing", "ease_out"))
            tf = _get_element_transform(kf, time_s, easing_fn)
            transforms.append(tf)

        frame_img = _render_frame(
            elements, transforms, positions, font, resolution,
            color=color, outline_color=outline_color,
            outline_width=outline_width, shadow_color=shadow_color,
            shadow_offset=shadow_offset, bg_color=background_color,
        )
        frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
        frame_img.save(frame_path, "PNG")

        if on_progress and total_frames > 1:
            pct = int((frame_idx + 1) / total_frames * 90)
            on_progress(pct)

    # Encode to video
    if not output_path:
        output_path = os.path.join(effective_dir, f"kinetic_{preset}_{mode}.mp4")

    if on_progress:
        on_progress(92)

    _encode_frames_to_video(frame_dir, output_path, fps, resolution)

    if on_progress:
        on_progress(100)

    return KineticResult(
        output_path=output_path,
        frames_rendered=total_frames,
        duration=duration,
        preset_used=preset,
        resolution=resolution,
        fps=fps,
    )


def preview_frame(
    text: str,
    preset: str = "bounce",
    mode: str = "char",
    time_s: float = 0.5,
    duration: float = 3.0,
    fps: int = 30,
    resolution: Tuple[int, int] = (1920, 1080),
    font_name: Optional[str] = None,
    font_size: int = 72,
    color: str = "#FFFFFF",
    outline_color: Optional[str] = None,
    outline_width: int = 0,
    shadow_color: Optional[str] = None,
    shadow_offset: Tuple[int, int] = (2, 2),
    background_color: str = "#00000000",
    output_path: Optional[str] = None,
) -> str:
    """Render a single preview frame and save as PNG.

    Returns the path to the saved PNG.
    """
    from PIL import Image, ImageDraw  # noqa: F821

    if not text:
        raise ValueError("Text cannot be empty")

    preset_fn = ANIMATION_PRESETS.get(preset)
    if preset_fn is None:
        raise ValueError(f"Unknown preset: {preset}")

    elements = _segment_text(text, mode)
    font = _load_font(font_name, font_size)

    tmp_img = Image.new("RGBA", resolution, (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp_img)
    positions = _compute_element_positions(elements, tmp_draw, font, resolution)
    del tmp_img, tmp_draw

    keyframe_list = []
    for i, elem in enumerate(elements):
        kf = preset_fn(i, len(elements), duration)
        keyframe_list.append(kf)

    transforms = []
    for kf in keyframe_list:
        easing_fn = get_easing(kf.get("easing", "ease_out"))
        tf = _get_element_transform(kf, time_s, easing_fn)
        transforms.append(tf)

    frame_img = _render_frame(
        elements, transforms, positions, font, resolution,
        color=color, outline_color=outline_color,
        outline_width=outline_width, shadow_color=shadow_color,
        shadow_offset=shadow_offset, bg_color=background_color,
    )

    if not output_path:
        output_path = os.path.join(tempfile.gettempdir(), "kinetic_preview.png")
    frame_img.save(output_path, "PNG")
    return output_path
