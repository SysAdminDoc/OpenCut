"""
OpenCut AI Intro Generator Module

Auto-generate animated video intros from a brand kit:
- Accept brand kit definition (logo, colors, fonts, tagline)
- Generate animated intro sequences (3-7 seconds)
- Multiple intro styles: logo_reveal, text_sweep, particles, minimal_fade, kinetic
- Background: solid color, gradient, or blurred clip frame
- Music bed integration (auto-trim intro music to duration)
- Export as standalone clip or prepend to main video

Uses Pillow for frame rendering, FFmpeg for video assembly.
"""

import logging
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Intro Styles
# ---------------------------------------------------------------------------
INTRO_STYLES = {
    "logo_reveal": {
        "name": "Logo Reveal",
        "description": "Logo fades in with glow effect, then tagline slides up",
        "min_duration": 3.0,
        "max_duration": 7.0,
        "default_duration": 5.0,
        "supports_logo": True,
        "supports_tagline": True,
    },
    "text_sweep": {
        "name": "Text Sweep",
        "description": "Brand name sweeps in letter by letter with trailing glow",
        "min_duration": 3.0,
        "max_duration": 6.0,
        "default_duration": 4.0,
        "supports_logo": False,
        "supports_tagline": True,
    },
    "particles": {
        "name": "Particles",
        "description": "Particles converge to form logo/text, then settle",
        "min_duration": 4.0,
        "max_duration": 7.0,
        "default_duration": 5.0,
        "supports_logo": True,
        "supports_tagline": True,
    },
    "minimal_fade": {
        "name": "Minimal Fade",
        "description": "Simple centered logo/text with elegant fade in/out",
        "min_duration": 2.5,
        "max_duration": 5.0,
        "default_duration": 3.5,
        "supports_logo": True,
        "supports_tagline": True,
    },
    "kinetic": {
        "name": "Kinetic",
        "description": "Dynamic motion graphics with bouncing text and shapes",
        "min_duration": 3.0,
        "max_duration": 6.0,
        "default_duration": 4.5,
        "supports_logo": True,
        "supports_tagline": True,
    },
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class BrandKit:
    """Brand identity definition for intro generation."""
    name: str = ""
    tagline: str = ""
    logo_path: str = ""
    font_path: str = ""
    font_name: str = "Arial"
    primary_color: Tuple[int, int, int] = (80, 140, 255)
    secondary_color: Tuple[int, int, int] = (255, 200, 80)
    accent_color: Tuple[int, int, int] = (100, 220, 180)
    background_color: Tuple[int, int, int] = (12, 12, 18)
    text_color: Tuple[int, int, int] = (240, 240, 245)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> List[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        if not self.name and not self.logo_path:
            errors.append("Brand kit must have at least a name or logo_path")
        if self.logo_path and not os.path.isfile(self.logo_path):
            errors.append(f"Logo file not found: {self.logo_path}")
        return errors


@dataclass
class IntroConfig:
    """Configuration for intro generation."""
    style: str = "logo_reveal"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    duration: float = 5.0
    background_mode: str = "solid"
    background_image: str = ""
    background_blur: int = 30
    music_path: str = ""
    music_volume: float = 0.8
    fade_in: float = 0.5
    fade_out: float = 0.8
    prepend_to: str = ""
    glow_intensity: float = 1.0
    animation_speed: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> List[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        if self.style not in INTRO_STYLES:
            errors.append(f"Unknown style '{self.style}', must be one of {list(INTRO_STYLES.keys())}")
        if self.width < 64 or self.width > 7680:
            errors.append(f"Width {self.width} out of range [64, 7680]")
        if self.height < 64 or self.height > 4320:
            errors.append(f"Height {self.height} out of range [64, 4320]")
        if self.fps < 1 or self.fps > 120:
            errors.append(f"FPS {self.fps} out of range [1, 120]")

        style_def = INTRO_STYLES.get(self.style, {})
        min_d = style_def.get("min_duration", 2.0)
        max_d = style_def.get("max_duration", 10.0)
        if self.duration < min_d or self.duration > max_d:
            errors.append(f"Duration {self.duration}s out of range [{min_d}, {max_d}] for style '{self.style}'")
        if self.music_volume < 0.0 or self.music_volume > 1.0:
            errors.append(f"music_volume {self.music_volume} out of range [0.0, 1.0]")
        if self.background_mode not in ("solid", "gradient", "blur", "image"):
            errors.append(f"Unknown background_mode '{self.background_mode}'")
        return errors


@dataclass
class IntroResult:
    """Result of intro generation."""
    output_path: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: int = 0
    frame_count: int = 0
    style: str = ""
    has_music: bool = False
    prepended_to: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Font Loading
# ---------------------------------------------------------------------------
def _load_font(brand_kit: BrandKit, size: int):
    """Load font from brand kit or fall back to defaults."""
    ensure_package("PIL", "Pillow")
    from PIL import ImageFont

    if brand_kit.font_path and os.path.isfile(brand_kit.font_path):
        try:
            return ImageFont.truetype(brand_kit.font_path, size=size)
        except (IOError, OSError):
            pass

    for name in ["arial", "Arial", "Helvetica", "DejaVuSans", "segoeui"]:
        try:
            return ImageFont.truetype(name, size=size)
        except (IOError, OSError):
            continue

    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Background Generation
# ---------------------------------------------------------------------------
def _create_bg(config: IntroConfig, brand: BrandKit) -> "Image.Image":  # noqa: F821
    """Create background frame based on config."""
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageFilter

    w, h = config.width, config.height

    if config.background_mode == "image" and config.background_image:
        try:
            bg = Image.open(config.background_image).convert("RGBA")
            return bg.resize((w, h), Image.LANCZOS)
        except Exception:
            pass

    if config.background_mode == "blur" and config.background_image:
        try:
            bg = Image.open(config.background_image).convert("RGBA")
            bg = bg.resize((w, h), Image.LANCZOS)
            return bg.filter(ImageFilter.GaussianBlur(radius=config.background_blur))
        except Exception:
            pass

    if config.background_mode == "gradient":
        img = Image.new("RGBA", (w, h))
        pixels = img.load()
        c1 = brand.background_color
        c2 = brand.primary_color
        for y in range(h):
            t = y / max(h - 1, 1)
            # Ease-in-out curve
            t = t * t * (3 - 2 * t)
            r = int(c1[0] * (1 - t) + c2[0] * t * 0.3)
            g = int(c1[1] * (1 - t) + c2[1] * t * 0.3)
            b = int(c1[2] * (1 - t) + c2[2] * t * 0.3)
            for x in range(w):
                pixels[x, y] = (max(0, min(255, r)),
                                max(0, min(255, g)),
                                max(0, min(255, b)), 255)
        return img

    # Solid
    return Image.new("RGBA", (w, h), brand.background_color + (255,))


# ---------------------------------------------------------------------------
# Easing Functions
# ---------------------------------------------------------------------------
def _ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out curve."""
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


def _ease_out_elastic(t: float) -> float:
    """Elastic ease-out for bouncy effects."""
    if t <= 0 or t >= 1:
        return max(0.0, min(1.0, t))
    c4 = (2 * math.pi) / 3
    return 2.0 ** (-10 * t) * math.sin((t * 10 - 0.75) * c4) + 1


def _ease_out_quad(t: float) -> float:
    """Quadratic ease-out."""
    return 1 - (1 - t) ** 2


# ---------------------------------------------------------------------------
# Frame Renderers per Style
# ---------------------------------------------------------------------------
def _render_logo_reveal(frame_idx: int, total_frames: int,
                         brand: BrandKit, config: IntroConfig,
                         bg: "Image.Image") -> "Image.Image":  # noqa: F821
    """Render a single frame of the logo reveal intro."""
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageDraw

    frame = bg.copy()
    draw = ImageDraw.Draw(frame)
    w, h = config.width, config.height
    t = frame_idx / max(total_frames - 1, 1)

    # Phase 1: Logo fade in (0-0.5)
    # Phase 2: Glow pulse (0.3-0.7)
    # Phase 3: Tagline slide (0.5-0.9)
    # Phase 4: Fade out (0.85-1.0)

    alpha_master = 255
    if t < config.fade_in / config.duration:
        alpha_master = int(255 * _ease_in_out_cubic(t / (config.fade_in / config.duration)))
    elif t > 1.0 - config.fade_out / config.duration:
        fade_t = (t - (1.0 - config.fade_out / config.duration)) / (config.fade_out / config.duration)
        alpha_master = int(255 * (1.0 - _ease_in_out_cubic(fade_t)))

    # Logo
    logo_alpha = 0
    if t < 0.5:
        logo_alpha = int(alpha_master * _ease_in_out_cubic(t / 0.5))
    else:
        logo_alpha = alpha_master

    if brand.logo_path and os.path.isfile(brand.logo_path):
        try:
            logo = Image.open(brand.logo_path).convert("RGBA")
            logo_h = int(h * 0.2)
            logo_w = int(logo.width * (logo_h / max(logo.height, 1)))
            logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

            # Apply alpha
            r, g, b, a = logo.split()
            a = a.point(lambda p: int(p * logo_alpha / 255))
            logo = Image.merge("RGBA", (r, g, b, a))

            lx = (w - logo_w) // 2
            ly = int(h * 0.35) - logo_h // 2
            frame.paste(logo, (lx, ly), logo)
        except Exception:
            pass
    else:
        # Text fallback for logo
        font_size = max(30, int(h * 0.08))
        font = _load_font(brand, font_size)
        name = brand.name or "INTRO"
        bbox = draw.textbbox((0, 0), name, font=font)
        tw = bbox[2] - bbox[0]
        tx = (w - tw) // 2
        ty = int(h * 0.35)
        color = brand.primary_color + (logo_alpha,)
        draw.text((tx, ty), name, fill=color, font=font)

    # Glow effect (drawn as expanding circle behind logo)
    if 0.2 < t < 0.7:
        glow_t = (t - 0.2) / 0.5
        glow_r = int(min(w, h) * 0.15 * _ease_out_quad(glow_t))
        glow_alpha = int(40 * config.glow_intensity * (1 - glow_t) * alpha_master / 255)
        cx, cy = w // 2, int(h * 0.35)
        for ring in range(3):
            r = glow_r + ring * 8
            ring_alpha = max(0, glow_alpha - ring * 10)
            if ring_alpha > 0:
                color = brand.primary_color + (ring_alpha,)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                             outline=color, width=2)

    # Tagline
    if t > 0.45 and brand.tagline:
        tag_t = min(1.0, (t - 0.45) / 0.35)
        tag_alpha = int(alpha_master * _ease_in_out_cubic(tag_t))
        tag_y_offset = int(20 * (1 - _ease_out_quad(tag_t)))

        font_size = max(14, int(h * 0.03))
        font = _load_font(brand, font_size)
        bbox = draw.textbbox((0, 0), brand.tagline, font=font)
        tw = bbox[2] - bbox[0]
        tx = (w - tw) // 2
        ty = int(h * 0.55) + tag_y_offset
        draw.text((tx, ty), brand.tagline, fill=brand.text_color + (tag_alpha,), font=font)

    return frame.convert("RGB")


def _render_text_sweep(frame_idx: int, total_frames: int,
                        brand: BrandKit, config: IntroConfig,
                        bg: "Image.Image") -> "Image.Image":  # noqa: F821
    """Render a single frame of the text sweep intro."""
    ensure_package("PIL", "Pillow")
    from PIL import ImageDraw

    frame = bg.copy()
    draw = ImageDraw.Draw(frame)
    w, h = config.width, config.height
    t = frame_idx / max(total_frames - 1, 1)
    speed = config.animation_speed

    name = brand.name or "BRAND"
    font_size = max(24, int(h * 0.1))
    font = _load_font(brand, font_size)

    # Master fade
    alpha_master = 255
    if t < config.fade_in / config.duration:
        alpha_master = int(255 * (t / (config.fade_in / config.duration)))
    elif t > 1.0 - config.fade_out / config.duration:
        fade_t = (t - (1.0 - config.fade_out / config.duration)) / (config.fade_out / config.duration)
        alpha_master = int(255 * (1 - fade_t))

    # Draw each letter with staggered entrance
    total_chars = len(name)
    char_delay = 0.4 / max(total_chars, 1)
    full_bbox = draw.textbbox((0, 0), name, font=font)
    full_w = full_bbox[2] - full_bbox[0]
    start_x = (w - full_w) // 2
    cy = int(h * 0.42)

    x_offset = start_x
    for ci, char in enumerate(name):
        char_bbox = draw.textbbox((0, 0), char, font=font)
        char_w = char_bbox[2] - char_bbox[0]
        char_bbox[3] - char_bbox[1]

        # Each char appears at a staggered time
        char_start = ci * char_delay * speed
        char_t = max(0.0, (t - char_start) / (0.3 / speed))
        char_t = min(1.0, char_t)

        if char_t > 0:
            char_alpha = int(alpha_master * _ease_out_quad(char_t))
            # Sweep from below
            y_offset = int(30 * (1 - _ease_out_quad(char_t)))
            color = brand.primary_color + (char_alpha,)
            draw.text((x_offset, cy + y_offset), char, fill=color, font=font)

            # Trailing glow
            if char_t < 0.8:
                glow_alpha = int(30 * (1 - char_t / 0.8) * config.glow_intensity)
                glow_color = brand.accent_color + (glow_alpha,)
                draw.text((x_offset - 1, cy + y_offset - 1), char, fill=glow_color, font=font)

        x_offset += char_w + 1

    # Tagline appears after name
    if t > 0.55 and brand.tagline:
        tag_t = min(1.0, (t - 0.55) / 0.3)
        tag_font = _load_font(brand, max(12, int(h * 0.03)))
        tag_bbox = draw.textbbox((0, 0), brand.tagline, font=tag_font)
        tag_w = tag_bbox[2] - tag_bbox[0]
        tag_x = (w - tag_w) // 2
        tag_alpha = int(alpha_master * _ease_in_out_cubic(tag_t))
        draw.text((tag_x, cy + font_size + 20),
                  brand.tagline, fill=brand.text_color + (tag_alpha,), font=tag_font)

    return frame.convert("RGB")


def _render_particles(frame_idx: int, total_frames: int,
                       brand: BrandKit, config: IntroConfig,
                       bg: "Image.Image") -> "Image.Image":  # noqa: F821
    """Render a single frame of the particle convergence intro."""
    ensure_package("PIL", "Pillow")
    from PIL import ImageDraw

    frame = bg.copy()
    draw = ImageDraw.Draw(frame)
    w, h = config.width, config.height
    t = frame_idx / max(total_frames - 1, 1)

    alpha_master = 255
    if t < config.fade_in / config.duration:
        alpha_master = int(255 * (t / (config.fade_in / config.duration)))
    elif t > 1.0 - config.fade_out / config.duration:
        fade_t = (t - (1.0 - config.fade_out / config.duration)) / (config.fade_out / config.duration)
        alpha_master = int(255 * (1 - fade_t))

    cx, cy = w // 2, int(h * 0.42)
    n_particles = 60
    random.seed(42)  # Deterministic particles

    # Phase 1: Particles scatter (0-0.3)
    # Phase 2: Converge to center (0.3-0.7)
    # Phase 3: Hold as logo/text (0.7-1.0)

    for i in range(n_particles):
        angle = random.uniform(0, 2 * math.pi)
        max_dist = random.uniform(0.2, 0.5) * min(w, h)
        particle_size = random.randint(2, 6)

        if t < 0.3:
            # Scattered, moving inward slowly
            scatter_t = t / 0.3
            dist = max_dist * (1 - scatter_t * 0.3)
        elif t < 0.7:
            # Converging
            converge_t = (t - 0.3) / 0.4
            ease_t = _ease_in_out_cubic(converge_t)
            dist = max_dist * (0.7 * (1 - ease_t))
        else:
            dist = 0

        px = int(cx + dist * math.cos(angle))
        py = int(cy + dist * math.sin(angle))

        # Color variation
        color_choice = random.choice([brand.primary_color, brand.accent_color, brand.secondary_color])
        p_alpha = int(alpha_master * random.uniform(0.5, 1.0))
        color = color_choice + (p_alpha,)

        draw.ellipse([px - particle_size, py - particle_size,
                       px + particle_size, py + particle_size], fill=color)

    # Show text after convergence
    if t > 0.6:
        text_t = min(1.0, (t - 0.6) / 0.2)
        text_alpha = int(alpha_master * _ease_in_out_cubic(text_t))
        name = brand.name or "BRAND"
        font_size = max(24, int(h * 0.07))
        font = _load_font(brand, font_size)
        bbox = draw.textbbox((0, 0), name, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((w - tw) // 2, cy - font_size // 2), name,
                  fill=brand.text_color + (text_alpha,), font=font)

        if t > 0.75 and brand.tagline:
            tag_t = min(1.0, (t - 0.75) / 0.15)
            tag_alpha = int(alpha_master * _ease_in_out_cubic(tag_t))
            tag_font = _load_font(brand, max(12, int(h * 0.025)))
            tag_bbox = draw.textbbox((0, 0), brand.tagline, font=tag_font)
            tag_w = tag_bbox[2] - tag_bbox[0]
            draw.text(((w - tag_w) // 2, cy + font_size // 2 + 15),
                      brand.tagline, fill=brand.text_color + (tag_alpha,), font=tag_font)

    return frame.convert("RGB")


def _render_minimal_fade(frame_idx: int, total_frames: int,
                          brand: BrandKit, config: IntroConfig,
                          bg: "Image.Image") -> "Image.Image":  # noqa: F821
    """Render a single frame of the minimal fade intro."""
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageDraw

    frame = bg.copy()
    draw = ImageDraw.Draw(frame)
    w, h = config.width, config.height
    t = frame_idx / max(total_frames - 1, 1)

    # Simple fade in, hold, fade out
    if t < 0.3:
        alpha = int(255 * _ease_in_out_cubic(t / 0.3))
    elif t > 0.7:
        alpha = int(255 * (1 - _ease_in_out_cubic((t - 0.7) / 0.3)))
    else:
        alpha = 255

    # Logo
    if brand.logo_path and os.path.isfile(brand.logo_path):
        try:
            logo = Image.open(brand.logo_path).convert("RGBA")
            logo_h = int(h * 0.18)
            logo_w = int(logo.width * (logo_h / max(logo.height, 1)))
            logo = logo.resize((logo_w, logo_h), Image.LANCZOS)
            r, g, b, a = logo.split()
            a = a.point(lambda p: int(p * alpha / 255))
            logo = Image.merge("RGBA", (r, g, b, a))
            frame.paste(logo, ((w - logo_w) // 2, int(h * 0.38) - logo_h // 2), logo)
        except Exception:
            pass
    else:
        name = brand.name or "BRAND"
        font_size = max(28, int(h * 0.08))
        font = _load_font(brand, font_size)
        bbox = draw.textbbox((0, 0), name, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((w - tw) // 2, int(h * 0.38)),
                  name, fill=brand.primary_color + (alpha,), font=font)

    # Thin separator line
    line_alpha = int(alpha * 0.4)
    line_w = int(w * 0.08)
    line_y = int(h * 0.52)
    draw.line([(w // 2 - line_w, line_y), (w // 2 + line_w, line_y)],
              fill=brand.accent_color + (line_alpha,), width=2)

    # Tagline below
    if brand.tagline:
        tag_font = _load_font(brand, max(12, int(h * 0.025)))
        tag_bbox = draw.textbbox((0, 0), brand.tagline, font=tag_font)
        tag_w = tag_bbox[2] - tag_bbox[0]
        draw.text(((w - tag_w) // 2, line_y + 15),
                  brand.tagline, fill=brand.text_color + (alpha,), font=tag_font)

    return frame.convert("RGB")


def _render_kinetic(frame_idx: int, total_frames: int,
                     brand: BrandKit, config: IntroConfig,
                     bg: "Image.Image") -> "Image.Image":  # noqa: F821
    """Render a single frame of the kinetic motion intro."""
    ensure_package("PIL", "Pillow")
    from PIL import ImageDraw

    frame = bg.copy()
    draw = ImageDraw.Draw(frame)
    w, h = config.width, config.height
    t = frame_idx / max(total_frames - 1, 1)
    speed = config.animation_speed

    alpha_master = 255
    if t < config.fade_in / config.duration:
        alpha_master = int(255 * (t / (config.fade_in / config.duration)))
    elif t > 1.0 - config.fade_out / config.duration:
        fade_t = (t - (1.0 - config.fade_out / config.duration)) / (config.fade_out / config.duration)
        alpha_master = int(255 * (1 - fade_t))

    # Geometric shapes that bounce in
    shapes = [
        {"type": "rect", "x": 0.15, "y": 0.3, "w": 0.06, "h": 0.06, "color": brand.primary_color, "delay": 0.0},
        {"type": "circle", "x": 0.82, "y": 0.25, "r": 0.025, "color": brand.accent_color, "delay": 0.05},
        {"type": "rect", "x": 0.1, "y": 0.7, "w": 0.04, "h": 0.15, "color": brand.secondary_color, "delay": 0.1},
        {"type": "circle", "x": 0.88, "y": 0.65, "r": 0.03, "color": brand.primary_color, "delay": 0.08},
        {"type": "rect", "x": 0.75, "y": 0.75, "w": 0.08, "h": 0.04, "color": brand.accent_color, "delay": 0.12},
    ]

    for shape in shapes:
        shape_t = max(0.0, (t - shape["delay"] * speed) / (0.3 / speed))
        shape_t = min(1.0, shape_t)
        if shape_t <= 0:
            continue

        ease_t = _ease_out_elastic(shape_t) if shape_t < 1.0 else 1.0
        s_alpha = int(alpha_master * min(1.0, shape_t * 2) * 0.5)

        sx = int(shape["x"] * w)
        sy = int(shape["y"] * h)
        # Bounce from off-screen
        y_offset = int(h * 0.3 * (1 - ease_t))

        if shape["type"] == "rect":
            sw = int(shape["w"] * w)
            sh_val = int(shape["h"] * h)
            color = shape["color"] + (s_alpha,)
            draw.rectangle([sx, sy - y_offset, sx + sw, sy - y_offset + sh_val], fill=color)
        elif shape["type"] == "circle":
            sr = int(shape["r"] * min(w, h))
            color = shape["color"] + (s_alpha,)
            draw.ellipse([sx - sr, sy - y_offset - sr, sx + sr, sy - y_offset + sr], fill=color)

    # Brand name bounces in
    if t > 0.15:
        name_t = min(1.0, (t - 0.15) / 0.3 * speed)
        name_ease = _ease_out_elastic(name_t)
        name_alpha = int(alpha_master * min(1.0, name_t * 3))

        name = brand.name or "BRAND"
        font_size = max(28, int(h * 0.09))
        font = _load_font(brand, font_size)
        bbox = draw.textbbox((0, 0), name, font=font)
        tw = bbox[2] - bbox[0]
        # Slide from left
        target_x = (w - tw) // 2
        start_x = -tw
        nx = int(start_x + (target_x - start_x) * name_ease)
        ny = int(h * 0.42)
        draw.text((nx, ny), name, fill=brand.primary_color + (name_alpha,), font=font)

    # Tagline slides from right
    if t > 0.4 and brand.tagline:
        tag_t = min(1.0, (t - 0.4) / 0.25 * speed)
        tag_ease = _ease_out_quad(tag_t)
        tag_alpha = int(alpha_master * min(1.0, tag_t * 3))

        tag_font = _load_font(brand, max(12, int(h * 0.03)))
        tag_bbox = draw.textbbox((0, 0), brand.tagline, font=tag_font)
        tag_w = tag_bbox[2] - tag_bbox[0]
        target_tx = (w - tag_w) // 2
        start_tx = w + tag_w
        tx = int(start_tx + (target_tx - start_tx) * tag_ease)
        ty = int(h * 0.55)
        draw.text((tx, ty), brand.tagline,
                  fill=brand.text_color + (tag_alpha,), font=tag_font)

    return frame.convert("RGB")


# Style -> renderer mapping
_INTRO_RENDERERS = {
    "logo_reveal": _render_logo_reveal,
    "text_sweep": _render_text_sweep,
    "particles": _render_particles,
    "minimal_fade": _render_minimal_fade,
    "kinetic": _render_kinetic,
}


# ---------------------------------------------------------------------------
# Music Bed Integration
# ---------------------------------------------------------------------------
def _add_music_bed(video_path: str, music_path: str, volume: float,
                    duration: float, out_path: str) -> str:
    """Mix music bed into intro video at specified volume."""
    if not os.path.isfile(music_path):
        logger.warning("Music file not found: %s", music_path)
        return video_path

    cmd = (FFmpegCmd()
           .input(video_path)
           .input(music_path)
           .option("t", str(duration))
           .video_codec("copy")
           .audio_codec("aac")
           .option("b:a", "192k")
           .option("filter_complex",
                   f"[1:a]atrim=0:{duration},afade=t=in:d=0.5,afade=t=out:st={max(0, duration - 1)}:d=1,"
                   f"volume={volume}[music];[music]apad=whole_dur={duration}[a]")
           .option("map", "0:v")
           .option("map", "[a]")
           .option("shortest", None)
           .output(out_path)
           .build())

    try:
        run_ffmpeg(cmd)
        return out_path
    except Exception as exc:
        logger.warning("Failed to add music bed: %s", exc)
        return video_path


# ---------------------------------------------------------------------------
# Video Prepend
# ---------------------------------------------------------------------------
def _prepend_intro(intro_path: str, main_video: str, out_path: str) -> str:
    """Concatenate intro clip before the main video."""
    _fd, concat_file = tempfile.mkstemp(suffix=".txt", prefix="intro_concat_")
    os.close(_fd)
    try:
        with open(concat_file, "w") as f:
            f.write(f"file '{intro_path}'\n")
            f.write(f"file '{main_video}'\n")

        cmd = (FFmpegCmd()
               .option("f", "concat")
               .option("safe", "0")
               .input(concat_file)
               .video_codec("copy")
               .audio_codec("copy")
               .output(out_path)
               .build())
        run_ffmpeg(cmd)
        return out_path
    except Exception as exc:
        logger.error("Failed to prepend intro: %s", exc)
        return intro_path
    finally:
        if os.path.isfile(concat_file):
            try:
                os.unlink(concat_file)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main Generation Function
# ---------------------------------------------------------------------------
def generate_intro(
    brand_kit: Dict,
    style: str = "logo_reveal",
    duration: float = 5.0,
    config: Optional[IntroConfig] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> IntroResult:
    """Generate an animated video intro from a brand kit.

    Args:
        brand_kit: Dict with brand identity fields (name, tagline, logo_path,
            primary_color, secondary_color, etc.).
        style: Intro animation style ID.
        duration: Intro duration in seconds.
        config: Full intro configuration. Overrides style/duration if set.
        output_dir: Output directory. Defaults to temp directory.
        on_progress: Callback ``(pct, msg)`` for progress updates.

    Returns:
        IntroResult with output video path and metadata.

    Raises:
        ValueError: If brand kit or config validation fails.
    """
    # Build BrandKit from dict
    brand = BrandKit(
        name=brand_kit.get("name", ""),
        tagline=brand_kit.get("tagline", ""),
        logo_path=brand_kit.get("logo_path", ""),
        font_path=brand_kit.get("font_path", ""),
        font_name=brand_kit.get("font_name", "Arial"),
        primary_color=tuple(brand_kit.get("primary_color", (80, 140, 255))),
        secondary_color=tuple(brand_kit.get("secondary_color", (255, 200, 80))),
        accent_color=tuple(brand_kit.get("accent_color", (100, 220, 180))),
        background_color=tuple(brand_kit.get("background_color", (12, 12, 18))),
        text_color=tuple(brand_kit.get("text_color", (240, 240, 245))),
    )

    brand_errors = brand.validate()
    if brand_errors:
        raise ValueError("Invalid BrandKit: " + "; ".join(brand_errors))

    config = config or IntroConfig()
    config.style = style
    config.duration = duration
    config_errors = config.validate()
    if config_errors:
        raise ValueError("Invalid IntroConfig: " + "; ".join(config_errors))

    if on_progress:
        on_progress(5, f"Starting {style} intro generation...")

    out_dir = output_dir or tempfile.gettempdir()
    os.makedirs(out_dir, exist_ok=True)
    intro_name = f"intro_{style}_{brand.name or 'brand'}.mp4"
    intro_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in intro_name)
    out_path = os.path.join(out_dir, intro_name)

    total_frames = int(config.duration * config.fps)
    renderer = _INTRO_RENDERERS.get(config.style, _render_minimal_fade)

    if on_progress:
        on_progress(10, "Creating background...")

    bg = _create_bg(config, brand)

    # Render all frames
    frame_dir = tempfile.mkdtemp(prefix="intro_frames_")
    try:
        if on_progress:
            on_progress(15, f"Rendering {total_frames} frames...")

        for i in range(total_frames):
            frame = renderer(i, total_frames, brand, config, bg)
            frame_path = os.path.join(frame_dir, f"frame_{i:06d}.png")
            frame.save(frame_path, "PNG")

            if on_progress and (i % max(1, total_frames // 15) == 0):
                pct = 15 + int(55 * (i + 1) / total_frames)
                on_progress(pct, f"Rendered frame {i + 1}/{total_frames}")

        # Assemble video
        if on_progress:
            on_progress(75, "Assembling video...")

        pattern = os.path.join(frame_dir, "frame_%06d.png")
        cmd = (FFmpegCmd()
               .option("framerate", str(config.fps))
               .input(pattern)
               .video_codec("libx264")
               .option("pix_fmt", "yuv420p")
               .option("crf", "18")
               .option("preset", "medium")
               .output(out_path)
               .build())
        run_ffmpeg(cmd)

    finally:
        try:
            import shutil
            shutil.rmtree(frame_dir, ignore_errors=True)
        except Exception:
            pass

    if not os.path.isfile(out_path):
        raise RuntimeError("Intro video assembly failed")

    # Add music bed if provided
    has_music = False
    if config.music_path and os.path.isfile(config.music_path):
        if on_progress:
            on_progress(82, "Adding music bed...")
        music_out = os.path.splitext(out_path)[0] + "_music.mp4"
        result = _add_music_bed(out_path, config.music_path,
                                 config.music_volume, config.duration, music_out)
        if result != out_path and os.path.isfile(music_out):
            os.replace(music_out, out_path)
            has_music = True

    # Prepend to main video if requested
    prepended_to = ""
    if config.prepend_to and os.path.isfile(config.prepend_to):
        if on_progress:
            on_progress(90, "Prepending intro to main video...")
        combined_out = output_path(config.prepend_to, "with_intro")
        result = _prepend_intro(out_path, config.prepend_to, combined_out)
        if os.path.isfile(combined_out):
            out_path = combined_out
            prepended_to = config.prepend_to

    if on_progress:
        on_progress(100, "Intro generation complete")

    return IntroResult(
        output_path=out_path,
        duration=config.duration,
        width=config.width,
        height=config.height,
        fps=config.fps,
        frame_count=total_frames,
        style=config.style,
        has_music=has_music,
        prepended_to=prepended_to,
    )


def list_intro_styles() -> Dict:
    """Return the available intro style definitions."""
    return {
        k: {"name": v["name"], "description": v["description"],
             "min_duration": v["min_duration"], "max_duration": v["max_duration"]}
        for k, v in INTRO_STYLES.items()
    }
