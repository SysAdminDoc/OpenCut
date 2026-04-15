"""
OpenCut Auto Chapter Art Generation Module

Generate stylized chapter artwork and title cards:
- Extract key frames per chapter (from chapter detection data)
- Apply stylized treatments (blur + title overlay, gradient, split)
- Multiple card styles: minimal, bold, gradient, cinematic, split
- Auto-generate chapter titles from transcript segments
- Brand kit integration (fonts, colors, logo placement)
- Export as individual images or video overlay sequence

Uses Pillow for image composition, FFmpeg for video assembly.
"""

import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Card Styles
# ---------------------------------------------------------------------------
CARD_STYLES = {
    "minimal": {
        "name": "Minimal",
        "description": "Clean, subtle title card with light text on darkened key frame",
        "bg_darken": 0.6,
        "blur_radius": 8,
        "text_color": (240, 240, 245),
        "text_shadow": True,
        "font_size_ratio": 0.06,
        "text_position": "center",
        "show_chapter_number": True,
        "overlay_color": (0, 0, 0, 120),
    },
    "bold": {
        "name": "Bold",
        "description": "High-impact card with large text and colored bar accent",
        "bg_darken": 0.4,
        "blur_radius": 15,
        "text_color": (255, 255, 255),
        "text_shadow": True,
        "font_size_ratio": 0.09,
        "text_position": "center",
        "show_chapter_number": True,
        "accent_bar": True,
        "accent_color": (80, 140, 255),
        "overlay_color": (10, 10, 20, 160),
    },
    "gradient": {
        "name": "Gradient",
        "description": "Bottom gradient fade with title text anchored low",
        "bg_darken": 0.0,
        "blur_radius": 0,
        "text_color": (255, 255, 255),
        "text_shadow": True,
        "font_size_ratio": 0.065,
        "text_position": "bottom",
        "show_chapter_number": True,
        "gradient_height_ratio": 0.45,
        "gradient_color": (0, 0, 0),
        "overlay_color": None,
    },
    "cinematic": {
        "name": "Cinematic",
        "description": "Letterboxed frame with centered title between bars",
        "bg_darken": 0.3,
        "blur_radius": 3,
        "text_color": (220, 200, 170),
        "text_shadow": True,
        "font_size_ratio": 0.055,
        "text_position": "center",
        "show_chapter_number": False,
        "letterbox_ratio": 0.12,
        "letterbox_color": (0, 0, 0),
        "overlay_color": (0, 0, 0, 80),
    },
    "split": {
        "name": "Split",
        "description": "Half frame / half solid color with title on the solid side",
        "bg_darken": 0.0,
        "blur_radius": 0,
        "text_color": (255, 255, 255),
        "text_shadow": False,
        "font_size_ratio": 0.07,
        "text_position": "right",
        "show_chapter_number": True,
        "split_ratio": 0.5,
        "split_color": (25, 25, 35),
        "overlay_color": None,
    },
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class ChapterInfo:
    """Input chapter data (from chapter detection or manual)."""
    index: int = 0
    title: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    transcript_segment: str = ""
    key_frame_time: float = -1.0


@dataclass
class BrandKit:
    """Brand identity for consistent card styling."""
    font_path: str = ""
    font_name: str = "Arial"
    primary_color: Tuple[int, int, int] = (80, 140, 255)
    secondary_color: Tuple[int, int, int] = (255, 200, 80)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    logo_path: str = ""
    logo_position: str = "top_right"
    logo_scale: float = 0.08

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChapterArtConfig:
    """Configuration for chapter art generation."""
    style: str = "minimal"
    width: int = 1920
    height: int = 1080
    card_duration: float = 3.0
    fps: int = 30
    fade_in: float = 0.3
    fade_out: float = 0.3
    export_images: bool = True
    export_video: bool = False
    brand_kit: Optional[BrandKit] = None
    title_prefix: str = "Chapter"
    auto_title_from_transcript: bool = True
    max_title_length: int = 50
    image_format: str = "png"
    image_quality: int = 95

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def validate(self) -> List[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        if self.style not in CARD_STYLES:
            errors.append(f"Unknown style '{self.style}', must be one of {list(CARD_STYLES.keys())}")
        if self.width < 64 or self.width > 7680:
            errors.append(f"Width {self.width} out of range [64, 7680]")
        if self.height < 64 or self.height > 4320:
            errors.append(f"Height {self.height} out of range [64, 4320]")
        if self.card_duration < 0.5 or self.card_duration > 30.0:
            errors.append(f"card_duration {self.card_duration} out of range [0.5, 30.0]")
        if self.image_format not in ("png", "jpg", "jpeg", "webp"):
            errors.append(f"Unsupported image_format '{self.image_format}'")
        return errors


@dataclass
class ChapterCard:
    """A single generated chapter card."""
    chapter_index: int = 0
    title: str = ""
    image_path: str = ""
    start_time: float = 0.0
    width: int = 0
    height: int = 0
    style: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChapterArtResult:
    """Result of chapter art generation."""
    cards: List[ChapterCard] = field(default_factory=list)
    video_path: str = ""
    output_dir: str = ""
    total_chapters: int = 0
    style: str = ""

    def to_dict(self) -> dict:
        return {
            "cards": [c.to_dict() for c in self.cards],
            "video_path": self.video_path,
            "output_dir": self.output_dir,
            "total_chapters": self.total_chapters,
            "style": self.style,
        }


# ---------------------------------------------------------------------------
# Key Frame Extraction
# ---------------------------------------------------------------------------
def _extract_key_frame(video_path: str, timestamp: float,
                        width: int, height: int) -> Optional[str]:
    """Extract a single frame from video at the given timestamp.

    Returns path to the extracted PNG frame, or None on failure.
    """
    out_path = tempfile.mktemp(suffix=".png", prefix="chapter_frame_")
    cmd = (FFmpegCmd()
           .option("ss", str(timestamp))
           .input(video_path)
           .option("frames:v", "1")
           .video_filter(f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                         f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black")
           .output(out_path)
           .build())
    try:
        run_ffmpeg(cmd)
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 100:
            return out_path
    except Exception as exc:
        logger.warning("Frame extraction failed at %.2fs: %s", timestamp, exc)

    return None


def _select_key_frame_time(chapter: ChapterInfo) -> float:
    """Choose the best timestamp for a chapter's key frame.

    Prefers explicit key_frame_time, then 20% into the chapter, then start.
    """
    if chapter.key_frame_time >= 0:
        return chapter.key_frame_time

    duration = chapter.end_time - chapter.start_time
    if duration > 2.0:
        return chapter.start_time + duration * 0.2
    return chapter.start_time + 0.5


# ---------------------------------------------------------------------------
# Title Generation
# ---------------------------------------------------------------------------
def _auto_title_from_transcript(transcript: str, max_length: int = 50) -> str:
    """Generate a concise chapter title from a transcript segment.

    Extracts the first meaningful sentence or phrase.
    """
    if not transcript or not transcript.strip():
        return ""

    # Clean up
    text = transcript.strip()
    text = re.sub(r"\s+", " ", text)

    # Try to find a sentence
    sentences = re.split(r"[.!?]+", text)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) >= 5:
            # Capitalize first letter
            title = sent[0].upper() + sent[1:]
            if len(title) > max_length:
                # Truncate at word boundary
                truncated = title[:max_length].rsplit(" ", 1)[0]
                return truncated + "..."
            return title

    # Fallback: use first N words
    words = text.split()[:8]
    title = " ".join(words)
    if len(title) > max_length:
        title = title[:max_length - 3] + "..."
    return title.capitalize() if title else ""


# ---------------------------------------------------------------------------
# Card Rendering
# ---------------------------------------------------------------------------
def _load_font(brand_kit: Optional[BrandKit], size: int):
    """Load font from brand kit or fall back to system defaults."""
    ensure_package("PIL", "Pillow")
    from PIL import ImageFont

    if brand_kit and brand_kit.font_path and os.path.isfile(brand_kit.font_path):
        try:
            return ImageFont.truetype(brand_kit.font_path, size=size)
        except (IOError, OSError):
            logger.warning("Failed to load brand font: %s", brand_kit.font_path)

    # Try common system fonts
    for font_name in ["arial", "Arial", "Helvetica", "DejaVuSans", "segoeui"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except (IOError, OSError):
            continue

    return ImageFont.load_default()


def _add_logo(frame: "Image.Image", brand_kit: BrandKit) -> "Image.Image":  # noqa: F821
    """Overlay brand logo onto the card frame."""
    ensure_package("PIL", "Pillow")
    from PIL import Image

    if not brand_kit.logo_path or not os.path.isfile(brand_kit.logo_path):
        return frame

    try:
        logo = Image.open(brand_kit.logo_path).convert("RGBA")
        logo_h = int(frame.height * brand_kit.logo_scale)
        logo_w = int(logo.width * (logo_h / logo.height))
        logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

        margin = int(frame.width * 0.02)
        positions = {
            "top_right": (frame.width - logo_w - margin, margin),
            "top_left": (margin, margin),
            "bottom_right": (frame.width - logo_w - margin, frame.height - logo_h - margin),
            "bottom_left": (margin, frame.height - logo_h - margin),
        }
        pos = positions.get(brand_kit.logo_position, positions["top_right"])
        frame.paste(logo, pos, logo)
    except Exception as exc:
        logger.warning("Failed to overlay logo: %s", exc)

    return frame


def _render_card_minimal(key_frame: "Image.Image", title: str, chapter_idx: int,  # noqa: F821
                          config: ChapterArtConfig, style_def: Dict) -> "Image.Image":  # noqa: F821
    """Render a minimal-style chapter card."""
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageDraw, ImageFilter

    w, h = config.width, config.height
    card = key_frame.copy().resize((w, h), Image.LANCZOS)

    # Apply blur
    if style_def.get("blur_radius", 0) > 0:
        card = card.filter(ImageFilter.GaussianBlur(radius=style_def["blur_radius"]))

    # Darken overlay
    overlay_color = style_def.get("overlay_color")
    if overlay_color:
        overlay = Image.new("RGBA", (w, h), overlay_color)
        card = card.convert("RGBA")
        card = Image.alpha_composite(card, overlay)

    draw = ImageDraw.Draw(card)
    font_size = max(16, int(h * style_def["font_size_ratio"]))
    font = _load_font(config.brand_kit, font_size)
    small_font = _load_font(config.brand_kit, max(12, font_size // 2))

    text_color = style_def["text_color"]

    # Chapter number
    if style_def.get("show_chapter_number"):
        num_text = f"{config.title_prefix} {chapter_idx + 1}"
        num_bbox = draw.textbbox((0, 0), num_text, font=small_font)
        num_w = num_bbox[2] - num_bbox[0]
        nx = (w - num_w) // 2
        ny = h // 2 - font_size - 10
        if style_def.get("text_shadow"):
            draw.text((nx + 2, ny + 2), num_text, fill=(0, 0, 0, 180), font=small_font)
        draw.text((nx, ny), num_text, fill=(text_color[0], text_color[1], text_color[2], 180), font=small_font)

    # Title text
    title_bbox = draw.textbbox((0, 0), title, font=font)
    title_w = title_bbox[2] - title_bbox[0]
    title_h_val = title_bbox[3] - title_bbox[1]

    # Word wrap if too wide
    if title_w > w * 0.85:
        max_chars = int(len(title) * (w * 0.85) / max(title_w, 1))
        lines = []
        current = ""
        for word in title.split():
            test = current + " " + word if current else word
            if len(test) > max_chars and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)
    else:
        lines = [title]

    ty = h // 2 - (len(lines) * (title_h_val + 5)) // 2 + 10
    for line in lines:
        line_bbox = draw.textbbox((0, 0), line, font=font)
        lw = line_bbox[2] - line_bbox[0]
        lx = (w - lw) // 2
        if style_def.get("text_shadow"):
            draw.text((lx + 2, ty + 2), line, fill=(0, 0, 0, 200), font=font)
        draw.text((lx, ty), line, fill=text_color, font=font)
        ty += title_h_val + 5

    return card.convert("RGB")


def _render_card_bold(key_frame: "Image.Image", title: str, chapter_idx: int,  # noqa: F821
                       config: ChapterArtConfig, style_def: Dict) -> "Image.Image":  # noqa: F821
    """Render a bold-style chapter card with accent bar."""
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageDraw, ImageFilter

    w, h = config.width, config.height
    card = key_frame.copy().resize((w, h), Image.LANCZOS)

    if style_def.get("blur_radius", 0) > 0:
        card = card.filter(ImageFilter.GaussianBlur(radius=style_def["blur_radius"]))

    overlay_color = style_def.get("overlay_color")
    if overlay_color:
        overlay = Image.new("RGBA", (w, h), overlay_color)
        card = card.convert("RGBA")
        card = Image.alpha_composite(card, overlay)

    draw = ImageDraw.Draw(card)
    font_size = max(20, int(h * style_def["font_size_ratio"]))
    font = _load_font(config.brand_kit, font_size)
    small_font = _load_font(config.brand_kit, max(14, font_size // 2))

    accent_color = style_def.get("accent_color", (80, 140, 255))
    if config.brand_kit:
        accent_color = config.brand_kit.primary_color

    # Accent bar
    bar_h = max(4, h // 80)
    bar_w = int(w * 0.15)
    bar_y = h // 2 - font_size - 30
    bar_x = (w - bar_w) // 2
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                    fill=accent_color + (255,))

    # Chapter number
    if style_def.get("show_chapter_number"):
        num_text = f"{config.title_prefix.upper()} {chapter_idx + 1}"
        num_bbox = draw.textbbox((0, 0), num_text, font=small_font)
        nw = num_bbox[2] - num_bbox[0]
        draw.text(((w - nw) // 2, bar_y - (num_bbox[3] - num_bbox[1]) - 10),
                  num_text, fill=accent_color + (255,), font=small_font)

    # Title
    title_upper = title.upper()
    title_bbox = draw.textbbox((0, 0), title_upper, font=font)
    tw = title_bbox[2] - title_bbox[0]
    tx = (w - tw) // 2
    ty = bar_y + bar_h + 15
    if style_def.get("text_shadow"):
        draw.text((tx + 3, ty + 3), title_upper, fill=(0, 0, 0, 200), font=font)
    draw.text((tx, ty), title_upper, fill=style_def["text_color"], font=font)

    return card.convert("RGB")


def _render_card_gradient(key_frame: "Image.Image", title: str, chapter_idx: int,  # noqa: F821
                           config: ChapterArtConfig, style_def: Dict) -> "Image.Image":  # noqa: F821
    """Render a gradient-style chapter card with bottom fade."""
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageDraw

    w, h = config.width, config.height
    card = key_frame.copy().resize((w, h), Image.LANCZOS).convert("RGBA")

    # Bottom gradient
    grad_h = int(h * style_def.get("gradient_height_ratio", 0.45))
    grad_color = style_def.get("gradient_color", (0, 0, 0))
    gradient = Image.new("RGBA", (w, grad_h), (0, 0, 0, 0))
    gpx = gradient.load()
    for y in range(grad_h):
        alpha = int(220 * (y / max(grad_h - 1, 1)))
        for x in range(w):
            gpx[x, y] = (grad_color[0], grad_color[1], grad_color[2], alpha)

    card.paste(gradient, (0, h - grad_h), gradient)

    draw = ImageDraw.Draw(card)
    font_size = max(18, int(h * style_def["font_size_ratio"]))
    font = _load_font(config.brand_kit, font_size)
    small_font = _load_font(config.brand_kit, max(12, font_size // 2))

    # Chapter number at bottom-left
    margin = int(w * 0.05)
    if style_def.get("show_chapter_number"):
        num_text = f"{config.title_prefix} {chapter_idx + 1}"
        draw.text((margin, h - font_size * 2 - 30), num_text,
                  fill=(200, 200, 210, 180), font=small_font)

    # Title at bottom
    ty = h - font_size - 25
    if style_def.get("text_shadow"):
        draw.text((margin + 2, ty + 2), title, fill=(0, 0, 0, 200), font=font)
    draw.text((margin, ty), title, fill=style_def["text_color"], font=font)

    return card.convert("RGB")


def _render_card_cinematic(key_frame: "Image.Image", title: str, chapter_idx: int,  # noqa: F821
                            config: ChapterArtConfig, style_def: Dict) -> "Image.Image":  # noqa: F821
    """Render a cinematic letterboxed chapter card."""
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageDraw, ImageFilter

    w, h = config.width, config.height
    card = key_frame.copy().resize((w, h), Image.LANCZOS)

    if style_def.get("blur_radius", 0) > 0:
        card = card.filter(ImageFilter.GaussianBlur(radius=style_def["blur_radius"]))

    overlay_color = style_def.get("overlay_color")
    if overlay_color:
        overlay = Image.new("RGBA", (w, h), overlay_color)
        card = card.convert("RGBA")
        card = Image.alpha_composite(card, overlay)

    draw = ImageDraw.Draw(card)

    # Letterbox bars
    bar_height = int(h * style_def.get("letterbox_ratio", 0.12))
    bar_color = style_def.get("letterbox_color", (0, 0, 0))
    draw.rectangle([0, 0, w, bar_height], fill=bar_color)
    draw.rectangle([0, h - bar_height, w, h], fill=bar_color)

    # Title centered
    font_size = max(16, int(h * style_def["font_size_ratio"]))
    font = _load_font(config.brand_kit, font_size)
    text_color = style_def["text_color"]

    title_bbox = draw.textbbox((0, 0), title, font=font)
    tw = title_bbox[2] - title_bbox[0]
    th = title_bbox[3] - title_bbox[1]
    tx = (w - tw) // 2
    ty = (h - th) // 2

    if style_def.get("text_shadow"):
        draw.text((tx + 2, ty + 2), title, fill=(0, 0, 0, 180), font=font)
    draw.text((tx, ty), title, fill=text_color, font=font)

    return card.convert("RGB")


def _render_card_split(key_frame: "Image.Image", title: str, chapter_idx: int,  # noqa: F821
                        config: ChapterArtConfig, style_def: Dict) -> "Image.Image":  # noqa: F821
    """Render a split-screen chapter card: image left, text right."""
    ensure_package("PIL", "Pillow")
    from PIL import Image, ImageDraw

    w, h = config.width, config.height
    split = style_def.get("split_ratio", 0.5)
    split_px = int(w * split)

    card = Image.new("RGB", (w, h), style_def.get("split_color", (25, 25, 35)))

    # Left: key frame
    left_img = key_frame.copy().resize((split_px, h), Image.LANCZOS)
    card.paste(left_img, (0, 0))

    # Right: text panel
    draw = ImageDraw.Draw(card)
    font_size = max(18, int(h * style_def["font_size_ratio"]))
    font = _load_font(config.brand_kit, font_size)
    small_font = _load_font(config.brand_kit, max(12, font_size // 2))

    text_x = split_px + int(w * 0.04)
    text_color = style_def["text_color"]

    accent = (80, 140, 255)
    if config.brand_kit:
        accent = config.brand_kit.primary_color

    # Chapter number
    if style_def.get("show_chapter_number"):
        num_text = f"{config.title_prefix} {chapter_idx + 1}"
        draw.text((text_x, h // 2 - font_size - 20), num_text,
                  fill=accent, font=small_font)

    # Title
    max_text_w = w - split_px - int(w * 0.08)
    title_bbox = draw.textbbox((0, 0), title, font=font)
    tw = title_bbox[2] - title_bbox[0]
    th = title_bbox[3] - title_bbox[1]

    if tw > max_text_w:
        max_chars = int(len(title) * max_text_w / max(tw, 1))
        lines = []
        current = ""
        for word in title.split():
            test = current + " " + word if current else word
            if len(test) > max_chars and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)
    else:
        lines = [title]

    ty = h // 2
    for line in lines:
        draw.text((text_x, ty), line, fill=text_color, font=font)
        ty += th + 8

    # Accent line between halves
    draw.line([(split_px, 0), (split_px, h)], fill=accent, width=3)

    return card


# Style -> renderer mapping
_CARD_RENDERERS = {
    "minimal": _render_card_minimal,
    "bold": _render_card_bold,
    "gradient": _render_card_gradient,
    "cinematic": _render_card_cinematic,
    "split": _render_card_split,
}


# ---------------------------------------------------------------------------
# Video Overlay Assembly
# ---------------------------------------------------------------------------
def _assemble_cards_to_video(card_paths: List[str], chapters: List[ChapterInfo],
                              config: ChapterArtConfig, out_path: str) -> str:
    """Assemble chapter cards into a video overlay sequence.

    Each card is shown at its chapter start time for card_duration seconds
    with configurable fade in/out.
    """
    # Build concat input with timed images
    concat_file = tempfile.mktemp(suffix=".txt", prefix="chapter_concat_")
    try:
        with open(concat_file, "w") as f:
            for i, card_path in enumerate(card_paths):
                f.write(f"file '{card_path}'\n")
                f.write(f"duration {config.card_duration}\n")
            # Repeat last frame to avoid truncation
            if card_paths:
                f.write(f"file '{card_paths[-1]}'\n")

        cmd = (FFmpegCmd()
               .option("f", "concat")
               .option("safe", "0")
               .input(concat_file)
               .video_codec("libx264")
               .option("pix_fmt", "yuv420p")
               .option("crf", "18")
               .option("preset", "medium")
               .output(out_path)
               .build())
        run_ffmpeg(cmd)
    finally:
        if os.path.isfile(concat_file):
            try:
                os.unlink(concat_file)
            except OSError:
                pass

    return out_path


# ---------------------------------------------------------------------------
# Main Generation Function
# ---------------------------------------------------------------------------
def generate_chapter_art(
    video_path: str,
    chapters: List[Dict],
    config: Optional[ChapterArtConfig] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ChapterArtResult:
    """Generate chapter artwork cards for a video.

    Args:
        video_path: Path to the source video file.
        chapters: List of chapter dicts with keys: index, title, start_time,
            end_time, transcript_segment (optional), key_frame_time (optional).
        config: Art generation configuration. Uses defaults if None.
        output_dir: Output directory. Defaults to video directory.
        on_progress: Callback ``(pct, msg)`` for progress updates.

    Returns:
        ChapterArtResult with generated cards and optional video.

    Raises:
        FileNotFoundError: If video not found.
        ValueError: If no chapters or config invalid.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not chapters:
        raise ValueError("No chapters provided")

    config = config or ChapterArtConfig()
    errors = config.validate()
    if errors:
        raise ValueError("Invalid ChapterArtConfig: " + "; ".join(errors))

    # Match video dimensions if not explicitly set
    info = get_video_info(video_path)
    if config.width == 1920 and config.height == 1080:
        config.width = info.get("width", 1920)
        config.height = info.get("height", 1080)

    if on_progress:
        on_progress(5, "Preparing chapter art generation...")

    out_dir = output_dir or os.path.dirname(video_path)
    os.makedirs(out_dir, exist_ok=True)

    style_def = CARD_STYLES[config.style]
    renderer = _CARD_RENDERERS[config.style]

    ensure_package("PIL", "Pillow")
    from PIL import Image

    chapter_objects = []
    for ch in chapters:
        chapter_objects.append(ChapterInfo(
            index=ch.get("index", 0),
            title=ch.get("title", ""),
            start_time=ch.get("start_time", 0.0),
            end_time=ch.get("end_time", 0.0),
            transcript_segment=ch.get("transcript_segment", ""),
            key_frame_time=ch.get("key_frame_time", -1.0),
        ))

    cards = []
    card_paths = []
    total = len(chapter_objects)

    for i, chapter in enumerate(chapter_objects):
        if on_progress:
            pct = 10 + int(80 * i / total)
            on_progress(pct, f"Generating card {i + 1}/{total}...")

        # Determine title
        title = chapter.title
        if not title and config.auto_title_from_transcript and chapter.transcript_segment:
            title = _auto_title_from_transcript(
                chapter.transcript_segment, config.max_title_length)
        if not title:
            title = f"{config.title_prefix} {chapter.index + 1}"

        # Extract key frame
        frame_time = _select_key_frame_time(chapter)
        frame_path = _extract_key_frame(video_path, frame_time, config.width, config.height)

        if frame_path:
            key_frame = Image.open(frame_path).convert("RGBA")
        else:
            # Solid color fallback
            key_frame = Image.new("RGBA", (config.width, config.height), (30, 30, 40, 255))

        # Render card
        card_img = renderer(key_frame, title, i, config, style_def)

        # Add brand logo
        if config.brand_kit:
            card_img = _add_logo(card_img.convert("RGBA"), config.brand_kit).convert("RGB")

        # Save card
        ext = config.image_format.lower()
        if ext == "jpeg":
            ext = "jpg"
        card_filename = f"chapter_{i + 1:02d}_{config.style}.{ext}"
        card_out = os.path.join(out_dir, card_filename)
        save_kwargs = {}
        if ext in ("jpg", "jpeg"):
            save_kwargs["quality"] = config.image_quality
        elif ext == "webp":
            save_kwargs["quality"] = config.image_quality
        card_img.save(card_out, **save_kwargs)

        cards.append(ChapterCard(
            chapter_index=i,
            title=title,
            image_path=card_out,
            start_time=chapter.start_time,
            width=config.width,
            height=config.height,
            style=config.style,
        ))
        card_paths.append(card_out)

        # Clean up extracted frame
        if frame_path and os.path.isfile(frame_path):
            try:
                os.unlink(frame_path)
            except OSError:
                pass

    # Optional video assembly
    video_out = ""
    if config.export_video and card_paths:
        if on_progress:
            on_progress(90, "Assembling chapter cards into video...")
        video_out = os.path.join(out_dir, f"chapter_cards_{config.style}.mp4")
        try:
            _assemble_cards_to_video(card_paths, chapter_objects, config, video_out)
        except Exception as exc:
            logger.error("Video assembly failed: %s", exc)
            video_out = ""

    if on_progress:
        on_progress(100, "Chapter art generation complete")

    return ChapterArtResult(
        cards=cards,
        video_path=video_out,
        output_dir=out_dir,
        total_chapters=total,
        style=config.style,
    )


def list_card_styles() -> Dict:
    """Return the available card style definitions."""
    return {k: {"name": v["name"], "description": v["description"]} for k, v in CARD_STYLES.items()}
