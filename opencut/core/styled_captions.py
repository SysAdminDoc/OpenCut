"""
Styled caption overlay renderer.

Generates transparent video overlays with styled captions including:
- Multiple visual presets (YouTube Bold, Clean Modern, etc.)
- Word-by-word highlight animation
- Action word emphasis (auto-detected or user-specified)
- Stroke, shadow, background box styling

Uses Pillow for text rendering and FFmpeg for video assembly.
"""

import logging
import math
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from .captions import CaptionSegment, TranscriptionResult, Word

logger = logging.getLogger("opencut.styled_captions")

# ---------------------------------------------------------------------------
# Style definitions
# ---------------------------------------------------------------------------

@dataclass
class CaptionStyle:
    """Visual style for rendered captions."""
    name: str
    label: str                          # Human-readable label
    font_name: str = "Arial Bold"       # Font lookup name
    font_file: str = "arialbd.ttf"      # Fallback filename
    font_size: int = 56                 # Pixels at 1080p
    text_color: Tuple = (255, 255, 255, 255)
    highlight_color: Tuple = (255, 255, 0, 255)
    action_color: Tuple = (255, 80, 80, 255)
    stroke_color: Tuple = (0, 0, 0, 255)
    stroke_width: int = 3
    shadow_color: Tuple = (0, 0, 0, 0)  # alpha 0 = no shadow
    shadow_offset: Tuple = (0, 0)
    bg_color: Tuple = (0, 0, 0, 0)      # alpha 0 = no background
    bg_padding: Tuple = (16, 8)          # h, v padding
    bg_radius: int = 8
    margin_bottom: int = 80             # px from bottom
    max_width_pct: float = 0.82         # Max text width as fraction of video width
    line_spacing: float = 1.3           # Line height multiplier
    # CSS-like properties for panel preview
    preview_css: str = ""


STYLES: Dict[str, CaptionStyle] = {
    "youtube_bold": CaptionStyle(
        name="youtube_bold",
        label="YouTube Bold",
        font_name="Impact",
        font_file="impact.ttf",
        font_size=62,
        text_color=(255, 255, 255, 255),
        highlight_color=(255, 230, 0, 255),
        action_color=(255, 50, 50, 255),
        stroke_color=(0, 0, 0, 255),
        stroke_width=4,
        margin_bottom=80,
        preview_css="font-family:Impact,sans-serif;color:#fff;-webkit-text-stroke:2px #000;text-shadow:3px 3px 0 #000",
    ),
    "clean_modern": CaptionStyle(
        name="clean_modern",
        label="Clean Modern",
        font_name="Arial Bold",
        font_file="arialbd.ttf",
        font_size=52,
        text_color=(255, 255, 255, 255),
        highlight_color=(0, 200, 255, 255),
        action_color=(255, 100, 100, 255),
        stroke_color=(0, 0, 0, 200),
        stroke_width=2,
        bg_color=(0, 0, 0, 160),
        bg_padding=(20, 10),
        bg_radius=12,
        margin_bottom=70,
        preview_css="font-family:Arial,Helvetica,sans-serif;font-weight:700;color:#fff;background:rgba(0,0,0,0.6);padding:4px 10px;border-radius:6px;text-shadow:1px 1px 2px rgba(0,0,0,0.8)",
    ),
    "neon_pop": CaptionStyle(
        name="neon_pop",
        label="Neon Pop",
        font_name="Arial Black",
        font_file="ariblk.ttf",
        font_size=58,
        text_color=(255, 255, 80, 255),
        highlight_color=(80, 255, 80, 255),
        action_color=(255, 50, 200, 255),
        stroke_color=(100, 0, 150, 255),
        stroke_width=3,
        shadow_color=(150, 0, 255, 100),
        shadow_offset=(0, 0),
        margin_bottom=80,
        preview_css="font-family:'Arial Black',Arial,sans-serif;color:#ffff50;-webkit-text-stroke:2px #6400a0;text-shadow:0 0 12px rgba(150,0,255,0.7),0 0 24px rgba(150,0,255,0.3)",
    ),
    "minimal": CaptionStyle(
        name="minimal",
        label="Minimal",
        font_name="Arial",
        font_file="arial.ttf",
        font_size=44,
        text_color=(230, 230, 230, 255),
        highlight_color=(255, 255, 255, 255),
        action_color=(255, 180, 100, 255),
        stroke_color=(0, 0, 0, 180),
        stroke_width=1,
        shadow_color=(0, 0, 0, 120),
        shadow_offset=(2, 2),
        margin_bottom=60,
        preview_css="font-family:Arial,Helvetica,sans-serif;color:#e6e6e6;text-shadow:2px 2px 3px rgba(0,0,0,0.5);font-size:0.85em",
    ),
    "boxed": CaptionStyle(
        name="boxed",
        label="Boxed",
        font_name="Arial Bold",
        font_file="arialbd.ttf",
        font_size=50,
        text_color=(255, 255, 255, 255),
        highlight_color=(255, 200, 50, 255),
        action_color=(255, 80, 80, 255),
        stroke_color=(0, 0, 0, 0),
        stroke_width=0,
        bg_color=(15, 15, 15, 220),
        bg_padding=(24, 12),
        bg_radius=4,
        margin_bottom=70,
        preview_css="font-family:Arial,Helvetica,sans-serif;font-weight:700;color:#fff;background:rgba(15,15,15,0.85);padding:6px 14px;border-radius:3px",
    ),
    "cinematic": CaptionStyle(
        name="cinematic",
        label="Cinematic",
        font_name="Georgia",
        font_file="georgia.ttf",
        font_size=46,
        text_color=(220, 220, 210, 255),
        highlight_color=(255, 240, 200, 255),
        action_color=(255, 160, 80, 255),
        stroke_color=(0, 0, 0, 0),
        stroke_width=0,
        shadow_color=(0, 0, 0, 180),
        shadow_offset=(3, 3),
        margin_bottom=90,
        line_spacing=1.5,
        preview_css="font-family:Georgia,'Times New Roman',serif;color:#dcdcd2;text-shadow:3px 3px 5px rgba(0,0,0,0.7);letter-spacing:0.5px",
    ),

    # ---- Phase 2 Styles ----

    "glow": CaptionStyle(
        name="glow",
        label="Glow",
        font_name="Arial Black",
        font_file="ariblk.ttf",
        font_size=56,
        text_color=(255, 255, 255, 255),
        highlight_color=(0, 230, 255, 255),
        action_color=(255, 50, 150, 255),
        stroke_color=(0, 180, 255, 80),
        stroke_width=4,
        shadow_color=(0, 150, 255, 120),
        shadow_offset=(0, 0),
        margin_bottom=80,
        preview_css="font-family:'Arial Black',Arial,sans-serif;color:#fff;text-shadow:0 0 10px rgba(0,180,255,0.8),0 0 20px rgba(0,180,255,0.5),0 0 40px rgba(0,180,255,0.3);-webkit-text-stroke:1px rgba(0,180,255,0.3)",
    ),
    "karaoke": CaptionStyle(
        name="karaoke",
        label="Karaoke",
        font_name="Impact",
        font_file="impact.ttf",
        font_size=60,
        text_color=(200, 200, 220, 255),
        highlight_color=(255, 50, 200, 255),
        action_color=(255, 220, 50, 255),
        stroke_color=(0, 0, 0, 255),
        stroke_width=3,
        bg_color=(0, 0, 0, 180),
        bg_padding=(20, 10),
        bg_radius=6,
        margin_bottom=70,
        preview_css="font-family:Impact,sans-serif;color:#c8c8dc;-webkit-text-stroke:2px #000;background:rgba(0,0,0,0.7);padding:4px 12px;border-radius:4px",
    ),
    "outline": CaptionStyle(
        name="outline",
        label="Outline",
        font_name="Arial Bold",
        font_file="arialbd.ttf",
        font_size=58,
        text_color=(0, 0, 0, 0),
        highlight_color=(255, 255, 255, 255),
        action_color=(255, 100, 100, 255),
        stroke_color=(255, 255, 255, 255),
        stroke_width=3,
        margin_bottom=80,
        preview_css="font-family:Arial,Helvetica,sans-serif;font-weight:700;color:transparent;-webkit-text-stroke:2px #fff;letter-spacing:1px",
    ),
    "gradient_fire": CaptionStyle(
        name="gradient_fire",
        label="Fire Gradient",
        font_name="Impact",
        font_file="impact.ttf",
        font_size=60,
        text_color=(255, 200, 50, 255),
        highlight_color=(255, 80, 0, 255),
        action_color=(255, 30, 30, 255),
        stroke_color=(100, 0, 0, 255),
        stroke_width=3,
        shadow_color=(255, 100, 0, 80),
        shadow_offset=(0, 2),
        margin_bottom=80,
        preview_css="font-family:Impact,sans-serif;color:#ffc832;-webkit-text-stroke:2px #640000;text-shadow:0 2px 8px rgba(255,100,0,0.5),0 0 20px rgba(255,50,0,0.3)",
    ),
    "typewriter": CaptionStyle(
        name="typewriter",
        label="Typewriter",
        font_name="Courier New",
        font_file="cour.ttf",
        font_size=44,
        text_color=(220, 220, 200, 255),
        highlight_color=(255, 255, 255, 255),
        action_color=(100, 255, 100, 255),
        stroke_color=(0, 0, 0, 0),
        stroke_width=0,
        bg_color=(20, 20, 15, 220),
        bg_padding=(24, 12),
        bg_radius=2,
        margin_bottom=60,
        line_spacing=1.4,
        preview_css="font-family:'Courier New',Courier,monospace;color:#dcdcc8;background:rgba(20,20,15,0.85);padding:6px 14px;border-radius:2px;letter-spacing:0.5px",
    ),
    "bounce": CaptionStyle(
        name="bounce",
        label="Bounce",
        font_name="Arial Black",
        font_file="ariblk.ttf",
        font_size=62,
        text_color=(255, 255, 255, 255),
        highlight_color=(255, 230, 0, 255),
        action_color=(0, 255, 150, 255),
        stroke_color=(0, 0, 0, 255),
        stroke_width=4,
        margin_bottom=80,
        preview_css="font-family:'Arial Black',Arial,sans-serif;color:#fff;-webkit-text-stroke:3px #000;text-shadow:4px 4px 0 #000",
    ),
    "comic": CaptionStyle(
        name="comic",
        label="Comic",
        font_name="Comic Sans MS",
        font_file="comic.ttf",
        font_size=52,
        text_color=(30, 30, 30, 255),
        highlight_color=(255, 50, 50, 255),
        action_color=(0, 100, 255, 255),
        stroke_color=(0, 0, 0, 0),
        stroke_width=0,
        bg_color=(255, 255, 240, 240),
        bg_padding=(24, 14),
        bg_radius=16,
        margin_bottom=70,
        preview_css="font-family:'Comic Sans MS',cursive,sans-serif;color:#1e1e1e;background:rgba(255,255,240,0.94);padding:6px 14px;border-radius:10px;border:2px solid #222",
    ),
    "subtitle_classic": CaptionStyle(
        name="subtitle_classic",
        label="Classic Subtitle",
        font_name="Arial",
        font_file="arial.ttf",
        font_size=42,
        text_color=(255, 255, 255, 255),
        highlight_color=(255, 255, 100, 255),
        action_color=(255, 150, 80, 255),
        stroke_color=(0, 0, 0, 255),
        stroke_width=2,
        shadow_color=(0, 0, 0, 160),
        shadow_offset=(1, 1),
        margin_bottom=40,
        line_spacing=1.3,
        max_width_pct=0.90,
        preview_css="font-family:Arial,Helvetica,sans-serif;color:#fff;text-shadow:1px 1px 2px rgba(0,0,0,0.8),-1px -1px 2px rgba(0,0,0,0.8);font-size:0.85em",
    ),
    "news_ticker": CaptionStyle(
        name="news_ticker",
        label="News Ticker",
        font_name="Arial Bold",
        font_file="arialbd.ttf",
        font_size=38,
        text_color=(255, 255, 255, 255),
        highlight_color=(255, 220, 0, 255),
        action_color=(255, 80, 80, 255),
        stroke_color=(0, 0, 0, 0),
        stroke_width=0,
        bg_color=(180, 20, 20, 230),
        bg_padding=(28, 10),
        bg_radius=0,
        margin_bottom=30,
        max_width_pct=1.0,
        line_spacing=1.2,
        preview_css="font-family:Arial,Helvetica,sans-serif;font-weight:700;color:#fff;background:rgba(180,20,20,0.9);padding:4px 16px;font-size:0.8em;text-transform:uppercase;letter-spacing:1px",
    ),
    "instagram_stories": CaptionStyle(
        name="instagram_stories",
        label="Instagram Stories",
        font_name="Arial Black",
        font_file="ariblk.ttf",
        font_size=64,
        text_color=(255, 255, 255, 255),
        highlight_color=(225, 48, 108, 255),
        action_color=(252, 175, 69, 255),
        stroke_color=(0, 0, 0, 0),
        stroke_width=0,
        bg_color=(0, 0, 0, 0),
        shadow_color=(0, 0, 0, 200),
        shadow_offset=(0, 3),
        margin_bottom=200,
        max_width_pct=0.85,
        line_spacing=1.2,
        preview_css="font-family:'Arial Black',Arial,sans-serif;color:#fff;text-shadow:0 3px 12px rgba(0,0,0,0.6),0 1px 4px rgba(0,0,0,0.4);font-size:1.1em;text-transform:uppercase;letter-spacing:2px",
    ),
    "netflix": CaptionStyle(
        name="netflix",
        label="Netflix",
        font_name="Arial Bold",
        font_file="arialbd.ttf",
        font_size=48,
        text_color=(255, 255, 255, 255),
        highlight_color=(229, 9, 20, 255),
        action_color=(255, 255, 255, 255),
        stroke_color=(0, 0, 0, 0),
        stroke_width=0,
        bg_color=(0, 0, 0, 0),
        shadow_color=(0, 0, 0, 220),
        shadow_offset=(0, 2),
        margin_bottom=60,
        max_width_pct=0.90,
        line_spacing=1.35,
        preview_css="font-family:Arial,Helvetica,sans-serif;font-weight:700;color:#fff;text-shadow:0 2px 8px rgba(0,0,0,0.8),0 0 2px rgba(0,0,0,0.6);font-size:0.95em;letter-spacing:0.5px",
    ),
    "documentary": CaptionStyle(
        name="documentary",
        label="Documentary",
        font_name="Georgia",
        font_file="georgia.ttf",
        font_size=44,
        text_color=(240, 235, 220, 255),
        highlight_color=(200, 180, 120, 255),
        action_color=(255, 200, 100, 255),
        stroke_color=(0, 0, 0, 0),
        stroke_width=0,
        bg_color=(15, 15, 12, 180),
        bg_padding=(24, 12),
        bg_radius=4,
        shadow_color=(0, 0, 0, 0),
        shadow_offset=(0, 0),
        margin_bottom=50,
        max_width_pct=0.88,
        line_spacing=1.4,
        preview_css="font-family:Georgia,'Times New Roman',serif;color:#f0ebdc;background:rgba(15,15,12,0.7);padding:6px 14px;border-radius:3px;font-size:0.88em;letter-spacing:0.3px;font-style:italic",
    ),
    "sports": CaptionStyle(
        name="sports",
        label="Sports",
        font_name="Impact",
        font_file="impact.ttf",
        font_size=66,
        text_color=(255, 255, 255, 255),
        highlight_color=(0, 200, 83, 255),
        action_color=(255, 50, 50, 255),
        stroke_color=(0, 0, 0, 255),
        stroke_width=5,
        bg_color=(0, 0, 0, 0),
        shadow_color=(0, 0, 0, 180),
        shadow_offset=(3, 3),
        margin_bottom=70,
        max_width_pct=0.80,
        line_spacing=1.15,
        preview_css="font-family:Impact,sans-serif;color:#fff;-webkit-text-stroke:3px #000;text-shadow:3px 3px 0 #000,4px 4px 8px rgba(0,0,0,0.5);font-size:1.2em;text-transform:uppercase;letter-spacing:3px",
    ),
}

DEFAULT_STYLE = "youtube_bold"

# ---------------------------------------------------------------------------
# Action word detection
# ---------------------------------------------------------------------------

_ACTION_KEYWORDS: Set[str] = {
    "amazing", "awesome", "incredible", "insane", "crazy", "unbelievable",
    "massive", "huge", "enormous", "epic", "legendary", "ultimate",
    "absolutely", "completely", "totally", "exactly", "definitely",
    "perfect", "fantastic", "brilliant", "stunning", "gorgeous",
    "crush", "smash", "destroy", "explode", "launch", "blast",
    "dominate", "transform", "revolutionize", "skyrocket",
    "never", "worst", "terrible", "horrible", "disaster",
    "impossible", "ridiculous", "outrageous",
    "everything", "nothing", "always", "million", "billion",
    "free", "secret", "hack", "trick", "game-changer", "breakthrough",
    "important", "critical", "essential", "key", "crucial",
}


def detect_action_words_by_energy(
    filepath: str,
    words: List[Word],
    threshold: float = 1.6,
) -> Set[int]:
    """Detect action words by comparing per-word audio energy to the average."""
    if not words:
        return set()

    try:
        from .audio import extract_audio_pcm
        pcm_bytes, sr = extract_audio_pcm(filepath, sample_rate=16000, mono=True)
    except Exception:
        return set()

    n_samples = len(pcm_bytes) // 2
    if n_samples == 0:
        return set()

    samples = struct.unpack(f"<{n_samples}h", pcm_bytes)

    word_rms = []
    for w in words:
        s_start = max(0, int(w.start * sr))
        s_end = min(n_samples, int(w.end * sr))
        if s_end <= s_start:
            word_rms.append(0.0)
            continue
        chunk = samples[s_start:s_end]
        rms = math.sqrt(sum(s * s for s in chunk) / len(chunk)) / 32768.0
        word_rms.append(rms)

    if not word_rms:
        return set()

    mean_rms = sum(word_rms) / len(word_rms)
    if mean_rms < 0.001:
        return set()

    return {i for i, rms in enumerate(word_rms) if rms > mean_rms * threshold}


def get_action_word_indices(
    all_words: List[Word],
    custom_words: Optional[List[str]] = None,
    use_keywords: bool = True,
    energy_indices: Optional[Set[int]] = None,
) -> Set[int]:
    """Combine keyword list, custom words, and energy analysis for action words."""
    result = set()

    keywords = set()
    if use_keywords:
        keywords.update(_ACTION_KEYWORDS)
    if custom_words:
        for cw in custom_words:
            keywords.add(cw.strip().lower())

    for i, w in enumerate(all_words):
        norm = w.text.strip().lower().strip(".,!?;:\"'()-")
        if norm in keywords:
            result.add(i)

    if energy_indices:
        result.update(energy_indices)

    return result


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

_FONT_DIRS = []
_font_cache: Dict[str, str] = {}


def _get_font_dirs() -> List[str]:
    """Get platform-specific font directories."""
    global _FONT_DIRS
    if _FONT_DIRS:
        return _FONT_DIRS

    dirs = []
    if os.name == "nt":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(os.path.join(windir, "Fonts"))
        localappdata = os.environ.get("LOCALAPPDATA", "")
        if localappdata:
            dirs.append(os.path.join(localappdata, "Microsoft", "Windows", "Fonts"))
    else:
        dirs.extend([
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
            "/Library/Fonts",
            "/System/Library/Fonts",
        ])

    _FONT_DIRS = [d for d in dirs if os.path.isdir(d)]
    return _FONT_DIRS


def find_font(filename: str) -> Optional[str]:
    """Find a font file by filename across system font directories."""
    if filename in _font_cache:
        return _font_cache[filename]

    if os.path.isfile(filename):
        _font_cache[filename] = filename
        return filename

    fn_lower = filename.lower()
    for d in _get_font_dirs():
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower() == fn_lower:
                    path = os.path.join(root, f)
                    _font_cache[filename] = path
                    return path

    return None


def _load_font(style: "CaptionStyle", size_override: Optional[int] = None):
    """Load a PIL ImageFont for the given style."""
    from PIL import ImageFont

    size = size_override or style.font_size

    # Try the style's font file
    font_path = find_font(style.font_file)
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass

    # Try common fallbacks (Windows and Linux names)
    for fallback in [
        "arialbd.ttf", "arial.ttf", "ARIALBD.TTF", "ARIAL.TTF",
        "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
        "LiberationSans-Bold.ttf", "LiberationSans-Regular.ttf",
        "segoeui.ttf", "SEGOEUI.TTF",
    ]:
        fp = find_font(fallback)
        if fp:
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue

    # Last resort: Pillow default (bitmap, small but won't crash)
    logger.warning(
        f"Could not find font '{style.font_file}' or any fallback. "
        f"Using Pillow default font. Searched: {_get_font_dirs()}"
    )
    try:
        return ImageFont.truetype("arial", size)
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Text layout
# ---------------------------------------------------------------------------

@dataclass
class WordLayout:
    """Layout information for a single word."""
    text: str
    x: float
    y: float
    width: float
    line_idx: int
    word_idx: int
    is_highlight: bool
    is_action: bool


def _font_getlength(font, text: str) -> float:
    """Version-safe font width measurement."""
    try:
        return font.getlength(text)
    except AttributeError:
        try:
            return font.getsize(text)[0]
        except AttributeError:
            return len(text) * 10


def _font_metrics(font):
    """Version-safe font metrics."""
    try:
        return font.getmetrics()
    except AttributeError:
        try:
            return (font.getsize("Ay")[1], 4)
        except AttributeError:
            return (20, 4)


def layout_caption_text(
    words: List[str],
    font,
    max_width: int,
    video_width: int,
    y_base: int,
    line_spacing: float,
    highlight_idx: int = -1,
    action_indices: Optional[Set[int]] = None,
    global_offset: int = 0,
) -> List[WordLayout]:
    """Calculate word positions for a caption, with line wrapping and centering."""
    if not words:
        return []

    act = action_indices or set()

    space_width = _font_getlength(font, " ")
    word_widths = [_font_getlength(font, w) for w in words]

    # Wrap into lines
    lines: List[List[int]] = [[]]
    line_widths: List[float] = [0.0]

    for i, (w, ww) in enumerate(zip(words, word_widths)):
        needed = ww + (space_width if lines[-1] else 0)
        if lines[-1] and line_widths[-1] + needed > max_width:
            lines.append([])
            line_widths.append(0.0)
        if lines[-1]:
            line_widths[-1] += space_width
        lines[-1].append(i)
        line_widths[-1] += ww

    ascent, descent = _font_metrics(font)
    line_height = int((ascent + descent) * line_spacing)
    total_height = line_height * len(lines)

    results = []
    for line_num, (line_indices, lw) in enumerate(zip(lines, line_widths)):
        y = y_base - total_height + (line_num * line_height)
        x_start = (video_width - lw) / 2
        x = x_start

        for local_idx, word_idx in enumerate(line_indices):
            global_idx = global_offset + word_idx
            results.append(WordLayout(
                text=words[word_idx],
                x=x,
                y=y,
                width=word_widths[word_idx],
                line_idx=line_num,
                word_idx=global_idx,
                is_highlight=(word_idx == highlight_idx),
                is_action=(global_idx in act),
            ))
            x += word_widths[word_idx] + space_width

    return results


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_frame(
    layouts: List[WordLayout],
    style: CaptionStyle,
    video_size: Tuple[int, int],
    font,
) -> "Image":
    """Render a single caption frame as a transparent RGBA image."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", video_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if not layouts:
        return img

    # Draw background box if style has one
    if style.bg_color[3] > 0:
        min_x = min(wl.x for wl in layouts) - style.bg_padding[0]
        max_x = max(wl.x + wl.width for wl in layouts) + style.bg_padding[0]
        min_y = min(wl.y for wl in layouts) - style.bg_padding[1]
        ascent, descent = _font_metrics(font)
        max_y = max(wl.y for wl in layouts) + ascent + descent + style.bg_padding[1]

        try:
            draw.rounded_rectangle(
                [(int(min_x), int(min_y)), (int(max_x), int(max_y))],
                radius=style.bg_radius,
                fill=style.bg_color,
            )
        except (AttributeError, TypeError):
            draw.rectangle(
                [(int(min_x), int(min_y)), (int(max_x), int(max_y))],
                fill=style.bg_color,
            )

    for wl in layouts:
        if wl.is_action:
            color = style.action_color
        elif wl.is_highlight:
            color = style.highlight_color
        else:
            color = style.text_color

        x, y = int(wl.x), int(wl.y)

        # Shadow pass
        if style.shadow_color[3] > 0:
            sx = x + style.shadow_offset[0]
            sy = y + style.shadow_offset[1]
            if style.shadow_offset == (0, 0):
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        draw.text((sx + dx, sy + dy), wl.text,
                                  font=font, fill=style.shadow_color)
            else:
                draw.text((sx, sy), wl.text, font=font,
                          fill=style.shadow_color)

        # Main text with stroke
        if style.stroke_width > 0:
            try:
                draw.text(
                    (x, y), wl.text, font=font, fill=color,
                    stroke_width=style.stroke_width,
                    stroke_fill=style.stroke_color,
                )
            except TypeError:
                # Pillow < 8.0: stroke_width not supported, fake it
                sw = style.stroke_width
                for dx in range(-sw, sw + 1):
                    for dy in range(-sw, sw + 1):
                        if dx != 0 or dy != 0:
                            draw.text((x + dx, y + dy), wl.text,
                                      font=font, fill=style.stroke_color)
                draw.text((x, y), wl.text, font=font, fill=color)
        else:
            draw.text((x, y), wl.text, font=font, fill=color)

    return img


# ---------------------------------------------------------------------------
# Pillow bootstrap
# ---------------------------------------------------------------------------

def _ensure_pillow():
    """Make sure Pillow is importable, installing if needed."""
    try:
        from PIL import Image  # noqa: F401
        return True
    except ImportError:
        pass

    import sys
    logger.info("Pillow not found, installing...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "Pillow", "--quiet"],
            timeout=120,
        )
    except Exception as e:
        logger.error(f"Failed to install Pillow: {e}")
        raise RuntimeError(
            "Pillow is required for styled captions. "
            "Install it manually: pip install Pillow"
        ) from e

    # CRITICAL: invalidate Python's import cache so it finds the new package
    import importlib
    importlib.invalidate_caches()

    try:
        from PIL import Image  # noqa: F401
        logger.info("Pillow installed and imported successfully")
        return True
    except ImportError as e:
        raise RuntimeError(
            "Pillow was installed but cannot be imported. "
            "Restart the OpenCut server and try again."
        ) from e


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def render_styled_caption_video(
    transcription: TranscriptionResult,
    output_path: str,
    style_name: str = DEFAULT_STYLE,
    video_width: int = 1920,
    video_height: int = 1080,
    fps: float = 30.0,
    action_indices: Optional[Set[int]] = None,
    total_duration: Optional[float] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> dict:
    """
    Render a transparent video overlay with styled, word-highlighted captions.

    Uses FFmpeg rawvideo pipe for reliability (no temp files, no concat quirks).
    Output is PNG codec in MOV container for universal alpha support.
    """
    _ensure_pillow()
    from PIL import Image

    style = STYLES.get(style_name, STYLES[DEFAULT_STYLE])
    font = _load_font(style)
    video_size = (video_width, video_height)
    max_text_width = int(video_width * style.max_width_pct)
    y_base = video_height - style.margin_bottom
    act = action_indices or set()

    # Flatten all words with global indices
    all_words_flat = []
    seg_word_map = []  # (seg_start_global, seg_end_global)
    for seg in transcription.segments:
        start_idx = len(all_words_flat)
        for w in seg.words:
            all_words_flat.append(w)
        seg_word_map.append((start_idx, len(all_words_flat)))

    # Determine total duration
    if total_duration is None:
        if transcription.segments:
            total_duration = transcription.segments[-1].end + 1.0
        else:
            total_duration = 1.0

    total_frames = max(1, int(total_duration * fps))

    # -------------------------------------------------------------------
    # Pre-build timeline of visual states:
    #   (start_time, end_time, seg_idx, highlight_local_idx)
    #   seg_idx = -1 means blank (no caption visible)
    #   highlight_local_idx = -1 means show segment, no word highlighted
    # -------------------------------------------------------------------
    states = []

    prev_end = 0.0
    for seg_idx, seg in enumerate(transcription.segments):
        # Blank gap before segment
        if seg.start > prev_end + 0.001:
            states.append((prev_end, seg.start, -1, -1))

        if not seg.words:
            states.append((seg.start, seg.end, seg_idx, -1))
        else:
            seg_time = seg.start
            for w_local, word_obj in enumerate(seg.words):
                if word_obj.start > seg_time + 0.001:
                    states.append((seg_time, word_obj.start, seg_idx, -1))
                w_end = max(word_obj.end, word_obj.start + 0.01)
                states.append((word_obj.start, w_end, seg_idx, w_local))
                seg_time = w_end
            if seg.end > seg_time + 0.001:
                states.append((seg_time, seg.end, seg_idx, -1))

        prev_end = seg.end

    if total_duration > prev_end + 0.001:
        states.append((prev_end, total_duration, -1, -1))

    # -------------------------------------------------------------------
    # Pre-render each unique visual state to raw RGBA bytes
    # -------------------------------------------------------------------
    blank_bytes = bytes(video_width * video_height * 4)  # all zeros = transparent

    rendered_states = []
    for si, (t0, t1, seg_idx, hl_local) in enumerate(states):
        if seg_idx < 0:
            rendered_states.append(blank_bytes)
            continue

        seg = transcription.segments[seg_idx]
        seg_start_global = seg_word_map[seg_idx][0]
        words_text = ([w.text for w in seg.words]
                      if seg.words else seg.text.strip().split())

        layouts = layout_caption_text(
            words_text, font, max_text_width, video_width, y_base,
            style.line_spacing, highlight_idx=hl_local,
            action_indices=act, global_offset=seg_start_global,
        )
        frame = render_frame(layouts, style, video_size, font)
        rendered_states.append(frame.tobytes("raw", "RGBA"))

    logger.info(
        f"Styled captions: {len(states)} visual states pre-rendered, "
        f"{total_frames} frames to encode at {fps}fps"
    )
    if on_progress:
        on_progress(65, f"Encoding {total_frames} frames...")

    # -------------------------------------------------------------------
    # Pipe raw RGBA frames to FFmpeg
    # -------------------------------------------------------------------
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{video_width}x{video_height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "png",
        "-pix_fmt", "rgba",
        output_path,
    ]

    logger.info(f"FFmpeg command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    state_idx = 0
    frames_written = 0
    last_pct = -1

    try:
        for frame_num in range(total_frames):
            t = frame_num / fps

            # Advance state pointer
            while (state_idx < len(states) - 1
                   and t >= states[state_idx][1] - 0.0001):
                state_idx += 1

            proc.stdin.write(rendered_states[state_idx])
            frames_written += 1

            # Progress callback every 5%
            if on_progress and total_frames > 0:
                pct = int(frame_num * 100 / total_frames)
                if pct != last_pct and pct % 5 == 0:
                    last_pct = pct
                    on_progress(
                        65 + int(pct * 0.30),
                        f"Encoding frame {frame_num}/{total_frames}",
                    )

    except BrokenPipeError:
        logger.warning("FFmpeg pipe closed early")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass

    stderr_out = b""
    try:
        stderr_out = proc.stderr.read()
    except Exception:
        pass
    proc.wait()

    if proc.returncode != 0:
        err_msg = stderr_out.decode("utf-8", errors="replace").strip()
        logger.error(f"FFmpeg failed (rc={proc.returncode}): {err_msg}")
        raise RuntimeError(f"FFmpeg encoding failed: {err_msg or 'unknown error'}")

    if not os.path.isfile(output_path) or os.path.getsize(output_path) < 100:
        raise RuntimeError(
            f"Output file missing or empty: {output_path}. "
            f"FFmpeg stderr: {stderr_out.decode('utf-8', errors='replace')}"
        )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(
        f"Styled caption overlay: {output_path} "
        f"({file_size_mb:.1f}MB, {frames_written} frames, {total_duration:.1f}s)"
    )

    return {
        "output_path": output_path,
        "frames_rendered": frames_written,
        "duration": total_duration,
        "style": style_name,
    }


def get_style_info() -> List[dict]:
    """Get serializable info for all styles (for the panel preview system)."""
    result = []
    for key, s in STYLES.items():
        result.append({
            "name": s.name,
            "label": s.label,
            "preview_css": s.preview_css,
            "highlight_color": f"rgb({s.highlight_color[0]},{s.highlight_color[1]},{s.highlight_color[2]})",
            "action_color": f"rgb({s.action_color[0]},{s.action_color[1]},{s.action_color[2]})",
        })
    return result
