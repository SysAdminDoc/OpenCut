"""
OpenCut Animated Caption Style Library

50+ built-in caption styles with word-by-word animation support.
Renders via FFmpeg drawtext filters with per-word timing.
Inspired by CapCut and Captions.ai animated caption systems.
"""

import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Set

from opencut.helpers import escape_drawtext, escape_filter_path, get_ffmpeg_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class CaptionStyle:
    """Definition of a single caption style."""
    id: str = ""
    name: str = ""
    category: str = ""
    preview_description: str = ""
    animation_type: str = "none"
    colors: Dict = field(default_factory=dict)
    font_family: str = "Arial"
    font_size: int = 48
    position: str = "bottom"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["font_resolution"] = resolve_caption_font(self.font_family).to_dict()
        return data


# ---------------------------------------------------------------------------
# Built-in Styles (50+)
# ---------------------------------------------------------------------------
BUILTIN_STYLES: Dict[str, CaptionStyle] = {}

_STYLE_DEFS: List[Dict] = [
    # --- minimal ---
    {"id": "minimal_clean", "name": "Minimal Clean", "category": "minimal",
     "preview_description": "Simple white text, no frills",
     "animation_type": "fade", "font_family": "Arial", "font_size": 42,
     "colors": {"text": "#FFFFFF", "shadow": "#00000080"}, "position": "bottom"},
    {"id": "minimal_lowercase", "name": "Minimal Lowercase", "category": "minimal",
     "preview_description": "Lowercase white text, subtle fade",
     "animation_type": "fade", "font_family": "Helvetica", "font_size": 38,
     "colors": {"text": "#E0E0E0", "shadow": "#00000060"}, "position": "bottom"},
    {"id": "minimal_center", "name": "Minimal Center", "category": "minimal",
     "preview_description": "Centered minimal white text",
     "animation_type": "fade", "font_family": "Arial", "font_size": 44,
     "colors": {"text": "#FFFFFF", "shadow": "#00000070"}, "position": "center"},
    {"id": "minimal_thin", "name": "Minimal Thin", "category": "minimal",
     "preview_description": "Thin weight clean text",
     "animation_type": "fade", "font_family": "Arial", "font_size": 40,
     "colors": {"text": "#F5F5F5", "shadow": "#00000050"}, "position": "bottom"},
    {"id": "minimal_top", "name": "Minimal Top", "category": "minimal",
     "preview_description": "Clean text at top of frame",
     "animation_type": "fade", "font_family": "Arial", "font_size": 40,
     "colors": {"text": "#FFFFFF", "shadow": "#00000060"}, "position": "top"},

    # --- bold ---
    {"id": "tiktok_bold", "name": "TikTok Bold", "category": "bold",
     "preview_description": "Bold white text with strong shadow, TikTok-style",
     "animation_type": "bounce", "font_family": "Impact", "font_size": 56,
     "colors": {"text": "#FFFFFF", "shadow": "#000000", "highlight": "#FFE500"},
     "position": "center"},
    {"id": "youtube_clean", "name": "YouTube Clean", "category": "bold",
     "preview_description": "Clean bold white text for YouTube",
     "animation_type": "fade", "font_family": "Arial", "font_size": 52,
     "colors": {"text": "#FFFFFF", "shadow": "#000000CC", "highlight": "#FF0000"},
     "position": "bottom"},
    {"id": "bold_impact", "name": "Bold Impact", "category": "bold",
     "preview_description": "Heavy impact text with drop shadow",
     "animation_type": "bounce", "font_family": "Impact", "font_size": 60,
     "colors": {"text": "#FFFFFF", "shadow": "#000000"}, "position": "center"},
    {"id": "bold_uppercase", "name": "Bold Uppercase", "category": "bold",
     "preview_description": "All caps bold text",
     "animation_type": "slide", "font_family": "Arial Black", "font_size": 54,
     "colors": {"text": "#FFFFFF", "shadow": "#000000"}, "position": "center"},
    {"id": "bold_red", "name": "Bold Red Accent", "category": "bold",
     "preview_description": "Bold white text with red highlight word",
     "animation_type": "bounce", "font_family": "Impact", "font_size": 56,
     "colors": {"text": "#FFFFFF", "shadow": "#000000", "highlight": "#FF3333"},
     "position": "center"},

    # --- karaoke ---
    {"id": "karaoke_highlight", "name": "Karaoke Highlight", "category": "karaoke",
     "preview_description": "Words light up as they are spoken, karaoke-style",
     "animation_type": "karaoke", "font_family": "Arial", "font_size": 50,
     "colors": {"text": "#888888", "highlight": "#FFFF00", "shadow": "#000000"},
     "position": "bottom"},
    {"id": "karaoke_blue", "name": "Karaoke Blue", "category": "karaoke",
     "preview_description": "Blue highlight sweep across words",
     "animation_type": "karaoke", "font_family": "Arial", "font_size": 50,
     "colors": {"text": "#666666", "highlight": "#00BFFF", "shadow": "#000000"},
     "position": "bottom"},
    {"id": "karaoke_green", "name": "Karaoke Green", "category": "karaoke",
     "preview_description": "Green highlight sweep",
     "animation_type": "karaoke", "font_family": "Arial", "font_size": 50,
     "colors": {"text": "#777777", "highlight": "#00FF88", "shadow": "#000000"},
     "position": "center"},
    {"id": "karaoke_pink", "name": "Karaoke Pink", "category": "karaoke",
     "preview_description": "Pink highlight karaoke style",
     "animation_type": "karaoke", "font_family": "Arial", "font_size": 48,
     "colors": {"text": "#777777", "highlight": "#FF69B4", "shadow": "#000000"},
     "position": "bottom"},
    {"id": "karaoke_bold", "name": "Karaoke Bold", "category": "karaoke",
     "preview_description": "Bold karaoke with yellow highlight",
     "animation_type": "karaoke", "font_family": "Impact", "font_size": 54,
     "colors": {"text": "#555555", "highlight": "#FFD700", "shadow": "#000000"},
     "position": "center"},

    # --- typewriter ---
    {"id": "typewriter_classic", "name": "Typewriter Classic", "category": "typewriter",
     "preview_description": "Words appear letter-by-letter like a typewriter",
     "animation_type": "typewriter", "font_family": "Courier New", "font_size": 44,
     "colors": {"text": "#FFFFFF", "shadow": "#00000080"}, "position": "bottom"},
    {"id": "typewriter_green", "name": "Typewriter Terminal", "category": "typewriter",
     "preview_description": "Green terminal text typing effect",
     "animation_type": "typewriter", "font_family": "Courier New", "font_size": 42,
     "colors": {"text": "#00FF00", "shadow": "#003300"}, "position": "bottom"},
    {"id": "typewriter_amber", "name": "Typewriter Amber", "category": "typewriter",
     "preview_description": "Amber monospace typing",
     "animation_type": "typewriter", "font_family": "Courier New", "font_size": 42,
     "colors": {"text": "#FFBF00", "shadow": "#33200080"}, "position": "bottom"},
    {"id": "typewriter_modern", "name": "Typewriter Modern", "category": "typewriter",
     "preview_description": "Modern sans-serif typing effect",
     "animation_type": "typewriter", "font_family": "Arial", "font_size": 44,
     "colors": {"text": "#FFFFFF", "shadow": "#00000070"}, "position": "center"},
    {"id": "typewriter_retro", "name": "Typewriter Retro", "category": "typewriter",
     "preview_description": "Retro sepia toned typing",
     "animation_type": "typewriter", "font_family": "Courier New", "font_size": 40,
     "colors": {"text": "#D2B48C", "shadow": "#3E250080"}, "position": "bottom"},

    # --- bounce ---
    {"id": "bounce_pop", "name": "Bounce Pop", "category": "bounce",
     "preview_description": "Words pop in with bouncy spring animation",
     "animation_type": "bounce", "font_family": "Arial Black", "font_size": 52,
     "colors": {"text": "#FFFFFF", "shadow": "#000000", "highlight": "#FF6B6B"},
     "position": "center"},
    {"id": "bounce_playful", "name": "Bounce Playful", "category": "bounce",
     "preview_description": "Playful bouncing colored words",
     "animation_type": "bounce", "font_family": "Comic Sans MS", "font_size": 48,
     "colors": {"text": "#FFD93D", "shadow": "#00000080"}, "position": "center"},
    {"id": "bounce_drop", "name": "Bounce Drop", "category": "bounce",
     "preview_description": "Words drop in from above with bounce",
     "animation_type": "bounce", "font_family": "Arial", "font_size": 50,
     "colors": {"text": "#FFFFFF", "shadow": "#000000"}, "position": "center"},
    {"id": "bounce_elastic", "name": "Bounce Elastic", "category": "bounce",
     "preview_description": "Elastic spring bounce effect",
     "animation_type": "bounce", "font_family": "Impact", "font_size": 54,
     "colors": {"text": "#FFFFFF", "shadow": "#000000", "highlight": "#00FF88"},
     "position": "center"},
    {"id": "bounce_scale", "name": "Bounce Scale", "category": "bounce",
     "preview_description": "Words scale up with bounce easing",
     "animation_type": "bounce", "font_family": "Arial Black", "font_size": 50,
     "colors": {"text": "#FFFFFF", "shadow": "#000000CC"}, "position": "center"},

    # --- wave ---
    {"id": "gradient_wave", "name": "Gradient Wave", "category": "wave",
     "preview_description": "Text with gradient color wave animation",
     "animation_type": "wave", "font_family": "Arial", "font_size": 48,
     "colors": {"text": "#FF6B6B", "highlight": "#4ECDC4", "shadow": "#000000"},
     "position": "center"},
    {"id": "wave_rainbow", "name": "Wave Rainbow", "category": "wave",
     "preview_description": "Rainbow color wave across words",
     "animation_type": "wave", "font_family": "Arial", "font_size": 48,
     "colors": {"text": "#FF0000", "highlight": "#00FF00", "shadow": "#000000"},
     "position": "center"},
    {"id": "wave_ocean", "name": "Wave Ocean", "category": "wave",
     "preview_description": "Blue-teal wave motion",
     "animation_type": "wave", "font_family": "Arial", "font_size": 46,
     "colors": {"text": "#0077B6", "highlight": "#00B4D8", "shadow": "#000000"},
     "position": "bottom"},
    {"id": "wave_sunset", "name": "Wave Sunset", "category": "wave",
     "preview_description": "Orange-pink sunset wave",
     "animation_type": "wave", "font_family": "Arial", "font_size": 46,
     "colors": {"text": "#FF6B35", "highlight": "#FF006E", "shadow": "#000000"},
     "position": "center"},
    {"id": "wave_pulse", "name": "Wave Pulse", "category": "wave",
     "preview_description": "Pulsing wave effect on text",
     "animation_type": "wave", "font_family": "Arial", "font_size": 48,
     "colors": {"text": "#FFFFFF", "highlight": "#FF00FF", "shadow": "#000000"},
     "position": "center"},

    # --- glow ---
    {"id": "neon_glow", "name": "Neon Glow", "category": "glow",
     "preview_description": "Neon-lit glowing text effect",
     "animation_type": "glow", "font_family": "Arial", "font_size": 50,
     "colors": {"text": "#FF00FF", "glow": "#FF00FF44", "shadow": "#000000"},
     "position": "center"},
    {"id": "glow_blue", "name": "Blue Glow", "category": "glow",
     "preview_description": "Electric blue glow effect",
     "animation_type": "glow", "font_family": "Arial", "font_size": 48,
     "colors": {"text": "#00BFFF", "glow": "#00BFFF44", "shadow": "#000000"},
     "position": "center"},
    {"id": "glow_green", "name": "Green Glow", "category": "glow",
     "preview_description": "Matrix-style green glow",
     "animation_type": "glow", "font_family": "Courier New", "font_size": 46,
     "colors": {"text": "#00FF00", "glow": "#00FF0044", "shadow": "#000000"},
     "position": "center"},
    {"id": "glow_gold", "name": "Gold Glow", "category": "glow",
     "preview_description": "Warm gold glowing text",
     "animation_type": "glow", "font_family": "Arial", "font_size": 50,
     "colors": {"text": "#FFD700", "glow": "#FFD70044", "shadow": "#000000"},
     "position": "center"},
    {"id": "glow_white", "name": "White Glow", "category": "glow",
     "preview_description": "Soft white glow halo",
     "animation_type": "glow", "font_family": "Arial", "font_size": 48,
     "colors": {"text": "#FFFFFF", "glow": "#FFFFFF44", "shadow": "#00000080"},
     "position": "center"},

    # --- gradient ---
    {"id": "gradient_fire", "name": "Gradient Fire", "category": "gradient",
     "preview_description": "Red-to-yellow fire gradient text",
     "animation_type": "fade", "font_family": "Impact", "font_size": 54,
     "colors": {"text": "#FF4500", "highlight": "#FFD700", "shadow": "#000000"},
     "position": "center"},
    {"id": "gradient_ice", "name": "Gradient Ice", "category": "gradient",
     "preview_description": "White-to-blue icy gradient",
     "animation_type": "fade", "font_family": "Arial", "font_size": 48,
     "colors": {"text": "#E0FFFF", "highlight": "#4169E1", "shadow": "#000000"},
     "position": "center"},
    {"id": "gradient_purple", "name": "Gradient Purple", "category": "gradient",
     "preview_description": "Purple-to-pink gradient",
     "animation_type": "fade", "font_family": "Arial", "font_size": 48,
     "colors": {"text": "#9B59B6", "highlight": "#E91E63", "shadow": "#000000"},
     "position": "center"},
    {"id": "gradient_mint", "name": "Gradient Mint", "category": "gradient",
     "preview_description": "Mint green gradient text",
     "animation_type": "fade", "font_family": "Arial", "font_size": 46,
     "colors": {"text": "#98FB98", "highlight": "#20B2AA", "shadow": "#000000"},
     "position": "bottom"},
    {"id": "gradient_coral", "name": "Gradient Coral", "category": "gradient",
     "preview_description": "Coral-to-peach warm gradient",
     "animation_type": "fade", "font_family": "Arial", "font_size": 48,
     "colors": {"text": "#FF7F50", "highlight": "#FFDAB9", "shadow": "#000000"},
     "position": "center"},

    # --- box ---
    {"id": "box_black", "name": "Box Black", "category": "box",
     "preview_description": "White text on black background box",
     "animation_type": "fade", "font_family": "Arial", "font_size": 44,
     "colors": {"text": "#FFFFFF", "background": "#000000CC"}, "position": "bottom"},
    {"id": "box_yellow", "name": "Box Yellow", "category": "box",
     "preview_description": "Black text on yellow highlight box",
     "animation_type": "fade", "font_family": "Arial Black", "font_size": 46,
     "colors": {"text": "#000000", "background": "#FFE500CC"}, "position": "bottom"},
    {"id": "box_red", "name": "Box Red", "category": "box",
     "preview_description": "White text on red box background",
     "animation_type": "slide", "font_family": "Arial", "font_size": 44,
     "colors": {"text": "#FFFFFF", "background": "#FF0000CC"}, "position": "bottom"},
    {"id": "box_blue", "name": "Box Blue", "category": "box",
     "preview_description": "White text on blue box",
     "animation_type": "fade", "font_family": "Arial", "font_size": 44,
     "colors": {"text": "#FFFFFF", "background": "#1E90FFCC"}, "position": "bottom"},
    {"id": "box_green", "name": "Box Green", "category": "box",
     "preview_description": "White text on green box",
     "animation_type": "fade", "font_family": "Arial", "font_size": 44,
     "colors": {"text": "#FFFFFF", "background": "#2ECC71CC"}, "position": "bottom"},

    # --- outline ---
    {"id": "outline_white", "name": "Outline White", "category": "outline",
     "preview_description": "White text with black outline stroke",
     "animation_type": "fade", "font_family": "Arial Black", "font_size": 52,
     "colors": {"text": "#FFFFFF", "outline": "#000000"}, "position": "center"},
    {"id": "outline_neon", "name": "Outline Neon", "category": "outline",
     "preview_description": "Neon outlined text with glow",
     "animation_type": "fade", "font_family": "Arial", "font_size": 50,
     "colors": {"text": "#000000", "outline": "#FF00FF"}, "position": "center"},
    {"id": "outline_yellow", "name": "Outline Yellow", "category": "outline",
     "preview_description": "Yellow outlined bold text",
     "animation_type": "bounce", "font_family": "Impact", "font_size": 52,
     "colors": {"text": "#000000", "outline": "#FFD700"}, "position": "center"},
    {"id": "outline_red", "name": "Outline Red", "category": "outline",
     "preview_description": "Red outlined text",
     "animation_type": "fade", "font_family": "Arial Black", "font_size": 50,
     "colors": {"text": "#FFFFFF", "outline": "#FF0000"}, "position": "center"},
    {"id": "outline_blue", "name": "Outline Blue", "category": "outline",
     "preview_description": "Blue outlined text",
     "animation_type": "fade", "font_family": "Arial", "font_size": 50,
     "colors": {"text": "#FFFFFF", "outline": "#0066FF"}, "position": "center"},

    # --- specialty ---
    {"id": "instagram_story", "name": "Instagram Story", "category": "bold",
     "preview_description": "Instagram Stories-style bold text",
     "animation_type": "slide", "font_family": "Arial", "font_size": 56,
     "colors": {"text": "#FFFFFF", "shadow": "#000000", "highlight": "#E1306C"},
     "position": "center"},
    {"id": "news_ticker", "name": "News Ticker", "category": "box",
     "preview_description": "Lower-third news ticker style",
     "animation_type": "slide", "font_family": "Arial", "font_size": 36,
     "colors": {"text": "#FFFFFF", "background": "#CC0000DD"}, "position": "bottom"},
    {"id": "cinematic_subtitle", "name": "Cinematic Subtitle", "category": "minimal",
     "preview_description": "Elegant cinematic subtitles",
     "animation_type": "fade", "font_family": "Georgia", "font_size": 40,
     "colors": {"text": "#FFFFFFDD", "shadow": "#00000088"}, "position": "bottom"},
    {"id": "podcast_caption", "name": "Podcast Caption", "category": "box",
     "preview_description": "Podcast-style caption with background",
     "animation_type": "fade", "font_family": "Arial", "font_size": 42,
     "colors": {"text": "#FFFFFF", "background": "#1A1A2EDD"}, "position": "bottom"},
    {"id": "gaming_hud", "name": "Gaming HUD", "category": "glow",
     "preview_description": "Gaming-style HUD text with glow",
     "animation_type": "glow", "font_family": "Impact", "font_size": 48,
     "colors": {"text": "#00FF88", "glow": "#00FF8844", "shadow": "#000000"},
     "position": "top"},
]

# Populate registry
for _def in _STYLE_DEFS:
    _s = CaptionStyle(
        id=_def["id"],
        name=_def["name"],
        category=_def["category"],
        preview_description=_def["preview_description"],
        animation_type=_def["animation_type"],
        colors=_def.get("colors", {}),
        font_family=_def.get("font_family", "Arial"),
        font_size=_def.get("font_size", 48),
        position=_def.get("position", "bottom"),
    )
    BUILTIN_STYLES[_s.id] = _s

CATEGORIES = [
    "minimal", "bold", "karaoke", "typewriter", "bounce",
    "wave", "glow", "gradient", "box", "outline",
]

_FONT_FAMILY_FILES: Dict[str, List[str]] = {
    "arial": ["arial.ttf", "Arial.ttf", "ARIAL.TTF", "segoeui.ttf", "DejaVuSans.ttf"],
    "arial black": ["ariblk.ttf", "Arial Black.ttf", "ARIBLK.TTF", "ArialBlack.ttf"],
    "comic sans ms": ["comic.ttf", "Comic Sans MS.ttf", "comicbd.ttf"],
    "consolas": ["consola.ttf", "Consolas.ttf", "DejaVuSansMono.ttf"],
    "courier new": ["cour.ttf", "Courier New.ttf", "courbd.ttf", "DejaVuSansMono.ttf"],
    "georgia": ["georgia.ttf", "Georgia.ttf", "DejaVuSerif.ttf"],
    "helvetica": ["Helvetica.ttf", "helvetica.ttf", "Arial.ttf", "arial.ttf"],
    "impact": ["impact.ttf", "Impact.ttf", "IMPACT.TTF"],
    "segoe script": ["segoesc.ttf", "Segoe Script.ttf"],
}

_COMMON_FONT_FALLBACKS = [
    "arialbd.ttf",
    "arial.ttf",
    "ARIALBD.TTF",
    "ARIAL.TTF",
    "DejaVuSans-Bold.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Bold.ttf",
    "LiberationSans-Regular.ttf",
    "segoeui.ttf",
    "SEGOEUI.TTF",
]

_CJK_FALLBACKS = [
    "NotoSansCJK-Regular.ttc",
    "NotoSansCJKsc-Regular.otf",
    "NotoSansJP-Regular.otf",
    "NotoSansSC-Regular.otf",
    "msgothic.ttc",
    "SimSun.ttf",
    "SimHei.ttf",
    "malgun.ttf",
    "meiryo.ttc",
    "YuGothR.ttc",
]

_INDIC_FALLBACKS = [
    "NotoSansDevanagari-Regular.ttf",
    "NotoSansBengali-Regular.ttf",
    "NotoSans-Regular.ttf",
    "mangal.ttf",
    "Mangal.ttf",
    "vrinda.ttf",
    "Vrinda.ttf",
    "nirmala.ttf",
    "Nirmala.ttf",
]

_RTL_FALLBACKS = [
    "NotoSansArabic-Regular.ttf",
    "NotoNaskhArabic-Regular.ttf",
    "NotoSansHebrew-Regular.ttf",
    "tahoma.ttf",
    "Tahoma.ttf",
]


@dataclass(frozen=True)
class CaptionFontResolution:
    """Resolved font information for FFmpeg drawtext captions."""

    requested_family: str
    font_family: str
    font_path: Optional[str]
    source: str
    scripts: List[str] = field(default_factory=list)
    warning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _find_font(filename: str) -> Optional[str]:
    try:
        from opencut.core.styled_captions import find_font

        return find_font(filename)
    except Exception:  # noqa: BLE001 - font discovery must degrade to family fallback.
        return filename if os.path.isfile(filename) else None


def _detect_scripts(text: str) -> Set[str]:
    try:
        from opencut.core.styled_captions import _detect_text_scripts

        return set(_detect_text_scripts(text))
    except Exception:  # noqa: BLE001 - keep caption filters buildable without renderer extras.
        scripts: Set[str] = set()
        for ch in text or "":
            code = ord(ch)
            if 0x0590 <= code <= 0x05FF:
                scripts.add("hebrew")
            elif 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
                scripts.add("arabic")
            elif 0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF:
                scripts.add("cjk")
            elif 0xAC00 <= code <= 0xD7AF:
                scripts.add("cjk")
            elif 0x0900 <= code <= 0x0D7F:
                scripts.add("indic")
        return scripts


def _unique(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _safe_font_family(font_family: str) -> str:
    cleaned = "".join(ch for ch in str(font_family or "Arial") if ch.isalnum() or ch in " _-")
    cleaned = " ".join(cleaned.split())
    return cleaned or "Arial"


def _script_font_candidates(scripts: Set[str]) -> List[str]:
    candidates: List[str] = []
    if "cjk" in scripts:
        candidates.extend(_CJK_FALLBACKS)
    if "indic" in scripts:
        candidates.extend(_INDIC_FALLBACKS)
    if "arabic" in scripts or "hebrew" in scripts:
        candidates.extend(_RTL_FALLBACKS)
    return candidates


@lru_cache(maxsize=256)
def resolve_caption_font(font_family: str, text_hint: str = "") -> CaptionFontResolution:
    """Resolve a caption font to a real file or safe FFmpeg family fallback."""
    requested = _safe_font_family(font_family)
    scripts = _detect_scripts(text_hint)
    script_candidates = _script_font_candidates(scripts)
    family_candidates = _FONT_FAMILY_FILES.get(requested.lower(), [])
    if "." in requested:
        family_candidates = [requested, *family_candidates]
    candidates = _unique([*script_candidates, *family_candidates, *_COMMON_FONT_FALLBACKS])

    for candidate in candidates:
        path = _find_font(candidate)
        if not path:
            continue
        if candidate in script_candidates:
            return CaptionFontResolution(
                requested_family=requested,
                font_family=requested,
                font_path=path,
                source="script_fallback_file",
                scripts=sorted(scripts),
                warning=f"Using script-aware font file for {', '.join(sorted(scripts)) or 'caption'} text.",
            )
        if candidate in family_candidates:
            return CaptionFontResolution(
                requested_family=requested,
                font_family=requested,
                font_path=path,
                source="preferred_file",
                scripts=sorted(scripts),
            )
        return CaptionFontResolution(
            requested_family=requested,
            font_family=requested,
            font_path=path,
            source="fallback_file",
            scripts=sorted(scripts),
            warning=f"Preferred font '{requested}' was not found; using a bundled/system fallback file.",
        )

    warning = f"No font file found for '{requested}'. FFmpeg will use the font family fallback."
    if "cjk" in scripts:
        warning = "No CJK font file found; install Noto Sans CJK for reliable glyph rendering."
    return CaptionFontResolution(
        requested_family=requested,
        font_family=requested,
        font_path=None,
        source="family_fallback",
        scripts=sorted(scripts),
        warning=warning,
    )


def _drawtext_font_option(resolution: CaptionFontResolution) -> str:
    if resolution.font_path:
        return f"fontfile='{escape_filter_path(resolution.font_path)}'"
    return f"font='{_escape_drawtext(resolution.font_family)}'"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_available_styles() -> List[CaptionStyle]:
    """Return all built-in caption styles."""
    return list(BUILTIN_STYLES.values())


def get_style(style_id: str) -> CaptionStyle:
    """Return a style by ID. Raises ``ValueError`` if not found."""
    if style_id not in BUILTIN_STYLES:
        raise ValueError(
            f"Unknown caption style: {style_id}. "
            f"Available: {', '.join(sorted(BUILTIN_STYLES))}"
        )
    return BUILTIN_STYLES[style_id]


def _position_y(position: str, height: int, font_size: int) -> int:
    """Compute Y coordinate for text placement."""
    if position == "top":
        return int(height * 0.08)
    if position == "center":
        return int((height - font_size) / 2)
    # bottom
    return int(height * 0.82)


def _escape_drawtext(text: str) -> str:
    """Escape text for FFmpeg drawtext filter."""
    return escape_drawtext(text)


def _build_drawtext_filter(
    style: CaptionStyle,
    text: str,
    start: float,
    end: float,
    width: int = 1920,
    height: int = 1080,
) -> str:
    """Build an FFmpeg drawtext filter string for a single caption segment."""
    escaped = _escape_drawtext(text)
    y = _position_y(style.position, height, style.font_size)
    text_color = style.colors.get("text", "#FFFFFF")
    shadow_color = style.colors.get("shadow", "#000000")
    bg_color = style.colors.get("background", "")
    font_resolution = resolve_caption_font(style.font_family, text)

    parts = [
        f"drawtext=expansion=none:text='{escaped}'",
        _drawtext_font_option(font_resolution),
        f"fontsize={style.font_size}",
        f"fontcolor={text_color}",
        f"shadowcolor={shadow_color}",
        "shadowx=2", "shadowy=2",
        "x=(w-text_w)/2",
        f"y={y}",
        f"enable='between(t,{start},{end})'",
    ]

    if bg_color:
        parts.append("box=1")
        parts.append(f"boxcolor={bg_color}")
        parts.append("boxborderw=8")

    return ":".join(parts)


def apply_caption_style(
    video_path: str,
    captions_data: List[Dict],
    style_id: str,
    output: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply a caption style to a video using word-timed captions.

    Args:
        video_path: Input video path.
        captions_data: List of ``{"text": str, "start": float, "end": float}``.
        style_id: ID of the style to apply.
        output: Optional output path (generated if omitted).
        on_progress: Progress callback ``(pct, msg)``.

    Returns:
        Output video path.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not captions_data:
        raise ValueError("captions_data must contain at least one segment")

    style = get_style(style_id)
    out = output or output_path(video_path, f"captions_{style_id}")

    if on_progress:
        on_progress(10, f"Applying caption style: {style.name}")

    # Probe dimensions
    from opencut.helpers import get_video_info
    info = get_video_info(video_path)
    w = info.get("width", 1920)
    h = info.get("height", 1080)

    # Build filter chain -- one drawtext per caption segment
    filters = []
    total = len(captions_data)
    for i, cap in enumerate(captions_data):
        text = cap.get("text", "")
        start = float(cap.get("start", 0))
        end = float(cap.get("end", start + 2))
        filt = _build_drawtext_filter(style, text, start, end, w, h)
        filters.append(filt)

        if on_progress and total > 0:
            pct = 10 + int(60 * (i + 1) / total)
            on_progress(pct, f"Building caption {i + 1}/{total}")

    vf = ",".join(filters) if filters else "null"

    cmd = [
        get_ffmpeg_path(), "-y", "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy", out,
    ]

    if on_progress:
        on_progress(75, "Rendering captioned video...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Captions applied")

    return out


def generate_style_preview(
    style_id: str, sample_text: str = "Hello World"
) -> bytes:
    """
    Render a single-frame PNG preview of a caption style.

    Returns raw PNG bytes.
    """
    style = get_style(style_id)

    # Build a 640x360 preview image using a minimal PNG with FFmpeg
    width, height = 640, 360
    y = _position_y(style.position, height, style.font_size)
    text_color = style.colors.get("text", "#FFFFFF")
    shadow_color = style.colors.get("shadow", "#000000")
    bg_color = style.colors.get("background", "")
    escaped = _escape_drawtext(sample_text)
    font_resolution = resolve_caption_font(style.font_family, sample_text)

    dt_parts = [
        f"drawtext=expansion=none:text='{escaped}'",
        _drawtext_font_option(font_resolution),
        f"fontsize={min(style.font_size, 36)}",
        f"fontcolor={text_color}",
        f"shadowcolor={shadow_color}",
        "shadowx=2", "shadowy=2",
        "x=(w-text_w)/2",
        f"y={y}",
    ]
    if bg_color:
        dt_parts.extend(["box=1", f"boxcolor={bg_color}", "boxborderw=6"])

    dt_filter = ":".join(dt_parts)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            get_ffmpeg_path(), "-y",
            "-f", "lavfi",
            "-i", f"color=c=#1A1A2E:s={width}x{height}:d=1",
            "-vf", dt_filter,
            "-frames:v", "1",
            tmp_path,
        ]
        run_ffmpeg(cmd)

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
