"""
OpenCut A/B Variant Generator

Create 3-5 versions of the same clip with different hooks, captions,
thumbnails, music, or pacing for A/B testing.

Each variant gets a unique combination from available options.
Output files are named ``{base}_variant_A.mp4``, ``_variant_B.mp4``, etc.
"""

import logging
import os
import string
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")

VARY_ELEMENTS = ("hook", "caption_style", "thumbnail", "music", "pacing")
_VARIANT_LETTERS = list(string.ascii_uppercase)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    """A single A/B test variant."""
    variant_id: str = ""
    output_path: str = ""
    changes: Dict = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VariantSetResult:
    """Result containing all generated variants."""
    variants: List[Variant] = field(default_factory=list)
    output_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "variants": [v.to_dict() for v in self.variants],
            "output_dir": self.output_dir,
        }


# ---------------------------------------------------------------------------
# Variant generation helpers
# ---------------------------------------------------------------------------
# Pacing speed presets
_PACING_PRESETS = [
    {"label": "Original", "atempo": 1.0, "setpts": "PTS"},
    {"label": "Slightly Faster", "atempo": 1.1, "setpts": "PTS/1.1"},
    {"label": "Faster", "atempo": 1.2, "setpts": "PTS/1.2"},
    {"label": "Slightly Slower", "atempo": 0.9, "setpts": "PTS/0.9"},
    {"label": "Slower", "atempo": 0.85, "setpts": "PTS/0.85"},
]

# Caption style IDs to cycle through for variants
_VARIANT_CAPTION_STYLES = [
    "tiktok_bold", "youtube_clean", "karaoke_highlight",
    "neon_glow", "bounce_pop",
]

# Hook types to cycle through
_VARIANT_HOOK_TYPES = ["teaser", "question", "statistic", "quote", "auto"]


def _apply_pacing_variant(
    video_path: str, out_path: str, pacing: Dict
) -> str:
    """Create a pacing variant by adjusting video and audio tempo."""
    atempo = pacing["atempo"]
    setpts = pacing["setpts"]

    if atempo == 1.0:
        # Just copy
        cmd = [
            get_ffmpeg_path(), "-y",
            "-i", video_path,
            "-c", "copy",
            out_path,
        ]
    else:
        vf = f"setpts={setpts}"
        af = f"atempo={atempo}"
        cmd = [
            get_ffmpeg_path(), "-y",
            "-i", video_path,
            "-vf", vf,
            "-af", af,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            out_path,
        ]

    run_ffmpeg(cmd)
    return out_path


def _apply_caption_variant(
    video_path: str, out_path: str, style_id: str
) -> str:
    """Create a caption variant using a specific style overlay on first 5s."""
    from opencut.core.caption_styles import _build_drawtext_filter, get_style

    style = get_style(style_id)
    info = get_video_info(video_path)
    w = info.get("width", 1920)
    h = info.get("height", 1080)

    sample_text = f"Style: {style.name}"
    dt = _build_drawtext_filter(style, sample_text, 0.0, 5.0, w, h)

    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", video_path,
        "-vf", dt,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        out_path,
    ]
    run_ffmpeg(cmd)
    return out_path


def _apply_hook_variant(
    video_path: str, out_path: str, hook_type: str
) -> str:
    """Create a hook variant with a text overlay identifying the hook type."""
    info = get_video_info(video_path)
    w = info.get("width", 1920)
    h = info.get("height", 1080)

    hook_labels = {
        "teaser": "What happens next will surprise you...",
        "question": "Want to know the secret?",
        "statistic": "9 out of 10 people miss this.",
        "quote": "This changed everything.",
        "auto": "You need to see this.",
    }
    text = hook_labels.get(hook_type, "Watch this.")
    escaped = (
        text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace("%", "%%")
    )

    font_size = max(24, min(56, int(w / 28)))
    y_pos = int(h * 0.45)

    drawtext = (
        f"drawtext=text='{escaped}'"
        f":fontsize={font_size}"
        f":fontcolor=#FFFFFF"
        f":shadowcolor=#000000:shadowx=3:shadowy=3"
        f":x=(w-text_w)/2:y={y_pos}"
        f":enable='between(t,0,3)'"
    )

    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", video_path,
        "-vf", drawtext,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        out_path,
    ]
    run_ffmpeg(cmd)
    return out_path


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def generate_variants(
    video_path: str,
    variant_count: int = 3,
    vary_elements: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> VariantSetResult:
    """
    Generate A/B test variants of a video.

    Args:
        video_path: Input video file.
        variant_count: Number of variants to generate (1-5).
        vary_elements: List of elements to vary. Defaults to
                       ``["hook", "caption_style"]``.
        on_progress: Progress callback ``(pct, msg)``.

    Returns:
        :class:`VariantSetResult` with all generated variants.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    variant_count = max(1, min(5, variant_count))

    if vary_elements is None:
        vary_elements = ["hook", "caption_style"]
    for elem in vary_elements:
        if elem not in VARY_ELEMENTS:
            raise ValueError(
                f"Invalid vary element: {elem}. Must be one of {VARY_ELEMENTS}"
            )

    if on_progress:
        on_progress(5, f"Generating {variant_count} variants...")

    out_dir = os.path.dirname(os.path.abspath(video_path))
    base = os.path.splitext(os.path.basename(video_path))[0]

    result = VariantSetResult(output_dir=out_dir)

    for i in range(variant_count):
        letter = _VARIANT_LETTERS[i] if i < len(_VARIANT_LETTERS) else str(i)
        variant_out = os.path.join(out_dir, f"{base}_variant_{letter}.mp4")

        if on_progress:
            pct = 5 + int(90 * i / variant_count)
            on_progress(pct, f"Creating variant {letter}...")

        changes: Dict = {}
        description_parts: List[str] = []

        # Start from the source for each variant
        current_input = video_path

        if "pacing" in vary_elements:
            preset = _PACING_PRESETS[i % len(_PACING_PRESETS)]
            if preset["atempo"] != 1.0:
                _apply_pacing_variant(current_input, variant_out, preset)
                current_input = variant_out
                changes["pacing"] = preset["label"]
                description_parts.append(f"Pacing: {preset['label']}")

        if "caption_style" in vary_elements:
            style_id = _VARIANT_CAPTION_STYLES[
                i % len(_VARIANT_CAPTION_STYLES)
            ]
            _apply_caption_variant(current_input, variant_out, style_id)
            current_input = variant_out
            changes["caption_style"] = style_id
            description_parts.append(f"Caption: {style_id}")

        if "hook" in vary_elements:
            hook_type = _VARIANT_HOOK_TYPES[i % len(_VARIANT_HOOK_TYPES)]
            _apply_hook_variant(current_input, variant_out, hook_type)
            current_input = variant_out
            changes["hook"] = hook_type
            description_parts.append(f"Hook: {hook_type}")

        # If no changes were applied (e.g. only unsupported elements), copy
        if current_input == video_path:
            cmd = [
                get_ffmpeg_path(), "-y",
                "-i", video_path, "-c", "copy", variant_out,
            ]
            run_ffmpeg(cmd)
            changes["copy"] = True
            description_parts.append("Copy of original")

        variant = Variant(
            variant_id=letter,
            output_path=variant_out,
            changes=changes,
            description=", ".join(description_parts) if description_parts else f"Variant {letter}",
        )
        result.variants.append(variant)

    if on_progress:
        on_progress(100, f"Generated {len(result.variants)} variants")

    return result
