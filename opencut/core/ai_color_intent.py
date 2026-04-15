"""
OpenCut AI Color Intent

Natural language color grading — describe the desired look in words and get
FFmpeg filter parameters:
- Parse intent phrases ("warm sunset", "cold corporate", "noir", "bleach bypass")
- Map to FFmpeg eq/colorbalance/curves filter chains
- Built-in library of 30+ named looks
- LLM fallback for custom descriptions

Uses FFmpeg only — no additional dependencies required.
"""

import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Color intent library — 30+ named looks
# ---------------------------------------------------------------------------
COLOR_INTENTS: Dict[str, Dict] = {
    "warm sunset": {
        "description": "Golden hour warmth with orange-red highlights",
        "filters": "eq=brightness=0.04:saturation=1.3,colorbalance=rs=0.15:gs=-0.05:bs=-0.2:rh=0.2:gh=0.05:bh=-0.15",
        "category": "warm",
    },
    "cold corporate": {
        "description": "Cool blue-tinted professional look",
        "filters": "eq=brightness=-0.02:saturation=0.85,colorbalance=rs=-0.1:gs=-0.05:bs=0.15:rh=-0.1:gh=0.0:bh=0.12",
        "category": "cool",
    },
    "noir": {
        "description": "Classic film noir — high contrast, desaturated, dark",
        "filters": "eq=brightness=-0.08:contrast=1.4:saturation=0.15,colorbalance=rs=-0.05:gs=-0.05:bs=0.05",
        "category": "dramatic",
    },
    "bleach bypass": {
        "description": "Silver retention look — low saturation, high contrast, metallic",
        "filters": "eq=contrast=1.3:saturation=0.4:brightness=-0.03,colorbalance=rs=-0.02:gs=-0.02:bs=0.04",
        "category": "stylized",
    },
    "vintage film": {
        "description": "Faded warm tones reminiscent of 70s film stock",
        "filters": "eq=brightness=0.03:saturation=0.7:contrast=0.9,colorbalance=rs=0.1:gs=0.05:bs=-0.05:rm=0.05:gm=0.02:bm=-0.08",
        "category": "retro",
    },
    "teal and orange": {
        "description": "Hollywood blockbuster color grade — teal shadows, orange highlights",
        "filters": "eq=saturation=1.2:contrast=1.1,colorbalance=rs=-0.1:gs=-0.05:bs=0.2:rh=0.2:gh=0.1:bh=-0.15",
        "category": "cinematic",
    },
    "cyberpunk": {
        "description": "Neon-soaked purple-cyan sci-fi look",
        "filters": "eq=saturation=1.5:contrast=1.2:brightness=-0.05,colorbalance=rs=0.1:gs=-0.1:bs=0.2:rh=0.15:gh=-0.05:bh=0.1",
        "category": "stylized",
    },
    "pastel dream": {
        "description": "Soft, airy pastel tones — lifted shadows, low contrast",
        "filters": "eq=brightness=0.08:contrast=0.8:saturation=0.7,colorbalance=rs=0.05:gs=0.05:bs=0.1:rm=0.03:gm=0.05:bm=0.08",
        "category": "soft",
    },
    "moody blue": {
        "description": "Dark blue-tinted moody atmosphere",
        "filters": "eq=brightness=-0.06:saturation=0.9:contrast=1.15,colorbalance=rs=-0.1:gs=-0.08:bs=0.12:rm=-0.05:gm=-0.03:bm=0.08",
        "category": "cool",
    },
    "golden hour": {
        "description": "Warm golden sunlight with soft contrast",
        "filters": "eq=brightness=0.06:saturation=1.2:contrast=0.95,colorbalance=rs=0.18:gs=0.08:bs=-0.1:rh=0.12:gh=0.06:bh=-0.08",
        "category": "warm",
    },
    "forest green": {
        "description": "Rich earthy greens with warm undertones",
        "filters": "eq=saturation=1.15:contrast=1.05,colorbalance=rs=-0.05:gs=0.1:bs=-0.08:rm=-0.03:gm=0.08:bm=-0.05",
        "category": "nature",
    },
    "arctic": {
        "description": "Ultra-cold blue-white icy look",
        "filters": "eq=brightness=0.05:saturation=0.6:contrast=1.1,colorbalance=rs=-0.15:gs=-0.05:bs=0.2:rh=-0.1:gh=0.0:bh=0.15",
        "category": "cool",
    },
    "sepia": {
        "description": "Classic warm brown sepia tone",
        "filters": "eq=saturation=0.3,colorbalance=rs=0.2:gs=0.1:bs=-0.1:rm=0.15:gm=0.08:bm=-0.05",
        "category": "retro",
    },
    "cross process": {
        "description": "Cross-processed film look — shifted colors, high contrast",
        "filters": "eq=contrast=1.25:saturation=1.3,colorbalance=rs=-0.1:gs=0.15:bs=-0.05:rh=0.15:gh=-0.1:bh=0.1",
        "category": "stylized",
    },
    "horror": {
        "description": "Desaturated green-tinted horror atmosphere",
        "filters": "eq=brightness=-0.1:contrast=1.3:saturation=0.4,colorbalance=rs=-0.08:gs=0.05:bs=-0.05:rm=-0.05:gm=0.08:bm=-0.03",
        "category": "dramatic",
    },
    "dreamy": {
        "description": "Soft glow with lifted blacks and warm tint",
        "filters": "eq=brightness=0.07:contrast=0.85:saturation=0.9,colorbalance=rs=0.08:gs=0.05:bs=0.08:rm=0.05:gm=0.03:bm=0.05",
        "category": "soft",
    },
    "high key": {
        "description": "Bright, clean, overexposed look",
        "filters": "eq=brightness=0.12:contrast=0.9:saturation=0.85",
        "category": "bright",
    },
    "low key": {
        "description": "Dark, dramatic, heavy shadows",
        "filters": "eq=brightness=-0.12:contrast=1.4:saturation=0.8",
        "category": "dramatic",
    },
    "matte": {
        "description": "Matte finish — lifted blacks, reduced contrast",
        "filters": "eq=contrast=0.8:brightness=0.03:saturation=0.9,colorbalance=rs=0.02:gs=0.02:bs=0.02",
        "category": "soft",
    },
    "technicolor": {
        "description": "Saturated vintage Technicolor film look",
        "filters": "eq=saturation=1.6:contrast=1.15,colorbalance=rs=0.1:gs=-0.05:bs=-0.1:rh=0.08:gh=0.05:bh=-0.05",
        "category": "retro",
    },
    "underwater": {
        "description": "Blue-green aquatic color cast",
        "filters": "eq=brightness=-0.03:saturation=0.85,colorbalance=rs=-0.12:gs=0.05:bs=0.15:rm=-0.08:gm=0.08:bm=0.1",
        "category": "nature",
    },
    "autumn": {
        "description": "Warm fall colors — orange, red, gold",
        "filters": "eq=saturation=1.25:contrast=1.05,colorbalance=rs=0.15:gs=0.03:bs=-0.12:rm=0.1:gm=0.02:bm=-0.08",
        "category": "warm",
    },
    "moonlight": {
        "description": "Cool blue moonlit night scene",
        "filters": "eq=brightness=-0.08:saturation=0.5:contrast=1.1,colorbalance=rs=-0.1:gs=-0.05:bs=0.18:rh=-0.08:gh=-0.03:bh=0.12",
        "category": "cool",
    },
    "candy": {
        "description": "Vibrant saturated pop-art colors",
        "filters": "eq=saturation=1.8:contrast=1.15:brightness=0.03,colorbalance=rs=0.1:gs=-0.05:bs=0.1",
        "category": "stylized",
    },
    "desaturated": {
        "description": "Nearly monochrome with faint color retention",
        "filters": "eq=saturation=0.2:contrast=1.1",
        "category": "dramatic",
    },
    "infrared": {
        "description": "False-color infrared photography look",
        "filters": "eq=saturation=1.4:contrast=1.2,colorbalance=rs=0.2:gs=-0.15:bs=-0.1:rm=0.15:gm=-0.1:bm=-0.08",
        "category": "stylized",
    },
    "newsroom": {
        "description": "Neutral, balanced broadcast look",
        "filters": "eq=brightness=0.02:contrast=1.05:saturation=0.95",
        "category": "neutral",
    },
    "romantic": {
        "description": "Soft pink warmth with gentle glow",
        "filters": "eq=brightness=0.05:contrast=0.9:saturation=0.95,colorbalance=rs=0.12:gs=0.02:bs=0.05:rm=0.08:gm=0.0:bm=0.03",
        "category": "soft",
    },
    "western": {
        "description": "Dusty desert tones — yellow-brown warmth",
        "filters": "eq=saturation=0.85:contrast=1.1:brightness=0.02,colorbalance=rs=0.12:gs=0.08:bs=-0.1:rm=0.08:gm=0.05:bm=-0.08",
        "category": "warm",
    },
    "matrix": {
        "description": "Green-tinted digital rain aesthetic",
        "filters": "eq=saturation=0.6:contrast=1.2:brightness=-0.05,colorbalance=rs=-0.1:gs=0.15:bs=-0.1:rm=-0.08:gm=0.12:bm=-0.08",
        "category": "stylized",
    },
    "clean": {
        "description": "Neutral, properly exposed, balanced colors",
        "filters": "eq=brightness=0.01:contrast=1.02:saturation=1.0",
        "category": "neutral",
    },
    "blockbuster": {
        "description": "High-budget cinematic — rich contrast, controlled saturation",
        "filters": "eq=contrast=1.15:saturation=1.1:brightness=-0.02,colorbalance=rs=0.02:gs=-0.02:bs=0.02:rh=0.05:gh=0.0:bh=-0.05",
        "category": "cinematic",
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ColorIntent:
    """Parsed color intent with resolved parameters."""
    intent: str = ""
    resolved_name: str = ""     # matched intent name from library
    description: str = ""
    filters: str = ""           # FFmpeg filter string
    intensity: float = 1.0      # 0..2 multiplier
    category: str = ""
    source: str = "library"     # "library", "llm", "custom"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ColorIntentResult:
    """Result of applying a color intent to video."""
    output_path: str = ""
    intent: Dict = field(default_factory=dict)
    applied_filters: str = ""
    preview_frame: str = ""     # path to preview frame if generated
    duration: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Intent resolution
# ---------------------------------------------------------------------------
def _fuzzy_match_intent(query: str) -> Optional[str]:
    """Find the closest matching intent name via keyword overlap."""
    query_lower = query.lower().strip()

    # Exact match
    if query_lower in COLOR_INTENTS:
        return query_lower

    # Keyword overlap scoring
    query_words = set(query_lower.split())
    best_name = None
    best_score = 0

    for name, data in COLOR_INTENTS.items():
        name_words = set(name.lower().split())
        desc_words = set(data.get("description", "").lower().split())
        category = data.get("category", "")

        # Score: word overlap with name (weighted 3x) + description + category match
        name_overlap = len(query_words & name_words) * 3
        desc_overlap = len(query_words & desc_words)
        cat_match = 2 if category.lower() in query_lower else 0

        score = name_overlap + desc_overlap + cat_match
        if score > best_score:
            best_score = score
            best_name = name

    return best_name if best_score >= 2 else None


def _apply_intensity(filters: str, intensity: float) -> str:
    """Scale filter parameters by intensity factor.

    intensity=1.0 is default, 0.5 is subtle, 2.0 is extreme.
    """
    if abs(intensity - 1.0) < 0.01:
        return filters

    def _scale_param(match):
        key = match.group(1)
        val = float(match.group(2))

        if key in ("saturation",):
            # Scale deviation from 1.0
            new_val = 1.0 + (val - 1.0) * intensity
        elif key in ("brightness",):
            new_val = val * intensity
        elif key in ("contrast",):
            new_val = 1.0 + (val - 1.0) * intensity
        elif key in ("rs", "gs", "bs", "rm", "gm", "bm", "rh", "gh", "bh"):
            new_val = val * intensity
        else:
            new_val = val

        return f"{key}={new_val:.3f}"

    result = re.sub(r"(\w+)=(-?[0-9.]+)", _scale_param, filters)
    return result


def _resolve_intent_via_llm(intent_text: str) -> Optional[ColorIntent]:
    """Use LLM to parse a custom color intent description into filter params."""
    try:
        from opencut.core.llm import query_llm

        prompt = (
            "You are a color grading expert. Given a color intent description, "
            "output FFmpeg eq and colorbalance filter parameters.\n\n"
            "Format: eq=brightness=X:contrast=Y:saturation=Z,"
            "colorbalance=rs=A:gs=B:bs=C:rh=D:gh=E:bh=F\n\n"
            "Brightness: -0.15 to 0.15\n"
            "Contrast: 0.7 to 1.5\n"
            "Saturation: 0.1 to 2.0\n"
            "Color balance values: -0.2 to 0.2\n\n"
            f"Intent: {intent_text}\n\n"
            "Output ONLY the filter string, nothing else."
        )

        result = query_llm(prompt)
        if not isinstance(result, str):
            return None

        # Validate the response looks like FFmpeg filters
        result = result.strip().strip('"').strip("'")
        if "eq=" not in result and "colorbalance=" not in result:
            return None

        # Basic sanity check on parameter values
        for match in re.finditer(r"=(-?[0-9.]+)", result):
            val = float(match.group(1))
            if abs(val) > 10:
                return None

        return ColorIntent(
            intent=intent_text,
            resolved_name=intent_text,
            description=f"Custom LLM-generated grade for '{intent_text}'",
            filters=result,
            category="custom",
            source="llm",
        )

    except Exception as exc:
        logger.debug("LLM color intent resolution failed: %s", exc)
        return None


def resolve_color_intent(intent_text: str, intensity: float = 1.0) -> ColorIntent:
    """Resolve an intent description to FFmpeg filter parameters.

    Tries library match first, then LLM fallback, then default neutral.
    """
    # 1. Try library match
    matched_name = _fuzzy_match_intent(intent_text)
    if matched_name:
        data = COLOR_INTENTS[matched_name]
        filters = _apply_intensity(data["filters"], intensity)
        return ColorIntent(
            intent=intent_text,
            resolved_name=matched_name,
            description=data["description"],
            filters=filters,
            intensity=intensity,
            category=data.get("category", ""),
            source="library",
        )

    # 2. Try LLM
    llm_intent = _resolve_intent_via_llm(intent_text)
    if llm_intent:
        llm_intent.intensity = intensity
        if intensity != 1.0:
            llm_intent.filters = _apply_intensity(llm_intent.filters, intensity)
        return llm_intent

    # 3. Fallback to neutral
    logger.info("Could not resolve color intent '%s', using clean/neutral", intent_text)
    data = COLOR_INTENTS["clean"]
    return ColorIntent(
        intent=intent_text,
        resolved_name="clean",
        description=f"Fallback: {data['description']}",
        filters=_apply_intensity(data["filters"], intensity),
        intensity=intensity,
        category="neutral",
        source="library",
    )


# ---------------------------------------------------------------------------
# Preview frame generation
# ---------------------------------------------------------------------------
def preview_color_intent(
    video_path: str,
    intent: str = "warm sunset",
    intensity: float = 1.0,
    timestamp: float = -1.0,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Generate a single preview frame with the color intent applied.

    Args:
        video_path: Path to source video.
        intent: Color intent text.
        intensity: Intensity multiplier (0.0-2.0).
        timestamp: Frame to capture (-1 for auto midpoint).
        on_progress: Progress callback.

    Returns:
        Dict with preview_path, intent info, and applied filters.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(10, "Resolving color intent...")

    color_intent = resolve_color_intent(intent, intensity)

    # Determine timestamp
    if timestamp < 0:
        info = get_video_info(video_path)
        timestamp = info.get("duration", 10) / 2

    if on_progress:
        on_progress(30, "Generating preview frame...")

    # Generate preview frame
    preview_path = output_path(video_path, f"preview_{intent.replace(' ', '_')}")
    preview_path = os.path.splitext(preview_path)[0] + ".jpg"

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-vf", color_intent.filters,
        "-q:v", "2",
        preview_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30, check=True)
    except Exception as exc:
        logger.warning("Preview generation failed: %s", exc)
        raise RuntimeError(f"Preview generation failed: {exc}")

    if on_progress:
        on_progress(100, "Preview ready")

    return {
        "preview_path": preview_path,
        "intent": color_intent.to_dict(),
        "applied_filters": color_intent.filters,
        "timestamp": timestamp,
    }


# ---------------------------------------------------------------------------
# Full video color intent application
# ---------------------------------------------------------------------------
def apply_color_intent(
    video_path: str,
    intent: str = "warm sunset",
    intensity: float = 1.0,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ColorIntentResult:
    """
    Apply a natural-language color intent to an entire video.

    Args:
        video_path: Path to the source video file.
        intent: Color intent description (name or free-form text).
        intensity: Intensity multiplier (0.0 = none, 1.0 = default, 2.0 = extreme).
        output_dir: Optional output directory.
        on_progress: Callback ``(pct, msg)`` for progress updates.

    Returns:
        ColorIntentResult with output path and applied filter info.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If FFmpeg encoding fails.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    intensity = max(0.0, min(2.0, intensity))

    if on_progress:
        on_progress(5, "Resolving color intent...")

    color_intent = resolve_color_intent(intent, intensity)

    if on_progress:
        on_progress(10, f"Applying '{color_intent.resolved_name}' look...")

    info = get_video_info(video_path)
    duration = info.get("duration", 0)

    # Build output path
    safe_name = re.sub(r"[^a-z0-9]+", "_", intent.lower())[:30]
    out_path = output_path(video_path, f"color_{safe_name}", output_dir)

    # Build FFmpeg command
    cmd = (FFmpegCmd()
           .input(video_path)
           .video_filter(color_intent.filters)
           .video_codec("libx264", crf=18, preset="medium")
           .audio_codec("copy")
           .faststart()
           .output(out_path)
           .build())

    if on_progress:
        on_progress(15, "Encoding...")

    try:
        run_ffmpeg(cmd, timeout=3600)
    except RuntimeError as exc:
        raise RuntimeError(f"Color grading failed: {exc}")

    if on_progress:
        on_progress(95, "Verifying output...")

    if not os.path.isfile(out_path):
        raise RuntimeError("Output file was not created")

    result = ColorIntentResult(
        output_path=out_path,
        intent=color_intent.to_dict(),
        applied_filters=color_intent.filters,
        duration=duration,
    )

    if on_progress:
        on_progress(100, "Color intent applied")

    return result
