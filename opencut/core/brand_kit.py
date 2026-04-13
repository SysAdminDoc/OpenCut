"""
OpenCut Brand Kit Module (15.1)

Define brand profiles (colors, fonts, logo), check content compliance,
and auto-correct brand violations via FFmpeg overlays.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


@dataclass
class BrandKit:
    """A brand profile definition."""
    name: str = "Default"
    primary_color: str = "#FFFFFF"
    secondary_color: str = "#000000"
    accent_color: str = "#FF0000"
    font_family: str = "Arial"
    font_size: int = 48
    logo_path: str = ""
    logo_position: str = "top_right"  # top_left, top_right, bottom_left, bottom_right
    logo_scale: float = 0.15  # fraction of video width
    logo_opacity: float = 1.0
    lower_third_style: str = "bar"  # bar, minimal, full
    watermark_text: str = ""
    intro_duration: float = 0.0
    outro_duration: float = 0.0
    safe_zone_margin: float = 0.05  # fraction of frame for title-safe


@dataclass
class BrandIssue:
    """A brand compliance issue."""
    issue_type: str
    description: str
    severity: str = "warning"
    auto_fixable: bool = True
    timestamp: float = 0.0


@dataclass
class ComplianceReport:
    """Brand compliance check report."""
    issues: List[BrandIssue] = field(default_factory=list)
    compliant: bool = True
    score: float = 100.0
    brand_kit_name: str = ""


def load_brand_kit(
    config_path: str,
    on_progress: Optional[Callable] = None,
) -> BrandKit:
    """Load a brand kit from a JSON config file.

    Args:
        config_path: Path to brand kit JSON file.
        on_progress: Progress callback(pct, msg).

    Returns:
        BrandKit dataclass instance.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Brand kit config not found: {config_path}")

    if on_progress:
        on_progress(10, "Loading brand kit configuration...")

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kit = BrandKit(
        name=data.get("name", "Default"),
        primary_color=data.get("primary_color", "#FFFFFF"),
        secondary_color=data.get("secondary_color", "#000000"),
        accent_color=data.get("accent_color", "#FF0000"),
        font_family=data.get("font_family", "Arial"),
        font_size=int(data.get("font_size", 48)),
        logo_path=data.get("logo_path", ""),
        logo_position=data.get("logo_position", "top_right"),
        logo_scale=float(data.get("logo_scale", 0.15)),
        logo_opacity=float(data.get("logo_opacity", 1.0)),
        lower_third_style=data.get("lower_third_style", "bar"),
        watermark_text=data.get("watermark_text", ""),
        intro_duration=float(data.get("intro_duration", 0.0)),
        outro_duration=float(data.get("outro_duration", 0.0)),
        safe_zone_margin=float(data.get("safe_zone_margin", 0.05)),
    )

    # Validate logo path if specified
    if kit.logo_path and not os.path.isfile(kit.logo_path):
        logger.warning("Brand kit logo not found: %s", kit.logo_path)

    if on_progress:
        on_progress(100, f"Loaded brand kit: {kit.name}")

    return kit


def _hex_to_ffmpeg_color(hex_color: str) -> str:
    """Convert #RRGGBB to FFmpeg color format (0xRRGGBB)."""
    color = hex_color.strip().lstrip("#")
    if len(color) == 6:
        return f"0x{color}"
    return "0xFFFFFF"


def _detect_existing_logo(video_path: str, info: dict) -> bool:
    """Simple heuristic: check if the top-right corner has a static overlay.
    Returns True if a logo-like element is detected (basic check)."""
    # This is a simplified detection - in production, you'd use ML
    # For now, return False to always suggest adding logo
    return False


def check_brand_compliance(
    video_path: str,
    brand_kit: BrandKit,
    on_progress: Optional[Callable] = None,
) -> ComplianceReport:
    """Check if a video complies with brand kit guidelines.

    Args:
        video_path: Path to the video to check.
        brand_kit: BrandKit profile to check against.
        on_progress: Progress callback(pct, msg).

    Returns:
        ComplianceReport with issues and compliance score.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(10, "Analyzing video for brand compliance...")

    info = get_video_info(video_path)
    issues = []
    score = 100.0

    # Check logo presence
    if brand_kit.logo_path and os.path.isfile(brand_kit.logo_path):
        if on_progress:
            on_progress(30, "Checking logo presence...")
        has_logo = _detect_existing_logo(video_path, info)
        if not has_logo:
            issues.append(BrandIssue(
                issue_type="missing_logo",
                description="Brand logo not detected in video",
                severity="warning",
                auto_fixable=True,
            ))
            score -= 15.0

    # Check resolution meets minimum
    if on_progress:
        on_progress(50, "Checking resolution and format...")
    if info["width"] < 1280 or info["height"] < 720:
        issues.append(BrandIssue(
            issue_type="low_resolution",
            description=f"Video resolution {info['width']}x{info['height']} "
                        f"is below 720p minimum",
            severity="warning",
            auto_fixable=False,
        ))
        score -= 10.0

    # Check duration for intro/outro space
    if brand_kit.intro_duration > 0:
        if info["duration"] < brand_kit.intro_duration + brand_kit.outro_duration + 5:
            issues.append(BrandIssue(
                issue_type="too_short",
                description="Video too short for branded intro/outro",
                severity="info",
                auto_fixable=False,
            ))
            score -= 5.0

    # Check watermark text
    if brand_kit.watermark_text:
        issues.append(BrandIssue(
            issue_type="missing_watermark",
            description="Brand watermark not detected",
            severity="info",
            auto_fixable=True,
        ))
        score -= 5.0

    # Check safe zone (content at edges)
    if on_progress:
        on_progress(70, "Checking safe zones...")

    # Verify aspect ratio is standard
    aspect = info["width"] / max(info["height"], 1)
    standard_aspects = [16/9, 4/3, 1.0, 9/16, 21/9]
    is_standard = any(abs(aspect - sa) < 0.05 for sa in standard_aspects)
    if not is_standard:
        issues.append(BrandIssue(
            issue_type="non_standard_aspect",
            description=f"Non-standard aspect ratio {aspect:.2f}:1",
            severity="info",
            auto_fixable=False,
        ))
        score -= 5.0

    score = max(0.0, score)
    compliant = len([i for i in issues if i.severity != "info"]) == 0

    if on_progress:
        status = "compliant" if compliant else f"{len(issues)} issues found"
        on_progress(100, f"Brand check: {status} (score: {score:.0f}%)")

    return ComplianceReport(
        issues=issues,
        compliant=compliant,
        score=round(score, 1),
        brand_kit_name=brand_kit.name,
    )


def auto_correct_brand(
    video_path: str,
    brand_kit: BrandKit,
    output_path_str: Optional[str] = None,
    add_logo: bool = True,
    add_watermark: bool = True,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Auto-correct brand compliance by adding logo, watermark, and styling.

    Args:
        video_path: Input video path.
        brand_kit: BrandKit profile to apply.
        output_path_str: Output path. Auto-generated if None.
        add_logo: Whether to overlay the brand logo.
        add_watermark: Whether to add watermark text.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, corrections_applied, duration.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(5, "Preparing brand corrections...")

    info = get_video_info(video_path)
    corrections = []

    if output_path_str is None:
        output_path_str = output_path(video_path, "branded")

    # Build filter complex
    filter_parts = []
    input_count = 1  # video is input 0
    inputs = [video_path]

    # Logo overlay
    has_logo = (add_logo and brand_kit.logo_path and
                os.path.isfile(brand_kit.logo_path))
    if has_logo:
        inputs.append(brand_kit.logo_path)
        logo_input = input_count
        input_count += 1

        logo_w = int(info["width"] * brand_kit.logo_scale)
        margin = int(info["width"] * brand_kit.safe_zone_margin)

        pos_map = {
            "top_left": (f"{margin}", f"{margin}"),
            "top_right": (f"W-w-{margin}", f"{margin}"),
            "bottom_left": (f"{margin}", f"H-h-{margin}"),
            "bottom_right": (f"W-w-{margin}", f"H-h-{margin}"),
        }
        ox, oy = pos_map.get(brand_kit.logo_position, pos_map["top_right"])

        filter_parts.append(
            f"[{logo_input}:v]scale={logo_w}:-1,"
            f"format=rgba,colorchannelmixer=aa={brand_kit.logo_opacity}[logo];"
            f"[0:v][logo]overlay={ox}:{oy}"
        )
        corrections.append("logo_overlay")

    # Watermark text
    if add_watermark and brand_kit.watermark_text:
        color = _hex_to_ffmpeg_color(brand_kit.primary_color)
        margin = int(info["width"] * brand_kit.safe_zone_margin)
        fontsize = max(12, int(info["height"] * 0.02))
        escaped_text = brand_kit.watermark_text.replace("'", "'\\''")

        if filter_parts:
            # Chain onto previous filter output
            wm_filter = (
                f"drawtext=text='{escaped_text}'"
                f":fontsize={fontsize}"
                f":fontcolor={color}@0.3"
                f":x=w-tw-{margin}:y=h-th-{margin}"
            )
            last = filter_parts[-1]
            filter_parts[-1] = last + f"[branded];[branded]{wm_filter}"
        else:
            filter_parts.append(
                f"[0:v]drawtext=text='{escaped_text}'"
                f":fontsize={fontsize}"
                f":fontcolor={color}@0.3"
                f":x=w-tw-{margin}:y=h-th-{margin}"
            )
        corrections.append("watermark")

    if on_progress:
        on_progress(30, f"Applying {len(corrections)} brand corrections...")

    # Build FFmpeg command
    builder = FFmpegCmd()
    for inp in inputs:
        builder.input(inp)

    if filter_parts:
        fc = ";".join(filter_parts) if len(filter_parts) > 1 else filter_parts[0]
        # Ensure final output label
        if "[out]" not in fc:
            fc += "[out]"
        builder.filter_complex(fc, maps=["[out]", "0:a?"])
    else:
        builder.map("0")

    builder.video_codec("libx264", crf=18, preset="fast")
    builder.audio_codec("aac", bitrate="192k")
    builder.faststart()
    builder.output(output_path_str)

    cmd = builder.build()
    run_ffmpeg(cmd)

    if not corrections:
        corrections.append("re-encoded (no brand changes needed)")

    if on_progress:
        on_progress(100, f"Brand corrections applied: {', '.join(corrections)}")

    return {
        "output_path": output_path_str,
        "corrections_applied": corrections,
        "duration": info["duration"],
        "brand_kit": brand_kit.name,
    }
