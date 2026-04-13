"""
OpenCut Platform Safe Zone Overlay

Defines safe zone templates for major social platforms and burns
semi-transparent colored rectangles onto video copies showing
areas that may be obscured by platform UI elements.

Uses FFmpeg only - no additional dependencies required.
"""

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


@dataclass
class SafeZone:
    """A rectangular region that may be obscured by platform UI."""
    x: int
    y: int
    w: int
    h: int
    label: str
    color: str  # FFmpeg hex color, e.g. "red@0.3"


# ---------------------------------------------------------------------------
# Platform templates (fractional coordinates relative to frame size)
# Each entry: (x_frac, y_frac, w_frac, h_frac, label, color)
# ---------------------------------------------------------------------------
_PLATFORM_TEMPLATES = {
    "youtube": [
        # End screen: bottom 20% of frame
        (0.0, 0.80, 1.0, 0.20, "End Screen Zone", "red"),
        # Title/description overlay: top 10%
        (0.0, 0.0, 1.0, 0.10, "Title Bar Zone", "yellow"),
    ],
    "tiktok": [
        # Bottom 20% for caption / description text
        (0.0, 0.80, 1.0, 0.20, "Caption Zone", "red"),
        # Right 15% for like/comment/share buttons
        (0.85, 0.30, 0.15, 0.50, "Action Buttons", "orange"),
        # Top 10% for following/search bar
        (0.0, 0.0, 1.0, 0.10, "Top Bar Zone", "yellow"),
    ],
    "instagram": [
        # Bottom 15% for caption overlay (Reels)
        (0.0, 0.85, 1.0, 0.15, "Caption Zone", "red"),
        # Right 10% for action buttons (Reels)
        (0.90, 0.40, 0.10, 0.40, "Action Buttons", "orange"),
        # Top 8% for header
        (0.0, 0.0, 1.0, 0.08, "Header Zone", "yellow"),
    ],
    "twitter": [
        # Bottom 12% for engagement bar
        (0.0, 0.88, 1.0, 0.12, "Engagement Bar", "red"),
        # Top 8% for handle/timestamp
        (0.0, 0.0, 1.0, 0.08, "Header Zone", "yellow"),
    ],
}

# Aliases
_PLATFORM_TEMPLATES["x"] = _PLATFORM_TEMPLATES["twitter"]
_PLATFORM_TEMPLATES["reels"] = _PLATFORM_TEMPLATES["instagram"]

SUPPORTED_PLATFORMS = sorted(_PLATFORM_TEMPLATES.keys())


def get_safe_zones(
    platform: str,
    width: int,
    height: int,
) -> List[SafeZone]:
    """
    Return safe zone rectangles for a given platform and resolution.

    Pure data function - no FFmpeg calls.

    Args:
        platform: Platform name (youtube, tiktok, instagram, twitter, x).
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        List of SafeZone dataclass instances with pixel coordinates.

    Raises:
        ValueError: If platform is not supported.
    """
    key = platform.lower().strip()
    templates = _PLATFORM_TEMPLATES.get(key)
    if templates is None:
        raise ValueError(
            f"Unsupported platform: {platform!r}. "
            f"Supported: {', '.join(SUPPORTED_PLATFORMS)}"
        )

    zones = []
    for x_frac, y_frac, w_frac, h_frac, label, color in templates:
        x = int(round(x_frac * width))
        y = int(round(y_frac * height))
        w = int(round(w_frac * width))
        h = int(round(h_frac * height))
        # Clamp to frame bounds
        w = min(w, width - x)
        h = min(h, height - y)
        zones.append(SafeZone(x=x, y=y, w=w, h=h, label=label, color=color))

    return zones


def generate_safe_zone_overlay(
    input_path: str,
    platform: str,
    out_path: str = None,
    opacity: float = 0.3,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Burn semi-transparent safe zone rectangles onto a copy of the video.

    Args:
        input_path: Source video file path.
        platform: Target platform (youtube, tiktok, instagram, twitter, x).
        out_path: Output file path. Auto-generated if None.
        opacity: Overlay opacity (0.0-1.0). Default 0.3.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path and zones list.
    """
    if on_progress:
        on_progress(5, "Reading video info...")

    info = get_video_info(input_path)
    width = info["width"]
    height = info["height"]
    duration = info["duration"]

    if on_progress:
        on_progress(10, f"Computing safe zones for {platform}...")

    opacity = max(0.0, min(1.0, float(opacity)))
    zones = get_safe_zones(platform, width, height)

    if not zones:
        raise ValueError(f"No safe zone templates defined for {platform}")

    if on_progress:
        on_progress(15, "Building FFmpeg filter chain...")

    # Build drawbox filter chain: one drawbox per zone
    # drawbox=x=X:y=Y:w=W:h=H:color=COLOR@OPACITY:t=fill
    filter_parts = []
    for zone in zones:
        color_str = f"{zone.color}@{opacity}"
        filter_parts.append(
            f"drawbox=x={zone.x}:y={zone.y}:w={zone.w}:h={zone.h}"
            f":color={color_str}:t=fill"
        )

    # Also draw labels with drawtext for each zone
    for zone in zones:
        # Position label text at center of zone
        text_x = zone.x + zone.w // 2
        text_y = zone.y + zone.h // 2
        escaped_label = zone.label.replace("'", "\\'").replace(":", "\\:")
        filter_parts.append(
            f"drawtext=text='{escaped_label}'"
            f":x={text_x}-(text_w/2):y={text_y}-(text_h/2)"
            f":fontsize=20:fontcolor=white:borderw=2:bordercolor=black"
        )

    vf_chain = ",".join(filter_parts)

    if out_path is None:
        out_path = output_path(input_path, f"safezone_{platform}")

    if on_progress:
        on_progress(20, "Rendering safe zone overlay...")

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(vf_chain)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(out_path)
        .build()
    )

    run_ffmpeg(cmd, timeout=max(300, int(duration * 3)))

    if on_progress:
        on_progress(100, "Safe zone overlay complete")

    return {
        "output_path": out_path,
        "platform": platform,
        "zones": [
            {
                "x": z.x,
                "y": z.y,
                "w": z.w,
                "h": z.h,
                "label": z.label,
                "color": z.color,
            }
            for z in zones
        ],
    }
