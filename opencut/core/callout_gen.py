"""
OpenCut Callout & Annotation Generator (Feature 11.3)

Numbered step callouts, spotlight boxes, blur/redact regions,
and arrow annotations at timestamps.

Uses FFmpeg drawtext/drawbox/boxblur filters - no additional dependencies.
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Region:
    """A rectangular region on the video frame."""
    x: int
    y: int
    w: int
    h: int


@dataclass
class Annotation:
    """A callout annotation at a specific time range."""
    type: str              # "callout", "spotlight", "blur", "arrow", "step"
    start_time: float      # seconds
    end_time: float        # seconds
    region: Optional[Region] = None  # target region for spotlight/blur
    text: str = ""         # text for callout/step
    number: int = 0        # step number for step callouts
    color: str = "yellow"  # annotation color
    font_size: int = 24    # text size
    arrow_from: Optional[Tuple[int, int]] = None  # arrow start (x, y)
    arrow_to: Optional[Tuple[int, int]] = None     # arrow end (x, y)


@dataclass
class StepCallout:
    """A numbered step callout configuration."""
    text: str
    number: int
    style: str = "circle"    # circle, square, badge
    color: str = "yellow"
    bg_color: str = "black"
    font_size: int = 24

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "number": self.number,
            "style": self.style,
            "color": self.color,
            "bg_color": self.bg_color,
            "font_size": self.font_size,
        }


@dataclass
class CalloutResult:
    """Result of generating callouts."""
    output_path: str
    annotation_count: int
    types_used: List[str]
    duration: float

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "annotation_count": self.annotation_count,
            "types_used": self.types_used,
            "duration": self.duration,
        }


@dataclass
class SpotlightResult:
    """Result of creating a spotlight effect."""
    output_path: str
    region: dict
    time_range: Tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "region": self.region,
            "time_range": list(self.time_range),
        }


# ---------------------------------------------------------------------------
# Drawtext escaping
# ---------------------------------------------------------------------------

def _escape_text(text: str) -> str:
    """Escape special characters for FFmpeg drawtext."""
    text = text.replace("\\", "\\\\\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("%", "%%")
    return text


# ---------------------------------------------------------------------------
# Step callout builder
# ---------------------------------------------------------------------------

def create_step_callout(
    text: str,
    number: int,
    style: str = "circle",
    color: str = "yellow",
    bg_color: str = "black",
    font_size: int = 24,
) -> StepCallout:
    """Create a step callout configuration.

    Args:
        text: Step description text.
        number: Step number (1-based).
        style: Visual style - "circle", "square", or "badge".
        color: Text/border color.
        bg_color: Background color.
        font_size: Font size in pixels.

    Returns:
        StepCallout dataclass.

    Raises:
        ValueError: If number < 1 or style is invalid.
    """
    if number < 1:
        raise ValueError(f"Step number must be >= 1, got {number}")
    valid_styles = ("circle", "square", "badge")
    if style not in valid_styles:
        raise ValueError(f"Invalid style: {style!r}. Valid: {', '.join(valid_styles)}")
    if not text.strip():
        raise ValueError("Step text cannot be empty")

    return StepCallout(
        text=text.strip(),
        number=number,
        style=style,
        color=color,
        bg_color=bg_color,
        font_size=font_size,
    )


# ---------------------------------------------------------------------------
# Filter builders
# ---------------------------------------------------------------------------

def _build_callout_filter(ann: Annotation, video_w: int, video_h: int) -> List[str]:
    """Build FFmpeg filter(s) for a text callout annotation."""
    filters = []
    t_start = ann.start_time
    t_end = ann.end_time
    enable = f"between(t,{t_start:.3f},{t_end:.3f})"
    escaped = _escape_text(ann.text)

    # Position: if region given, place above it; otherwise center
    if ann.region:
        x_pos = str(ann.region.x)
        y_pos = str(max(0, ann.region.y - ann.font_size - 16))
    else:
        x_pos = "(w-tw)/2"
        y_pos = "h-th-40"

    filters.append(
        f"drawtext=text='{escaped}'"
        f":fontsize={ann.font_size}:fontcolor={ann.color}"
        f":x={x_pos}:y={y_pos}"
        f":box=1:boxcolor=black@0.6:boxborderw=6"
        f":enable='{enable}'"
    )
    return filters


def _build_step_filter(ann: Annotation, video_w: int, video_h: int) -> List[str]:
    """Build FFmpeg filter(s) for a numbered step callout."""
    filters = []
    t_start = ann.start_time
    t_end = ann.end_time
    enable = f"between(t,{t_start:.3f},{t_end:.3f})"

    # Step number badge
    num_text = _escape_text(str(ann.number))
    badge_x = "20"
    badge_y = "20"

    if ann.region:
        badge_x = str(max(0, ann.region.x - 40))
        badge_y = str(ann.region.y)

    # Number badge (circle-like box)
    filters.append(
        f"drawbox=x={badge_x}:y={badge_y}:w=32:h=32"
        f":color={ann.color}@0.9:t=fill:enable='{enable}'"
    )
    filters.append(
        f"drawtext=text='{num_text}'"
        f":fontsize=20:fontcolor=black"
        f":x={badge_x}+8:y={badge_y}+6"
        f":enable='{enable}'"
    )

    # Step text next to badge
    if ann.text:
        escaped = _escape_text(ann.text)
        text_x = f"{badge_x}+40"
        text_y = f"{badge_y}+4"
        filters.append(
            f"drawtext=text='{escaped}'"
            f":fontsize={ann.font_size}:fontcolor={ann.color}"
            f":x={text_x}:y={text_y}"
            f":box=1:boxcolor=black@0.6:boxborderw=4"
            f":enable='{enable}'"
        )

    return filters


def _build_spotlight_filter(ann: Annotation, video_w: int, video_h: int) -> List[str]:
    """Build FFmpeg filter for spotlight (darken everything outside region)."""
    filters = []
    if not ann.region:
        return filters

    r = ann.region
    enable = f"between(t,{ann.start_time:.3f},{ann.end_time:.3f})"

    # Darken the entire frame, then brighten the region
    # We use drawbox with invert fill to create a "spotlight" effect
    # Top dark band
    if r.y > 0:
        filters.append(
            f"drawbox=x=0:y=0:w={video_w}:h={r.y}"
            f":color=black@0.6:t=fill:enable='{enable}'"
        )
    # Bottom dark band
    bottom_y = r.y + r.h
    if bottom_y < video_h:
        filters.append(
            f"drawbox=x=0:y={bottom_y}:w={video_w}:h={video_h - bottom_y}"
            f":color=black@0.6:t=fill:enable='{enable}'"
        )
    # Left dark band (between top and bottom)
    if r.x > 0:
        filters.append(
            f"drawbox=x=0:y={r.y}:w={r.x}:h={r.h}"
            f":color=black@0.6:t=fill:enable='{enable}'"
        )
    # Right dark band
    right_x = r.x + r.w
    if right_x < video_w:
        filters.append(
            f"drawbox=x={right_x}:y={r.y}:w={video_w - right_x}:h={r.h}"
            f":color=black@0.6:t=fill:enable='{enable}'"
        )

    # Highlight border around the spotlight region
    filters.append(
        f"drawbox=x={r.x}:y={r.y}:w={r.w}:h={r.h}"
        f":color={ann.color}@0.8:t=3:enable='{enable}'"
    )

    return filters


def _build_blur_filter(ann: Annotation, video_w: int, video_h: int) -> str:
    """Build FFmpeg filter_complex for blur/redact a region.

    Returns a filter_complex string that crops, blurs, and overlays
    the region back onto the source.
    """
    if not ann.region:
        return ""

    r = ann.region
    enable = f"between(t,{ann.start_time:.3f},{ann.end_time:.3f})"

    # We use drawbox with fill to create a solid redact rectangle
    # For actual blur, filter_complex would be needed, but for
    # simplicity and to keep it as a -vf chain, we use a solid fill
    return (
        f"drawbox=x={r.x}:y={r.y}:w={r.w}:h={r.h}"
        f":color=black@0.95:t=fill:enable='{enable}'"
    )


def _build_arrow_filter(ann: Annotation, video_w: int, video_h: int) -> List[str]:
    """Build FFmpeg drawbox filters to approximate an arrow annotation.

    Since FFmpeg doesn't have a native draw-line/arrow filter,
    we simulate an arrow with a series of small boxes along the path
    and a triangle-like head.
    """
    filters = []
    if not ann.arrow_from or not ann.arrow_to:
        return filters

    x1, y1 = ann.arrow_from
    x2, y2 = ann.arrow_to
    enable = f"between(t,{ann.start_time:.3f},{ann.end_time:.3f})"

    # Draw line segments using small boxes
    import math
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return filters

    steps = max(int(length / 4), 1)
    box_size = 4
    for i in range(steps + 1):
        frac = i / max(steps, 1)
        px = int(x1 + dx * frac)
        py = int(y1 + dy * frac)
        px = max(0, min(px, video_w - box_size))
        py = max(0, min(py, video_h - box_size))
        filters.append(
            f"drawbox=x={px}:y={py}:w={box_size}:h={box_size}"
            f":color={ann.color}@0.9:t=fill:enable='{enable}'"
        )

    # Arrowhead: 3 larger boxes at the tip
    for offset in [(-6, -6), (0, -6), (-6, 0)]:
        ax = max(0, min(x2 + offset[0], video_w - 8))
        ay = max(0, min(y2 + offset[1], video_h - 8))
        filters.append(
            f"drawbox=x={ax}:y={ay}:w=8:h=8"
            f":color={ann.color}@0.9:t=fill:enable='{enable}'"
        )

    return filters


# ---------------------------------------------------------------------------
# Main callout generator
# ---------------------------------------------------------------------------

def generate_callout(
    video_path: str,
    annotations: List[Annotation],
    output_path_str: str = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Generate callout annotations on a video.

    Supports callout text, step numbers, spotlight, blur/redact,
    and arrow annotations.

    Args:
        video_path: Source video file path.
        annotations: List of Annotation objects.
        output_path_str: Output file path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path, annotation_count, types_used, duration.

    Raises:
        FileNotFoundError: If video_path doesn't exist.
        ValueError: If annotations is empty.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not annotations:
        raise ValueError("No annotations provided")

    if on_progress:
        on_progress(5, "Reading video info...")

    info = get_video_info(video_path)
    w, h = info["width"], info["height"]
    duration = info["duration"]

    if on_progress:
        on_progress(10, f"Building {len(annotations)} annotations...")

    # Sort by start time
    sorted_anns = sorted(annotations, key=lambda a: a.start_time)

    all_filters = []
    types_used = set()

    for ann in sorted_anns:
        if ann.start_time < 0 or ann.end_time <= ann.start_time:
            logger.warning("Invalid time range [%.2f, %.2f], skipping", ann.start_time, ann.end_time)
            continue

        types_used.add(ann.type)

        if ann.type == "callout":
            all_filters.extend(_build_callout_filter(ann, w, h))
        elif ann.type == "step":
            all_filters.extend(_build_step_filter(ann, w, h))
        elif ann.type == "spotlight":
            all_filters.extend(_build_spotlight_filter(ann, w, h))
        elif ann.type == "blur":
            blur_f = _build_blur_filter(ann, w, h)
            if blur_f:
                all_filters.append(blur_f)
        elif ann.type == "arrow":
            all_filters.extend(_build_arrow_filter(ann, w, h))
        else:
            logger.warning("Unknown annotation type: %s", ann.type)

    if not all_filters:
        raise ValueError("No valid annotations to render")

    vf_chain = ",".join(all_filters)

    if on_progress:
        on_progress(25, "Encoding video with annotations...")

    out = output_path_str or output_path(video_path, "annotated")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(vf_chain)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(out)
        .build()
    )

    if on_progress:
        on_progress(35, "Running FFmpeg...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Done")

    result = CalloutResult(
        output_path=out,
        annotation_count=len(sorted_anns),
        types_used=sorted(types_used),
        duration=duration,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Spotlight creator
# ---------------------------------------------------------------------------

def create_spotlight(
    video_path: str,
    region: Region,
    timestamp_range: Tuple[float, float],
    output_path_str: str = None,
    border_color: str = "yellow",
    darkness: float = 0.6,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Create a spotlight effect on a specific region of a video.

    Darkens everything outside the region and adds a highlight border.

    Args:
        video_path: Source video file path.
        region: Region to spotlight.
        timestamp_range: (start_time, end_time) in seconds.
        output_path_str: Output file path (auto-generated if None).
        border_color: Highlight border color.
        darkness: Darkness level for non-spotlit areas (0.0 - 1.0).
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path, region, time_range.

    Raises:
        FileNotFoundError: If video_path doesn't exist.
        ValueError: If timestamp_range is invalid.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if len(timestamp_range) != 2 or timestamp_range[0] >= timestamp_range[1]:
        raise ValueError("timestamp_range must be (start, end) with start < end")
    if timestamp_range[0] < 0:
        raise ValueError("Start time cannot be negative")

    ann = Annotation(
        type="spotlight",
        start_time=timestamp_range[0],
        end_time=timestamp_range[1],
        region=region,
        color=border_color,
    )

    result = generate_callout(
        video_path=video_path,
        annotations=[ann],
        output_path_str=output_path_str,
        on_progress=on_progress,
    )

    return SpotlightResult(
        output_path=result["output_path"],
        region={"x": region.x, "y": region.y, "w": region.w, "h": region.h},
        time_range=timestamp_range,
    ).to_dict()
