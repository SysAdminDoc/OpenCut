"""
OpenCut End Screen / CTA Template v1.0.0

YouTube-style end screen templates with:
  - Subscribe button placement
  - Video recommendation card placeholders
  - Channel branding areas
  - Animated transitions (fade in/out)
  - 5-20 second duration support
"""

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Template Definitions
# ---------------------------------------------------------------------------

@dataclass
class EndScreenElement:
    """A positioned element in an end screen template."""
    element_type: str = ""    # "subscribe", "video_card", "text", "logo"
    x: float = 0.0           # x position as fraction of width (0-1)
    y: float = 0.0           # y position as fraction of height (0-1)
    width: float = 0.0       # width as fraction of total width
    height: float = 0.0      # height as fraction of total height
    label: str = ""


@dataclass
class EndScreenTemplate:
    """End screen template definition."""
    name: str = ""
    label: str = ""
    description: str = ""
    elements: List[EndScreenElement] = field(default_factory=list)
    background_color: str = "000000"
    accent_color: str = "FF0000"
    text_color: str = "FFFFFF"


# Pre-defined templates
_TEMPLATES: Dict[str, EndScreenTemplate] = {
    "youtube_classic": EndScreenTemplate(
        name="youtube_classic",
        label="YouTube Classic",
        description="Two video cards side by side with subscribe button below.",
        elements=[
            EndScreenElement("video_card", 0.05, 0.15, 0.42, 0.45, "Video 1"),
            EndScreenElement("video_card", 0.53, 0.15, 0.42, 0.45, "Video 2"),
            EndScreenElement("subscribe", 0.30, 0.70, 0.40, 0.12, "Subscribe"),
        ],
        background_color="1A1A2E",
        accent_color="FF0000",
    ),
    "youtube_split": EndScreenTemplate(
        name="youtube_split",
        label="YouTube Split",
        description="One large video card left, subscribe and playlist right.",
        elements=[
            EndScreenElement("video_card", 0.05, 0.10, 0.50, 0.55, "Featured Video"),
            EndScreenElement("subscribe", 0.60, 0.10, 0.35, 0.12, "Subscribe"),
            EndScreenElement("video_card", 0.60, 0.30, 0.35, 0.35, "Next Video"),
            EndScreenElement("text", 0.60, 0.70, 0.35, 0.10, "Thanks for watching!"),
        ],
        background_color="16213E",
        accent_color="E94560",
    ),
    "minimal": EndScreenTemplate(
        name="minimal",
        label="Minimal",
        description="Clean minimal end screen with centered subscribe CTA.",
        elements=[
            EndScreenElement("text", 0.20, 0.20, 0.60, 0.15, "Channel Name"),
            EndScreenElement("subscribe", 0.30, 0.45, 0.40, 0.12, "Subscribe"),
            EndScreenElement("video_card", 0.25, 0.65, 0.50, 0.25, "Recommended"),
        ],
        background_color="0F0F0F",
        accent_color="FF0000",
        text_color="FFFFFF",
    ),
    "podcast": EndScreenTemplate(
        name="podcast",
        label="Podcast",
        description="Podcast-style with episode links and subscribe.",
        elements=[
            EndScreenElement("logo", 0.35, 0.05, 0.30, 0.25, "Show Logo"),
            EndScreenElement("text", 0.15, 0.35, 0.70, 0.10, "Show Title"),
            EndScreenElement("video_card", 0.05, 0.50, 0.42, 0.30, "Previous Episode"),
            EndScreenElement("video_card", 0.53, 0.50, 0.42, 0.30, "Next Episode"),
            EndScreenElement("subscribe", 0.25, 0.85, 0.50, 0.10, "Subscribe"),
        ],
        background_color="2C3333",
        accent_color="00ADB5",
        text_color="EEEEEE",
    ),
    "gaming": EndScreenTemplate(
        name="gaming",
        label="Gaming",
        description="Gaming-style with bold colors and multiple video slots.",
        elements=[
            EndScreenElement("text", 0.10, 0.05, 0.80, 0.12, "Channel Name"),
            EndScreenElement("video_card", 0.03, 0.22, 0.30, 0.35, "Video 1"),
            EndScreenElement("video_card", 0.35, 0.22, 0.30, 0.35, "Video 2"),
            EndScreenElement("video_card", 0.67, 0.22, 0.30, 0.35, "Video 3"),
            EndScreenElement("subscribe", 0.20, 0.65, 0.60, 0.12, "Subscribe Now!"),
            EndScreenElement("text", 0.25, 0.82, 0.50, 0.08, "Like & Share"),
        ],
        background_color="1A1A2E",
        accent_color="E94560",
        text_color="EAEAEA",
    ),
}


def list_templates() -> List[dict]:
    """List available end screen templates.

    Returns:
        List of dicts with template name, label, description, and element info.
    """
    result = []
    for name, tmpl in _TEMPLATES.items():
        result.append({
            "name": tmpl.name,
            "label": tmpl.label,
            "description": tmpl.description,
            "element_count": len(tmpl.elements),
            "elements": [
                {"type": e.element_type, "label": e.label}
                for e in tmpl.elements
            ],
        })
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _build_end_screen_filter(
    template: EndScreenTemplate,
    data: dict,
    width: int,
    height: int,
    duration: float,
    fade_duration: float = 1.0,
) -> str:
    """Build FFmpeg filter_complex string for end screen rendering.

    This generates colored rectangles, text overlays, and fade effects
    to represent the end screen layout.
    """
    bg_color = template.background_color
    accent = template.accent_color
    text_color = template.text_color

    # Start with background color
    parts = [f"color=c=0x{bg_color}:s={width}x{height}:d={duration}[bg]"]

    prev_label = "bg"
    for i, elem in enumerate(template.elements):
        ex = int(elem.x * width)
        ey = int(elem.y * height)
        ew = int(elem.width * width)
        eh = int(elem.height * height)

        # Get custom text from data
        custom_text = data.get(f"element_{i}_text", elem.label)
        safe_text = custom_text.replace("'", "\\'").replace(":", "\\:")

        out_label = f"e{i}"

        if elem.element_type == "subscribe":
            # Red/accent subscribe button
            font_size = max(16, eh // 3)
            parts.append(
                f"[{prev_label}]drawbox=x={ex}:y={ey}:w={ew}:h={eh}"
                f":color=0x{accent}@0.9:t=fill,"
                f"drawtext=text='{safe_text}'"
                f":fontsize={font_size}:fontcolor=0x{text_color}"
                f":x={ex}+({ew}-text_w)/2:y={ey}+({eh}-text_h)/2"
                f"[{out_label}]"
            )
        elif elem.element_type == "video_card":
            # Dark rectangle with border for video card placeholder
            font_size = max(14, eh // 6)
            parts.append(
                f"[{prev_label}]drawbox=x={ex}:y={ey}:w={ew}:h={eh}"
                f":color=0x333333@0.8:t=fill,"
                f"drawbox=x={ex}:y={ey}:w={ew}:h={eh}"
                f":color=0x{accent}@0.6:t=3,"
                f"drawtext=text='{safe_text}'"
                f":fontsize={font_size}:fontcolor=0x{text_color}@0.8"
                f":x={ex}+({ew}-text_w)/2:y={ey}+({eh}-text_h)/2"
                f"[{out_label}]"
            )
        elif elem.element_type == "logo":
            # Circular-ish logo placeholder
            font_size = max(14, min(ew, eh) // 5)
            parts.append(
                f"[{prev_label}]drawbox=x={ex}:y={ey}:w={ew}:h={eh}"
                f":color=0x{accent}@0.4:t=fill,"
                f"drawtext=text='{safe_text}'"
                f":fontsize={font_size}:fontcolor=0x{text_color}"
                f":x={ex}+({ew}-text_w)/2:y={ey}+({eh}-text_h)/2"
                f"[{out_label}]"
            )
        else:
            # Text element
            font_size = max(16, eh // 2)
            parts.append(
                f"[{prev_label}]drawtext=text='{safe_text}'"
                f":fontsize={font_size}:fontcolor=0x{text_color}"
                f":x={ex}+({ew}-text_w)/2:y={ey}+({eh}-text_h)/2"
                f"[{out_label}]"
            )
        prev_label = out_label

    # Add fade in/out
    if fade_duration > 0:
        fade_out_start = max(0, duration - fade_duration)
        parts.append(
            f"[{prev_label}]fade=t=in:st=0:d={fade_duration},"
            f"fade=t=out:st={fade_out_start}:d={fade_duration}[out]"
        )
    else:
        parts.append(f"[{prev_label}]null[out]")

    return ";".join(parts)


def generate_end_screen(
    template: str,
    data: dict,
    duration: float = 10.0,
    output_path_str: str = "",
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    fade_duration: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Generate an animated end screen video from a template.

    Args:
        template: Template name (e.g. "youtube_classic").
        data: Dict with element customization (channel_name, element_N_text, etc.).
        duration: End screen duration in seconds (5-20).
        output_path_str: Output file path. Auto-generated if empty.
        width: Video width.
        height: Video height.
        fps: Frames per second.
        fade_duration: Fade in/out duration in seconds.
        on_progress: Optional callback (percent, message).

    Returns:
        Dict with output_path, duration, template used, dimensions.

    Raises:
        ValueError: If template is unknown or duration out of range.
    """
    template_name = template.lower().strip()
    if template_name not in _TEMPLATES:
        raise ValueError(
            f"Unknown template: {template_name}. "
            f"Available: {', '.join(_TEMPLATES.keys())}"
        )

    duration = max(5.0, min(20.0, float(duration)))
    fade_duration = max(0.0, min(duration / 3, float(fade_duration)))

    tmpl = _TEMPLATES[template_name]

    if not output_path_str:
        output_path_str = os.path.join(
            tempfile.gettempdir(),
            f"opencut_endscreen_{template_name}_{int(time.time())}.mp4"
        )

    if on_progress:
        on_progress(10, f"Building end screen ({tmpl.label})...")

    fc = _build_end_screen_filter(tmpl, data, width, height, duration, fade_duration)

    cmd = (FFmpegCmd()
           .filter_complex(fc, maps=["[out]"])
           .video_codec("libx264", crf=18, preset="fast")
           .option("r", str(fps))
           .option("t", str(duration))
           .faststart()
           .output(output_path_str)
           .build())

    if on_progress:
        on_progress(30, "Rendering end screen...")

    try:
        run_ffmpeg(cmd, timeout=120)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to render end screen: {e}")

    if on_progress:
        on_progress(100, "End screen generated.")

    return {
        "output_path": output_path_str,
        "template": template_name,
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "elements": len(tmpl.elements),
    }


def preview_template(
    template: str,
    data: dict,
    width: int = 1280,
    height: int = 720,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Generate a single-frame preview of an end screen template.

    Args:
        template: Template name.
        data: Dict with element customization.
        width: Preview width.
        height: Preview height.
        on_progress: Optional callback (percent, message).

    Returns:
        Dict with output_path for the preview image.
    """
    template_name = template.lower().strip()
    if template_name not in _TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}")

    tmpl = _TEMPLATES[template_name]
    output_file = os.path.join(
        tempfile.gettempdir(),
        f"opencut_endscreen_preview_{template_name}.jpg"
    )

    if on_progress:
        on_progress(20, "Generating preview...")

    fc = _build_end_screen_filter(tmpl, data, width, height, 1.0, 0.0)

    cmd = (FFmpegCmd()
           .filter_complex(fc, maps=["[out]"])
           .frames(1)
           .option("q:v", "2")
           .output(output_file)
           .build())

    try:
        run_ffmpeg(cmd, timeout=30)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to generate preview: {e}")

    if on_progress:
        on_progress(100, "Preview generated.")

    return {
        "output_path": output_file,
        "template": template_name,
        "width": width,
        "height": height,
    }
