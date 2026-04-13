"""
OpenCut Lower-Thirds Module

Auto lower-thirds generation from data sources:
- Render styled lower-third graphics via Pillow
- Styles: modern, corporate, news, minimal
- Animate with fade + slide, export as transparent overlay video
- Batch generation from list of dicts or CSV file

Requires Pillow for rendering, FFmpeg for video export.
"""

import csv
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

from opencut.helpers import get_ffmpeg_path, run_ffmpeg

logger = logging.getLogger("opencut")


@dataclass
class LowerThirdEntry:
    """A single lower-third data entry."""
    name: str
    title: str = ""
    organization: str = ""
    timestamp: float = 0.0


@dataclass
class LowerThirdResult:
    """Result of lower-third generation."""
    output_path: str = ""
    style: str = "modern"
    duration: float = 5.0
    width: int = 1920
    height: int = 1080


@dataclass
class BatchLowerThirdResult:
    """Result of batch lower-third generation."""
    output_dir: str = ""
    files: List[str] = field(default_factory=list)
    count: int = 0
    style: str = "modern"


# ---------------------------------------------------------------------------
# Style Definitions
# ---------------------------------------------------------------------------

STYLES = {
    "modern": {
        "bg_color": (30, 30, 30, 220),
        "accent_color": (0, 150, 255, 255),
        "name_color": (255, 255, 255, 255),
        "title_color": (180, 180, 180, 255),
        "name_font_size": 42,
        "title_font_size": 28,
        "padding": 30,
        "accent_width": 4,
        "bar_height_factor": 1.0,
    },
    "corporate": {
        "bg_color": (20, 40, 80, 230),
        "accent_color": (200, 170, 50, 255),
        "name_color": (255, 255, 255, 255),
        "title_color": (200, 200, 220, 255),
        "name_font_size": 38,
        "title_font_size": 26,
        "padding": 35,
        "accent_width": 6,
        "bar_height_factor": 1.1,
    },
    "news": {
        "bg_color": (180, 20, 20, 240),
        "accent_color": (255, 255, 255, 255),
        "name_color": (255, 255, 255, 255),
        "title_color": (255, 220, 220, 255),
        "name_font_size": 46,
        "title_font_size": 30,
        "padding": 25,
        "accent_width": 0,
        "bar_height_factor": 1.2,
    },
    "minimal": {
        "bg_color": (0, 0, 0, 150),
        "accent_color": (255, 255, 255, 200),
        "name_color": (255, 255, 255, 255),
        "title_color": (200, 200, 200, 255),
        "name_font_size": 36,
        "title_font_size": 24,
        "padding": 20,
        "accent_width": 2,
        "bar_height_factor": 0.8,
    },
}


# ---------------------------------------------------------------------------
# Pillow Rendering
# ---------------------------------------------------------------------------

def _get_font(size: int):
    """Get a Pillow font, falling back to default if no system fonts available."""
    from PIL import ImageFont

    # Try common sans-serif fonts
    font_names = [
        "arial.ttf", "Arial.ttf",
        "DejaVuSans.ttf", "dejavu-sans/DejaVuSans.ttf",
        "Helvetica.ttf", "LiberationSans-Regular.ttf",
        "NotoSans-Regular.ttf",
    ]
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue

    # Fallback to default
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _render_lower_third_frame(
    name: str,
    title: str = "",
    organization: str = "",
    style: str = "modern",
    width: int = 1920,
    height: int = 1080,
    slide_progress: float = 1.0,
    opacity: float = 1.0,
) -> object:
    """Render a single lower-third frame as a Pillow RGBA image.

    Args:
        name: Primary name text.
        title: Secondary title/role text.
        organization: Organization text (appended to title).
        style: Style preset name.
        width: Canvas width.
        height: Canvas height.
        slide_progress: 0.0 (fully left) to 1.0 (final position) for slide animation.
        opacity: Overall opacity 0.0-1.0 for fade animation.

    Returns:
        Pillow RGBA Image.
    """
    from PIL import Image, ImageDraw

    s = STYLES.get(style, STYLES["modern"])
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    name_font = _get_font(s["name_font_size"])
    title_font = _get_font(s["title_font_size"])

    # Measure text
    name_bbox = draw.textbbox((0, 0), name, font=name_font)
    name_w = name_bbox[2] - name_bbox[0]
    name_h = name_bbox[3] - name_bbox[1]

    full_title = title
    if organization and title:
        full_title = f"{title}  |  {organization}"
    elif organization:
        full_title = organization

    title_w, title_h = 0, 0
    if full_title:
        title_bbox = draw.textbbox((0, 0), full_title, font=title_font)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]

    padding = s["padding"]
    text_block_h = name_h + (title_h + 8 if full_title else 0)
    bar_w = max(name_w, title_w) + padding * 2 + s["accent_width"]
    bar_h = int((text_block_h + padding * 2) * s["bar_height_factor"])

    # Position: bottom-left with margin
    margin_x = 80
    margin_y = 120
    bar_x = margin_x + int((1.0 - slide_progress) * (-bar_w - margin_x))
    bar_y = height - margin_y - bar_h

    # Apply opacity to colors
    def _apply_opacity(color):
        r, g, b, a = color
        return (r, g, b, int(a * opacity))

    bg = _apply_opacity(s["bg_color"])
    accent = _apply_opacity(s["accent_color"])
    name_c = _apply_opacity(s["name_color"])
    title_c = _apply_opacity(s["title_color"])

    # Draw background bar
    draw.rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
        fill=bg,
    )

    # Draw accent line/bar
    if s["accent_width"] > 0:
        draw.rectangle(
            [bar_x, bar_y, bar_x + s["accent_width"], bar_y + bar_h],
            fill=accent,
        )

    # Draw name text
    text_x = bar_x + s["accent_width"] + padding
    text_y = bar_y + padding
    draw.text((text_x, text_y), name, fill=name_c, font=name_font)

    # Draw title text
    if full_title:
        draw.text(
            (text_x, text_y + name_h + 8),
            full_title, fill=title_c, font=title_font,
        )

    return canvas


# ---------------------------------------------------------------------------
# Video Export
# ---------------------------------------------------------------------------

def generate_lower_third(
    name: str,
    title: str = "",
    organization: str = "",
    style: str = "modern",
    duration: float = 5.0,
    width: int = 1920,
    height: int = 1080,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Generate a lower-third overlay video with animation.

    Renders lower-third graphic via Pillow with fade in + slide from left,
    hold, and fade out. Exports as transparent overlay video segment.

    Args:
        name: Primary name/label text.
        title: Secondary title/role.
        organization: Organization name.
        style: Style preset: "modern", "corporate", "news", "minimal".
        duration: Total duration in seconds.
        width: Output width.
        height: Output height.
        output_path_str: Output file path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, style, duration, width, height.
    """
    from opencut.helpers import ensure_package
    if not ensure_package("PIL", "Pillow", on_progress=on_progress):
        raise RuntimeError("Pillow is required for lower-thirds. Install with: pip install Pillow")

    if on_progress:
        on_progress(5, "Generating lower-third frames...")

    style = style if style in STYLES else "modern"
    duration = max(1.0, min(30.0, duration))
    fps = 30
    total_frames = int(duration * fps)

    # Animation timing
    fade_in_dur = min(0.5, duration * 0.15)
    slide_in_dur = min(0.8, duration * 0.2)
    fade_out_dur = min(0.5, duration * 0.15)
    hold_end = duration - fade_out_dur

    if output_path_str is None:
        safe_name = "".join(c for c in name[:30] if c.isalnum() or c in " _-").strip().replace(" ", "_")
        output_path_str = os.path.join(tempfile.gettempdir(), f"lower_third_{safe_name}.mov")

    tmp_dir = tempfile.mkdtemp(prefix="opencut_lt_")
    try:
        # Render frames
        for i in range(total_frames):
            t = i / fps

            # Calculate animation values
            if t < slide_in_dur:
                slide_progress = t / slide_in_dur
                # Ease out for slide
                slide_progress = 1 - (1 - slide_progress) * (1 - slide_progress)
            else:
                slide_progress = 1.0

            if t < fade_in_dur:
                opacity = t / fade_in_dur
            elif t > hold_end:
                opacity = max(0.0, 1.0 - (t - hold_end) / fade_out_dur)
            else:
                opacity = 1.0

            frame = _render_lower_third_frame(
                name=name, title=title, organization=organization,
                style=style, width=width, height=height,
                slide_progress=slide_progress, opacity=opacity,
            )

            frame_path = os.path.join(tmp_dir, f"frame_{i:05d}.png")
            frame.save(frame_path, "PNG")

            if on_progress and total_frames > 0:
                pct = 10 + int(70 * (i + 1) / total_frames)
                on_progress(pct, f"Rendering frame {i + 1}/{total_frames}...")

        if on_progress:
            on_progress(85, "Encoding overlay video...")

        # Encode frames to video with alpha (MOV/ProRes 4444 or VP9 webm)
        frame_pattern = os.path.join(tmp_dir, "frame_%05d.png")

        if output_path_str.endswith(".webm"):
            cmd = [
                get_ffmpeg_path(),
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
                "-b:v", "2M", "-y", output_path_str,
            ]
        else:
            # Default to MOV with PNG codec for lossless alpha
            if not output_path_str.endswith(".mov"):
                output_path_str = os.path.splitext(output_path_str)[0] + ".mov"
            cmd = [
                get_ffmpeg_path(),
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", "png", "-pix_fmt", "rgba",
                "-y", output_path_str,
            ]

        run_ffmpeg(cmd)

        if on_progress:
            on_progress(100, "Lower-third generated")

        return {
            "output_path": output_path_str,
            "style": style,
            "duration": duration,
            "width": width,
            "height": height,
        }

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CSV Parsing
# ---------------------------------------------------------------------------

def parse_csv_data(csv_path: str) -> List[dict]:
    """Parse a CSV file into a list of lower-third data dicts.

    Expected columns: name (required), title, organization, timestamp.
    Accepts any column order. Column headers are case-insensitive.

    Args:
        csv_path: Path to CSV file.

    Returns:
        List of dicts with keys: name, title, organization, timestamp.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    entries = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalize header names
        if reader.fieldnames is None:
            return entries

        header_map = {}
        for h in reader.fieldnames:
            lower = h.strip().lower()
            header_map[h] = lower

        for row in reader:
            normalized = {header_map.get(k, k.strip().lower()): v.strip() for k, v in row.items() if v}
            name = normalized.get("name", "")
            if not name:
                continue
            timestamp = 0.0
            ts_str = normalized.get("timestamp", "")
            if ts_str:
                try:
                    timestamp = float(ts_str)
                except ValueError:
                    pass
            entries.append({
                "name": name,
                "title": normalized.get("title", ""),
                "organization": normalized.get("organization", ""),
                "timestamp": timestamp,
            })

    return entries


# ---------------------------------------------------------------------------
# Batch Generation
# ---------------------------------------------------------------------------

def batch_lower_thirds(
    data_source: Union[str, List[dict]],
    style: str = "modern",
    duration: float = 5.0,
    width: int = 1920,
    height: int = 1080,
    output_dir: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Generate lower-thirds in batch from a data source.

    Args:
        data_source: Either a list of dicts [{name, title, organization, timestamp}]
                     or a path to a CSV file.
        style: Style preset for all lower-thirds.
        duration: Duration for each lower-third.
        width: Output width.
        height: Output height.
        output_dir: Output directory. Uses temp dir if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_dir, files list, count, style.
    """
    # Parse data source
    if isinstance(data_source, str):
        entries = parse_csv_data(data_source)
    elif isinstance(data_source, list):
        entries = data_source
    else:
        raise ValueError("data_source must be a list of dicts or a CSV file path")

    if not entries:
        raise ValueError("No entries found in data source")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="opencut_lt_batch_")
    os.makedirs(output_dir, exist_ok=True)

    if on_progress:
        on_progress(5, f"Generating {len(entries)} lower-thirds...")

    files = []
    for idx, entry in enumerate(entries):
        name = entry.get("name", f"Person {idx + 1}")
        title = entry.get("title", "")
        organization = entry.get("organization", "")

        safe_name = "".join(c for c in name[:30] if c.isalnum() or c in " _-").strip().replace(" ", "_")
        out_path = os.path.join(output_dir, f"lower_third_{idx:03d}_{safe_name}.mov")

        def _sub_progress(pct, msg):
            base = int(5 + 90 * idx / len(entries))
            per_item = 90 / len(entries)
            overall = base + int(per_item * pct / 100)
            if on_progress:
                on_progress(min(95, overall), f"[{idx + 1}/{len(entries)}] {msg}")

        result = generate_lower_third(
            name=name,
            title=title,
            organization=organization,
            style=style,
            duration=duration,
            width=width,
            height=height,
            output_path_str=out_path,
            on_progress=_sub_progress,
        )
        files.append(result["output_path"])

    if on_progress:
        on_progress(100, f"Generated {len(files)} lower-thirds")

    return {
        "output_dir": output_dir,
        "files": files,
        "count": len(files),
        "style": style,
    }
