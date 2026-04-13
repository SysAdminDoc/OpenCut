"""
OpenCut Batch Thumbnail Extraction

Extract representative thumbnails from multiple video files using FFmpeg.
Supports fixed-time, scene-detect, and auto (most interesting frame) modes.

Also generates contact-sheet montages via Pillow.
"""

import logging
import math
import os
import tempfile
from typing import Callable, List, Optional

from opencut.helpers import (
    _get_file_duration,
    get_ffmpeg_path,
    get_ffprobe_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


def extract_thumbnails(
    file_paths: List[str],
    mode: str = "auto",
    timestamp_pct: float = 0.1,
    output_dir: Optional[str] = None,
    width: int = 640,
    format: str = "jpg",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Extract a single representative thumbnail from each video file.

    Args:
        file_paths: List of video file paths.
        mode: Extraction strategy:
            - ``"fixed"``  — extract at ``timestamp_pct`` fraction of duration.
            - ``"scene"``  — first scene-change keyframe.
            - ``"auto"``   — frame at 10% duration (avoids black intros).
        timestamp_pct: Fraction of duration for fixed mode (0.0-1.0).
        output_dir: Directory for thumbnails; defaults to temp dir.
        width: Output thumbnail width (height auto-scaled).
        format: ``"jpg"`` or ``"png"``.
        on_progress: Callback(percent, message).

    Returns:
        dict with ``thumbnails`` list, ``success_count``, ``error_count``,
        ``output_dir``.
    """
    total = len(file_paths)
    if total == 0:
        return {"thumbnails": [], "success_count": 0, "error_count": 0, "output_dir": ""}

    out_dir = output_dir or tempfile.mkdtemp(prefix="opencut_thumbs_")
    os.makedirs(out_dir, exist_ok=True)

    fmt_ext = ".png" if format == "png" else ".jpg"
    thumbnails = []
    success = 0
    errors = 0

    for idx, fp in enumerate(file_paths):
        pct = int((idx / total) * 100)
        if on_progress:
            on_progress(pct, f"Extracting thumbnail {idx + 1}/{total}")

        base = os.path.splitext(os.path.basename(fp))[0]
        thumb_path = os.path.join(out_dir, f"{base}_thumb{fmt_ext}")

        try:
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"File not found: {fp}")

            _extract_single(fp, thumb_path, mode, timestamp_pct, width, fmt_ext)

            thumbnails.append({
                "input": fp,
                "thumbnail": thumb_path,
                "status": "ok",
            })
            success += 1
        except Exception as exc:
            logger.error("Thumbnail extraction error for %s: %s", fp, exc)
            thumbnails.append({
                "input": fp,
                "error": str(exc),
                "status": "error",
            })
            errors += 1

    if on_progress:
        on_progress(100, f"Thumbnails extracted: {success} ok, {errors} errors")

    return {
        "thumbnails": thumbnails,
        "success_count": success,
        "error_count": errors,
        "output_dir": out_dir,
    }


def _extract_single(
    input_path: str,
    output_path: str,
    mode: str,
    timestamp_pct: float,
    width: int,
    fmt_ext: str,
) -> None:
    """Extract one thumbnail from a video file."""
    duration = _get_file_duration(input_path)

    if mode == "scene":
        seek_time = _find_scene_keyframe(input_path, duration)
    elif mode == "fixed":
        seek_time = max(0, duration * max(0.0, min(1.0, timestamp_pct)))
    else:
        # "auto" — use 10% to skip black intros, or 1s for very short clips
        seek_time = max(1.0, duration * 0.1) if duration > 2 else 0

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
        "-y", "-ss", f"{seek_time:.3f}",
        "-i", input_path,
        "-vf", f"scale={width}:-1",
        "-vframes", "1",
        "-q:v", "2",
        output_path,
    ]
    run_ffmpeg(cmd, timeout=60)


def _find_scene_keyframe(input_path: str, duration: float) -> float:
    """Find the timestamp of the first scene change, falling back to 10% of duration."""
    import json
    import subprocess as _sp

    try:
        # Use ffprobe to detect scene changes in the first 30s
        limit = min(30, duration) if duration > 0 else 30
        cmd = [
            get_ffprobe_path(), "-v", "error",
            "-read_intervals", f"%+{limit:.0f}",
            "-show_frames", "-select_streams", "v",
            "-of", "json",
            "-f", "lavfi",
            f"movie={input_path},select=gt(scene\\,0.3)",
        ]
        result = _sp.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        frames = data.get("frames", [])
        if frames:
            pts = float(frames[0].get("pts_time", 0))
            if pts > 0:
                return pts
    except Exception as exc:
        logger.debug("Scene detection fallback for %s: %s", input_path, exc)

    # Fallback
    return max(1.0, duration * 0.1) if duration > 2 else 0


def generate_contact_sheet(
    thumbnails: List[str],
    columns: int = 4,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Generate a contact sheet (montage) from thumbnail images.

    Args:
        thumbnails: List of thumbnail image file paths.
        columns: Number of columns in the grid.
        output_path: Output image path; auto-generated if None.
        on_progress: Callback(percent, message).

    Returns:
        dict with ``output_path``, ``columns``, ``rows``, ``total``.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    except ImportError:
        raise ImportError("Pillow is required for contact sheet generation. Install with: pip install Pillow")

    if not thumbnails:
        raise ValueError("No thumbnails provided for contact sheet")

    if on_progress:
        on_progress(10, "Loading thumbnails...")

    # Load all images
    images = []
    labels = []
    for tp in thumbnails:
        if os.path.isfile(tp):
            images.append(Image.open(tp))
            labels.append(os.path.basename(tp))

    if not images:
        raise ValueError("No valid thumbnail images found")

    # Determine cell size from the first image
    cell_w = images[0].width
    cell_h = images[0].height
    label_h = 20  # space for filename label

    rows = math.ceil(len(images) / columns)
    sheet_w = cell_w * columns
    sheet_h = (cell_h + label_h) * rows

    if on_progress:
        on_progress(30, f"Building {columns}x{rows} contact sheet...")

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(32, 32, 32))
    draw = ImageDraw.Draw(sheet)

    for idx, (img, label) in enumerate(zip(images, labels)):
        col = idx % columns
        row = idx // columns
        x = col * cell_w
        y = row * (cell_h + label_h)

        # Resize to cell dimensions
        img_resized = img.resize((cell_w, cell_h), Image.LANCZOS)
        sheet.paste(img_resized, (x, y))

        # Draw filename label
        label_y = y + cell_h + 2
        # Truncate long names
        display_label = label[:30] + "..." if len(label) > 33 else label
        try:
            draw.text((x + 4, label_y), display_label, fill=(200, 200, 200))
        except Exception:
            pass  # Font rendering failure is non-fatal

    # Save
    out = output_path or os.path.join(
        tempfile.mkdtemp(prefix="opencut_sheet_"),
        "contact_sheet.jpg",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if on_progress:
        on_progress(80, "Saving contact sheet...")

    sheet.save(out, quality=90)

    if on_progress:
        on_progress(100, "Contact sheet generated")

    return {
        "output_path": out,
        "columns": columns,
        "rows": rows,
        "total": len(images),
    }
