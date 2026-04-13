"""
Programmatic Video from Data (21.5)

Generate videos from data sources + templates, enabling batch
production of personalized video variants.
"""

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import FFmpegCmd, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class DataVideoTemplate:
    """Template for data-driven video generation."""
    name: str
    background: str = "black"  # color name or path to background video/image
    width: int = 1920
    height: int = 1080
    duration: float = 10.0
    fps: float = 30.0
    text_fields: List[Dict[str, Any]] = field(default_factory=list)
    # Each text_field: {"key": "name", "x": 100, "y": 200, "fontsize": 48,
    #                   "fontcolor": "white", "default": ""}
    audio_path: str = ""
    overlay_path: str = ""  # optional overlay image/logo

    def validate(self):
        """Validate template configuration."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        if self.width % 2 != 0 or self.height % 2 != 0:
            raise ValueError("Width and height must be even numbers")
        for tf in self.text_fields:
            if "key" not in tf:
                raise ValueError("Each text_field must have a 'key'")


# ---------------------------------------------------------------------------
# Single Video Generation
# ---------------------------------------------------------------------------
def create_data_video(
    template: DataVideoTemplate,
    data_row: Dict[str, str],
    output_path_str: str,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Generate a single video from a template and data row.

    Args:
        template: DataVideoTemplate definition.
        data_row: Dict mapping field keys to values.
        output_path_str: Output file path.
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with output_path, duration, fields_used.
    """
    template.validate()

    if on_progress:
        on_progress(5, "Preparing video generation...")

    os.makedirs(os.path.dirname(os.path.abspath(output_path_str)), exist_ok=True)

    # Build FFmpeg filter for text overlays
    bg = template.background
    is_bg_file = os.path.isfile(bg) if bg and bg not in (
        "black", "white", "blue", "green", "red", "gray"
    ) else False

    # Build drawtext filters
    drawtext_parts = []
    fields_used = []

    for tf in template.text_fields:
        key = tf["key"]
        text = data_row.get(key, tf.get("default", ""))
        if not text:
            continue

        # Escape for FFmpeg drawtext
        safe = (text
                .replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace(":", "\\:")
                .replace("%", "%%"))

        x = tf.get("x", "(w-text_w)/2")
        y = tf.get("y", "(h-text_h)/2")
        fontsize = tf.get("fontsize", 48)
        fontcolor = tf.get("fontcolor", "white")

        drawtext_parts.append(
            f"drawtext=text='{safe}':fontsize={fontsize}:"
            f"fontcolor={fontcolor}:x={x}:y={y}"
        )
        fields_used.append(key)

    if on_progress:
        on_progress(20, f"Building video with {len(fields_used)} text fields...")

    if is_bg_file:
        # Use background file as input
        cmd = FFmpegCmd().input(bg)
        if drawtext_parts:
            vf = ",".join(drawtext_parts)
            cmd.video_filter(vf)
        cmd = (cmd
               .video_codec("libx264", crf=18, preset="fast")
               .audio_codec("aac", bitrate="192k")
               .faststart()
               .output(output_path_str))
    else:
        # Generate color background
        color_input = f"color=c={bg}:s={template.width}x{template.height}:d={template.duration}:r={template.fps}"
        cmd = FFmpegCmd().option("f", "lavfi").input(color_input)

        # Add silent audio
        cmd.option("f", "lavfi").input(
            f"anullsrc=r=48000:cl=stereo:d={template.duration}"
        )

        if drawtext_parts:
            vf = ",".join(drawtext_parts)
            cmd.video_filter(vf)

        cmd = (cmd
               .video_codec("libx264", crf=18, preset="fast")
               .audio_codec("aac", bitrate="192k")
               .option("shortest")
               .faststart()
               .output(output_path_str))

    if on_progress:
        on_progress(40, "Encoding video...")

    run_ffmpeg(cmd.build())

    # Overlay logo if specified
    if template.overlay_path and os.path.isfile(template.overlay_path):
        if on_progress:
            on_progress(80, "Adding overlay...")
        _apply_overlay(output_path_str, template.overlay_path)

    if on_progress:
        on_progress(100, "Video generated")

    return {
        "output_path": output_path_str,
        "duration": template.duration,
        "fields_used": fields_used,
        "data_row": {k: v for k, v in data_row.items() if k in fields_used},
    }


def _apply_overlay(video_path: str, overlay_path: str):
    """Apply an overlay image to the video (in-place via temp file)."""
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="opencut_overlay_")
    os.close(tmp_fd)

    try:
        fc = "[0:v][1:v]overlay=10:10:shortest=1[outv]"
        cmd = (FFmpegCmd()
               .input(video_path)
               .input(overlay_path)
               .filter_complex(fc, maps=["[outv]", "0:a"])
               .video_codec("libx264", crf=18, preset="fast")
               .audio_codec("aac", bitrate="192k")
               .faststart()
               .output(tmp_path)
               .build())
        run_ffmpeg(cmd)
        os.replace(tmp_path, video_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Batch Video Generation
# ---------------------------------------------------------------------------
def batch_data_videos(
    template: DataVideoTemplate,
    csv_path: str,
    output_dir: str,
    filename_field: str = "",
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Generate batch videos from a CSV file and template.

    Args:
        template: DataVideoTemplate definition.
        csv_path: Path to CSV file with data rows.
        output_dir: Directory for output videos.
        filename_field: CSV column to use for output filenames.
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with output_dir, total, succeeded, failed, files.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    template.validate()
    os.makedirs(output_dir, exist_ok=True)

    if on_progress:
        on_progress(5, "Reading CSV data...")

    # Read CSV
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("CSV file is empty or has no data rows")

    if on_progress:
        on_progress(10, f"Processing {len(rows)} rows...")

    total = len(rows)
    succeeded = 0
    failed = 0
    files = []
    errors = []

    for i, row in enumerate(rows):
        if on_progress:
            pct = 10 + int((i / total) * 85)
            on_progress(pct, f"Generating video {i + 1}/{total}...")

        # Determine output filename
        if filename_field and filename_field in row:
            safe_name = "".join(
                c if c.isalnum() or c in "-_ " else "_"
                for c in row[filename_field]
            ).strip()[:80]
            out_file = os.path.join(output_dir, f"{safe_name}.mp4")
        else:
            out_file = os.path.join(output_dir, f"video_{i + 1:04d}.mp4")

        try:
            create_data_video(template, row, out_file)
            files.append(out_file)
            succeeded += 1
        except Exception as e:
            logger.warning("Failed to generate video %d: %s", i + 1, e)
            errors.append({"row": i + 1, "error": str(e)})
            failed += 1

    if on_progress:
        on_progress(100, f"Batch complete: {succeeded}/{total} succeeded")

    return {
        "output_dir": output_dir,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "files": files,
        "errors": errors[:20],
    }
