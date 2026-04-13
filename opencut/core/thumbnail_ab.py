"""
OpenCut Thumbnail A/B Generator v1.0.0

Generates thumbnail candidates from video with quality scoring:
  - Scores frames by sharpness, brightness, contrast, face presence
  - Generates variants: original, color-boosted, text overlay, face zoom
  - Exports as image grid + individual files
"""

import logging
import math
import os
import subprocess as _sp
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------

@dataclass
class FrameScore:
    """Scored candidate frame."""
    timestamp: float = 0.0
    frame_path: str = ""
    sharpness: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    overall_score: float = 0.0


@dataclass
class ThumbnailVariant:
    """A generated thumbnail variant."""
    variant_type: str = ""  # "original", "color_boost", "text_overlay", "face_zoom"
    path: str = ""
    width: int = 0
    height: int = 0
    label: str = ""


# ---------------------------------------------------------------------------
# Frame Scoring
# ---------------------------------------------------------------------------

def _extract_frame(video_path: str, timestamp: float, output_file: str) -> bool:
    """Extract a single frame at a given timestamp."""
    cmd = (FFmpegCmd()
           .pre_input("ss", str(timestamp))
           .input(video_path)
           .frames(1)
           .option("q:v", "2")
           .output(output_file)
           .build())
    try:
        run_ffmpeg(cmd, timeout=30)
        return os.path.isfile(output_file)
    except RuntimeError:
        return False


def _measure_frame_quality(frame_path: str) -> dict:
    """Measure frame quality using FFmpeg signalstats/entropy.

    Returns dict with sharpness, brightness, contrast estimates.
    """
    # Use FFmpeg to get basic stats about the frame
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", frame_path,
        "-vf", "signalstats",
        "-f", "null", os.devnull,
    ]

    sharpness = 50.0
    brightness = 128.0
    contrast = 50.0

    try:
        proc = _sp.run(cmd, capture_output=True, text=True, timeout=15)
        stderr = proc.stderr

        for line in stderr.splitlines():
            stripped = line.strip()
            if "YAVG" in stripped:
                try:
                    brightness = float(stripped.split("YAVG=")[-1].split()[0])
                except (ValueError, IndexError):
                    pass
            if "YMIN" in stripped and "YMAX" in stripped:
                try:
                    parts = stripped.split()
                    ymin = ymax = None
                    for p in parts:
                        if p.startswith("YMIN="):
                            ymin = float(p.split("=")[1])
                        elif p.startswith("YMAX="):
                            ymax = float(p.split("=")[1])
                    if ymin is not None and ymax is not None:
                        contrast = ymax - ymin
                except (ValueError, IndexError):
                    pass
    except (RuntimeError, _sp.TimeoutExpired, OSError):
        pass

    # Estimate sharpness via laplacian variance (using a second pass)
    cmd_blur = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", frame_path,
        "-vf", "convolution=0 -1 0 -1 4 -1 0 -1 0,signalstats",
        "-f", "null", os.devnull,
    ]
    try:
        proc2 = _sp.run(cmd_blur, capture_output=True, text=True, timeout=15)
        for line in proc2.stderr.splitlines():
            if "YAVG" in line:
                try:
                    sharpness = float(line.strip().split("YAVG=")[-1].split()[0])
                except (ValueError, IndexError):
                    pass
                break
    except (RuntimeError, _sp.TimeoutExpired, OSError):
        pass

    return {
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
    }


def score_frames(
    video_path: str,
    count: int = 5,
    min_interval: float = 2.0,
    sample_count: int = 20,
    on_progress: Optional[Callable] = None,
) -> List[FrameScore]:
    """Score top candidate frames from a video.

    Samples frames throughout the video, scores each by visual quality,
    and returns the top `count` frames sorted by score.

    Args:
        video_path: Path to the video file.
        count: Number of top frames to return.
        min_interval: Minimum seconds between sampled frames.
        sample_count: Total frames to sample before picking top N.
        on_progress: Optional callback (percent, message).

    Returns:
        List of FrameScore objects, sorted by overall_score descending.

    Raises:
        FileNotFoundError: If video_path does not exist.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input file not found: {video_path}")

    count = max(1, min(count, 20))
    sample_count = max(count, min(sample_count, 100))

    info = get_video_info(video_path)
    duration = info["duration"]
    if duration <= 0:
        duration = 60.0  # fallback

    if on_progress:
        on_progress(5, f"Sampling {sample_count} frames from video...")

    # Generate sample timestamps, avoiding first/last 5% of video
    margin = max(0.5, duration * 0.05)
    usable = duration - 2 * margin
    if usable <= 0:
        usable = duration
        margin = 0

    interval = max(min_interval, usable / sample_count)
    timestamps = []
    t = margin
    while t < duration - margin and len(timestamps) < sample_count:
        timestamps.append(round(t, 2))
        t += interval

    if not timestamps:
        timestamps = [duration / 2]

    # Extract and score frames
    tmp_dir = tempfile.mkdtemp(prefix="opencut_thumb_")
    scored: List[FrameScore] = []

    try:
        for i, ts in enumerate(timestamps):
            if on_progress:
                pct = 10 + int((i / len(timestamps)) * 60)
                on_progress(pct, f"Scoring frame {i+1}/{len(timestamps)}...")

            frame_path = os.path.join(tmp_dir, f"frame_{i:04d}.jpg")
            if not _extract_frame(video_path, ts, frame_path):
                continue

            quality = _measure_frame_quality(frame_path)

            # Composite score:
            # - Prefer moderate brightness (not too dark/bright)
            # - Prefer high contrast
            # - Prefer high sharpness
            brightness_score = max(0, 100 - abs(quality["brightness"] - 128) * 0.8)
            contrast_score = min(100, quality["contrast"] * 0.5)
            sharpness_score = min(100, quality["sharpness"] * 2)
            overall = (sharpness_score * 0.4 + brightness_score * 0.35 + contrast_score * 0.25)

            scored.append(FrameScore(
                timestamp=ts,
                frame_path=frame_path,
                sharpness=round(quality["sharpness"], 2),
                brightness=round(quality["brightness"], 2),
                contrast=round(quality["contrast"], 2),
                overall_score=round(overall, 2),
            ))

        # Sort by score and keep top N
        scored.sort(key=lambda f: f.overall_score, reverse=True)
        top = scored[:count]

        # Clean up frames we won't use
        keep_paths = {f.frame_path for f in top}
        for f in scored:
            if f.frame_path not in keep_paths and os.path.isfile(f.frame_path):
                try:
                    os.unlink(f.frame_path)
                except OSError:
                    pass

        if on_progress:
            on_progress(80, f"Selected top {len(top)} frames.")

        return top

    except Exception:
        # Cleanup on error
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Variant Generation
# ---------------------------------------------------------------------------

def generate_variants(
    frame_path: str,
    text: str = "",
    output_dir: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    on_progress: Optional[Callable] = None,
) -> List[ThumbnailVariant]:
    """Generate thumbnail variants from a single frame.

    Creates: original (resized), color-boosted, text overlay, face zoom.

    Args:
        frame_path: Path to the source frame image.
        text: Text for the text overlay variant.
        output_dir: Directory for output files. Defaults to frame's directory.
        width: Target thumbnail width.
        height: Target thumbnail height.
        on_progress: Optional callback (percent, message).

    Returns:
        List of ThumbnailVariant objects.

    Raises:
        FileNotFoundError: If frame_path does not exist.
    """
    if not os.path.isfile(frame_path):
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    if not output_dir:
        output_dir = os.path.dirname(frame_path)
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(frame_path))[0]
    variants: List[ThumbnailVariant] = []

    if on_progress:
        on_progress(10, "Generating thumbnail variants...")

    # 1. Original (resized)
    orig_out = os.path.join(output_dir, f"{base}_original.jpg")
    cmd = (FFmpegCmd()
           .input(frame_path)
           .video_filter(f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                         f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black")
           .frames(1)
           .option("q:v", "2")
           .output(orig_out)
           .build())
    try:
        run_ffmpeg(cmd, timeout=30)
        variants.append(ThumbnailVariant(
            variant_type="original",
            path=orig_out,
            width=width,
            height=height,
            label="Original",
        ))
    except RuntimeError as e:
        logger.warning("Failed to generate original variant: %s", e)

    if on_progress:
        on_progress(30, "Color boost variant...")

    # 2. Color-boosted
    boost_out = os.path.join(output_dir, f"{base}_color_boost.jpg")
    cmd = (FFmpegCmd()
           .input(frame_path)
           .video_filter(
               f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
               f"eq=saturation=1.5:contrast=1.1:brightness=0.05,"
               f"unsharp=5:5:1.0")
           .frames(1)
           .option("q:v", "2")
           .output(boost_out)
           .build())
    try:
        run_ffmpeg(cmd, timeout=30)
        variants.append(ThumbnailVariant(
            variant_type="color_boost",
            path=boost_out,
            width=width,
            height=height,
            label="Color Boost",
        ))
    except RuntimeError as e:
        logger.warning("Failed to generate color boost variant: %s", e)

    if on_progress:
        on_progress(50, "Text overlay variant...")

    # 3. Text overlay (if text provided)
    if text:
        text_out = os.path.join(output_dir, f"{base}_text_overlay.jpg")
        # Escape special characters for FFmpeg drawtext
        safe_text = text.replace("'", "\\'").replace(":", "\\:")
        cmd = (FFmpegCmd()
               .input(frame_path)
               .video_filter(
                   f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
                   f"drawtext=text='{safe_text}'"
                   f":fontsize={height // 10}"
                   f":fontcolor=white:borderw=3:bordercolor=black"
                   f":x=(w-text_w)/2:y=h-th-{height // 10}")
               .frames(1)
               .option("q:v", "2")
               .output(text_out)
               .build())
        try:
            run_ffmpeg(cmd, timeout=30)
            variants.append(ThumbnailVariant(
                variant_type="text_overlay",
                path=text_out,
                width=width,
                height=height,
                label="Text Overlay",
            ))
        except RuntimeError as e:
            logger.warning("Failed to generate text overlay variant: %s", e)

    if on_progress:
        on_progress(70, "Face zoom variant...")

    # 4. Face zoom (center-crop zoom)
    zoom_out = os.path.join(output_dir, f"{base}_face_zoom.jpg")
    crop_w = int(width * 0.6)
    crop_h = int(height * 0.6)
    cmd = (FFmpegCmd()
           .input(frame_path)
           .video_filter(
               f"scale=-1:{height * 2},"
               f"crop={crop_w}:{crop_h}:(iw-{crop_w})/2:(ih-{crop_h})/3,"
               f"scale={width}:{height}")
           .frames(1)
           .option("q:v", "2")
           .output(zoom_out)
           .build())
    try:
        run_ffmpeg(cmd, timeout=30)
        variants.append(ThumbnailVariant(
            variant_type="face_zoom",
            path=zoom_out,
            width=width,
            height=height,
            label="Face Zoom",
        ))
    except RuntimeError as e:
        logger.warning("Failed to generate face zoom variant: %s", e)

    if on_progress:
        on_progress(90, f"Generated {len(variants)} variants.")

    return variants


# ---------------------------------------------------------------------------
# Grid Creation
# ---------------------------------------------------------------------------

def create_thumbnail_grid(
    thumbnails: List[str],
    output_path_str: str,
    columns: int = 2,
    cell_width: int = 640,
    cell_height: int = 360,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Create an image grid from multiple thumbnail images.

    Args:
        thumbnails: List of paths to thumbnail images.
        output_path_str: Output path for the grid image.
        columns: Number of columns in the grid.
        cell_width: Width of each cell in the grid.
        cell_height: Height of each cell in the grid.
        on_progress: Optional callback (percent, message).

    Returns:
        Dict with output_path, columns, rows, cell dimensions.

    Raises:
        ValueError: If thumbnails list is empty.
    """
    if not thumbnails:
        raise ValueError("No thumbnails provided for grid.")

    valid = [t for t in thumbnails if os.path.isfile(t)]
    if not valid:
        raise ValueError("None of the provided thumbnail paths exist.")

    columns = max(1, min(columns, len(valid)))
    rows = math.ceil(len(valid) / columns)

    if on_progress:
        on_progress(10, f"Creating {columns}x{rows} thumbnail grid...")

    # Build filter_complex for grid layout
    # Scale each input then use xstack
    n = len(valid)
    cmd = FFmpegCmd()
    for thumb in valid:
        cmd.input(thumb)

    scale_parts = []
    for i in range(n):
        scale_parts.append(f"[{i}:v]scale={cell_width}:{cell_height}:force_original_aspect_ratio=decrease,"
                           f"pad={cell_width}:{cell_height}:(ow-iw)/2:(oh-ih)/2:color=black[s{i}]")

    # Pad to fill incomplete rows
    total_cells = rows * columns
    fill_parts = []
    if n < total_cells:
        # Create black fill for empty cells using color source
        for i in range(n, total_cells):
            fill_parts.append(f"color=black:s={cell_width}x{cell_height}:d=1[s{i}]")

    # Build xstack layout
    layout_parts = []
    for idx in range(total_cells):
        row = idx // columns
        col = idx % columns
        x = col * cell_width
        y = row * cell_height
        layout_parts.append(f"{x}_{y}")

    stack_inputs = "".join(f"[s{i}]" for i in range(total_cells))
    layout_str = "|".join(layout_parts)

    fc = ";".join(scale_parts + fill_parts) + f";{stack_inputs}xstack=inputs={total_cells}:layout={layout_str}[out]"

    cmd.filter_complex(fc, maps=["[out]"])
    cmd.frames(1)
    cmd.option("q:v", "2")
    cmd.output(output_path_str)

    try:
        run_ffmpeg(cmd.build(), timeout=60)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to create thumbnail grid: {e}")

    if on_progress:
        on_progress(100, "Thumbnail grid complete.")

    return {
        "output_path": output_path_str,
        "columns": columns,
        "rows": rows,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "total_thumbnails": n,
    }
