"""
OpenCut Star Trail Compositing Module v1.0.0

Lighten-mode stacking, gap filling, airplane/streak removal,
and progressive animation video generation.
"""

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

from opencut.helpers import FFmpegCmd, ensure_package, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StarTrailResult:
    """Result for a star trail composite operation."""
    output_path: str = ""
    frames_processed: int = 0
    frames_skipped: int = 0
    streaks_removed: int = 0
    duration_seconds: float = 0.0
    status: str = "pending"
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _collect_images(image_paths: List[str]) -> List[str]:
    """Filter and sort valid image paths."""
    valid = []
    for p in image_paths:
        if os.path.isfile(p):
            ext = os.path.splitext(p)[1].lower()
            if ext in _IMAGE_EXTENSIONS:
                valid.append(p)
    return sorted(valid)


def _load_image(path: str):
    """Load an image as a numpy array."""
    import numpy as np
    from PIL import Image
    img = Image.open(path)
    return np.array(img, dtype=np.float32)


def _save_image(arr, path: str):
    """Save a numpy array as an image."""
    import numpy as np
    from PIL import Image
    arr_clipped = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr_clipped).save(path, quality=95)


# ---------------------------------------------------------------------------
# Streak / airplane removal
# ---------------------------------------------------------------------------

def remove_streaks(
    frames: List[str],
    threshold: float = 50.0,
    min_length: int = 20,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Detect and mask airplane/satellite streaks in frames.

    Compares consecutive frames to find fast-moving linear bright
    features. Returns a dict of frame_path -> mask information.

    Args:
        frames: List of image paths.
        threshold: Brightness difference threshold for streak detection.
        min_length: Minimum pixel length to classify as streak.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        Dict with ``frames_analyzed``, ``streaks_found``, ``streak_map``.
    """
    if not ensure_package("numpy"):
        raise RuntimeError("numpy required for streak removal")
    if not ensure_package("PIL", "Pillow"):
        raise RuntimeError("Pillow required for streak removal")

    import numpy as np

    valid = _collect_images(frames)
    if len(valid) < 2:
        return {"frames_analyzed": 0, "streaks_found": 0, "streak_map": {}}

    streak_map = {}
    total_streaks = 0

    if on_progress:
        on_progress(5, "Analyzing frames for streaks...")

    prev_img = _load_image(valid[0])

    for i in range(1, len(valid)):
        curr_img = _load_image(valid[i])

        # Compute brightness difference
        if prev_img.shape == curr_img.shape:
            diff = np.abs(curr_img.mean(axis=2) - prev_img.mean(axis=2))
            bright_pixels = int(np.sum(diff > threshold))

            # Simple heuristic: if bright pixel count exceeds min_length
            # but is a small fraction of total, likely a streak
            total_pixels = diff.shape[0] * diff.shape[1]
            if bright_pixels > min_length and bright_pixels < total_pixels * 0.05:
                streak_map[valid[i]] = {
                    "bright_pixels": bright_pixels,
                    "fraction": round(bright_pixels / total_pixels, 6),
                }
                total_streaks += 1

        prev_img = curr_img

        if on_progress and i % 10 == 0:
            pct = min(int((i / len(valid)) * 90) + 5, 95)
            on_progress(pct, f"Analyzed {i}/{len(valid)} frames")

    if on_progress:
        on_progress(100, f"Streak analysis complete: {total_streaks} found")

    return {
        "frames_analyzed": len(valid),
        "streaks_found": total_streaks,
        "streak_map": streak_map,
    }


# ---------------------------------------------------------------------------
# Star trail compositing (lighten-mode stacking)
# ---------------------------------------------------------------------------

def composite_star_trails(
    image_paths: List[str],
    output_path: str,
    mode: str = "lighten",
    gap_fill: bool = True,
    skip_streaks: bool = False,
    streak_threshold: float = 50.0,
    on_progress: Optional[Callable] = None,
) -> StarTrailResult:
    """Composite star trails from a series of images.

    Uses lighten-mode blending (per-pixel maximum) to stack long-exposure
    frames into a single star trail image.

    Args:
        image_paths: List of input image paths (chronological order).
        output_path: Path for the output composite image.
        mode: Blending mode - 'lighten' (max), 'average', or 'additive'.
        gap_fill: Interpolate between frames to fill temporal gaps.
        skip_streaks: Skip frames detected as containing airplane streaks.
        streak_threshold: Threshold for streak detection.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        :class:`StarTrailResult` with composite details.
    """
    if not ensure_package("numpy"):
        raise RuntimeError("numpy required for star trail compositing")
    if not ensure_package("PIL", "Pillow"):
        raise RuntimeError("Pillow required for star trail compositing")

    import numpy as np

    result = StarTrailResult()
    start_time = time.time()

    valid = _collect_images(image_paths)
    if len(valid) < 2:
        raise ValueError("At least 2 valid images required for star trails")

    if on_progress:
        on_progress(5, f"Processing {len(valid)} frames...")

    # Optionally detect streaks
    streak_frames = set()
    if skip_streaks:
        streak_info = remove_streaks(valid, threshold=streak_threshold)
        streak_frames = set(streak_info.get("streak_map", {}).keys())
        result.streaks_removed = len(streak_frames)

    # Initialize composite with first non-streak frame
    composite = None
    frames_processed = 0
    prev_frame = None

    for i, img_path in enumerate(valid):
        if img_path in streak_frames:
            result.frames_skipped += 1
            continue

        frame = _load_image(img_path)

        if composite is None:
            composite = frame.copy()
            prev_frame = frame.copy()
            frames_processed += 1
            continue

        # Gap filling: blend intermediate frame if shapes match
        if gap_fill and prev_frame is not None and prev_frame.shape == frame.shape:
            mid = (prev_frame + frame) / 2.0
            if mode == "lighten":
                composite = np.maximum(composite, mid)
            elif mode == "additive":
                composite = np.minimum(composite + mid * 0.5, 255.0)

        # Apply blending
        if composite.shape == frame.shape:
            if mode == "lighten":
                composite = np.maximum(composite, frame)
            elif mode == "average":
                alpha = 1.0 / (frames_processed + 1)
                composite = composite * (1 - alpha) + frame * alpha
            elif mode == "additive":
                composite = np.minimum(composite + frame * (1.0 / len(valid)), 255.0)

        prev_frame = frame.copy()
        frames_processed += 1

        if on_progress and (i + 1) % 5 == 0:
            pct = min(int(((i + 1) / len(valid)) * 85) + 5, 90)
            on_progress(pct, f"Stacked {frames_processed}/{len(valid)} frames")

    if composite is None:
        raise ValueError("No valid frames could be processed")

    # Save output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_image(composite, output_path)

    result.output_path = output_path
    result.frames_processed = frames_processed
    result.status = "complete"
    result.duration_seconds = round(time.time() - start_time, 2)

    if on_progress:
        on_progress(100, f"Star trail complete: {frames_processed} frames stacked")

    return result


# ---------------------------------------------------------------------------
# Progressive trail animation
# ---------------------------------------------------------------------------

def create_trail_animation(
    image_paths: List[str],
    output_path: str,
    fps: float = 24.0,
    trail_length: int = 0,
    mode: str = "lighten",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Create a progressive star trail animation video.

    Each frame of the video shows the cumulative trail up to that point,
    creating a growing-trail effect.

    Args:
        image_paths: List of input image paths.
        output_path: Path for the output video.
        fps: Output video frame rate.
        trail_length: If > 0, use a sliding window of this many frames
            instead of full accumulation.
        mode: Blending mode - 'lighten' or 'additive'.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        Dict with ``output_path``, ``frames``, ``duration_seconds``.
    """
    if not ensure_package("numpy"):
        raise RuntimeError("numpy required for trail animation")
    if not ensure_package("PIL", "Pillow"):
        raise RuntimeError("Pillow required for trail animation")

    import tempfile

    import numpy as np

    valid = _collect_images(image_paths)
    if len(valid) < 2:
        raise ValueError("At least 2 valid images required for animation")

    if on_progress:
        on_progress(5, f"Creating trail animation from {len(valid)} frames...")

    # Create temp directory for intermediate frames
    temp_dir = tempfile.mkdtemp(prefix="opencut_trail_")
    start_time = time.time()

    try:
        composite = None
        window = []

        for i, img_path in enumerate(valid):
            frame = _load_image(img_path)

            if trail_length > 0:
                window.append(frame)
                if len(window) > trail_length:
                    window.pop(0)
                # Rebuild composite from window
                composite = window[0].copy()
                for wf in window[1:]:
                    if composite.shape == wf.shape:
                        if mode == "lighten":
                            composite = np.maximum(composite, wf)
                        else:
                            composite = np.minimum(composite + wf * (1.0 / len(window)), 255.0)
            else:
                if composite is None:
                    composite = frame.copy()
                elif composite.shape == frame.shape:
                    if mode == "lighten":
                        composite = np.maximum(composite, frame)
                    else:
                        composite = np.minimum(
                            composite + frame * (1.0 / len(valid)), 255.0
                        )

            # Save intermediate frame
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            _save_image(composite, frame_path)

            if on_progress and (i + 1) % 5 == 0:
                pct = min(int(((i + 1) / len(valid)) * 80) + 5, 85)
                on_progress(pct, f"Rendered {i + 1}/{len(valid)} frames")

        # Encode frames to video with FFmpeg
        if on_progress:
            on_progress(90, "Encoding video...")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        frame_pattern = os.path.join(temp_dir, "frame_%06d.png")

        cmd = (
            FFmpegCmd()
            .option("framerate", str(fps))
            .input(frame_pattern)
            .video_codec("libx264", crf=18, preset="medium")
            .faststart()
            .output(output_path)
            .build()
        )
        run_ffmpeg(cmd)

    finally:
        # Clean up temp frames
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    duration = round(time.time() - start_time, 2)

    if on_progress:
        on_progress(100, f"Trail animation complete: {len(valid)} frames")

    return {
        "output_path": output_path,
        "frames": len(valid),
        "duration_seconds": duration,
    }
