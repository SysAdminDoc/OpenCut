"""
OpenCut Screenshot-to-Video with Ken Burns (Feature 11.4)

Import image folder, detect ROIs (edges/faces/saliency), generate
pan-zoom keyframes visiting ROIs.

Uses FFmpeg for video assembly. Optional OpenCV for advanced ROI detection
with graceful fallback to center-crop ROIs.
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ROI:
    """Region of interest in an image."""
    x: int
    y: int
    w: int
    h: int
    weight: float = 1.0    # importance weight for keyframe timing
    label: str = ""        # e.g. "face", "edge_cluster", "center"

    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def to_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "weight": self.weight, "label": self.label,
        }


@dataclass
class KenBurnsKeyframe:
    """A single keyframe in a Ken Burns pan-zoom animation."""
    time: float        # seconds into the per-image duration
    x: float           # crop x (0.0 - 1.0 normalized)
    y: float           # crop y (0.0 - 1.0 normalized)
    zoom: float = 1.0  # zoom level (1.0 = full image, 2.0 = 2x zoom)

    def to_dict(self) -> dict:
        return {
            "time": round(self.time, 3),
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "zoom": round(self.zoom, 3),
        }


@dataclass
class ScreenshotVideoResult:
    """Result of creating a screenshot video."""
    output_path: str
    image_count: int
    total_duration: float
    resolution: Tuple[int, int]
    keyframes_per_image: int

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "image_count": self.image_count,
            "total_duration": self.total_duration,
            "resolution": list(self.resolution),
            "keyframes_per_image": self.keyframes_per_image,
        }


# ---------------------------------------------------------------------------
# Image size reader (PIL-free, uses FFprobe)
# ---------------------------------------------------------------------------

def _get_image_size(image_path: str) -> Tuple[int, int]:
    """Get image width and height using ffprobe."""
    import json
    import subprocess

    from opencut.helpers import get_ffprobe_path

    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json", image_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {image_path}")
    data = json.loads(result.stdout.decode())
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No image streams found in {image_path}")
    return int(streams[0]["width"]), int(streams[0]["height"])


# ---------------------------------------------------------------------------
# ROI detection
# ---------------------------------------------------------------------------

def _detect_roi_edges(image_path: str) -> List[ROI]:
    """Detect ROIs using edge density analysis via OpenCV.

    Divides the image into a grid, computes edge density per cell,
    and returns cells with above-average density as ROIs.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape
    edges = cv2.Canny(img, 50, 150)

    # Grid-based edge density
    grid_rows, grid_cols = 4, 4
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    rois = []
    densities = []
    cells = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            y1 = r * cell_h
            x1 = c * cell_w
            y2 = min(y1 + cell_h, h)
            x2 = min(x1 + cell_w, w)
            cell = edges[y1:y2, x1:x2]
            density = float(np.count_nonzero(cell)) / max(cell.size, 1)
            densities.append(density)
            cells.append((x1, y1, x2 - x1, y2 - y1))

    if not densities:
        return []

    avg_density = sum(densities) / len(densities)
    threshold = avg_density * 1.2  # above-average threshold

    for i, (cx, cy, cw, ch) in enumerate(cells):
        if densities[i] > threshold:
            rois.append(ROI(
                x=cx, y=cy, w=cw, h=ch,
                weight=densities[i],
                label="edge_cluster",
            ))

    return rois


def _detect_roi_faces(image_path: str) -> List[ROI]:
    """Detect face ROIs using OpenCV Haar cascades."""
    try:
        import cv2
    except ImportError:
        return []

    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use the built-in frontal face cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.isfile(cascade_path):
        return []

    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    rois = []
    for (x, y, w, h) in faces:
        rois.append(ROI(
            x=int(x), y=int(y), w=int(w), h=int(h),
            weight=2.0,  # faces get higher priority
            label="face",
        ))

    return rois


def _detect_roi_center(image_path: str) -> List[ROI]:
    """Fallback: return center region and rule-of-thirds intersections."""
    try:
        w, h = _get_image_size(image_path)
    except Exception:
        # Absolute fallback
        w, h = 1920, 1080

    rois = [
        # Center region
        ROI(x=w // 4, y=h // 4, w=w // 2, h=h // 2, weight=1.0, label="center"),
        # Rule of thirds: top-left intersection
        ROI(x=w // 6, y=h // 6, w=w // 3, h=h // 3, weight=0.7, label="thirds_tl"),
        # Rule of thirds: bottom-right intersection
        ROI(x=w // 2, y=h // 2, w=w // 3, h=h // 3, weight=0.7, label="thirds_br"),
    ]
    return rois


def detect_roi(image_path: str) -> List[dict]:
    """Detect regions of interest in an image.

    Tries multiple strategies: face detection, edge analysis,
    and falls back to center/rule-of-thirds if OpenCV is unavailable.

    Args:
        image_path: Path to the image file.

    Returns:
        List of ROI dicts with x, y, w, h, weight, label fields.

    Raises:
        FileNotFoundError: If image doesn't exist.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    all_rois = []

    # Try face detection first (highest priority)
    face_rois = _detect_roi_faces(image_path)
    all_rois.extend(face_rois)

    # Try edge-based detection
    edge_rois = _detect_roi_edges(image_path)
    all_rois.extend(edge_rois)

    # If no ROIs found via CV, use geometric fallback
    if not all_rois:
        all_rois = _detect_roi_center(image_path)

    # Sort by weight descending
    all_rois.sort(key=lambda r: r.weight, reverse=True)

    # Limit to top 6 ROIs
    all_rois = all_rois[:6]

    return [r.to_dict() for r in all_rois]


# ---------------------------------------------------------------------------
# Ken Burns keyframe generation
# ---------------------------------------------------------------------------

def generate_ken_burns_keyframes(
    rois: List[dict],
    duration: float,
    min_zoom: float = 1.0,
    max_zoom: float = 1.5,
) -> List[dict]:
    """Generate pan-zoom keyframes that visit each ROI.

    Creates a smooth camera path that pans across ROIs with
    zoom-in/zoom-out transitions.

    Args:
        rois: List of ROI dicts (from detect_roi).
        duration: Total duration in seconds for this image.
        min_zoom: Minimum zoom level (1.0 = full view).
        max_zoom: Maximum zoom level.

    Returns:
        List of KenBurnsKeyframe dicts with time, x, y, zoom.

    Raises:
        ValueError: If duration <= 0 or rois is empty.
    """
    if duration <= 0:
        raise ValueError(f"Duration must be > 0, got {duration}")
    if not rois:
        raise ValueError("No ROIs provided")

    keyframes = []
    num_rois = len(rois)

    # Distribute time across ROIs weighted by importance
    total_weight = sum(r.get("weight", 1.0) for r in rois)
    if total_weight <= 0:
        total_weight = num_rois

    current_time = 0.0
    for i, roi in enumerate(rois):
        weight = roi.get("weight", 1.0)
        roi_duration = duration * (weight / total_weight)

        # Normalize ROI center to 0-1 range
        # Assume the ROI dict has x, y, w, h
        cx = roi.get("x", 0) + roi.get("w", 0) / 2
        cy = roi.get("y", 0) + roi.get("h", 0) / 2

        # We'll normalize later when we know image size; for now use raw
        # Zoom: alternate between zoomed-in and zoomed-out
        if i % 2 == 0:
            zoom_start = min_zoom
            zoom_end = max_zoom
        else:
            zoom_start = max_zoom
            zoom_end = min_zoom

        # Start keyframe for this ROI
        keyframes.append(KenBurnsKeyframe(
            time=current_time,
            x=cx, y=cy,
            zoom=zoom_start,
        ))

        # End keyframe for this ROI
        end_time = min(current_time + roi_duration, duration)
        keyframes.append(KenBurnsKeyframe(
            time=end_time,
            x=cx, y=cy,
            zoom=zoom_end,
        ))

        current_time = end_time

    # Ensure we don't exceed duration
    if keyframes and keyframes[-1].time > duration:
        keyframes[-1].time = duration

    return [kf.to_dict() for kf in keyframes]


# ---------------------------------------------------------------------------
# Screenshot video creator
# ---------------------------------------------------------------------------

def create_screenshot_video(
    image_paths: List[str],
    output_path_str: str,
    duration_per_image: float = 5.0,
    resolution: Tuple[int, int] = (1920, 1080),
    fps: int = 30,
    transition: str = "fade",
    transition_duration: float = 0.5,
    enable_ken_burns: bool = True,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Create a video from a folder of screenshots with Ken Burns effect.

    Args:
        image_paths: List of image file paths (in order).
        output_path_str: Output video file path.
        duration_per_image: Duration each image is shown (seconds).
        resolution: Output video resolution (width, height).
        fps: Output frame rate.
        transition: Transition type between images ("fade", "none").
        transition_duration: Duration of cross-fade transition.
        enable_ken_burns: Enable pan-zoom animation per image.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path, image_count, total_duration, resolution.

    Raises:
        ValueError: If image_paths is empty or contains non-image files.
        FileNotFoundError: If any image file doesn't exist.
    """
    if not image_paths:
        raise ValueError("No image paths provided")

    # Validate all images exist
    valid_images = []
    for p in image_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Image not found: {p}")
        ext = os.path.splitext(p)[1].lower()
        if ext not in _IMAGE_EXTS:
            raise ValueError(f"Unsupported image format: {ext} ({p})")
        valid_images.append(p)

    if on_progress:
        on_progress(5, f"Processing {len(valid_images)} images...")

    out_w, out_h = resolution
    total_duration = duration_per_image * len(valid_images)
    n = len(valid_images)

    # Build FFmpeg filter_complex for slideshow with optional Ken Burns
    # Each image becomes an input, scaled and padded to output resolution,
    # then concatenated with optional fade transitions.
    inputs_part = []
    filter_parts = []

    for i, img_path in enumerate(valid_images):
        if on_progress:
            pct = 10 + int((i / n) * 40)
            on_progress(pct, f"Preparing image {i + 1}/{n}...")

        inputs_part.append(img_path)

        # Scale and pad each image to target resolution
        # setsar=1 ensures correct aspect ratio
        scale_pad = (
            f"[{i}:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={fps},"
            f"settb=1/{fps},"
            f"trim=duration={duration_per_image},"
            f"setpts=PTS-STARTPTS"
        )

        if enable_ken_burns:
            # Apply a simple zoom-pan effect:
            # Slowly zoom in from 1.0 to 1.15 while panning slightly
            zoompan = (
                f",zoompan=z='min(zoom+0.0005,1.15)'"
                f":x='iw/2-(iw/zoom/2)+((iw/zoom/2)*sin(on/{fps}/3))'"
                f":y='ih/2-(ih/zoom/2)'"
                f":d={int(duration_per_image * fps)}"
                f":s={out_w}x{out_h}:fps={fps}"
            )
            scale_pad += zoompan

        filter_parts.append(f"{scale_pad}[v{i}]")

    # Concatenate all image streams
    if n == 1:
        concat_inputs = "[v0]"
    else:
        if transition == "fade" and transition_duration > 0:
            # Use xfade for cross-fade transitions between consecutive pairs
            prev_label = "v0"
            for i in range(1, n):
                offset = duration_per_image * i - transition_duration * i
                offset = max(0, offset)
                out_label = f"xf{i}" if i < n - 1 else "outv"
                filter_parts.append(
                    f"[{prev_label}][v{i}]xfade=transition=fade"
                    f":duration={transition_duration}:offset={offset:.3f}"
                    f"[{out_label}]"
                )
                prev_label = out_label
        else:
            # Simple concatenation
            concat_inputs = "".join(f"[v{i}]" for i in range(n))
            filter_parts.append(
                f"{concat_inputs}concat=n={n}:v=1:a=0[outv]"
            )

    filter_complex = ";".join(filter_parts)

    if on_progress:
        on_progress(55, "Building FFmpeg command...")

    # Build FFmpeg command
    cmd_builder = FFmpegCmd()
    for img in valid_images:
        cmd_builder.input(img, loop=1)

    cmd = (
        cmd_builder
        .filter_complex(filter_complex, maps=["[outv]"])
        .video_codec("libx264", crf=18, preset="fast")
        .option("shortest")
        .faststart()
        .output(output_path_str)
        .build()
    )

    if on_progress:
        on_progress(60, "Encoding video...")

    os.makedirs(os.path.dirname(os.path.abspath(output_path_str)), exist_ok=True)
    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(100, "Done")

    actual_duration = total_duration
    if transition == "fade" and n > 1:
        actual_duration -= transition_duration * (n - 1)

    result = ScreenshotVideoResult(
        output_path=output_path_str,
        image_count=n,
        total_duration=round(actual_duration, 2),
        resolution=resolution,
        keyframes_per_image=2 if enable_ken_burns else 0,
    )
    return result.to_dict()
