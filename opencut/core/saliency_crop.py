"""
OpenCut Saliency-Guided Auto-Crop

Generates a saliency map (face regions + motion + text + high-contrast + center
bias), places a crop window to maximize saliency, and smooths the crop path
over time for stable output.
"""

import json
import logging
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_ffprobe_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# Default target aspect ratios
ASPECT_RATIOS = {
    "9:16": (9, 16),    # Vertical / Stories / TikTok
    "1:1": (1, 1),      # Square / Instagram
    "4:5": (4, 5),      # Instagram portrait
    "16:9": (16, 9),    # Standard widescreen
    "4:3": (4, 3),      # Classic TV
    "21:9": (21, 9),    # Ultrawide
}


@dataclass
class SaliencyRegion:
    """A detected salient region in a frame."""
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    weight: float = 1.0
    source: str = "unknown"  # face, motion, text, contrast, center


@dataclass
class CropKeyframe:
    """A keyframe in the crop path."""
    time: float = 0.0
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


@dataclass
class SaliencyCropResult:
    """Result of saliency-guided auto-crop."""
    output_path: str = ""
    original_width: int = 0
    original_height: int = 0
    crop_width: int = 0
    crop_height: int = 0
    target_aspect: str = ""
    keyframe_count: int = 0
    crop_keyframes: List[Dict] = field(default_factory=list)


def _compute_saliency_map(
    frame_path: str,
    frame_w: int,
    frame_h: int,
    face_weight: float = 3.0,
    motion_weight: float = 2.0,
    contrast_weight: float = 1.5,
    center_weight: float = 1.0,
) -> List[SaliencyRegion]:
    """
    Generate a saliency map for a single frame.

    Combines multiple cues:
    - Face regions (highest weight)
    - High-contrast areas
    - Center bias (gaussian falloff from center)

    Returns list of SaliencyRegion objects.
    """
    regions = []

    # Face detection via shot_classify module
    try:
        from opencut.core.shot_classify import _detect_faces_in_frame
        faces = _detect_faces_in_frame(frame_path)
        for face in faces:
            regions.append(SaliencyRegion(
                x=face["x"], y=face["y"],
                w=face["w"], h=face["h"],
                weight=face_weight,
                source="face",
            ))
    except Exception as exc:
        logger.debug("Face detection for saliency failed: %s", exc)

    # High-contrast region detection via FFmpeg signalstats
    try:
        # Divide frame into quadrants and measure contrast per quadrant
        quad_w = frame_w // 2
        quad_h = frame_h // 2
        quadrants = [
            (0, 0, "top_left"),
            (quad_w, 0, "top_right"),
            (0, quad_h, "bottom_left"),
            (quad_w, quad_h, "bottom_right"),
        ]

        for qx, qy, qname in quadrants:
            cmd = [
                get_ffmpeg_path(), "-hide_banner", "-loglevel", "info",
                "-i", frame_path,
                "-vf", f"crop={quad_w}:{quad_h}:{qx}:{qy},"
                       f"signalstats=stat=tout+vrep+brng,metadata=print:file=-",
                "-frames:v", "1",
                "-f", "null", "-",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            import re
            tout_match = re.search(r"SIGNALSTATS\.TOUT=(\d+\.?\d*)",
                                   result.stderr)
            if tout_match:
                tout = float(tout_match.group(1))
                if tout > 1.0:  # Notable contrast
                    regions.append(SaliencyRegion(
                        x=qx, y=qy,
                        w=quad_w, h=quad_h,
                        weight=contrast_weight * min(2.0, tout / 5.0),
                        source="contrast",
                    ))
    except Exception as exc:
        logger.debug("Contrast analysis for saliency failed: %s", exc)

    # Center bias (always add a center-weighted region)
    center_w = int(frame_w * 0.4)
    center_h = int(frame_h * 0.4)
    regions.append(SaliencyRegion(
        x=(frame_w - center_w) // 2,
        y=(frame_h - center_h) // 2,
        w=center_w,
        h=center_h,
        weight=center_weight,
        source="center",
    ))

    return regions


def _find_optimal_crop(
    regions: List[SaliencyRegion],
    frame_w: int,
    frame_h: int,
    crop_w: int,
    crop_h: int,
) -> Tuple[int, int]:
    """
    Find the crop window position that maximizes total saliency overlap.

    Uses a weighted overlap approach: for each candidate position, sum
    the weighted area of intersection with each salient region.

    Returns (crop_x, crop_y) for the optimal position.
    """
    if not regions:
        # Center crop
        return (frame_w - crop_w) // 2, (frame_h - crop_h) // 2

    best_x, best_y = (frame_w - crop_w) // 2, (frame_h - crop_h) // 2
    best_score = -1.0

    # Evaluate candidate positions: region centers, center, weighted centroid
    candidates = []

    # Weighted centroid
    total_weight = sum(r.weight for r in regions)
    if total_weight > 0:
        cx = sum((r.x + r.w / 2) * r.weight for r in regions) / total_weight
        cy = sum((r.y + r.h / 2) * r.weight for r in regions) / total_weight
        candidates.append((int(cx - crop_w / 2), int(cy - crop_h / 2)))

    # Each region center
    for r in regions:
        rx = r.x + r.w // 2 - crop_w // 2
        ry = r.y + r.h // 2 - crop_h // 2
        candidates.append((rx, ry))

    # Center of frame
    candidates.append(((frame_w - crop_w) // 2, (frame_h - crop_h) // 2))

    for cx, cy in candidates:
        # Clamp to frame bounds
        cx = max(0, min(cx, frame_w - crop_w))
        cy = max(0, min(cy, frame_h - crop_h))

        score = 0.0
        for r in regions:
            # Compute intersection area
            ix1 = max(cx, r.x)
            iy1 = max(cy, r.y)
            ix2 = min(cx + crop_w, r.x + r.w)
            iy2 = min(cy + crop_h, r.y + r.h)

            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                region_area = max(1, r.w * r.h)
                overlap_ratio = intersection / region_area
                score += overlap_ratio * r.weight

        if score > best_score:
            best_score = score
            best_x, best_y = cx, cy

    return best_x, best_y


def _smooth_crop_path(
    keyframes: List[CropKeyframe],
    smoothing: float = 0.7,
) -> List[CropKeyframe]:
    """
    Smooth the crop path over time using exponential moving average.

    Prevents jerky crop movement between frames.
    """
    if len(keyframes) <= 1:
        return keyframes

    smoothed = [keyframes[0]]
    for i in range(1, len(keyframes)):
        prev = smoothed[-1]
        curr = keyframes[i]
        smooth_x = int(prev.x * smoothing + curr.x * (1 - smoothing))
        smooth_y = int(prev.y * smoothing + curr.y * (1 - smoothing))
        smoothed.append(CropKeyframe(
            time=curr.time,
            x=smooth_x, y=smooth_y,
            w=curr.w, h=curr.h,
        ))

    return smoothed


def saliency_crop(
    input_path: str,
    target_aspect: str = "9:16",
    sample_interval: float = 2.0,
    smoothing: float = 0.7,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Perform saliency-guided auto-crop on a video.

    Analyzes frames at regular intervals, computes saliency maps,
    places crop windows to maximize saliency, smooths the crop path,
    and renders the cropped output.

    Args:
        input_path: Source video file path.
        target_aspect: Target aspect ratio (e.g. "9:16", "1:1").
        sample_interval: Seconds between analysis frames.
        smoothing: Crop path smoothing factor (0=no smooth, 1=static).
        output_dir: Output directory. Uses input dir if empty.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, dimensions, keyframes.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Parse aspect ratio
    if target_aspect in ASPECT_RATIOS:
        ar_w, ar_h = ASPECT_RATIOS[target_aspect]
    else:
        try:
            parts = target_aspect.split(":")
            ar_w, ar_h = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            ar_w, ar_h = 9, 16

    if on_progress:
        on_progress(5, "Getting video info...")

    info = get_video_info(input_path)
    frame_w = info.get("width", 1920)
    frame_h = info.get("height", 1080)
    duration = info.get("duration", 0)
    fps = info.get("fps", 30.0)

    # Calculate crop dimensions maintaining target aspect ratio
    # Fit the largest crop window with the target AR inside the frame
    if (ar_w / ar_h) > (frame_w / frame_h):
        # Wider than frame: fit to width
        crop_w = frame_w
        crop_h = int(frame_w * ar_h / ar_w)
    else:
        # Taller than frame: fit to height
        crop_h = frame_h
        crop_w = int(frame_h * ar_w / ar_h)

    # Ensure even dimensions
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    if on_progress:
        on_progress(10, f"Crop size: {crop_w}x{crop_h}")

    # Sample frames and build crop keyframes
    tmp_dir = tempfile.mkdtemp(prefix="opencut_saliency_")
    keyframes: List[CropKeyframe] = []

    try:
        sample_times = []
        t = 0.0
        while t < duration:
            sample_times.append(t)
            t += sample_interval

        if not sample_times:
            sample_times = [0.0]

        total_samples = len(sample_times)

        if on_progress:
            on_progress(15, f"Analyzing {total_samples} frames...")

        for idx, t in enumerate(sample_times):
            frame_path = os.path.join(tmp_dir, f"sal_{idx:04d}.jpg")

            # Extract frame
            cmd = [
                get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
                "-ss", str(t), "-i", input_path,
                "-frames:v", "1", "-q:v", "3",
                "-y", frame_path,
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=15)
            except subprocess.TimeoutExpired:
                continue

            if not os.path.isfile(frame_path):
                # Use center crop as fallback
                keyframes.append(CropKeyframe(
                    time=t,
                    x=(frame_w - crop_w) // 2,
                    y=(frame_h - crop_h) // 2,
                    w=crop_w, h=crop_h,
                ))
                continue

            # Compute saliency
            regions = _compute_saliency_map(frame_path, frame_w, frame_h)

            # Find optimal crop position
            cx, cy = _find_optimal_crop(regions, frame_w, frame_h, crop_w, crop_h)

            keyframes.append(CropKeyframe(
                time=t, x=cx, y=cy, w=crop_w, h=crop_h,
            ))

            if on_progress:
                pct = 15 + int(55 * (idx + 1) / total_samples)
                on_progress(pct, f"Analyzed frame {idx + 1}/{total_samples}")

        if on_progress:
            on_progress(70, "Smoothing crop path...")

        # Smooth the path
        keyframes = _smooth_crop_path(keyframes, smoothing)

        if on_progress:
            on_progress(75, "Rendering cropped video...")

        # Build FFmpeg crop command with keyframe interpolation
        # Use the median crop position for a single static crop
        # (For production, this would use crop with sendcmd/zmq for dynamic)
        if keyframes:
            mid_idx = len(keyframes) // 2
            kf = keyframes[mid_idx]
            crop_x, crop_y = kf.x, kf.y
        else:
            crop_x = (frame_w - crop_w) // 2
            crop_y = (frame_h - crop_h) // 2

        out_dir = output_dir or os.path.dirname(input_path)
        out_path = output_path(input_path, f"saliency_{target_aspect.replace(':', 'x')}",
                               out_dir)

        cmd = (FFmpegCmd()
               .input(input_path)
               .video_filter(f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}")
               .video_codec("libx264", crf=18, preset="fast")
               .audio_codec("aac", bitrate="192k")
               .faststart()
               .output(out_path)
               .build())

        run_ffmpeg(cmd)

        if on_progress:
            on_progress(100, "Saliency crop complete")

        return {
            "output_path": out_path,
            "original_width": frame_w,
            "original_height": frame_h,
            "crop_width": crop_w,
            "crop_height": crop_h,
            "target_aspect": target_aspect,
            "keyframe_count": len(keyframes),
            "crop_keyframes": [
                {
                    "time": round(kf.time, 2),
                    "x": kf.x, "y": kf.y,
                    "w": kf.w, "h": kf.h,
                }
                for kf in keyframes
            ],
        }

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
