"""
OpenCut Vertical-First Intelligent Reframe Module

Smart reframing of horizontal video for vertical/portrait formats:
- Center crop: simple center extraction
- Face-aware: crop centered on detected face position
- Auto: face detection with fallback to motion-center
- Split: wide shots stacked as two halves

All via FFmpeg - OpenCV optional for face tracking.
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


@dataclass
class ReframeResult:
    """Result of video reframing."""
    output_path: str = ""
    method: str = "center"
    source_aspect: str = ""
    target_aspect: str = "9:16"
    width: int = 0
    height: int = 0
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Aspect Ratio Parsing
# ---------------------------------------------------------------------------

def _parse_aspect(aspect: str) -> Tuple[int, int]:
    """Parse aspect ratio string like '9:16' into (width_ratio, height_ratio)."""
    parts = aspect.split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return 9, 16


def _calc_crop_dims(
    src_w: int, src_h: int, target_w_ratio: int, target_h_ratio: int,
) -> Tuple[int, int]:
    """Calculate crop dimensions to achieve target aspect ratio.

    Returns (crop_w, crop_h) that fits within source dimensions.
    """
    target_ratio = target_w_ratio / target_h_ratio
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # Source is wider than target - crop width
        crop_h = src_h
        crop_w = int(src_h * target_ratio)
    else:
        # Source is taller - crop height
        crop_w = src_w
        crop_h = int(src_w / target_ratio)

    # Ensure even dimensions (required by most codecs)
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    return max(2, crop_w), max(2, crop_h)


# ---------------------------------------------------------------------------
# Reframe Methods
# ---------------------------------------------------------------------------

def _reframe_center(
    input_path: str, info: dict,
    crop_w: int, crop_h: int,
    output_path_str: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Simple center crop reframe."""
    src_w, src_h = info["width"], info["height"]
    crop_x = (src_w - crop_w) // 2
    crop_y = (src_h - crop_h) // 2

    if on_progress:
        on_progress(20, f"Applying center crop {crop_w}x{crop_h}...")

    vf = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
    cmd = [
        get_ffmpeg_path(), "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        "-y", output_path_str,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Center crop complete")

    return {
        "output_path": output_path_str,
        "method": "center",
        "width": crop_w,
        "height": crop_h,
        "duration": info["duration"],
    }


def _reframe_split(
    input_path: str, info: dict,
    crop_w: int, crop_h: int,
    output_path_str: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Split wide shot into two stacked halves for vertical format."""
    src_w, src_h = info["width"], info["height"]

    if on_progress:
        on_progress(20, "Splitting wide shot into stacked halves...")

    # Split horizontally into left and right halves, stack vertically
    half_w = src_w // 2
    # Each half cropped to target aspect
    half_target_h = int(half_w / (crop_w / crop_h))
    half_crop_h = min(half_target_h, src_h)
    half_crop_y = (src_h - half_crop_h) // 2

    # Ensure even dimensions
    half_w = half_w - (half_w % 2)
    half_crop_h = half_crop_h - (half_crop_h % 2)

    # Use filter_complex to split and stack
    filter_complex = (
        f"[0:v]split[left][right];"
        f"[left]crop={half_w}:{half_crop_h}:0:{half_crop_y}[ltop];"
        f"[right]crop={half_w}:{half_crop_h}:{half_w}:{half_crop_y}[lbot];"
        f"[ltop][lbot]vstack[out]"
    )

    cmd = [
        get_ffmpeg_path(), "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        "-y", output_path_str,
    ]
    run_ffmpeg(cmd)

    # Get actual output dimensions
    out_info = get_video_info(output_path_str)

    if on_progress:
        on_progress(100, "Split reframe complete")

    return {
        "output_path": output_path_str,
        "method": "split",
        "width": out_info["width"],
        "height": out_info["height"],
        "duration": info["duration"],
    }


def _detect_face_positions(
    input_path: str, info: dict,
    sample_interval: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> List[dict]:
    """Detect face positions at regular intervals for tracking.

    Returns list of {time, x, y, w, h} dicts.
    """
    try:
        import cv2
    except ImportError:
        logger.debug("OpenCV not available for face tracking")
        return []

    cascade_path = None
    for path in [
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"),
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ]:
        if os.path.isfile(path):
            cascade_path = path
            break

    if cascade_path is None:
        return []

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return []

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return []

    fps = info["fps"]
    duration = info["duration"]
    sample_frames = max(1, int(sample_interval * fps))
    positions = []

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_frames == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                )

                t = frame_idx / fps
                if len(faces) > 0:
                    # Use largest face
                    largest = max(faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = largest
                    positions.append({
                        "time": t,
                        "x": int(fx + fw // 2),
                        "y": int(fy + fh // 2),
                        "w": int(fw),
                        "h": int(fh),
                    })

                if on_progress:
                    pct = 10 + int(30 * t / duration)
                    on_progress(pct, f"Tracking faces at {t:.1f}s...")

            frame_idx += 1
    finally:
        cap.release()

    return positions


def _detect_motion_center(
    input_path: str, info: dict,
    on_progress: Optional[Callable] = None,
) -> List[dict]:
    """Detect center of motion per scene using FFmpeg.

    Uses scene change detection + crop detection to find content center.
    Returns list of {time, x, y} dicts.
    """
    # Use FFmpeg cropdetect to find content region
    import subprocess as _sp

    cmd = [
        get_ffmpeg_path(), "-i", input_path,
        "-vf", "cropdetect=24:16:0",
        "-f", "null", "-",
    ]
    result = _sp.run(cmd, capture_output=True, timeout=300)
    stderr = result.stderr.decode(errors="replace")

    # Parse cropdetect output: "crop=W:H:X:Y"
    import re
    positions = []
    for match in re.finditer(r"crop=(\d+):(\d+):(\d+):(\d+)", stderr):
        cw, ch, cx, cy = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
        center_x = cx + cw // 2
        center_y = cy + ch // 2
        positions.append({"x": center_x, "y": center_y})

    if positions:
        # Average position as single reference point
        avg_x = sum(p["x"] for p in positions) // len(positions)
        avg_y = sum(p["y"] for p in positions) // len(positions)
        return [{"time": 0, "x": avg_x, "y": avg_y, "w": 0, "h": 0}]

    return []


def _smooth_positions(positions: List[dict], duration: float) -> List[dict]:
    """Smooth face/content tracking positions to avoid jerky crop movement.

    Applies simple moving average over nearby positions.
    """
    if len(positions) <= 1:
        return positions

    smoothed = []
    window = 3  # average over 3 samples

    for i in range(len(positions)):
        start = max(0, i - window // 2)
        end = min(len(positions), i + window // 2 + 1)
        subset = positions[start:end]

        avg_x = sum(p["x"] for p in subset) // len(subset)
        avg_y = sum(p["y"] for p in subset) // len(subset)

        smoothed.append({
            "time": positions[i]["time"],
            "x": avg_x,
            "y": avg_y,
            "w": positions[i].get("w", 0),
            "h": positions[i].get("h", 0),
        })

    return smoothed


def _reframe_face(
    input_path: str, info: dict,
    crop_w: int, crop_h: int,
    output_path_str: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Reframe centered on detected face with smooth tracking."""
    src_w, src_h = info["width"], info["height"]

    if on_progress:
        on_progress(10, "Detecting face positions...")

    positions = _detect_face_positions(input_path, info, sample_interval=0.5, on_progress=on_progress)

    if not positions:
        if on_progress:
            on_progress(40, "No faces found, falling back to center crop")
        return _reframe_center(input_path, info, crop_w, crop_h, output_path_str, on_progress)

    # Smooth positions
    positions = _smooth_positions(positions, info["duration"])

    if on_progress:
        on_progress(50, "Building face-tracking crop...")

    # For simplicity with FFmpeg, use average face position for static crop
    # (dynamic per-frame crop would require complex sendcmd or frame-by-frame processing)
    avg_x = sum(p["x"] for p in positions) // len(positions)
    avg_y = sum(p["y"] for p in positions) // len(positions)

    # Center crop on average face position
    crop_x = max(0, min(avg_x - crop_w // 2, src_w - crop_w))
    crop_y = max(0, min(avg_y - crop_h // 2, src_h - crop_h))

    vf = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
    cmd = [
        get_ffmpeg_path(), "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        "-y", output_path_str,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Face-tracked reframe complete")

    return {
        "output_path": output_path_str,
        "method": "face",
        "width": crop_w,
        "height": crop_h,
        "duration": info["duration"],
    }


def _reframe_auto(
    input_path: str, info: dict,
    crop_w: int, crop_h: int,
    output_path_str: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Auto reframe: try face detection, fall back to motion center."""
    src_w, src_h = info["width"], info["height"]

    if on_progress:
        on_progress(10, "Auto-detecting content focus...")

    # Try face detection first
    positions = _detect_face_positions(input_path, info, sample_interval=2.0, on_progress=on_progress)

    if positions:
        if on_progress:
            on_progress(40, f"Found {len(positions)} face samples, using face-aware crop")
        return _reframe_face(input_path, info, crop_w, crop_h, output_path_str, on_progress)

    # Fall back to motion/content center
    if on_progress:
        on_progress(40, "No faces found, analyzing content center...")

    motion = _detect_motion_center(input_path, info, on_progress)

    if motion:
        avg_x = motion[0]["x"]
        avg_y = motion[0]["y"]
        crop_x = max(0, min(avg_x - crop_w // 2, src_w - crop_w))
        crop_y = max(0, min(avg_y - crop_h // 2, src_h - crop_h))
    else:
        # Default to center
        crop_x = (src_w - crop_w) // 2
        crop_y = (src_h - crop_h) // 2

    if on_progress:
        on_progress(60, "Applying content-aware crop...")

    vf = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
    cmd = [
        get_ffmpeg_path(), "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        "-y", output_path_str,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Auto reframe complete")

    return {
        "output_path": output_path_str,
        "method": "auto",
        "width": crop_w,
        "height": crop_h,
        "duration": info["duration"],
    }


# ---------------------------------------------------------------------------
# Main Reframe Function
# ---------------------------------------------------------------------------

def reframe_vertical(
    input_path: str,
    target_aspect: str = "9:16",
    method: str = "auto",
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Reframe a horizontal video for vertical/portrait display.

    Args:
        input_path: Source video file.
        target_aspect: Target aspect ratio (e.g., "9:16", "4:5", "1:1").
        method: Reframe method:
            "center" - simple center crop
            "face" - crop centered on detected face
            "auto" - face if detected, otherwise content-aware center
            "split" - wide shots split into two stacked halves
        output_path_str: Output file path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, method, source_aspect, target_aspect,
        width, height, duration.
    """
    method = method if method in ("center", "face", "auto", "split") else "auto"

    if on_progress:
        on_progress(5, f"Preparing {method} reframe to {target_aspect}...")

    info = get_video_info(input_path)
    src_w, src_h = info["width"], info["height"]
    src_aspect = f"{src_w}:{src_h}"

    # Parse target aspect
    tw, th = _parse_aspect(target_aspect)
    crop_w, crop_h = _calc_crop_dims(src_w, src_h, tw, th)

    if output_path_str is None:
        aspect_tag = target_aspect.replace(":", "x")
        output_path_str = output_path(input_path, f"reframe_{aspect_tag}")

    if method == "center":
        result = _reframe_center(input_path, info, crop_w, crop_h, output_path_str, on_progress)
    elif method == "face":
        result = _reframe_face(input_path, info, crop_w, crop_h, output_path_str, on_progress)
    elif method == "split":
        result = _reframe_split(input_path, info, crop_w, crop_h, output_path_str, on_progress)
    else:  # auto
        result = _reframe_auto(input_path, info, crop_w, crop_h, output_path_str, on_progress)

    result["source_aspect"] = src_aspect
    result["target_aspect"] = target_aspect
    return result
