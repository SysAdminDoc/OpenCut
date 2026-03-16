"""
OpenCut Face Reframe Module

Auto-crop video to keep face centered for vertical/social media formats.
Uses MediaPipe face detection to track face positions across frames,
applies smoothing, and generates a dynamic crop via FFmpeg.

Uses: mediapipe (already a dependency via face_tools.py), opencv-python-headless, numpy
"""

import logging
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


@dataclass
class FaceTrack:
    """A face detection result for a single frame."""
    frame: int
    time: float
    cx: float  # normalized center x (0-1)
    cy: float  # normalized center y (0-1)
    w: float   # normalized width
    h: float   # normalized height
    confidence: float


def _ensure_package(pkg_name: str, pip_name: str = None, on_progress: Callable = None):
    """Import a package, installing it if missing."""
    try:
        __import__(pkg_name)
        return True
    except ImportError:
        pip_name = pip_name or pkg_name
        if on_progress:
            on_progress(5, f"Installing {pip_name}...")
        logger.info(f"Installing missing dependency: {pip_name}")
        from opencut.security import safe_pip_install
        safe_pip_install(pip_name)
        return True


def _run_ffmpeg(cmd: List[str], timeout: int = 3600) -> str:
    """Run FFmpeg command, return stderr."""
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        raise RuntimeError(f"FFmpeg error: {err[-500:]}")
    return result.stderr.decode(errors="replace")


def _get_video_info(filepath: str) -> Dict:
    """Get video metadata via ffprobe."""
    import json as _json
    cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json", filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    try:
        data = _json.loads(result.stdout.decode())
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})
        fps_parts = stream.get("r_frame_rate", "30/1").split("/")
        fps = (float(fps_parts[0]) / float(fps_parts[1])) if len(fps_parts) == 2 and float(fps_parts[1]) else 30.0
        duration = float(stream.get("duration", 0) or fmt.get("duration", 0) or 0)
        nb_frames = int(stream.get("nb_frames", 0) or 0)
        if nb_frames == 0 and duration > 0:
            nb_frames = int(duration * fps)
        return {
            "width": int(stream.get("width", 1920)),
            "height": int(stream.get("height", 1080)),
            "fps": fps,
            "duration": duration,
            "total_frames": nb_frames,
        }
    except Exception:
        return {"width": 1920, "height": 1080, "fps": 30.0, "duration": 0.0, "total_frames": 0}


def _detect_faces_in_frames(video_path: str, sample_rate: int,
                            on_progress: Optional[Callable] = None) -> List[FaceTrack]:
    """Open video with cv2, sample every Nth frame, run MediaPipe face detection."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tracks = []

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)

                if results.detections:
                    # Pick the largest / most confident face
                    best = None
                    best_area = 0.0
                    for det in results.detections:
                        bb = det.location_data.relative_bounding_box
                        area = bb.width * bb.height
                        if area > best_area:
                            best_area = area
                            best = det
                            best_conf = det.score[0] if det.score else 0.5

                    if best is not None:
                        bb = best.location_data.relative_bounding_box
                        cx = bb.xmin + bb.width / 2.0
                        cy = bb.ymin + bb.height / 2.0
                        tracks.append(FaceTrack(
                            frame=frame_idx,
                            time=frame_idx / fps,
                            cx=max(0.0, min(1.0, cx)),
                            cy=max(0.0, min(1.0, cy)),
                            w=bb.width,
                            h=bb.height,
                            confidence=best_conf,
                        ))

                if on_progress and total > 0:
                    pct = 5 + int((frame_idx / total) * 35)
                    if frame_idx % (sample_rate * 10) == 0:
                        on_progress(pct, f"Analyzing frame {frame_idx}/{total}...")

            frame_idx += 1
    finally:
        cap.release()
        detector.close()

    return tracks


def _smooth_tracks(tracks: List[FaceTrack], smoothing: float,
                   total_frames: int, fps: float) -> List[Tuple[float, float]]:
    """
    Interpolate positions for all frames and apply exponential moving average.

    Returns list of (cx, cy) for every frame from 0..total_frames-1.
    """
    if not tracks:
        return [(0.5, 0.5)] * total_frames

    # Build a per-frame position array by interpolating between detected frames
    positions = [None] * total_frames

    # Place detected positions
    for t in tracks:
        if 0 <= t.frame < total_frames:
            positions[t.frame] = (t.cx, t.cy)

    # Fill gaps via linear interpolation between nearest detected frames
    detected_indices = [t.frame for t in tracks if 0 <= t.frame < total_frames]
    if not detected_indices:
        return [(0.5, 0.5)] * total_frames

    # Fill before first detection
    first_pos = positions[detected_indices[0]]
    for i in range(detected_indices[0]):
        positions[i] = first_pos

    # Fill after last detection
    last_pos = positions[detected_indices[-1]]
    for i in range(detected_indices[-1] + 1, total_frames):
        positions[i] = last_pos

    # Interpolate between detected frames
    for idx in range(len(detected_indices) - 1):
        f_a = detected_indices[idx]
        f_b = detected_indices[idx + 1]
        pos_a = positions[f_a]
        pos_b = positions[f_b]
        span = f_b - f_a
        if span <= 1:
            continue
        for f in range(f_a + 1, f_b):
            t = (f - f_a) / span
            positions[f] = (
                pos_a[0] + t * (pos_b[0] - pos_a[0]),
                pos_a[1] + t * (pos_b[1] - pos_a[1]),
            )

    # Ensure no None values remain
    for i in range(total_frames):
        if positions[i] is None:
            positions[i] = (0.5, 0.5)

    # Apply exponential moving average smoothing
    smoothed = list(positions)
    for i in range(1, total_frames):
        sx = smoothing * smoothed[i - 1][0] + (1 - smoothing) * positions[i][0]
        sy = smoothing * smoothed[i - 1][1] + (1 - smoothing) * positions[i][1]
        smoothed[i] = (sx, sy)

    return smoothed


def _build_crop_expression(positions: List[Tuple[float, float]],
                           src_w: int, src_h: int,
                           crop_w: int, crop_h: int,
                           fps: float) -> Tuple[str, str]:
    """
    Build FFmpeg crop x/y expressions from per-frame positions.

    Groups frames into 1-second windows, averages position per window,
    and builds nested if(between(t,...),pos,...) expressions.
    """
    if not positions:
        cx = max(0, (src_w - crop_w) // 2)
        cy = max(0, (src_h - crop_h) // 2)
        return str(cx), str(cy)

    total_frames = len(positions)
    duration = total_frames / fps
    window_sec = 1.0

    # Group into windows and average positions
    windows = []
    t = 0.0
    while t < duration:
        t_end = min(t + window_sec, duration)
        f_start = int(t * fps)
        f_end = min(int(t_end * fps), total_frames)
        if f_end <= f_start:
            f_end = f_start + 1

        avg_cx = sum(positions[f][0] for f in range(f_start, min(f_end, total_frames))) / max(1, min(f_end, total_frames) - f_start)
        avg_cy = sum(positions[f][1] for f in range(f_start, min(f_end, total_frames))) / max(1, min(f_end, total_frames) - f_start)

        # Convert normalized position to pixel offset for crop
        px = int(avg_cx * src_w - crop_w / 2)
        py = int(avg_cy * src_h - crop_h / 2)
        px = max(0, min(px, src_w - crop_w))
        py = max(0, min(py, src_h - crop_h))

        windows.append((round(t, 3), round(t_end, 3), px, py))
        t = t_end

    if len(windows) == 1:
        return str(windows[0][2]), str(windows[0][3])

    # Build nested if(between(t,...),val,...) expression iteratively
    # FFmpeg expression: if(between(t,0,1),100,if(between(t,1,2),150,...))
    def _build_nested(windows_list, coord_idx):
        # Start from the last window (innermost expression / default value)
        expr = str(windows_list[-1][coord_idx])
        # Work backwards to build the nested if() chain
        for i in range(len(windows_list) - 2, -1, -1):
            w = windows_list[i]
            val = w[coord_idx]
            expr = f"if(between(t\\,{w[0]}\\,{w[1]})\\,{val}\\,{expr})"
        return expr

    x_expr = _build_nested(windows, 2)
    y_expr = _build_nested(windows, 3)
    return x_expr, y_expr


def face_reframe(input_path: str, target_w: int = 1080, target_h: int = 1920,
                 smoothing: float = 0.85, face_padding: float = 0.3,
                 output_path: Optional[str] = None, output_dir: str = "",
                 sample_rate: int = 5, on_progress: Optional[Callable] = None) -> str:
    """
    Reframe video to keep face centered using MediaPipe tracking.

    Args:
        input_path: Source video path.
        target_w: Output width (default 1080 for vertical).
        target_h: Output height (default 1920 for vertical).
        smoothing: Smoothing factor 0-1 (higher = smoother pan).
        face_padding: Extra padding around face (0-1, fraction of face size).
        output_path: Explicit output path.
        output_dir: Output directory.
        sample_rate: Analyze every Nth frame (higher = faster but less smooth).
        on_progress: Progress callback(pct, msg).

    Returns:
        Output file path.
    """
    _ensure_package("cv2", "opencv-python-headless", on_progress)
    _ensure_package("mediapipe", "mediapipe", on_progress)
    _ensure_package("numpy", "numpy", on_progress)

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1] or ".mp4"
        directory = output_dir or os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_reframed{ext}")

    if on_progress:
        on_progress(5, "Analyzing face positions...")

    # Get source video info
    info = _get_video_info(input_path)
    src_w = info["width"]
    src_h = info["height"]
    fps = info["fps"]
    total_frames = info["total_frames"]
    duration = info["duration"]

    if total_frames <= 0:
        total_frames = max(1, int(duration * fps))

    # Detect faces in sampled frames
    tracks = _detect_faces_in_frames(input_path, sample_rate, on_progress)

    if on_progress:
        on_progress(40, "Smoothing camera path...")

    # Calculate crop dimensions to match target aspect ratio
    target_aspect = target_w / target_h
    src_aspect = src_w / src_h

    if src_aspect > target_aspect:
        # Source is wider than target: crop width
        crop_h = src_h
        crop_w = int(src_h * target_aspect)
    else:
        # Source is taller than target: crop height
        crop_w = src_w
        crop_h = int(src_w / target_aspect)

    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)

    if not tracks:
        # No faces detected: center crop
        logger.warning("No faces detected, using center crop")
        cx = max(0, (src_w - crop_w) // 2)
        cy = max(0, (src_h - crop_h) // 2)
        x_expr = str(cx)
        y_expr = str(cy)
    else:
        # Smooth and interpolate face positions
        positions = _smooth_tracks(tracks, smoothing, total_frames, fps)

        if on_progress:
            on_progress(50, "Building crop path...")

        # Build FFmpeg crop expressions
        x_expr, y_expr = _build_crop_expression(positions, src_w, src_h, crop_w, crop_h, fps)

    if on_progress:
        on_progress(60, "Rendering reframed video...")

    # Build FFmpeg command
    vf = f"crop={crop_w}:{crop_h}:{x_expr}:{y_expr},scale={target_w}:{target_h}"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path,
    ]

    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Reframe complete!")

    logger.info(f"Reframed video saved: {output_path}")
    return output_path
