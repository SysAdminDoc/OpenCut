"""
OpenCut License Plate Detection & Blur

Detect license plates via YOLO/PaddleOCR when available, falling back to
color-threshold + contour detection.  Track detections across frames with
IoU matching, apply Gaussian blur, and export redaction metadata JSON.

Uses FFmpeg for frame extraction and final compositing; OpenCV or NumPy for
detection when available.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PlateDetection:
    """A single detected license plate region."""
    x: int
    y: int
    w: int
    h: int
    frame: int = 0
    timestamp: float = 0.0
    confidence: float = 0.0
    text: str = ""
    track_id: int = -1


@dataclass
class PlateBlurResult:
    """Result from plate blur processing."""
    output_path: str = ""
    plates_found: int = 0
    tracks: int = 0
    method: str = "contour"
    metadata_path: str = ""


# ---------------------------------------------------------------------------
# IoU Tracking
# ---------------------------------------------------------------------------
def _iou(a: dict, b: dict) -> float:
    """Compute Intersection-over-Union between two boxes {x,y,w,h}."""
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    return inter / (area_a + area_b - inter)


def _track_detections(detections: List[PlateDetection], iou_threshold: float = 0.3) -> List[PlateDetection]:
    """Assign track IDs to detections using greedy IoU matching."""
    tracks: List[dict] = []  # list of {x,y,w,h} for last-known position
    next_id = 0

    for det in detections:
        det_box = {"x": det.x, "y": det.y, "w": det.w, "h": det.h}
        best_iou = 0.0
        best_tid = -1
        for tid, tbox in enumerate(tracks):
            score = _iou(det_box, tbox)
            if score > best_iou:
                best_iou = score
                best_tid = tid

        if best_iou >= iou_threshold and best_tid >= 0:
            det.track_id = best_tid
            tracks[best_tid] = det_box
        else:
            det.track_id = next_id
            tracks.append(det_box)
            next_id += 1

    return detections


# ---------------------------------------------------------------------------
# Detection back-ends
# ---------------------------------------------------------------------------
def _detect_plates_yolo(frames_dir: str, info: dict, sample_fps: float) -> Optional[List[PlateDetection]]:
    """Attempt YOLO-based plate detection."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return None

    try:
        model = YOLO("yolov8n.pt")
    except Exception:
        return None

    detections: List[PlateDetection] = []
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    fps = info.get("fps", 25.0)

    for idx, fname in enumerate(frame_files):
        fpath = os.path.join(frames_dir, fname)
        try:
            results = model(fpath, verbose=False)
        except Exception:
            continue

        t = idx / sample_fps
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                # Accept any high-confidence rectangular detection as possible plate
                if conf < 0.25:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append(PlateDetection(
                    x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                    frame=idx, timestamp=t, confidence=conf,
                ))

    return detections if detections else None


def _detect_plates_contour(frames_dir: str, info: dict, sample_fps: float) -> List[PlateDetection]:
    """Fallback: color-threshold + contour detection for rectangular plate-like regions."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.info("OpenCV not available; install opencv-python for contour plate detection")
        return []

    detections: List[PlateDetection] = []
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    width, height = info.get("width", 1920), info.get("height", 1080)
    min_area = (width * height) * 0.0005  # plates are small
    max_area = (width * height) * 0.05

    for idx, fname in enumerate(frame_files):
        fpath = os.path.join(frames_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Bilateral filter to reduce noise, keep edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(filtered, 30, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        t = idx / sample_fps
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / max(h, 1)
                # Plates are typically 2:1 to 5:1 aspect ratio
                if 1.5 <= aspect <= 6.0:
                    detections.append(PlateDetection(
                        x=x, y=y, w=w, h=h,
                        frame=idx, timestamp=t,
                        confidence=min(1.0, aspect / 4.0),
                    ))

    return detections


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def _extract_sample_frames(input_path: str, output_dir: str, sample_fps: float = 2.0) -> None:
    """Extract frames at sample_fps rate for detection."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", f"fps={sample_fps}",
        os.path.join(output_dir, "frame_%06d.png"),
    ]
    run_ffmpeg(cmd)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_plates(
    input_path: str,
    sample_fps: float = 2.0,
    iou_threshold: float = 0.3,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Detect license plates in a video.

    Args:
        input_path: Source video file.
        sample_fps: Frames per second to sample for detection.
        iou_threshold: IoU threshold for cross-frame tracking.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with detections list, tracks count, method used.
    """
    sample_fps = max(0.5, min(10.0, float(sample_fps)))
    iou_threshold = max(0.1, min(0.9, float(iou_threshold)))

    info = get_video_info(input_path)

    if on_progress:
        on_progress(10, "Extracting sample frames...")

    tmpdir = tempfile.mkdtemp(prefix="opencut_plates_")
    try:
        _extract_sample_frames(input_path, tmpdir, sample_fps)

        if on_progress:
            on_progress(30, "Running plate detection...")

        # Try YOLO first, fall back to contour
        detections = _detect_plates_yolo(tmpdir, info, sample_fps)
        method = "yolo"
        if detections is None:
            detections = _detect_plates_contour(tmpdir, info, sample_fps)
            method = "contour"

        if on_progress:
            on_progress(70, "Tracking detections across frames...")

        detections = _track_detections(detections, iou_threshold)
        track_ids = set(d.track_id for d in detections)

        if on_progress:
            on_progress(100, f"Found {len(detections)} plate detections in {len(track_ids)} tracks")

        return {
            "detections": [asdict(d) for d in detections],
            "plates_found": len(detections),
            "tracks": len(track_ids),
            "method": method,
        }
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def blur_plates(
    input_path: str,
    detections: Optional[List[dict]] = None,
    blur_strength: int = 30,
    sample_fps: float = 2.0,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Detect and blur license plates in a video.

    If detections is None, runs detect_plates first.

    Args:
        input_path: Source video file.
        detections: Pre-computed detections (optional).
        blur_strength: Gaussian blur kernel size (1-100).
        sample_fps: Frames per second for detection sampling.
        output_path_str: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, plates_found, tracks, method, metadata_path.
    """
    blur_strength = max(1, min(100, int(blur_strength)))

    if output_path_str is None:
        output_path_str = output_path(input_path, "plates_blurred")

    info = get_video_info(input_path)
    duration = info["duration"]
    width, height = info["width"], info["height"]
    fps = info.get("fps", 25.0)

    if detections is None:
        if on_progress:
            on_progress(5, "Detecting plates...")
        det_result = detect_plates(
            input_path, sample_fps=sample_fps,
            on_progress=lambda p, m: on_progress(5 + int(p * 0.4), m) if on_progress else None,
        )
        detections = det_result["detections"]
        method = det_result["method"]
        tracks = det_result["tracks"]
    else:
        method = "provided"
        track_ids = set(d.get("track_id", i) for i, d in enumerate(detections))
        tracks = len(track_ids)

    if not detections:
        if on_progress:
            on_progress(90, "No plates detected, copying original...")
        ffmpeg = get_ffmpeg_path()
        cmd = [ffmpeg, "-i", input_path, "-c", "copy", "-y", output_path_str]
        run_ffmpeg(cmd)
        return {
            "output_path": output_path_str,
            "plates_found": 0,
            "tracks": 0,
            "method": method,
            "metadata_path": "",
        }

    if on_progress:
        on_progress(50, f"Blurring {len(detections)} plate regions...")

    # Group detections by time window and build blur filter
    # Each detection gets a time-windowed boxblur overlay
    filter_parts = []
    current_label = "[0:v]"

    for i, det in enumerate(detections):
        rx = max(0, min(int(det.get("x", 0)), width - 1))
        ry = max(0, min(int(det.get("y", 0)), height - 1))
        rw = max(2, min(int(det.get("w", 50)), width - rx))
        rh = max(2, min(int(det.get("h", 30)), height - ry))
        t = float(det.get("timestamp", 0))
        t_end = min(t + (1.0 / sample_fps) + 0.1, duration)

        enable = f"between(t,{t},{t_end})"
        next_label = f"[v{i}]"

        filter_parts.append(
            f"{current_label}split[base{i}][crop{i}];"
            f"[crop{i}]crop={rw}:{rh}:{rx}:{ry},"
            f"boxblur={blur_strength}:{blur_strength}[blurred{i}];"
            f"[base{i}][blurred{i}]overlay={rx}:{ry}:enable='{enable}'{next_label}"
        )
        current_label = next_label

    filter_str = ";".join(filter_parts)
    if filter_str:
        last_label = f"[v{len(detections) - 1}]"
        filter_str = filter_str[:filter_str.rfind(last_label)] + "[out]"
    else:
        filter_str = "[0:v]copy[out]"

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-i", input_path,
        "-filter_complex", filter_str,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        "-y", output_path_str,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(90, "Generating redaction metadata...")

    # Export metadata JSON
    meta_path = os.path.splitext(output_path_str)[0] + "_plate_metadata.json"
    metadata = {
        "plate_redaction": {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source_file": os.path.basename(input_path),
            "method": method,
            "blur_strength": blur_strength,
            "total_detections": len(detections),
            "total_tracks": tracks,
            "detections": detections,
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if on_progress:
        on_progress(100, "Plate blur complete")

    return {
        "output_path": output_path_str,
        "plates_found": len(detections),
        "tracks": tracks,
        "method": method,
        "metadata_path": meta_path,
    }
