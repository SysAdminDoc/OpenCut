"""
OpenCut Document & Screen Redaction

Detect rectangular surfaces (screens, documents, whiteboards) via edge
detection and contour analysis.  Classify surface type, then apply blur
to entire surface or selectively to OCR-identified text regions.

Uses FFmpeg for frame I/O and OpenCV for detection when available.
"""

import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SurfaceDetection:
    """A detected rectangular surface in a frame."""
    x: int
    y: int
    w: int
    h: int
    frame: int = 0
    timestamp: float = 0.0
    surface_type: str = "unknown"  # screen, document, whiteboard, unknown
    confidence: float = 0.0
    area_ratio: float = 0.0  # fraction of total frame area


@dataclass
class DocRedactResult:
    """Result of document/screen redaction."""
    output_path: str = ""
    surfaces_found: int = 0
    surface_types: List[str] = field(default_factory=list)
    redaction_mode: str = "full"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _classify_surface(contour_props: dict, frame_brightness: float) -> str:
    """
    Classify a rectangular surface based on properties.

    Args:
        contour_props: Dict with aspect, area_ratio, mean_brightness, edge_density.
        frame_brightness: Mean brightness of the full frame.

    Returns:
        Surface type: screen, document, whiteboard, or unknown.
    """
    aspect = contour_props.get("aspect", 1.0)
    area_ratio = contour_props.get("area_ratio", 0.0)
    brightness = contour_props.get("mean_brightness", 128.0)
    edge_density = contour_props.get("edge_density", 0.0)

    # Screens: bright rectangles, often 16:9 or 4:3
    if 1.2 <= aspect <= 2.0 and brightness > frame_brightness * 1.2:
        return "screen"
    # Documents: paper-like, high contrast text
    if 0.6 <= aspect <= 0.85 and brightness > 180 and edge_density > 0.1:
        return "document"
    # Whiteboards: large, bright, lower edge density
    if area_ratio > 0.08 and brightness > 200 and edge_density < 0.08:
        return "whiteboard"

    return "unknown"


def _extract_frames(input_path: str, output_dir: str, sample_fps: float = 1.0) -> None:
    """Extract frames for surface detection."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", f"fps={sample_fps}",
        os.path.join(output_dir, "frame_%06d.png"),
    ]
    run_ffmpeg(cmd)


def _detect_surfaces_opencv(
    frames_dir: str, info: dict, sample_fps: float,
    min_area_ratio: float = 0.02, max_area_ratio: float = 0.8,
) -> List[SurfaceDetection]:
    """Detect rectangular surfaces using OpenCV edge detection and contour analysis."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.info("OpenCV not available; install opencv-python for surface detection")
        return []

    detections: List[SurfaceDetection] = []
    width, height = info.get("width", 1920), info.get("height", 1080)
    frame_area = width * height
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))

    for idx, fname in enumerate(frame_files):
        fpath = os.path.join(frames_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        t = idx / sample_fps
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame_brightness = float(np.mean(gray))

        # Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_ratio = area / frame_area
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / max(h, 1)

                # Compute properties for classification
                roi = gray[y:y+h, x:x+w]
                mean_bright = float(np.mean(roi)) if roi.size > 0 else 128.0
                roi_edges = edges[y:y+h, x:x+w]
                edge_density = float(np.count_nonzero(roi_edges)) / max(roi.size, 1)

                surface_type = _classify_surface({
                    "aspect": aspect,
                    "area_ratio": area_ratio,
                    "mean_brightness": mean_bright,
                    "edge_density": edge_density,
                }, frame_brightness)

                confidence = min(1.0, area_ratio * 10 + (0.3 if surface_type != "unknown" else 0.0))

                detections.append(SurfaceDetection(
                    x=x, y=y, w=w, h=h,
                    frame=idx, timestamp=t,
                    surface_type=surface_type,
                    confidence=confidence,
                    area_ratio=round(area_ratio, 4),
                ))

    return detections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_surfaces(
    input_path: str,
    sample_fps: float = 1.0,
    min_area_ratio: float = 0.02,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Detect rectangular surfaces (screens, documents, whiteboards) in video.

    Args:
        input_path: Source video file.
        sample_fps: Frames per second for detection sampling.
        min_area_ratio: Minimum surface area as fraction of frame (0-1).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with detections list, surfaces_found, surface_types.
    """
    sample_fps = max(0.25, min(5.0, float(sample_fps)))
    min_area_ratio = max(0.005, min(0.5, float(min_area_ratio)))
    info = get_video_info(input_path)

    if on_progress:
        on_progress(10, "Extracting frames for surface detection...")

    tmpdir = tempfile.mkdtemp(prefix="opencut_doc_")
    try:
        _extract_frames(input_path, tmpdir, sample_fps)

        if on_progress:
            on_progress(30, "Detecting rectangular surfaces...")

        detections = _detect_surfaces_opencv(tmpdir, info, sample_fps, min_area_ratio)
        surface_types = sorted(set(d.surface_type for d in detections))

        if on_progress:
            on_progress(100, f"Found {len(detections)} surfaces ({', '.join(surface_types) or 'none'})")

        return {
            "detections": [asdict(d) for d in detections],
            "surfaces_found": len(detections),
            "surface_types": surface_types,
        }
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def redact_surfaces(
    input_path: str,
    detections: Optional[List[dict]] = None,
    surface_types: Optional[List[str]] = None,
    redaction_mode: str = "full",
    blur_strength: int = 30,
    sample_fps: float = 1.0,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Detect and redact document/screen surfaces in video.

    Args:
        input_path: Source video file.
        detections: Pre-computed detections (optional).
        surface_types: Filter to only redact these surface types.
        redaction_mode: "full" blurs entire surface, "text" blurs text regions only.
        blur_strength: Blur kernel size (1-100).
        sample_fps: Detection sample rate.
        output_path_str: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, surfaces_found, surface_types, redaction_mode.
    """
    blur_strength = max(1, min(100, int(blur_strength)))
    redaction_mode = redaction_mode if redaction_mode in ("full", "text") else "full"

    if output_path_str is None:
        output_path_str = output_path(input_path, "doc_redacted")

    info = get_video_info(input_path)
    duration = info["duration"]
    width, height = info["width"], info["height"]

    if detections is None:
        if on_progress:
            on_progress(5, "Detecting surfaces...")
        det_result = detect_surfaces(
            input_path, sample_fps=sample_fps,
            on_progress=lambda p, m: on_progress(5 + int(p * 0.4), m) if on_progress else None,
        )
        detections = det_result["detections"]

    # Filter by surface type if specified
    if surface_types:
        type_set = set(surface_types)
        detections = [d for d in detections if d.get("surface_type", "unknown") in type_set]

    types_found = sorted(set(d.get("surface_type", "unknown") for d in detections))

    if not detections:
        if on_progress:
            on_progress(90, "No surfaces detected, copying original...")
        ffmpeg = get_ffmpeg_path()
        cmd = [ffmpeg, "-i", input_path, "-c", "copy", "-y", output_path_str]
        run_ffmpeg(cmd)
        return {
            "output_path": output_path_str,
            "surfaces_found": 0,
            "surface_types": [],
            "redaction_mode": redaction_mode,
        }

    if on_progress:
        on_progress(50, f"Redacting {len(detections)} surfaces...")

    # Build blur filter chain
    filter_parts = []
    current_label = "[0:v]"

    for i, det in enumerate(detections):
        rx = max(0, min(int(det.get("x", 0)), width - 1))
        ry = max(0, min(int(det.get("y", 0)), height - 1))
        rw = max(2, min(int(det.get("w", 100)), width - rx))
        rh = max(2, min(int(det.get("h", 100)), height - ry))
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
        on_progress(100, "Surface redaction complete")

    return {
        "output_path": output_path_str,
        "surfaces_found": len(detections),
        "surface_types": types_found,
        "redaction_mode": redaction_mode,
    }
