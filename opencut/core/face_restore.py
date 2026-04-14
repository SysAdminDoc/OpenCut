"""
OpenCut AI Face Restoration Module

Detect and restore/enhance faces in video frames.  Applies targeted
sharpening and denoising to face regions via FFmpeg crop+overlay pipeline.

Optional: uses RetinaFace for improved detection accuracy, falling back
to FFmpeg-based face metadata if unavailable.

Functions:
    detect_faces         - Detect faces in sampled video frames
    restore_faces        - Enhance face regions across entire video
    restore_single_frame - Restore faces in a single image
"""

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class FaceBox:
    """Bounding box for a detected face."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    confidence: float = 0.0


@dataclass
class FaceDetectionResult:
    """Result of face detection on video frames."""
    faces_detected: int = 0
    avg_face_size: float = 0.0
    boxes: List[FaceBox] = field(default_factory=list)
    sample_frames: int = 0


@dataclass
class FaceRestoreResult:
    """Result of face restoration."""
    output_path: str = ""
    faces_detected: int = 0
    faces_restored: int = 0
    method: str = "ffmpeg"


# ---------------------------------------------------------------------------
# Face detection helpers
# ---------------------------------------------------------------------------
def _extract_sample_frames(video_path: str, count: int = 10) -> List[str]:
    """Extract evenly-spaced sample frames from a video.

    Returns list of temporary image paths.
    """
    info = get_video_info(video_path)
    duration = info.get("duration", 10.0)
    if duration <= 0:
        duration = 10.0

    ffmpeg = get_ffmpeg_path()
    frame_paths: List[str] = []
    interval = duration / max(1, count + 1)

    for i in range(count):
        ts = interval * (i + 1)
        fd, frame_path = tempfile.mkstemp(suffix=".jpg", prefix=f"opencut_face_sample_{i}_")
        os.close(fd)
        frame_paths.append(frame_path)

        cmd = [
            ffmpeg, "-hide_banner", "-y",
            "-ss", str(round(ts, 2)),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            frame_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception as exc:
            logger.debug("Frame extraction at %.2fs failed: %s", ts, exc)

    return frame_paths


def _detect_faces_retinaface(image_path: str) -> List[FaceBox]:
    """Detect faces using RetinaFace (high accuracy)."""
    try:
        from retinaface import RetinaFace
    except ImportError:
        return []

    try:
        faces = RetinaFace.detect_faces(image_path)
        if not isinstance(faces, dict):
            return []

        boxes: List[FaceBox] = []
        for _key, face_info in faces.items():
            area = face_info.get("facial_area", [0, 0, 0, 0])
            score = face_info.get("score", 0.0)
            x1, y1, x2, y2 = area
            boxes.append(FaceBox(
                x=int(x1),
                y=int(y1),
                width=int(x2 - x1),
                height=int(y2 - y1),
                confidence=round(float(score), 4),
            ))
        return boxes
    except Exception as exc:
        logger.debug("RetinaFace detection failed: %s", exc)
        return []


def _detect_faces_ffmpeg(image_path: str) -> List[FaceBox]:
    """Detect faces using FFmpeg metadata (lower accuracy fallback).

    Uses cropdetect as a rough heuristic for finding significant regions.
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner",
        "-i", image_path,
        "-vf", "cropdetect=24:16:0",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        crop_re = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
        matches = crop_re.findall(result.stderr)
        if matches:
            w, h, x, y = (int(v) for v in matches[-1])
            face_w = w // 3
            face_h = h // 3
            face_x = x + w // 3
            face_y = y + h // 6
            return [FaceBox(
                x=face_x, y=face_y,
                width=face_w, height=face_h,
                confidence=0.5,
            )]
    except Exception as exc:
        logger.debug("FFmpeg face heuristic failed: %s", exc)

    return []


# ---------------------------------------------------------------------------
# Public: face detection
# ---------------------------------------------------------------------------
def detect_faces(
    video_path: str,
    sample_frames: int = 10,
    on_progress: Optional[Callable] = None,
) -> FaceDetectionResult:
    """Detect faces in sampled video frames.

    Samples evenly-spaced frames from the video and runs face detection
    on each.  Uses RetinaFace if available, otherwise falls back to
    FFmpeg-based heuristic detection.

    Args:
        video_path: Path to input video.
        sample_frames: Number of frames to sample (default 10).
        on_progress: Optional progress callback ``(pct, msg)``.

    Returns:
        FaceDetectionResult with face count, average size, and bounding boxes.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    sample_frames = max(1, min(100, int(sample_frames)))

    if on_progress:
        on_progress(5, "Extracting sample frames...")

    frame_paths = _extract_sample_frames(video_path, count=sample_frames)

    if on_progress:
        on_progress(20, f"Extracted {len(frame_paths)} frames, detecting faces...")

    use_retinaface = ensure_package("retinaface", "retinaface-pytorch")

    all_boxes: List[FaceBox] = []
    for i, fp in enumerate(frame_paths):
        if not os.path.isfile(fp) or os.path.getsize(fp) == 0:
            continue

        if use_retinaface:
            boxes = _detect_faces_retinaface(fp)
        else:
            boxes = _detect_faces_ffmpeg(fp)

        all_boxes.extend(boxes)

        if on_progress and i % max(1, len(frame_paths) // 5) == 0:
            pct = 20 + int(60 * i / max(1, len(frame_paths)))
            on_progress(min(pct, 80), f"Processed frame {i + 1}/{len(frame_paths)}")

    # Clean up sample frames
    for fp in frame_paths:
        try:
            os.unlink(fp)
        except OSError:
            pass

    faces_detected = len(all_boxes)
    avg_size = 0.0
    if all_boxes:
        sizes = [b.width * b.height for b in all_boxes]
        avg_size = sum(sizes) / len(sizes)

    if on_progress:
        on_progress(90, f"Detected {faces_detected} faces across {len(frame_paths)} frames")

    return FaceDetectionResult(
        faces_detected=faces_detected,
        avg_face_size=round(avg_size, 1),
        boxes=all_boxes,
        sample_frames=len(frame_paths),
    )


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------
def _build_face_enhance_filter(
    x: int, y: int, w: int, h: int,
    frame_w: int, frame_h: int,
    strength: float = 1.0,
) -> str:
    """Build FFmpeg filter_complex for targeted face enhancement.

    Crops the face region, applies sharpening + denoising, then overlays
    back onto the original frame.
    """
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(16, min(w, frame_w - x))
    h = max(16, min(h, frame_h - y))

    sharp_amount = min(3.0, 1.0 + 0.8 * strength)
    nlm_s = min(6.0, 2.0 + 1.5 * strength)

    fc = (
        f"[0:v]split[bg][src];"
        f"[src]crop={w}:{h}:{x}:{y},"
        f"unsharp=5:5:{sharp_amount:.1f}:5:5:0.0,"
        f"nlmeans=s={nlm_s:.1f}:p=5:pc=3:r=9:rc=5"
        f"[face];"
        f"[bg][face]overlay={x}:{y}[outv]"
    )
    return fc


# ---------------------------------------------------------------------------
# Public: face restoration -- video
# ---------------------------------------------------------------------------
def restore_faces(
    video_path: str,
    output_path: Optional[str] = None,
    strength: float = 1.0,
    method: str = "ffmpeg",
    on_progress: Optional[Callable] = None,
) -> FaceRestoreResult:
    """Restore and enhance faces in a video.

    Detects face regions, then applies targeted sharpening and denoising
    to each face area using an FFmpeg crop+overlay pipeline.

    Args:
        video_path: Path to input video.
        output_path: Output video path. Auto-generated if None.
        strength: Enhancement strength 0.0-2.0 (1.0 = standard).
        method: Enhancement method (currently ``"ffmpeg"`` only).
        on_progress: Optional progress callback ``(pct, msg)``.

    Returns:
        FaceRestoreResult with output path and face counts.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    strength = max(0.0, min(2.0, float(strength)))

    if on_progress:
        on_progress(5, "Detecting faces for restoration...")

    detection = detect_faces(video_path, sample_frames=5, on_progress=on_progress)

    if on_progress:
        on_progress(40, f"Found {detection.faces_detected} faces, preparing restoration...")

    if output_path is None:
        output_path = _output_path(video_path, "face_restored", "")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # If no faces detected, copy original
    if detection.faces_detected == 0:
        logger.info("No faces detected, copying original video")
        import shutil
        shutil.copy2(video_path, output_path)

        if on_progress:
            on_progress(100, "No faces detected, copied original")

        return FaceRestoreResult(
            output_path=output_path,
            faces_detected=0,
            faces_restored=0,
            method=method,
        )

    if on_progress:
        on_progress(50, "Building face enhancement pipeline...")

    info = get_video_info(video_path)
    frame_w = info.get("width", 1920)
    frame_h = info.get("height", 1080)

    # Use the most confident/largest face for enhancement
    best_box = max(detection.boxes, key=lambda b: b.confidence * b.width * b.height)

    fc = _build_face_enhance_filter(
        best_box.x, best_box.y, best_box.width, best_box.height,
        frame_w, frame_h, strength,
    )

    if on_progress:
        on_progress(60, "Enhancing face regions...")

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .filter_complex(fc, maps=["[outv]", "0:a?"])
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("copy")
        .faststart()
        .output(output_path)
        .build()
    )

    run_ffmpeg(cmd)

    faces_restored = min(detection.faces_detected, 1)

    if on_progress:
        on_progress(95, f"Restored {faces_restored} face(s)")

    return FaceRestoreResult(
        output_path=output_path,
        faces_detected=detection.faces_detected,
        faces_restored=faces_restored,
        method=method,
    )


# ---------------------------------------------------------------------------
# Public: single-frame restoration
# ---------------------------------------------------------------------------
def restore_single_frame(
    image_path: str,
    output_path: Optional[str] = None,
    strength: float = 1.0,
) -> FaceRestoreResult:
    """Restore faces in a single image.

    Args:
        image_path: Path to input image.
        output_path: Output image path. Auto-generated if None.
        strength: Enhancement strength 0.0-2.0.

    Returns:
        FaceRestoreResult with output path and face counts.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    strength = max(0.0, min(2.0, float(strength)))

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        ext = ext or ".jpg"
        output_path = f"{base}_face_restored{ext}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Try RetinaFace first, fall back to FFmpeg heuristic
    use_retinaface = ensure_package("retinaface", "retinaface-pytorch")
    if use_retinaface:
        boxes = _detect_faces_retinaface(image_path)
    else:
        boxes = _detect_faces_ffmpeg(image_path)

    if not boxes:
        import shutil
        shutil.copy2(image_path, output_path)
        return FaceRestoreResult(
            output_path=output_path,
            faces_detected=0,
            faces_restored=0,
            method="ffmpeg",
        )

    # Get image dimensions via FFmpeg
    ffmpeg = get_ffmpeg_path()
    cmd_probe = [
        ffmpeg, "-hide_banner",
        "-i", image_path,
        "-f", "null", "-",
    ]
    try:
        probe_result = subprocess.run(cmd_probe, capture_output=True, text=True, timeout=10)
        size_re = re.compile(r"(\d{2,5})x(\d{2,5})")
        m = size_re.search(probe_result.stderr)
        if m:
            img_w, img_h = int(m.group(1)), int(m.group(2))
        else:
            img_w, img_h = 1920, 1080
    except Exception:
        img_w, img_h = 1920, 1080

    best_box = max(boxes, key=lambda b: b.confidence * b.width * b.height)

    sharp_amount = min(3.0, 1.0 + 0.8 * strength)
    nlm_s = min(6.0, 2.0 + 1.5 * strength)

    bx = max(0, min(best_box.x, img_w - 16))
    by = max(0, min(best_box.y, img_h - 16))
    bw = max(16, min(best_box.width, img_w - bx))
    bh = max(16, min(best_box.height, img_h - by))

    fc = (
        f"[0:v]split[bg][src];"
        f"[src]crop={bw}:{bh}:{bx}:{by},"
        f"unsharp=5:5:{sharp_amount:.1f}:5:5:0.0,"
        f"nlmeans=s={nlm_s:.1f}:p=5:pc=3:r=9:rc=5"
        f"[face];"
        f"[bg][face]overlay={bx}:{by}[outv]"
    )

    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-i", image_path,
        "-filter_complex", fc,
        "-map", "[outv]",
        "-q:v", "2",
        output_path,
    ]
    run_ffmpeg(cmd)

    return FaceRestoreResult(
        output_path=output_path,
        faces_detected=len(boxes),
        faces_restored=1,
        method="ffmpeg",
    )
