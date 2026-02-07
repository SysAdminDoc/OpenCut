"""
OpenCut Auto-Reframe

Converts video aspect ratios with intelligent face-aware cropping.
Detects faces in sampled frames to determine optimal crop regions,
then renders the reframed output via FFmpeg.

Face detection:
  - Primary:  MediaPipe Face Detection (pip install mediapipe)
  - Fallback: OpenCV Haar cascades (bundled with opencv-python)
  - Minimal:  Center crop (no face detection)

Uses FFmpeg for final rendering - face detection only for crop planning.
"""

import json
import logging
import math
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CropRegion:
    """A crop region within a frame."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class FaceDetection:
    """A detected face in a frame."""
    x_center: float     # 0.0-1.0 normalized
    y_center: float
    width: float         # 0.0-1.0 normalized
    height: float
    confidence: float


@dataclass
class ReframeResult:
    """Result of an auto-reframe operation."""
    output_path: str = ""
    original_width: int = 0
    original_height: int = 0
    output_width: int = 0
    output_height: int = 0
    target_aspect: str = ""
    faces_detected: int = 0
    method: str = ""  # "mediapipe", "opencv", "center"


# ---------------------------------------------------------------------------
# Aspect ratio presets
# ---------------------------------------------------------------------------
ASPECT_PRESETS = {
    "9:16":  {"label": "Vertical (9:16)",   "description": "TikTok, Reels, Shorts", "ratio": 9 / 16},
    "1:1":   {"label": "Square (1:1)",       "description": "Instagram Post",         "ratio": 1 / 1},
    "4:5":   {"label": "Portrait (4:5)",     "description": "Instagram Portrait",     "ratio": 4 / 5},
    "4:3":   {"label": "Standard (4:3)",     "description": "Classic TV",             "ratio": 4 / 3},
    "16:9":  {"label": "Widescreen (16:9)",  "description": "YouTube, Standard",      "ratio": 16 / 9},
    "21:9":  {"label": "Ultrawide (21:9)",   "description": "Cinematic",              "ratio": 21 / 9},
    "2.35:1": {"label": "Anamorphic (2.35:1)", "description": "Movie scope",          "ratio": 2.35},
}


def get_aspect_presets() -> List[Dict]:
    """Return available aspect ratio presets."""
    return [
        {"name": name, "label": data["label"], "description": data["description"]}
        for name, data in ASPECT_PRESETS.items()
    ]


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------
def _probe_video_info(filepath: str) -> Dict:
    """Get video stream info."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        "-select_streams", "v:0",
        filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    info = {"fps": 30.0, "width": 1920, "height": 1080, "duration": 0.0}
    try:
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        info["width"] = int(stream.get("width", 1920))
        info["height"] = int(stream.get("height", 1080))

        r_frame = stream.get("r_frame_rate", "30/1")
        if "/" in str(r_frame):
            num, den = r_frame.split("/")
            info["fps"] = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            info["fps"] = float(r_frame)

        # Duration from format
        fmt = data.get("format") if "format" in data else {}
        if not fmt:
            # Re-probe with format
            cmd2 = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", filepath,
            ]
            r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
            try:
                fmt = json.loads(r2.stdout).get("format", {})
            except Exception:
                fmt = {}
        info["duration"] = float(fmt.get("duration", 0.0))
    except (json.JSONDecodeError, ValueError, TypeError, IndexError):
        pass
    return info


# ---------------------------------------------------------------------------
# Face detection backends
# ---------------------------------------------------------------------------
def _detect_faces_mediapipe(frame_path: str) -> List[FaceDetection]:
    """Detect faces using MediaPipe."""
    try:
        import cv2
        import mediapipe as mp

        img = cv2.imread(frame_path)
        if img is None:
            return []

        h, w = img.shape[:2]
        mp_face = mp.solutions.face_detection

        with mp_face.FaceDetection(
            model_selection=1,   # 0=short-range, 1=full-range (up to 5m)
            min_detection_confidence=0.5,
        ) as face_det:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_det.process(rgb)

            if not results.detections:
                return []

            faces = []
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                faces.append(FaceDetection(
                    x_center=bbox.xmin + bbox.width / 2,
                    y_center=bbox.ymin + bbox.height / 2,
                    width=bbox.width,
                    height=bbox.height,
                    confidence=det.score[0] if det.score else 0.0,
                ))
            return faces

    except ImportError:
        return []
    except Exception as e:
        logger.warning(f"MediaPipe face detection error: {e}")
        return []


def _detect_faces_opencv(frame_path: str) -> List[FaceDetection]:
    """Detect faces using OpenCV Haar cascades."""
    try:
        import cv2

        img = cv2.imread(frame_path)
        if img is None:
            return []

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use the frontal face cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        rects = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        faces = []
        for (x, y, fw, fh) in rects:
            faces.append(FaceDetection(
                x_center=(x + fw / 2) / w,
                y_center=(y + fh / 2) / h,
                width=fw / w,
                height=fh / h,
                confidence=0.8,
            ))
        return faces

    except ImportError:
        return []
    except Exception as e:
        logger.warning(f"OpenCV face detection error: {e}")
        return []


def detect_faces(frame_path: str) -> Tuple[List[FaceDetection], str]:
    """
    Detect faces using best available backend.
    Returns (faces, method) where method is "mediapipe", "opencv", or "none".
    """
    # Try MediaPipe first
    faces = _detect_faces_mediapipe(frame_path)
    if faces:
        return faces, "mediapipe"

    # Fall back to OpenCV
    faces = _detect_faces_opencv(frame_path)
    if faces:
        return faces, "opencv"

    return [], "none"


# ---------------------------------------------------------------------------
# Frame sampling + face analysis
# ---------------------------------------------------------------------------
def _extract_sample_frames(
    input_path: str,
    count: int = 10,
    temp_dir: str = "",
) -> List[str]:
    """Extract evenly-spaced sample frames from the video."""
    if not temp_dir:
        temp_dir = os.path.join(os.path.dirname(input_path), ".opencut_reframe_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    # Use FFmpeg to extract frames
    frame_paths = []
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-y", "-i", input_path,
        "-vf", f"fps=1/{max(1, count)}",  # Will be overridden below
        "-frames:v", str(count),
        os.path.join(temp_dir, "frame_%04d.jpg"),
    ]

    # Better approach: select specific frames spread across the video
    info = _probe_video_info(input_path)
    duration = info["duration"]
    if duration <= 0:
        duration = 10.0

    interval = duration / (count + 1)
    select_expr = "+".join(
        [f"eq(n\\,{int(i * interval * info['fps'])})" for i in range(1, count + 1)]
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-y", "-i", input_path,
        "-vf", f"select='{select_expr}',scale=640:-1",
        "-vsync", "vfr",
        "-frames:v", str(count),
        "-q:v", "5",
        os.path.join(temp_dir, "frame_%04d.jpg"),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        logger.warning("Frame extraction timed out")

    # Collect extracted frames
    for i in range(1, count + 1):
        path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
        if os.path.isfile(path):
            frame_paths.append(path)

    return frame_paths


def analyze_face_positions(
    input_path: str,
    sample_count: int = 10,
    temp_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> Tuple[Optional[Dict], str]:
    """
    Analyze face positions across sampled frames.

    Returns:
        (average_face_center, method)
        average_face_center: {"x": 0.0-1.0, "y": 0.0-1.0} or None
        method: detection method used
    """
    if on_progress:
        on_progress(10, "Extracting sample frames...")

    frames = _extract_sample_frames(input_path, sample_count, temp_dir)
    if not frames:
        return None, "none"

    all_faces = []
    method = "none"

    for i, frame_path in enumerate(frames):
        faces, m = detect_faces(frame_path)
        if faces:
            method = m
            # Use the most confident face
            best = max(faces, key=lambda f: f.confidence)
            all_faces.append(best)

        if on_progress:
            pct = 10 + int((i / len(frames)) * 60)
            on_progress(pct, f"Analyzing frame {i + 1}/{len(frames)}...")

    # Clean up temp frames
    for f in frames:
        try:
            os.remove(f)
        except OSError:
            pass
    if temp_dir and os.path.isdir(temp_dir):
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

    if not all_faces:
        return None, "none"

    # Average face center
    avg_x = sum(f.x_center for f in all_faces) / len(all_faces)
    avg_y = sum(f.y_center for f in all_faces) / len(all_faces)

    return {"x": avg_x, "y": avg_y, "detections": len(all_faces)}, method


# ---------------------------------------------------------------------------
# Crop calculation
# ---------------------------------------------------------------------------
def calculate_crop(
    src_width: int,
    src_height: int,
    target_aspect: str,
    face_center: Optional[Dict] = None,
    padding: float = 0.1,
) -> CropRegion:
    """
    Calculate the optimal crop region for a target aspect ratio.

    If face_center is provided, the crop is positioned to center on the face.
    Otherwise, centers the crop.

    Args:
        src_width:    Source video width.
        src_height:   Source video height.
        target_aspect: Aspect ratio string (e.g., "9:16").
        face_center:  {"x": 0.0-1.0, "y": 0.0-1.0} or None.
        padding:      Extra padding around face (0.0-0.5).

    Returns:
        CropRegion with x, y, width, height.
    """
    preset = ASPECT_PRESETS.get(target_aspect)
    if not preset:
        raise ValueError(f"Unknown aspect ratio: {target_aspect}")

    target_ratio = preset["ratio"]
    src_ratio = src_width / src_height

    if target_ratio <= src_ratio:
        # Target is narrower/taller - crop width
        crop_height = src_height
        crop_width = int(src_height * target_ratio)
    else:
        # Target is wider/shorter - crop height
        crop_width = src_width
        crop_height = int(src_width / target_ratio)

    # Ensure even dimensions
    crop_width = crop_width - (crop_width % 2)
    crop_height = crop_height - (crop_height % 2)

    # Clamp to source
    crop_width = min(crop_width, src_width)
    crop_height = min(crop_height, src_height)

    # Position: center on face or center of frame
    if face_center:
        face_x_px = face_center["x"] * src_width
        face_y_px = face_center["y"] * src_height

        # Center crop on face
        x = int(face_x_px - crop_width / 2)
        y = int(face_y_px - crop_height / 2)
    else:
        x = (src_width - crop_width) // 2
        y = (src_height - crop_height) // 2

    # Clamp to bounds
    x = max(0, min(x, src_width - crop_width))
    y = max(0, min(y, src_height - crop_height))

    return CropRegion(x=x, y=y, width=crop_width, height=crop_height)


# ---------------------------------------------------------------------------
# Reframe rendering
# ---------------------------------------------------------------------------
def auto_reframe(
    input_path: str,
    target_aspect: str = "9:16",
    use_face_detection: bool = True,
    output_dir: str = "",
    output_width: int = 0,
    quality: str = "medium",
    sample_count: int = 10,
    on_progress: Optional[Callable] = None,
) -> ReframeResult:
    """
    Auto-reframe a video to a target aspect ratio with face-aware cropping.

    Args:
        input_path:         Source video file.
        target_aspect:      Target aspect ratio (e.g., "9:16", "1:1").
        use_face_detection: Enable face-aware positioning.
        output_dir:         Output directory.
        output_width:       Force output width (0 = auto from source).
        quality:            Encoding quality.
        sample_count:       Frames to sample for face detection.
        on_progress:        Callback(pct, msg).

    Returns:
        ReframeResult with output path and metadata.
    """
    if on_progress:
        on_progress(5, "Analyzing source video...")

    video_info = _probe_video_info(input_path)
    src_w = video_info["width"]
    src_h = video_info["height"]

    # Face detection
    face_center = None
    method = "center"

    if use_face_detection:
        if on_progress:
            on_progress(10, "Detecting faces...")

        face_center, method = analyze_face_positions(
            input_path, sample_count,
            on_progress=on_progress,
        )

        if face_center:
            if on_progress:
                on_progress(70, f"Face detected ({method}), calculating crop...")
        else:
            method = "center"
            if on_progress:
                on_progress(70, "No faces found, using center crop...")
    else:
        if on_progress:
            on_progress(70, "Using center crop...")

    # Calculate crop
    crop = calculate_crop(src_w, src_h, target_aspect, face_center)

    # Determine output resolution
    if output_width > 0:
        # Scale to specified width
        preset = ASPECT_PRESETS.get(target_aspect, {"ratio": 9 / 16})
        out_w = output_width - (output_width % 2)
        out_h = int(out_w / preset["ratio"])
        out_h = out_h - (out_h % 2)
    else:
        out_w = crop.width
        out_h = crop.height

    if on_progress:
        on_progress(75, f"Rendering {out_w}x{out_h}...")

    # Output path
    if not output_dir:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    aspect_label = target_aspect.replace(":", "x").replace(".", "")
    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base}_reframe_{aspect_label}.mp4")

    # Quality settings
    crf_map = {"low": "28", "medium": "23", "high": "18", "lossless": "0"}
    crf = crf_map.get(quality, "23")

    # Build FFmpeg command
    vf_filter = f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y}"
    if out_w != crop.width or out_h != crop.height:
        vf_filter += f",scale={out_w}:{out_h}"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-y",
        "-i", input_path,
        "-vf", vf_filter,
        "-c:v", "libx264", "-crf", crf, "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]

    if on_progress:
        on_progress(80, "Encoding reframed video...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"FFmpeg reframe error: {result.stderr[-1000:]}")
            raise RuntimeError(f"FFmpeg encoding failed: {result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Reframe encoding timed out (>60 minutes)")

    if on_progress:
        on_progress(100, "Reframe complete")

    return ReframeResult(
        output_path=output_path,
        original_width=src_w,
        original_height=src_h,
        output_width=out_w,
        output_height=out_h,
        target_aspect=target_aspect,
        faces_detected=face_center.get("detections", 0) if face_center else 0,
        method=method,
    )
