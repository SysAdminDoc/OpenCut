"""
Auto Zoom / Dynamic Push-In

Detects face positions in video frames and generates
scale + position keyframes for an engaging push-in zoom effect.
Uses OpenCV face detection (Haar cascade fallback if no DNN model available).
"""

import logging
import os
import threading
from typing import List, Optional

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False
    logger.debug("opencv-python not installed; auto_zoom features unavailable. "
                 "Install with: pip install opencv-python")


def _require_cv2():
    if not _CV2_AVAILABLE:
        raise RuntimeError(
            "opencv-python is required for auto-zoom. "
            "Install with: pip install opencv-python"
        )


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------

def _ease(t: float, mode: str) -> float:
    """
    Apply an easing curve to a normalised time value t ∈ [0.0, 1.0].

    Args:
        t: Normalised time (0.0 = start, 1.0 = end).
        mode: "linear", "ease_in", "ease_out", or "ease_in_out".

    Returns:
        Eased value in [0.0, 1.0].
    """
    t = max(0.0, min(1.0, t))
    if mode == "linear":
        return t
    elif mode == "ease_in":
        return t * t
    elif mode == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    elif mode == "ease_in_out":
        if t < 0.5:
            return 2.0 * t * t
        else:
            return 1.0 - 2.0 * (1.0 - t) * (1.0 - t)
    # Fallback
    return t


# ---------------------------------------------------------------------------
# Haar cascade loader
# ---------------------------------------------------------------------------

_CASCADE: Optional[object] = None
_CASCADE_LOCK = threading.Lock()


def _get_cascade():
    """Load the frontal-face Haar cascade (cached after first load). Thread-safe."""
    global _CASCADE
    if _CASCADE is not None:
        return _CASCADE

    with _CASCADE_LOCK:
        # Double-check after acquiring lock
        if _CASCADE is not None:
            return _CASCADE

        # Try the built-in OpenCV data path first
        cascade_name = "haarcascade_frontalface_default.xml"
        builtin_path = cv2.data.haarcascades + cascade_name  # type: ignore[union-attr]
        if os.path.exists(builtin_path):
            _CASCADE = cv2.CascadeClassifier(builtin_path)
            return _CASCADE

        # Fallback: search common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), cascade_name),
            os.path.join(os.path.expanduser("~"), ".opencut", cascade_name),
        ]
        for path in candidates:
            if os.path.exists(path):
                _CASCADE = cv2.CascadeClassifier(path)
                return _CASCADE

        # Return empty classifier — detection will always return no faces,
        # and we fall back to centre-crop anchors.
        logger.warning(
            "Haar cascade not found at %s; face detection disabled, using centre crop.",
            builtin_path,
        )
        _CASCADE = cv2.CascadeClassifier()
        return _CASCADE


# ---------------------------------------------------------------------------
# Face detection on a single frame
# ---------------------------------------------------------------------------

def _detect_face_centre(gray_frame: "cv2.Mat") -> Optional[tuple]:  # type: ignore[name-defined]
    """
    Detect the largest face in a greyscale frame.

    Returns:
        (cx, cy) normalised [0.0, 1.0] face centre, or None if no face found.
    """
    cascade = _get_cascade()
    if cascade is None or cascade.empty():
        return None

    faces = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return None

    # Pick the largest face
    faces_list = list(faces)
    faces_list.sort(key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces_list[0]

    height, width = gray_frame.shape[:2]
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    return cx, cy


# ---------------------------------------------------------------------------
# Keyframe interpolation
# ---------------------------------------------------------------------------

def _interpolate_anchors(
    raw_anchors: List[dict],
    easing: str,
) -> List[dict]:
    """
    Smooth anchor transitions between sampled keyframes using easing.

    Returns the same list with eased anchor_x / anchor_y values filled in
    based on transitions between adjacent anchor points.
    """
    if len(raw_anchors) <= 1:
        return raw_anchors

    smoothed = [dict(raw_anchors[0])]
    n = len(raw_anchors)

    for i in range(1, n):
        prev = smoothed[-1]  # use already-smoothed previous anchor
        curr = raw_anchors[i]
        # Blend factor: ease the normalised position through the sequence
        t_raw = i / (n - 1) if n > 1 else 1.0
        t = _ease(t_raw, easing)
        # Exponential moving average blend — 0.4 weight to new position
        blend = 0.4 * t + 0.3
        ax = prev["anchor_x"] + (curr["anchor_x"] - prev["anchor_x"]) * blend
        ay = prev["anchor_y"] + (curr["anchor_y"] - prev["anchor_y"]) * blend
        smoothed.append({**curr, "anchor_x": round(ax, 4), "anchor_y": round(ay, 4)})

    return smoothed


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def generate_zoom_keyframes(
    video_path: str,
    zoom_amount: float = 1.15,
    easing: str = "ease_in_out",
    sample_rate: float = 2.0,
    face_padding: float = 0.2,
    on_progress=None,
) -> dict:
    """
    Generate scale + anchor keyframes for an auto push-in zoom effect.

    Samples frames at sample_rate per second, detects face position, and
    returns a keyframe list suitable for Premiere Pro's Motion effect.

    Args:
        video_path: Path to the source video file.
        zoom_amount: Scale factor for the zoom (1.15 = 15% zoom in).
        easing: Easing mode — "linear", "ease_in", "ease_out", "ease_in_out".
        sample_rate: Frames to sample per second.
        face_padding: Fraction of frame size to pad around the face when
                      computing the anchor point (prevents extreme cropping).

    Returns:
        Dict with:
            "keyframes": list of {"time": float, "scale": float,
                                  "anchor_x": float, "anchor_y": float}
            "fps": float
            "duration": float

    Raises:
        RuntimeError: If cv2 is not installed or the video cannot be opened.
    """
    _require_cv2()

    zoom_amount = max(1.0, float(zoom_amount))
    sample_rate = max(0.1, float(sample_rate))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    frame_interval = max(1, int(fps / sample_rate))
    total_samples = max(1, total_frames // frame_interval)
    sample_times = []
    raw_keyframes = []

    frame_idx = 0
    sample_idx = 0
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                t = frame_idx / fps
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                face = _detect_face_centre(gray)

                if face is not None:
                    cx, cy = face
                    # Clamp anchor so face stays within frame after zoom
                    margin = face_padding
                    ax = max(margin, min(1.0 - margin, cx))
                    ay = max(margin, min(1.0 - margin, cy))
                else:
                    # No face — use centre crop
                    ax, ay = 0.5, 0.5

                # Compute eased scale: ramp from 1.0 at start to zoom_amount
                t_normalised = t / duration if duration > 0 else 0.0
                scale = 1.0 + (zoom_amount - 1.0) * _ease(t_normalised, easing)

                raw_keyframes.append({
                    "time": round(t, 4),
                    "scale": round(scale, 4),
                    "anchor_x": round(ax, 4),
                    "anchor_y": round(ay, 4),
                })
                sample_times.append(t)

                if on_progress and total_samples > 0:
                    on_progress(int(sample_idx / total_samples * 100))
                sample_idx += 1

            frame_idx += 1
    finally:
        cap.release()

    if not raw_keyframes:
        # Nothing could be read — return safe defaults
        return {
            "keyframes": [{"time": 0.0, "scale": 1.0, "anchor_x": 0.5, "anchor_y": 0.5}],
            "fps": fps,
            "duration": duration,
        }

    keyframes = _interpolate_anchors(raw_keyframes, easing)

    logger.info(
        "Generated %d zoom keyframes for %s (zoom=%.2f, easing=%s)",
        len(keyframes), video_path, zoom_amount, easing,
    )

    return {
        "keyframes": keyframes,
        "fps": fps,
        "duration": round(duration, 4),
    }
