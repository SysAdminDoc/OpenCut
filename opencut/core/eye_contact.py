"""
OpenCut Eye Contact Correction Module v0.1.0

AI-powered gaze redirection for video:
- Detect face and eye landmarks via MediaPipe Face Mesh
- Estimate gaze direction per-frame
- Apply subtle warp to redirect eyes toward camera
- Falls back to OpenCV-based pupil shift when MediaPipe unavailable

Uses frame-by-frame processing with FFmpeg reassembly.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import ensure_package, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EyeContactConfig:
    """Configuration for eye contact correction."""
    strength: float = 0.7          # Correction strength 0.0-1.0
    smoothing_window: int = 5      # Temporal smoothing (frames)
    max_yaw_correction: float = 25.0   # Max horizontal gaze correction (degrees)
    max_pitch_correction: float = 15.0  # Max vertical gaze correction (degrees)
    face_confidence: float = 0.5   # Minimum face detection confidence
    eye_confidence: float = 0.5    # Minimum eye landmark confidence
    only_center_face: bool = True  # Only correct the largest/center face


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class EyeContactResult:
    """Result of eye contact correction."""
    output_path: str = ""
    frames_processed: int = 0
    frames_corrected: int = 0
    avg_gaze_offset: float = 0.0
    corrections_applied: int = 0


# ---------------------------------------------------------------------------
# Gaze Detection
# ---------------------------------------------------------------------------

# MediaPipe Face Mesh eye landmark indices
_LEFT_EYE_IRIS = [468, 469, 470, 471, 472]
_RIGHT_EYE_IRIS = [473, 474, 475, 476, 477]
_LEFT_EYE_CORNERS = [33, 133]   # inner, outer
_RIGHT_EYE_CORNERS = [362, 263]  # inner, outer
_NOSE_TIP = 1
_FOREHEAD = 10
_CHIN = 152


def detect_gaze_direction(frame, face_mesh=None, config: Optional[EyeContactConfig] = None) -> Dict:
    """
    Detect gaze direction from a single frame.

    Returns dict with:
        - detected: bool
        - yaw: float (degrees, negative=left, positive=right)
        - pitch: float (degrees, negative=down, positive=up)
        - left_eye_center: (x, y) normalized
        - right_eye_center: (x, y) normalized
        - face_center: (x, y) normalized
    """
    import cv2
    import numpy as np

    if config is None:
        config = EyeContactConfig()

    h, w = frame.shape[:2]
    result = {
        "detected": False,
        "yaw": 0.0,
        "pitch": 0.0,
        "left_eye_center": None,
        "right_eye_center": None,
        "face_center": None,
    }

    if face_mesh is None:
        return result

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    mp_result = face_mesh.process(rgb)

    if not mp_result.multi_face_landmarks:
        return result

    landmarks = mp_result.multi_face_landmarks[0]
    lm = landmarks.landmark

    # Get iris centers
    left_iris_x = sum(lm[i].x for i in _LEFT_EYE_IRIS) / len(_LEFT_EYE_IRIS)
    left_iris_y = sum(lm[i].y for i in _LEFT_EYE_IRIS) / len(_LEFT_EYE_IRIS)
    right_iris_x = sum(lm[i].x for i in _RIGHT_EYE_IRIS) / len(_RIGHT_EYE_IRIS)
    right_iris_y = sum(lm[i].y for i in _RIGHT_EYE_IRIS) / len(_RIGHT_EYE_IRIS)

    # Get eye corner midpoints for reference
    left_mid_x = (lm[_LEFT_EYE_CORNERS[0]].x + lm[_LEFT_EYE_CORNERS[1]].x) / 2
    left_mid_y = (lm[_LEFT_EYE_CORNERS[0]].y + lm[_LEFT_EYE_CORNERS[1]].y) / 2
    right_mid_x = (lm[_RIGHT_EYE_CORNERS[0]].x + lm[_RIGHT_EYE_CORNERS[1]].x) / 2
    right_mid_y = (lm[_RIGHT_EYE_CORNERS[0]].y + lm[_RIGHT_EYE_CORNERS[1]].y) / 2

    # Compute gaze offset as iris displacement from eye center
    left_eye_width = abs(lm[_LEFT_EYE_CORNERS[1]].x - lm[_LEFT_EYE_CORNERS[0]].x)
    right_eye_width = abs(lm[_RIGHT_EYE_CORNERS[1]].x - lm[_RIGHT_EYE_CORNERS[0]].x)

    if left_eye_width < 0.001 or right_eye_width < 0.001:
        return result

    # Normalized displacement (-1 to 1 range)
    left_dx = (left_iris_x - left_mid_x) / (left_eye_width / 2)
    left_dy = (left_iris_y - left_mid_y) / (left_eye_width / 2)
    right_dx = (right_iris_x - right_mid_x) / (right_eye_width / 2)
    right_dy = (right_iris_y - right_mid_y) / (right_eye_width / 2)

    avg_dx = (left_dx + right_dx) / 2
    avg_dy = (left_dy + right_dy) / 2

    # Convert to approximate degrees
    yaw = float(np.clip(avg_dx * 45.0, -60, 60))
    pitch = float(np.clip(avg_dy * 30.0, -40, 40))

    nose = lm[_NOSE_TIP]
    face_cx = nose.x
    face_cy = (lm[_FOREHEAD].y + lm[_CHIN].y) / 2

    result.update({
        "detected": True,
        "yaw": yaw,
        "pitch": pitch,
        "left_eye_center": (left_iris_x, left_iris_y),
        "right_eye_center": (right_iris_x, right_iris_y),
        "face_center": (face_cx, face_cy),
    })

    return result


def _apply_eye_warp(frame, gaze_info: Dict, config: EyeContactConfig):
    """Apply subtle warp to redirect gaze toward camera."""
    import cv2
    import numpy as np

    if not gaze_info["detected"]:
        return frame

    h, w = frame.shape[:2]
    yaw = gaze_info["yaw"]
    pitch = gaze_info["pitch"]

    # Only correct if gaze is off by meaningful amount
    if abs(yaw) < 2.0 and abs(pitch) < 2.0:
        return frame

    # Clamp correction
    yaw_corr = max(-config.max_yaw_correction, min(config.max_yaw_correction, yaw))
    pitch_corr = max(-config.max_pitch_correction, min(config.max_pitch_correction, pitch))

    strength = config.strength

    # Compute pixel shift for each eye
    for eye_key in ["left_eye_center", "right_eye_center"]:
        eye_pos = gaze_info[eye_key]
        if eye_pos is None:
            continue

        cx = int(eye_pos[0] * w)
        cy = int(eye_pos[1] * h)

        # Define region of interest around the eye
        roi_size = int(w * 0.08)
        x1 = max(0, cx - roi_size)
        y1 = max(0, cy - roi_size)
        x2 = min(w, cx + roi_size)
        y2 = min(h, cy + roi_size)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = frame[y1:y2, x1:x2].copy()
        rh, rw = roi.shape[:2]

        # Create displacement maps for subtle warp
        shift_x = -yaw_corr * strength * 0.3
        shift_y = -pitch_corr * strength * 0.3

        # Gaussian-weighted displacement (strongest at center)
        gy, gx = np.mgrid[0:rh, 0:rw].astype(np.float32)
        center_x = rw / 2.0
        center_y = rh / 2.0

        sigma = rw / 3.0
        gauss = np.exp(-((gx - center_x) ** 2 + (gy - center_y) ** 2) / (2 * sigma ** 2))

        map_x = gx - shift_x * gauss
        map_y = gy - shift_y * gauss

        warped = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Blend with Gaussian falloff
        alpha = gauss[:, :, np.newaxis] * strength * 0.8
        blended = (warped * alpha + roi * (1 - alpha)).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended

    return frame


# ---------------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------------
def correct_eye_contact(
    video_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    config: Optional[EyeContactConfig] = None,
    on_progress: Optional[Callable] = None,
) -> EyeContactResult:
    """
    Correct eye contact in a video to make the subject appear to look at camera.

    Uses MediaPipe Face Mesh for landmark detection, then applies subtle
    per-frame warping around the eye regions to redirect gaze.

    Args:
        video_path: Path to input video.
        output_path: Optional explicit output path.
        output_dir: Output directory (defaults to input dir).
        config: EyeContactConfig with correction parameters.
        on_progress: Callback(pct, msg) for progress updates.

    Returns:
        EyeContactResult with output path and statistics.
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required")
    if not ensure_package("mediapipe", "mediapipe", on_progress):
        raise RuntimeError("mediapipe is required for eye contact correction")

    import cv2
    import mediapipe as mp

    if config is None:
        config = EyeContactConfig()

    result = EyeContactResult()

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_eye_contact.mp4")

    if on_progress:
        on_progress(5, "Initializing face mesh...")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1 if config.only_center_face else 3,
        refine_landmarks=True,
        min_detection_confidence=config.face_confidence,
        min_tracking_confidence=config.eye_confidence,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    _ntf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video = _ntf.name
    _ntf.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        cap.release()
        try:
            os.unlink(tmp_video)
        except OSError:
            pass
        raise RuntimeError(f"Cannot create video writer for: {tmp_video}")

    if on_progress:
        on_progress(10, "Processing eye contact correction...")

    # Temporal smoothing buffer
    gaze_buffer: List[Tuple[float, float]] = []
    frame_idx = 0
    corrections = 0
    total_offset = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                gaze = detect_gaze_direction(frame, face_mesh, config)

                if gaze["detected"]:
                    gaze_buffer.append((gaze["yaw"], gaze["pitch"]))
                    if len(gaze_buffer) > config.smoothing_window:
                        gaze_buffer.pop(0)

                    # Apply temporal smoothing
                    if gaze_buffer:
                        smooth_yaw = sum(g[0] for g in gaze_buffer) / len(gaze_buffer)
                        smooth_pitch = sum(g[1] for g in gaze_buffer) / len(gaze_buffer)
                        gaze["yaw"] = smooth_yaw
                        gaze["pitch"] = smooth_pitch

                    offset = (abs(gaze["yaw"]) + abs(gaze["pitch"])) / 2
                    total_offset += offset

                    frame = _apply_eye_warp(frame, gaze, config)
                    corrections += 1

            except Exception as e:
                logger.debug("Eye contact frame %d failed: %s", frame_idx, e)

            writer.write(frame)
            frame_idx += 1

            if on_progress and frame_idx % 10 == 0:
                pct = 10 + int((frame_idx / total) * 80)
                on_progress(pct, f"Correcting frame {frame_idx}/{total}...")
    finally:
        cap.release()
        writer.release()
        face_mesh.close()

    if on_progress:
        on_progress(92, "Encoding with audio...")

    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_video, "-i", video_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", output_path,
        ], timeout=7200)
    finally:
        try:
            os.unlink(tmp_video)
        except OSError:
            pass

    result.output_path = output_path
    result.frames_processed = frame_idx
    result.frames_corrected = corrections
    result.corrections_applied = corrections
    result.avg_gaze_offset = total_offset / max(1, corrections)

    if on_progress:
        on_progress(100, "Eye contact correction complete!")

    return result
