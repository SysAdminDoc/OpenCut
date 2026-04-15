"""
AI Eye Contact Correction

Detect face and eye landmarks using MediaPipe Face Mesh, estimate gaze
direction from iris position, and apply affine corrections to redirect
gaze toward the camera.  Supports temporal smoothing and intensity control.
"""

import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# MediaPipe Face Mesh landmark indices
# Left eye corners + iris center
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
LEFT_IRIS_CENTER = 468
# Right eye corners + iris center
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_IRIS_CENTER = 473
# Eye top/bottom for bounding box
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


@dataclass
class GazeEstimate:
    """Gaze direction estimate for a single frame."""
    frame_index: int = 0
    left_gaze_x: float = 0.0
    left_gaze_y: float = 0.0
    right_gaze_x: float = 0.0
    right_gaze_y: float = 0.0
    face_detected: bool = False
    correction_magnitude: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "left_gaze": [self.left_gaze_x, self.left_gaze_y],
            "right_gaze": [self.right_gaze_x, self.right_gaze_y],
            "face_detected": self.face_detected,
            "correction_magnitude": round(self.correction_magnitude, 4),
        }


@dataclass
class EyeContactResult:
    """Result of eye contact correction."""
    output_path: str = ""
    frames_processed: int = 0
    faces_detected: int = 0
    average_correction_magnitude: float = 0.0
    intensity: float = 1.0

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "frames_processed": self.frames_processed,
            "faces_detected": self.faces_detected,
            "average_correction_magnitude": round(self.average_correction_magnitude, 4),
            "intensity": self.intensity,
        }


# ---------------------------------------------------------------------------
# Gaze estimation
# ---------------------------------------------------------------------------
def _estimate_gaze_from_landmarks(
    landmarks: list,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    """Estimate gaze direction from face mesh landmarks.

    Returns (left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y)
    where values represent offset from eye center (0 = centered/looking at camera).
    """
    def _lm(idx):
        lm = landmarks[idx]
        return (lm.x * image_width, lm.y * image_height)

    # Left eye gaze: iris position relative to eye corners
    l_inner = _lm(LEFT_EYE_INNER)
    l_outer = _lm(LEFT_EYE_OUTER)
    l_iris = _lm(LEFT_IRIS_CENTER)
    l_top = _lm(LEFT_EYE_TOP)
    l_bottom = _lm(LEFT_EYE_BOTTOM)

    l_eye_width = max(1.0, abs(l_inner[0] - l_outer[0]))
    l_eye_height = max(1.0, abs(l_top[1] - l_bottom[1]))
    l_center_x = (l_inner[0] + l_outer[0]) / 2
    l_center_y = (l_top[1] + l_bottom[1]) / 2

    left_gaze_x = (l_iris[0] - l_center_x) / l_eye_width
    left_gaze_y = (l_iris[1] - l_center_y) / l_eye_height

    # Right eye gaze
    r_inner = _lm(RIGHT_EYE_INNER)
    r_outer = _lm(RIGHT_EYE_OUTER)
    r_iris = _lm(RIGHT_IRIS_CENTER)
    r_top = _lm(RIGHT_EYE_TOP)
    r_bottom = _lm(RIGHT_EYE_BOTTOM)

    r_eye_width = max(1.0, abs(r_inner[0] - r_outer[0]))
    r_eye_height = max(1.0, abs(r_top[1] - r_bottom[1]))
    r_center_x = (r_inner[0] + r_outer[0]) / 2
    r_center_y = (r_top[1] + r_bottom[1]) / 2

    right_gaze_x = (r_iris[0] - r_center_x) / r_eye_width
    right_gaze_y = (r_iris[1] - r_center_y) / r_eye_height

    return (left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y)


def _compute_correction_magnitude(gaze: Tuple[float, float, float, float]) -> float:
    """Compute the magnitude of gaze correction needed."""
    lx, ly, rx, ry = gaze
    left_mag = math.sqrt(lx * lx + ly * ly)
    right_mag = math.sqrt(rx * rx + ry * ry)
    return (left_mag + right_mag) / 2


# ---------------------------------------------------------------------------
# Temporal smoothing (exponential moving average)
# ---------------------------------------------------------------------------
class GazeSmoother:
    """Exponential moving average smoother for gaze vectors."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = max(0.01, min(1.0, alpha))
        self._prev: Optional[Tuple[float, float, float, float]] = None

    def smooth(
        self, gaze: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        """Apply EMA smoothing to gaze vector."""
        if self._prev is None:
            self._prev = gaze
            return gaze

        smoothed = tuple(
            self.alpha * cur + (1 - self.alpha) * prev
            for cur, prev in zip(gaze, self._prev)
        )
        self._prev = smoothed
        return smoothed  # type: ignore[return-value]

    def reset(self):
        self._prev = None


# ---------------------------------------------------------------------------
# Eye region correction via affine transform
# ---------------------------------------------------------------------------
def _correct_eye_region(
    frame,  # numpy array (H, W, 3)
    landmarks: list,
    eye_indices: Dict[str, int],
    gaze_offset: Tuple[float, float],
    intensity: float,
    image_width: int,
    image_height: int,
):
    """Apply affine transform to eye region to correct gaze direction.

    Args:
        frame: Video frame (modified in-place).
        landmarks: Face mesh landmarks.
        eye_indices: Dict with 'inner', 'outer', 'top', 'bottom', 'iris' keys.
        gaze_offset: (gaze_x, gaze_y) offset to correct.
        intensity: Correction intensity 0.0 - 1.0.
        image_width: Frame width.
        image_height: Frame height.
    """
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401

    def _lm(idx):
        lm = landmarks[idx]
        return (lm.x * image_width, lm.y * image_height)

    inner = _lm(eye_indices["inner"])
    outer = _lm(eye_indices["outer"])
    top = _lm(eye_indices["top"])
    bottom = _lm(eye_indices["bottom"])

    # Compute eye bounding box with padding
    pad = 5
    x_min = int(max(0, min(inner[0], outer[0]) - pad))
    x_max = int(min(image_width, max(inner[0], outer[0]) + pad))
    y_min = int(max(0, min(top[1], bottom[1]) - pad))
    y_max = int(min(image_height, max(top[1], bottom[1]) + pad))

    roi_w = x_max - x_min
    roi_h = y_max - y_min
    if roi_w < 4 or roi_h < 4:
        return

    # Compute translation to shift iris toward center
    gaze_x, gaze_y = gaze_offset
    shift_x = -gaze_x * roi_w * intensity * 0.3
    shift_y = -gaze_y * roi_h * intensity * 0.3

    # Affine transform matrix (translation)
    m_matrix = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y],
    ])

    # Extract ROI, apply transform, blend back
    roi = frame[y_min:y_max, x_min:x_max].copy()
    shifted = cv2.warpAffine(roi, m_matrix, (roi_w, roi_h), borderMode=cv2.BORDER_REPLICATE)

    # Create smooth blend mask (elliptical feather)
    mask = np.zeros((roi_h, roi_w), dtype=np.float32)
    center = (roi_w // 2, roi_h // 2)
    cv2.ellipse(mask, center, (roi_w // 2, roi_h // 2), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=roi_w * 0.15, sigmaY=roi_h * 0.15)
    mask = mask[:, :, np.newaxis]

    blended = (shifted * mask + roi * (1.0 - mask)).astype(np.uint8)
    frame[y_min:y_max, x_min:x_max] = blended


# ---------------------------------------------------------------------------
# Single frame preview
# ---------------------------------------------------------------------------
def preview_eye_contact(
    input_path: str,
    frame_number: int = 0,
    intensity: float = 1.0,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Apply eye contact correction to a single frame for preview.

    Args:
        input_path: Path to video file.
        frame_number: Frame to preview (0-indexed).
        intensity: Correction intensity 0.0 - 1.0.
        output_dir: Output directory for preview image.
        on_progress: Progress callback(pct).

    Returns:
        Dict with preview_path, face_detected, gaze_estimate.
    """
    ensure_package("cv2", "opencv-python-headless")
    ensure_package("mediapipe")

    import cv2  # noqa: F401
    import mediapipe as mp  # noqa: F401

    if on_progress:
        on_progress(10)

    intensity = max(0.0, min(1.0, intensity))

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_number}")

    if on_progress:
        on_progress(30)

    h, w = frame.shape[:2]
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    face_mesh.close()

    gaze_est = GazeEstimate(frame_index=frame_number)

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark
        gaze = _estimate_gaze_from_landmarks(lms, w, h)
        gaze_est.left_gaze_x, gaze_est.left_gaze_y = gaze[0], gaze[1]
        gaze_est.right_gaze_x, gaze_est.right_gaze_y = gaze[2], gaze[3]
        gaze_est.face_detected = True
        gaze_est.correction_magnitude = _compute_correction_magnitude(gaze)

        if on_progress:
            on_progress(60)

        # Apply correction
        _correct_eye_region(
            frame, lms,
            {"inner": LEFT_EYE_INNER, "outer": LEFT_EYE_OUTER,
             "top": LEFT_EYE_TOP, "bottom": LEFT_EYE_BOTTOM, "iris": LEFT_IRIS_CENTER},
            (gaze[0], gaze[1]), intensity, w, h,
        )
        _correct_eye_region(
            frame, lms,
            {"inner": RIGHT_EYE_INNER, "outer": RIGHT_EYE_OUTER,
             "top": RIGHT_EYE_TOP, "bottom": RIGHT_EYE_BOTTOM, "iris": RIGHT_IRIS_CENTER},
            (gaze[2], gaze[3]), intensity, w, h,
        )

    if on_progress:
        on_progress(80)

    out_dir = output_dir or os.path.dirname(input_path)
    preview_path = os.path.join(out_dir, f"eye_contact_preview_{frame_number}.png")
    cv2.imwrite(preview_path, frame)

    if on_progress:
        on_progress(100)

    return {
        "preview_path": preview_path,
        "face_detected": gaze_est.face_detected,
        "gaze_estimate": gaze_est.to_dict(),
    }


# ---------------------------------------------------------------------------
# Full video eye contact correction
# ---------------------------------------------------------------------------
def fix_eye_contact(
    input_path: str,
    intensity: float = 1.0,
    smoothing_alpha: float = 0.3,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> EyeContactResult:
    """Apply eye contact correction to entire video.

    Args:
        input_path: Path to video file.
        intensity: Correction intensity 0.0 - 1.0.
        smoothing_alpha: Temporal smoothing factor (higher = less smoothing).
        output_dir: Output directory.
        on_progress: Progress callback(pct).

    Returns:
        EyeContactResult with output path and statistics.
    """
    ensure_package("cv2", "opencv-python-headless")
    ensure_package("mediapipe")

    import cv2  # noqa: F401
    import mediapipe as mp  # noqa: F401

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if on_progress:
        on_progress(2)

    intensity = max(0.0, min(1.0, intensity))
    smoothing_alpha = max(0.01, min(1.0, smoothing_alpha))

    info = get_video_info(input_path)
    fps = info.get("fps", 30.0)
    total_frames = int(info.get("duration", 0) * fps) or 1

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    # Write corrected frames to temp file, then mux audio
    tmp_dir = tempfile.mkdtemp(prefix="opencut_eye_")
    tmp_video = os.path.join(tmp_dir, "corrected_noaudio.mp4")

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, actual_fps, (w, h))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    smoother = GazeSmoother(alpha=smoothing_alpha)
    frames_processed = 0
    faces_detected = 0
    total_correction = 0.0

    if on_progress:
        on_progress(5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            raw_gaze = _estimate_gaze_from_landmarks(lms, w, h)
            smoothed = smoother.smooth(raw_gaze)
            mag = _compute_correction_magnitude(smoothed)

            _correct_eye_region(
                frame, lms,
                {"inner": LEFT_EYE_INNER, "outer": LEFT_EYE_OUTER,
                 "top": LEFT_EYE_TOP, "bottom": LEFT_EYE_BOTTOM, "iris": LEFT_IRIS_CENTER},
                (smoothed[0], smoothed[1]), intensity, w, h,
            )
            _correct_eye_region(
                frame, lms,
                {"inner": RIGHT_EYE_INNER, "outer": RIGHT_EYE_OUTER,
                 "top": RIGHT_EYE_TOP, "bottom": RIGHT_EYE_BOTTOM, "iris": RIGHT_IRIS_CENTER},
                (smoothed[2], smoothed[3]), intensity, w, h,
            )

            faces_detected += 1
            total_correction += mag
        else:
            smoother.reset()

        writer.write(frame)
        frames_processed += 1

        if on_progress and frames_processed % 10 == 0:
            pct = 5 + int((frames_processed / max(1, total_frames)) * 85)
            on_progress(min(90, pct))

    cap.release()
    writer.release()
    face_mesh.close()

    if on_progress:
        on_progress(92)

    # Mux audio from original into corrected video
    out_path = output_path(input_path, "eyecontact", output_dir)
    cmd = (FFmpegCmd()
           .input(tmp_video)
           .input(input_path)
           .map("0:v", "1:a?")
           .video_codec("libx264", crf=18, preset="fast")
           .audio_codec("copy")
           .faststart()
           .output(out_path)
           .build())

    try:
        run_ffmpeg(cmd)
    finally:
        # Clean up temp
        try:
            os.unlink(tmp_video)
            os.rmdir(tmp_dir)
        except OSError:
            logger.debug("Failed to clean up temp dir: %s", tmp_dir)

    if on_progress:
        on_progress(100)

    avg_correction = total_correction / max(1, faces_detected)

    return EyeContactResult(
        output_path=out_path,
        frames_processed=frames_processed,
        faces_detected=faces_detected,
        average_correction_magnitude=avg_correction,
        intensity=intensity,
    )
