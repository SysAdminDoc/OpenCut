"""
OpenCut AI Face Reshape Module v1.29.0

Liquify-style face warping using MediaPipe FaceMesh landmarks.

Operations:
  slim_face       — Translate outer jawline landmarks inward (face slimming).
  enlarge_eyes    — Push eye-corner landmarks outward (eye opening).
  shrink_nose     — Pull nose wing/tip landmarks toward the nose bridge.
  raise_cheekbones — Lift mid-cheek landmarks upward.
  smooth_jaw      — Smooth/soften jawline contour.

Algorithm:
  1. Extract FaceMesh landmarks for each frame (468 pts).
  2. Compute per-pixel displacement map by scattering landmark deltas
     onto a grid and applying Gaussian smoothing (smooth deformation).
  3. Apply displacement via cv2.remap.
  4. Re-encode output video with audio copy.

Requirements: mediapipe, opencv-python-headless, numpy, torch (optional).
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from opencut.helpers import _try_import, get_ffmpeg_path, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")

INSTALL_HINT = "pip install mediapipe opencv-python-headless numpy"

OPERATIONS = ("slim_face", "enlarge_eyes", "shrink_nose", "raise_cheekbones", "smooth_jaw")

# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark index groups
# (indices into the 468-point canonical face mesh)
# ---------------------------------------------------------------------------

# Outer jawline (left side → right side)
_JAWLINE_INDICES = [
    172, 136, 150, 149, 176, 148, 152, 377, 400, 378,
    379, 365, 397, 288, 361, 323,
]
# Left eye corners
_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133]
# Right eye corners
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263]
# Nose tip and wings
_NOSE = [4, 5, 6, 168, 197, 195, 5, 48, 115, 220, 45, 44, 4, 274, 344, 440, 275, 278]
# Mid-cheek (below eyes)
_CHEEKS = [117, 118, 119, 346, 347, 348]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FaceReshapeResult:
    output: str = ""
    operation: str = ""
    frames_processed: int = 0
    faces_found: int = 0
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "operation", "frames_processed", "faces_found", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_face_reshape_available() -> bool:
    return (
        _try_import("mediapipe") is not None
        and _try_import("cv2") is not None
        and _try_import("numpy") is not None
    )


# ---------------------------------------------------------------------------
# Displacement field helpers
# ---------------------------------------------------------------------------

def _build_displacement_field(
    h: int,
    w: int,
    src_pts: List[Tuple[float, float]],
    dst_pts: List[Tuple[float, float]],
    sigma: float = 60.0,
) -> Tuple:
    """
    Build a smooth pixel-level displacement map (dx, dy) by scattering
    point deltas and applying Gaussian smoothing.

    Returns (map_x, map_y) for cv2.remap (float32).
    """
    import cv2
    import numpy as np

    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    for (sx, sy), (tx, ty) in zip(src_pts, dst_pts):
        px, py = int(round(sx)), int(round(sy))
        if 0 <= px < w and 0 <= py < h:
            dx[py, px] += tx - sx
            dy[py, px] += ty - sy
            weight[py, px] += 1.0

    if weight.max() == 0:
        # No valid points — identity map
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
        return map_x, map_y

    k_size = max(3, int(sigma * 6) | 1)  # must be odd
    dx_smooth = cv2.GaussianBlur(dx, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)
    dy_smooth = cv2.GaussianBlur(dy, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)
    w_smooth = cv2.GaussianBlur(weight, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)

    # Normalise displacement by smoothed weight to avoid scale issues
    safe_w = np.where(w_smooth > 1e-6, w_smooth, 1.0)
    dx_norm = dx_smooth / safe_w * (weight.max() > 0)
    dy_norm = dy_smooth / safe_w * (weight.max() > 0)

    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1)) + dx_norm
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w)) + dy_norm

    return map_x, map_y


# ---------------------------------------------------------------------------
# Per-operation landmark displacement
# ---------------------------------------------------------------------------

def _get_deltas(
    lm: List,
    operation: str,
    strength: float,
    h: int,
    w: int,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Given FaceMesh landmark list, return (src_pts, dst_pts) for the operation.

    `lm` entries are landmark objects with .x, .y in [0, 1] normalised coords.
    `strength` is [0, 1].
    """
    src: List[Tuple[float, float]] = []
    dst: List[Tuple[float, float]] = []

    def _lm_px(idx: int) -> Tuple[float, float]:
        p = lm[idx]
        return p.x * w, p.y * h

    def _centroid(indices: List[int]) -> Tuple[float, float]:
        pts = [_lm_px(i) for i in indices]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        return cx, cy

    if operation == "slim_face":
        cx, _ = _centroid(list(range(0, 468)))
        for idx in _JAWLINE_INDICES:
            sx, sy = _lm_px(idx)
            # Move toward horizontal centre
            tx = sx + (cx - sx) * strength * 0.35
            ty = sy
            src.append((sx, sy))
            dst.append((tx, ty))

    elif operation == "enlarge_eyes":
        for eye_indices in (_LEFT_EYE, _RIGHT_EYE):
            ecx, ecy = _centroid(eye_indices)
            for idx in eye_indices:
                sx, sy = _lm_px(idx)
                tx = sx + (sx - ecx) * strength * 0.4
                ty = sy + (sy - ecy) * strength * 0.4
                src.append((sx, sy))
                dst.append((tx, ty))

    elif operation == "shrink_nose":
        ncx, ncy = _centroid(_NOSE)
        for idx in _NOSE:
            sx, sy = _lm_px(idx)
            tx = sx + (ncx - sx) * strength * 0.3
            ty = sy + (ncy - sy) * strength * 0.2
            src.append((sx, sy))
            dst.append((tx, ty))

    elif operation == "raise_cheekbones":
        for idx in _CHEEKS:
            sx, sy = _lm_px(idx)
            tx = sx
            ty = sy - h * 0.02 * strength
            src.append((sx, sy))
            dst.append((tx, ty))

    elif operation == "smooth_jaw":
        # Smooth the jawline by blending toward the average jawline position
        jaw_pts = [_lm_px(idx) for idx in _JAWLINE_INDICES]
        avg_y = sum(p[1] for p in jaw_pts) / len(jaw_pts)
        for idx, (sx, sy) in zip(_JAWLINE_INDICES, jaw_pts):
            ty = sy + (avg_y - sy) * strength * 0.3
            src.append((sx, sy))
            dst.append((sx, ty))

    return src, dst


# ---------------------------------------------------------------------------
# Core reshape function
# ---------------------------------------------------------------------------

def reshape(
    video_path: str,
    operation: str = "slim_face",
    strength: float = 0.5,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> FaceReshapeResult:
    """
    Apply face reshaping to a video.

    Args:
        video_path: Input video.
        operation:  One of OPERATIONS.
        strength:   Warp intensity 0–1.
        output:     Output path.  Auto-generated if None.
        on_progress: Progress callback ``(percent, message)``.
    """
    if not check_face_reshape_available():
        raise RuntimeError(f"face_reshape dependencies not installed.\n{INSTALL_HINT}")
    if operation not in OPERATIONS:
        raise ValueError(f"operation must be one of {OPERATIONS}")
    strength = float(max(0.0, min(1.0, strength)))

    import cv2
    import mediapipe as mp
    if on_progress:
        on_progress(5, "Initialising FaceMesh model...")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if output is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.expanduser("~"), ".opencut")
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, f"{base}_{operation}.mp4")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    h, w = info.get("height", 480), info.get("width", 640)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frames_processed = 0
    faces_found = 0

    try:
        if on_progress:
            on_progress(10, f"Processing {total} frames ({operation})...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = frame[:, :, ::-1]
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                src_pts, dst_pts = _get_deltas(lm, operation, strength, h, w)
                if src_pts:
                    map_x, map_y = _build_displacement_field(h, w, src_pts, dst_pts)
                    frame = cv2.remap(
                        frame, map_x, map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101,
                    )
                    faces_found += 1

            writer.write(frame)
            frames_processed += 1
            if on_progress and frames_processed % 15 == 0:
                pct = 10 + int((frames_processed / total) * 82)
                on_progress(pct, f"Frame {frames_processed}/{total}")
    finally:
        cap.release()
        writer.release()
        face_mesh.close()

    if on_progress:
        on_progress(93, "Encoding output with audio...")

    try:
        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_path, "-i", video_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", output,
        ], timeout=14400)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if on_progress:
        on_progress(100, f"Face reshape complete — {faces_found}/{frames_processed} frames with faces")

    logger.info(
        "face_reshape: op=%s strength=%.2f faces=%d/%d output=%s",
        operation, strength, faces_found, frames_processed, output,
    )

    return FaceReshapeResult(
        output=output,
        operation=operation,
        frames_processed=frames_processed,
        faces_found=faces_found,
        notes=[
            f"operation: {operation}",
            f"strength: {strength:.2f}",
            f"faces detected in {faces_found}/{frames_processed} frames",
        ],
    )


__all__ = [
    "FaceReshapeResult",
    "INSTALL_HINT",
    "OPERATIONS",
    "check_face_reshape_available",
    "reshape",
]
