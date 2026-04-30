"""
OpenCut AI Skin Retouching Module v1.29.0

Automated skin retouching and blemish removal for video.

Pipeline:
  1. Detect face region per-frame (MediaPipe FaceDetection or Haar fallback).
  2. Build a skin-tone mask in HSV colour space.
  3. Apply frequency-separation smoothing:
       - Low-frequency layer  → bilateral filter (preserves edges, smooths skin)
       - High-frequency layer → keep (preserves pores, fine texture)
     Blend: output = low_freq * intensity + original * (1 - intensity)
  4. Optionally brighten face region slightly (radiance boost).
  5. Re-encode with audio copy.

Optional GFPGAN mode: uses `gfpgan` package for deep face restoration
on top of the bilateral pass.  Gated by check_gfpgan_available().

Requirements: mediapipe (or cv2 only for haar fallback), opencv-python-headless, numpy.
Optional:     gfpgan, torch.
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
GFPGAN_HINT = "pip install gfpgan torch  # for deep face restoration mode"

# HSV skin colour range (broad to cover various skin tones)
_SKIN_HSV_LOWER = (0, 20, 70)
_SKIN_HSV_UPPER = (35, 255, 255)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SkinRetouchResult:
    output: str = ""
    frames_processed: int = 0
    faces_found: int = 0
    mode: str = "bilateral"
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("output", "frames_processed", "faces_found", "mode", "notes")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_skin_retouch_available() -> bool:
    return (
        _try_import("cv2") is not None
        and _try_import("numpy") is not None
    )


def check_gfpgan_available() -> bool:
    return (
        _try_import("gfpgan") is not None
        and _try_import("torch") is not None
    )


# ---------------------------------------------------------------------------
# Face detection helpers
# ---------------------------------------------------------------------------

def _detect_face_rect(frame, mp_det) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the dominant face using MediaPipe FaceDetection.
    Returns (x, y, w, h) or None.
    """
    h_img, w_img = frame.shape[:2]
    rgb = frame[:, :, ::-1]
    results = mp_det.process(rgb)
    if not results.detections:
        return None
    det = max(results.detections, key=lambda d: d.score[0])
    bb = det.location_data.relative_bounding_box
    x = max(0, int(bb.xmin * w_img))
    y = max(0, int(bb.ymin * h_img))
    bw = min(w_img - x, int(bb.width * w_img))
    bh = min(h_img - y, int(bb.height * h_img))
    # Expand ROI slightly
    pad_x = int(bw * 0.1)
    pad_y = int(bh * 0.1)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    bw = min(w_img - x, bw + 2 * pad_x)
    bh = min(h_img - y, bh + 2 * pad_y)
    if bw < 20 or bh < 20:
        return None
    return (x, y, bw, bh)


def _detect_face_haar(frame, cascade) -> Optional[Tuple[int, int, int, int]]:
    """Haar cascade fallback. Returns largest detected face or None."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    # Pick largest
    x, y, bw, bh = max(faces, key=lambda r: r[2] * r[3])
    return (int(x), int(y), int(bw), int(bh))


# ---------------------------------------------------------------------------
# Skin mask
# ---------------------------------------------------------------------------

def _skin_mask(roi):
    """Build a binary mask of skin pixels in the face ROI (HSV range)."""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(_SKIN_HSV_LOWER), np.array(_SKIN_HSV_UPPER))
    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# ---------------------------------------------------------------------------
# Frequency-separation smoothing
# ---------------------------------------------------------------------------

def _bilateral_smooth(roi, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0):
    """Apply bilateral filter (edge-preserving smoothing)."""
    import cv2
    return cv2.bilateralFilter(roi, d, sigma_color, sigma_space)


def _frequency_separation_blend(original_roi, smooth_roi, mask, intensity: float):
    """
    Blend smooth and original using frequency separation.

    low_freq  = smooth_roi  (broad colour/tone)
    high_freq = original - smooth_roi  (fine texture)
    output    = low_freq * intensity + (low_freq + high_freq) * (1 - intensity)
              = smooth_roi * intensity + original_roi * (1 - intensity)
    with the skin mask applied — non-skin areas stay unchanged.
    """
    import cv2
    import numpy as np
    orig_f = original_roi.astype(np.float32)
    smth_f = smooth_roi.astype(np.float32)
    blended_f = smth_f * intensity + orig_f * (1.0 - intensity)
    blended = np.clip(blended_f, 0, 255).astype(np.uint8)

    # Mask: only apply to skin pixels
    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
    out_f = blended.astype(np.float32) * mask_3ch + orig_f * (1.0 - mask_3ch)
    return np.clip(out_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# GFPGAN pass
# ---------------------------------------------------------------------------

def _gfpgan_enhance_frame(frame, restorer, scale: int = 1):
    """Run one frame through GFPGAN restorer. Returns enhanced frame."""
    import cv2
    _, _, restored_imgs = restorer.enhance(
        frame,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    if restored_imgs:
        out = restored_imgs[0]
        if out.shape != frame.shape:
            out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
        return out
    return frame


# ---------------------------------------------------------------------------
# Main retouch function
# ---------------------------------------------------------------------------

def retouch(
    video_path: str,
    intensity: float = 0.6,
    mode: str = "bilateral",
    radiance: float = 0.0,
    output: Optional[str] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> SkinRetouchResult:
    """
    Skin retouching for video.

    Args:
        video_path: Input video.
        intensity:  Retouching strength 0–1.
        mode:       "bilateral" (fast) or "gfpgan" (deep, GPU recommended).
        radiance:   Subtle brightness boost on face 0–1.  0 = disabled.
        output:     Output path.  Auto-generated if None.
        on_progress: Callback ``(percent, message)``.
    """
    if not check_skin_retouch_available():
        raise RuntimeError(f"skin_retouch dependencies not installed.\n{INSTALL_HINT}")
    if mode == "gfpgan" and not check_gfpgan_available():
        logger.warning("GFPGAN not available — falling back to bilateral mode")
        mode = "bilateral"

    import cv2
    import numpy as np

    intensity = float(max(0.0, min(1.0, intensity)))
    radiance = float(max(0.0, min(1.0, radiance)))

    if output is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.expanduser("~"), ".opencut")
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, f"{base}_retouched.mp4")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    h, w = info.get("height", 480), info.get("width", 640)

    # --- Set up face detector ---
    use_mediapipe = _try_import("mediapipe") is not None
    mp_det = None
    haar_cascade = None

    if use_mediapipe:
        import mediapipe as mp_module
        mp_det = mp_module.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
    else:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        haar_cascade = cv2.CascadeClassifier(cascade_path)

    # --- Set up GFPGAN restorer if needed ---
    gfpgan_restorer = None
    if mode == "gfpgan":
        if on_progress:
            on_progress(5, "Loading GFPGAN model...")
        try:
            from gfpgan import GFPGANer
            gfpgan_restorer = GFPGANer(
                model_path=None,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
            )
        except Exception as exc:
            logger.warning("GFPGAN init failed (%s) — falling back to bilateral", exc)
            mode = "bilateral"
            gfpgan_restorer = None

    # --- Video I/O ---
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
            on_progress(10, f"Retouching {total} frames (mode={mode}, intensity={intensity:.0%})...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if mode == "gfpgan" and gfpgan_restorer is not None:
                try:
                    frame = _gfpgan_enhance_frame(frame, gfpgan_restorer)
                    faces_found += 1
                except Exception as exc:
                    logger.debug("GFPGAN frame error: %s", exc)
            else:
                # Detect face rect
                face_rect = None
                if mp_det is not None:
                    face_rect = _detect_face_rect(frame, mp_det)
                elif haar_cascade is not None:
                    face_rect = _detect_face_haar(frame, haar_cascade)

                if face_rect is not None:
                    x, y, bw, bh = face_rect
                    roi = frame[y:y + bh, x:x + bw]
                    skin_mask = _skin_mask(roi)
                    smooth_roi = _bilateral_smooth(roi, d=9,
                                                   sigma_color=75.0 * intensity,
                                                   sigma_space=75.0 * intensity)
                    blended = _frequency_separation_blend(roi, smooth_roi, skin_mask, intensity)

                    if radiance > 0:
                        brightness = np.ones_like(blended, dtype=np.float32) * (1 + radiance * 0.15)
                        bright = np.clip(blended.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
                        # Only brighten skin areas
                        skin_3ch = cv2.merge([skin_mask, skin_mask, skin_mask]).astype(np.float32) / 255.0
                        blended_f = bright.astype(np.float32) * skin_3ch + blended.astype(np.float32) * (1 - skin_3ch)
                        blended = np.clip(blended_f, 0, 255).astype(np.uint8)

                    frame[y:y + bh, x:x + bw] = blended
                    faces_found += 1

            writer.write(frame)
            frames_processed += 1
            if on_progress and frames_processed % 20 == 0:
                pct = 10 + int((frames_processed / total) * 83)
                on_progress(pct, f"Frame {frames_processed}/{total}")
    finally:
        cap.release()
        writer.release()
        if mp_det is not None:
            mp_det.close()
        if gfpgan_restorer is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    if on_progress:
        on_progress(94, "Encoding output with audio...")

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
        on_progress(100, f"Skin retouch complete — {faces_found}/{frames_processed} frames with faces")

    logger.info(
        "skin_retouch: mode=%s intensity=%.2f faces=%d/%d output=%s",
        mode, intensity, faces_found, frames_processed, output,
    )

    return SkinRetouchResult(
        output=output,
        frames_processed=frames_processed,
        faces_found=faces_found,
        mode=mode,
        notes=[
            f"mode: {mode}",
            f"intensity: {intensity:.2f}",
            f"faces detected in {faces_found}/{frames_processed} frames",
        ],
    )


__all__ = [
    "SkinRetouchResult",
    "INSTALL_HINT",
    "GFPGAN_HINT",
    "check_skin_retouch_available",
    "check_gfpgan_available",
    "retouch",
]
