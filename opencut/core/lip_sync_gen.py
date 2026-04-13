"""
OpenCut AI Lip Sync Module v0.1.0

Replace mouth movements in video to match new/replacement audio:
- Detect mouth/jaw region via MediaPipe Face Mesh
- Generate lip sync frames using audio-driven mouth shape estimation
- Blend synthesized mouth region back onto original face
- Falls back to simple jaw-movement overlay when full model unavailable

Uses frame-by-frame processing with FFmpeg reassembly.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from opencut.helpers import ensure_package, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Configuration / Result
# ---------------------------------------------------------------------------
@dataclass
class LipSyncConfig:
    """Configuration for lip sync generation."""
    face_confidence: float = 0.5    # Minimum face detection confidence
    blend_radius: int = 15          # Pixel radius for mouth region blending
    jaw_sensitivity: float = 1.0    # Jaw movement sensitivity multiplier
    smooth_frames: int = 3          # Temporal smoothing window
    only_lower_face: bool = True    # Only modify lower face region


@dataclass
class LipSyncResult:
    """Result of lip sync operation."""
    output_path: str = ""
    frames_processed: int = 0
    frames_synced: int = 0
    audio_duration: float = 0.0


# ---------------------------------------------------------------------------
# Mouth Region Detection
# ---------------------------------------------------------------------------

# MediaPipe Face Mesh mouth landmark indices
_MOUTH_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]
_MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14]
_JAW = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]
_UPPER_LIP_TOP = [0, 37, 39, 40, 185, 61]
_LOWER_LIP_BOTTOM = [17, 314, 405, 321, 375, 291]


def detect_mouth_region(frame, face_mesh=None) -> Dict:
    """
    Detect mouth region in a single frame using MediaPipe Face Mesh.

    Returns dict with:
        - detected: bool
        - mouth_center: (x, y) in pixels
        - mouth_bbox: (x1, y1, x2, y2) bounding box in pixels
        - mouth_open_ratio: float (0=closed, 1=wide open)
        - jaw_landmarks: list of (x, y) points
        - mouth_landmarks: list of (x, y) points
    """
    import cv2

    h, w = frame.shape[:2]
    result = {
        "detected": False,
        "mouth_center": None,
        "mouth_bbox": None,
        "mouth_open_ratio": 0.0,
        "jaw_landmarks": [],
        "mouth_landmarks": [],
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

    # Extract mouth landmark pixel positions
    mouth_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in _MOUTH_OUTER]
    jaw_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in _JAW]

    if not mouth_pts:
        return result

    xs = [p[0] for p in mouth_pts]
    ys = [p[1] for p in mouth_pts]
    cx = sum(xs) // len(xs)
    cy = sum(ys) // len(ys)

    # Bounding box with padding
    pad = 15
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)

    # Mouth open ratio: vertical distance between upper and lower lip
    upper_y = sum(lm[i].y for i in _UPPER_LIP_TOP) / len(_UPPER_LIP_TOP)
    lower_y = sum(lm[i].y for i in _LOWER_LIP_BOTTOM) / len(_LOWER_LIP_BOTTOM)
    mouth_height = abs(lower_y - upper_y) * h
    mouth_width = abs(max(xs) - min(xs))
    open_ratio = min(1.0, mouth_height / max(1, mouth_width) * 2.0)

    result.update({
        "detected": True,
        "mouth_center": (cx, cy),
        "mouth_bbox": (x1, y1, x2, y2),
        "mouth_open_ratio": open_ratio,
        "jaw_landmarks": jaw_pts,
        "mouth_landmarks": mouth_pts,
    })

    return result


# ---------------------------------------------------------------------------
# Audio-driven mouth shapes
# ---------------------------------------------------------------------------
def _extract_audio_energy(audio_path: str, fps: float) -> List[float]:
    """Extract per-frame audio energy levels from a WAV file."""
    ensure_package("numpy", "numpy")
    import struct
    import wave

    import numpy as np

    try:
        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception:
        return []

    if sampwidth == 2:
        fmt = f"<{n_frames * n_channels}h"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float32)
    else:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Compute RMS energy per video frame
    samples_per_frame = int(framerate / fps)
    energies = []
    for i in range(0, len(samples), samples_per_frame):
        chunk = samples[i:i + samples_per_frame]
        if len(chunk) == 0:
            break
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        energies.append(rms)

    # Normalize
    max_e = max(energies) if energies else 1.0
    if max_e > 0:
        energies = [e / max_e for e in energies]

    return energies


def _apply_mouth_shape(frame, mouth_info: Dict, energy: float,
                       config: LipSyncConfig):
    """Apply audio-driven mouth shape modification to a frame."""
    import cv2
    import numpy as np

    if not mouth_info["detected"] or mouth_info["mouth_bbox"] is None:
        return frame

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = mouth_info["mouth_bbox"]
    cx, cy = mouth_info["mouth_center"]

    if x2 <= x1 or y2 <= y1:
        return frame

    # Jaw movement scale based on audio energy
    jaw_scale = energy * config.jaw_sensitivity
    jaw_shift = int(jaw_scale * 8)  # pixels of vertical stretch

    if jaw_shift < 1:
        return frame

    roi = frame[y1:y2, x1:x2].copy()
    rh, rw = roi.shape[:2]

    if rh < 4 or rw < 4:
        return frame

    # Create vertical stretch map centered on mouth
    gy, gx = np.mgrid[0:rh, 0:rw].astype(np.float32)
    center_y = rh / 2.0
    center_x = rw / 2.0

    # Gaussian weight: strongest at center of mouth
    sigma_x = rw / 2.5
    sigma_y = rh / 2.5
    weight = np.exp(-((gx - center_x) ** 2 / (2 * sigma_x ** 2) +
                       (gy - center_y) ** 2 / (2 * sigma_y ** 2)))

    # Vertical displacement (open jaw)
    map_x = gx
    map_y = gy - jaw_shift * weight * np.sign(gy - center_y)

    warped = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)

    # Feathered blending
    blend_mask = weight[:, :, np.newaxis] * 0.85
    blended = (warped * blend_mask + roi * (1 - blend_mask)).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended

    return frame


# ---------------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------------
def apply_lip_sync(
    video_path: str,
    audio_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    config: Optional[LipSyncConfig] = None,
    on_progress: Optional[Callable] = None,
) -> LipSyncResult:
    """
    Apply lip sync to a video using a replacement audio track.

    Detects mouth regions via MediaPipe, then warps the mouth area
    frame-by-frame based on audio energy to simulate lip movement
    matching the new audio.

    Args:
        video_path: Path to input video.
        audio_path: Path to replacement audio track.
        output_path: Optional explicit output path.
        output_dir: Output directory (defaults to input dir).
        config: LipSyncConfig with processing parameters.
        on_progress: Callback(pct, msg) for progress updates.

    Returns:
        LipSyncResult with output path and statistics.
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required")
    if not ensure_package("mediapipe", "mediapipe", on_progress):
        raise RuntimeError("mediapipe is required for lip sync")

    import cv2
    import mediapipe as mp

    if config is None:
        config = LipSyncConfig()

    result = LipSyncResult()

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_lipsync.mp4")

    if on_progress:
        on_progress(5, "Initializing lip sync...")

    # Extract audio energy profile
    _ntf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav = _ntf.name
    _ntf.close()

    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", audio_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            tmp_wav,
        ], timeout=120)

        info = get_video_info(video_path)
        fps = info.get("fps", 30)
        audio_info = get_video_info(audio_path)
        result.audio_duration = audio_info.get("duration", 0)

        if on_progress:
            on_progress(10, "Analyzing audio energy...")

        energies = _extract_audio_energy(tmp_wav, fps)
    finally:
        try:
            os.unlink(tmp_wav)
        except OSError:
            pass

    # Initialize face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=config.face_confidence,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    _ntf2 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video = _ntf2.name
    _ntf2.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, vid_fps, (orig_w, orig_h))
    if not writer.isOpened():
        cap.release()
        try:
            os.unlink(tmp_video)
        except OSError:
            pass
        raise RuntimeError(f"Cannot create video writer for: {tmp_video}")

    if on_progress:
        on_progress(15, "Processing lip sync frames...")

    frame_idx = 0
    synced = 0
    energy_buffer: List[float] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            energy = energies[frame_idx] if frame_idx < len(energies) else 0.0

            # Temporal smoothing
            energy_buffer.append(energy)
            if len(energy_buffer) > config.smooth_frames:
                energy_buffer.pop(0)
            smooth_energy = sum(energy_buffer) / len(energy_buffer)

            try:
                mouth = detect_mouth_region(frame, face_mesh)
                if mouth["detected"]:
                    frame = _apply_mouth_shape(frame, mouth, smooth_energy, config)
                    synced += 1
            except Exception as e:
                logger.debug("Lip sync frame %d failed: %s", frame_idx, e)

            writer.write(frame)
            frame_idx += 1

            if on_progress and frame_idx % 10 == 0:
                pct = 15 + int((frame_idx / total) * 75)
                on_progress(pct, f"Syncing frame {frame_idx}/{total}...")
    finally:
        cap.release()
        writer.release()
        face_mesh.close()

    if on_progress:
        on_progress(92, "Encoding with new audio...")

    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_video, "-i", audio_path,
            "-map", "0:v", "-map", "1:a",
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
    result.frames_synced = synced

    if on_progress:
        on_progress(100, "Lip sync complete!")

    return result
