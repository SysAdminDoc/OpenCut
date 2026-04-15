"""
AI Lip Sync — Match mouth movements to replacement audio.

Given a video and replacement audio, modify the mouth region to match.
Simplified approach uses audio amplitude to drive mouth openness (jaw
landmark distance).  Advanced approach delegates to Wav2Lip/SadTalker
subprocess if available.
"""

import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# MediaPipe mouth landmark indices
MOUTH_OUTER_TOP = 13
MOUTH_OUTER_BOTTOM = 14
MOUTH_INNER_TOP = 82
MOUTH_INNER_BOTTOM = 87
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
JAW_TIP = 152
NOSE_TIP = 1
# Full mouth contour indices for ROI
MOUTH_CONTOUR = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]
# Min/max mouth openness ratios for mapping
MOUTH_CLOSED_RATIO = 0.02
MOUTH_OPEN_RATIO = 0.25


@dataclass
class LipSyncResult:
    """Result of lip sync operation."""
    output_path: str = ""
    frames_processed: int = 0
    audio_duration: float = 0.0
    sync_quality_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "frames_processed": self.frames_processed,
            "audio_duration": round(self.audio_duration, 4),
            "sync_quality_score": round(self.sync_quality_score, 4),
        }


# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------
def _extract_audio_features(
    audio_path: str,
    fps: float = 30.0,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Extract amplitude envelope and spectral features per video frame.

    Args:
        audio_path: Path to audio file.
        fps: Video frame rate for alignment.
        on_progress: Progress callback(pct).

    Returns:
        Dict with 'amplitudes' (per-frame), 'duration', 'frame_count'.
    """
    ensure_package("numpy", "numpy")

    import numpy as np  # noqa: F401

    if on_progress:
        on_progress(5)

    # Extract raw PCM via FFmpeg
    tmp_dir = tempfile.mkdtemp(prefix="opencut_ls_")
    raw_pcm = os.path.join(tmp_dir, "audio.raw")

    cmd = (FFmpegCmd()
           .input(audio_path)
           .no_video()
           .option("f", "s16le")
           .option("ar", "16000")
           .option("ac", "1")
           .output(raw_pcm)
           .build())
    run_ffmpeg(cmd)

    # Read raw PCM
    with open(raw_pcm, "rb") as f:
        raw_data = f.read()

    # Clean up
    try:
        os.unlink(raw_pcm)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    if on_progress:
        on_progress(20)

    samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
    sample_rate = 16000
    duration = len(samples) / sample_rate
    samples_per_frame = int(sample_rate / fps)
    frame_count = int(math.ceil(duration * fps))

    if on_progress:
        on_progress(30)

    # Compute per-frame amplitude envelope (RMS)
    amplitudes = []
    for i in range(frame_count):
        start_idx = i * samples_per_frame
        end_idx = min(start_idx + samples_per_frame, len(samples))
        if start_idx >= len(samples):
            amplitudes.append(0.0)
            continue
        chunk = samples[start_idx:end_idx]
        rms = float(np.sqrt(np.mean(chunk ** 2))) / 32768.0
        amplitudes.append(min(1.0, rms * 3.0))  # normalize + boost

    if on_progress:
        on_progress(50)

    # Simple spectral analysis: high-frequency ratio per frame for consonant detection
    spectral_ratios: List[float] = []
    for i in range(frame_count):
        start_idx = i * samples_per_frame
        end_idx = min(start_idx + samples_per_frame, len(samples))
        if start_idx >= len(samples) or end_idx - start_idx < 16:
            spectral_ratios.append(0.0)
            continue
        chunk = samples[start_idx:end_idx]
        fft = np.abs(np.fft.rfft(chunk))
        mid = len(fft) // 2
        if mid == 0:
            spectral_ratios.append(0.0)
            continue
        low_energy = float(np.sum(fft[:mid]))
        high_energy = float(np.sum(fft[mid:]))
        total = low_energy + high_energy
        ratio = high_energy / max(1.0, total)
        spectral_ratios.append(ratio)

    if on_progress:
        on_progress(60)

    return {
        "amplitudes": amplitudes,
        "spectral_ratios": spectral_ratios,
        "duration": duration,
        "frame_count": frame_count,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# Mouth region manipulation
# ---------------------------------------------------------------------------
def _get_mouth_roi(
    landmarks: list,
    image_width: int,
    image_height: int,
    padding: int = 20,
) -> Tuple[int, int, int, int]:
    """Get mouth bounding box from face landmarks.

    Returns (x_min, y_min, x_max, y_max).
    """
    xs = []
    ys = []
    for idx in MOUTH_CONTOUR:
        lm = landmarks[idx]
        xs.append(lm.x * image_width)
        ys.append(lm.y * image_height)

    x_min = int(max(0, min(xs) - padding))
    y_min = int(max(0, min(ys) - padding))
    x_max = int(min(image_width, max(xs) + padding))
    y_max = int(min(image_height, max(ys) + padding))

    return (x_min, y_min, x_max, y_max)


def _compute_mouth_openness(
    landmarks: list,
    image_height: int,
) -> float:
    """Compute mouth openness ratio from landmarks."""
    top = landmarks[MOUTH_OUTER_TOP]
    bottom = landmarks[MOUTH_OUTER_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    vertical = abs(bottom.y - top.y) * image_height
    horizontal = abs(right.x - left.x) * image_height
    if horizontal < 1.0:
        return 0.0
    return vertical / horizontal


def _apply_mouth_transform(
    frame,  # numpy array
    landmarks: list,
    target_openness: float,
    current_openness: float,
    image_width: int,
    image_height: int,
    blend_strength: float = 0.7,
):
    """Transform mouth region to match target openness.

    Applies vertical scaling to mouth ROI based on difference between
    current and target openness.
    """
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401

    x_min, y_min, x_max, y_max = _get_mouth_roi(
        landmarks, image_width, image_height,
    )

    roi_w = x_max - x_min
    roi_h = y_max - y_min
    if roi_w < 10 or roi_h < 10:
        return

    openness_diff = target_openness - current_openness
    if abs(openness_diff) < 0.01:
        return

    # Scale factor: positive diff = open more, negative = close more
    scale_y = 1.0 + openness_diff * 2.0
    scale_y = max(0.6, min(1.5, scale_y))

    roi = frame[y_min:y_max, x_min:x_max].copy()

    # Apply vertical scale around center
    center_y = roi_h // 2
    m_matrix = np.float32([
        [1, 0, 0],
        [0, scale_y, center_y * (1 - scale_y)],
    ])
    scaled = cv2.warpAffine(roi, m_matrix, (roi_w, roi_h), borderMode=cv2.BORDER_REPLICATE)

    # Elliptical blend mask
    mask = np.zeros((roi_h, roi_w), dtype=np.float32)
    center = (roi_w // 2, roi_h // 2)
    axes = (int(roi_w * 0.4), int(roi_h * 0.4))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=roi_w * 0.15, sigmaY=roi_h * 0.15)
    mask = mask[:, :, np.newaxis] * blend_strength

    blended = (scaled * mask + roi * (1.0 - mask)).astype(np.uint8)
    frame[y_min:y_max, x_min:x_max] = blended


# ---------------------------------------------------------------------------
# Check for external lip sync tools
# ---------------------------------------------------------------------------
def _check_wav2lip() -> Optional[str]:
    """Check if Wav2Lip is available. Returns path or None."""
    w2l = shutil.which("wav2lip_inference")
    if w2l:
        return w2l
    # Check common install locations
    for candidate in [
        os.path.expanduser("~/.opencut/models/Wav2Lip/inference.py"),
        os.path.expanduser("~/Wav2Lip/inference.py"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _run_wav2lip(
    video_path: str,
    audio_path: str,
    output_path_file: str,
    on_progress: Optional[Callable] = None,
) -> LipSyncResult:
    """Run Wav2Lip as subprocess for high-quality lip sync."""
    wav2lip_path = _check_wav2lip()
    if not wav2lip_path:
        raise RuntimeError("Wav2Lip not found")

    if on_progress:
        on_progress(10)

    import sys
    cmd = [
        sys.executable, wav2lip_path,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path_file,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(f"Wav2Lip failed: {stderr[-500:]}")

    info = get_video_info(output_path_file)

    if on_progress:
        on_progress(100)

    return LipSyncResult(
        output_path=output_path_file,
        frames_processed=int(info.get("duration", 0) * info.get("fps", 30)),
        audio_duration=info.get("duration", 0),
        sync_quality_score=0.85,  # Wav2Lip typically high quality
    )


# ---------------------------------------------------------------------------
# Single frame preview
# ---------------------------------------------------------------------------
def preview_lip_sync(
    video_path: str,
    audio_path: str,
    frame_number: int = 0,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Preview lip sync on a single frame.

    Args:
        video_path: Path to video file.
        audio_path: Path to replacement audio.
        frame_number: Frame to preview.
        output_dir: Output directory.
        on_progress: Progress callback(pct).

    Returns:
        Dict with preview_path, face_detected, mouth_openness.
    """
    ensure_package("cv2", "opencv-python-headless")
    ensure_package("mediapipe")
    ensure_package("numpy", "numpy")

    import cv2  # noqa: F401
    import mediapipe as mp  # noqa: F401
    import numpy as np  # noqa: F401

    if on_progress:
        on_progress(10)

    # Extract audio features
    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    features = _extract_audio_features(audio_path, fps)

    if on_progress:
        on_progress(40)

    # Read target frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_number}")

    h, w = frame.shape[:2]

    # Get target amplitude for this frame
    target_amp = 0.0
    if frame_number < len(features["amplitudes"]):
        target_amp = features["amplitudes"][frame_number]

    if on_progress:
        on_progress(60)

    # Detect face and apply mouth transform
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    face_mesh.close()

    face_detected = False
    mouth_openness = 0.0

    if results.multi_face_landmarks:
        face_detected = True
        lms = results.multi_face_landmarks[0].landmark
        mouth_openness = _compute_mouth_openness(lms, h)

        # Map amplitude to target openness
        target_openness = MOUTH_CLOSED_RATIO + target_amp * (MOUTH_OPEN_RATIO - MOUTH_CLOSED_RATIO)
        _apply_mouth_transform(frame, lms, target_openness, mouth_openness, w, h)

    if on_progress:
        on_progress(85)

    out_dir = output_dir or os.path.dirname(video_path)
    preview_path = os.path.join(out_dir, f"lip_sync_preview_{frame_number}.png")
    cv2.imwrite(preview_path, frame)

    if on_progress:
        on_progress(100)

    return {
        "preview_path": preview_path,
        "face_detected": face_detected,
        "mouth_openness": round(mouth_openness, 4),
        "target_amplitude": round(target_amp, 4),
    }


# ---------------------------------------------------------------------------
# Full video lip sync
# ---------------------------------------------------------------------------
def apply_lip_sync(
    video_path: str,
    audio_path: str,
    use_external: bool = True,
    blend_strength: float = 0.7,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> LipSyncResult:
    """Apply lip sync to video with replacement audio.

    Tries Wav2Lip if available and use_external=True, otherwise uses
    built-in amplitude-driven mouth animation.

    Args:
        video_path: Path to video file.
        audio_path: Path to replacement audio.
        use_external: Try external tools (Wav2Lip) first.
        blend_strength: Mouth blend intensity 0.0-1.0.
        output_dir: Output directory.
        on_progress: Progress callback(pct).

    Returns:
        LipSyncResult with output path and statistics.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    out_path = output_path(video_path, "lipsync", output_dir)

    # Try external tool first
    if use_external and _check_wav2lip():
        try:
            logger.info("Using Wav2Lip for lip sync")
            return _run_wav2lip(video_path, audio_path, out_path, on_progress)
        except Exception as e:
            logger.warning("Wav2Lip failed, falling back to built-in: %s", e)

    # Built-in amplitude-driven lip sync
    return _builtin_lip_sync(
        video_path, audio_path, out_path,
        blend_strength=blend_strength,
        on_progress=on_progress,
    )


def _builtin_lip_sync(
    video_path: str,
    audio_path: str,
    out_path: str,
    blend_strength: float = 0.7,
    on_progress: Optional[Callable] = None,
) -> LipSyncResult:
    """Built-in amplitude-driven lip sync."""
    ensure_package("cv2", "opencv-python-headless")
    ensure_package("mediapipe")
    ensure_package("numpy", "numpy")

    import cv2  # noqa: F401
    import mediapipe as mp  # noqa: F401
    import numpy as np  # noqa: F401

    if on_progress:
        on_progress(2)

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)

    # Extract audio features
    features = _extract_audio_features(audio_path, fps)
    amplitudes = features["amplitudes"]
    audio_duration = features["duration"]

    if on_progress:
        on_progress(15)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or features["frame_count"]

    # Write to temp, then mux with audio
    tmp_dir = tempfile.mkdtemp(prefix="opencut_ls_")
    tmp_video = os.path.join(tmp_dir, "lipsync_noaudio.mp4")

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, actual_fps, (w, h))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames_processed = 0
    quality_scores: List[float] = []

    if on_progress:
        on_progress(20)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get target amplitude for this frame
        target_amp = 0.0
        if frames_processed < len(amplitudes):
            target_amp = amplitudes[frames_processed]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            current_openness = _compute_mouth_openness(lms, h)
            target_openness = (
                MOUTH_CLOSED_RATIO + target_amp * (MOUTH_OPEN_RATIO - MOUTH_CLOSED_RATIO)
            )
            _apply_mouth_transform(
                frame, lms, target_openness, current_openness, w, h,
                blend_strength=blend_strength,
            )
            # Quality score: how well amplitude maps to mouth position
            diff = abs(target_openness - current_openness)
            score = max(0.0, 1.0 - diff * 5.0)
            quality_scores.append(score)

        writer.write(frame)
        frames_processed += 1

        if on_progress and frames_processed % 10 == 0:
            pct = 20 + int((frames_processed / max(1, total_frames)) * 65)
            on_progress(min(85, pct))

    cap.release()
    writer.release()
    face_mesh.close()

    if on_progress:
        on_progress(88)

    # Mux: use replacement audio with modified video
    cmd = (FFmpegCmd()
           .input(tmp_video)
           .input(audio_path)
           .map("0:v", "1:a")
           .video_codec("libx264", crf=18, preset="fast")
           .audio_codec("aac", bitrate="192k")
           .faststart()
           .option("shortest")
           .output(out_path)
           .build())

    try:
        run_ffmpeg(cmd)
    finally:
        try:
            os.unlink(tmp_video)
            os.rmdir(tmp_dir)
        except OSError:
            logger.debug("Failed to clean up temp dir: %s", tmp_dir)

    if on_progress:
        on_progress(100)

    avg_quality = sum(quality_scores) / max(1, len(quality_scores)) if quality_scores else 0.5

    return LipSyncResult(
        output_path=out_path,
        frames_processed=frames_processed,
        audio_duration=audio_duration,
        sync_quality_score=avg_quality,
    )
