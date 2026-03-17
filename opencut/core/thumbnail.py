"""
OpenCut Auto-Thumbnail Generator v0.7.2

Generates thumbnails from video by scoring frames on:
- Visual interest (color variance, contrast, sharpness)
- Face presence and size (MediaPipe or Haar)
- Scene change proximity (interesting moments)
- Rule of thirds composition

Extracts top N candidate frames and returns them as JPEGs.
"""

import logging
import os
import tempfile
from typing import Callable, Dict, List, Optional

from opencut.helpers import ensure_package, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")

def _score_frame(frame, has_face_detector=False, face_detector=None) -> float:
    """Score a frame based on visual quality metrics."""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    score = 0.0

    # 1. Color variance (more colorful = more interesting)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    score += min(sat_mean / 128.0, 1.0) * 20  # 0-20 points

    # 2. Contrast (Michelson contrast on luminance)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lum_min = float(np.min(gray))
    lum_max = float(np.max(gray))
    if lum_max + lum_min > 0:
        contrast = (lum_max - lum_min) / (lum_max + lum_min)
    else:
        contrast = 0
    score += contrast * 15  # 0-15 points

    # 3. Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())
    score += min(sharpness / 500.0, 1.0) * 15  # 0-15 points

    # 4. Not too dark or bright
    brightness = float(np.mean(gray))
    if 60 < brightness < 200:
        score += 10  # Bonus for good exposure
    elif 40 < brightness < 220:
        score += 5

    # 5. Face detection bonus
    if has_face_detector and face_detector is not None:
        try:
            faces = face_detector(frame)
            if faces:
                # More/bigger faces = better thumbnail
                total_face_area = sum(f[2] * f[3] for f in faces)
                frame_area = w * h
                face_ratio = total_face_area / frame_area
                score += min(face_ratio * 200, 30)  # 0-30 points for faces
                # Bonus for centered faces
                for (fx, fy, fw, fh) in faces:
                    cx = (fx + fw / 2) / w
                    cy = (fy + fh / 2) / h
                    # Rule of thirds bonus
                    for tx in [0.33, 0.5, 0.67]:
                        for ty in [0.33, 0.5, 0.67]:
                            dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                            if dist < 0.15:
                                score += 5
                                break
        except Exception:
            pass

    # 6. Penalty for very uniform frames (likely title cards)
    std_dev = float(np.std(gray))
    if std_dev < 20:
        score -= 15  # Probably a solid color or title card

    return score


def generate_thumbnails(
    input_path: str,
    output_dir: str = "",
    count: int = 5,
    width: int = 1920,
    use_faces: bool = True,
    on_progress: Optional[Callable] = None,
) -> List[Dict]:
    """
    Generate thumbnail candidates from video.

    Samples frames at intervals, scores each for visual quality,
    and returns the top N as JPEG files.

    Args:
        count: Number of thumbnails to generate.
        width: Output thumbnail width (height scales proportionally).
        use_faces: Use face detection for scoring bonus.

    Returns:
        List of dicts with path, timestamp, score.
    """
    ensure_package("cv2", "opencv-python-headless", on_progress)
    import cv2

    info = get_video_info(input_path)
    duration = info["duration"]
    if duration <= 0:
        raise RuntimeError("Could not determine video duration")

    directory = output_dir or os.path.dirname(input_path)
    base = os.path.splitext(os.path.basename(input_path))[0]

    if on_progress:
        on_progress(5, "Analyzing video for thumbnails...")

    # Sample ~30 frames spread across the video (skip first/last 5%)
    start_pct = 0.05
    end_pct = 0.95
    num_samples = min(60, max(30, int(duration)))
    sample_times = [
        start_pct * duration + (end_pct - start_pct) * duration * i / (num_samples - 1)
        for i in range(num_samples)
    ]

    # Set up face detector
    face_detect_fn = None
    has_faces = False
    if use_faces:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)

            def _haar_detect(frame):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                  minNeighbors=5, minSize=(30, 30))
                return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]

            face_detect_fn = _haar_detect
            has_faces = True
        except Exception:
            pass

    # Extract and score frames
    scored_frames = []
    for i, t in enumerate(sample_times):
        if on_progress and i % 5 == 0:
            pct = 5 + int((i / num_samples) * 70)
            on_progress(pct, f"Scoring frame {i+1}/{num_samples}...")

        # Extract single frame at timestamp
        _ntf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp = _ntf.name
        _ntf.close()
        try:
            run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-ss", f"{t:.3f}", "-i", input_path,
                "-vframes", "1", "-q:v", "2", tmp,
            ], timeout=300)
            frame = cv2.imread(tmp)
            if frame is None:
                continue

            score = _score_frame(frame, has_faces, face_detect_fn)
            scored_frames.append((t, score, frame))
        except Exception:
            pass
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    if not scored_frames:
        raise RuntimeError("No frames could be extracted for thumbnails")

    # Sort by score, take top N
    scored_frames.sort(key=lambda x: x[1], reverse=True)
    top_frames = scored_frames[:count]
    # Re-sort by timestamp for display
    top_frames.sort(key=lambda x: x[0])

    if on_progress:
        on_progress(80, "Saving thumbnails...")

    results = []
    for idx, (timestamp, score, frame) in enumerate(top_frames):
        # Scale to target width
        h, w = frame.shape[:2]
        if width and w != width:
            scale = width / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_LANCZOS4)

        out_path = os.path.join(directory, f"{base}_thumb_{idx+1}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        results.append({
            "path": out_path,
            "timestamp": round(timestamp, 2),
            "timestamp_display": f"{minutes}:{seconds:02d}",
            "score": round(score, 1),
            "index": idx + 1,
        })

    if on_progress:
        on_progress(100, f"Generated {len(results)} thumbnails")

    return results
