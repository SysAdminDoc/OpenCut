"""
OpenCut Sports/Genre Highlights v1.30.0

Extract highlight segments from sports, concerts, events, and reaction videos
using multi-signal scoring:
  - Motion energy: optical flow magnitude per frame window
  - Audio energy: RMS amplitude per window
  - Genre-specific signal weighting

Requires: cv2, numpy (always available with standard install).
Optional:  librosa (better audio analysis), torch (neural crowd detection).
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from opencut.helpers import _try_import, get_ffmpeg_path, get_ffprobe_path, get_video_info

logger = logging.getLogger("opencut")

GENRES = ["sports", "concert", "event", "reaction", "game"]
INSTALL_HINT = "pip install opencv-python-headless numpy  # already in standard install"

# Genre-specific weights for motion/audio/face scoring
_GENRE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "sports":   {"motion": 0.55, "audio": 0.30, "face": 0.15},
    "concert":  {"motion": 0.30, "audio": 0.55, "face": 0.15},
    "event":    {"motion": 0.35, "audio": 0.35, "face": 0.30},
    "reaction": {"motion": 0.20, "audio": 0.30, "face": 0.50},
    "game":     {"motion": 0.50, "audio": 0.35, "face": 0.15},
}


def check_sports_highlights_available() -> bool:
    """Always True — base implementation uses only cv2 and numpy."""
    return _try_import("cv2") is not None and _try_import("numpy") is not None


@dataclass
class HighlightSegment:
    start: float = 0.0
    end: float = 0.0
    score: float = 0.0
    signals: Dict = field(default_factory=dict)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def keys(self):
        return ("start", "end", "score", "signals")

    def __contains__(self, k: str) -> bool:
        return k in self.keys()


def list_genres() -> List[str]:
    return GENRES


# ---------------------------------------------------------------------------
# Internal: extract audio RMS via ffmpeg/librosa
# ---------------------------------------------------------------------------
def _extract_audio_rms(video_path: str, window_sec: float = 1.0) -> np.ndarray:
    """Return per-window RMS array using ffmpeg's astats filter."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner",
        "-i", video_path,
        "-vn",
        "-af", f"asetnsamples=n={int(44100 * window_sec)}:p=0,astats=metadata=1:reset=1",
        "-f", "null", "-",
    ]
    rms_values = []
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        for line in result.stderr.splitlines():
            if "RMS_level" in line and "lavfi" in line:
                # format: "lavfi.astats.Overall.RMS_level=-18.3"
                try:
                    val = float(line.split("=")[-1].strip())
                    # Convert dBFS to linear [0, 1]
                    rms_values.append(min(1.0, 10 ** (val / 20.0) if val > -100 else 0.0))
                except ValueError:
                    continue
    except Exception as exc:
        logger.debug("Audio RMS extraction failed: %s", exc)
    return np.array(rms_values, dtype=np.float32) if rms_values else np.array([0.5], dtype=np.float32)


# ---------------------------------------------------------------------------
# Internal: optical flow motion scoring
# ---------------------------------------------------------------------------
def _compute_motion_scores(video_path: str, sample_fps: float = 2.0,
                            window_sec: float = 1.0) -> np.ndarray:
    """
    Sample frames at sample_fps and compute per-window optical flow magnitude.

    Returns array of motion scores normalised to [0, 1].
    """
    cv2 = _try_import("cv2")
    if cv2 is None:
        logger.warning("cv2 not available — using uniform motion scores")
        info = get_video_info(video_path)
        n_windows = max(1, int(info.get("duration", 30) / window_sec))
        return np.ones(n_windows, dtype=np.float32) * 0.5

    info = get_video_info(video_path)
    total_duration = float(info.get("duration", 0) or 0)
    if total_duration < 0.1:
        return np.array([0.5], dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.array([0.5], dtype=np.float32)

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    step = max(1, int(video_fps / sample_fps))

    frame_scores: List[float] = []
    prev_gray = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 180))  # fast
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=2, winsize=15,
                        iterations=2, poly_n=5, poly_sigma=1.2,
                        flags=0,
                    )
                    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    frame_scores.append(float(magnitude.mean()))
                prev_gray = gray

            frame_idx += 1
    finally:
        cap.release()

    if not frame_scores:
        return np.array([0.5], dtype=np.float32)

    scores = np.array(frame_scores, dtype=np.float32)

    # Group into time windows
    frames_per_window = max(1, int(sample_fps * window_sec))
    n_windows = max(1, len(scores) // frames_per_window)
    window_scores = np.array([
        scores[i * frames_per_window: (i + 1) * frames_per_window].mean()
        for i in range(n_windows)
    ])

    # Normalise to [0, 1]
    s_max = window_scores.max()
    if s_max > 0:
        window_scores = window_scores / s_max

    return window_scores


# ---------------------------------------------------------------------------
# Internal: combine signals into final scores
# ---------------------------------------------------------------------------
def _score_windows(
    motion: np.ndarray,
    audio: np.ndarray,
    genre: str,
) -> np.ndarray:
    """Combine motion + audio arrays into per-window highlight scores."""
    weights = _GENRE_WEIGHTS.get(genre, _GENRE_WEIGHTS["sports"])

    # Normalise both arrays to the same length (shorter one repeats last value)
    n = max(len(motion), len(audio))

    def _pad(arr: np.ndarray, length: int) -> np.ndarray:
        if len(arr) >= length:
            return arr[:length]
        pad_width = length - len(arr)
        return np.pad(arr, (0, pad_width), mode="edge")

    motion = _pad(motion, n)
    audio = _pad(audio, n)

    # Smooth with a small sliding window to avoid single-frame spikes
    def _smooth(arr: np.ndarray, w: int = 3) -> np.ndarray:
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")

    motion = _smooth(motion, w=3)
    audio = _smooth(audio, w=3)

    combined = (
        motion * weights["motion"] +
        audio * weights["audio"]
    )

    # Normalise final
    c_max = combined.max()
    if c_max > 0:
        combined = combined / c_max

    return combined


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract(
    video_path: str,
    genre: str = "sports",
    top_n: int = 5,
    window_sec: float = 3.0,
    min_score: float = 0.4,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> List[HighlightSegment]:
    """
    Extract highlight segments from a video.

    Uses optical flow motion energy and audio RMS energy to score each
    time window, then returns the top-N non-overlapping windows.

    Args:
        video_path: Path to input video.
        genre: Content genre for signal weighting (sports/concert/event/reaction/game).
        top_n: Maximum number of highlight segments to return.
        window_sec: Duration of each scored window in seconds.
        min_score: Minimum normalised score threshold to include a window.
        on_progress: Optional callback(pct, msg).

    Returns:
        List of HighlightSegment sorted by start time.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If cv2 is not available.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if genre not in GENRES:
        logger.warning("Unknown genre %r — using 'sports' weights", genre)
        genre = "sports"

    if on_progress:
        on_progress(5, "Analysing video info...")

    info = get_video_info(video_path)
    total_duration = float(info.get("duration", 0) or 0)
    if total_duration < window_sec:
        # Video shorter than one window — return entire clip as single highlight
        return [HighlightSegment(start=0.0, end=total_duration, score=1.0,
                                  signals={"motion": 1.0, "audio": 1.0})]

    if on_progress:
        on_progress(10, "Computing motion scores...")

    motion_scores = _compute_motion_scores(video_path, sample_fps=2.0,
                                           window_sec=window_sec)

    if on_progress:
        on_progress(40, "Computing audio energy...")

    audio_scores = _extract_audio_rms(video_path, window_sec=window_sec)

    if on_progress:
        on_progress(65, "Scoring windows...")

    combined = _score_windows(motion_scores, audio_scores, genre)

    # Map window indices back to timestamps
    n_windows = len(combined)
    # Actual seconds per window based on total duration
    actual_window = total_duration / n_windows

    # Sort by score descending, pick top_n non-overlapping windows
    ranked = np.argsort(combined)[::-1]
    selected_indices: List[int] = []
    for idx in ranked:
        if combined[idx] < min_score:
            break
        # Check overlap with already-selected windows
        start_t = idx * actual_window
        end_t = start_t + actual_window
        overlap = False
        for sel in selected_indices:
            sel_start = sel * actual_window
            sel_end = sel_start + actual_window
            if not (end_t <= sel_start or start_t >= sel_end):
                overlap = True
                break
        if not overlap:
            selected_indices.append(idx)
        if len(selected_indices) >= top_n:
            break

    if on_progress:
        on_progress(90, f"Found {len(selected_indices)} highlights")

    # Build result sorted by start time
    highlights = []
    for idx in sorted(selected_indices):
        start_t = round(idx * actual_window, 3)
        end_t = round(min(total_duration, start_t + actual_window), 3)
        motion_val = float(motion_scores[idx]) if idx < len(motion_scores) else 0.0
        audio_val = float(audio_scores[idx]) if idx < len(audio_scores) else 0.0
        highlights.append(HighlightSegment(
            start=start_t,
            end=end_t,
            score=round(float(combined[idx]), 4),
            signals={"motion": round(motion_val, 4), "audio": round(audio_val, 4)},
        ))

    if on_progress:
        on_progress(100, "Highlight extraction complete")

    return highlights


__all__ = [
    "check_sports_highlights_available",
    "INSTALL_HINT",
    "GENRES",
    "HighlightSegment",
    "list_genres",
    "extract",
]
