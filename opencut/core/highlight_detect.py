"""
OpenCut Gaming Highlight Detection Module v0.9.0

Multi-signal fusion for automatic highlight extraction:
- Audio peak detection (cheering, explosions, voice spikes)
- Visual change detection (scene cuts, flash events, HUD changes)
- Optional chat activity correlation (Twitch/YouTube chat logs)
- Segment scoring and ranked highlight extraction

Requires: pip install opencv-python-headless numpy
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

from opencut.helpers import ensure_package, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class HighlightScore:
    """Score for a video segment."""
    start_time: float
    end_time: float
    score: float
    audio_score: float = 0.0
    visual_score: float = 0.0
    chat_score: float = 0.0
    label: str = ""


# ---------------------------------------------------------------------------
# Segment Scoring
# ---------------------------------------------------------------------------
def score_segments(
    video_path: str,
    segment_duration: float = 10.0,
    audio_weight: float = 0.5,
    visual_weight: float = 0.4,
    chat_weight: float = 0.1,
    chat_log_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> List[HighlightScore]:
    """
    Score video segments by multi-signal analysis.

    Divides the video into fixed-length segments and assigns a
    normalised highlight score based on audio energy, visual change
    magnitude, and optional chat activity.

    Args:
        video_path: Input video path.
        segment_duration: Duration of each segment in seconds.
        audio_weight: Weight for audio signal (0-1).
        visual_weight: Weight for visual signal (0-1).
        chat_weight: Weight for chat signal (0-1).
        chat_log_path: Optional path to chat log (JSON array of
            {timestamp, message} objects).
        on_progress: Progress callback(pct, msg).

    Returns:
        List of HighlightScore objects sorted by score descending.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("Failed to install opencv-python-headless")
    if not ensure_package("numpy", "numpy"):
        raise RuntimeError("Failed to install numpy")
    import cv2
    import numpy as np

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    duration = info.get("duration", 0.0)
    if duration <= 0:
        raise ValueError("Cannot determine video duration")

    # Normalise weights
    total_w = audio_weight + visual_weight + chat_weight
    if total_w > 0:
        audio_weight /= total_w
        visual_weight /= total_w
        chat_weight /= total_w

    num_segments = max(1, int(duration / segment_duration))
    audio_scores = [0.0] * num_segments
    visual_scores = [0.0] * num_segments

    if on_progress:
        on_progress(5, "Analysing visual changes...")

    # --- Visual analysis: frame differencing ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    prev_gray = None
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample every Nth frame for speed
    sample_step = max(1, int(fps / 4))  # ~4 samples per second

    segment_visual_accum = [0.0] * num_segments
    segment_visual_count = [0] * num_segments

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 180))

                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    change = float(np.mean(diff))

                    t = frame_idx / fps
                    seg_idx = min(int(t / segment_duration), num_segments - 1)
                    segment_visual_accum[seg_idx] += change
                    segment_visual_count[seg_idx] += 1

                prev_gray = gray

            frame_idx += 1

            if on_progress and frame_idx % (sample_step * 30) == 0:
                pct = 5 + int((frame_idx / max(1, total_frames)) * 35)
                on_progress(min(pct, 40), f"Visual analysis {frame_idx}/{total_frames}...")
    finally:
        cap.release()

    # Average visual change per segment
    for i in range(num_segments):
        if segment_visual_count[i] > 0:
            visual_scores[i] = segment_visual_accum[i] / segment_visual_count[i]

    # Normalise visual scores to 0-1
    v_max = max(visual_scores) if visual_scores else 1.0
    if v_max > 0:
        visual_scores = [v / v_max for v in visual_scores]

    if on_progress:
        on_progress(45, "Analysing audio energy...")

    # --- Audio analysis: extract audio and measure RMS energy ---
    import tempfile
    audio_tmp = os.path.join(tempfile.gettempdir(), "oc_highlight_audio.wav")
    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
            audio_tmp,
        ], timeout=600)

        # Read WAV and compute per-segment RMS
        import wave
        try:
            with wave.open(audio_tmp, "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                audio_data /= 32768.0

                samples_per_seg = int(sr * segment_duration)
                for i in range(num_segments):
                    start = i * samples_per_seg
                    end = min(start + samples_per_seg, len(audio_data))
                    if start < len(audio_data):
                        seg = audio_data[start:end]
                        rms = float(np.sqrt(np.mean(seg ** 2)))
                        audio_scores[i] = rms
        except Exception as e:
            logger.warning("Audio analysis failed, using visual only: %s", e)
    except Exception as e:
        logger.warning("Audio extraction failed: %s", e)
    finally:
        try:
            os.unlink(audio_tmp)
        except OSError:
            pass

    # Normalise audio scores
    a_max = max(audio_scores) if audio_scores else 1.0
    if a_max > 0:
        audio_scores = [a / a_max for a in audio_scores]

    if on_progress:
        on_progress(75, "Computing highlight scores...")

    # --- Chat analysis (optional) ---
    chat_scores = [0.0] * num_segments
    if chat_log_path and os.path.isfile(chat_log_path):
        try:
            with open(chat_log_path, "r", encoding="utf-8") as f:
                chat_data = json.load(f)
            # Count messages per segment
            for msg in chat_data:
                t = float(msg.get("timestamp", 0))
                seg_idx = min(int(t / segment_duration), num_segments - 1)
                if 0 <= seg_idx < num_segments:
                    chat_scores[seg_idx] += 1.0
            c_max = max(chat_scores) if chat_scores else 1.0
            if c_max > 0:
                chat_scores = [c / c_max for c in chat_scores]
        except Exception as e:
            logger.warning("Chat log parsing failed: %s", e)

    # --- Combine scores ---
    highlights = []
    for i in range(num_segments):
        combined = (
            audio_weight * audio_scores[i]
            + visual_weight * visual_scores[i]
            + chat_weight * chat_scores[i]
        )
        highlights.append(HighlightScore(
            start_time=i * segment_duration,
            end_time=min((i + 1) * segment_duration, duration),
            score=round(combined, 4),
            audio_score=round(audio_scores[i], 4),
            visual_score=round(visual_scores[i], 4),
            chat_score=round(chat_scores[i], 4),
        ))

    highlights.sort(key=lambda h: h.score, reverse=True)

    if on_progress:
        on_progress(100, "Highlight scoring complete!")

    return highlights


# ---------------------------------------------------------------------------
# Detect Highlights (convenience wrapper)
# ---------------------------------------------------------------------------
def detect_highlights(
    video_path: str,
    top_n: int = 5,
    segment_duration: float = 10.0,
    min_score: float = 0.3,
    chat_log_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Detect top highlight moments in a video.

    Args:
        video_path: Input video path.
        top_n: Number of top highlights to return.
        segment_duration: Segment length in seconds.
        min_score: Minimum score threshold.
        chat_log_path: Optional chat log path.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with highlights (list of HighlightScore dicts),
        total_segments, duration.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    all_scores = score_segments(
        video_path,
        segment_duration=segment_duration,
        chat_log_path=chat_log_path,
        on_progress=on_progress,
    )

    # Filter by min_score and take top_n
    filtered = [h for h in all_scores if h.score >= min_score]
    top = filtered[:top_n]

    info = get_video_info(video_path)

    return {
        "highlights": [asdict(h) for h in top],
        "total_segments": len(all_scores),
        "duration": info.get("duration", 0.0),
    }


# ---------------------------------------------------------------------------
# Extract Highlight Clips
# ---------------------------------------------------------------------------
def extract_highlight_clips(
    video_path: str,
    highlights: List[dict],
    output_dir: str = "",
    padding: float = 2.0,
    on_progress: Optional[Callable] = None,
) -> List[str]:
    """
    Extract individual highlight clips from a video.

    Args:
        video_path: Source video path.
        highlights: List of highlight dicts with start_time, end_time.
        output_dir: Directory for output clips. Auto-generated if empty.
        padding: Extra seconds before/after each highlight.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of paths to extracted clip files.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not highlights:
        raise ValueError("No highlights provided")

    if not output_dir:
        output_dir = os.path.dirname(video_path)

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]
    info = get_video_info(video_path)
    total_dur = info.get("duration", 0.0)

    clip_paths = []
    for i, hl in enumerate(highlights):
        start = max(0.0, float(hl["start_time"]) - padding)
        end = min(total_dur, float(hl["end_time"]) + padding) if total_dur > 0 else float(hl["end_time"]) + padding
        clip_dur = end - start

        clip_path = os.path.join(output_dir, f"{base}_highlight_{i + 1:03d}.mp4")

        if on_progress:
            pct = int((i / len(highlights)) * 90) + 5
            on_progress(pct, f"Extracting highlight {i + 1}/{len(highlights)}...")

        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-i", video_path,
            "-t", str(clip_dur),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            clip_path,
        ], timeout=600)

        clip_paths.append(clip_path)

    if on_progress:
        on_progress(100, f"Extracted {len(clip_paths)} highlight clips!")

    return clip_paths
