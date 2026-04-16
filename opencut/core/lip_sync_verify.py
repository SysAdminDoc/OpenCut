"""
OpenCut Lip Sync Verification Module (16.2)

Compare audio energy envelope to mouth movement energy extracted from
video frames, flag drift segments where audio and visual go out of sync.
"""

import logging
import math
import os
import struct
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LipSyncResult:
    """Result of a lip-sync verification analysis."""
    overall_correlation: float = 0.0
    drift_segments: List[Dict] = field(default_factory=list)
    max_drift_ms: float = 0.0
    audio_energy_samples: int = 0
    mouth_energy_samples: int = 0
    in_sync: bool = True


# ---------------------------------------------------------------------------
# Audio energy computation
# ---------------------------------------------------------------------------

def compute_audio_energy(
    audio_path: str,
    window_ms: float = 40.0,
    sample_rate: int = 16000,
    on_progress: Optional[Callable] = None,
) -> List[float]:
    """Compute RMS audio energy envelope from an audio/video file.

    Extracts mono 16-bit PCM, then computes per-window RMS values.

    Args:
        audio_path: Path to audio or video file.
        window_ms: Analysis window in milliseconds.
        sample_rate: Resample rate for analysis.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        List of RMS energy values, one per window.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if on_progress:
        on_progress(5, "Extracting audio for energy analysis")

    # Extract raw PCM
    _fd, tmp_wav = tempfile.mkstemp(suffix="_lipsync_audio.wav")
    os.close(_fd)
    try:
        ffmpeg = get_ffmpeg_path()
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", audio_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-ac", "1",
            tmp_wav,
        ]
        run_ffmpeg(cmd)

        # Read PCM samples (skip 44-byte WAV header)
        with open(tmp_wav, "rb") as f:
            f.read(44)
            raw = f.read()

        n_samples = len(raw) // 2
        if n_samples == 0:
            return []
        samples = struct.unpack(f"<{n_samples}h", raw[:n_samples * 2])

        if on_progress:
            on_progress(30, "Computing audio energy envelope")

        # Compute RMS per window
        window_samples = max(1, int(sample_rate * window_ms / 1000.0))
        energy = []
        for i in range(0, n_samples - window_samples + 1, window_samples):
            chunk = samples[i:i + window_samples]
            rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
            energy.append(rms / 32768.0)  # normalise to 0..1

        if on_progress:
            on_progress(50, f"Computed {len(energy)} audio energy windows")

        return energy
    finally:
        if os.path.isfile(tmp_wav):
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Mouth movement energy computation
# ---------------------------------------------------------------------------

def compute_mouth_energy(
    video_path: str,
    window_ms: float = 40.0,
    on_progress: Optional[Callable] = None,
) -> List[float]:
    """Estimate mouth movement energy from video frames.

    Uses frame differencing in the lower-third of each frame as a proxy
    for mouth/face movement. No ML model required.

    Args:
        video_path: Path to video file.
        window_ms: Target window size in milliseconds.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        List of movement energy values, one per analysis window.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    info = get_video_info(video_path)
    fps = info.get("fps", 25.0) or 25.0
    width = info.get("width", 320) or 320
    height = info.get("height", 240) or 240
    info.get("duration", 0.0) or 0.0

    if on_progress:
        on_progress(5, "Extracting frames for mouth energy analysis")

    # Determine the lower third region for face/mouth area
    crop_y = int(height * 2 / 3)
    crop_h = height - crop_y
    scale_w = min(width, 160)
    scale_h = max(1, int(crop_h * scale_w / max(width, 1)))

    # Extract lower-third grayscale frames as raw bytes
    _fd, tmp_raw = tempfile.mkstemp(suffix="_mouth_raw.gray")
    os.close(_fd)
    try:
        ffmpeg = get_ffmpeg_path()
        vf = f"crop={width}:{crop_h}:0:{crop_y},scale={scale_w}:{scale_h},format=gray"
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "gray",
            tmp_raw,
        ]
        run_ffmpeg(cmd, timeout=600)

        if not os.path.isfile(tmp_raw):
            return []

        raw_size = os.path.getsize(tmp_raw)
        frame_size = scale_w * scale_h
        if frame_size == 0:
            return []
        n_frames = raw_size // frame_size

        if on_progress:
            on_progress(40, f"Analyzing {n_frames} frames for movement")

        with open(tmp_raw, "rb") as f:
            raw_data = f.read()

        # Frame-to-frame difference in the mouth region
        frames_per_window = max(1, int(fps * window_ms / 1000.0))
        energy = []
        prev_frame = None
        window_accum = []

        for fi in range(n_frames):
            offset = fi * frame_size
            frame_bytes = raw_data[offset:offset + frame_size]
            if len(frame_bytes) < frame_size:
                break

            if prev_frame is not None:
                # Mean absolute difference
                diff_sum = sum(abs(a - b) for a, b in zip(frame_bytes, prev_frame))
                diff_mean = diff_sum / frame_size / 255.0  # normalise to 0..1
                window_accum.append(diff_mean)

            prev_frame = frame_bytes

            if len(window_accum) >= frames_per_window:
                energy.append(sum(window_accum) / len(window_accum))
                window_accum = []

        # Flush remaining
        if window_accum:
            energy.append(sum(window_accum) / len(window_accum))

        if on_progress:
            on_progress(70, f"Computed {len(energy)} mouth energy windows")

        return energy
    finally:
        if os.path.isfile(tmp_raw):
            try:
                os.unlink(tmp_raw)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def find_drift_segments(
    correlation: List[float],
    threshold: float = 0.3,
    min_duration_windows: int = 5,
    window_ms: float = 40.0,
) -> List[Dict]:
    """Find segments where audio-visual correlation drops below threshold.

    Args:
        correlation: Per-window correlation values.
        threshold: Minimum acceptable correlation (0..1).
        min_duration_windows: Minimum consecutive low-correlation windows
            to constitute a drift segment.
        window_ms: Window size in ms (for time calculation).

    Returns:
        List of drift segment dicts with start_ms, end_ms, avg_correlation.
    """
    segments = []
    drift_start = None
    drift_values = []

    for i, corr in enumerate(correlation):
        if corr < threshold:
            if drift_start is None:
                drift_start = i
                drift_values = []
            drift_values.append(corr)
        else:
            if drift_start is not None and len(drift_values) >= min_duration_windows:
                segments.append({
                    "start_ms": drift_start * window_ms,
                    "end_ms": i * window_ms,
                    "duration_ms": (i - drift_start) * window_ms,
                    "avg_correlation": sum(drift_values) / len(drift_values),
                    "windows": len(drift_values),
                })
            drift_start = None
            drift_values = []

    # Close open segment
    if drift_start is not None and len(drift_values) >= min_duration_windows:
        end_i = len(correlation)
        segments.append({
            "start_ms": drift_start * window_ms,
            "end_ms": end_i * window_ms,
            "duration_ms": (end_i - drift_start) * window_ms,
            "avg_correlation": sum(drift_values) / len(drift_values),
            "windows": len(drift_values),
        })

    return segments


# ---------------------------------------------------------------------------
# Main verification function
# ---------------------------------------------------------------------------

def verify_lip_sync(
    video_path: str,
    threshold: float = 0.3,
    window_ms: float = 40.0,
    min_drift_windows: int = 5,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Verify lip sync by comparing audio and mouth movement energy.

    Args:
        video_path: Path to video file.
        threshold: Correlation threshold below which drift is flagged.
        window_ms: Analysis window in milliseconds.
        min_drift_windows: Minimum consecutive low-correlation windows.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        Dict with correlation, drift segments, and sync status.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if on_progress:
        on_progress(0, "Starting lip sync verification")

    # Compute both energy envelopes
    audio_energy = compute_audio_energy(
        video_path, window_ms=window_ms,
        on_progress=lambda p, m: on_progress(p * 0.4, m) if on_progress else None,
    )
    mouth_energy = compute_mouth_energy(
        video_path, window_ms=window_ms,
        on_progress=lambda p, m: on_progress(40 + p * 0.4, m) if on_progress else None,
    )

    if on_progress:
        on_progress(80, "Computing correlation")

    # Align lengths
    min_len = min(len(audio_energy), len(mouth_energy))
    if min_len == 0:
        result = LipSyncResult()
        return {
            "overall_correlation": result.overall_correlation,
            "drift_segments": result.drift_segments,
            "max_drift_ms": result.max_drift_ms,
            "audio_energy_samples": len(audio_energy),
            "mouth_energy_samples": len(mouth_energy),
            "in_sync": True,
        }

    audio_energy = audio_energy[:min_len]
    mouth_energy = mouth_energy[:min_len]

    # Normalised cross-correlation per window
    a_mean = sum(audio_energy) / min_len
    m_mean = sum(mouth_energy) / min_len
    math.sqrt(max(1e-12, sum((v - a_mean) ** 2 for v in audio_energy) / min_len))
    math.sqrt(max(1e-12, sum((v - m_mean) ** 2 for v in mouth_energy) / min_len))

    # Sliding correlation
    correlation = []
    half_window = max(1, min_drift_windows)
    for i in range(min_len):
        lo = max(0, i - half_window)
        hi = min(min_len, i + half_window + 1)
        seg_a = audio_energy[lo:hi]
        seg_m = mouth_energy[lo:hi]
        n = len(seg_a)
        sa_mean = sum(seg_a) / n
        sm_mean = sum(seg_m) / n
        sa_std = math.sqrt(max(1e-12, sum((v - sa_mean) ** 2 for v in seg_a) / n))
        sm_std = math.sqrt(max(1e-12, sum((v - sm_mean) ** 2 for v in seg_m) / n))
        cov = sum((a - sa_mean) * (b - sm_mean) for a, b in zip(seg_a, seg_m)) / n
        corr = cov / (sa_std * sm_std)
        correlation.append(max(0.0, min(1.0, (corr + 1.0) / 2.0)))

    overall_corr = sum(correlation) / len(correlation) if correlation else 0.0

    # Find drift segments
    drift_segments = find_drift_segments(
        correlation, threshold=threshold,
        min_duration_windows=min_drift_windows,
        window_ms=window_ms,
    )

    max_drift = max((s["duration_ms"] for s in drift_segments), default=0.0)

    if on_progress:
        on_progress(100, "Lip sync verification complete")

    return {
        "overall_correlation": round(overall_corr, 4),
        "drift_segments": drift_segments,
        "max_drift_ms": round(max_drift, 2),
        "audio_energy_samples": len(audio_energy),
        "mouth_energy_samples": len(mouth_energy),
        "in_sync": len(drift_segments) == 0,
    }
