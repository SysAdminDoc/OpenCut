"""
OpenCut AI Environmental Noise Classifier

Classify audio events (traffic, wind, HVAC, typing, etc.) per time segment
using simple energy + frequency heuristics or YAMNet when available.
Supports selective removal of classified noise types.

Uses FFmpeg for audio extraction, numpy for frequency analysis.
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, output_path as _output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Noise classes and frequency profiles
# ---------------------------------------------------------------------------
NOISE_CLASSES = {
    "traffic": {"freq_range": (20, 500), "energy_profile": "sustained_low"},
    "wind": {"freq_range": (20, 300), "energy_profile": "fluctuating_low"},
    "hvac": {"freq_range": (50, 500), "energy_profile": "sustained_constant"},
    "typing": {"freq_range": (800, 4000), "energy_profile": "transient_mid"},
    "speech": {"freq_range": (85, 8000), "energy_profile": "sustained_mid"},
    "music": {"freq_range": (20, 16000), "energy_profile": "sustained_wide"},
    "siren": {"freq_range": (500, 4000), "energy_profile": "sweeping_mid"},
    "dog_bark": {"freq_range": (300, 4000), "energy_profile": "transient_mid"},
    "construction": {"freq_range": (50, 2000), "energy_profile": "transient_low"},
    "electrical_hum": {"freq_range": (50, 300), "energy_profile": "sustained_narrow"},
}


@dataclass
class NoiseSegment:
    """A classified noise segment."""
    start: float
    end: float
    duration: float
    noise_type: str
    confidence: float
    energy_db: float = 0.0


@dataclass
class NoiseClassifyResult:
    """Result of noise classification."""
    segments: List[NoiseSegment] = field(default_factory=list)
    noise_types_found: List[str] = field(default_factory=list)
    total_duration: float = 0.0
    method: str = "heuristic"


# ---------------------------------------------------------------------------
# Classification backends
# ---------------------------------------------------------------------------
def _classify_yamnet(input_path: str, segment_duration: float = 1.0) -> Optional[List[NoiseSegment]]:
    """Classify using TensorFlow YAMNet model."""
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        import numpy as np
        import soundfile as sf
    except ImportError:
        return None

    try:
        model = hub.load("https://tfhub.dev/google/yamnet/1")
    except Exception:
        return None

    try:
        audio, sr = sf.read(input_path, dtype="float32")
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        # YAMNet expects 16kHz
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
    except Exception as e:
        logger.warning("Audio loading for YAMNet failed: %s", e)
        return None

    try:
        scores, embeddings, spectrogram = model(audio)
        class_map = model.class_map_path()
        class_names = list(tf.io.read_file(class_map).numpy().decode("utf-8").split("\n"))
    except Exception as e:
        logger.warning("YAMNet inference failed: %s", e)
        return None

    segments = []
    scores_np = scores.numpy()
    for i, frame_scores in enumerate(scores_np):
        top_idx = int(np.argmax(frame_scores))
        conf = float(frame_scores[top_idx])
        if conf > 0.3 and top_idx < len(class_names):
            name = class_names[top_idx].strip().lower().replace(" ", "_")
            t_start = i * 0.48  # YAMNet uses ~0.48s frames
            segments.append(NoiseSegment(
                start=t_start,
                end=t_start + 0.48,
                duration=0.48,
                noise_type=name,
                confidence=round(conf, 3),
            ))

    return segments


def _classify_heuristic(
    input_path: str,
    segment_duration: float = 1.0,
) -> List[NoiseSegment]:
    """
    Classify audio segments using energy and frequency heuristics.

    Analyzes audio in chunks, computing energy per frequency band and
    matching against known noise profiles.
    """
    try:
        import numpy as np
    except ImportError:
        logger.info("numpy not available for heuristic classification")
        return []

    info = get_video_info(input_path)
    total_duration = info.get("duration", 0)
    if total_duration <= 0:
        return []

    # Extract raw audio via FFmpeg
    ffmpeg = get_ffmpeg_path()
    tmpfile = tempfile.NamedTemporaryFile(suffix=".raw", delete=False, prefix="noise_cls_")
    tmpfile.close()

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-ac", "1", "-ar", "16000", "-f", "s16le",
        tmpfile.name,
    ]

    try:
        run_ffmpeg(cmd)
        raw_data = np.fromfile(tmpfile.name, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        logger.warning("Audio extraction failed: %s", e)
        return []
    finally:
        try:
            os.unlink(tmpfile.name)
        except OSError:
            pass

    sr = 16000
    segment_samples = int(segment_duration * sr)
    segments = []

    for i in range(0, len(raw_data) - segment_samples + 1, segment_samples):
        chunk = raw_data[i:i + segment_samples]
        t_start = i / sr

        # Compute FFT
        fft = np.fft.rfft(chunk)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)

        # Compute energy in different bands
        total_energy = np.sum(magnitude ** 2)
        if total_energy < 1e-10:
            continue

        energy_db = 10 * np.log10(total_energy + 1e-10)

        # Band energies
        low_mask = (freqs >= 20) & (freqs < 300)
        mid_mask = (freqs >= 300) & (freqs < 2000)
        high_mask = (freqs >= 2000) & (freqs < 8000)

        low_energy = np.sum(magnitude[low_mask] ** 2) / max(total_energy, 1e-10)
        mid_energy = np.sum(magnitude[mid_mask] ** 2) / max(total_energy, 1e-10)
        high_energy = np.sum(magnitude[high_mask] ** 2) / max(total_energy, 1e-10)

        # Classify based on band distribution
        noise_type = "unknown"
        confidence = 0.3

        if low_energy > 0.7 and energy_db > -40:
            noise_type = "traffic"
            confidence = min(0.8, low_energy)
        elif low_energy > 0.6 and energy_db < -30:
            noise_type = "wind"
            confidence = min(0.7, low_energy)
        elif 0.3 < mid_energy < 0.7 and 0.2 < low_energy < 0.5:
            noise_type = "speech"
            confidence = min(0.7, mid_energy)
        elif high_energy > 0.5:
            noise_type = "typing"
            confidence = min(0.6, high_energy)
        elif low_energy > 0.5 and mid_energy > 0.3:
            noise_type = "hvac"
            confidence = 0.4

        # Check for narrow-band (electrical hum)
        peak_idx = np.argmax(magnitude)
        peak_freq = freqs[peak_idx]
        peak_ratio = float(magnitude[peak_idx]) / (np.mean(magnitude) + 1e-10)
        if peak_ratio > 10 and 45 < peak_freq < 65:
            noise_type = "electrical_hum"
            confidence = min(0.9, peak_ratio / 20)

        if noise_type != "unknown":
            segments.append(NoiseSegment(
                start=round(t_start, 3),
                end=round(t_start + segment_duration, 3),
                duration=round(segment_duration, 3),
                noise_type=noise_type,
                confidence=round(confidence, 3),
                energy_db=round(energy_db, 1),
            ))

    return segments


# ---------------------------------------------------------------------------
# Selective removal
# ---------------------------------------------------------------------------
def _remove_noise_ffmpeg(
    input_path: str,
    segments: List[dict],
    noise_types: List[str],
) -> str:
    """Remove classified noise segments using FFmpeg filters."""
    ffmpeg = get_ffmpeg_path()
    info = get_video_info(input_path)
    duration = info.get("duration", 0)

    filters = []
    for seg in segments:
        if seg.get("noise_type") not in noise_types:
            continue
        s = float(seg.get("start", 0))
        e = float(seg.get("end", 0))
        ntype = seg.get("noise_type", "")

        profile = NOISE_CLASSES.get(ntype, {})
        freq_range = profile.get("freq_range", (20, 8000))

        center = (freq_range[0] + freq_range[1]) / 2
        width = freq_range[1] - freq_range[0]
        enable = f"between(t,{s},{e})"
        filters.append(f"bandreject=f={center}:width_type=h:w={width}:enable='{enable}'")

    if not filters:
        filters.append("anull")

    af = ",".join(filters)
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="noise_rm_")
    tmpfile.close()

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", af,
        "-c:a", "pcm_s16le",
        tmpfile.name,
    ]
    run_ffmpeg(cmd)

    return tmpfile.name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_noise(
    input_path: str,
    segment_duration: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Classify environmental noise in audio by time segment.

    Args:
        input_path: Source audio/video file.
        segment_duration: Duration of each analysis segment in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with segments list, noise_types_found, total_duration, method.
    """
    segment_duration = max(0.25, min(10.0, float(segment_duration)))

    if on_progress:
        on_progress(10, "Classifying noise...")

    # Try YAMNet first
    segments = _classify_yamnet(input_path, segment_duration)
    method = "yamnet"

    if segments is None:
        segments = _classify_heuristic(input_path, segment_duration)
        method = "heuristic"

    types_found = sorted(set(s.noise_type for s in segments))
    total_dur = sum(s.duration for s in segments)

    if on_progress:
        on_progress(100, f"Classified {len(segments)} segments ({', '.join(types_found) or 'none'})")

    return {
        "segments": [asdict(s) for s in segments],
        "noise_types_found": types_found,
        "total_duration": round(total_dur, 3),
        "method": method,
    }


def remove_classified_noise(
    input_path: str,
    noise_types: List[str],
    segments: Optional[List[dict]] = None,
    segment_duration: float = 1.0,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Remove specific noise types from audio.

    Args:
        input_path: Source audio/video file.
        noise_types: List of noise types to remove (e.g. ["traffic", "wind"]).
        segments: Pre-classified segments (auto-classified if None).
        segment_duration: Segment size for classification.
        output_path_str: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, segments_removed, noise_types_removed.
    """
    if output_path_str is None:
        output_path_str = _output_path(input_path, "noise_cleaned")
        output_path_str = os.path.splitext(output_path_str)[0] + ".wav"

    if segments is None:
        if on_progress:
            on_progress(10, "Classifying noise segments...")
        cls_result = classify_noise(input_path, segment_duration)
        segments = cls_result["segments"]

    # Filter to target types
    target_segments = [s for s in segments if s.get("noise_type") in noise_types]

    if not target_segments:
        if on_progress:
            on_progress(100, "No matching noise segments found")
        ffmpeg = get_ffmpeg_path()
        cmd = [ffmpeg, "-i", input_path, "-c:a", "pcm_s16le", "-y", output_path_str]
        run_ffmpeg(cmd)
        return {
            "output_path": output_path_str,
            "segments_removed": 0,
            "noise_types_removed": [],
        }

    if on_progress:
        on_progress(50, f"Removing {len(target_segments)} noise segments...")

    tmp_path = _remove_noise_ffmpeg(input_path, target_segments, noise_types)

    # Copy to final output
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", tmp_path,
        "-c:a", "pcm_s16le",
        output_path_str,
    ]
    run_ffmpeg(cmd)

    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    removed_types = sorted(set(s.get("noise_type") for s in target_segments))

    if on_progress:
        on_progress(100, f"Removed {len(target_segments)} noise segments")

    return {
        "output_path": output_path_str,
        "segments_removed": len(target_segments),
        "noise_types_removed": removed_types,
    }
