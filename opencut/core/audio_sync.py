"""
OpenCut Multi-Camera Audio Sync Module

Synchronize multiple camera angles by cross-correlating their audio tracks.
Extracts audio as raw PCM, computes amplitude envelopes, and finds the
time offset via cross-correlation.
"""

import logging
import os
import struct
import subprocess as _sp
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# SyncResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    """Result of an audio sync offset computation."""
    offset_seconds: float = 0.0
    confidence: float = 0.0
    method: str = "cross_correlation"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_audio_pcm(input_path: str, wav_path: str) -> None:
    """Extract audio from a video/audio file as mono 16-bit 16kHz WAV."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        wav_path,
    ]
    try:
        result = _sp.run(cmd, capture_output=True, timeout=300)
    except _sp.TimeoutExpired as exc:
        # 5-minute extraction cap is generous for audio-only PCM convert.
        # Surface the timeout as a normal failure instead of letting the
        # raw TimeoutExpired bubble up through the worker thread.
        raise RuntimeError(
            f"Audio extraction timed out after {exc.timeout}s for '{input_path}'"
        ) from exc
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"Audio extraction failed: {stderr}")
    if not os.path.isfile(wav_path) or os.path.getsize(wav_path) < 44:
        raise RuntimeError(f"Audio extraction produced empty output for '{input_path}'")


def _read_wav_samples(wav_path: str) -> list:
    """Read raw PCM samples from a 16-bit mono WAV file, skipping the 44-byte header."""
    with open(wav_path, "rb") as f:
        header = f.read(44)
        if len(header) < 44:
            raise RuntimeError("WAV file too short")
        raw = f.read()

    # Unpack as signed 16-bit little-endian
    n_samples = len(raw) // 2
    samples = list(struct.unpack(f"<{n_samples}h", raw[:n_samples * 2]))
    return samples


def _compute_envelope(samples: list, window: int = 160) -> list:
    """Compute amplitude envelope by taking max absolute value per window."""
    envelope = []
    for i in range(0, len(samples) - window, window):
        chunk = samples[i:i + window]
        envelope.append(max(abs(s) for s in chunk))
    return envelope


def _cross_correlate(ref_env: list, tgt_env: list, max_lag: int = 0) -> tuple:
    """
    Compute cross-correlation between two envelopes.
    Returns (best_lag, max_correlation_value, confidence).
    """
    len_ref = len(ref_env)
    len_tgt = len(tgt_env)

    if max_lag <= 0:
        max_lag = min(len_ref, len_tgt) // 2

    best_lag = 0
    best_val = -1.0

    # Normalize envelopes
    ref_mean = sum(ref_env) / max(len_ref, 1)
    tgt_mean = sum(tgt_env) / max(len_tgt, 1)
    ref_norm = [v - ref_mean for v in ref_env]
    tgt_norm = [v - tgt_mean for v in tgt_env]

    ref_energy = sum(v * v for v in ref_norm) ** 0.5
    tgt_energy = sum(v * v for v in tgt_norm) ** 0.5
    denom = max(ref_energy * tgt_energy, 1e-10)

    for lag in range(-max_lag, max_lag + 1):
        acc = 0.0
        count = 0
        for i in range(len_ref):
            j = i + lag
            if 0 <= j < len_tgt:
                acc += ref_norm[i] * tgt_norm[j]
                count += 1
        if count > 0:
            val = acc / denom
            if val > best_val:
                best_val = val
                best_lag = lag

    confidence = min(max(best_val, 0.0), 1.0)
    return best_lag, best_val, confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_sync_offset(
    reference_path: str,
    target_path: str,
    on_progress: Optional[Callable] = None,
) -> SyncResult:
    """
    Compute the audio sync offset of target relative to reference.

    Extracts audio from both files, computes amplitude envelopes,
    and cross-correlates to find the time offset.

    Args:
        reference_path: Path to the reference video/audio.
        target_path: Path to the target video/audio.
        on_progress: Progress callback (percent, message).

    Returns:
        SyncResult with offset_seconds, confidence, and method.
    """
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"Target file not found: {target_path}")

    if on_progress:
        on_progress(5, "Extracting audio from reference...")

    tmp_dir = tempfile.mkdtemp(prefix="opencut_sync_")
    ref_wav = os.path.join(tmp_dir, "ref.wav")
    tgt_wav = os.path.join(tmp_dir, "tgt.wav")

    try:
        _extract_audio_pcm(reference_path, ref_wav)

        if on_progress:
            on_progress(25, "Extracting audio from target...")

        _extract_audio_pcm(target_path, tgt_wav)

        if on_progress:
            on_progress(45, "Computing audio envelopes...")

        ref_samples = _read_wav_samples(ref_wav)
        tgt_samples = _read_wav_samples(tgt_wav)

        # Envelope at 16kHz with window=160 gives 100 values per second
        window = 160
        sample_rate = 16000
        envelope_rate = sample_rate / window  # 100 Hz

        ref_env = _compute_envelope(ref_samples, window)
        tgt_env = _compute_envelope(tgt_samples, window)

        if on_progress:
            on_progress(60, "Cross-correlating audio...")

        # Allow up to 30 seconds of offset
        max_lag = int(30 * envelope_rate)
        lag, _, confidence = _cross_correlate(ref_env, tgt_env, max_lag)

        offset_seconds = lag / envelope_rate

        if on_progress:
            on_progress(100, "Sync computation complete!")

        return SyncResult(
            offset_seconds=round(offset_seconds, 4),
            confidence=round(confidence, 4),
            method="cross_correlation",
        )
    finally:
        # Cleanup temp files
        for f in (ref_wav, tgt_wav):
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def sync_multiple(
    reference_path: str,
    target_paths: list,
    on_progress: Optional[Callable] = None,
) -> List[SyncResult]:
    """
    Compute sync offsets for multiple target files against one reference.

    Args:
        reference_path: Path to the reference video/audio.
        target_paths: List of paths to target videos/audio files.
        on_progress: Progress callback (percent, message).

    Returns:
        List of SyncResult, one per target.
    """
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    if not target_paths:
        raise ValueError("No target paths provided")

    results = []
    total = len(target_paths)

    for i, tgt_path in enumerate(target_paths):
        if on_progress:
            base_pct = int((i / total) * 100)
            on_progress(base_pct, f"Syncing target {i + 1}/{total}...")

        def _sub_progress(pct, msg=""):
            if on_progress:
                scaled = int((i / total) * 100 + (pct / total))
                on_progress(scaled, msg)

        result = compute_sync_offset(reference_path, tgt_path, on_progress=_sub_progress)
        results.append(result)

    if on_progress:
        on_progress(100, f"Synced {total} target(s)!")

    return results


def apply_sync_offset(
    input_path: str,
    offset_seconds: float,
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Shift audio in a video file by the given offset.

    Positive offset: delay audio (adelay filter).
    Negative offset: trim audio from beginning.

    Args:
        input_path: Path to the input video file.
        offset_seconds: Time offset in seconds (positive=delay, negative=trim).
        output_path_override: Custom output path (auto-generated if None).
        on_progress: Progress callback (percent, message).

    Returns:
        dict with output_path and applied_offset_seconds.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    out = output_path_override or output_path(input_path, "synced")

    if on_progress:
        on_progress(10, "Applying sync offset...")

    if abs(offset_seconds) < 0.001:
        # No meaningful offset, just copy
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-c", "copy",
            out,
        ], timeout=3600)
    elif offset_seconds > 0:
        # Delay audio by offset_seconds using adelay filter
        delay_ms = int(offset_seconds * 1000)
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-c:v", "copy",
            "-af", f"adelay={delay_ms}|{delay_ms}",
            "-c:a", "aac", "-b:a", "192k",
            out,
        ], timeout=3600)
    else:
        # Negative offset: trim audio from beginning
        trim_sec = abs(offset_seconds)
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-c:v", "copy",
            "-af", f"atrim=start={trim_sec},asetpts=PTS-STARTPTS",
            "-c:a", "aac", "-b:a", "192k",
            out,
        ], timeout=3600)

    if on_progress:
        on_progress(100, "Sync offset applied!")

    return {
        "output_path": out,
        "applied_offset_seconds": offset_seconds,
    }
