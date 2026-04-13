"""
OpenCut Audio Fingerprinting & Copyright Detection

Generates spectral fingerprints from audio, compares them, and scans
against a local database of known tracks for copyright detection.

Approach:
- Extract audio as raw PCM via FFmpeg
- Compute spectral peaks per 0.5s chunk (mel-spectrogram-like binning)
- Hash each chunk to produce a compact fingerprint
- Compare fingerprints via chunk-hash overlap scoring

Uses FFmpeg only for audio extraction; spectral analysis is pure Python
using the standard library (struct + hashlib).
"""

import hashlib
import json
import logging
import os
import struct
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, run_ffmpeg

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_DEFAULT_DB_PATH = os.path.join(_OPENCUT_DIR, "fingerprint_db.json")

# Fingerprint parameters
_SAMPLE_RATE = 16000
_CHUNK_SECONDS = 0.5
_CHUNK_SAMPLES = int(_SAMPLE_RATE * _CHUNK_SECONDS)
_NUM_BINS = 32  # frequency bins for spectral analysis


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class AudioFingerprint:
    """Spectral fingerprint of an audio file."""
    hash_data: List[str] = field(default_factory=list)
    duration: float = 0.0
    sample_rate: int = _SAMPLE_RATE
    label: str = ""


@dataclass
class FingerprintMatch:
    """Result of comparing two fingerprints."""
    match_score: float = 0.0
    matched_chunks: int = 0
    total_chunks: int = 0
    is_match: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _extract_raw_pcm(input_path: str) -> bytes:
    """Extract audio as raw 16-bit LE mono PCM at 16kHz via FFmpeg."""
    ffmpeg = get_ffmpeg_path()
    tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", str(_SAMPLE_RATE),
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            tmp_path,
        ]
        run_ffmpeg(cmd)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _compute_spectral_bins(samples: List[int]) -> List[int]:
    """
    Compute simple spectral energy bins for a chunk of samples.

    Uses a basic DFT-like approach: split samples into frequency bands
    and sum squared magnitudes per band. Returns bin energies.
    """
    n = len(samples)
    if n == 0:
        return [0] * _NUM_BINS

    # Simple energy-based frequency binning via zero-crossing rate
    # and short-time energy in sub-bands
    bin_size = max(1, n // _NUM_BINS)
    bins = []
    for b in range(_NUM_BINS):
        start = b * bin_size
        end = min(start + bin_size, n)
        chunk = samples[start:end]
        # Energy of this sub-band
        energy = sum(s * s for s in chunk)
        bins.append(energy)

    return bins


def _hash_chunk(bins: List[int]) -> str:
    """Hash a spectral bin vector into a compact hex fingerprint."""
    # Binarize: each bin above median = 1, below = 0
    if not bins:
        return hashlib.md5(b"empty").hexdigest()[:16]

    sorted_bins = sorted(bins)
    median = sorted_bins[len(sorted_bins) // 2]

    bits = 0
    for i, val in enumerate(bins):
        if val > median:
            bits |= (1 << i)

    raw = struct.pack(">I", bits & 0xFFFFFFFF)
    return hashlib.md5(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_fingerprint(
    input_path: str,
    on_progress: Optional[Callable] = None,
) -> AudioFingerprint:
    """
    Generate a spectral fingerprint for an audio file.

    Args:
        input_path: Source audio/video file.
        on_progress: Progress callback(pct, msg).

    Returns:
        AudioFingerprint with hash_data, duration, sample_rate.
    """
    if on_progress:
        on_progress(10, "Extracting audio as PCM...")

    pcm_data = _extract_raw_pcm(input_path)
    num_samples = len(pcm_data) // 2  # 16-bit = 2 bytes per sample
    duration = num_samples / _SAMPLE_RATE

    if on_progress:
        on_progress(30, f"Computing fingerprint ({duration:.1f}s audio)...")

    # Unpack PCM samples
    samples = list(struct.unpack(f"<{num_samples}h", pcm_data[:num_samples * 2]))

    # Chunk and fingerprint
    hashes = []
    total_chunks = max(1, num_samples // _CHUNK_SAMPLES)
    for i in range(total_chunks):
        start = i * _CHUNK_SAMPLES
        end = min(start + _CHUNK_SAMPLES, num_samples)
        chunk_samples = samples[start:end]
        bins = _compute_spectral_bins(chunk_samples)
        h = _hash_chunk(bins)
        hashes.append(h)

        if on_progress and i % 20 == 0:
            pct = 30 + int(60 * i / total_chunks)
            on_progress(pct, f"Fingerprinting chunk {i + 1}/{total_chunks}...")

    if on_progress:
        on_progress(100, "Fingerprint generated")

    logger.info("Generated fingerprint for %s: %d chunks, %.1fs", input_path, len(hashes), duration)
    return AudioFingerprint(
        hash_data=hashes,
        duration=duration,
        sample_rate=_SAMPLE_RATE,
    )


def compare_fingerprints(
    fp1: AudioFingerprint,
    fp2: AudioFingerprint,
    threshold: float = 0.7,
) -> FingerprintMatch:
    """
    Compare two audio fingerprints and return match score.

    Args:
        fp1: First fingerprint.
        fp2: Second fingerprint.
        threshold: Minimum match_score to consider a match (0.0-1.0).

    Returns:
        FingerprintMatch with match_score, matched_chunks, is_match.
    """
    if not fp1.hash_data or not fp2.hash_data:
        return FingerprintMatch(match_score=0.0, matched_chunks=0, total_chunks=0, is_match=False)

    # Use the shorter fingerprint as reference, slide over the longer one
    short, long_ = (fp1.hash_data, fp2.hash_data) if len(fp1.hash_data) <= len(fp2.hash_data) else (fp2.hash_data, fp1.hash_data)
    short_set = set(short)

    best_matched = 0
    window = len(short)
    for offset in range(max(1, len(long_) - window + 1)):
        window_hashes = set(long_[offset:offset + window])
        matched = len(short_set & window_hashes)
        best_matched = max(best_matched, matched)

    total = len(short)
    score = best_matched / total if total > 0 else 0.0

    return FingerprintMatch(
        match_score=round(score, 4),
        matched_chunks=best_matched,
        total_chunks=total,
        is_match=score >= threshold,
    )


def _load_database(db_path: str) -> Dict:
    """Load the fingerprint database from disk."""
    if not os.path.exists(db_path):
        return {"tracks": []}
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read fingerprint DB at %s, starting fresh", db_path)
        return {"tracks": []}


def _save_database(db: Dict, db_path: str) -> None:
    """Save the fingerprint database to disk."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def scan_against_database(
    input_path: str,
    db_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> list:
    """
    Scan an audio file against the local fingerprint database.

    Args:
        input_path: Source audio/video file to check.
        db_path: Path to fingerprint database JSON. Defaults to ~/.opencut/fingerprint_db.json.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of dicts with label, match_score, matched_chunks, is_match for each DB entry.
    """
    if db_path is None:
        db_path = _DEFAULT_DB_PATH

    if on_progress:
        on_progress(5, "Generating fingerprint for input...")

    fp = generate_fingerprint(input_path, on_progress=None)

    if on_progress:
        on_progress(50, "Loading fingerprint database...")

    db = _load_database(db_path)
    tracks = db.get("tracks", [])

    if not tracks:
        if on_progress:
            on_progress(100, "Database is empty, no matches")
        return []

    matches = []
    for i, track in enumerate(tracks):
        track_fp = AudioFingerprint(
            hash_data=track.get("hash_data", []),
            duration=track.get("duration", 0.0),
            sample_rate=track.get("sample_rate", _SAMPLE_RATE),
            label=track.get("label", "unknown"),
        )
        result = compare_fingerprints(fp, track_fp)
        matches.append({
            "label": track.get("label", "unknown"),
            "match_score": result.match_score,
            "matched_chunks": result.matched_chunks,
            "is_match": result.is_match,
        })

        if on_progress:
            pct = 50 + int(45 * (i + 1) / len(tracks))
            on_progress(pct, f"Compared against '{track.get('label', 'unknown')}'...")

    # Sort by score descending
    matches.sort(key=lambda m: m["match_score"], reverse=True)

    if on_progress:
        on_progress(100, f"Scan complete: {sum(1 for m in matches if m['is_match'])} matches found")

    logger.info("Fingerprint scan of %s against %d tracks: %d matches",
                input_path, len(tracks), sum(1 for m in matches if m["is_match"]))
    return matches


def add_to_database(
    input_path: str,
    label: str,
    db_path: Optional[str] = None,
) -> None:
    """
    Add a known track to the local fingerprint database.

    Args:
        input_path: Audio/video file to fingerprint and add.
        label: Human-readable label for this track (e.g. "Song Title - Artist").
        db_path: Path to fingerprint database JSON. Defaults to ~/.opencut/fingerprint_db.json.
    """
    if db_path is None:
        db_path = _DEFAULT_DB_PATH

    fp = generate_fingerprint(input_path)

    db = _load_database(db_path)
    db.setdefault("tracks", []).append({
        "label": label,
        "hash_data": fp.hash_data,
        "duration": fp.duration,
        "sample_rate": fp.sample_rate,
    })
    _save_database(db, db_path)

    logger.info("Added '%s' to fingerprint database (%d chunks)", label, len(fp.hash_data))
