"""
OpenCut Best Take Selection

Score alternative recordings of the same content on audio quality metrics
(clarity/SNR, energy/RMS, consistency/loudness variance) and recommend the best.
"""

import logging
import os
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TakeScore:
    """Score for a single take."""
    file_path: str
    overall_score: float = 0.0
    clarity_score: float = 0.0
    energy_score: float = 0.0
    consistency_score: float = 0.0
    recommended: bool = False


@dataclass
class TakeScoreResult:
    """Result of scoring multiple takes."""
    takes: List[TakeScore] = field(default_factory=list)
    best_take: str = ""
    total_scored: int = 0


# ---------------------------------------------------------------------------
# Audio analysis helpers
# ---------------------------------------------------------------------------
def _get_audio_stats(filepath: str) -> Dict:
    """Run FFmpeg astats filter and parse results.

    Returns dict with rms_level, peak_level, noise_floor, dynamic_range.
    """
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-nostats",
        "-i", filepath,
        "-filter:a", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:key=lavfi.astats.Overall.Peak_level:key=lavfi.astats.Overall.Noise_floor:key=lavfi.astats.Overall.Dynamic_range",
        "-f", "null", "-",
    ]
    try:
        result = _sp.run(cmd, capture_output=True, timeout=120)
        stderr = result.stderr.decode(errors="replace")
        stdout = result.stdout.decode(errors="replace")
        combined = stderr + "\n" + stdout

        stats = {
            "rms_level": -30.0,
            "peak_level": -6.0,
            "noise_floor": -60.0,
            "dynamic_range": 30.0,
        }

        import re
        # Parse from metadata lines
        for key, field_name in [
            ("RMS_level", "rms_level"),
            ("Peak_level", "peak_level"),
            ("Noise_floor", "noise_floor"),
            ("Dynamic_range", "dynamic_range"),
        ]:
            pattern = rf"{key}[=:]\s*(-?\d+\.?\d*)"
            match = re.search(pattern, combined)
            if match:
                try:
                    stats[field_name] = float(match.group(1))
                except ValueError:
                    pass

        return stats
    except Exception as e:
        logger.debug("astats failed for %s: %s", filepath, e)
        return {
            "rms_level": -30.0,
            "peak_level": -6.0,
            "noise_floor": -60.0,
            "dynamic_range": 30.0,
        }


def _get_loudness_stats(filepath: str) -> Dict:
    """Run FFmpeg ebur128 filter and parse integrated loudness + LRA.

    Returns dict with integrated_lufs, loudness_range, true_peak.
    """
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-nostats",
        "-i", filepath,
        "-filter:a", "ebur128=peak=true",
        "-f", "null", "-",
    ]
    try:
        result = _sp.run(cmd, capture_output=True, timeout=120)
        stderr = result.stderr.decode(errors="replace")

        stats = {
            "integrated_lufs": -23.0,
            "loudness_range": 10.0,
            "true_peak": -1.0,
        }

        import re
        # Integrated loudness
        match = re.search(r"I:\s*(-?\d+\.?\d*)\s*LUFS", stderr)
        if match:
            stats["integrated_lufs"] = float(match.group(1))

        # Loudness range
        match = re.search(r"LRA:\s*(-?\d+\.?\d*)\s*LU", stderr)
        if match:
            stats["loudness_range"] = float(match.group(1))

        # True peak
        match = re.search(r"Peak:\s*(-?\d+\.?\d*)\s*dBFS", stderr)
        if match:
            stats["true_peak"] = float(match.group(1))

        return stats
    except Exception as e:
        logger.debug("ebur128 failed for %s: %s", filepath, e)
        return {
            "integrated_lufs": -23.0,
            "loudness_range": 10.0,
            "true_peak": -1.0,
        }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _normalize_score(value: float, low: float, high: float) -> float:
    """Normalize a value to 0.0-1.0 range. Clamps to bounds."""
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _score_clarity(audio_stats: Dict, loudness_stats: Dict) -> float:
    """Score audio clarity based on SNR estimation.

    Higher SNR (RMS - noise floor) = better clarity. Score 0-100.
    """
    rms = audio_stats.get("rms_level", -30.0)
    noise = audio_stats.get("noise_floor", -60.0)
    # SNR approximation: difference between RMS and noise floor
    # Typical good SNR: 30-60 dB, poor: < 20 dB
    snr = abs(rms - noise) if noise < rms else 5.0
    return _normalize_score(snr, 5.0, 60.0) * 100.0


def _score_energy(audio_stats: Dict, loudness_stats: Dict) -> float:
    """Score audio energy based on RMS level and dynamic range.

    Good energy: RMS not too quiet, reasonable dynamic range. Score 0-100.
    """
    rms = audio_stats.get("rms_level", -30.0)
    dynamic_range = audio_stats.get("dynamic_range", 30.0)

    # RMS score: -20 is great, -40 is bad
    rms_score = _normalize_score(rms, -45.0, -12.0)

    # Dynamic range: 10-30 is ideal for speech, too high = inconsistent
    dr_score = 1.0 - abs(dynamic_range - 20.0) / 30.0
    dr_score = max(0.0, min(1.0, dr_score))

    return ((rms_score * 0.6) + (dr_score * 0.4)) * 100.0


def _score_consistency(loudness_stats: Dict) -> float:
    """Score loudness consistency (lower variance = better).

    Low LRA (loudness range) = more consistent. Score 0-100.
    """
    lra = loudness_stats.get("loudness_range", 10.0)
    # Good LRA for speech: 3-8 LU, bad: > 20 LU
    return _normalize_score(25.0 - lra, 0.0, 25.0) * 100.0


def score_takes(
    takes: List[str],
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> TakeScoreResult:
    """Score multiple takes and recommend the best one.

    Args:
        takes: List of file paths to alternative recordings.
        on_progress: Progress callback ``(percent, message)``.

    Returns:
        TakeScoreResult with scored takes and best recommendation.
    """
    if not takes:
        return TakeScoreResult()

    result = TakeScoreResult()
    total = len(takes)

    for i, filepath in enumerate(takes):
        if on_progress:
            pct = int(10 + (80 * i / total))
            on_progress(pct, f"Analyzing take {i+1}/{total}: {os.path.basename(filepath)}")

        if not os.path.isfile(filepath):
            logger.warning("Take file not found: %s", filepath)
            result.takes.append(TakeScore(
                file_path=filepath,
                overall_score=0.0,
            ))
            continue

        # Gather audio statistics
        audio_stats = _get_audio_stats(filepath)
        loudness_stats = _get_loudness_stats(filepath)

        # Compute sub-scores
        clarity = _score_clarity(audio_stats, loudness_stats)
        energy = _score_energy(audio_stats, loudness_stats)
        consistency = _score_consistency(loudness_stats)

        # Weighted overall score
        overall = (clarity * 0.4) + (energy * 0.3) + (consistency * 0.3)

        result.takes.append(TakeScore(
            file_path=filepath,
            overall_score=round(overall, 1),
            clarity_score=round(clarity, 1),
            energy_score=round(energy, 1),
            consistency_score=round(consistency, 1),
        ))

    result.total_scored = len(result.takes)

    # Mark the best take
    if result.takes:
        best = max(result.takes, key=lambda t: t.overall_score)
        best.recommended = True
        result.best_take = best.file_path

    if on_progress:
        on_progress(100, "Take scoring complete")

    return result


# ---------------------------------------------------------------------------
# Find repeated takes (group similar files)
# ---------------------------------------------------------------------------
def find_repeated_takes(
    file_paths: List[str],
    similarity_threshold: float = 0.8,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> List[List[str]]:
    """Group files that appear to be repeated takes of the same content.

    Uses filename similarity and duration matching to identify groups
    of related takes.

    Args:
        file_paths: List of file paths to analyze.
        similarity_threshold: 0.0-1.0 threshold for grouping (default 0.8).
        on_progress: Progress callback.

    Returns:
        List of groups, where each group is a list of file paths.
    """
    if not file_paths:
        return []

    if on_progress:
        on_progress(5, "Gathering file metadata...")

    # Build metadata for each file
    file_info = []
    for i, fp in enumerate(file_paths):
        if not os.path.isfile(fp):
            continue
        info = get_video_info(fp)
        base = os.path.splitext(os.path.basename(fp))[0]
        file_info.append({
            "path": fp,
            "basename": base,
            "duration": info.get("duration", 0),
        })
        if on_progress:
            pct = int(5 + (40 * i / len(file_paths)))
            on_progress(pct, f"Probing {i+1}/{len(file_paths)}...")

    if on_progress:
        on_progress(50, "Finding similar takes...")

    # Group by name similarity + duration proximity
    groups: List[List[str]] = []
    assigned = set()

    for i, fi in enumerate(file_info):
        if fi["path"] in assigned:
            continue

        group = [fi["path"]]
        assigned.add(fi["path"])

        for j, fj in enumerate(file_info):
            if i == j or fj["path"] in assigned:
                continue

            # Name similarity: strip trailing numbers and compare
            name_sim = _name_similarity(fi["basename"], fj["basename"])

            # Duration similarity: within 20% of each other
            dur_sim = _duration_similarity(fi["duration"], fj["duration"])

            combined = (name_sim * 0.6) + (dur_sim * 0.4)
            if combined >= similarity_threshold:
                group.append(fj["path"])
                assigned.add(fj["path"])

        if len(group) > 1:
            groups.append(group)

    if on_progress:
        on_progress(100, f"Found {len(groups)} take groups")

    return groups


def _name_similarity(a: str, b: str) -> float:
    """Compute name similarity between two basenames.

    Strips trailing numbers/underscores/take markers and compares stems.
    """
    import re

    def _stem(name: str) -> str:
        # Remove trailing take markers: _take1, _v2, _02, (1), etc.
        name = re.sub(r'[_\-\s]*(take|v|ver|version)?[_\-\s]*\d+\s*$', '', name, flags=re.I)
        name = re.sub(r'\s*\(\d+\)\s*$', '', name)
        return name.lower().strip()

    sa = _stem(a)
    sb = _stem(b)

    if not sa or not sb:
        return 0.0
    if sa == sb:
        return 1.0

    # Character overlap ratio
    common = sum(1 for c in sa if c in sb)
    return (2 * common) / (len(sa) + len(sb))


def _duration_similarity(a: float, b: float) -> float:
    """Compute duration similarity (1.0 = identical, 0.0 = very different)."""
    if a <= 0 or b <= 0:
        return 0.0
    ratio = min(a, b) / max(a, b)
    return ratio
