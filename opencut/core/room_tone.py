"""
OpenCut Room Tone Matching & Generation

Tools for extracting, generating, and inserting room tone:
- Extract room tone from quietest non-silent segments of audio
- Generate seamless room tone loops from a reference
- Fill silence gaps in audio with room tone for continuity

Uses FFmpeg astats, aloop, acrossfade, and volume-based gap detection.
"""

import json
import logging
import os
from typing import Callable, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg
from opencut.helpers import output_path as _output_path

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_duration(filepath: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    ffprobe = get_ffprobe_path()
    if not ffprobe:
        raise RuntimeError("ffprobe not found")
    import subprocess
    result = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", filepath],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    info = json.loads(result.stdout)
    return float(info.get("format", {}).get("duration", 0.0))


def _detect_silence_segments(filepath: str, threshold_db: float = -50.0, min_duration: float = 0.3) -> list:
    """
    Detect silence segments in audio using FFmpeg silencedetect.

    Returns list of dicts with start, end, duration.
    """
    ffmpeg = get_ffmpeg_path()
    import subprocess
    cmd = [
        ffmpeg, "-hide_banner", "-i", filepath,
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)

    segments = []
    lines = result.stderr.split("\n")
    current_start = None
    for line in lines:
        if "silence_start:" in line:
            try:
                current_start = float(line.split("silence_start:")[1].strip().split()[0])
            except (ValueError, IndexError):
                current_start = None
        elif "silence_end:" in line and current_start is not None:
            try:
                parts = line.split("silence_end:")[1].strip().split()
                end = float(parts[0])
                segments.append({
                    "start": current_start,
                    "end": end,
                    "duration": end - current_start,
                })
            except (ValueError, IndexError):
                pass
            current_start = None

    return segments


def _find_quietest_segments(filepath: str, chunk_seconds: float = 2.0) -> list:
    """
    Analyze audio in chunks to find the quietest non-silent segments.

    Returns list of (start_time, rms_level) tuples sorted by RMS ascending.
    """
    ffmpeg = get_ffmpeg_path()
    total_duration = _get_duration(filepath)
    if total_duration <= 0:
        return []

    import subprocess
    # Use astats to measure RMS per chunk
    chunks = []
    t = 0.0
    while t < total_duration:
        chunk_dur = min(chunk_seconds, total_duration - t)
        if chunk_dur < 0.1:
            break
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-ss", str(t), "-t", str(chunk_dur),
            "-i", filepath,
            "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        # Parse RMS from stderr
        rms = -100.0
        for line in result.stderr.split("\n"):
            if "RMS_level" in line:
                try:
                    val = line.split("=")[-1].strip()
                    rms = float(val)
                except (ValueError, IndexError):
                    pass
        # Ignore true silence (below -80dB) - we want room tone, not dead air
        if rms > -80.0:
            chunks.append((t, rms))
        t += chunk_seconds

    # Sort by RMS ascending (quietest first)
    chunks.sort(key=lambda x: x[1])
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_room_tone(
    input_path: str,
    duration: float = 5.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Extract room tone from the quietest non-silent segments of audio.

    Finds segments with the lowest RMS energy that still contain content
    (not dead silence) and extracts the best one as a room tone reference.

    Args:
        input_path: Source audio/video file.
        duration: Desired room tone duration in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, tone_profile (start_time, rms_level, duration).
    """
    duration = max(0.5, min(60.0, float(duration)))

    if on_progress:
        on_progress(10, "Analyzing audio for quiet segments...")

    quietest = _find_quietest_segments(input_path, chunk_seconds=duration)

    if not quietest:
        raise RuntimeError("No suitable room tone segments found in audio")

    best_start, best_rms = quietest[0]

    if on_progress:
        on_progress(50, f"Extracting room tone from {best_start:.1f}s (RMS: {best_rms:.1f}dB)...")

    out = _output_path(input_path, "roomtone")
    # Change extension to .wav for clean audio
    out = os.path.splitext(out)[0] + ".wav"

    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(best_start), "-t", str(duration),
        "-i", input_path,
        "-vn",
        "-c:a", "pcm_s16le",
        out,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Room tone extracted")

    logger.info("Extracted room tone from %s at %.1fs (RMS: %.1fdB)", input_path, best_start, best_rms)
    return {
        "output_path": out,
        "tone_profile": {
            "start_time": best_start,
            "rms_level": best_rms,
            "duration": duration,
        },
    }


def generate_room_tone(
    reference_path: str,
    duration: float = 30.0,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Generate seamless room tone by looping a reference with crossfade.

    Takes a short room tone reference and loops it with crossfade at loop
    points to create a longer seamless fill.

    Args:
        reference_path: Short room tone reference file.
        duration: Desired output duration in seconds.
        output_path: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, duration.
    """
    duration = max(1.0, min(3600.0, float(duration)))

    if output_path is None:
        output_path = _output_path(reference_path, "roomtone_gen")
        output_path = os.path.splitext(output_path)[0] + ".wav"

    if on_progress:
        on_progress(10, f"Generating {duration:.1f}s room tone from reference...")

    ref_duration = _get_duration(reference_path)
    if ref_duration <= 0:
        raise RuntimeError("Could not determine reference audio duration")

    ffmpeg = get_ffmpeg_path()

    if duration <= ref_duration:
        # Just trim reference to desired duration
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", reference_path,
            "-t", str(duration),
            "-vn", "-c:a", "pcm_s16le",
            output_path,
        ]
        run_ffmpeg(cmd)
    else:
        # Loop with crossfade
        # Number of loops needed (round up)
        loops_needed = int(duration / ref_duration) + 2
        min(0.5, ref_duration * 0.2)  # 20% of ref or 0.5s max

        # Use aloop to repeat, then trim
        # aloop loop count is additional loops, so loops_needed - 1
        loop_count = max(1, loops_needed - 1)
        af = f"aloop=loop={loop_count}:size={int(ref_duration * 48000)},afade=t=out:st={duration - 0.1}:d=0.1"

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", reference_path,
            "-af", af,
            "-t", str(duration),
            "-vn", "-c:a", "pcm_s16le",
            output_path,
        ]
        run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Room tone generated")

    logger.info("Generated %.1fs room tone: %s", duration, output_path)
    return {"output_path": output_path, "duration": duration}


def fill_gaps_with_tone(
    input_path: str,
    tone_path: str,
    gap_threshold_db: float = -50.0,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Detect silence gaps in audio and fill them with room tone.

    Args:
        input_path: Source audio/video file with silence gaps.
        tone_path: Room tone file to fill gaps with.
        gap_threshold_db: dB threshold below which audio is considered a gap.
        output_path: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, gaps_filled (count), total_gap_duration.
    """
    if output_path is None:
        output_path = _output_path(input_path, "filled")

    gap_threshold_db = max(-80.0, min(-10.0, float(gap_threshold_db)))

    if on_progress:
        on_progress(10, "Detecting silence gaps...")

    gaps = _detect_silence_segments(input_path, threshold_db=gap_threshold_db)

    if not gaps:
        # No gaps found, just copy input
        if on_progress:
            on_progress(100, "No silence gaps detected")
        ffmpeg = get_ffmpeg_path()
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-c", "copy",
            output_path,
        ]
        run_ffmpeg(cmd)
        return {"output_path": output_path, "gaps_filled": 0, "total_gap_duration": 0.0}

    if on_progress:
        on_progress(30, f"Found {len(gaps)} gaps, preparing room tone fill...")

    total_gap_dur = sum(g["duration"] for g in gaps)

    # Strategy: use FFmpeg to overlay room tone on the silence segments.
    # Build a complex filter that mixes room tone into gap regions.
    ffmpeg = get_ffmpeg_path()

    # Generate volume envelope: full volume during gaps, zero otherwise
    # We use the sidechaingate approach: invert the silence detection
    # Simpler approach: overlay tone at reduced volume on the full track,
    # but only enable it during gaps using volume automation.
    # Simplest FFmpeg approach: use amerge + volume keyframes

    # Build volume expression for the tone track that's only audible during gaps
    total_duration = _get_duration(input_path)
    volume_expr_parts = []
    for g in gaps:
        # Enable tone during this gap with small fade in/out
        fade_ms = 0.02
        gs = max(0, g["start"] - fade_ms)
        ge = min(total_duration, g["end"] + fade_ms)
        volume_expr_parts.append(f"between(t,{gs},{ge})")

    if volume_expr_parts:
        volume_enable = "+".join(volume_expr_parts)
        volume_expr = f"volume=enable='{volume_enable}':volume=1"
    else:
        volume_expr = "volume=0"

    # Use filter_complex to mix original with gated room tone
    fc = (
        f"[1:a]aloop=loop=-1:size={int(48000 * total_duration)},atrim=duration={total_duration},"
        f"{volume_expr}[tone];"
        f"[0:a][tone]amix=inputs=2:duration=first:dropout_transition=0[out]"
    )

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-i", tone_path,
        "-filter_complex", fc,
        "-map", "[out]",
        "-map", "0:v?",
        "-c:v", "copy",
        "-c:a", "pcm_s16le",
        output_path,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, f"Filled {len(gaps)} gaps with room tone")

    logger.info("Filled %d gaps (%.1fs total) in %s", len(gaps), total_gap_dur, input_path)
    return {
        "output_path": output_path,
        "gaps_filled": len(gaps),
        "total_gap_duration": round(total_gap_dur, 3),
    }
