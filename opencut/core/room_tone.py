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
import tempfile
import wave
from typing import Callable, List, Optional

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


# ---------------------------------------------------------------------------
# Enhanced Room Tone: Spectral Envelope Analysis & Synthesis
# ---------------------------------------------------------------------------

def analyze_room_tone(
    audio_path: str,
    n_fft: int = 2048,
    hop_length: int = 512,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Analyze room tone spectral envelope profile.

    Extracts the spectral shape of ambient room tone from the quietest
    segments of audio, producing a profile that can be used to synthesize
    matching room tone.

    Args:
        audio_path: Source audio/video file.
        n_fft: FFT window size for spectral analysis.
        hop_length: STFT hop length.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with spectral_envelope (list of {freq, magnitude_db}),
        rms_db, duration, sample_rate, method.
    """
    n_fft = max(256, min(8192, int(n_fft)))
    hop_length = max(64, min(n_fft, int(hop_length)))

    if on_progress:
        on_progress(10, "Analyzing room tone spectral envelope...")

    # Try librosa path for detailed spectral analysis. Any failure
    # (missing file, codec error, librosa internals) falls back to the
    # FFmpeg-based envelope estimator below — librosa.load can raise a
    # variety of exceptions (FileNotFoundError, RuntimeError from
    # audioread, NoBackendError, etc.) and surfacing any of them here
    # would defeat the whole point of having a fallback.
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = len(y) / sr

        # Find quietest 2-second segment
        chunk_samples = int(2.0 * sr)
        min_rms = float("inf")
        best_start = 0
        for i in range(0, len(y) - chunk_samples, chunk_samples // 2):
            chunk = y[i:i + chunk_samples]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if 0 < rms < min_rms:
                min_rms = rms
                best_start = i

        tone_segment = y[best_start:best_start + chunk_samples]

        # Compute STFT
        stft = librosa.stft(tone_segment, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        # Average across time to get spectral envelope
        avg_magnitude = np.mean(magnitude, axis=1)
        mag_db = librosa.amplitude_to_db(avg_magnitude, ref=np.max)

        freq_resolution = sr / n_fft
        envelope = []
        for i, db_val in enumerate(mag_db):
            freq = i * freq_resolution
            envelope.append({
                "freq": round(freq, 1),
                "magnitude_db": round(float(db_val), 2),
            })

        rms_db = round(20 * np.log10(min_rms + 1e-10), 1)

        if on_progress:
            on_progress(100, "Room tone analysis complete")

        return {
            "spectral_envelope": envelope,
            "rms_db": rms_db,
            "duration": round(duration, 3),
            "sample_rate": int(sr),
            "n_fft": n_fft,
            "method": "librosa",
        }
    except ImportError:
        pass
    except Exception as exc:
        # Log at info — the FFmpeg fallback below produces a usable
        # result; a load failure is interesting but not fatal.
        logger.info("librosa room-tone path failed (%s) — falling back to FFmpeg", exc)

    # Fallback: use FFmpeg astats for basic analysis
    if on_progress:
        on_progress(50, "Analyzing with FFmpeg (basic)...")

    quietest = _find_quietest_segments(audio_path, chunk_seconds=2.0)
    rms_db = quietest[0][1] if quietest else -60.0

    # Generate a simplified spectral envelope via FFmpeg
    # This is a rough approximation without full FFT analysis
    envelope = [
        {"freq": 50.0, "magnitude_db": rms_db - 5},
        {"freq": 100.0, "magnitude_db": rms_db - 3},
        {"freq": 200.0, "magnitude_db": rms_db - 2},
        {"freq": 500.0, "magnitude_db": rms_db - 4},
        {"freq": 1000.0, "magnitude_db": rms_db - 8},
        {"freq": 2000.0, "magnitude_db": rms_db - 15},
        {"freq": 4000.0, "magnitude_db": rms_db - 25},
        {"freq": 8000.0, "magnitude_db": rms_db - 40},
    ]

    total_dur = _get_duration(audio_path)

    if on_progress:
        on_progress(100, "Room tone analysis complete (basic)")

    return {
        "spectral_envelope": envelope,
        "rms_db": round(rms_db, 1),
        "duration": round(total_dur, 3),
        "sample_rate": 48000,
        "n_fft": n_fft,
        "method": "ffmpeg_basic",
    }


def synthesize_room_tone(
    profile: dict,
    duration: float = 10.0,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Synthesize room tone matching a spectral envelope profile.

    Generates pink-noise-like audio shaped to match the spectral profile
    from analyze_room_tone().

    Args:
        profile: Spectral profile dict from analyze_room_tone().
        duration: Desired output duration in seconds.
        output_path_str: Output WAV path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, duration, method.
    """
    duration = max(0.5, min(3600.0, float(duration)))
    envelope = profile.get("spectral_envelope", [])
    sr = int(profile.get("sample_rate", 48000))
    rms_db = float(profile.get("rms_db", -50.0))

    if output_path_str is None:
        output_path_str = os.path.join(
            tempfile.gettempdir(), f"opencut_synth_roomtone_{os.getpid()}.wav"
        )

    if on_progress:
        on_progress(10, "Synthesizing room tone from profile...")

    # Try numpy/scipy for spectral synthesis
    try:
        import numpy as np

        n_samples = int(sr * duration)

        # Generate white noise
        noise = np.random.randn(n_samples).astype(np.float32)

        # Apply spectral shaping via FFT
        fft = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(n_samples, 1.0 / sr)

        # Interpolate envelope to match FFT bins
        if envelope:
            env_freqs = [e["freq"] for e in envelope]
            env_mags = [10 ** (e["magnitude_db"] / 20.0) for e in envelope]

            # Linear interpolation
            shaped_mag = np.interp(freqs, env_freqs, env_mags, left=env_mags[0], right=env_mags[-1])
            fft *= shaped_mag

        # Inverse FFT
        audio = np.fft.irfft(fft, n=n_samples)

        # Normalize to target RMS
        target_rms = 10 ** (rms_db / 20.0)
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)

        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)

        # Write WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(output_path_str, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())

        if on_progress:
            on_progress(100, "Room tone synthesized")

        return {
            "output_path": output_path_str,
            "duration": duration,
            "method": "spectral_synthesis",
        }
    except ImportError:
        pass

    # Fallback: generate pink-ish noise via FFmpeg
    if on_progress:
        on_progress(50, "Synthesizing via FFmpeg anoisesrc...")

    ffmpeg = get_ffmpeg_path()
    # anoisesrc generates white noise; apply lowpass for approximate pink noise
    target_amp = max(0.001, 10 ** (rms_db / 20.0))
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", f"anoisesrc=d={duration}:c=pink:r={sr}:a={target_amp}",
        "-c:a", "pcm_s16le",
        output_path_str,
    ]
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Room tone synthesized (FFmpeg)")

    return {
        "output_path": output_path_str,
        "duration": duration,
        "method": "ffmpeg_pink_noise",
    }


def fill_cuts_with_room_tone(
    audio_path: str,
    cut_points: List[dict],
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Fill edit cut points with synthesized room tone for seamless audio.

    Analyzes the room tone of the source audio, synthesizes matching tone,
    and crossfades it into the specified cut points.

    Args:
        audio_path: Source audio/video file.
        cut_points: List of cut point dicts with {time, duration} where
                    time is the cut position and duration is the fill length.
        output_path_str: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, cuts_filled, total_fill_duration, method.
    """
    if output_path_str is None:
        output_path_str = _output_path(audio_path, "cuts_filled")

    if not cut_points:
        if on_progress:
            on_progress(100, "No cut points provided")
        ffmpeg = get_ffmpeg_path()
        cmd = [ffmpeg, "-i", audio_path, "-c", "copy", "-y", output_path_str]
        run_ffmpeg(cmd)
        return {
            "output_path": output_path_str,
            "cuts_filled": 0,
            "total_fill_duration": 0.0,
            "method": "none",
        }

    if on_progress:
        on_progress(10, "Analyzing room tone...")

    # Analyze room tone profile
    profile = analyze_room_tone(audio_path)

    # Find max fill duration needed
    max_fill = max(float(cp.get("duration", 0.5)) for cp in cut_points)
    total_fill_dur = sum(float(cp.get("duration", 0.5)) for cp in cut_points)

    if on_progress:
        on_progress(30, "Synthesizing room tone fill...")

    # Synthesize tone
    tone_result = synthesize_room_tone(profile, duration=max_fill + 1.0)
    tone_path = tone_result["output_path"]

    try:
        if on_progress:
            on_progress(50, f"Filling {len(cut_points)} cut points...")

        total_duration = _get_duration(audio_path)

        # Build filter that overlays room tone at each cut point
        enable_parts = []
        for cp in cut_points:
            t = max(0, float(cp.get("time", 0)))
            dur = max(0.05, float(cp.get("duration", 0.5)))
            min(0.02, dur * 0.1)
            s = max(0, t - dur / 2)
            e = min(total_duration, t + dur / 2)
            enable_parts.append(f"between(t,{s},{e})")

        enable_expr = "+".join(enable_parts)
        tone_vol = f"volume=enable='{enable_expr}':volume=1"

        fc = (
            f"[1:a]aloop=loop=-1:size={int(48000 * total_duration)},"
            f"atrim=duration={total_duration},{tone_vol}[tone];"
            f"[0:a][tone]amix=inputs=2:duration=first:dropout_transition=0[out]"
        )

        ffmpeg = get_ffmpeg_path()
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", audio_path,
            "-i", tone_path,
            "-filter_complex", fc,
            "-map", "[out]",
            "-map", "0:v?",
            "-c:v", "copy",
            "-c:a", "pcm_s16le",
            output_path_str,
        ]
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(100, f"Filled {len(cut_points)} cuts with room tone")

        method = f"synthesized_{profile.get('method', 'unknown')}"

        return {
            "output_path": output_path_str,
            "cuts_filled": len(cut_points),
            "total_fill_duration": round(total_fill_dur, 3),
            "method": method,
        }
    finally:
        try:
            os.unlink(tone_path)
        except OSError:
            pass
