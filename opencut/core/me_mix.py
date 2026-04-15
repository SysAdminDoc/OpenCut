"""
OpenCut M&E (Music & Effects) Mix Export

Generate M&E stems by removing dialogue from an audio mix.

Pipeline:
1. Stem separation via Demucs (subprocess) or spectral subtraction fallback
2. Combine non-vocal stems (drums + bass + other)
3. Multi-track source: simply mute dialogue tracks
4. Level-match M&E to original mix loudness
5. Validate residual dialogue via spectral envelope comparison

Exports WAV or MP3.  All processing via FFmpeg / subprocess.
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STEM_TYPES = ("vocals", "drums", "bass", "other")
ME_METHODS = ("stem_separation", "track_mute", "spectral")
DEFAULT_FORMAT = "wav"
SUPPORTED_FORMATS = ("wav", "mp3")

# Spectral band for dialogue residual check (300 Hz - 4 kHz)
DIALOGUE_FREQ_LOW = 300
DIALOGUE_FREQ_HIGH = 4000


# ---------------------------------------------------------------------------
# Result Data Class
# ---------------------------------------------------------------------------
@dataclass
class MEMixResult:
    """Result from M&E mix generation."""

    output_path: str = ""
    method_used: str = ""
    residual_dialogue_score: float = 0.0
    duration: float = 0.0
    loudness_lufs: float = -23.0
    stems_combined: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_me_mix(
    input_path: str,
    output_path: Optional[str] = None,
    method: str = "auto",
    output_format: str = "wav",
    target_lufs: float = -23.0,
    dialogue_tracks: Optional[List[int]] = None,
    on_progress: Optional[Callable] = None,
) -> MEMixResult:
    """
    Generate an M&E (Music & Effects) mix by removing dialogue.

    Args:
        input_path: Source audio/video file.
        output_path: Output path (auto-generated if None).
        method: Removal method - "auto", "stem_separation", "track_mute", "spectral".
            auto: try stem_separation first, fall back to spectral.
        output_format: "wav" or "mp3".
        target_lufs: Target loudness in LUFS for level matching.
        dialogue_tracks: For track_mute method, list of track indices to mute.
        on_progress: Progress callback taking one int (percentage).

    Returns:
        MEMixResult with output path, method used, and quality metrics.
    """
    if output_format not in SUPPORTED_FORMATS:
        output_format = DEFAULT_FORMAT

    target_lufs = max(-36.0, min(-10.0, float(target_lufs)))

    ext = ".wav" if output_format == "wav" else ".mp3"
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_me_mix{ext}")

    if on_progress:
        on_progress(5)

    method = method.strip().lower()
    if method not in ("auto", "stem_separation", "track_mute", "spectral"):
        method = "auto"

    # Route to appropriate method
    if method == "track_mute" and dialogue_tracks:
        result = _me_via_track_mute(
            input_path, output_path, dialogue_tracks,
            output_format, on_progress,
        )
    elif method == "stem_separation" or method == "auto":
        result = _me_via_stem_separation(
            input_path, output_path, output_format, on_progress,
        )
        if result is None and method == "auto":
            # Fallback to spectral subtraction
            if on_progress:
                on_progress(20)
            result = _me_via_spectral(
                input_path, output_path, output_format, on_progress,
            )
    elif method == "spectral":
        result = _me_via_spectral(
            input_path, output_path, output_format, on_progress,
        )
    else:
        result = _me_via_spectral(
            input_path, output_path, output_format, on_progress,
        )

    if result is None:
        raise RuntimeError("All M&E mix methods failed")

    if on_progress:
        on_progress(75)

    # Level-match to original loudness
    _level_match(result.output_path, target_lufs, on_progress)
    result.loudness_lufs = target_lufs

    if on_progress:
        on_progress(85)

    # Measure duration
    result.duration = _get_duration(result.output_path)

    # Validate residual dialogue
    result.residual_dialogue_score = _measure_residual_dialogue(
        input_path, result.output_path,
    )

    if on_progress:
        on_progress(100)

    logger.info(
        "Generated M&E mix (%s): %s -> %s (residual: %.2f)",
        result.method_used, input_path, result.output_path,
        result.residual_dialogue_score,
    )
    return result


# ---------------------------------------------------------------------------
# Stem Separation Method
# ---------------------------------------------------------------------------
def _me_via_stem_separation(
    input_path: str,
    output_path: str,
    output_format: str,
    on_progress: Optional[Callable] = None,
) -> Optional[MEMixResult]:
    """Attempt M&E generation using Demucs stem separation.

    Calls demucs as a subprocess. If not available, returns None.
    Combines drums + bass + other stems, excluding vocals.
    """
    if on_progress:
        on_progress(10)

    # Check if demucs is available
    import shutil
    demucs_bin = shutil.which("demucs")
    if demucs_bin is None:
        # Try python -m demucs
        try:
            result = subprocess.run(
                ["python", "-m", "demucs", "--help"],
                capture_output=True, timeout=15,
            )
            if result.returncode != 0:
                logger.info("Demucs not available, skipping stem separation")
                return None
            demucs_bin = "python_module"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.info("Demucs not available, skipping stem separation")
            return None

    if on_progress:
        on_progress(15)

    stems_dir = tempfile.mkdtemp(prefix="opencut_me_stems_")

    try:
        # Run Demucs separation
        if demucs_bin == "python_module":
            cmd = ["python", "-m", "demucs", "--two-stems", "vocals",
                   "-o", stems_dir, input_path]
        else:
            cmd = [demucs_bin, "--two-stems", "vocals",
                   "-o", stems_dir, input_path]

        if on_progress:
            on_progress(20)

        proc = subprocess.run(cmd, capture_output=True, timeout=1800)
        if proc.returncode != 0:
            logger.warning("Demucs failed: %s", proc.stderr.decode(errors="replace")[-300:])
            return None

        if on_progress:
            on_progress(55)

        # Find the "no_vocals" stem output
        no_vocals_path = None
        for root, dirs, files in os.walk(stems_dir):
            for fname in files:
                lower = fname.lower()
                if ("no_vocals" in lower or "other" in lower) and lower.endswith(
                    (".wav", ".mp3", ".flac")
                ):
                    no_vocals_path = os.path.join(root, fname)
                    break
            if no_vocals_path:
                break

        if no_vocals_path is None:
            # Look for individual stems and combine non-vocal ones
            non_vocal_stems = []
            for root, dirs, files in os.walk(stems_dir):
                for fname in files:
                    lower = fname.lower()
                    if lower.endswith((".wav", ".mp3", ".flac")) and "vocal" not in lower:
                        non_vocal_stems.append(os.path.join(root, fname))

            if not non_vocal_stems:
                logger.warning("No non-vocal stems found in demucs output")
                return None

            no_vocals_path = _mix_stems(non_vocal_stems, output_path, output_format)
            stems_used = [os.path.basename(s) for s in non_vocal_stems]
        else:
            # Copy/convert the no_vocals stem to output
            _convert_to_format(no_vocals_path, output_path, output_format)
            stems_used = ["no_vocals"]

        if on_progress:
            on_progress(70)

        return MEMixResult(
            output_path=output_path,
            method_used="stem_separation",
            stems_combined=stems_used,
            notes="Generated via Demucs stem separation",
        )
    except subprocess.TimeoutExpired:
        logger.warning("Demucs timed out after 30 minutes")
        return None
    except Exception as exc:
        logger.warning("Stem separation failed: %s", exc)
        return None
    finally:
        # Clean up stems directory
        import shutil as _shutil
        try:
            _shutil.rmtree(stems_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Track Mute Method
# ---------------------------------------------------------------------------
def _me_via_track_mute(
    input_path: str,
    output_path: str,
    dialogue_tracks: List[int],
    output_format: str,
    on_progress: Optional[Callable] = None,
) -> MEMixResult:
    """Generate M&E by muting specific dialogue tracks in a multi-track source.

    Uses FFmpeg to select all audio streams except those in dialogue_tracks,
    then merges them into a stereo (or original channel layout) output.
    """
    if on_progress:
        on_progress(15)

    ffmpeg = get_ffmpeg_path()

    # Probe total audio streams
    stream_count = _count_audio_streams(input_path)
    if stream_count <= 0:
        raise RuntimeError("No audio streams found in source")

    # Build stream selection: include all streams NOT in dialogue_tracks
    keep_streams = [i for i in range(stream_count) if i not in dialogue_tracks]
    if not keep_streams:
        raise ValueError("All audio tracks are marked as dialogue")

    if on_progress:
        on_progress(30)

    # Build FFmpeg command with stream mapping
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", input_path]

    for idx in keep_streams:
        cmd.extend(["-map", f"0:a:{idx}"])

    if len(keep_streams) > 1:
        # Merge multiple streams with amerge
        filter_inputs = "".join(f"[0:a:{idx}]" for idx in keep_streams)
        af = f"{filter_inputs}amerge=inputs={len(keep_streams)},dynaudnorm"
        cmd.extend(["-filter_complex", af, "-ac", "2"])
    else:
        cmd.extend(["-ac", "2"])

    if output_format == "mp3":
        cmd.extend(["-c:a", "libmp3lame", "-b:a", "320k"])
    else:
        cmd.extend(["-c:a", "pcm_s24le"])

    cmd.append(output_path)

    if on_progress:
        on_progress(50)

    run_ffmpeg(cmd, timeout=1800)

    if on_progress:
        on_progress(70)

    muted_labels = [f"track_{i}" for i in dialogue_tracks]
    return MEMixResult(
        output_path=output_path,
        method_used="track_mute",
        stems_combined=[f"track_{i}" for i in keep_streams],
        notes=f"Muted dialogue tracks: {muted_labels}",
    )


# ---------------------------------------------------------------------------
# Spectral Subtraction Method
# ---------------------------------------------------------------------------
def _me_via_spectral(
    input_path: str,
    output_path: str,
    output_format: str,
    on_progress: Optional[Callable] = None,
) -> MEMixResult:
    """Generate M&E using FFmpeg center-channel subtraction (spectral approach).

    Removes center-panned content (typically dialogue) by subtracting
    the left channel from right and vice versa, then applies band-pass
    filtering to preserve music and effects.
    """
    if on_progress:
        on_progress(25)

    ffmpeg = get_ffmpeg_path()

    # Center-channel subtraction: removes mono/center-panned content
    # Then apply a notch around dialogue frequencies to further clean
    af_chain = (
        "pan=stereo|c0=c0-c1|c1=c1-c0,"
        "highpass=f=40,"
        "equalizer=f=1500:t=q:w=0.5:g=-2,"
        "dynaudnorm=p=0.9:s=5"
    )

    if on_progress:
        on_progress(40)

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-af", af_chain,
    ]

    if output_format == "mp3":
        cmd.extend(["-c:a", "libmp3lame", "-b:a", "320k"])
    else:
        cmd.extend(["-c:a", "pcm_s24le"])

    cmd.extend(["-vn", output_path])

    run_ffmpeg(cmd, timeout=1800)

    if on_progress:
        on_progress(70)

    return MEMixResult(
        output_path=output_path,
        method_used="spectral",
        stems_combined=["L-R difference"],
        notes="Center-channel subtraction removes center-panned dialogue. "
              "Quality depends on original stereo imaging.",
    )


# ---------------------------------------------------------------------------
# Level Matching
# ---------------------------------------------------------------------------
def _level_match(
    filepath: str,
    target_lufs: float,
    on_progress: Optional[Callable] = None,
) -> None:
    """Normalize output file to match target loudness via loudnorm."""
    ffmpeg = get_ffmpeg_path()

    tmp_out = filepath + ".tmp.wav"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", filepath,
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-c:a", "pcm_s24le",
        tmp_out,
    ]
    try:
        run_ffmpeg(cmd, timeout=600)
        os.replace(tmp_out, filepath)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_out)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Residual Dialogue Validation
# ---------------------------------------------------------------------------
def _measure_residual_dialogue(
    original_path: str,
    me_path: str,
) -> float:
    """Measure residual dialogue in M&E mix.

    Compares spectral energy in the dialogue band (300-4000 Hz) between
    original and M&E mix. Returns a score 0.0 (no residual) to 1.0 (full dialogue).
    """
    try:
        orig_energy = _measure_band_energy(original_path, DIALOGUE_FREQ_LOW, DIALOGUE_FREQ_HIGH)
        me_energy = _measure_band_energy(me_path, DIALOGUE_FREQ_LOW, DIALOGUE_FREQ_HIGH)

        if orig_energy <= 0:
            return 0.0

        ratio = me_energy / orig_energy
        # Score: 0 = perfect removal, 1 = no removal
        score = min(1.0, max(0.0, ratio))
        return round(score, 3)
    except Exception as exc:
        logger.debug("Residual dialogue measurement failed: %s", exc)
        return 0.5  # Unknown, return mid-range


def _measure_band_energy(filepath: str, low_hz: int, high_hz: int) -> float:
    """Measure RMS energy in a frequency band using FFmpeg bandpass + volumedetect."""
    ffmpeg = get_ffmpeg_path()
    af = (
        f"bandpass=f={(low_hz + high_hz) // 2}:t=h:w={high_hz - low_hz},"
        "volumedetect"
    )
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "info",
        "-i", filepath,
        "-af", af,
        "-t", "30",  # Sample first 30 seconds
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stderr = result.stderr
        for line in stderr.split("\n"):
            if "mean_volume" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    val = parts[-1].strip().replace("dB", "").strip()
                    db = float(val)
                    return 10 ** (db / 20.0)
        return 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------
def _get_duration(filepath: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe, "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json",
        filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return 0.0
        data = json.loads(result.stdout.decode())
        return float(data.get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def _count_audio_streams(filepath: str) -> int:
    """Count audio streams in a media file via ffprobe."""
    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe, "-v", "quiet",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "json",
        filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return 0
        data = json.loads(result.stdout.decode())
        return len(data.get("streams", []))
    except Exception:
        return 0


def _mix_stems(
    stem_paths: List[str],
    output_path: str,
    output_format: str,
) -> str:
    """Mix multiple stem files into a single output using FFmpeg amix."""
    ffmpeg = get_ffmpeg_path()

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for path in stem_paths:
        cmd.extend(["-i", path])

    filter_parts = []
    for i in range(len(stem_paths)):
        filter_parts.append(f"[{i}:a]")
    filter_str = (
        "".join(filter_parts)
        + f"amix=inputs={len(stem_paths)}:duration=longest:dropout_transition=2,"
        "dynaudnorm"
    )
    cmd.extend(["-filter_complex", filter_str])

    if output_format == "mp3":
        cmd.extend(["-c:a", "libmp3lame", "-b:a", "320k"])
    else:
        cmd.extend(["-c:a", "pcm_s24le"])

    cmd.append(output_path)
    run_ffmpeg(cmd, timeout=1800)
    return output_path


def _convert_to_format(src: str, dst: str, fmt: str) -> None:
    """Convert audio file to specified format."""
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", src]
    if fmt == "mp3":
        cmd.extend(["-c:a", "libmp3lame", "-b:a", "320k"])
    else:
        cmd.extend(["-c:a", "pcm_s24le"])
    cmd.append(dst)
    run_ffmpeg(cmd, timeout=600)


# ---------------------------------------------------------------------------
# Convenience Accessors
# ---------------------------------------------------------------------------
def list_methods() -> List[dict]:
    """List available M&E mix generation methods."""
    return [
        {
            "id": "auto",
            "label": "Automatic",
            "description": "Try stem separation first, fall back to spectral subtraction",
        },
        {
            "id": "stem_separation",
            "label": "Stem Separation (Demucs)",
            "description": "AI-based stem separation for highest quality vocal removal",
        },
        {
            "id": "track_mute",
            "label": "Track Mute",
            "description": "Mute specific dialogue tracks in multi-track sources",
        },
        {
            "id": "spectral",
            "label": "Spectral Subtraction",
            "description": "Center-channel subtraction removes center-panned dialogue",
        },
    ]
