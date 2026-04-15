"""
OpenCut Music Mood Morph Module (Category 75)

Transform music mood and energy over time.  Analyze audio for tempo, energy,
and spectral characteristics using FFmpeg.  Apply transformations like brighten,
darken, energize, calm, build, and drop using keyframeable mood curves with
FFmpeg EQ, atempo, and compand filters.

Functions:
    analyze_audio_properties  - Get tempo, energy, spectral centroid from audio
    apply_mood_morph          - Apply mood transformation to audio
    apply_keyframed_morph     - Apply time-varying mood curve
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants & Transform Definitions
# ---------------------------------------------------------------------------
MOOD_TRANSFORMS = {
    "brighten": {
        "description": "Boost highs, increase tempo slightly for uplifting feel",
        "eq_treble": 4.0,     # dB boost at high shelf
        "eq_bass": -1.0,      # dB at low shelf
        "tempo_factor": 1.04,  # Speed up 4%
        "compress": False,
    },
    "darken": {
        "description": "Boost lows, reduce highs for moody atmosphere",
        "eq_treble": -5.0,
        "eq_bass": 4.0,
        "tempo_factor": 0.97,
        "compress": False,
    },
    "energize": {
        "description": "Compress, boost, speed up for high-energy feel",
        "eq_treble": 3.0,
        "eq_bass": 3.0,
        "tempo_factor": 1.08,
        "compress": True,
        "compand": "0.3|0.3:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2",
    },
    "calm": {
        "description": "Expand dynamics, slow down, reduce highs for relaxation",
        "eq_treble": -3.0,
        "eq_bass": 1.0,
        "tempo_factor": 0.93,
        "compress": False,
    },
    "build": {
        "description": "Gradual energy increase over time",
        "eq_treble": 2.0,
        "eq_bass": 2.0,
        "tempo_factor": 1.03,
        "compress": True,
        "compand": "0.3|0.3:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2",
        "is_gradient": True,
    },
    "drop": {
        "description": "Sudden energy decrease (bass drop effect)",
        "eq_treble": -6.0,
        "eq_bass": 8.0,
        "tempo_factor": 0.90,
        "compress": True,
        "compand": "0.05|0.1:1|1:-90/-60|-60/-20|-20/-10:6:0:-90:0.1",
    },
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class AudioProperties:
    """Audio analysis results."""
    duration: float = 0.0
    sample_rate: int = 44100
    channels: int = 2
    bitrate: int = 0
    codec: str = ""
    estimated_tempo: float = 120.0
    estimated_energy: float = 0.5
    spectral_centroid: float = 2000.0

    def to_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bitrate": self.bitrate,
            "codec": self.codec,
            "estimated_tempo": round(self.estimated_tempo, 1),
            "estimated_energy": round(self.estimated_energy, 3),
            "spectral_centroid": round(self.spectral_centroid, 1),
        }


@dataclass
class MoodKeyframe:
    """A single keyframe in a mood curve."""
    time: float        # Position in seconds
    mood: str          # Transform name
    intensity: float   # 0.0-1.0

    def to_dict(self) -> dict:
        return {
            "time": round(self.time, 3),
            "mood": self.mood,
            "intensity": round(self.intensity, 2),
        }


@dataclass
class MoodMorphResult:
    """Result of mood morph operation."""
    output_path: str = ""
    applied_transforms: List[str] = field(default_factory=list)
    duration: float = 0.0
    keyframes: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "applied_transforms": self.applied_transforms,
            "duration": round(self.duration, 3),
            "keyframes": self.keyframes,
        }


# ---------------------------------------------------------------------------
# Audio Analysis
# ---------------------------------------------------------------------------
def _get_audio_info(filepath: str) -> dict:
    """Get audio stream info via ffprobe."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,codec_name,bit_rate,duration",
        "-show_entries", "format=duration,bit_rate",
        "-of", "json", filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}
        return json.loads(result.stdout)
    except Exception as e:
        logger.warning("ffprobe failed for %s: %s", filepath, e)
        return {}


def _estimate_energy_via_volumedetect(filepath: str) -> Tuple[float, float]:
    """Estimate audio energy and spectral centroid using ffmpeg volumedetect.

    Returns (energy_0_to_1, estimated_spectral_centroid_hz).
    """
    cmd = [
        get_ffmpeg_path(), "-i", filepath,
        "-af", "volumedetect",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stderr = result.stderr
    except Exception:
        return 0.5, 2000.0

    mean_volume = -20.0
    peak_volume = -1.0
    for line in stderr.split("\n"):
        if "mean_volume:" in line:
            try:
                mean_volume = float(line.split("mean_volume:")[1].strip().split(" ")[0])
            except (ValueError, IndexError):
                pass
        elif "max_volume:" in line:
            try:
                peak_volume = float(line.split("max_volume:")[1].strip().split(" ")[0])
            except (ValueError, IndexError):
                pass

    # Normalize energy: -60dB = 0.0, 0dB = 1.0
    energy = max(0.0, min(1.0, (mean_volume + 60.0) / 60.0))
    # Refine with peak volume if available
    peak_energy = max(0.0, min(1.0, (peak_volume + 60.0) / 60.0))
    energy = max(energy, peak_energy * 0.5)

    # Rough spectral centroid estimate from energy level
    # Higher energy content often correlates with higher spectral centroid
    centroid = 1000.0 + energy * 4000.0

    return energy, centroid


def _estimate_tempo_via_onset(filepath: str) -> float:
    """Estimate tempo using FFmpeg onset detection.

    Counts transient peaks in a short segment to estimate BPM.
    """
    # Extract first 30s for analysis
    cmd = [
        get_ffmpeg_path(), "-i", filepath,
        "-t", "30",
        "-af", "silencedetect=noise=-30dB:d=0.1",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stderr = result.stderr
    except Exception:
        return 120.0

    # Count silence_end events as rough onset approximation
    onsets = []
    for line in stderr.split("\n"):
        if "silence_end:" in line:
            try:
                ts = float(line.split("silence_end:")[1].strip().split(" ")[0])
                onsets.append(ts)
            except (ValueError, IndexError):
                pass

    if len(onsets) < 2:
        return 120.0

    # Calculate intervals between onsets
    intervals = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
    if not intervals:
        return 120.0

    avg_interval = sum(intervals) / len(intervals)
    if avg_interval <= 0:
        return 120.0

    bpm = 60.0 / avg_interval
    # Clamp to reasonable range
    return max(40.0, min(300.0, bpm))


def analyze_audio_properties(
    filepath: str,
    on_progress: Optional[Callable] = None,
) -> AudioProperties:
    """Analyze audio file for tempo, energy, and spectral characteristics.

    Args:
        filepath: Path to audio file.
        on_progress: Progress callback (int percentage).

    Returns:
        AudioProperties with analysis results.
    """
    if on_progress:
        on_progress(5)

    props = AudioProperties()

    # Get basic info from ffprobe
    info = _get_audio_info(filepath)
    streams = info.get("streams", [])
    fmt = info.get("format", {})

    if streams:
        s = streams[0]
        props.sample_rate = int(s.get("sample_rate", 44100))
        props.channels = int(s.get("channels", 2))
        props.codec = s.get("codec_name", "")
        props.bitrate = int(s.get("bit_rate", 0) or fmt.get("bit_rate", 0) or 0)
        dur = float(s.get("duration", 0) or fmt.get("duration", 0) or 0)
        props.duration = dur
    elif fmt:
        props.duration = float(fmt.get("duration", 0))
        props.bitrate = int(fmt.get("bit_rate", 0) or 0)

    if on_progress:
        on_progress(30)

    # Estimate energy
    energy, centroid = _estimate_energy_via_volumedetect(filepath)
    props.estimated_energy = energy
    props.spectral_centroid = centroid

    if on_progress:
        on_progress(60)

    # Estimate tempo
    props.estimated_tempo = _estimate_tempo_via_onset(filepath)

    if on_progress:
        on_progress(90)

    logger.info(
        "Audio analysis: tempo=%.1f BPM, energy=%.2f, centroid=%.0f Hz",
        props.estimated_tempo, props.estimated_energy, props.spectral_centroid,
    )
    return props


# ---------------------------------------------------------------------------
# Filter Chain Builder
# ---------------------------------------------------------------------------
def _build_filter_chain(
    mood: str,
    intensity: float = 1.0,
) -> Tuple[List[str], float]:
    """Build FFmpeg audio filter chain for a mood transform.

    Returns (filter_parts, tempo_factor).
    """
    transform = MOOD_TRANSFORMS.get(mood)
    if transform is None:
        logger.warning("Unknown mood '%s', using brighten", mood)
        transform = MOOD_TRANSFORMS["brighten"]

    filters = []
    intensity = max(0.0, min(1.0, intensity))

    # EQ: treble and bass adjustments
    treble = transform["eq_treble"] * intensity
    bass = transform["eq_bass"] * intensity

    if abs(treble) > 0.1 or abs(bass) > 0.1:
        eq_parts = []
        if abs(treble) > 0.1:
            eq_parts.append(f"treble=g={treble:.1f}:f=3000")
        if abs(bass) > 0.1:
            eq_parts.append(f"bass=g={bass:.1f}:f=100")
        filters.extend(eq_parts)

    # Compressor / Compand
    if transform.get("compress") and intensity > 0.3:
        compand = transform.get("compand", "0.3|0.3:1|1:-90/-60|-60/-40|-40/-30|-20/-20:6:0:-90:0.2")
        filters.append(f"compand={compand}")

    # Tempo factor (interpolated toward 1.0 by intensity)
    base_tempo = transform["tempo_factor"]
    tempo_factor = 1.0 + (base_tempo - 1.0) * intensity

    return filters, tempo_factor


def _build_atempo_chain(factor: float) -> List[str]:
    """Build chained atempo filters for factors outside 0.5-2.0 range.

    FFmpeg atempo only accepts 0.5-100.0, but best in 0.5-2.0.
    """
    if abs(factor - 1.0) < 0.001:
        return []

    parts = []
    remaining = max(0.5, min(100.0, factor))

    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5

    if abs(remaining - 1.0) > 0.001:
        parts.append(f"atempo={remaining:.4f}")

    return parts


# ---------------------------------------------------------------------------
# Mood Morph Application
# ---------------------------------------------------------------------------
def apply_mood_morph(
    input_path: str,
    mood: str = "brighten",
    intensity: float = 1.0,
    output_path_val: str = "",
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> MoodMorphResult:
    """Apply a mood transformation to an audio file.

    Args:
        input_path: Path to input audio/video file.
        mood: Transform name from MOOD_TRANSFORMS.
        intensity: Transform intensity 0.0-1.0.
        output_path_val: Explicit output path (optional).
        output_dir: Output directory (optional).
        on_progress: Progress callback (int percentage).

    Returns:
        MoodMorphResult with output path and applied transforms.
    """
    if on_progress:
        on_progress(5)

    if mood not in MOOD_TRANSFORMS:
        logger.warning("Unknown mood '%s', falling back to 'brighten'", mood)
        mood = "brighten"

    intensity = max(0.0, min(1.0, intensity))

    # Analyze input duration
    info = _get_audio_info(input_path)
    duration = 0.0
    for s in info.get("streams", []):
        duration = float(s.get("duration", 0))
        if duration > 0:
            break
    if duration <= 0:
        duration = float(info.get("format", {}).get("duration", 0))

    if on_progress:
        on_progress(15)

    # Build filter chain
    filters, tempo_factor = _build_filter_chain(mood, intensity)
    tempo_filters = _build_atempo_chain(tempo_factor)
    all_filters = filters + tempo_filters

    if on_progress:
        on_progress(25)

    # Determine output path
    if output_path_val:
        out = output_path_val
    elif output_dir:
        base = os.path.splitext(os.path.basename(input_path))[0]
        out = os.path.join(output_dir, f"{base}_mood_{mood}.wav")
    else:
        out = output_path(input_path, f"mood_{mood}")
        # Ensure audio output extension
        if not out.lower().endswith((".wav", ".mp3", ".flac", ".aac", ".m4a")):
            out = os.path.splitext(out)[0] + ".wav"

    # Build FFmpeg command
    cmd = [get_ffmpeg_path(), "-y", "-i", input_path]

    if all_filters:
        cmd.extend(["-af", ",".join(all_filters)])

    cmd.extend(["-vn", out])

    if on_progress:
        on_progress(35)

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(90)

    # Calculate output duration (adjusted by tempo factor)
    out_duration = duration / tempo_factor if tempo_factor > 0 else duration

    result = MoodMorphResult(
        output_path=out,
        applied_transforms=[mood],
        duration=out_duration,
        keyframes=[{"time": 0.0, "mood": mood, "intensity": intensity}],
    )
    logger.info("Mood morph complete: %s (intensity=%.2f) -> %s", mood, intensity, out)
    return result


# ---------------------------------------------------------------------------
# Keyframed Mood Morph
# ---------------------------------------------------------------------------
def _interpolate_keyframes(
    keyframes: List[MoodKeyframe],
    total_duration: float,
    segment_duration: float = 5.0,
) -> List[Tuple[float, float, str, float]]:
    """Interpolate keyframes into segments.

    Returns list of (start_time, end_time, mood, intensity) tuples.
    """
    if not keyframes:
        return [(0.0, total_duration, "brighten", 1.0)]

    # Sort by time
    kfs = sorted(keyframes, key=lambda k: k.time)
    segments = []
    t = 0.0

    while t < total_duration:
        seg_end = min(t + segment_duration, total_duration)

        # Find surrounding keyframes
        prev_kf = kfs[0]
        next_kf = kfs[-1]
        for i, kf in enumerate(kfs):
            if kf.time <= t:
                prev_kf = kf
            if kf.time >= t and i > 0:
                next_kf = kf
                break

        # Linear interpolation of intensity
        if prev_kf.time == next_kf.time or prev_kf is next_kf:
            mood = prev_kf.mood
            intensity = prev_kf.intensity
        else:
            blend = (t - prev_kf.time) / (next_kf.time - prev_kf.time)
            blend = max(0.0, min(1.0, blend))
            intensity = prev_kf.intensity + blend * (next_kf.intensity - prev_kf.intensity)
            # Use the mood of whichever keyframe is closer
            mood = prev_kf.mood if blend < 0.5 else next_kf.mood

        segments.append((t, seg_end, mood, intensity))
        t = seg_end

    return segments


def apply_keyframed_morph(
    input_path: str,
    keyframes: List[MoodKeyframe],
    segment_duration: float = 5.0,
    output_path_val: str = "",
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> MoodMorphResult:
    """Apply time-varying mood curve defined by keyframes.

    Splits audio into segments, applies per-segment mood transforms,
    then concatenates.

    Args:
        input_path: Path to input audio file.
        keyframes: List of MoodKeyframe defining the mood curve.
        segment_duration: Duration of each processing segment in seconds.
        output_path_val: Explicit output path.
        output_dir: Output directory.
        on_progress: Progress callback (int percentage).

    Returns:
        MoodMorphResult with output path and applied transforms.
    """
    if on_progress:
        on_progress(5)

    # Get duration
    info = _get_audio_info(input_path)
    duration = 0.0
    for s in info.get("streams", []):
        duration = float(s.get("duration", 0))
        if duration > 0:
            break
    if duration <= 0:
        duration = float(info.get("format", {}).get("duration", 0))

    if duration <= 0:
        logger.warning("Could not determine duration for %s", input_path)
        return MoodMorphResult()

    if on_progress:
        on_progress(10)

    # For simple single-mood cases, use direct application
    unique_moods = set(kf.mood for kf in keyframes)
    if len(keyframes) == 1 or (len(unique_moods) == 1 and
            all(abs(kf.intensity - keyframes[0].intensity) < 0.05 for kf in keyframes)):
        return apply_mood_morph(
            input_path, mood=keyframes[0].mood, intensity=keyframes[0].intensity,
            output_path_val=output_path_val, output_dir=output_dir,
            on_progress=on_progress,
        )

    # Interpolate keyframes into segments
    segments = _interpolate_keyframes(keyframes, duration, segment_duration)
    temp_parts = []
    applied_moods = set()

    if on_progress:
        on_progress(15)

    try:
        for idx, (start, end, mood, intensity) in enumerate(segments):
            if on_progress:
                pct = 15 + int(((idx + 1) / len(segments)) * 65)
                on_progress(pct)

            seg_dur = end - start
            if seg_dur <= 0:
                continue

            # Extract segment
            fd, seg_path = tempfile.mkstemp(suffix=f"_seg{idx}.wav")
            os.close(fd)

            extract_cmd = [
                get_ffmpeg_path(), "-y", "-i", input_path,
                "-ss", f"{start:.3f}", "-t", f"{seg_dur:.3f}",
                "-vn", seg_path,
            ]
            try:
                run_ffmpeg(extract_cmd)
            except RuntimeError as e:
                logger.warning("Segment extraction failed at %.1f: %s", start, e)
                try:
                    os.unlink(seg_path)
                except OSError:
                    pass
                continue

            # Apply mood to segment
            filters, tempo_factor = _build_filter_chain(mood, intensity)
            tempo_filters = _build_atempo_chain(tempo_factor)
            all_filters = filters + tempo_filters

            fd, morphed_path = tempfile.mkstemp(suffix=f"_morphed{idx}.wav")
            os.close(fd)

            morph_cmd = [get_ffmpeg_path(), "-y", "-i", seg_path]
            if all_filters:
                morph_cmd.extend(["-af", ",".join(all_filters)])
            morph_cmd.extend(["-vn", morphed_path])

            try:
                run_ffmpeg(morph_cmd)
                temp_parts.append(morphed_path)
                applied_moods.add(mood)
            except RuntimeError as e:
                logger.warning("Mood morph failed for segment %d: %s", idx, e)
                # Use original segment as fallback
                temp_parts.append(seg_path)
                continue
            finally:
                try:
                    os.unlink(seg_path)
                except OSError:
                    pass

        if not temp_parts:
            return MoodMorphResult()

        if on_progress:
            on_progress(82)

        # Concatenate segments
        if output_path_val:
            out = output_path_val
        elif output_dir:
            base = os.path.splitext(os.path.basename(input_path))[0]
            out = os.path.join(output_dir, f"{base}_mood_keyframed.wav")
        else:
            out = output_path(input_path, "mood_keyframed")
            if not out.lower().endswith((".wav", ".mp3", ".flac")):
                out = os.path.splitext(out)[0] + ".wav"

        # Create concat list file
        fd, list_path = tempfile.mkstemp(suffix="_concat.txt")
        os.close(fd)
        try:
            with open(list_path, "w", encoding="utf-8") as f:
                for p in temp_parts:
                    safe_p = p.replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{safe_p}'\n")

            concat_cmd = [
                get_ffmpeg_path(), "-y", "-f", "concat", "-safe", "0",
                "-i", list_path, "-c", "copy", out,
            ]
            run_ffmpeg(concat_cmd)
        finally:
            try:
                os.unlink(list_path)
            except OSError:
                pass

        if on_progress:
            on_progress(95)

        result = MoodMorphResult(
            output_path=out,
            applied_transforms=sorted(applied_moods),
            duration=duration,
            keyframes=[kf.to_dict() for kf in keyframes],
        )
        logger.info("Keyframed morph complete: %d segments, moods=%s -> %s",
                     len(segments), sorted(applied_moods), out)
        return result

    finally:
        # Clean up temp files
        for p in temp_parts:
            try:
                os.unlink(p)
            except OSError:
                pass


def list_mood_transforms() -> List[Dict]:
    """Return available mood transforms with descriptions."""
    return [
        {"name": name, "description": info["description"]}
        for name, info in MOOD_TRANSFORMS.items()
    ]
