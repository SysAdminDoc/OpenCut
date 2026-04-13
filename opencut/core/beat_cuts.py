"""
OpenCut Beat-Synced Auto-Cuts Module

Generate beat-synchronized video cuts from music:
- Detect beats via FFmpeg audio energy analysis
- Assign clips to beat intervals (round-robin or random)
- Assemble clips trimmed to beat intervals with music track

All via FFmpeg - librosa optional for higher quality beat detection.
"""

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


@dataclass
class BeatMarker:
    """A detected beat position."""
    time: float
    strength: float = 1.0
    bar_position: int = 0  # 0-3 for 4/4 time


@dataclass
class CutEntry:
    """A single cut in the beat-synced sequence."""
    clip_path: str
    start: float
    duration: float
    beat_time: float


@dataclass
class BeatCutResult:
    """Result of beat-synced cut list generation."""
    cuts: List[CutEntry] = field(default_factory=list)
    beats: List[BeatMarker] = field(default_factory=list)
    total_duration: float = 0.0
    density: str = "every_beat"
    bpm: float = 0.0


# ---------------------------------------------------------------------------
# Audio Analysis
# ---------------------------------------------------------------------------

def _get_audio_duration(filepath: str) -> float:
    """Get audio duration via ffprobe."""
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json", filepath,
    ]
    import subprocess as _sp
    result = _sp.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        return 0.0
    try:
        data = json.loads(result.stdout.decode())
        return float(data.get("format", {}).get("duration", 0))
    except (json.JSONDecodeError, ValueError):
        return 0.0


def _detect_beats_energy(music_path: str, on_progress: Optional[Callable] = None) -> List[BeatMarker]:
    """Detect beats using FFmpeg audio energy analysis.

    Extracts audio energy levels at high resolution and finds peaks
    that correspond to rhythmic beats.

    Args:
        music_path: Path to audio/music file.
        on_progress: Progress callback.

    Returns:
        List of BeatMarker with detected beat positions.
    """
    if on_progress:
        on_progress(10, "Analyzing audio energy...")

    # Extract audio volume levels using FFmpeg volumedetect + astats
    tmp_dir = tempfile.mkdtemp(prefix="opencut_beats_")
    energy_file = os.path.join(tmp_dir, "energy.txt")

    try:
        # Use FFmpeg to extract audio energy at ~43fps (1024 samples at 44100Hz)
        cmd = [
            get_ffmpeg_path(), "-i", music_path,
            "-af", "astats=metadata=1:reset=1024,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=" + energy_file.replace("\\", "/"),
            "-f", "null", "-",
        ]
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(40, "Parsing energy data...")

        # Parse energy data
        energy_samples = []
        frame_time = 1024.0 / 44100.0  # time per analysis frame

        if os.path.isfile(energy_file):
            with open(energy_file, "r") as f:
                for line in f:
                    line = line.strip()
                    # Format: frame:N    pts_time:T
                    # lavfi.astats.Overall.RMS_level=-XX.XX
                    if "RMS_level=" in line:
                        match = re.search(r"RMS_level=(-?[\d.]+)", line)
                        if match:
                            try:
                                db = float(match.group(1))
                                # Convert dB to linear energy (higher = louder)
                                energy = max(0, db + 60)  # shift so -60dB = 0
                                energy_samples.append(energy)
                            except ValueError:
                                pass

        if not energy_samples:
            # Fallback: generate beats at regular intervals
            duration = _get_audio_duration(music_path)
            if duration <= 0:
                return []
            # Assume 120 BPM
            beat_interval = 0.5
            beats = []
            t = beat_interval
            while t < duration:
                beats.append(BeatMarker(time=round(t, 3), strength=0.8, bar_position=int((t / beat_interval) % 4)))
                t += beat_interval
            return beats

        if on_progress:
            on_progress(60, "Finding beat peaks...")

        # Find energy peaks (beats)
        # Simple peak detection: local maxima above threshold
        if not energy_samples:
            return []

        avg_energy = sum(energy_samples) / len(energy_samples)
        threshold = avg_energy * 1.3  # beats are louder than average

        # Minimum time between beats (fastest = 200 BPM = 0.3s)
        min_gap_samples = int(0.3 / frame_time)

        beats = []
        last_beat_idx = -min_gap_samples

        for i in range(1, len(energy_samples) - 1):
            if energy_samples[i] > threshold:
                if energy_samples[i] >= energy_samples[i - 1] and energy_samples[i] >= energy_samples[i + 1]:
                    if i - last_beat_idx >= min_gap_samples:
                        t = i * frame_time
                        strength = min(1.0, energy_samples[i] / 60.0)
                        beats.append(BeatMarker(
                            time=round(t, 3),
                            strength=round(strength, 3),
                            bar_position=len(beats) % 4,
                        ))
                        last_beat_idx = i

        if on_progress:
            on_progress(80, f"Detected {len(beats)} beats")

        return beats

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass


def _detect_beats_librosa(music_path: str, on_progress: Optional[Callable] = None) -> List[BeatMarker]:
    """Detect beats using librosa for higher quality results."""
    try:
        import librosa
        import numpy as np  # noqa: F401
    except ImportError:
        logger.debug("librosa not available, falling back to FFmpeg energy detection")
        return _detect_beats_energy(music_path, on_progress)

    if on_progress:
        on_progress(10, "Loading audio with librosa...")

    y, sr = librosa.load(music_path, sr=22050, mono=True)

    if on_progress:
        on_progress(30, "Detecting beats...")

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if on_progress:
        on_progress(70, f"Found {len(beat_times)} beats at ~{float(tempo):.0f} BPM")

    # Get onset strengths for confidence
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    beats = []
    for i, t in enumerate(beat_times):
        frame_idx = beat_frames[i] if i < len(beat_frames) else 0
        strength = float(onset_env[frame_idx]) if frame_idx < len(onset_env) else 0.5
        strength = min(1.0, strength / max(onset_env.max(), 1e-6))
        beats.append(BeatMarker(
            time=round(float(t), 3),
            strength=round(strength, 3),
            bar_position=i % 4,
        ))

    return beats


# ---------------------------------------------------------------------------
# Cut List Generation
# ---------------------------------------------------------------------------

def generate_beat_cut_list(
    music_path: str,
    clip_paths: List[str],
    density: str = "every_beat",
    assignment: str = "round_robin",
    on_progress: Optional[Callable] = None,
) -> BeatCutResult:
    """Generate a beat-synchronized cut list from music and video clips.

    Args:
        music_path: Path to music/audio file.
        clip_paths: List of video clip paths to cut between.
        density: Beat density: "every_beat", "every_2nd", "every_bar" (every 4th).
        assignment: Clip assignment: "round_robin" or "random".
        on_progress: Progress callback(pct, msg).

    Returns:
        BeatCutResult with cuts list, beats, and metadata.
    """
    if not clip_paths:
        raise ValueError("At least one clip path is required")

    if on_progress:
        on_progress(5, "Detecting beats in music...")

    # Try librosa first, fall back to FFmpeg
    try:
        import librosa  # noqa: F401
        beats = _detect_beats_librosa(music_path, on_progress)
    except ImportError:
        beats = _detect_beats_energy(music_path, on_progress)

    if not beats:
        raise RuntimeError("No beats detected in music track")

    if on_progress:
        on_progress(80, "Building cut list...")

    # Filter beats by density
    density = density if density in ("every_beat", "every_2nd", "every_bar") else "every_beat"
    if density == "every_2nd":
        filtered = [b for i, b in enumerate(beats) if i % 2 == 0]
    elif density == "every_bar":
        filtered = [b for b in beats if b.bar_position == 0]
    else:
        filtered = list(beats)

    if not filtered:
        filtered = beats[:1]

    # Get clip durations
    clip_durations = {}
    for cp in clip_paths:
        info = get_video_info(cp)
        clip_durations[cp] = info["duration"]

    # Calculate BPM from beat intervals
    if len(filtered) >= 2:
        intervals = [filtered[i + 1].time - filtered[i].time for i in range(len(filtered) - 1)]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0.5
        bpm = 60.0 / avg_interval if avg_interval > 0 else 120.0
    else:
        bpm = 120.0

    # Assign clips to beat intervals
    import random
    cuts = []
    clip_index = 0

    for i in range(len(filtered)):
        beat = filtered[i]
        # Duration until next beat (or end of music)
        if i + 1 < len(filtered):
            beat_duration = filtered[i + 1].time - beat.time
        else:
            music_dur = _get_audio_duration(music_path)
            beat_duration = max(0.5, music_dur - beat.time)

        # Select clip
        if assignment == "random":
            cp = random.choice(clip_paths)
        else:
            cp = clip_paths[clip_index % len(clip_paths)]
            clip_index += 1

        # Select start position within clip
        clip_dur = clip_durations.get(cp, 10.0)
        max_start = max(0, clip_dur - beat_duration)
        if assignment == "random" and max_start > 0:
            start = random.uniform(0, max_start)
        else:
            # Cycle through clip evenly
            start = (i * beat_duration) % max(0.1, clip_dur - beat_duration) if clip_dur > beat_duration else 0

        cuts.append(CutEntry(
            clip_path=cp,
            start=round(start, 3),
            duration=round(beat_duration, 3),
            beat_time=beat.time,
        ))

    if on_progress:
        on_progress(100, f"Generated {len(cuts)} cuts at {bpm:.0f} BPM")

    total_dur = sum(c.duration for c in cuts)
    return BeatCutResult(
        cuts=cuts,
        beats=beats,
        total_duration=round(total_dur, 3),
        density=density,
        bpm=round(bpm, 1),
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_beat_synced(
    music_path: str,
    cut_list: List[dict],
    output_path_str: Optional[str] = None,
    transition: str = "cut",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Assemble beat-synced clips into a final video with music.

    Args:
        music_path: Path to music track.
        cut_list: List of cut dicts [{clip_path, start, duration, beat_time}].
        output_path_str: Output file path. Auto-generated if None.
        transition: Transition type: "cut" (hard cut) or "crossfade".
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, duration, clip_count.
    """
    if not cut_list:
        raise ValueError("cut_list must not be empty")

    if on_progress:
        on_progress(5, "Preparing beat-synced assembly...")

    if output_path_str is None:
        output_path_str = output_path(music_path, "beat_synced", ".mp4")
        # Ensure .mp4 extension
        if not output_path_str.endswith(".mp4"):
            output_path_str = os.path.splitext(output_path_str)[0] + ".mp4"

    # Normalize cut_list entries
    cuts = []
    for entry in cut_list:
        if isinstance(entry, dict):
            cuts.append(CutEntry(
                clip_path=entry.get("clip_path", ""),
                start=float(entry.get("start", 0)),
                duration=float(entry.get("duration", 1)),
                beat_time=float(entry.get("beat_time", 0)),
            ))
        elif isinstance(entry, CutEntry):
            cuts.append(entry)

    # Get output resolution from first clip
    first_info = get_video_info(cuts[0].clip_path)
    out_w, out_h = first_info["width"], first_info["height"]
    first_info["fps"]

    tmp_dir = tempfile.mkdtemp(prefix="opencut_beatasm_")
    try:
        if on_progress:
            on_progress(10, f"Trimming {len(cuts)} clips...")

        # Trim each clip segment
        segment_files = []
        for idx, cut in enumerate(cuts):
            seg_path = os.path.join(tmp_dir, f"seg_{idx:04d}.mp4")
            cmd = [
                get_ffmpeg_path(), "-i", cut.clip_path,
                "-ss", str(cut.start), "-t", str(cut.duration),
                "-vf", f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-an",  # strip original audio, we'll use music track
                "-y", seg_path,
            ]
            run_ffmpeg(cmd)
            segment_files.append(seg_path)

            if on_progress:
                pct = 10 + int(60 * (idx + 1) / len(cuts))
                on_progress(pct, f"Trimmed clip {idx + 1}/{len(cuts)}")

        if on_progress:
            on_progress(75, "Concatenating clips...")

        # Write concat list
        concat_file = os.path.join(tmp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for sp in segment_files:
                f.write(f"file '{sp}'\n")

        # Concat video segments
        concat_video = os.path.join(tmp_dir, "concat_video.mp4")
        cmd = [
            get_ffmpeg_path(), "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an", "-y", concat_video,
        ]
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(85, "Merging with music track...")

        # Merge concatenated video with music audio
        total_video_dur = sum(c.duration for c in cuts)
        cmd = [
            get_ffmpeg_path(),
            "-i", concat_video,
            "-i", music_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(total_video_dur),
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            "-y", output_path_str,
        ]
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(100, "Beat-synced assembly complete")

        return {
            "output_path": output_path_str,
            "duration": round(total_video_dur, 3),
            "clip_count": len(cuts),
            "bpm": 0,  # caller can pass from BeatCutResult
        }

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass
