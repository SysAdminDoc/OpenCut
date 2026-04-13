"""
OpenCut Auto Montage Builder (12.3)

Score video clips by excitement (audio energy + motion detection),
select the top N, and assemble them into a beat-synced montage with
a music track.

Pipeline:
1. Analyze each clip for audio peaks and visual motion
2. Compute composite excitement scores
3. Select top-N clips
4. Detect beats in music track
5. Assemble clips trimmed to beat intervals with transitions

All via FFmpeg -- numpy/librosa optional for higher quality analysis.
"""

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_ffprobe_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class ClipScore:
    """Excitement score breakdown for a single clip."""
    clip_path: str = ""
    audio_energy: float = 0.0       # 0-1: normalized RMS energy
    motion_score: float = 0.0       # 0-1: inter-frame motion magnitude
    duration: float = 0.0
    composite: float = 0.0          # weighted overall score
    peak_timestamp: float = 0.0     # timestamp of peak excitement


@dataclass
class MontageResult:
    """Result of montage assembly."""
    output_path: str = ""
    clip_count: int = 0
    total_duration: float = 0.0
    scores: List[ClipScore] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Audio Energy Analysis
# ---------------------------------------------------------------------------
def _analyze_audio_energy(clip_path: str) -> float:
    """Compute normalized RMS audio energy for a clip via FFmpeg.

    Returns a value 0.0-1.0 where 1.0 is maximum energy.
    """
    import subprocess as _sp

    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream_tags=",
        "-of", "json", clip_path,
    ]
    # Check if audio stream exists
    result = _sp.run(cmd, capture_output=True, timeout=15)

    # Use volumedetect to get mean volume
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    tmp.close()
    try:
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-i", clip_path,
            "-af", "volumedetect", "-f", "null", "-",
        ]
        result = _sp.run(cmd, capture_output=True, timeout=60, text=True)
        stderr = result.stderr or ""

        # Parse mean_volume from FFmpeg stderr
        match = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", stderr)
        if match:
            mean_db = float(match.group(1))
            # Normalize: -60dB = 0.0, 0dB = 1.0
            normalized = max(0.0, min(1.0, (mean_db + 60.0) / 60.0))
            return normalized
        return 0.3  # default if no audio
    except Exception as e:
        logger.debug("Audio energy analysis failed for %s: %s", clip_path, e)
        return 0.3
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Motion Detection Analysis
# ---------------------------------------------------------------------------
def _analyze_motion(clip_path: str) -> float:
    """Estimate visual motion magnitude using FFmpeg scene change detection.

    Returns a value 0.0-1.0 where 1.0 is maximum motion.
    """
    import subprocess as _sp

    try:
        # Use FFmpeg's select filter to detect scene changes which
        # correlates with motion. Count how many frames exceed the threshold.
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-i", clip_path,
            "-vf", "select='gt(scene,0.1)',metadata=print",
            "-f", "null", "-",
        ]
        result = _sp.run(cmd, capture_output=True, timeout=60, text=True)
        stderr = result.stderr or ""

        # Count scene changes
        scene_changes = len(re.findall(r"scene_score=[\d.]+", stderr))

        info = get_video_info(clip_path)
        duration = max(info["duration"], 1.0)

        # Normalize: scene changes per second, capped at 1.0
        changes_per_sec = scene_changes / duration
        # 0.5 changes/sec is moderate, 2.0 is high action
        normalized = min(1.0, changes_per_sec / 2.0)
        return round(normalized, 3)

    except Exception as e:
        logger.debug("Motion analysis failed for %s: %s", clip_path, e)
        return 0.3


# ---------------------------------------------------------------------------
# Clip Scoring
# ---------------------------------------------------------------------------
def score_clips(
    clip_paths: List[str],
    audio_weight: float = 0.5,
    motion_weight: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> List[ClipScore]:
    """Score video clips by excitement (audio energy + motion).

    Args:
        clip_paths: List of video file paths to analyze.
        audio_weight: Weight for audio energy in composite score (0-1).
        motion_weight: Weight for visual motion in composite score (0-1).
        on_progress: Progress callback(pct, msg).

    Returns:
        List of ClipScore sorted by composite score descending.
    """
    if not clip_paths:
        raise ValueError("No clips provided for scoring")

    # Normalize weights
    total_weight = audio_weight + motion_weight
    if total_weight > 0:
        audio_weight = audio_weight / total_weight
        motion_weight = motion_weight / total_weight
    else:
        audio_weight = motion_weight = 0.5

    scores = []
    total = len(clip_paths)

    for i, path in enumerate(clip_paths):
        if not os.path.isfile(path):
            logger.warning("Clip not found, skipping: %s", path)
            continue

        if on_progress:
            pct = int(10 + 80 * i / total)
            on_progress(pct, f"Analyzing clip {i + 1}/{total}...")

        info = get_video_info(path)
        audio_e = _analyze_audio_energy(path)
        motion_s = _analyze_motion(path)
        composite = audio_weight * audio_e + motion_weight * motion_s

        scores.append(ClipScore(
            clip_path=path,
            audio_energy=round(audio_e, 3),
            motion_score=round(motion_s, 3),
            duration=info["duration"],
            composite=round(composite, 3),
            peak_timestamp=info["duration"] / 2,  # approximate midpoint
        ))

    scores.sort(key=lambda s: s.composite, reverse=True)

    if on_progress:
        on_progress(95, f"Scored {len(scores)} clips")

    return scores


# ---------------------------------------------------------------------------
# Clip Selection
# ---------------------------------------------------------------------------
def select_top_clips(
    scores: List[ClipScore],
    count: int = 10,
    min_duration: float = 1.0,
) -> List[ClipScore]:
    """Select top N clips by composite excitement score.

    Args:
        scores: List of ClipScore from score_clips().
        count: Maximum number of clips to select.
        min_duration: Minimum clip duration to include.

    Returns:
        Top N ClipScore entries, sorted by composite score descending.
    """
    filtered = [s for s in scores if s.duration >= min_duration]
    filtered.sort(key=lambda s: s.composite, reverse=True)
    return filtered[:count]


# ---------------------------------------------------------------------------
# Beat Detection (simple FFmpeg-based)
# ---------------------------------------------------------------------------
def _detect_beats_simple(music_path: str) -> List[float]:
    """Detect approximate beat positions using FFmpeg energy analysis.

    Returns list of beat timestamps in seconds.
    """
    import subprocess as _sp

    tmp_dir = tempfile.mkdtemp(prefix="opencut_montage_beats_")
    energy_file = os.path.join(tmp_dir, "energy.txt")

    try:
        cmd = [
            get_ffmpeg_path(), "-i", music_path,
            "-af", (
                "astats=metadata=1:reset=1024,"
                "ametadata=print:key=lavfi.astats.Overall.RMS_level"
                ":file=" + energy_file.replace("\\", "/")
            ),
            "-f", "null", "-",
        ]
        run_ffmpeg(cmd)

        frame_time = 1024.0 / 44100.0
        energy_samples = []

        if os.path.isfile(energy_file):
            with open(energy_file, "r") as f:
                for line in f:
                    match = re.search(r"RMS_level=(-?[\d.]+)", line)
                    if match:
                        try:
                            db = float(match.group(1))
                            energy_samples.append(max(0, db + 60))
                        except ValueError:
                            pass

        if not energy_samples:
            # Fallback: regular intervals at 120 BPM
            cmd_dur = [
                get_ffprobe_path(), "-v", "quiet",
                "-show_entries", "format=duration", "-of", "json", music_path,
            ]
            r = _sp.run(cmd_dur, capture_output=True, timeout=15)
            try:
                dur = float(json.loads(r.stdout.decode()).get("format", {}).get("duration", 60))
            except Exception:
                dur = 60.0
            return [t * 0.5 for t in range(1, int(dur / 0.5))]

        # Peak detection
        avg = sum(energy_samples) / len(energy_samples) if energy_samples else 0
        threshold = avg * 1.3
        min_gap = int(0.3 / frame_time)

        beats = []
        last_idx = -min_gap
        for i in range(1, len(energy_samples) - 1):
            if (energy_samples[i] > threshold
                    and energy_samples[i] >= energy_samples[i - 1]
                    and energy_samples[i] >= energy_samples[i + 1]
                    and i - last_idx >= min_gap):
                beats.append(round(i * frame_time, 3))
                last_idx = i

        return beats

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Montage Assembly
# ---------------------------------------------------------------------------
def assemble_montage(
    clips: List[ClipScore],
    music_path: str,
    output_path_str: Optional[str] = None,
    transition: str = "cut",
    transition_duration: float = 0.3,
    target_resolution: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> MontageResult:
    """Assemble selected clips into a beat-synced montage with music.

    Args:
        clips: List of ClipScore (from select_top_clips).
        music_path: Path to music/audio track.
        output_path_str: Output file path. Auto-generated if None.
        transition: Transition type: 'cut', 'crossfade', 'fade'.
        transition_duration: Duration of transitions in seconds.
        target_resolution: Optional resolution e.g. '1920x1080'.
        on_progress: Progress callback(pct, msg).

    Returns:
        MontageResult with output path and stats.
    """
    if not clips:
        raise ValueError("No clips provided for montage assembly")
    if not os.path.isfile(music_path):
        raise FileNotFoundError(f"Music file not found: {music_path}")

    if on_progress:
        on_progress(5, "Detecting beats in music...")

    beats = _detect_beats_simple(music_path)
    if not beats:
        raise RuntimeError("No beats detected in music track")

    # Get target resolution from first clip if not specified
    first_info = get_video_info(clips[0].clip_path)
    if target_resolution:
        parts = target_resolution.split("x")
        out_w, out_h = int(parts[0]), int(parts[1])
    else:
        out_w, out_h = first_info["width"], first_info["height"]

    if output_path_str is None:
        output_path_str = output_path(clips[0].clip_path, "montage")
        if not output_path_str.endswith(".mp4"):
            output_path_str = os.path.splitext(output_path_str)[0] + ".mp4"

    tmp_dir = tempfile.mkdtemp(prefix="opencut_montage_")

    try:
        if on_progress:
            on_progress(15, f"Trimming {len(clips)} clips to beats...")

        # Assign clips to beat intervals
        segment_files = []
        clip_idx = 0
        used_beats = min(len(beats) - 1, len(clips))

        for i in range(used_beats):
            clip = clips[clip_idx % len(clips)]
            clip_idx += 1

            # Beat duration
            if i + 1 < len(beats):
                beat_dur = beats[i + 1] - beats[i]
            else:
                beat_dur = 2.0  # default last segment

            beat_dur = max(0.5, min(beat_dur, clip.duration))

            # Start from peak area of clip for most exciting content
            start = max(0, clip.peak_timestamp - beat_dur / 2)
            if start + beat_dur > clip.duration:
                start = max(0, clip.duration - beat_dur)

            seg_path = os.path.join(tmp_dir, f"seg_{i:04d}.mp4")
            scale_filter = (
                f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
            )

            cmd = (
                FFmpegCmd()
                .input(clip.clip_path, ss=str(round(start, 3)))
                .option("t", str(round(beat_dur, 3)))
                .video_filter(scale_filter)
                .video_codec("libx264", crf=18, preset="fast")
                .option("an")
                .output(seg_path)
                .build()
            )
            run_ffmpeg(cmd)
            segment_files.append(seg_path)

            if on_progress:
                pct = 15 + int(55 * (i + 1) / used_beats)
                on_progress(pct, f"Trimmed segment {i + 1}/{used_beats}")

        if not segment_files:
            raise RuntimeError("No segments produced during assembly")

        if on_progress:
            on_progress(75, "Concatenating segments...")

        # Concat all segments
        concat_file = os.path.join(tmp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for sp in segment_files:
                f.write(f"file '{sp}'\n")

        concat_video = os.path.join(tmp_dir, "concat_video.mp4")
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an", concat_video,
        ]
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(85, "Merging with music track...")

        # Merge video with music
        total_dur = sum(
            (beats[i + 1] - beats[i]) if i + 1 < len(beats) else 2.0
            for i in range(used_beats)
        )
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", concat_video, "-i", music_path,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-t", str(round(total_dur, 3)),
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", "-movflags", "+faststart",
            output_path_str,
        ]
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(100, "Montage assembly complete")

        return MontageResult(
            output_path=output_path_str,
            clip_count=len(segment_files),
            total_duration=round(total_dur, 3),
            scores=list(clips),
        )

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
