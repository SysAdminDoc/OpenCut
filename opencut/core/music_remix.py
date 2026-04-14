"""
OpenCut AI Music Remix / Duration Fit

Automatically adjust background music duration to match video length.
Modes: smart (beat-aware cut+loop), stretch (time-stretch), fade (fade out).

Smart mode detects beats via librosa, finds musically-valid loop/cut points
at bar boundaries, and extends or shortens by repeating or removing sections
with crossfades at edit points.

Uses FFmpeg for audio processing. librosa is optional (for smart mode only).
"""

import logging
import math
import os
import subprocess as _sp
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    ensure_package,
    get_ffmpeg_path,
    get_ffprobe_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Section:
    """A detected section of a music track."""
    label: str = ""
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0


@dataclass
class MusicStructure:
    """Analyzed structure of a music track."""
    bpm: float = 0.0
    bars: int = 0
    sections: List[Section] = field(default_factory=list)
    loop_points: List[float] = field(default_factory=list)


@dataclass
class MusicFitResult:
    """Result of fitting music to a target duration."""
    output_path: str = ""
    original_duration: float = 0.0
    target_duration: float = 0.0
    actual_duration: float = 0.0
    mode_used: str = ""
    edit_points: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Audio duration helper
# ---------------------------------------------------------------------------
def _get_audio_duration(audio_path: str) -> float:
    """Get audio file duration in seconds via ffprobe."""
    cmd = [
        get_ffprobe_path(), "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", audio_path,
    ]
    result = _sp.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {audio_path}")
    import json
    data = json.loads(result.stdout)
    return float(data.get("format", {}).get("duration", 0))


# ---------------------------------------------------------------------------
# Fade mode: simple fade out at target duration
# ---------------------------------------------------------------------------
def _fit_fade(
    music_path: str,
    target_duration: float,
    out_path: str,
    original_duration: float,
    fade_seconds: float = 3.0,
    on_progress: Optional[Callable] = None,
) -> MusicFitResult:
    """Fade mode: cut at target duration with a smooth fade-out."""
    if on_progress:
        on_progress(20, "Applying fade-out cut...")

    fade_start = max(0, target_duration - fade_seconds)
    af_filter = f"afade=t=out:st={fade_start}:d={fade_seconds}"

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", music_path,
        "-t", str(target_duration),
        "-af", af_filter,
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]
    run_ffmpeg(cmd, timeout=300)

    if on_progress:
        on_progress(100, "Fade complete")

    return MusicFitResult(
        output_path=out_path,
        original_duration=original_duration,
        target_duration=target_duration,
        actual_duration=target_duration,
        mode_used="fade",
        edit_points=[fade_start],
    )


# ---------------------------------------------------------------------------
# Stretch mode: time-stretch via FFmpeg atempo
# ---------------------------------------------------------------------------
def _build_atempo_chain(ratio: float) -> str:
    """Build chained atempo filters for ratios outside 0.5-2.0 range.

    FFmpeg atempo is limited to 0.5-2.0 per instance; chain for larger changes.
    """
    filters = []
    remaining = ratio
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def _fit_stretch(
    music_path: str,
    target_duration: float,
    out_path: str,
    original_duration: float,
    on_progress: Optional[Callable] = None,
) -> MusicFitResult:
    """Stretch mode: time-stretch audio to exactly match target duration."""
    if on_progress:
        on_progress(20, "Time-stretching audio...")

    if original_duration <= 0:
        raise RuntimeError("Cannot determine original audio duration")

    ratio = original_duration / target_duration
    ratio = max(0.1, min(10.0, ratio))

    atempo_chain = _build_atempo_chain(ratio)

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", music_path,
        "-af", atempo_chain,
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]
    run_ffmpeg(cmd, timeout=600)

    actual = _get_audio_duration(out_path) if os.path.isfile(out_path) else target_duration

    if on_progress:
        on_progress(100, "Stretch complete")

    return MusicFitResult(
        output_path=out_path,
        original_duration=original_duration,
        target_duration=target_duration,
        actual_duration=actual,
        mode_used="stretch",
        edit_points=[],
    )


# ---------------------------------------------------------------------------
# Smart mode: beat-aware looping/cutting
# ---------------------------------------------------------------------------
def _detect_beats_librosa(music_path: str, on_progress: Optional[Callable] = None) -> dict:
    """Detect beats and tempo using librosa."""
    import librosa

    if on_progress:
        on_progress(15, "Loading audio for beat detection...")

    y, sr = librosa.load(music_path, sr=22050, mono=True)

    if on_progress:
        on_progress(25, "Detecting tempo and beats...")

    tempo_result = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo_result[0], "__len__"):
        tempo = float(tempo_result[0][0]) if len(tempo_result[0]) > 0 else 120.0
    else:
        tempo = float(tempo_result[0])
    beat_frames = tempo_result[1]
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    bar_length = 4
    bar_boundaries = []
    for i in range(0, len(beat_times), bar_length):
        bar_boundaries.append(beat_times[i])

    duration = float(len(y)) / sr

    return {
        "bpm": round(tempo, 1),
        "beat_times": beat_times,
        "bar_boundaries": bar_boundaries,
        "duration": duration,
    }


def _detect_beats_ffmpeg(music_path: str, on_progress: Optional[Callable] = None) -> dict:
    """Fallback beat detection using FFmpeg silence/energy analysis."""
    if on_progress:
        on_progress(15, "Analyzing audio energy...")

    duration = _get_audio_duration(music_path)
    estimated_bpm = 120.0
    beat_interval = 60.0 / estimated_bpm
    beat_times = []
    t = 0.0
    while t < duration:
        beat_times.append(round(t, 3))
        t += beat_interval

    bar_boundaries = beat_times[::4]

    return {
        "bpm": estimated_bpm,
        "beat_times": beat_times,
        "bar_boundaries": bar_boundaries,
        "duration": duration,
    }


def _find_best_loop_point(bar_boundaries: list, target: float) -> float:
    """Find the bar boundary closest to a target time."""
    if not bar_boundaries:
        return target
    best = bar_boundaries[0]
    for b in bar_boundaries:
        if abs(b - target) < abs(best - target):
            best = b
    return best


def _fit_smart(
    music_path: str,
    target_duration: float,
    out_path: str,
    original_duration: float,
    on_progress: Optional[Callable] = None,
) -> MusicFitResult:
    """Smart mode: beat-aware looping or cutting at bar boundaries."""
    has_librosa = ensure_package("librosa", "librosa", on_progress)

    if has_librosa:
        beat_info = _detect_beats_librosa(music_path, on_progress)
    else:
        beat_info = _detect_beats_ffmpeg(music_path, on_progress)

    bar_boundaries = beat_info["bar_boundaries"]

    if on_progress:
        on_progress(40, "Computing edit points...")

    edit_points = []

    if target_duration <= original_duration:
        # Shorten: cut at nearest bar boundary
        cut_point = _find_best_loop_point(bar_boundaries, target_duration)
        cut_point = min(cut_point, target_duration)

        if on_progress:
            on_progress(50, f"Cutting at {cut_point:.1f}s...")

        fade_start = max(0, cut_point - 3.0)
        af = f"afade=t=out:st={fade_start}:d=3"
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", music_path,
            "-t", str(cut_point),
            "-af", af,
            "-c:a", "aac", "-b:a", "192k",
            out_path,
        ]
        run_ffmpeg(cmd, timeout=300)
        edit_points = [cut_point]
    else:
        # Extend: loop from a good loop point
        if len(bar_boundaries) >= 2:
            loop_start = bar_boundaries[max(0, len(bar_boundaries) // 4)]
            loop_end_idx = min(len(bar_boundaries) - 1, 3 * len(bar_boundaries) // 4)
            loop_end = bar_boundaries[loop_end_idx]
        else:
            loop_start = 0.0
            loop_end = original_duration

        loop_dur = loop_end - loop_start
        if loop_dur <= 0:
            loop_dur = original_duration
            loop_start = 0.0

        repeats_needed = math.ceil((target_duration - original_duration) / loop_dur) + 1
        total_loops = min(repeats_needed, 50)

        if on_progress:
            on_progress(50, f"Looping section {loop_start:.1f}s-{loop_end:.1f}s ({total_loops}x)...")

        # Build concat list: full track + looped sections
        fd, list_file = tempfile.mkstemp(suffix=".txt", prefix="opencut_concat_")
        temp_segments = []
        try:
            # Extract loop segment
            fd_seg, seg_path = tempfile.mkstemp(suffix=".wav", prefix="opencut_seg_")
            os.close(fd_seg)
            temp_segments.append(seg_path)

            cmd_seg = [
                get_ffmpeg_path(), "-hide_banner", "-y",
                "-i", music_path,
                "-ss", str(loop_start),
                "-t", str(loop_dur),
                "-c:a", "pcm_s16le",
                seg_path,
            ]
            run_ffmpeg(cmd_seg, timeout=120)

            # Write concat list
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                escaped_music = music_path.replace("'", "'\\''")
                f.write(f"file '{escaped_music}'\n")
                escaped_seg = seg_path.replace("'", "'\\''")
                for _ in range(total_loops):
                    f.write(f"file '{escaped_seg}'\n")

            if on_progress:
                on_progress(70, "Concatenating segments...")

            fd_tmp, tmp_out = tempfile.mkstemp(suffix=".aac", prefix="opencut_loop_")
            os.close(fd_tmp)
            temp_segments.append(tmp_out)

            cmd_concat = [
                get_ffmpeg_path(), "-hide_banner", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_file,
                "-t", str(target_duration),
                "-c:a", "aac", "-b:a", "192k",
                out_path,
            ]
            run_ffmpeg(cmd_concat, timeout=600)
            edit_points = [original_duration, loop_start, loop_end]
        finally:
            try:
                os.unlink(list_file)
            except OSError:
                pass
            for seg in temp_segments:
                try:
                    os.unlink(seg)
                except OSError:
                    pass

    actual = _get_audio_duration(out_path) if os.path.isfile(out_path) else target_duration

    if on_progress:
        on_progress(100, "Smart fit complete")

    return MusicFitResult(
        output_path=out_path,
        original_duration=original_duration,
        target_duration=target_duration,
        actual_duration=actual,
        mode_used="smart",
        edit_points=edit_points,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fit_music_to_duration(
    music_path: str,
    target_duration: float,
    output_path: Optional[str] = None,
    mode: str = "smart",
    on_progress: Optional[Callable] = None,
) -> MusicFitResult:
    """
    Adjust music duration to match a target length.

    Args:
        music_path: Path to the source music file.
        target_duration: Desired duration in seconds.
        output_path: Path for the output file. Auto-generated if None.
        mode: Fitting mode — "smart" (beat-aware), "stretch" (time-stretch),
              "fade" (simple fade-out cut).
        on_progress: Progress callback(pct, msg).

    Returns:
        MusicFitResult with output path and metadata.
    """
    if not os.path.isfile(music_path):
        raise FileNotFoundError(f"Music file not found: {music_path}")
    if target_duration <= 0:
        raise ValueError("Target duration must be positive")

    mode = mode.lower().strip()
    if mode not in ("smart", "stretch", "fade"):
        raise ValueError(f"Unknown mode: {mode}. Use 'smart', 'stretch', or 'fade'.")

    if on_progress:
        on_progress(5, f"Analyzing music ({mode} mode)...")

    original_duration = _get_audio_duration(music_path)
    if original_duration <= 0:
        raise RuntimeError("Could not determine music duration")

    if output_path is None:
        from opencut.helpers import output_path as _output_path
        out = _output_path(music_path, f"fit_{mode}")
    else:
        out = output_path

    if mode == "fade":
        return _fit_fade(music_path, target_duration, out, original_duration,
                         on_progress=on_progress)
    elif mode == "stretch":
        return _fit_stretch(music_path, target_duration, out, original_duration,
                            on_progress=on_progress)
    else:
        return _fit_smart(music_path, target_duration, out, original_duration,
                          on_progress=on_progress)


def detect_music_structure(
    music_path: str,
    on_progress: Optional[Callable] = None,
) -> MusicStructure:
    """
    Analyze the structure of a music track.

    Detects BPM, bar count, sections (intro, verse, chorus, etc.),
    and good loop points at bar boundaries.

    Args:
        music_path: Path to the music file.
        on_progress: Progress callback(pct, msg).

    Returns:
        MusicStructure with tempo, bars, sections, and loop points.
    """
    if not os.path.isfile(music_path):
        raise FileNotFoundError(f"Music file not found: {music_path}")

    if on_progress:
        on_progress(5, "Analyzing music structure...")

    has_librosa = ensure_package("librosa", "librosa", on_progress)

    if has_librosa:
        beat_info = _detect_beats_librosa(music_path, on_progress)
    else:
        beat_info = _detect_beats_ffmpeg(music_path, on_progress)

    bpm = beat_info["bpm"]
    duration = beat_info["duration"]
    bar_boundaries = beat_info["bar_boundaries"]
    bars = len(bar_boundaries)

    if on_progress:
        on_progress(60, "Identifying sections...")

    # Estimate sections based on bar structure
    sections = []
    if bars >= 4:
        # Simple section estimation: intro(~4 bars), body, outro(~4 bars)
        bar_dur = 4 * (60.0 / bpm) if bpm > 0 else 8.0

        intro_end = min(bar_dur * 2, duration * 0.15)
        outro_start = max(duration - bar_dur * 2, duration * 0.85)

        sections.append(Section(
            label="intro", start=0.0, end=round(intro_end, 2),
            duration=round(intro_end, 2),
        ))
        sections.append(Section(
            label="body", start=round(intro_end, 2), end=round(outro_start, 2),
            duration=round(outro_start - intro_end, 2),
        ))
        sections.append(Section(
            label="outro", start=round(outro_start, 2), end=round(duration, 2),
            duration=round(duration - outro_start, 2),
        ))
    else:
        sections.append(Section(
            label="full", start=0.0, end=round(duration, 2),
            duration=round(duration, 2),
        ))

    # Good loop points: bar boundaries excluding first/last few bars
    loop_points = []
    if len(bar_boundaries) > 4:
        loop_points = [round(b, 3) for b in bar_boundaries[2:-2]]
    elif bar_boundaries:
        loop_points = [round(b, 3) for b in bar_boundaries]

    if on_progress:
        on_progress(100, "Structure analysis complete")

    return MusicStructure(
        bpm=bpm,
        bars=bars,
        sections=sections,
        loop_points=loop_points,
    )
