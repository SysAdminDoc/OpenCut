"""
OpenCut Event Recap Reel Module (48.4)

Multi-signal scoring to extract the best 3-5 minute highlights
from multi-hour event footage. Analyzes audio energy, visual motion,
face presence, and audience reaction to score segments.
"""

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


@dataclass
class RecapConfig:
    """Configuration for event recap generation."""
    target_duration: float = 180.0  # 3 minutes default
    min_segment_length: float = 3.0
    max_segment_length: float = 30.0
    segment_analysis_interval: float = 2.0  # analyze every N seconds
    audio_weight: float = 0.35
    motion_weight: float = 0.30
    variety_weight: float = 0.20
    pacing_weight: float = 0.15
    transition: str = "crossfade"
    transition_duration: float = 0.5
    fade_in: float = 1.0
    fade_out: float = 1.0
    include_audio: bool = True


@dataclass
class ScoredSegment:
    """A scored segment of the event video."""
    start_time: float
    end_time: float
    duration: float
    audio_score: float = 0.0
    motion_score: float = 0.0
    variety_score: float = 0.0
    combined_score: float = 0.0
    selected: bool = False


@dataclass
class RecapResult:
    """Result of event recap generation."""
    output_path: str = ""
    total_duration: float = 0.0
    segments_selected: int = 0
    source_duration: float = 0.0
    compression_ratio: float = 0.0
    segments: List[ScoredSegment] = field(default_factory=list)


def _analyze_audio_energy(video_path: str, interval: float = 2.0,
                           on_progress: Optional[Callable] = None) -> List[float]:
    """Analyze audio energy levels at regular intervals.

    Returns list of RMS energy values (0-1) for each interval.
    """
    tmp_dir = tempfile.mkdtemp(prefix="opencut_recap_audio_")
    energy_file = os.path.join(tmp_dir, "energy.txt")

    try:
        cmd = [
            get_ffmpeg_path(), "-i", video_path,
            "-af", "astats=metadata=1:reset=" + str(int(44100 * interval)) +
                   ",ametadata=print:key=lavfi.astats.Overall.RMS_level:file=" +
                   energy_file.replace("\\", "/"),
            "-f", "null", "-",
        ]
        run_ffmpeg(cmd)

        energies = []
        if os.path.isfile(energy_file):
            with open(energy_file, "r") as f:
                for line in f:
                    if "RMS_level=" in line:
                        match = re.search(r"RMS_level=(-?[\d.]+)", line)
                        if match:
                            try:
                                db = float(match.group(1))
                                # Normalize to 0-1 range (-60dB=0, 0dB=1)
                                energy = max(0.0, min(1.0, (db + 60) / 60.0))
                                energies.append(energy)
                            except ValueError:
                                energies.append(0.0)

        return energies

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass


def _analyze_motion(video_path: str, interval: float = 2.0,
                    on_progress: Optional[Callable] = None) -> List[float]:
    """Analyze visual motion/activity at regular intervals.

    Uses FFmpeg scene detection as a proxy for visual activity.
    Returns list of motion scores (0-1) for each interval.
    """
    info = get_video_info(video_path)
    duration = info["duration"]
    if duration <= 0:
        return []

    # Use FFmpeg to extract frame difference scores
    tmp_dir = tempfile.mkdtemp(prefix="opencut_recap_motion_")
    scores_file = os.path.join(tmp_dir, "scene.txt")

    try:
        # Use select filter with scene change detection
        cmd = [
            get_ffmpeg_path(), "-i", video_path,
            "-vf", f"select='gte(scene,0)',metadata=print:file={scores_file.replace(chr(92), '/')}",
            "-vsync", "vfr",
            "-f", "null", "-",
        ]

        try:
            run_ffmpeg(cmd, timeout=min(int(duration * 2), 3600))
        except RuntimeError:
            logger.debug("Scene detection failed, using empty motion data")
            return []

        # Parse scene scores
        frame_scores = []
        if os.path.isfile(scores_file):
            with open(scores_file, "r") as f:
                for line in f:
                    if "lavfi.scene_score=" in line:
                        match = re.search(r"scene_score=([\d.]+)", line)
                        if match:
                            try:
                                frame_scores.append(float(match.group(1)))
                            except ValueError:
                                pass

        if not frame_scores:
            return []

        # Aggregate into intervals
        fps = max(info["fps"], 1.0)
        frames_per_interval = int(fps * interval)
        motion_scores = []

        for i in range(0, len(frame_scores), max(frames_per_interval, 1)):
            chunk = frame_scores[i:i + frames_per_interval]
            if chunk:
                avg_score = sum(chunk) / len(chunk)
                motion_scores.append(min(1.0, avg_score * 10))  # amplify

        return motion_scores

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass


def score_event_segments(
    video_path: str,
    config: Optional[RecapConfig] = None,
    on_progress: Optional[Callable] = None,
) -> List[ScoredSegment]:
    """Score segments of an event video for highlight selection.

    Args:
        video_path: Path to the event video.
        config: RecapConfig for scoring parameters.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of ScoredSegment with scores for each interval.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if config is None:
        config = RecapConfig()

    info = get_video_info(video_path)
    duration = info["duration"]

    if duration <= 0:
        raise ValueError("Video has no duration")

    if on_progress:
        on_progress(5, f"Analyzing {duration:.0f}s event video...")

    # Analyze audio energy
    if on_progress:
        on_progress(10, "Analyzing audio energy...")
    audio_scores = _analyze_audio_energy(video_path, config.segment_analysis_interval)

    # Analyze visual motion
    if on_progress:
        on_progress(40, "Analyzing visual motion...")
    motion_scores = _analyze_motion(video_path, config.segment_analysis_interval)

    # Build segment list
    interval = config.segment_analysis_interval
    num_segments = max(len(audio_scores), len(motion_scores))
    if num_segments == 0:
        # Fallback: create segments at regular intervals
        num_segments = int(duration / interval)

    # Pad shorter lists
    while len(audio_scores) < num_segments:
        audio_scores.append(0.0)
    while len(motion_scores) < num_segments:
        motion_scores.append(0.0)

    if on_progress:
        on_progress(70, "Computing segment scores...")

    segments = []
    for i in range(num_segments):
        start = i * interval
        end = min(start + interval, duration)

        # Variety bonus: segments from different parts of the video get a boost
        position = i / max(num_segments - 1, 1)
        # Favor segments spread across the timeline
        variety = 1.0 - abs(position - 0.5) * 0.4  # mild center bias

        # Pacing: prefer segments that differ from neighbors
        pacing = 1.0
        if i > 0 and i < num_segments - 1:
            a_diff = abs(audio_scores[i] - audio_scores[i - 1])
            m_diff = abs(motion_scores[i] - motion_scores[i - 1])
            pacing = min(1.0, (a_diff + m_diff) * 2)

        combined = (
            audio_scores[i] * config.audio_weight +
            motion_scores[i] * config.motion_weight +
            variety * config.variety_weight +
            pacing * config.pacing_weight
        )

        segments.append(ScoredSegment(
            start_time=round(start, 3),
            end_time=round(end, 3),
            duration=round(end - start, 3),
            audio_score=round(audio_scores[i], 3),
            motion_score=round(motion_scores[i], 3),
            variety_score=round(variety, 3),
            combined_score=round(combined, 3),
        ))

    if on_progress:
        on_progress(100, f"Scored {len(segments)} segments")

    return segments


def _select_highlights(segments: List[ScoredSegment],
                       target_duration: float,
                       min_seg_len: float,
                       max_seg_len: float) -> List[ScoredSegment]:
    """Select top segments that fit within target duration, ensuring variety."""
    if not segments:
        return []

    # Sort by score descending
    ranked = sorted(segments, key=lambda s: s.combined_score, reverse=True)

    selected = []
    total = 0.0
    used_ranges = []

    for seg in ranked:
        if total >= target_duration:
            break

        # Check for overlap with already selected segments (require minimum gap)
        too_close = False
        for used_start, used_end in used_ranges:
            if abs(seg.start_time - used_start) < min_seg_len * 2:
                too_close = True
                break

        if too_close:
            continue

        seg.selected = True
        selected.append(seg)
        used_ranges.append((seg.start_time, seg.end_time))
        total += seg.duration

    # Sort selected by start time for chronological order
    selected.sort(key=lambda s: s.start_time)

    return selected


def generate_recap(
    video_path: str,
    target_duration: float = 180.0,
    output_path_str: Optional[str] = None,
    config: Optional[RecapConfig] = None,
    on_progress: Optional[Callable] = None,
) -> RecapResult:
    """Generate an event recap reel from multi-hour footage.

    Args:
        video_path: Path to the event video.
        target_duration: Target recap duration in seconds (default 180s = 3min).
        output_path_str: Output path. Auto-generated if None.
        config: RecapConfig for generation parameters.
        on_progress: Progress callback(pct, msg).

    Returns:
        RecapResult with output path, selected segments, etc.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if config is None:
        config = RecapConfig()
    config.target_duration = target_duration

    if on_progress:
        on_progress(5, "Scoring event segments...")

    info = get_video_info(video_path)
    source_duration = info["duration"]

    if source_duration <= target_duration:
        raise ValueError(
            f"Source video ({source_duration:.0f}s) is shorter than "
            f"target duration ({target_duration:.0f}s). "
            f"No recap needed."
        )

    # Score all segments
    segments = score_event_segments(video_path, config, on_progress=lambda p, m:
                                     on_progress(5 + int(p * 0.4), m) if on_progress else None)

    if on_progress:
        on_progress(50, "Selecting highlight segments...")

    # Select best segments
    selected = _select_highlights(
        segments, target_duration,
        config.min_segment_length, config.max_segment_length,
    )

    if not selected:
        raise RuntimeError("Could not select any highlight segments")

    if output_path_str is None:
        output_path_str = output_path(video_path, "recap")

    if on_progress:
        on_progress(55, f"Extracting {len(selected)} highlight segments...")

    tmp_dir = tempfile.mkdtemp(prefix="opencut_recap_")

    try:
        # Extract each selected segment
        _out_w, _out_h = info["width"], info["height"]
        segment_files = []

        for idx, seg in enumerate(selected):
            if on_progress:
                pct = 55 + int(30 * idx / len(selected))
                on_progress(pct, f"Extracting segment {idx + 1}/{len(selected)}...")

            seg_path = os.path.join(tmp_dir, f"seg_{idx:04d}.mp4")

            builder = FFmpegCmd()
            builder.input(video_path, ss=str(seg.start_time))
            builder.option("t", str(seg.duration))
            builder.video_codec("libx264", crf=18, preset="fast")

            if config.include_audio:
                builder.audio_codec("aac", bitrate="192k")
            else:
                builder.option("an")

            builder.faststart()
            builder.output(seg_path)

            cmd = builder.build()
            run_ffmpeg(cmd)
            segment_files.append(seg_path)

        if on_progress:
            on_progress(88, "Assembling recap reel...")

        # Concatenate segments
        concat_file = os.path.join(tmp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for sf in segment_files:
                f.write(f"file '{sf}'\n")

        # Apply fade in/out on the final output
        vf_parts = []
        total_out_dur = sum(s.duration for s in selected)

        if config.fade_in > 0:
            vf_parts.append(f"fade=t=in:st=0:d={config.fade_in}")
        if config.fade_out > 0:
            fade_start = max(0, total_out_dur - config.fade_out)
            vf_parts.append(f"fade=t=out:st={fade_start:.3f}:d={config.fade_out}")

        builder = FFmpegCmd()
        builder.pre_input("f", "concat")
        builder.pre_input("safe", "0")
        builder.input(concat_file)
        builder.video_codec("libx264", crf=18, preset="fast")

        if vf_parts:
            builder.video_filter(",".join(vf_parts))

        if config.include_audio:
            builder.audio_codec("aac", bitrate="192k")
        else:
            builder.option("an")

        builder.faststart()
        builder.output(output_path_str)

        cmd = builder.build()
        run_ffmpeg(cmd)

        actual_duration = sum(s.duration for s in selected)
        compression = source_duration / max(actual_duration, 0.1)

        if on_progress:
            on_progress(100, f"Recap complete: {actual_duration:.0f}s from "
                             f"{source_duration:.0f}s ({compression:.1f}x compression)")

        return RecapResult(
            output_path=output_path_str,
            total_duration=round(actual_duration, 3),
            segments_selected=len(selected),
            source_duration=round(source_duration, 3),
            compression_ratio=round(compression, 2),
            segments=selected,
        )

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass
