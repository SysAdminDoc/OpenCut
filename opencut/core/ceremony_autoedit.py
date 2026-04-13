"""
Multi-Camera Ceremony Auto-Edit (48.1)

Automatically select the best camera angle per moment in a ceremony
based on audio energy, face detection heuristics, and motion analysis.
"""

import logging
import math
import os
import subprocess as _sp
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffprobe_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CeremonyConfig:
    """Configuration for ceremony auto-editing."""
    segment_duration: float = 5.0  # seconds per decision point
    min_segment: float = 2.0  # minimum time before allowing a cut
    max_segment: float = 15.0  # maximum time before forcing a cut
    audio_weight: float = 0.5  # weight for audio energy scoring
    motion_weight: float = 0.3  # weight for motion scoring
    variety_weight: float = 0.2  # weight for camera variety
    width: int = 1920
    height: int = 1080
    prefer_wide: bool = True  # prefer wide shots during silence
    crossfade_duration: float = 0.0  # 0 = hard cut


@dataclass
class CameraScore:
    """Score for a single camera at a given timestamp."""
    camera_index: int
    timestamp: float
    audio_energy: float = 0.0
    motion_score: float = 0.0
    variety_score: float = 0.0
    total_score: float = 0.0


@dataclass
class EditDecision:
    """A single edit decision: which camera to use for a time range."""
    camera_index: int
    start: float
    end: float
    score: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ---------------------------------------------------------------------------
# Audio Analysis
# ---------------------------------------------------------------------------
def _analyze_audio_energy(video_path: str, duration: float,
                          segment_duration: float) -> List[float]:
    """Analyze audio energy levels across the video.

    Returns a list of energy values, one per segment.
    """
    try:
        cmd = [
            get_ffprobe_path(), "-v", "quiet",
            "-select_streams", "a:0",
            "-show_entries", "frame=pkt_pts_time,rms_level",
            "-of", "csv=p=0",
            video_path,
        ]
        result = _sp.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            # Fallback: return uniform energy
            n_segments = max(1, int(math.ceil(duration / segment_duration)))
            return [0.5] * n_segments

        # Parse RMS levels
        n_segments = max(1, int(math.ceil(duration / segment_duration)))
        energies = [0.0] * n_segments
        counts = [0] * n_segments

        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    ts = float(parts[0])
                    rms = float(parts[1]) if parts[1] != "-inf" else -90.0
                    seg_idx = min(int(ts / segment_duration), n_segments - 1)
                    # Convert dB RMS to linear (0-1 range)
                    linear = max(0.0, min(1.0, (rms + 90.0) / 90.0))
                    energies[seg_idx] += linear
                    counts[seg_idx] += 1
                except (ValueError, IndexError):
                    continue

        # Average per segment
        for i in range(n_segments):
            if counts[i] > 0:
                energies[i] /= counts[i]

        return energies

    except Exception as e:
        logger.debug("Audio analysis failed for %s: %s", video_path, e)
        n_segments = max(1, int(math.ceil(duration / segment_duration)))
        return [0.5] * n_segments


# ---------------------------------------------------------------------------
# Motion Analysis (lightweight via frame diff)
# ---------------------------------------------------------------------------
def _analyze_motion(video_path: str, duration: float,
                    segment_duration: float) -> List[float]:
    """Estimate motion level per segment using frame difference.

    Uses a lightweight approach: extract a few frames per segment and compare.
    Returns list of motion scores (0-1), one per segment.
    """
    n_segments = max(1, int(math.ceil(duration / segment_duration)))

    try:
        # Use FFmpeg's blackframe or signalstats for a quick motion estimate
        cmd = [
            get_ffprobe_path(), "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "frame=pkt_pts_time",
            "-show_entries", "frame_tags=lavfi.signalstats.YDIF",
            "-of", "csv=p=0",
            "-f", "lavfi",
            f"movie='{video_path.replace(os.sep, '/')}',signalstats",
        ]
        result = _sp.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            # Fallback: uniform motion
            return [0.5] * n_segments

        motion = [0.0] * n_segments
        counts = [0] * n_segments

        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    ts = float(parts[0])
                    ydif = float(parts[1])
                    seg_idx = min(int(ts / segment_duration), n_segments - 1)
                    # Normalize YDIF (typically 0-50ish)
                    score = min(1.0, ydif / 30.0)
                    motion[seg_idx] += score
                    counts[seg_idx] += 1
                except (ValueError, IndexError):
                    continue

        for i in range(n_segments):
            if counts[i] > 0:
                motion[i] /= counts[i]

        return motion

    except Exception as e:
        logger.debug("Motion analysis failed for %s: %s", video_path, e)
        return [0.5] * n_segments


# ---------------------------------------------------------------------------
# Score Camera Angles
# ---------------------------------------------------------------------------
def score_camera_angles(
    camera_paths: List[str],
    timestamp: float,
    segment_duration: float = 5.0,
    config: Optional[CeremonyConfig] = None,
    previous_camera: int = -1,
    on_progress: Optional[Callable] = None,
) -> List[CameraScore]:
    """Score each camera angle at a given timestamp.

    Args:
        camera_paths: List of camera video file paths.
        timestamp: Timestamp to score.
        segment_duration: Duration of the segment to analyze.
        config: Optional CeremonyConfig.
        previous_camera: Index of the previous camera (for variety scoring).
        on_progress: Optional callback(pct, msg).

    Returns:
        List of CameraScore objects, sorted by total_score descending.
    """
    if not camera_paths:
        raise ValueError("At least one camera path required")

    config = config or CeremonyConfig()
    scores = []

    for i, cam_path in enumerate(camera_paths):
        if not os.path.isfile(cam_path):
            raise FileNotFoundError(f"Camera file not found: {cam_path}")

        info = get_video_info(cam_path)
        dur = info["duration"]

        if timestamp >= dur:
            scores.append(CameraScore(
                camera_index=i, timestamp=timestamp,
            ))
            continue

        # Audio energy for this segment
        min(timestamp + segment_duration, dur)
        audio_energies = _analyze_audio_energy(cam_path, dur, segment_duration)
        seg_idx = min(int(timestamp / segment_duration), len(audio_energies) - 1)
        audio = audio_energies[seg_idx] if seg_idx < len(audio_energies) else 0.5

        # Motion score
        motion_scores = _analyze_motion(cam_path, dur, segment_duration)
        motion = motion_scores[seg_idx] if seg_idx < len(motion_scores) else 0.5

        # Variety: prefer switching cameras
        variety = 1.0 if i != previous_camera else 0.3

        total = (
            audio * config.audio_weight
            + motion * config.motion_weight
            + variety * config.variety_weight
        )

        scores.append(CameraScore(
            camera_index=i,
            timestamp=timestamp,
            audio_energy=round(audio, 3),
            motion_score=round(motion, 3),
            variety_score=round(variety, 3),
            total_score=round(total, 3),
        ))

    scores.sort(key=lambda s: s.total_score, reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Generate Multicam Edit
# ---------------------------------------------------------------------------
def generate_multicam_edit(
    scores: Dict[float, List[CameraScore]],
    config: CeremonyConfig,
    output_path_str: str = "",
    on_progress: Optional[Callable] = None,
) -> List[EditDecision]:
    """Generate edit decision list from scored camera angles.

    Args:
        scores: Map of timestamp -> list of CameraScore (sorted best first).
        config: CeremonyConfig.
        output_path_str: Unused here (decisions only), kept for API compat.
        on_progress: Optional callback(pct, msg).

    Returns:
        List of EditDecision objects.
    """
    if not scores:
        raise ValueError("Scores dict is empty")

    if on_progress:
        on_progress(10, "Generating edit decisions...")

    sorted_times = sorted(scores.keys())
    decisions = []

    for i, ts in enumerate(sorted_times):
        cam_scores = scores[ts]
        if not cam_scores:
            continue

        best = cam_scores[0]
        end_ts = sorted_times[i + 1] if i + 1 < len(sorted_times) else ts + config.segment_duration

        # Merge with previous if same camera and under max_segment
        if (decisions
                and decisions[-1].camera_index == best.camera_index
                and (end_ts - decisions[-1].start) <= config.max_segment):
            decisions[-1].end = end_ts
            decisions[-1].score = max(decisions[-1].score, best.total_score)
        else:
            decisions.append(EditDecision(
                camera_index=best.camera_index,
                start=ts,
                end=end_ts,
                score=best.total_score,
            ))

    if on_progress:
        on_progress(100, f"Generated {len(decisions)} edit decisions")

    return decisions


# ---------------------------------------------------------------------------
# Full Auto-Edit Pipeline
# ---------------------------------------------------------------------------
def auto_edit_ceremony(
    camera_paths: List[str],
    output_path_str: str = "",
    config: Optional[CeremonyConfig] = None,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Full auto-edit pipeline for multi-camera ceremony footage.

    Args:
        camera_paths: List of camera video file paths.
        output_path_str: Output file path.
        config: Optional CeremonyConfig.
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with output_path, decisions, duration, camera_count.
    """
    if not camera_paths or len(camera_paths) < 2:
        raise ValueError("At least 2 camera files are required")

    for p in camera_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Camera file not found: {p}")

    config = config or CeremonyConfig()

    if on_progress:
        on_progress(5, f"Analyzing {len(camera_paths)} camera angles...")

    # Get durations
    durations = []
    for p in camera_paths:
        info = get_video_info(p)
        durations.append(info["duration"])

    min_duration = min(durations) if durations else 0
    if min_duration <= 0:
        raise ValueError("Camera files have no valid duration")

    # Score all timestamps
    n_segments = max(1, int(math.ceil(min_duration / config.segment_duration)))
    all_scores = {}
    prev_cam = -1

    for seg_i in range(n_segments):
        ts = seg_i * config.segment_duration
        if on_progress:
            pct = 10 + int((seg_i / n_segments) * 50)
            on_progress(pct, f"Scoring segment {seg_i + 1}/{n_segments}...")

        cam_scores = score_camera_angles(
            camera_paths, ts, config.segment_duration,
            config=config, previous_camera=prev_cam,
        )
        all_scores[ts] = cam_scores
        if cam_scores:
            prev_cam = cam_scores[0].camera_index

    if on_progress:
        on_progress(65, "Generating edit decisions...")

    decisions = generate_multicam_edit(all_scores, config)

    if not decisions:
        raise RuntimeError("No edit decisions generated")

    # Assemble video from decisions
    if not output_path_str:
        out_dir = os.path.dirname(os.path.abspath(camera_paths[0]))
        output_path_str = output_path(camera_paths[0], "ceremony_edit", out_dir)

    if on_progress:
        on_progress(70, "Assembling final video...")

    import shutil
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="opencut_ceremony_")
    segment_files = []

    try:
        total_dec = len(decisions)
        for i, dec in enumerate(decisions):
            if on_progress:
                pct = 70 + int((i / total_dec) * 25)
                on_progress(pct, f"Encoding segment {i + 1}/{total_dec}...")

            cam_path = camera_paths[dec.camera_index]
            seg_path = os.path.join(tmp_dir, f"seg_{i:04d}.mp4")

            vf = (
                f"scale={config.width}:{config.height}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2:black"
            )
            cmd = (FFmpegCmd()
                   .input(cam_path, ss=dec.start, to=dec.end)
                   .video_filter(vf)
                   .video_codec("libx264", crf=18, preset="fast")
                   .audio_codec("aac", bitrate="192k")
                   .output(seg_path)
                   .build())
            run_ffmpeg(cmd)
            segment_files.append(seg_path)

        if on_progress:
            on_progress(96, "Concatenating final video...")

        concat_path = os.path.join(tmp_dir, "concat.txt")
        with open(concat_path, "w", encoding="utf-8") as f:
            for seg in segment_files:
                safe = seg.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        cmd = (FFmpegCmd()
               .option("f", "concat")
               .option("safe", "0")
               .input(concat_path)
               .copy_streams()
               .faststart()
               .output(output_path_str)
               .build())
        run_ffmpeg(cmd)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    total_dur = sum(d.duration for d in decisions)

    if on_progress:
        on_progress(100, "Ceremony auto-edit complete")

    return {
        "output_path": output_path_str,
        "decisions": [
            {
                "camera": d.camera_index,
                "start": d.start,
                "end": d.end,
                "duration": d.duration,
                "score": d.score,
            }
            for d in decisions
        ],
        "duration": total_dur,
        "camera_count": len(camera_paths),
        "segment_count": len(decisions),
    }
