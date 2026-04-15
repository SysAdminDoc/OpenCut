"""
OpenCut Beat-Synced Video Editing Module (Category 75)

Auto-edit video cuts to music beats.  Detect beats in audio track using
onset detection via FFmpeg.  Align video cuts to beats with multiple modes:
every beat, every bar, accent only, or custom interval.

Functions:
    detect_beats         - Detect beats/onsets in audio track
    plan_beat_sync_cuts  - Plan video cuts aligned to beats
    assemble_beat_sync   - Execute beat-synced assembly via FFmpeg concat
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BEAT_SYNC_MODES = {
    "every_beat": {"description": "Cut on every detected beat", "beat_divisor": 1},
    "every_bar": {"description": "Cut every 4 beats (one bar in 4/4)", "beat_divisor": 4},
    "accent_only": {"description": "Cut on strong beats/downbeats only", "beat_divisor": 4},
    "every_2_beats": {"description": "Cut every 2 beats", "beat_divisor": 2},
    "every_8_beats": {"description": "Cut every 8 beats (2 bars)", "beat_divisor": 8},
    "custom": {"description": "Cut every N beats (user-specified)", "beat_divisor": 1},
}

DEFAULT_MODE = "every_beat"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class Beat:
    """A detected beat in the audio."""
    timestamp: float
    strength: float = 1.0  # Relative strength 0.0-1.0
    beat_number: int = 0
    is_downbeat: bool = False

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 4),
            "strength": round(self.strength, 3),
            "beat_number": self.beat_number,
            "is_downbeat": self.is_downbeat,
        }


@dataclass
class CutPoint:
    """A planned video cut point aligned to a beat."""
    clip_index: int
    clip_path: str
    in_point: float
    out_point: float
    beat_number: int
    beat_timestamp: float

    def to_dict(self) -> dict:
        return {
            "clip_index": self.clip_index,
            "clip_path": self.clip_path,
            "in_point": round(self.in_point, 4),
            "out_point": round(self.out_point, 4),
            "beat_number": self.beat_number,
            "beat_timestamp": round(self.beat_timestamp, 4),
        }


@dataclass
class BeatDetectResult:
    """Result of beat detection."""
    beats: List[Beat] = field(default_factory=list)
    tempo_bpm: float = 0.0
    total_beats: int = 0
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "beats": [b.to_dict() for b in self.beats],
            "tempo_bpm": round(self.tempo_bpm, 1),
            "total_beats": self.total_beats,
            "duration": round(self.duration, 3),
        }


@dataclass
class BeatSyncResult:
    """Result of beat-synced video assembly."""
    cuts: List[CutPoint] = field(default_factory=list)
    tempo_bpm: float = 0.0
    total_beats: int = 0
    output_path: str = ""
    mode: str = ""
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cuts": [c.to_dict() for c in self.cuts],
            "tempo_bpm": round(self.tempo_bpm, 1),
            "total_beats": self.total_beats,
            "output_path": self.output_path,
            "mode": self.mode,
            "duration": round(self.duration, 3),
        }


# ---------------------------------------------------------------------------
# Beat Detection
# ---------------------------------------------------------------------------
def _detect_onsets_ffmpeg(audio_path: str, threshold: float = 0.1) -> List[Tuple[float, float]]:
    """Detect onsets using FFmpeg silencedetect as a proxy for transients.

    Returns list of (timestamp, relative_strength) tuples.
    """
    cmd = [
        get_ffmpeg_path(), "-i", audio_path,
        "-af", f"silencedetect=noise=-25dB:d={threshold}",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stderr = result.stderr
    except Exception as e:
        logger.warning("Onset detection failed: %s", e)
        return []

    onsets = []
    for line in stderr.split("\n"):
        if "silence_end:" in line:
            try:
                parts = line.split("silence_end:")[1].strip().split("|")
                ts = float(parts[0].strip().split(" ")[0])
                # Extract duration (used as proxy for onset strength)
                strength = 0.5
                if "silence_duration:" in line:
                    dur_str = line.split("silence_duration:")[1].strip()
                    sil_dur = float(dur_str.split(" ")[0] if " " in dur_str else dur_str)
                    # Shorter silence before onset = stronger beat
                    strength = max(0.1, min(1.0, 1.0 / (1.0 + sil_dur)))
                onsets.append((ts, strength))
            except (ValueError, IndexError):
                continue

    return onsets


def _detect_onsets_astats(audio_path: str) -> List[Tuple[float, float]]:
    """Alternative onset detection using energy analysis with astats.

    Splits audio into short frames and detects energy peaks.
    """
    frame_dur = 0.05  # 50ms frames
    cmd = [
        get_ffmpeg_path(), "-i", audio_path,
        "-af", f"asetnsamples=n={int(44100 * frame_dur)},astats=metadata=1:reset=1",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stderr = result.stderr
    except Exception:
        return []

    # Parse RMS levels from astats output
    rms_values = []
    current_time = 0.0
    for line in stderr.split("\n"):
        if "RMS_level" in line:
            try:
                rms = float(line.split("=")[1].strip())
                if rms > -100:  # Filter out -inf
                    rms_values.append((current_time, rms))
                current_time += frame_dur
            except (ValueError, IndexError):
                current_time += frame_dur

    if not rms_values:
        return []

    # Find peaks (onset = local maximum in RMS)
    onsets = []
    for i in range(1, len(rms_values) - 1):
        prev_rms = rms_values[i - 1][1]
        curr_rms = rms_values[i][1]
        next_rms = rms_values[i + 1][1]

        if curr_rms > prev_rms and curr_rms > next_rms:
            # Normalize strength: -60dB = 0.0, 0dB = 1.0
            strength = max(0.0, min(1.0, (curr_rms + 60.0) / 60.0))
            if strength > 0.2:
                onsets.append((rms_values[i][0], strength))

    return onsets


def _quantize_to_grid(
    onsets: List[Tuple[float, float]], tempo_bpm: float, duration: float,
) -> List[Beat]:
    """Quantize detected onsets to a beat grid and assign beat numbers."""
    if tempo_bpm <= 0:
        tempo_bpm = 120.0

    beat_interval = 60.0 / tempo_bpm
    total_beats = int(duration / beat_interval) + 1

    # Create beat grid
    beat_grid = []
    for i in range(total_beats):
        ts = i * beat_interval
        if ts > duration:
            break
        beat_grid.append(Beat(
            timestamp=ts,
            strength=0.3,  # Default weak strength
            beat_number=i,
            is_downbeat=(i % 4 == 0),
        ))

    # Map onsets to nearest beats, boosting strength
    for onset_ts, onset_strength in onsets:
        closest_idx = -1
        closest_dist = float("inf")
        for idx, beat in enumerate(beat_grid):
            dist = abs(beat.timestamp - onset_ts)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = idx

        if closest_idx >= 0 and closest_dist < beat_interval * 0.5:
            beat_grid[closest_idx].strength = max(
                beat_grid[closest_idx].strength, onset_strength
            )

    return beat_grid


def _estimate_tempo(onsets: List[Tuple[float, float]]) -> float:
    """Estimate tempo from onset intervals."""
    if len(onsets) < 3:
        return 120.0

    intervals = [onsets[i + 1][0] - onsets[i][0] for i in range(len(onsets) - 1)]
    # Filter to reasonable intervals (50-250 BPM range)
    valid_intervals = [iv for iv in intervals if 0.24 < iv < 1.2]

    if not valid_intervals:
        return 120.0

    # Use median interval for robustness
    valid_intervals.sort()
    median_interval = valid_intervals[len(valid_intervals) // 2]

    if median_interval <= 0:
        return 120.0

    bpm = 60.0 / median_interval
    return max(40.0, min(300.0, bpm))


def detect_beats(
    audio_path: str,
    sensitivity: float = 0.5,
    on_progress: Optional[Callable] = None,
) -> BeatDetectResult:
    """Detect beats in an audio track.

    Args:
        audio_path: Path to audio file.
        sensitivity: Detection sensitivity 0.0-1.0 (higher = more beats).
        on_progress: Progress callback (int percentage).

    Returns:
        BeatDetectResult with beats, tempo, count.
    """
    if on_progress:
        on_progress(5)

    # Get duration
    info_cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json", audio_path,
    ]
    try:
        result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
    except Exception:
        duration = 0.0

    if on_progress:
        on_progress(15)

    # Detect onsets using silence detection
    threshold = 0.05 + (1.0 - sensitivity) * 0.15
    onsets = _detect_onsets_ffmpeg(audio_path, threshold=threshold)

    if on_progress:
        on_progress(50)

    # Try supplemental onset detection if few onsets found
    if len(onsets) < 5:
        extra_onsets = _detect_onsets_astats(audio_path)
        # Merge, removing near-duplicates
        for ts, strength in extra_onsets:
            if not any(abs(ts - ots) < 0.1 for ots, _ in onsets):
                onsets.append((ts, strength))
        onsets.sort(key=lambda x: x[0])

    if on_progress:
        on_progress(65)

    # Estimate tempo
    tempo = _estimate_tempo(onsets)

    if on_progress:
        on_progress(75)

    # Quantize to grid
    beats = _quantize_to_grid(onsets, tempo, duration)

    if on_progress:
        on_progress(90)

    result = BeatDetectResult(
        beats=beats,
        tempo_bpm=tempo,
        total_beats=len(beats),
        duration=duration,
    )
    logger.info("Beat detection: %.1f BPM, %d beats in %.1fs", tempo, len(beats), duration)
    return result


# ---------------------------------------------------------------------------
# Cut Planning
# ---------------------------------------------------------------------------
def _get_clip_info(clip_path: str) -> dict:
    """Get clip duration and basic info."""
    info = get_video_info(clip_path)
    return {
        "path": clip_path,
        "duration": info.get("duration", 0),
        "fps": info.get("fps", 30.0),
        "width": info.get("width", 1920),
        "height": info.get("height", 1080),
    }


def _select_cut_beats(
    beats: List[Beat],
    mode: str = "every_beat",
    custom_n: int = 1,
) -> List[Beat]:
    """Select beats to cut on based on mode."""
    if mode == "accent_only":
        # Only downbeats (beat 0, 4, 8, ...)
        return [b for b in beats if b.is_downbeat]

    if mode == "custom":
        divisor = max(1, custom_n)
    else:
        mode_info = BEAT_SYNC_MODES.get(mode, BEAT_SYNC_MODES[DEFAULT_MODE])
        divisor = mode_info["beat_divisor"]

    return [b for b in beats if b.beat_number % divisor == 0]


def plan_beat_sync_cuts(
    clip_paths: List[str],
    beats: List[Beat],
    mode: str = "every_beat",
    custom_n: int = 1,
    energy_match: bool = False,
) -> List[CutPoint]:
    """Plan video cuts aligned to beat positions.

    Args:
        clip_paths: List of video clip file paths.
        beats: Detected beats from detect_beats().
        mode: Beat sync mode name.
        custom_n: Custom beat interval (for 'custom' mode).
        energy_match: If True, pair high-energy clips with stronger beats.

    Returns:
        List of CutPoint instances defining the assembly.
    """
    if not clip_paths or not beats:
        return []

    cut_beats = _select_cut_beats(beats, mode=mode, custom_n=custom_n)
    if not cut_beats:
        return []

    # Get clip info
    clips = [_get_clip_info(p) for p in clip_paths]
    clip_count = len(clips)

    # Calculate energy scores for clips if energy matching is enabled
    if energy_match and clip_count > 1:
        # Sort clips by duration as proxy for energy
        # (longer clips = more content = potentially higher energy)
        indexed = list(enumerate(clips))
        indexed.sort(key=lambda x: x[1]["duration"], reverse=True)

        # Sort cut_beats by strength
        sorted_beats = sorted(
            enumerate(cut_beats), key=lambda x: x[1].strength, reverse=True
        )

        # Map: strongest beat -> "highest energy" clip
        beat_to_clip = {}
        for rank, (beat_idx, _beat) in enumerate(sorted_beats):
            clip_rank = rank % clip_count
            beat_to_clip[beat_idx] = indexed[clip_rank][0]

    cuts = []
    for beat_idx in range(len(cut_beats) - 1):
        beat_start = cut_beats[beat_idx]
        beat_end = cut_beats[beat_idx + 1]
        segment_dur = beat_end.timestamp - beat_start.timestamp

        if segment_dur <= 0:
            continue

        # Select clip
        if energy_match and clip_count > 1:
            clip_idx = beat_to_clip.get(beat_idx, beat_idx % clip_count)
        else:
            clip_idx = beat_idx % clip_count

        clip = clips[clip_idx]
        clip_dur = clip["duration"]

        # Determine in/out points within the clip
        if clip_dur <= 0:
            in_point = 0.0
            out_point = segment_dur
        elif clip_dur >= segment_dur:
            # Start from a position that allows the full segment
            max_start = clip_dur - segment_dur
            # Spread evenly across the clip
            in_point = (beat_idx * segment_dur) % max(0.01, max_start)
            out_point = in_point + segment_dur
        else:
            # Clip is shorter than segment — use whole clip
            in_point = 0.0
            out_point = min(clip_dur, segment_dur)

        cuts.append(CutPoint(
            clip_index=clip_idx,
            clip_path=clip["path"],
            in_point=in_point,
            out_point=out_point,
            beat_number=beat_start.beat_number,
            beat_timestamp=beat_start.timestamp,
        ))

    return cuts


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _extract_segment(clip_path: str, in_point: float, out_point: float, idx: int) -> str:
    """Extract a video segment to a temp file."""
    duration = out_point - in_point
    fd, seg_path = tempfile.mkstemp(suffix=f"_bseg{idx}.mp4")
    os.close(fd)

    cmd = [
        get_ffmpeg_path(), "-y",
        "-ss", f"{in_point:.4f}",
        "-i", clip_path,
        "-t", f"{duration:.4f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-avoid_negative_ts", "1",
        seg_path,
    ]
    run_ffmpeg(cmd)
    return seg_path


def assemble_beat_sync(
    audio_path: str,
    clip_paths: List[str],
    mode: str = "every_beat",
    custom_n: int = 1,
    sensitivity: float = 0.5,
    energy_match: bool = False,
    output_path_val: str = "",
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> BeatSyncResult:
    """Full beat-sync assembly pipeline: detect beats, plan cuts, render.

    Args:
        audio_path: Path to music/audio track.
        clip_paths: List of video clip paths to cut between.
        mode: Beat sync mode.
        custom_n: Custom beat interval for 'custom' mode.
        sensitivity: Beat detection sensitivity 0.0-1.0.
        energy_match: Pair high-energy clips with stronger beats.
        output_path_val: Explicit output path.
        output_dir: Output directory.
        on_progress: Progress callback (int percentage).

    Returns:
        BeatSyncResult with cuts, tempo, output path.
    """
    if on_progress:
        on_progress(2)

    if not clip_paths:
        raise ValueError("At least one video clip path is required")

    # Step 1: Detect beats
    beat_result = detect_beats(audio_path, sensitivity=sensitivity, on_progress=None)

    if on_progress:
        on_progress(25)

    # Step 2: Plan cuts
    cuts = plan_beat_sync_cuts(
        clip_paths=clip_paths,
        beats=beat_result.beats,
        mode=mode,
        custom_n=custom_n,
        energy_match=energy_match,
    )

    if on_progress:
        on_progress(35)

    if not cuts:
        logger.warning("No cuts planned — insufficient beats or clips")
        return BeatSyncResult(
            tempo_bpm=beat_result.tempo_bpm,
            total_beats=beat_result.total_beats,
            mode=mode,
            duration=beat_result.duration,
        )

    # Step 3: Extract segments
    segment_paths = []
    try:
        for idx, cut in enumerate(cuts):
            if on_progress:
                pct = 35 + int(((idx + 1) / len(cuts)) * 40)
                on_progress(pct)

            try:
                seg = _extract_segment(
                    cut.clip_path, cut.in_point, cut.out_point, idx
                )
                segment_paths.append(seg)
            except RuntimeError as e:
                logger.warning("Segment extraction failed for cut %d: %s", idx, e)

        if not segment_paths:
            return BeatSyncResult(
                tempo_bpm=beat_result.tempo_bpm,
                total_beats=beat_result.total_beats,
                mode=mode,
            )

        if on_progress:
            on_progress(78)

        # Step 4: Concatenate segments
        fd, list_path = tempfile.mkstemp(suffix="_beatconcat.txt")
        os.close(fd)
        try:
            with open(list_path, "w", encoding="utf-8") as f:
                for p in segment_paths:
                    safe_p = p.replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{safe_p}'\n")

            # Determine output path
            if output_path_val:
                out = output_path_val
            elif output_dir:
                base = os.path.splitext(os.path.basename(audio_path))[0]
                out = os.path.join(output_dir, f"{base}_beatsync.mp4")
            else:
                out = output_path(audio_path, "beatsync")
                if not out.lower().endswith((".mp4", ".mov", ".mkv")):
                    out = os.path.splitext(out)[0] + ".mp4"

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
            on_progress(88)

        # Step 5: Mix audio track onto the concatenated video
        fd, final_path = tempfile.mkstemp(suffix="_beatsync_final.mp4")
        os.close(fd)
        try:
            mix_cmd = [
                get_ffmpeg_path(), "-y",
                "-i", out, "-i", audio_path,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                final_path,
            ]
            run_ffmpeg(mix_cmd)
            # Replace intermediate with final
            try:
                os.replace(final_path, out)
            except OSError:
                # If replace fails, final_path is the output
                out = final_path
        except RuntimeError as e:
            logger.warning("Audio mix step failed, using video-only output: %s", e)
            try:
                os.unlink(final_path)
            except OSError:
                pass

        if on_progress:
            on_progress(95)

        result = BeatSyncResult(
            cuts=cuts,
            tempo_bpm=beat_result.tempo_bpm,
            total_beats=beat_result.total_beats,
            output_path=out,
            mode=mode,
            duration=beat_result.duration,
        )
        logger.info(
            "Beat-sync assembly: %d cuts, %.1f BPM, mode=%s -> %s",
            len(cuts), beat_result.tempo_bpm, mode, out,
        )
        return result

    finally:
        # Clean up segment temp files
        for p in segment_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def list_beat_sync_modes() -> List[Dict]:
    """Return available beat sync modes with descriptions."""
    return [
        {"name": name, "description": info["description"], "beat_divisor": info["beat_divisor"]}
        for name, info in BEAT_SYNC_MODES.items()
    ]
