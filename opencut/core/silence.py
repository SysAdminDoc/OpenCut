"""
Silence detection engine using FFmpeg's silencedetect filter.

Detects silent segments in audio/video files and returns time intervals
for both silent and non-silent (speech) regions.
"""

import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from ..utils.config import SilenceConfig
from ..utils.media import MediaInfo, probe


@dataclass
class TimeSegment:
    """A time segment with start and end in seconds."""
    start: float
    end: float
    label: str = ""  # "speech", "silence", or speaker label

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"TimeSegment({self.start:.3f}s - {self.end:.3f}s, {self.duration:.3f}s, '{self.label}')"


def detect_silences(
    filepath: str,
    threshold_db: float = -30.0,
    min_duration: float = 0.5,
) -> List[TimeSegment]:
    """
    Detect silent segments in an audio/video file using FFmpeg.

    Args:
        filepath: Path to the media file.
        threshold_db: Noise floor in dB. Audio quieter than this is silence.
                      Typical values: -30 (aggressive) to -50 (conservative).
        min_duration: Minimum silence duration in seconds to detect.

    Returns:
        List of TimeSegment objects representing silent regions.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i", filepath,
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
        "-vn",       # ignore video (faster)
        "-sn",       # ignore subtitles
        "-f", "null",
        "-",
    ]

    # Scale timeout: 10 min base + 3x file duration (long podcasts need more time)
    try:
        info = probe(filepath)
        timeout = max(600, int(info.duration * 3) + 120)
    except Exception:
        timeout = 1800  # 30 min fallback for very long files

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Install FFmpeg: https://ffmpeg.org/download.html")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timed out processing '{filepath}'")

    # Parse silencedetect output from stderr
    # Format: [silencedetect @ 0x...] silence_start: 1.234
    #         [silencedetect @ 0x...] silence_end: 5.678 | silence_duration: 4.444
    stderr = result.stderr

    silence_starts = []
    silence_ends = []

    for line in stderr.split("\n"):
        # Match silence_start
        start_match = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if start_match:
            silence_starts.append(float(start_match.group(1)))

        # Match silence_end
        end_match = re.search(r"silence_end:\s*(-?[\d.]+)", line)
        if end_match:
            silence_ends.append(float(end_match.group(1)))

    # Build silence segments
    silences = []
    for i, start in enumerate(silence_starts):
        if i < len(silence_ends):
            end = silence_ends[i]
        else:
            # Silence extends to end of file — get duration
            info = probe(filepath)
            end = info.duration

        if end > start:
            silences.append(TimeSegment(start=start, end=end, label="silence"))

    return silences


def detect_speech(
    filepath: str,
    config: Optional[SilenceConfig] = None,
) -> List[TimeSegment]:
    """
    Detect speech segments by inverting silence detection results.

    This is the primary function for silence removal workflows.
    Returns non-silent time segments with padding applied.

    Args:
        filepath: Path to the media file.
        config: Silence detection configuration. Uses defaults if None.

    Returns:
        List of TimeSegment objects representing speech regions.
    """
    if config is None:
        config = SilenceConfig()

    # Get file duration
    info = probe(filepath)
    total_duration = info.duration

    if total_duration <= 0:
        raise ValueError(f"Could not determine duration of '{filepath}'")

    # Detect silences
    silences = detect_silences(
        filepath,
        threshold_db=config.threshold_db,
        min_duration=config.min_duration,
    )

    # Invert: get speech segments (gaps between silences)
    speech_segments = []

    if not silences:
        # No silence detected — entire file is speech
        return [TimeSegment(start=0.0, end=total_duration, label="speech")]

    # Speech before first silence
    if silences[0].start > 0:
        speech_segments.append(TimeSegment(
            start=0.0,
            end=silences[0].start,
            label="speech",
        ))

    # Speech between silences
    for i in range(len(silences) - 1):
        gap_start = silences[i].end
        gap_end = silences[i + 1].start
        if gap_end > gap_start:
            speech_segments.append(TimeSegment(
                start=gap_start,
                end=gap_end,
                label="speech",
            ))

    # Speech after last silence
    if silences[-1].end < total_duration:
        speech_segments.append(TimeSegment(
            start=silences[-1].end,
            end=total_duration,
            label="speech",
        ))

    # Apply padding (extend speech segments slightly into silence)
    padded = []
    for seg in speech_segments:
        new_start = max(0.0, seg.start - config.padding_before)
        new_end = min(total_duration, seg.end + config.padding_after)
        padded.append(TimeSegment(start=new_start, end=new_end, label="speech"))

    # Merge overlapping segments (can happen after padding)
    merged = _merge_overlapping(padded)

    # Filter out very short speech segments
    filtered = [s for s in merged if s.duration >= config.min_speech_duration]

    return filtered


def _merge_overlapping(segments: List[TimeSegment]) -> List[TimeSegment]:
    """Merge overlapping or adjacent time segments."""
    if not segments:
        return []

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: s.start)
    merged = [TimeSegment(
        start=sorted_segs[0].start,
        end=sorted_segs[0].end,
        label=sorted_segs[0].label,
    )]

    for seg in sorted_segs[1:]:
        if seg.start <= merged[-1].end:
            # Overlapping — extend the current segment
            merged[-1].end = max(merged[-1].end, seg.end)
        else:
            merged.append(TimeSegment(start=seg.start, end=seg.end, label=seg.label))

    return merged


def get_edit_summary(
    filepath: str,
    speech_segments: List[TimeSegment],
) -> dict:
    """
    Generate a summary of the edit (how much was cut, time saved, etc.).

    Args:
        filepath: Path to the original media file.
        speech_segments: The detected speech segments.

    Returns:
        Dictionary with edit statistics.
    """
    info = probe(filepath)
    original_duration = info.duration

    kept_duration = sum(s.duration for s in speech_segments)
    removed_duration = original_duration - kept_duration

    return {
        "original_duration": original_duration,
        "kept_duration": kept_duration,
        "removed_duration": removed_duration,
        "segments_count": len(speech_segments),
        "reduction_percent": (removed_duration / original_duration * 100) if original_duration > 0 else 0,
        "original_formatted": _format_time(original_duration),
        "kept_formatted": _format_time(kept_duration),
        "removed_formatted": _format_time(removed_duration),
    }


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"
