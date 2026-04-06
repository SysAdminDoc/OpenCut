"""
Silence detection engine using FFmpeg's silencedetect filter.

Detects silent segments in audio/video files and returns time intervals
for both silent and non-silent (speech) regions.
"""

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path

from ..utils.config import SilenceConfig
from ..utils.media import probe

logger = logging.getLogger("opencut")


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
    file_duration: float = 0.0,
) -> List[TimeSegment]:
    """
    Detect silent segments in an audio/video file using FFmpeg.

    Args:
        filepath: Path to the media file.
        threshold_db: Noise floor in dB. Audio quieter than this is silence.
                      Typical values: -30 (aggressive) to -50 (conservative).
        min_duration: Minimum silence duration in seconds to detect.
        file_duration: Pre-probed file duration to avoid redundant ffprobe calls.
                       If 0, will probe the file.

    Returns:
        List of TimeSegment objects representing silent regions.
    """
    # Coerce to float for safe interpolation into FFmpeg filter
    threshold_db = float(threshold_db)
    min_duration = float(min_duration)

    cmd = [
        get_ffmpeg_path(),
        "-hide_banner",
        "-i", filepath,
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
        "-vn",       # ignore video (faster)
        "-sn",       # ignore subtitles
        "-f", "null",
        "-",
    ]

    # Scale timeout: 10 min base + 3x file duration (long podcasts need more time)
    if file_duration > 0:
        timeout = max(600, int(file_duration * 3) + 120)
    else:
        try:
            info = probe(filepath)
            file_duration = info.duration
            timeout = max(600, int(file_duration * 3) + 120)
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

    if result.returncode != 0:
        logger.warning("FFmpeg silencedetect failed (rc=%d) for %s", result.returncode, filepath)

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
            # Silence extends to end of file — use pre-probed duration
            if file_duration <= 0:
                try:
                    info = probe(filepath)
                    file_duration = info.duration
                except Exception:
                    # If probe fails, skip this trailing silence
                    continue
            end = file_duration

        if end > start:
            silences.append(TimeSegment(start=start, end=end, label="silence"))

    return silences


def detect_silences_vad(
    filepath: str,
    min_duration: float = 0.5,
    file_duration: float = 0.0,
) -> List[TimeSegment]:
    """
    Detect silent segments using Silero VAD (neural voice activity detection).

    Silero VAD is far more accurate than energy-based detection, especially
    in noisy environments. Uses a 1.8MB ONNX model, <1ms per 30ms chunk.

    Args:
        filepath: Path to the media file.
        min_duration: Minimum silence duration in seconds to report.
        file_duration: Pre-probed file duration.

    Returns:
        List of TimeSegment objects representing silent regions.

    Raises:
        ImportError: If torch/silero is not installed.
    """

    try:
        import torch
    except ImportError:
        raise ImportError(
            "Silero VAD requires PyTorch. Install with: pip install torch"
        )

    # Load Silero VAD v6 model (cached after first load)
    # ONNX mode avoids GPU memory overhead for inference
    use_onnx = True
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        use_onnx = False

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        onnx=use_onnx,
        force_reload=False,
        trust_repo=True,
    )
    (get_speech_timestamps, _, read_audio, _, _) = utils

    tmp_wav = None
    try:
        # Silero VAD requires 16kHz mono audio — read_audio handles conversion
        try:
            wav = read_audio(filepath, sampling_rate=16000)
        except Exception:
            # Fallback: extract audio with FFmpeg first, then load
            import os
            import tempfile
            fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="opencut_vad_")
            os.close(fd)
            _extract_audio_wav(filepath, tmp_wav)
            if not os.path.isfile(tmp_wav):
                raise RuntimeError(f"Audio extraction failed for '{filepath}' — FFmpeg produced no output")
            wav = read_audio(tmp_wav, sampling_rate=16000)

        # Get speech timestamps from Silero VAD
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=16000,
            min_silence_duration_ms=int(min_duration * 1000),
            min_speech_duration_ms=100,
            return_seconds=True,
        )

        # Determine total duration
        if file_duration <= 0:
            file_duration = len(wav) / 16000.0

        # Invert speech timestamps to get silence segments
        silences = []
        prev_end = 0.0

        for ts in speech_timestamps:
            # Handle both dict format {"start": x, "end": y} and tuple format (start, end)
            if isinstance(ts, dict):
                speech_start = float(ts.get("start", 0))
                speech_end = float(ts.get("end", 0))
            elif isinstance(ts, (list, tuple)) and len(ts) >= 2:
                speech_start = float(ts[0])
                speech_end = float(ts[1])
            else:
                continue

            if speech_start > prev_end + min_duration:
                silences.append(TimeSegment(
                    start=prev_end,
                    end=speech_start,
                    label="silence",
                ))
            prev_end = speech_end

        # Trailing silence
        if file_duration > prev_end + min_duration:
            silences.append(TimeSegment(
                start=prev_end,
                end=file_duration,
                label="silence",
            ))

        return silences
    finally:
        # Clean up temp file
        if tmp_wav is not None:
            try:
                import os
                os.remove(tmp_wav)
            except OSError:
                pass
        # Free GPU memory
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _extract_audio_wav(input_path: str, output_path: str) -> None:
    """Extract audio from a media file as 16kHz mono WAV for VAD processing."""
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found. Install FFmpeg: https://ffmpeg.org/download.html")

    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Audio extraction timed out for '{input_path}'")

    if result.returncode != 0:
        stderr = result.stderr[-300:] if result.stderr else "unknown error"
        raise RuntimeError(f"Audio extraction failed: {stderr}")


def detect_speech(
    filepath: str,
    config: Optional[SilenceConfig] = None,
    file_duration: float = 0.0,
    method: str = "energy",
) -> List[TimeSegment]:
    """
    Detect speech segments by inverting silence detection results.

    This is the primary function for silence removal workflows.
    Returns non-silent time segments with padding applied.

    Args:
        filepath: Path to the media file.
        config: Silence detection configuration. Uses defaults if None.
        file_duration: Pre-probed file duration to avoid redundant ffprobe calls.
        method: Detection method — "energy" (FFmpeg threshold), "vad" (Silero VAD),
                or "auto" (try VAD first, fall back to energy).

    Returns:
        List of TimeSegment objects representing speech regions.
    """
    if config is None:
        config = SilenceConfig()

    # Get file duration (reuse if already probed)
    if file_duration > 0:
        total_duration = file_duration
    else:
        info = probe(filepath)
        total_duration = info.duration

    if total_duration <= 0:
        raise ValueError(f"Could not determine duration of '{filepath}'")

    # Choose detection method
    if method == "vad":
        silences = detect_silences_vad(
            filepath,
            min_duration=config.min_duration,
            file_duration=total_duration,
        )
    elif method == "auto":
        try:
            silences = detect_silences_vad(
                filepath,
                min_duration=config.min_duration,
                file_duration=total_duration,
            )
            logger.info("Using Silero VAD for silence detection")
        except (ImportError, Exception) as e:
            logger.warning("Silero VAD unavailable (%s), falling back to energy-based detection", e)
            silences = detect_silences(
                filepath,
                threshold_db=config.threshold_db,
                min_duration=config.min_duration,
                file_duration=total_duration,
            )
    else:
        # Default: energy-based (FFmpeg silencedetect)
        silences = detect_silences(
            filepath,
            threshold_db=config.threshold_db,
            min_duration=config.min_duration,
            file_duration=total_duration,
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
    file_duration: float = 0.0,
) -> dict:
    """
    Generate a summary of the edit (how much was cut, time saved, etc.).

    Args:
        filepath: Path to the original media file.
        speech_segments: The detected speech segments.
        file_duration: Pre-probed duration to avoid redundant ffprobe calls.

    Returns:
        Dictionary with edit statistics.
    """
    if file_duration > 0:
        original_duration = file_duration
    else:
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
    """Format seconds as HH:MM:SS.mmm (always includes hours for consistent parsing)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


# ---------------------------------------------------------------------------
# Speed-Up-Silence Mode
# ---------------------------------------------------------------------------
def speed_up_silences(
    filepath: str,
    speed_factor: float = 4.0,
    threshold_db: float = -30.0,
    min_duration: float = 0.5,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Speed up silent segments instead of removing them.

    Detects silences and applies a speed multiplier to those regions,
    keeping context while dramatically shortening pauses. Produces
    a single output file with speech at normal speed and silences
    sped up.

    Args:
        filepath: Path to the media file.
        speed_factor: Speed multiplier for silent segments (2.0-8.0).
        threshold_db: Silence threshold in dB.
        min_duration: Minimum silence duration to speed up.
        output_path: Explicit output path. Auto-generated if None.
        output_dir: Output directory (used if output_path is None).
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path, original_duration, new_duration, reduction_percent.
    """
    import os

    if speed_factor < 1.5:
        speed_factor = 1.5
    elif speed_factor > 8.0:
        speed_factor = 8.0

    if on_progress:
        on_progress(5, "Detecting silences...")

    # Probe duration
    info = probe(filepath)
    total_duration = info.duration

    if total_duration <= 0:
        raise ValueError(f"Could not determine duration of '{filepath}'")

    # Detect silences
    silences = detect_silences(
        filepath,
        threshold_db=threshold_db,
        min_duration=min_duration,
        file_duration=total_duration,
    )

    if not silences:
        # No silences — just copy the file
        if output_path is None:
            base = os.path.splitext(os.path.basename(filepath))[0]
            ext = os.path.splitext(filepath)[1] or ".mp4"
            directory = output_dir or os.path.dirname(filepath)
            output_path = os.path.join(directory, f"{base}_speedsilence{ext}")

        import shutil
        shutil.copy2(filepath, output_path)
        return {
            "output_path": output_path,
            "original_duration": total_duration,
            "new_duration": total_duration,
            "reduction_percent": 0.0,
            "silences_found": 0,
        }

    if on_progress:
        on_progress(20, f"Found {len(silences)} silent segments, building filter...")

    # Build all segments (speech + silence) in order
    all_segments = []
    pos = 0.0

    for silence in silences:
        # Speech segment before this silence
        if silence.start > pos:
            all_segments.append(("speech", pos, silence.start))
        # Silent segment
        all_segments.append(("silence", silence.start, silence.end))
        pos = silence.end

    # Final speech segment after last silence
    if pos < total_duration:
        all_segments.append(("speech", pos, total_duration))

    # Detect if input has video (vs audio-only files like .mp3, .wav)
    has_video = info.has_video if hasattr(info, "has_video") else not filepath.lower().endswith(
        (".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus")
    )

    # Build FFmpeg filter_complex with concat
    # Each segment gets its own trim + setpts/atempo chain
    filter_parts = []
    concat_inputs_v = []
    concat_inputs_a = []

    for i, (seg_type, start, end) in enumerate(all_segments):
        # Trim the segment
        if has_video:
            filter_parts.append(
                f"[0:v]trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS[v{i}];"
            )
        filter_parts.append(
            f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{i}];"
        )

        if seg_type == "silence":
            if has_video:
                # Speed up the video segment
                filter_parts.append(
                    f"[v{i}]setpts=PTS/{speed_factor:.2f}[vs{i}];"
                )
                concat_inputs_v.append(f"[vs{i}]")
            # Speed up audio — atempo max is 2.0, so chain for higher speeds
            atempo_chain = _build_atempo_chain(speed_factor, f"a{i}", f"as{i}")
            filter_parts.append(atempo_chain)
            concat_inputs_a.append(f"[as{i}]")
        else:
            if has_video:
                concat_inputs_v.append(f"[v{i}]")
            concat_inputs_a.append(f"[a{i}]")

    # Concat all segments
    n = len(all_segments)
    if has_video:
        v_inputs = "".join(concat_inputs_v)
        a_inputs = "".join(concat_inputs_a)
        filter_parts.append(
            f"{v_inputs}{a_inputs}concat=n={n}:v=1:a=1[outv][outa]"
        )
    else:
        a_inputs = "".join(concat_inputs_a)
        filter_parts.append(
            f"{a_inputs}concat=n={n}:v=0:a=1[outa]"
        )

    filter_complex = "".join(filter_parts)

    # Build output path
    if output_path is None:
        base = os.path.splitext(os.path.basename(filepath))[0]
        ext = os.path.splitext(filepath)[1] or ".mp4"
        directory = output_dir or os.path.dirname(filepath)
        output_path = os.path.join(directory, f"{base}_speedsilence{ext}")

    if on_progress:
        on_progress(30, "Rendering output...")

    # Run FFmpeg
    if has_video:
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", filepath,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", filepath,
            "-filter_complex", filter_complex,
            "-map", "[outa]",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(600, int(total_duration * 5)),
        )
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Install FFmpeg: https://ffmpeg.org/download.html")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg timed out processing '{filepath}'")

    if result.returncode != 0:
        stderr = result.stderr[-500:] if result.stderr else "unknown error"
        raise RuntimeError(f"FFmpeg failed: {stderr}")

    # Calculate new duration
    new_info = probe(output_path)
    new_duration = new_info.duration if new_info.duration > 0 else total_duration
    reduction = ((total_duration - new_duration) / total_duration * 100) if total_duration > 0 else 0

    if on_progress:
        on_progress(100, f"Done: {_format_time(total_duration)} -> {_format_time(new_duration)} ({reduction:.0f}% shorter)")

    return {
        "output_path": output_path,
        "original_duration": total_duration,
        "new_duration": new_duration,
        "reduction_percent": round(reduction, 1),
        "silences_found": len(silences),
        "speed_factor": speed_factor,
    }


def _build_atempo_chain(speed: float, input_label: str, output_label: str) -> str:
    """
    Build chained atempo filters for speeds > 2.0.

    FFmpeg's atempo filter only supports 0.5-2.0 per instance,
    so we chain multiple: 4x = atempo=2.0,atempo=2.0
    """
    tempos = []
    remaining = speed
    while remaining > 2.0:
        tempos.append(2.0)
        remaining /= 2.0
    # Clamp to FFmpeg atempo range (0.5-100.0)
    remaining = max(0.5, min(100.0, round(remaining, 4)))
    tempos.append(remaining)

    chain = ",".join(f"atempo={t}" for t in tempos)
    return f"[{input_label}]{chain}[{output_label}];"
