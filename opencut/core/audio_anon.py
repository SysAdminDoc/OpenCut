"""
OpenCut Audio Redaction & Speaker Anonymization

Diarize a target speaker, then replace their voice via pitch + formant
shift (or TTS resynthesis when available), preserving prosody for
anonymization.

Uses FFmpeg for audio manipulation and optional pydub/librosa for
advanced processing.
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SpeakerSegment:
    """A segment of audio attributed to a speaker."""
    speaker: str
    start: float
    end: float
    duration: float = 0.0


@dataclass
class AnonymizationResult:
    """Result of speaker anonymization."""
    output_path: str = ""
    segments_processed: int = 0
    method: str = "pitch_shift"
    target_speaker: str = ""
    total_anonymized_duration: float = 0.0


# ---------------------------------------------------------------------------
# Diarization helpers
# ---------------------------------------------------------------------------
def _diarize_simple(input_path: str, num_speakers: int = 2) -> List[SpeakerSegment]:
    """
    Simple energy-based speaker segmentation.

    Uses silence detection to split audio into speech segments and alternates
    speaker labels. For production use, pyannote.audio is recommended.
    """
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-i", input_path,
        "-af", "silencedetect=noise=-35dB:d=0.5",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)

    segments = []
    speech_start = 0.0
    speaker_idx = 0
    current_start = 0.0

    lines = result.stderr.split("\n")
    for line in lines:
        if "silence_start:" in line:
            try:
                silence_start = float(line.split("silence_start:")[1].strip().split()[0])
                if silence_start > current_start + 0.1:
                    segments.append(SpeakerSegment(
                        speaker=f"speaker_{speaker_idx % num_speakers}",
                        start=current_start,
                        end=silence_start,
                        duration=silence_start - current_start,
                    ))
                    speaker_idx += 1
            except (ValueError, IndexError):
                pass
        elif "silence_end:" in line:
            try:
                current_start = float(line.split("silence_end:")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    # Add final segment if there's remaining audio
    info = get_video_info(input_path)
    total_dur = info.get("duration", 0)
    if current_start < total_dur - 0.1:
        segments.append(SpeakerSegment(
            speaker=f"speaker_{speaker_idx % num_speakers}",
            start=current_start,
            end=total_dur,
            duration=total_dur - current_start,
        ))

    return segments


def _diarize_pyannote(input_path: str, num_speakers: int = 2) -> Optional[List[SpeakerSegment]]:
    """Attempt pyannote.audio-based diarization."""
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError:
        return None

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        diarization = pipeline(input_path, num_speakers=num_speakers)
    except Exception as e:
        logger.warning("pyannote diarization failed: %s", e)
        return None

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(SpeakerSegment(
            speaker=speaker,
            start=turn.start,
            end=turn.end,
            duration=turn.end - turn.start,
        ))

    return segments


# ---------------------------------------------------------------------------
# Voice anonymization
# ---------------------------------------------------------------------------
def _pitch_shift_segment(
    input_path: str, output_path_str: str,
    start: float, end: float,
    semitones: float = 4.0,
) -> None:
    """Apply pitch + formant shift to an audio segment using FFmpeg rubberband or asetrate."""
    ffmpeg = get_ffmpeg_path()
    duration = end - start

    # Use asetrate + atempo for simple pitch shift
    # Raising pitch by semitones: rate_factor = 2^(semitones/12)
    import math
    rate_factor = 2 ** (semitones / 12.0)
    tempo_factor = 1.0 / rate_factor  # compensate to keep duration

    af = f"asetrate=48000*{rate_factor},atempo={tempo_factor},aresample=48000"

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-t", str(duration),
        "-i", input_path,
        "-af", af,
        "-c:a", "pcm_s16le",
        output_path_str,
    ]
    run_ffmpeg(cmd)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def diarize_speakers(
    input_path: str,
    num_speakers: int = 2,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Diarize audio to identify speaker segments.

    Args:
        input_path: Source audio/video file.
        num_speakers: Expected number of speakers.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with segments list, speaker_count, method.
    """
    num_speakers = max(1, min(10, int(num_speakers)))

    if on_progress:
        on_progress(10, "Diarizing speakers...")

    # Try pyannote first, fall back to simple energy-based
    segments = _diarize_pyannote(input_path, num_speakers)
    method = "pyannote"

    if segments is None:
        segments = _diarize_simple(input_path, num_speakers)
        method = "energy"

    speakers = sorted(set(s.speaker for s in segments))

    if on_progress:
        on_progress(100, f"Found {len(segments)} segments from {len(speakers)} speakers")

    return {
        "segments": [asdict(s) for s in segments],
        "segment_count": len(segments),
        "speaker_count": len(speakers),
        "speakers": speakers,
        "method": method,
    }


def anonymize_speaker(
    input_path: str,
    target_speaker: str = "speaker_0",
    pitch_semitones: float = 4.0,
    num_speakers: int = 2,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Anonymize a target speaker's voice via pitch/formant shift.

    Args:
        input_path: Source audio/video file.
        target_speaker: Speaker label to anonymize.
        pitch_semitones: Pitch shift amount in semitones.
        num_speakers: Expected number of speakers for diarization.
        output_path_str: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, segments_processed, method, target_speaker,
        total_anonymized_duration.
    """
    pitch_semitones = max(-12.0, min(12.0, float(pitch_semitones)))

    if output_path_str is None:
        output_path_str = output_path(input_path, "anonymized")

    if on_progress:
        on_progress(5, "Diarizing speakers...")

    diar_result = diarize_speakers(
        input_path, num_speakers,
        on_progress=lambda p, m: on_progress(5 + int(p * 0.3), m) if on_progress else None,
    )

    segments = diar_result["segments"]
    target_segments = [s for s in segments if s["speaker"] == target_speaker]

    if not target_segments:
        if on_progress:
            on_progress(90, f"Speaker '{target_speaker}' not found, copying original...")
        ffmpeg = get_ffmpeg_path()
        cmd = [ffmpeg, "-i", input_path, "-c", "copy", "-y", output_path_str]
        run_ffmpeg(cmd)
        return {
            "output_path": output_path_str,
            "segments_processed": 0,
            "method": "none",
            "target_speaker": target_speaker,
            "total_anonymized_duration": 0.0,
        }

    if on_progress:
        on_progress(40, f"Anonymizing {len(target_segments)} segments for {target_speaker}...")

    info = get_video_info(input_path)
    duration = info["duration"]

    # Build FFmpeg filter that pitch-shifts only target segments
    # Use volume envelope to swap in shifted audio during target segments
    tmpdir = tempfile.mkdtemp(prefix="opencut_anon_")
    try:
        # Create pitch-shifted version of full audio
        shifted_path = os.path.join(tmpdir, "shifted.wav")
        import math
        rate_factor = 2 ** (pitch_semitones / 12.0)
        tempo_factor = 1.0 / rate_factor

        ffmpeg = get_ffmpeg_path()
        shift_cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-af", f"asetrate=48000*{rate_factor},atempo={tempo_factor},aresample=48000",
            "-vn", "-c:a", "pcm_s16le",
            shifted_path,
        ]
        run_ffmpeg(shift_cmd)

        if on_progress:
            on_progress(60, "Mixing anonymized audio...")

        # Build mix filter: use shifted audio during target segments, original otherwise
        enable_parts = []
        total_anon_dur = 0.0
        for s in target_segments:
            start = max(0, s["start"])
            end = min(duration, s["end"])
            enable_parts.append(f"between(t,{start},{end})")
            total_anon_dur += end - start

        enable_expr = "+".join(enable_parts)

        # Original: mute during target segments
        # Shifted: only audible during target segments
        fc = (
            f"[0:a]volume=enable='{enable_expr}':volume=0[orig];"
            f"[1:a]volume=enable='{enable_expr}':volume=1[shifted];"
            f"[orig][shifted]amix=inputs=2:duration=first[aout]"
        )

        mix_cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-i", shifted_path,
            "-filter_complex", fc,
            "-map", "0:v?", "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            output_path_str,
        ]
        run_ffmpeg(mix_cmd)

        if on_progress:
            on_progress(100, "Speaker anonymization complete")

        return {
            "output_path": output_path_str,
            "segments_processed": len(target_segments),
            "method": "pitch_shift",
            "target_speaker": target_speaker,
            "total_anonymized_duration": round(total_anon_dur, 3),
        }
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
