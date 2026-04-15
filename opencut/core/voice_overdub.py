"""
Voice Overdub — Fix spoken mistakes by typing correct words.

Pipeline: identify replacement segment via transcript timestamps, clone
speaker voice from surrounding context, generate corrected audio via TTS,
time-stretch to match original duration, and crossfade at boundaries.
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_ffmpeg_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# Crossfade duration in seconds at segment boundaries
DEFAULT_CROSSFADE_MS = 50
# Voice cloning context: seconds of speaker audio to extract
VOICE_CLONE_MIN_SECONDS = 10
VOICE_CLONE_MAX_SECONDS = 30
# Supported TTS backends
TTS_BACKENDS = ("edge_tts", "external_api")


@dataclass
class ReplacementSegment:
    """A single audio segment to replace."""
    start_time: float
    end_time: float
    original_text: str = ""
    replacement_text: str = ""
    generated_audio_path: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> dict:
        return {
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "original_text": self.original_text,
            "replacement_text": self.replacement_text,
            "duration": round(self.duration, 4),
        }


@dataclass
class VoiceProfile:
    """Speaker voice profile extracted from audio context."""
    speaker_id: str = ""
    source_path: str = ""
    duration: float = 0.0
    sample_rate: int = 24000
    reference_audio_path: str = ""

    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "source_path": self.source_path,
            "duration": round(self.duration, 4),
            "sample_rate": self.sample_rate,
            "reference_audio_path": self.reference_audio_path,
        }


@dataclass
class OverdubResult:
    """Result of voice overdub operation."""
    output_path: str = ""
    replaced_segments: List[Dict] = field(default_factory=list)
    original_duration: float = 0.0
    new_duration: float = 0.0
    voice_profile: Optional[Dict] = None

    def to_dict(self) -> dict:
        return {
            "output_path": self.output_path,
            "replaced_segments": self.replaced_segments,
            "original_duration": round(self.original_duration, 4),
            "new_duration": round(self.new_duration, 4),
            "voice_profile": self.voice_profile,
        }


# ---------------------------------------------------------------------------
# Extract speaker audio for voice cloning
# ---------------------------------------------------------------------------
def extract_speaker_audio(
    input_path: str,
    segments: List[Dict],
    exclude_start: float = 0.0,
    exclude_end: float = 0.0,
    min_seconds: float = VOICE_CLONE_MIN_SECONDS,
    max_seconds: float = VOICE_CLONE_MAX_SECONDS,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> VoiceProfile:
    """Extract speaker audio context for voice cloning.

    Collects audio from transcript segments surrounding the replacement region,
    avoiding the segment being replaced.

    Args:
        input_path: Path to source audio/video file.
        segments: Transcript segments [{"start", "end", "speaker"}].
        exclude_start: Start time of region to exclude (the replacement).
        exclude_end: End time of region to exclude.
        min_seconds: Minimum speaker audio to collect.
        max_seconds: Maximum speaker audio to collect.
        output_dir: Output directory for reference audio.
        on_progress: Progress callback(pct).

    Returns:
        VoiceProfile with reference audio path.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if on_progress:
        on_progress(10)

    # Collect usable segments (outside exclusion zone)
    usable: List[Dict] = []
    for seg in segments:
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", 0))
        if seg_end <= exclude_start or seg_start >= exclude_end:
            usable.append({"start": seg_start, "end": seg_end})

    # Sort by proximity to exclusion zone (prefer nearby context)
    mid_exclude = (exclude_start + exclude_end) / 2
    usable.sort(key=lambda s: abs((s["start"] + s["end"]) / 2 - mid_exclude))

    if on_progress:
        on_progress(25)

    # Collect up to max_seconds of audio
    collected_dur = 0.0
    selected: List[Dict] = []
    for seg in usable:
        if collected_dur >= max_seconds:
            break
        dur = seg["end"] - seg["start"]
        selected.append(seg)
        collected_dur += dur

    if collected_dur < min_seconds and usable:
        # If not enough context, lower the bar and take what we have
        logger.warning(
            "Only %.1fs of speaker audio found (need %.1fs); using available context",
            collected_dur, min_seconds,
        )

    if not selected:
        raise ValueError("No usable speaker audio segments found for voice cloning")

    if on_progress:
        on_progress(40)

    # Extract and concatenate selected audio segments
    out_dir = output_dir or os.path.dirname(input_path)
    tmp_dir = tempfile.mkdtemp(prefix="opencut_vc_")
    segment_files: List[str] = []

    for i, seg in enumerate(selected):
        seg_path = os.path.join(tmp_dir, f"seg_{i:04d}.wav")
        cmd = (FFmpegCmd()
               .input(input_path, ss=str(seg["start"]), to=str(seg["end"]))
               .no_video()
               .audio_codec("pcm_s16le")
               .option("ar", "24000")
               .option("ac", "1")
               .output(seg_path)
               .build())
        run_ffmpeg(cmd)
        segment_files.append(seg_path)

        if on_progress:
            pct = 40 + int((i / max(1, len(selected))) * 30)
            on_progress(pct)

    # Concatenate via FFmpeg concat demuxer
    ref_audio_path = os.path.join(out_dir, "voice_reference.wav")
    concat_list = os.path.join(tmp_dir, "concat.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for sp in segment_files:
            f.write(f"file '{sp}'\n")

    cmd = (FFmpegCmd()
           .option("f", "concat")
           .option("safe", "0")
           .input(concat_list)
           .audio_codec("pcm_s16le")
           .output(ref_audio_path)
           .build())
    run_ffmpeg(cmd)

    # Clean up temp segment files
    for sp in segment_files:
        try:
            os.unlink(sp)
        except OSError:
            pass
    try:
        os.unlink(concat_list)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    if on_progress:
        on_progress(100)

    return VoiceProfile(
        speaker_id="speaker_0",
        source_path=input_path,
        duration=collected_dur,
        sample_rate=24000,
        reference_audio_path=ref_audio_path,
    )


# ---------------------------------------------------------------------------
# TTS Generation
# ---------------------------------------------------------------------------
def _generate_tts_audio(
    text: str,
    output_path_wav: str,
    voice_profile: Optional[VoiceProfile] = None,
    tts_endpoint: str = "",
    voice_name: str = "en-US-GuyNeural",
    on_progress: Optional[Callable] = None,
) -> str:
    """Generate speech audio from text via TTS.

    Uses external API endpoint if provided, otherwise falls back to edge_tts.

    Args:
        text: Text to speak.
        output_path_wav: Output WAV file path.
        voice_profile: Optional voice profile for cloning context.
        tts_endpoint: External TTS API endpoint URL.
        voice_name: edge_tts voice name (fallback).
        on_progress: Progress callback(pct).

    Returns:
        Path to generated WAV file.
    """
    if tts_endpoint:
        return _call_external_tts(text, output_path_wav, tts_endpoint, voice_profile)

    # Fallback to edge_tts
    return _generate_edge_tts(text, output_path_wav, voice_name, on_progress)


def _call_external_tts(
    text: str,
    output_wav: str,
    endpoint: str,
    voice_profile: Optional[VoiceProfile] = None,
) -> str:
    """Call external TTS API endpoint."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "text": text,
        "voice_reference": voice_profile.reference_audio_path if voice_profile else "",
        "sample_rate": 24000,
    }).encode()

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            audio_data = resp.read()
        with open(output_wav, "wb") as f:
            f.write(audio_data)
        return output_wav
    except urllib.error.URLError as e:
        logger.warning("External TTS endpoint failed: %s — falling back to edge_tts", e)
        return _generate_edge_tts(text, output_wav, "en-US-GuyNeural")


def _generate_edge_tts(
    text: str,
    output_wav: str,
    voice_name: str = "en-US-GuyNeural",
    on_progress: Optional[Callable] = None,
) -> str:
    """Generate speech via edge_tts (Microsoft Edge TTS)."""
    ensure_package("edge_tts")
    import asyncio

    import edge_tts  # noqa: F401

    tmp_mp3 = output_wav.rsplit(".", 1)[0] + "_tts.mp3"

    async def _run():
        communicate = edge_tts.Communicate(text, voice_name)
        await communicate.save(tmp_mp3)

    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_run())
        loop.close()
    except Exception as e:
        raise RuntimeError(f"edge_tts generation failed: {e}") from e

    if on_progress:
        on_progress(80)

    # Convert MP3 to WAV for consistent processing
    cmd = (FFmpegCmd()
           .input(tmp_mp3)
           .audio_codec("pcm_s16le")
           .option("ar", "24000")
           .option("ac", "1")
           .output(output_wav)
           .build())
    run_ffmpeg(cmd)

    try:
        os.unlink(tmp_mp3)
    except OSError:
        pass

    return output_wav


# ---------------------------------------------------------------------------
# Time-stretch audio to match duration
# ---------------------------------------------------------------------------
def _time_stretch_audio(
    input_wav: str,
    target_duration: float,
    output_wav: str,
) -> str:
    """Time-stretch audio to match target duration using FFmpeg atempo.

    Args:
        input_wav: Input WAV file.
        target_duration: Desired duration in seconds.
        output_wav: Output WAV file.

    Returns:
        Path to time-stretched WAV.
    """
    # Get current duration
    info_cmd = [
        get_ffmpeg_path(), "-i", input_wav, "-f", "null", "-",
    ]
    result = subprocess.run(info_cmd, capture_output=True, timeout=30)
    stderr = result.stderr.decode(errors="replace")

    # Parse duration from FFmpeg output
    current_dur = 0.0
    for line in stderr.split("\n"):
        if "Duration:" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip()
            time_parts = parts.split(":")
            if len(time_parts) == 3:
                current_dur = (
                    float(time_parts[0]) * 3600
                    + float(time_parts[1]) * 60
                    + float(time_parts[2])
                )
            break

    if current_dur <= 0:
        # Fallback: just copy
        cmd = (FFmpegCmd()
               .input(input_wav)
               .audio_codec("pcm_s16le")
               .output(output_wav)
               .build())
        run_ffmpeg(cmd)
        return output_wav

    ratio = current_dur / target_duration
    if abs(ratio - 1.0) < 0.02:
        # Close enough, just copy
        cmd = (FFmpegCmd()
               .input(input_wav)
               .audio_codec("pcm_s16le")
               .output(output_wav)
               .build())
        run_ffmpeg(cmd)
        return output_wav

    # atempo filter accepts 0.5 - 100.0; chain if needed
    filters = _build_atempo_chain(ratio)
    cmd = (FFmpegCmd()
           .input(input_wav)
           .audio_filter(",".join(filters))
           .audio_codec("pcm_s16le")
           .option("ar", "24000")
           .output(output_wav)
           .build())
    run_ffmpeg(cmd)
    return output_wav


def _build_atempo_chain(ratio: float) -> List[str]:
    """Build chained atempo filters for ratios outside 0.5-100.0 range."""
    ratio = max(0.1, min(100.0, ratio))
    filters: List[str] = []

    if ratio < 0.5:
        while ratio < 0.5:
            filters.append("atempo=0.5")
            ratio /= 0.5
        filters.append(f"atempo={ratio:.6f}")
    elif ratio > 100.0:
        while ratio > 100.0:
            filters.append("atempo=100.0")
            ratio /= 100.0
        filters.append(f"atempo={ratio:.6f}")
    else:
        filters.append(f"atempo={ratio:.6f}")

    return filters


# ---------------------------------------------------------------------------
# Crossfade + Mix
# ---------------------------------------------------------------------------
def _crossfade_mix(
    original_audio: str,
    replacement_audio: str,
    start_time: float,
    end_time: float,
    crossfade_ms: int,
    output_path_wav: str,
) -> str:
    """Mix replacement audio into original with crossfade at boundaries.

    Args:
        original_audio: Full original audio WAV.
        replacement_audio: Replacement segment WAV (already time-stretched).
        start_time: Start time of replacement in original.
        end_time: End time of replacement in original.
        crossfade_ms: Crossfade duration in ms.
        output_path_wav: Output WAV path.

    Returns:
        Path to mixed WAV.
    """
    cf_sec = crossfade_ms / 1000.0

    # Build filter: take original, duck the replacement region, overlay new audio
    # [0] = original, [1] = replacement
    cf_start = max(0.0, start_time - cf_sec)
    cf_end = end_time + cf_sec

    fc = (
        f"[0]volume=enable='between(t,{cf_start:.4f},{cf_end:.4f})':"
        f"volume=0:eval=frame[ducked];"
        f"[1]adelay={int(start_time * 1000)}|{int(start_time * 1000)}[delayed];"
        f"[ducked][delayed]amix=inputs=2:duration=first:dropout_transition=0[out]"
    )

    cmd = (FFmpegCmd()
           .input(original_audio)
           .input(replacement_audio)
           .filter_complex(fc, maps=["[out]"])
           .audio_codec("pcm_s16le")
           .option("ar", "24000")
           .output(output_path_wav)
           .build())
    run_ffmpeg(cmd)
    return output_path_wav


# ---------------------------------------------------------------------------
# Main overdub pipeline
# ---------------------------------------------------------------------------
def overdub(
    input_path: str,
    replacements: List[Dict],
    transcript_segments: Optional[List[Dict]] = None,
    tts_endpoint: str = "",
    voice_name: str = "en-US-GuyNeural",
    crossfade_ms: int = DEFAULT_CROSSFADE_MS,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> OverdubResult:
    """Replace audio segments with AI-generated corrected speech.

    Args:
        input_path: Path to source audio/video file.
        replacements: List of dicts with keys:
            - start_time: float
            - end_time: float
            - replacement_text: str
            - original_text: str (optional)
        transcript_segments: Full transcript segments for voice cloning context.
        tts_endpoint: External TTS API endpoint (empty = edge_tts fallback).
        voice_name: edge_tts voice (fallback).
        crossfade_ms: Crossfade duration at boundaries.
        output_dir: Output directory.
        on_progress: Progress callback(pct).

    Returns:
        OverdubResult with output path and statistics.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not replacements:
        raise ValueError("No replacement segments provided")

    if on_progress:
        on_progress(2)

    info = get_video_info(input_path)
    original_duration = info.get("duration", 0.0)
    has_video = info.get("width", 0) > 0

    # Sort replacements by start time
    sorted_reps = sorted(replacements, key=lambda r: float(r.get("start_time", 0)))

    # Step 1: Extract full audio from source
    tmp_dir = tempfile.mkdtemp(prefix="opencut_od_")
    original_audio = os.path.join(tmp_dir, "original.wav")
    cmd = (FFmpegCmd()
           .input(input_path)
           .no_video()
           .audio_codec("pcm_s16le")
           .option("ar", "24000")
           .option("ac", "1")
           .output(original_audio)
           .build())
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(10)

    # Step 2: Clone speaker voice from context (if transcript available)
    voice_profile = None
    if transcript_segments and len(sorted_reps) > 0:
        try:
            first_rep = sorted_reps[0]
            voice_profile = extract_speaker_audio(
                input_path,
                transcript_segments,
                exclude_start=float(first_rep.get("start_time", 0)),
                exclude_end=float(first_rep.get("end_time", 0)),
                output_dir=tmp_dir,
                on_progress=lambda p: on_progress(10 + int(p * 0.1)) if on_progress else None,
            )
        except Exception as e:
            logger.warning("Voice cloning failed, using default TTS voice: %s", e)

    if on_progress:
        on_progress(20)

    # Step 3: Generate replacement audio for each segment
    result_segments: List[Dict] = []
    current_audio = original_audio
    total_reps = len(sorted_reps)

    for i, rep in enumerate(sorted_reps):
        start_t = float(rep.get("start_time", 0))
        end_t = float(rep.get("end_time", 0))
        rep_text = rep.get("replacement_text", "")
        orig_text = rep.get("original_text", "")

        if not rep_text:
            logger.warning("Skipping empty replacement at %.2f-%.2f", start_t, end_t)
            continue

        seg_duration = end_t - start_t
        if seg_duration <= 0:
            logger.warning("Skipping zero-duration segment at %.2f", start_t)
            continue

        # Generate TTS audio
        raw_tts = os.path.join(tmp_dir, f"tts_raw_{i}.wav")
        _generate_tts_audio(
            rep_text, raw_tts,
            voice_profile=voice_profile,
            tts_endpoint=tts_endpoint,
            voice_name=voice_name,
            on_progress=lambda p: (
                on_progress(20 + int((i / total_reps) * 40 + (p / 100) * (40 / total_reps)))
                if on_progress else None
            ),
        )

        # Time-stretch to match original segment duration
        stretched_tts = os.path.join(tmp_dir, f"tts_stretched_{i}.wav")
        _time_stretch_audio(raw_tts, seg_duration, stretched_tts)

        # Crossfade mix into current audio
        mixed_audio = os.path.join(tmp_dir, f"mixed_{i}.wav")
        _crossfade_mix(
            current_audio, stretched_tts,
            start_t, end_t, crossfade_ms, mixed_audio,
        )
        current_audio = mixed_audio

        result_segments.append(ReplacementSegment(
            start_time=start_t,
            end_time=end_t,
            original_text=orig_text,
            replacement_text=rep_text,
        ).to_dict())

        if on_progress:
            pct = 20 + int(((i + 1) / total_reps) * 60)
            on_progress(pct)

    if on_progress:
        on_progress(85)

    # Step 4: Produce final output
    out_path = output_path(input_path, "overdub", output_dir)

    if has_video:
        # Mux corrected audio with original video
        cmd = (FFmpegCmd()
               .input(input_path)
               .input(current_audio)
               .map("0:v", "1:a")
               .video_codec("copy")
               .audio_codec("aac", bitrate="192k")
               .faststart()
               .option("shortest")
               .output(out_path)
               .build())
        run_ffmpeg(cmd)
    else:
        # Audio-only: convert to output format
        ext = os.path.splitext(input_path)[1].lower()
        if ext in (".mp3", ".aac", ".m4a"):
            cmd = (FFmpegCmd()
                   .input(current_audio)
                   .audio_codec("libmp3lame" if ext == ".mp3" else "aac", bitrate="192k")
                   .output(out_path)
                   .build())
            run_ffmpeg(cmd)
        else:
            cmd = (FFmpegCmd()
                   .input(current_audio)
                   .audio_codec("pcm_s16le")
                   .output(out_path)
                   .build())
            run_ffmpeg(cmd)

    # Clean up temp files
    _cleanup_temp_dir(tmp_dir)

    if on_progress:
        on_progress(100)

    return OverdubResult(
        output_path=out_path,
        replaced_segments=result_segments,
        original_duration=original_duration,
        new_duration=original_duration,  # overdub preserves duration
        voice_profile=voice_profile.to_dict() if voice_profile else None,
    )


def _cleanup_temp_dir(tmp_dir: str):
    """Remove temp directory and contents."""
    try:
        for f in os.listdir(tmp_dir):
            try:
                os.unlink(os.path.join(tmp_dir, f))
            except OSError:
                pass
        os.rmdir(tmp_dir)
    except OSError:
        logger.debug("Failed to clean up temp dir: %s", tmp_dir)
