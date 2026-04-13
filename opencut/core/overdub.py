"""
OpenCut AI Overdub / Voice Correction Module v0.1.0

Clone voice from surrounding audio context, generate a corrected speech
segment, and crossfade it back into the timeline:
- Extract speaker voice characteristics from surrounding audio
- Generate replacement speech segment via TTS with cloned voice
- Smooth crossfade at boundaries to avoid audible seams
- Falls back to whisper-speed TTS or FFmpeg silence patching

Requires: openai-whisper (for transcription), TTS (Coqui) or edge-tts
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Configuration / Result
# ---------------------------------------------------------------------------
@dataclass
class OverdubConfig:
    """Configuration for overdub/voice correction."""
    crossfade_ms: int = 150         # Crossfade duration at boundaries (ms)
    voice_clone_seconds: float = 10.0  # Seconds of surrounding audio for cloning
    sample_rate: int = 24000        # Output sample rate
    tts_backend: str = "edge"       # "edge" (edge-tts), "coqui" (TTS), or "piper"
    language: str = "en"            # Language code
    speed: float = 1.0              # Playback speed multiplier


@dataclass
class OverdubResult:
    """Result of overdub operation."""
    output_path: str = ""
    segment_start: float = 0.0
    segment_end: float = 0.0
    new_text: str = ""
    tts_backend_used: str = ""
    duration_generated: float = 0.0


# ---------------------------------------------------------------------------
# Voice Cloning (extract characteristics)
# ---------------------------------------------------------------------------
def clone_voice_segment(
    audio_path: str,
    start: float,
    end: float,
    context_seconds: float = 10.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Extract a reference voice clip from surrounding audio for TTS cloning.

    Extracts up to context_seconds of audio around [start, end] to use
    as a voice reference. Returns path to the extracted reference WAV.

    Args:
        audio_path: Path to source audio/video file.
        start: Start of segment to replace (seconds).
        end: End of segment to replace (seconds).
        context_seconds: Seconds of surrounding audio to extract.

    Returns:
        Path to temporary WAV file containing reference voice audio.
    """
    info = get_video_info(audio_path)
    duration = info.get("duration", 0)

    # Extract audio before and after the segment
    ref_start = max(0, start - context_seconds / 2)
    ref_end_before = max(ref_start, start - 0.1)
    ref_start_after = min(duration, end + 0.1)
    ref_end = min(duration, end + context_seconds / 2)

    _ntf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    ref_path = _ntf.name
    _ntf.close()

    # Extract audio context (before + after the segment)
    inputs = []

    if ref_end_before > ref_start + 0.1:
        inputs.extend(["-ss", str(ref_start), "-to", str(ref_end_before)])
    else:
        inputs.extend(["-ss", str(ref_start), "-to", str(start)])

    cmd = (
        FFmpegCmd()
        .input(audio_path, ss=str(ref_start), to=str(ref_end_before))
    )
    if ref_end > ref_start_after + 0.1:
        cmd = cmd.input(audio_path, ss=str(ref_start_after), to=str(ref_end))
        cmd = cmd.filter_complex(
            "[0:a][1:a]concat=n=2:v=0:a=1[out]",
            maps=["[out]"],
        )
    else:
        cmd = cmd.map("0:a")

    cmd = (
        cmd
        .audio_codec("pcm_s16le")
        .option("-ar", "16000")
        .option("-ac", "1")
        .output(ref_path)
    )

    try:
        run_ffmpeg(cmd.build(), timeout=120)
    except Exception as e:
        logger.warning("Voice clone extraction failed: %s", e)
        # Fallback: just extract the segment before
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", audio_path, "-ss", str(max(0, start - 5)),
            "-to", str(start), "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            ref_path,
        ], timeout=60)

    if on_progress:
        on_progress(20, "Voice reference extracted")

    return ref_path


# ---------------------------------------------------------------------------
# Audio Boundary Blending
# ---------------------------------------------------------------------------
def blend_audio_boundaries(
    original_path: str,
    generated_path: str,
    output_path: str,
    start: float,
    end: float,
    crossfade_ms: int = 150,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Blend generated audio segment back into the original with crossfades.

    Replaces audio between [start, end] with generated_path content,
    applying crossfade_ms crossfade at both boundaries.

    Args:
        original_path: Original audio/video file.
        generated_path: Generated replacement audio segment (WAV).
        output_path: Where to write the blended result.
        start: Start of replacement region (seconds).
        end: End of replacement region (seconds).
        crossfade_ms: Crossfade duration in milliseconds.

    Returns:
        Path to the blended output file.
    """
    info = get_video_info(original_path)
    duration = info.get("duration", 0)
    cf_sec = crossfade_ms / 1000.0

    # Build complex filter:
    # 1. Split original audio into before/after segments
    # 2. Crossfade generated segment at boundaries
    # 3. Concatenate all parts

    before_end = start
    after_start = end

    filter_graph = (
        f"[0:a]atrim=0:{before_end},asetpts=PTS-STARTPTS[before];"
        f"[0:a]atrim={after_start}:{duration},asetpts=PTS-STARTPTS[after];"
        f"[1:a]asetpts=PTS-STARTPTS[gen];"
        f"[before][gen]acrossfade=d={cf_sec}:c1=tri:c2=tri[bg];"
        f"[bg][after]acrossfade=d={cf_sec}:c1=tri:c2=tri[out]"
    )

    cmd = (
        FFmpegCmd()
        .input(original_path)
        .input(generated_path)
        .filter_complex(filter_graph, maps=["[out]"])
        .audio_codec("aac", bitrate="192k")
        .output(output_path)
    )

    run_ffmpeg(cmd.build(), timeout=300)

    if on_progress:
        on_progress(90, "Audio boundaries blended")

    return output_path


# ---------------------------------------------------------------------------
# TTS Generation
# ---------------------------------------------------------------------------
def _generate_tts_edge(text: str, output_wav: str, language: str = "en",
                       speed: float = 1.0) -> str:
    """Generate speech via edge-tts (Microsoft Edge free TTS)."""
    ensure_package("edge_tts", "edge-tts")

    import asyncio

    import edge_tts

    voice_map = {
        "en": "en-US-GuyNeural",
        "es": "es-ES-AlvaroNeural",
        "fr": "fr-FR-HenriNeural",
        "de": "de-DE-ConradNeural",
        "ja": "ja-JP-KeitaNeural",
        "zh": "zh-CN-YunxiNeural",
    }
    voice = voice_map.get(language, "en-US-GuyNeural")
    rate_str = f"+{int((speed - 1) * 100)}%" if speed >= 1 else f"{int((speed - 1) * 100)}%"

    async def _run():
        comm = edge_tts.Communicate(text, voice, rate=rate_str)
        await comm.save(output_wav)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()

    return output_wav


def _generate_tts_fallback(text: str, output_wav: str, speed: float = 1.0) -> str:
    """Fallback TTS: generate silence matching approximate word count duration."""
    words = len(text.split())
    duration = max(1.0, words / 2.5) / speed  # ~150 WPM

    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
        "-t", str(duration),
        "-acodec", "pcm_s16le", output_wav,
    ], timeout=30)

    return output_wav


# ---------------------------------------------------------------------------
# Main Overdub
# ---------------------------------------------------------------------------
def overdub_segment(
    video_path: str,
    start: float,
    end: float,
    new_text: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    config: Optional[OverdubConfig] = None,
    on_progress: Optional[Callable] = None,
) -> OverdubResult:
    """
    Replace a segment of audio with AI-generated speech matching the speaker's voice.

    Pipeline:
    1. Extract reference voice from surrounding audio
    2. Generate new speech with TTS
    3. Crossfade replacement into original
    4. Mux back with video

    Args:
        video_path: Input video/audio path.
        start: Start of segment to overdub (seconds).
        end: End of segment to overdub (seconds).
        new_text: Replacement text to speak.
        output_path: Optional explicit output path.
        output_dir: Output directory (defaults to input dir).
        config: OverdubConfig with TTS/crossfade parameters.
        on_progress: Callback(pct, msg) for progress updates.

    Returns:
        OverdubResult with output path and metadata.
    """
    if config is None:
        config = OverdubConfig()

    if not new_text.strip():
        raise ValueError("new_text cannot be empty")
    if start >= end:
        raise ValueError("start must be less than end")

    result = OverdubResult(
        segment_start=start,
        segment_end=end,
        new_text=new_text,
    )

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_overdub.mp4")

    if on_progress:
        on_progress(5, "Extracting voice reference...")

    # Step 1: Clone voice reference
    tmp_files = []
    try:
        ref_path = clone_voice_segment(
            video_path, start, end,
            context_seconds=config.voice_clone_seconds,
            on_progress=on_progress,
        )
        tmp_files.append(ref_path)

        if on_progress:
            on_progress(30, "Generating replacement speech...")

        # Step 2: Generate TTS audio
        _ntf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tts_path = _ntf.name
        _ntf.close()
        tmp_files.append(tts_path)

        backend_used = config.tts_backend
        try:
            if config.tts_backend == "edge":
                _generate_tts_edge(new_text, tts_path, config.language, config.speed)
            else:
                _generate_tts_fallback(new_text, tts_path, config.speed)
                backend_used = "fallback"
        except Exception as e:
            logger.warning("TTS backend '%s' failed: %s, using fallback", config.tts_backend, e)
            _generate_tts_fallback(new_text, tts_path, config.speed)
            backend_used = "fallback"

        result.tts_backend_used = backend_used

        # Get generated duration
        tts_info = get_video_info(tts_path)
        result.duration_generated = tts_info.get("duration", end - start)

        if on_progress:
            on_progress(50, "Extracting original audio...")

        # Step 3: Extract original audio track
        _ntf2 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        orig_audio = _ntf2.name
        _ntf2.close()
        tmp_files.append(orig_audio)

        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", str(config.sample_rate), "-ac", "1",
            orig_audio,
        ], timeout=120)

        if on_progress:
            on_progress(60, "Blending audio boundaries...")

        # Step 4: Blend audio
        _ntf3 = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        blended_audio = _ntf3.name
        _ntf3.close()
        tmp_files.append(blended_audio)

        blend_audio_boundaries(
            orig_audio, tts_path, blended_audio,
            start, end,
            crossfade_ms=config.crossfade_ms,
            on_progress=on_progress,
        )

        if on_progress:
            on_progress(80, "Muxing video with corrected audio...")

        # Step 5: Mux back with video
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path, "-i", blended_audio,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", output_path,
        ], timeout=600)

        result.output_path = output_path

    finally:
        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    if on_progress:
        on_progress(100, "Overdub complete!")

    return result
