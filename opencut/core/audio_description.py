"""
OpenCut Audio Description Track Generator v1.0.0

Generate audio description (AD) tracks for accessibility:
- Detect dialogue pauses/gaps suitable for descriptions
- Describe visual content at timestamps
- Synthesize description speech
- Mix descriptions into an AD track alongside original audio

Uses FFmpeg for audio processing and gap detection.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class DescriptionGap:
    """A gap in dialogue suitable for audio description."""
    start: float
    end: float
    duration: float = 0.0
    suitable: bool = True
    max_words: int = 0  # Estimated words that fit in this gap


@dataclass
class VisualDescription:
    """A description of visual content at a timestamp."""
    timestamp: float
    description: str
    duration: float = 0.0
    importance: str = "normal"  # low, normal, high


@dataclass
class ADResult:
    """Result from audio description generation."""
    output_path: str = ""
    gaps_found: int = 0
    descriptions_added: int = 0
    total_description_duration: float = 0.0
    original_duration: float = 0.0


# ---------------------------------------------------------------------------
# Gap Detection
# ---------------------------------------------------------------------------
def find_description_gaps(
    audio_path: str,
    transcript: Optional[List[Dict]] = None,
    min_gap_seconds: float = 2.0,
    max_gap_seconds: float = 15.0,
    silence_threshold_db: float = -35.0,
    on_progress: Optional[Callable] = None,
) -> List[DescriptionGap]:
    """
    Find gaps in dialogue suitable for audio descriptions.

    Uses FFmpeg silencedetect to find pauses, optionally cross-referenced
    with a transcript for more accurate gap identification.

    Args:
        audio_path: Source audio/video file.
        transcript: Optional transcript (list of dicts with start, end, text).
        min_gap_seconds: Minimum gap duration to consider.
        max_gap_seconds: Maximum gap duration to consider.
        silence_threshold_db: Silence detection threshold in dB.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of DescriptionGap objects.
    """
    if on_progress:
        on_progress(5, "Detecting dialogue gaps...")

    ffmpeg = get_ffmpeg_path()

    # Use transcript gaps if available
    if transcript:
        gaps = _gaps_from_transcript(transcript, min_gap_seconds, max_gap_seconds)
        if on_progress:
            on_progress(100, f"Found {len(gaps)} gaps from transcript")
        return gaps

    # Otherwise use silence detection
    cmd = [
        ffmpeg, "-hide_banner", "-i", audio_path,
        "-af", f"silencedetect=noise={silence_threshold_db}dB:d={min_gap_seconds}",
        "-f", "null", "-",
    ]

    try:
        stderr_output = run_ffmpeg(cmd, timeout=600, stderr_cap=1)
    except RuntimeError:
        stderr_output = ""

    if on_progress:
        on_progress(50, "Parsing silence regions...")

    gaps = []
    lines = stderr_output.split("\n") if stderr_output else []

    silence_start = None
    for line in lines:
        if "silence_start:" in line:
            try:
                silence_start = float(
                    line.split("silence_start:")[1].strip().split()[0]
                )
            except (ValueError, IndexError):
                silence_start = None
        elif "silence_end:" in line and silence_start is not None:
            try:
                silence_end = float(
                    line.split("silence_end:")[1].strip().split()[0]
                )
                duration = silence_end - silence_start
                if min_gap_seconds <= duration <= max_gap_seconds:
                    # Estimate words: ~3 words per second for AD narration
                    max_words = int(duration * 3)
                    gaps.append(DescriptionGap(
                        start=silence_start,
                        end=silence_end,
                        duration=duration,
                        suitable=True,
                        max_words=max_words,
                    ))
            except (ValueError, IndexError):
                pass
            silence_start = None

    if on_progress:
        on_progress(100, f"Found {len(gaps)} suitable gaps")

    return gaps


def _gaps_from_transcript(
    transcript: List[Dict],
    min_gap: float,
    max_gap: float,
) -> List[DescriptionGap]:
    """Extract gaps from transcript timing data."""
    gaps = []
    sorted_entries = sorted(transcript, key=lambda x: float(x.get("start", 0)))

    for i in range(len(sorted_entries) - 1):
        current_end = float(sorted_entries[i].get("end", 0))
        next_start = float(sorted_entries[i + 1].get("start", 0))
        duration = next_start - current_end

        if min_gap <= duration <= max_gap:
            gaps.append(DescriptionGap(
                start=current_end,
                end=next_start,
                duration=duration,
                suitable=True,
                max_words=int(duration * 3),
            ))

    return gaps


# ---------------------------------------------------------------------------
# Visual Content Description
# ---------------------------------------------------------------------------
def describe_visual_content(
    video_path: str,
    timestamp: float,
    on_progress: Optional[Callable] = None,
) -> VisualDescription:
    """
    Describe visual content at a specific timestamp.

    Extracts a frame at the given timestamp and generates a text
    description. Uses basic frame analysis for scene type detection.

    Args:
        video_path: Source video file.
        timestamp: Timestamp in seconds to describe.
        on_progress: Progress callback(pct, msg).

    Returns:
        VisualDescription with text description of the scene.
    """
    if on_progress:
        on_progress(10, f"Analyzing frame at {timestamp:.1f}s...")

    ffmpeg = get_ffmpeg_path()

    # Extract frame for analysis
    temp_frame = None
    try:
        ntf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_frame = ntf.name
        ntf.close()

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-y", "-ss", f"{timestamp:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            temp_frame,
        ]
        run_ffmpeg(cmd, timeout=60)

        # Basic frame analysis using FFmpeg signalstats
        stats_cmd = [
            ffmpeg, "-hide_banner",
            "-i", temp_frame,
            "-vf", "signalstats",
            "-f", "null", "-",
        ]
        try:
            stats_output = run_ffmpeg(stats_cmd, timeout=30, stderr_cap=1)
        except RuntimeError:
            stats_output = ""

        # Generate description based on available analysis
        description = _generate_scene_description(stats_output, timestamp)

        if on_progress:
            on_progress(100, "Frame analysis complete")

        return VisualDescription(
            timestamp=timestamp,
            description=description,
            importance="normal",
        )

    finally:
        if temp_frame and os.path.exists(temp_frame):
            try:
                os.unlink(temp_frame)
            except OSError:
                pass


def _generate_scene_description(stats_output: str, timestamp: float) -> str:
    """Generate a scene description from frame statistics."""
    # Basic analysis from signal stats
    brightness = "normal"
    if "YAVG" in stats_output:
        try:
            for line in stats_output.split("\n"):
                if "YAVG" in line:
                    val = float(line.split("YAVG:")[1].strip().split()[0])
                    if val < 50:
                        brightness = "dark"
                    elif val > 200:
                        brightness = "bright"
                    break
        except (ValueError, IndexError):
            pass

    # Return a placeholder description indicating the scene characteristics
    time_str = f"{int(timestamp // 60)}:{int(timestamp % 60):02d}"
    return (
        f"Scene at {time_str}: "
        f"{'Dark' if brightness == 'dark' else 'Bright' if brightness == 'bright' else 'Normal lighting'} scene. "
        f"[Visual description placeholder — connect an AI vision model for detailed descriptions.]"
    )


# ---------------------------------------------------------------------------
# Speech Synthesis
# ---------------------------------------------------------------------------
def synthesize_description(
    text: str,
    voice: str = "default",
    output_path_val: Optional[str] = None,
    speed: float = 1.1,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Synthesize audio description speech from text.

    Uses available TTS engine (pyttsx3 as fallback, or system TTS).
    Audio descriptions are typically spoken slightly faster than normal
    to fit within dialogue pauses.

    Args:
        text: Description text to synthesize.
        voice: Voice identifier (engine-dependent).
        output_path_val: Output audio path (auto-generated if None).
        speed: Speech rate multiplier (1.1 = slightly fast for AD).
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to synthesized audio file.
    """
    if on_progress:
        on_progress(10, "Synthesizing description audio...")

    if output_path_val is None:
        output_path_val = os.path.join(
            tempfile.gettempdir(),
            f"ad_synth_{hash(text) & 0xFFFFFF:06x}.wav",
        )

    ffmpeg = get_ffmpeg_path()

    # Try pyttsx3 first
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", int(150 * speed))

        if voice != "default":
            voices = engine.getProperty("voices")
            for v in voices:
                if voice.lower() in v.name.lower() or voice == v.id:
                    engine.setProperty("voice", v.id)
                    break

        engine.save_to_file(text, output_path_val)
        engine.runAndWait()

        if on_progress:
            on_progress(100, "Description audio synthesized")
        return output_path_val

    except (ImportError, Exception) as e:
        logger.debug("pyttsx3 not available: %s, using FFmpeg sine fallback", e)

    # Fallback: generate a placeholder tone (sine wave beep)
    # In production this would use a proper TTS API
    duration = max(1.0, len(text.split()) / (3 * speed))

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-y",
        "-f", "lavfi", "-i",
        f"sine=frequency=440:duration={duration:.2f}",
        "-af", f"volume=0.3,atempo={speed}",
        output_path_val,
    ]
    run_ffmpeg(cmd, timeout=60)

    if on_progress:
        on_progress(100, "Description audio synthesized (placeholder)")

    return output_path_val


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def generate_audio_description(
    video_path: str,
    output_path_val: Optional[str] = None,
    descriptions: Optional[List[Dict]] = None,
    transcript: Optional[List[Dict]] = None,
    voice: str = "default",
    min_gap_seconds: float = 2.0,
    on_progress: Optional[Callable] = None,
) -> ADResult:
    """
    Generate a complete audio description track for a video.

    Pipeline:
    1. Find gaps in dialogue
    2. Generate/use provided descriptions for visual content
    3. Synthesize description speech
    4. Mix descriptions into AD audio track

    Args:
        video_path: Source video file.
        output_path_val: Output path (auto-generated if None).
        descriptions: Pre-written descriptions (list of dicts with
            timestamp, text, duration).
        transcript: Transcript for gap detection.
        voice: TTS voice identifier.
        min_gap_seconds: Minimum gap for descriptions.
        on_progress: Progress callback(pct, msg).

    Returns:
        ADResult with output path and statistics.
    """
    if output_path_val is None:
        output_path_val = output_path(video_path, "_audio_described")

    ffmpeg = get_ffmpeg_path()
    temp_files = []
    result = ADResult()

    try:
        # Step 1: Find gaps
        if on_progress:
            on_progress(5, "Finding description gaps...")

        gaps = find_description_gaps(
            video_path,
            transcript=transcript,
            min_gap_seconds=min_gap_seconds,
            on_progress=None,
        )
        result.gaps_found = len(gaps)

        # Step 2: Match descriptions to gaps
        if on_progress:
            on_progress(20, "Preparing descriptions...")

        desc_audio_pairs = []  # (timestamp, audio_path)

        if descriptions:
            # Use provided descriptions
            for i, desc in enumerate(descriptions):
                ts = float(desc.get("timestamp", 0))
                text = desc.get("text", "")
                if not text:
                    continue

                # Descriptions are scheduled by their explicit ``timestamp``
                # regardless of whether they line up with a detected silent
                # gap — the previous implementation searched ``gaps`` for a
                # match but then guarded the synthesis with
                # ``if best_gap or True:``, which always entered the branch.
                # Skipping the lookup keeps the same effective behavior with
                # less noise.
                synth_path = os.path.join(
                    tempfile.gettempdir(), f"ad_desc_{i:04d}.wav",
                )
                temp_files.append(synth_path)

                synthesize_description(
                    text, voice=voice,
                    output_path_val=synth_path,
                    on_progress=None,
                )
                desc_audio_pairs.append((ts, synth_path))

                if on_progress:
                    pct = 20 + int(40 * (i + 1) / len(descriptions))
                    on_progress(pct, f"Synthesized {i + 1}/{len(descriptions)}")
        else:
            # Auto-describe at gap positions
            for i, gap in enumerate(gaps[:20]):  # Limit to 20 descriptions
                desc = describe_visual_content(video_path, gap.start, on_progress=None)
                synth_path = os.path.join(
                    tempfile.gettempdir(), f"ad_auto_{i:04d}.wav",
                )
                temp_files.append(synth_path)

                synthesize_description(
                    desc.description, voice=voice,
                    output_path_val=synth_path,
                    on_progress=None,
                )
                desc_audio_pairs.append((gap.start, synth_path))

        result.descriptions_added = len(desc_audio_pairs)

        if not desc_audio_pairs:
            # No descriptions to add, just copy original
            if on_progress:
                on_progress(90, "No descriptions to add, copying original...")

            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-y", "-i", video_path, "-c", "copy",
                output_path_val,
            ]
            run_ffmpeg(cmd, timeout=600)
            result.output_path = output_path_val
            return result

        # Step 3: Mix descriptions into audio
        if on_progress:
            on_progress(65, "Mixing descriptions into audio track...")

        result = _mix_descriptions(
            video_path, desc_audio_pairs, output_path_val, result, temp_files,
        )

        if on_progress:
            on_progress(100, f"Audio description complete: {result.descriptions_added} descriptions")

        return result

    finally:
        for f in temp_files:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except OSError:
                    pass


def _mix_descriptions(
    video_path: str,
    desc_pairs: List,
    output_path_val: str,
    result: ADResult,
    temp_files: List[str],
) -> ADResult:
    """Mix description audio clips into the video's audio track."""
    ffmpeg = get_ffmpeg_path()

    # Build FFmpeg command with multiple inputs and amix
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    cmd.extend(["-i", video_path])

    for _, audio_path in desc_pairs:
        cmd.extend(["-i", audio_path])

    # Build filter complex to delay and mix each description
    n_desc = len(desc_pairs)
    filter_parts = []
    mix_inputs = ["[0:a]"]

    for i, (timestamp, _) in enumerate(desc_pairs):
        delay_ms = int(timestamp * 1000)
        filter_parts.append(
            f"[{i + 1}:a]adelay={delay_ms}|{delay_ms},"
            f"volume=0.9[d{i}]"
        )
        mix_inputs.append(f"[d{i}]")

    # Mix all streams
    mix_str = "".join(mix_inputs)
    filter_parts.append(
        f"{mix_str}amix=inputs={n_desc + 1}:duration=longest:dropout_transition=2[outa]"
    )

    filter_complex = ";".join(filter_parts)

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[outa]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path_val,
    ])

    run_ffmpeg(cmd, timeout=3600)

    result.output_path = output_path_val
    result.total_description_duration = sum(
        3.0 for _ in desc_pairs  # approximate
    )
    return result
