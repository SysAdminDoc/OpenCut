"""
OpenCut AI Sound Effects Generation Module (Feature 2.4)

Analyze video scenes to identify on-screen actions, generate SFX prompts,
and produce sound effects via FFmpeg-based synthesis (tone, noise bursts)
as a fallback when AudioGen/AudioLDM2 models are not available.

Functions:
    detect_sfx_cues  - Analyze video/scene data and return SFX cue list
    generate_sfx     - Generate a single SFX audio file from a cue
    apply_sfx_to_video - Mix generated SFX tracks onto a video
"""

import logging
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# SFX Categories & Synthesis Parameters
# ---------------------------------------------------------------------------
SFX_CATEGORIES = {
    "impact": {"description": "Hit, punch, collision", "freq": 80, "type": "noise_burst", "duration": 0.3},
    "whoosh": {"description": "Fast movement, swipe", "freq": 400, "type": "sweep", "duration": 0.5},
    "click": {"description": "Button press, tap", "freq": 1200, "type": "tone_burst", "duration": 0.08},
    "explosion": {"description": "Blast, boom", "freq": 40, "type": "noise_burst", "duration": 1.5},
    "glass": {"description": "Glass breaking, shatter", "freq": 3000, "type": "noise_burst", "duration": 0.6},
    "footstep": {"description": "Walking, running", "freq": 200, "type": "tone_burst", "duration": 0.15},
    "door": {"description": "Door open/close", "freq": 150, "type": "tone_burst", "duration": 0.4},
    "water": {"description": "Splash, drip", "freq": 600, "type": "noise_filtered", "duration": 0.8},
    "wind": {"description": "Breeze, gust", "freq": 300, "type": "noise_filtered", "duration": 2.0},
    "bell": {"description": "Ring, chime", "freq": 880, "type": "sine_decay", "duration": 1.0},
    "alarm": {"description": "Alert, siren", "freq": 1000, "type": "sweep_loop", "duration": 1.5},
    "typing": {"description": "Keyboard typing", "freq": 2000, "type": "tone_burst", "duration": 0.05},
    "ambient": {"description": "Background atmosphere", "freq": 200, "type": "noise_filtered", "duration": 3.0},
}

# Scene action keywords mapped to SFX categories
ACTION_TO_SFX = {
    "hit": "impact", "punch": "impact", "kick": "impact", "slap": "impact",
    "crash": "explosion", "explode": "explosion", "boom": "explosion",
    "run": "footstep", "walk": "footstep", "step": "footstep",
    "swipe": "whoosh", "throw": "whoosh", "swing": "whoosh", "fly": "whoosh",
    "click": "click", "tap": "click", "press": "click", "type": "typing",
    "break": "glass", "shatter": "glass", "smash": "glass",
    "open": "door", "close": "door", "shut": "door",
    "splash": "water", "pour": "water", "drip": "water", "rain": "water",
    "wind": "wind", "breeze": "wind", "storm": "wind",
    "ring": "bell", "chime": "bell", "ding": "bell",
    "alarm": "alarm", "siren": "alarm", "alert": "alarm",
}


@dataclass
class SFXCue:
    """A detected sound effect cue point."""
    timestamp: float
    duration: float
    category: str
    description: str
    confidence: float = 0.8
    volume: float = 0.7


@dataclass
class SFXResult:
    """Result of SFX generation."""
    output_path: str
    category: str
    duration: float
    method: str  # "ffmpeg_synth" or "ai_model"


@dataclass
class SFXApplyResult:
    """Result of applying SFX to video."""
    output_path: str
    sfx_count: int
    duration: float


def detect_sfx_cues(
    video_path: str,
    scene_data: Optional[List[Dict]] = None,
    keywords: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> List[SFXCue]:
    """Analyze video scenes to identify on-screen actions and map to SFX categories.

    Args:
        video_path: Path to the video file.
        scene_data: Optional list of scene dicts with 'start', 'end', 'description' keys.
                    If None, generates placeholder cues based on video duration.
        keywords: Optional list of action keywords to search for in scene descriptions.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        List of SFXCue objects with timestamps and categories.
    """
    if on_progress:
        on_progress(10, "Analyzing video for SFX cues...")

    info = get_video_info(video_path)
    duration = info.get("duration", 0)

    cues = []

    if scene_data:
        # Parse scene descriptions to find action keywords
        for scene in scene_data:
            start = float(scene.get("start", 0))
            float(scene.get("end", start + 1.0))
            desc = scene.get("description", "").lower()

            # Check for action keywords in description
            for action, category in ACTION_TO_SFX.items():
                if action in desc:
                    cat_info = SFX_CATEGORIES.get(category, {})
                    cues.append(SFXCue(
                        timestamp=start,
                        duration=cat_info.get("duration", 0.5),
                        category=category,
                        description=f"Auto-detected: {action} in scene",
                        confidence=0.7,
                        volume=0.6,
                    ))
                    break  # One SFX per scene
    elif keywords:
        # Generate cues from keywords at even intervals
        interval = max(duration / (len(keywords) + 1), 1.0) if duration > 0 else 2.0
        for i, kw in enumerate(keywords):
            kw_lower = kw.lower().strip()
            category = ACTION_TO_SFX.get(kw_lower, "impact")
            cat_info = SFX_CATEGORIES.get(category, {})
            cues.append(SFXCue(
                timestamp=interval * (i + 1),
                duration=cat_info.get("duration", 0.5),
                category=category,
                description=f"Keyword cue: {kw}",
                confidence=0.9,
                volume=0.7,
            ))
    else:
        # No scene data: generate sample cues at regular intervals
        if duration > 0:
            num_cues = max(1, int(duration / 10))  # One cue every ~10 seconds
            for i in range(min(num_cues, 20)):
                timestamp = (i + 1) * (duration / (num_cues + 1))
                category = random.choice(list(SFX_CATEGORIES.keys()))
                cat_info = SFX_CATEGORIES[category]
                cues.append(SFXCue(
                    timestamp=round(timestamp, 2),
                    duration=cat_info["duration"],
                    category=category,
                    description=f"Auto-generated placeholder: {cat_info['description']}",
                    confidence=0.3,
                    volume=0.5,
                ))

    if on_progress:
        on_progress(80, f"Found {len(cues)} SFX cue points")

    return cues


def _build_synth_filter(category: str, duration: float) -> str:
    """Build an FFmpeg audio filter for synthesizing a sound effect.

    Args:
        category: SFX category name from SFX_CATEGORIES.
        duration: Desired duration in seconds.

    Returns:
        FFmpeg filter string for -filter_complex.
    """
    cat_info = SFX_CATEGORIES.get(category, SFX_CATEGORIES["impact"])
    freq = cat_info["freq"]
    synth_type = cat_info["type"]

    if synth_type == "tone_burst":
        # Short sine tone with fade
        return (
            f"sine=frequency={freq}:duration={duration},"
            f"afade=t=in:d={min(0.01, duration / 4)},"
            f"afade=t=out:st={max(0, duration - 0.05)}:d={min(0.05, duration / 2)}"
        )
    elif synth_type == "noise_burst":
        # White noise shaped with bandpass and envelope
        return (
            f"anoisesrc=d={duration}:c=white:a=0.5,"
            f"highpass=f={max(20, freq - 100)},"
            f"lowpass=f={freq + 500},"
            f"afade=t=in:d={min(0.02, duration / 4)},"
            f"afade=t=out:st={max(0, duration * 0.3)}:d={duration * 0.7}"
        )
    elif synth_type == "sweep":
        # Frequency sweep (whoosh)
        start_freq = freq * 2
        end_freq = freq // 2
        return (
            f"sine=frequency={start_freq}:duration={duration},"
            f"asetrate=44100*({end_freq}/{start_freq}),"
            f"aresample=44100,"
            f"afade=t=in:d={duration * 0.2},"
            f"afade=t=out:st={duration * 0.5}:d={duration * 0.5}"
        )
    elif synth_type == "noise_filtered":
        # Filtered noise (water, wind)
        return (
            f"anoisesrc=d={duration}:c=pink:a=0.3,"
            f"bandpass=f={freq}:width_type=o:w=2,"
            f"afade=t=in:d={duration * 0.1},"
            f"afade=t=out:st={duration * 0.7}:d={duration * 0.3}"
        )
    elif synth_type == "sine_decay":
        # Bell-like sine with exponential decay
        return (
            f"sine=frequency={freq}:duration={duration},"
            f"afade=t=out:st=0:d={duration}:curve=exp"
        )
    elif synth_type == "sweep_loop":
        # Alarm sweep (up-down)
        duration / 2
        return (
            f"sine=frequency={freq}:duration={duration},"
            f"tremolo=f=4:d=0.7,"
            f"afade=t=in:d=0.05,"
            f"afade=t=out:st={max(0, duration - 0.1)}:d=0.1"
        )
    else:
        # Default: simple tone
        return f"sine=frequency={freq}:duration={duration}"


def generate_sfx(
    cue: SFXCue,
    sfx_output_path: Optional[str] = None,
    use_ai: bool = False,
    on_progress: Optional[Callable] = None,
) -> SFXResult:
    """Generate a sound effect audio file from an SFX cue.

    Uses FFmpeg-based synthesis as the primary method. AI model integration
    (AudioGen/AudioLDM2) is a placeholder for future implementation.

    Args:
        cue: SFXCue object with category and duration.
        sfx_output_path: Output path for the generated WAV. Auto-generated if None.
        use_ai: If True, attempt AI model generation (falls back to FFmpeg).
        on_progress: Optional progress callback(pct, msg).

    Returns:
        SFXResult with the generated file path and metadata.
    """
    if on_progress:
        on_progress(10, f"Generating {cue.category} SFX...")

    if sfx_output_path is None:
        fd, sfx_output_path = tempfile.mkstemp(suffix=".wav", prefix=f"sfx_{cue.category}_")
        os.close(fd)

    method = "ffmpeg_synth"

    if use_ai:
        # Placeholder for AI model integration
        # In a full implementation, this would call AudioGen or AudioLDM2
        logger.info("AI SFX generation requested but not available, falling back to FFmpeg synthesis")

    # FFmpeg-based synthesis
    filter_str = _build_synth_filter(cue.category, cue.duration)
    cmd = (
        FFmpegCmd()
        .filter_complex(f"{filter_str}[out]", maps=["[out]"])
        .audio_codec("pcm_s16le")
        .option("ar", "44100")
        .option("ac", "1")
        .output(sfx_output_path)
        .build()
    )

    if on_progress:
        on_progress(50, "Running FFmpeg synthesis...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(90, "SFX generated")

    return SFXResult(
        output_path=sfx_output_path,
        category=cue.category,
        duration=cue.duration,
        method=method,
    )


def apply_sfx_to_video(
    video_path: str,
    sfx_list: List[Dict],
    output: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> SFXApplyResult:
    """Mix generated SFX tracks into a video at specified timestamps.

    Args:
        video_path: Path to the input video.
        sfx_list: List of dicts with 'audio_path', 'timestamp', and optional 'volume' (0.0-1.0).
        output: Output video path. Auto-generated if None.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        SFXApplyResult with the output path and SFX count.
    """
    if not sfx_list:
        raise ValueError("No SFX tracks provided")

    info = get_video_info(video_path)
    duration = info.get("duration", 0)

    if output is None:
        output = output_path(video_path, "sfx")

    if on_progress:
        on_progress(10, f"Mixing {len(sfx_list)} SFX tracks...")

    # Build FFmpeg command with multiple inputs and amerge
    cmd_builder = FFmpegCmd().input(video_path)

    # Add each SFX as an input
    for sfx in sfx_list:
        audio_path = sfx.get("audio_path", "")
        if not audio_path or not os.path.isfile(audio_path):
            logger.warning("Skipping missing SFX file: %s", audio_path)
            continue
        cmd_builder.input(audio_path)

    valid_sfx = [s for s in sfx_list if os.path.isfile(s.get("audio_path", ""))]
    if not valid_sfx:
        raise ValueError("No valid SFX audio files found")

    # Build filter_complex to delay and mix each SFX with the original audio
    filter_parts = []
    mix_inputs = ["[0:a]"]

    for i, sfx in enumerate(valid_sfx):
        idx = i + 1  # input index (0 is the video)
        timestamp_ms = int(float(sfx.get("timestamp", 0)) * 1000)
        volume = float(sfx.get("volume", 0.7))
        volume = max(0.0, min(1.0, volume))

        filter_parts.append(
            f"[{idx}:a]adelay={timestamp_ms}|{timestamp_ms},"
            f"volume={volume}[sfx{i}]"
        )
        mix_inputs.append(f"[sfx{i}]")

    # Amix all tracks together
    n_inputs = len(mix_inputs)
    mix_labels = "".join(mix_inputs)
    filter_parts.append(
        f"{mix_labels}amix=inputs={n_inputs}:duration=longest:dropout_transition=2[outa]"
    )

    fc = ";".join(filter_parts)
    cmd_builder.filter_complex(fc, maps=["0:v", "[outa]"])
    cmd_builder.video_codec("copy")
    cmd_builder.audio_codec("aac", bitrate="192k")
    cmd_builder.faststart()
    cmd_builder.output(output)

    cmd = cmd_builder.build()

    if on_progress:
        on_progress(40, "Encoding video with SFX...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(95, "Done")

    return SFXApplyResult(
        output_path=output,
        sfx_count=len(valid_sfx),
        duration=duration,
    )
