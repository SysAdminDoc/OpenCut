"""
OpenCut Transition Sound Effects Module (Feature 18.7)

Map video transition types to SFX categories (whoosh, impact, riser, etc.),
auto-apply SFX at transition points, and provide a library of available
transition sounds.

Functions:
    get_transition_sfx    - Get an SFX file for a given transition type/style
    apply_transition_sfx  - Apply SFX at all transition points in a video
    list_available_sfx    - List available transition SFX categories and styles
"""

import logging
import os
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
# Transition Type -> SFX Category Mapping
# ---------------------------------------------------------------------------
TRANSITION_SFX_MAP = {
    # Cut transitions
    "cut": {"category": "click", "style": "subtle", "duration": 0.1},
    "hard_cut": {"category": "impact", "style": "punchy", "duration": 0.15},
    "jump_cut": {"category": "pop", "style": "quick", "duration": 0.1},

    # Wipe transitions
    "wipe": {"category": "whoosh", "style": "smooth", "duration": 0.5},
    "wipe_left": {"category": "whoosh", "style": "directional", "duration": 0.5},
    "wipe_right": {"category": "whoosh", "style": "directional", "duration": 0.5},

    # Dissolve transitions
    "dissolve": {"category": "shimmer", "style": "gentle", "duration": 0.8},
    "cross_dissolve": {"category": "shimmer", "style": "smooth", "duration": 1.0},
    "fade_in": {"category": "riser", "style": "building", "duration": 1.0},
    "fade_out": {"category": "downer", "style": "fading", "duration": 1.0},
    "fade_to_black": {"category": "downer", "style": "dramatic", "duration": 1.0},

    # Slide transitions
    "slide": {"category": "whoosh", "style": "fast", "duration": 0.4},
    "push": {"category": "whoosh", "style": "forceful", "duration": 0.4},
    "pull": {"category": "whoosh", "style": "suction", "duration": 0.4},

    # Zoom transitions
    "zoom_in": {"category": "riser", "style": "accelerating", "duration": 0.6},
    "zoom_out": {"category": "downer", "style": "decelerating", "duration": 0.6},
    "zoom_through": {"category": "whoosh_deep", "style": "immersive", "duration": 0.8},

    # Spin transitions
    "spin": {"category": "whoosh", "style": "rotating", "duration": 0.6},
    "whip_pan": {"category": "whoosh_fast", "style": "aggressive", "duration": 0.3},

    # Glitch transitions
    "glitch": {"category": "digital", "style": "glitchy", "duration": 0.4},
    "static": {"category": "noise", "style": "electric", "duration": 0.5},

    # Impact transitions
    "impact": {"category": "impact", "style": "heavy", "duration": 0.3},
    "slam": {"category": "impact", "style": "massive", "duration": 0.4},
    "flash": {"category": "flash", "style": "bright", "duration": 0.2},

    # Cinematic
    "cinematic": {"category": "riser", "style": "epic", "duration": 1.5},
    "dramatic": {"category": "boom", "style": "dramatic", "duration": 1.0},
}

# SFX synthesis definitions (FFmpeg filter chains)
SFX_SYNTHESIS = {
    "click": lambda d: f"sine=frequency=1200:duration={d},afade=t=in:d=0.005,afade=t=out:st={max(0, d - 0.03)}:d=0.03",
    "impact": lambda d: f"anoisesrc=d={d}:c=white:a=0.7,highpass=f=30,lowpass=f=300,afade=t=out:st=0:d={d}:curve=exp",
    "pop": lambda d: f"sine=frequency=800:duration={d},afade=t=in:d=0.005,afade=t=out:st={max(0, d - 0.05)}:d=0.05",
    "whoosh": lambda d: f"anoisesrc=d={d}:c=pink:a=0.5,bandpass=f=500:width_type=o:w=3,afade=t=in:d={d * 0.3},afade=t=out:st={d * 0.5}:d={d * 0.5}",
    "whoosh_fast": lambda d: f"anoisesrc=d={d}:c=pink:a=0.6,bandpass=f=800:width_type=o:w=4,afade=t=in:d={d * 0.2},afade=t=out:st={d * 0.4}:d={d * 0.6}",
    "whoosh_deep": lambda d: f"anoisesrc=d={d}:c=brown:a=0.5,bandpass=f=200:width_type=o:w=2,afade=t=in:d={d * 0.2},afade=t=out:st={d * 0.5}:d={d * 0.5}",
    "shimmer": lambda d: f"sine=frequency=2000:duration={d},tremolo=f=8:d=0.6,afade=t=in:d={d * 0.3},afade=t=out:st={d * 0.5}:d={d * 0.5}",
    "riser": lambda d: f"sine=frequency=200:duration={d},asetrate=44100*2,aresample=44100,afade=t=in:d={d * 0.1},afade=t=out:st={max(0, d - 0.1)}:d=0.1",
    "downer": lambda d: f"sine=frequency=800:duration={d},asetrate=44100*0.5,aresample=44100,afade=t=in:d=0.05,afade=t=out:st={d * 0.3}:d={d * 0.7}",
    "digital": lambda d: f"anoisesrc=d={d}:c=white:a=0.4,bandreject=f=4000:w=1000,tremolo=f=20:d=0.9,afade=t=in:d=0.01,afade=t=out:st={max(0, d - 0.05)}:d=0.05",
    "noise": lambda d: f"anoisesrc=d={d}:c=white:a=0.3,afade=t=in:d={d * 0.1},afade=t=out:st={d * 0.6}:d={d * 0.4}",
    "flash": lambda d: f"sine=frequency=3000:duration={d},afade=t=in:d=0.005,afade=t=out:st=0:d={d}:curve=exp",
    "boom": lambda d: f"anoisesrc=d={d}:c=brown:a=0.8,lowpass=f=150,afade=t=out:st=0:d={d}:curve=exp",
}


@dataclass
class TransitionSFXResult:
    """Result of getting/generating a transition SFX."""
    output_path: str
    transition_type: str
    sfx_category: str
    style: str
    duration: float


@dataclass
class TransitionPoint:
    """A transition point in a video timeline."""
    timestamp: float
    transition_type: str
    duration: float = 0.5
    style: str = ""


@dataclass
class ApplyTransitionResult:
    """Result of applying transition SFX to a video."""
    output_path: str
    transitions_processed: int
    total_sfx_applied: int
    video_duration: float


def get_transition_sfx(
    transition_type: str,
    style: str = "",
    duration: Optional[float] = None,
    output: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> TransitionSFXResult:
    """Generate an SFX audio file for a given transition type.

    Args:
        transition_type: Transition type name (e.g. "wipe", "dissolve", "impact").
        style: Optional style override.
        duration: Duration override in seconds. Uses mapping default if None.
        output: Output WAV path. Auto-generated if None.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        TransitionSFXResult with the generated audio path.
    """
    if on_progress:
        on_progress(10, f"Getting SFX for {transition_type}...")

    # Look up transition mapping
    mapping = TRANSITION_SFX_MAP.get(transition_type.lower(), {
        "category": "whoosh",
        "style": "default",
        "duration": 0.5,
    })

    category = mapping["category"]
    effective_style = style or mapping.get("style", "default")
    effective_duration = duration if duration is not None else mapping.get("duration", 0.5)

    if output is None:
        fd, output = tempfile.mkstemp(suffix=".wav", prefix=f"transition_{transition_type}_")
        os.close(fd)

    if on_progress:
        on_progress(30, f"Synthesizing {category} SFX...")

    # Get synthesis function
    synth_fn = SFX_SYNTHESIS.get(category, SFX_SYNTHESIS["whoosh"])
    filter_str = synth_fn(effective_duration)

    cmd = (
        FFmpegCmd()
        .filter_complex(f"{filter_str}[out]", maps=["[out]"])
        .audio_codec("pcm_s16le")
        .option("ar", "44100")
        .option("ac", "2")
        .output(output)
        .build()
    )

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(90, "Transition SFX generated")

    return TransitionSFXResult(
        output_path=output,
        transition_type=transition_type,
        sfx_category=category,
        style=effective_style,
        duration=effective_duration,
    )


def apply_transition_sfx(
    video_path: str,
    transitions: List[Dict],
    output: Optional[str] = None,
    master_volume: float = 0.7,
    on_progress: Optional[Callable] = None,
) -> ApplyTransitionResult:
    """Apply SFX at each transition point in a video.

    Args:
        video_path: Path to the input video.
        transitions: List of dicts with 'timestamp', 'type', and optional 'duration', 'style'.
        output: Output video path. Auto-generated if None.
        master_volume: Master volume for all transition SFX (0.0-1.0).
        on_progress: Optional progress callback(pct, msg).

    Returns:
        ApplyTransitionResult with the output path and stats.
    """
    if not transitions:
        raise ValueError("No transition points provided")

    info = get_video_info(video_path)
    video_duration = info.get("duration", 0)

    if output is None:
        output = output_path(video_path, "transition_sfx")

    if on_progress:
        on_progress(5, f"Generating SFX for {len(transitions)} transitions...")

    # Generate SFX for each transition
    temp_sfx_files = []
    valid_transitions = []

    try:
        for i, trans in enumerate(transitions):
            timestamp = float(trans.get("timestamp", 0))
            trans_type = trans.get("type", "cut")
            trans_duration = float(trans.get("duration", 0)) or None
            trans_style = trans.get("style", "")

            pct = 10 + int((i / len(transitions)) * 40)
            if on_progress:
                on_progress(pct, f"Generating SFX {i + 1}/{len(transitions)}: {trans_type}")

            try:
                result = get_transition_sfx(
                    transition_type=trans_type,
                    style=trans_style,
                    duration=trans_duration,
                )
                temp_sfx_files.append(result.output_path)
                valid_transitions.append({
                    "audio_path": result.output_path,
                    "timestamp": timestamp,
                    "volume": master_volume,
                })
            except Exception as e:
                logger.warning("Failed to generate SFX for transition %s at %s: %s",
                               trans_type, timestamp, e)

        if not valid_transitions:
            raise RuntimeError("Failed to generate any transition SFX")

        if on_progress:
            on_progress(55, "Mixing SFX into video...")

        # Mix SFX into video using amerge approach
        cmd_builder = FFmpegCmd().input(video_path)
        for sfx in valid_transitions:
            cmd_builder.input(sfx["audio_path"])

        filter_parts = []
        mix_inputs = ["[0:a]"]

        for i, sfx in enumerate(valid_transitions):
            idx = i + 1
            timestamp_ms = int(sfx["timestamp"] * 1000)
            volume = sfx["volume"]
            filter_parts.append(
                f"[{idx}:a]adelay={timestamp_ms}|{timestamp_ms},"
                f"volume={volume}[tsfx{i}]"
            )
            mix_inputs.append(f"[tsfx{i}]")

        n_inputs = len(mix_inputs)
        labels_str = "".join(mix_inputs)
        filter_parts.append(
            f"{labels_str}amix=inputs={n_inputs}:duration=longest"
            f":dropout_transition=2[outa]"
        )

        fc = ";".join(filter_parts)
        cmd_builder.filter_complex(fc, maps=["0:v", "[outa]"])
        cmd_builder.video_codec("copy")
        cmd_builder.audio_codec("aac", bitrate="192k")
        cmd_builder.faststart()
        cmd_builder.output(output)

        cmd = cmd_builder.build()

        if on_progress:
            on_progress(70, "Encoding...")

        run_ffmpeg(cmd)

        if on_progress:
            on_progress(95, "Done")

        return ApplyTransitionResult(
            output_path=output,
            transitions_processed=len(transitions),
            total_sfx_applied=len(valid_transitions),
            video_duration=video_duration,
        )

    finally:
        # Clean up temp SFX files
        for tmp in temp_sfx_files:
            try:
                if os.path.isfile(tmp):
                    os.unlink(tmp)
            except OSError:
                pass


def list_available_sfx(
    on_progress: Optional[Callable] = None,
) -> Dict:
    """List all available transition SFX categories and transition type mappings.

    Args:
        on_progress: Optional progress callback(pct, msg).

    Returns:
        Dict with 'transition_types' (mapping) and 'sfx_categories' (list).
    """
    if on_progress:
        on_progress(50, "Listing available transition SFX...")

    categories = sorted(set(SFX_SYNTHESIS.keys()))
    transitions = {}

    for trans_type, mapping in TRANSITION_SFX_MAP.items():
        transitions[trans_type] = {
            "category": mapping["category"],
            "style": mapping["style"],
            "default_duration": mapping["duration"],
        }

    return {
        "transition_types": transitions,
        "sfx_categories": categories,
        "total_transitions": len(transitions),
        "total_categories": len(categories),
    }
