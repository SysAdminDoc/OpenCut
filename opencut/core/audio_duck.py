"""
OpenCut Audio Ducking Module v0.8.1

Auto-lower background music/audio during speech:
- Sidechain compression: music ducks when voice is detected
- Speech-gated ducking: analyze speech segments, apply volume automation
- Mix two audio tracks with ducking
- Adjustable duck amount, attack, release, threshold

Uses FFmpeg sidechaincompress for real-time ducking,
or volume keyframe automation for more precise control.
"""

import logging
import os
import subprocess
import tempfile
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


def _run_ffmpeg(cmd: List[str], timeout: int = 3600) -> str:
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode(errors='replace')[-500:]}")
    return result.stderr.decode(errors="replace")


# ---------------------------------------------------------------------------
# Sidechain Duck (two audio inputs)
# ---------------------------------------------------------------------------
def sidechain_duck(
    music_path: str,
    voice_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    duck_amount: float = 0.7,
    threshold: float = 0.015,
    attack: float = 20.0,
    release: float = 300.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Duck music track when voice is present using FFmpeg sidechaincompress.

    Args:
        music_path: Background music / audio to be ducked.
        voice_path: Voice / speech track that triggers ducking.
        duck_amount: How much to reduce music (0.0-1.0). 0.7 = reduce by 70%.
        threshold: Voice detection threshold (0.001-0.5).
        attack: How fast ducking kicks in (ms).
        release: How fast ducking releases (ms).
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(music_path))[0]
        directory = output_dir or os.path.dirname(music_path)
        output_path = os.path.join(directory, f"{base}_ducked.wav")

    if on_progress:
        on_progress(10, "Applying sidechain ducking...")

    # ratio controls compression depth. Higher = more ducking.
    ratio = max(2, int(duck_amount * 20))
    # level_sc scales the sidechain sensitivity
    level_sc = 1.0

    af = (
        f"[0:a][1:a]sidechaincompress="
        f"threshold={threshold}:"
        f"ratio={ratio}:"
        f"attack={attack}:"
        f"release={release}:"
        f"level_sc={level_sc}"
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", music_path, "-i", voice_path,
        "-filter_complex", af,
        "-c:a", "pcm_s16le",
        output_path,
    ]
    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Sidechain ducking applied!")
    return output_path


# ---------------------------------------------------------------------------
# Mix with Auto-Duck (voice + music -> mixed output)
# ---------------------------------------------------------------------------
def mix_with_duck(
    voice_path: str,
    music_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    music_volume: float = 0.3,
    duck_amount: float = 0.6,
    threshold: float = 0.02,
    attack: float = 50.0,
    release: float = 500.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Mix voice and music with automatic ducking.

    The music volume is reduced by duck_amount whenever voice is detected,
    then the two tracks are mixed together.

    Args:
        music_volume: Base music volume (0.0-1.0) before ducking.
        duck_amount: Additional reduction during speech (0.0-1.0).
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(voice_path))[0]
        directory = output_dir or os.path.dirname(voice_path)
        output_path = os.path.join(directory, f"{base}_mixed.wav")

    if on_progress:
        on_progress(10, "Mixing voice + music with ducking...")

    ratio = max(2, int(duck_amount * 20))

    # Complex filter: duck music using voice as sidechain, then mix
    fc = (
        f"[1:a]volume={music_volume}[music];"
        f"[music][0:a]sidechaincompress="
        f"threshold={threshold}:ratio={ratio}:"
        f"attack={attack}:release={release}[ducked];"
        f"[0:a][ducked]amix=inputs=2:duration=longest:dropout_transition=2"
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", voice_path, "-i", music_path,
        "-filter_complex", fc,
        "-c:a", "pcm_s16le",
        output_path,
    ]
    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Mixed with ducking!")
    return output_path


# ---------------------------------------------------------------------------
# Volume-Based Duck (single file, detect loud/quiet)
# ---------------------------------------------------------------------------
def auto_duck_video(
    video_path: str,
    music_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    music_volume: float = 0.25,
    duck_amount: float = 0.7,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Add background music to a video with auto-ducking.

    Extracts the video's speech audio, uses it as sidechain to duck
    the music, then mixes everything back together.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        ext = os.path.splitext(video_path)[1] or ".mp4"
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_with_music{ext}")

    if on_progress:
        on_progress(5, "Preparing audio mix...")

    ratio = max(2, int(duck_amount * 20))

    # Use video audio as sidechain to duck music, then mix, then combine with video
    fc = (
        f"[1:a]volume={music_volume}[music];"
        f"[music][0:a]sidechaincompress="
        f"threshold=0.02:ratio={ratio}:"
        f"attack=50:release=500[ducked];"
        f"[0:a][ducked]amix=inputs=2:duration=first:dropout_transition=2[mixed]"
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path, "-i", music_path,
        "-filter_complex", fc,
        "-map", "0:v", "-map", "[mixed]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Music added with ducking!")
    return output_path


# ---------------------------------------------------------------------------
# Simple audio mix (no ducking)
# ---------------------------------------------------------------------------
def mix_audio_tracks(
    tracks: List[Dict],
    output_path: Optional[str] = None,
    output_dir: str = "",
    duration_mode: str = "longest",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Mix multiple audio tracks together.

    Args:
        tracks: List of {"path": str, "volume": float (0-1)}.
        duration_mode: "longest", "shortest", or "first".
    """
    if not tracks:
        raise ValueError("No tracks to mix")

    if output_path is None:
        directory = output_dir or os.path.dirname(tracks[0]["path"])
        output_path = os.path.join(directory, "mixed_audio.wav")

    if on_progress:
        on_progress(10, f"Mixing {len(tracks)} audio tracks...")

    # Build filter chain
    inputs = []
    volume_filters = []
    for i, t in enumerate(tracks):
        vol = t.get("volume", 1.0)
        inputs += ["-i", t["path"]]
        volume_filters.append(f"[{i}:a]volume={vol}[a{i}]")

    mix_inputs = "".join(f"[a{i}]" for i in range(len(tracks)))
    fc = ";".join(volume_filters) + f";{mix_inputs}amix=inputs={len(tracks)}:duration={duration_mode}:dropout_transition=2"

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    cmd += inputs
    cmd += ["-filter_complex", fc, "-c:a", "pcm_s16le", output_path]
    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Audio mixed!")
    return output_path
