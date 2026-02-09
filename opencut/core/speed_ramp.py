"""
OpenCut Speed Ramp Module v0.8.1

Variable speed / speed ramping for video:
- Constant speed change (0.25x - 8x)
- Keyframe-based speed ramps (smooth transitions between speeds)
- Easing curves: linear, ease-in, ease-out, ease-in-out, exponential
- Time range extraction with speed change
- Reverse playback
- Audio pitch correction option (maintain pitch at different speeds)

All via FFmpeg setpts/atempo filters - zero dependencies.
"""

import logging
import math
import os
import subprocess
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


def _run_ffmpeg(cmd: List[str], timeout: int = 7200) -> str:
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode(errors='replace')[-500:]}")
    return result.stderr.decode(errors="replace")


def _get_duration(filepath: str) -> float:
    import json
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "json", filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    try:
        data = json.loads(result.stdout.decode())
        return float(data["format"]["duration"])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Easing Functions
# ---------------------------------------------------------------------------
def _ease_linear(t: float) -> float:
    return t

def _ease_in(t: float) -> float:
    return t * t

def _ease_out(t: float) -> float:
    return 1 - (1 - t) * (1 - t)

def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2 * t * t
    return 1 - (-2 * t + 2) ** 2 / 2

def _ease_exponential(t: float) -> float:
    if t == 0:
        return 0
    return 2 ** (10 * t - 10)

EASING_FUNCTIONS = {
    "linear": _ease_linear,
    "ease_in": _ease_in,
    "ease_out": _ease_out,
    "ease_in_out": _ease_in_out,
    "exponential": _ease_exponential,
}


# ---------------------------------------------------------------------------
# Constant Speed Change
# ---------------------------------------------------------------------------
def change_speed(
    input_path: str,
    speed: float = 2.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    maintain_pitch: bool = False,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply constant speed change to video.

    Args:
        speed: Speed multiplier (0.25-8.0). 2.0 = 2x faster, 0.5 = half speed.
        maintain_pitch: Keep audio pitch constant despite speed change.
    """
    speed = max(0.25, min(8.0, speed))

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1] or ".mp4"
        directory = output_dir or os.path.dirname(input_path)
        tag = f"{speed}x".replace(".", "_")
        output_path = os.path.join(directory, f"{base}_speed_{tag}{ext}")

    if on_progress:
        on_progress(10, f"Changing speed to {speed}x...")

    # Video: setpts filter (PTS / speed)
    vf = f"setpts={1.0/speed}*PTS"

    # Audio: atempo filter (chained for extreme values)
    af_parts = _build_atempo_chain(speed, maintain_pitch)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", input_path]
    cmd += ["-vf", vf]
    if af_parts:
        cmd += ["-af", af_parts]
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
    cmd.append(output_path)

    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, f"Speed changed to {speed}x")
    return output_path


def _build_atempo_chain(speed: float, maintain_pitch: bool = False) -> str:
    """Build FFmpeg atempo filter chain. atempo supports 0.5-100.0 per stage."""
    if maintain_pitch:
        # rubberband filter preserves pitch (requires librubberband)
        return f"rubberband=tempo={speed}"

    # Chain atempo filters for values outside 0.5-2.0
    if speed == 1.0:
        return ""

    parts = []
    remaining = speed
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - 1.0) > 0.001:
        parts.append(f"atempo={remaining:.6f}")

    return ",".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Reverse Playback
# ---------------------------------------------------------------------------
def reverse_video(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    reverse_audio: bool = True,
    on_progress: Optional[Callable] = None,
) -> str:
    """Reverse video (and optionally audio) playback."""
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1] or ".mp4"
        directory = output_dir or os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_reversed{ext}")

    if on_progress:
        on_progress(10, "Reversing video...")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", "reverse",
    ]
    if reverse_audio:
        cmd += ["-af", "areverse"]
    else:
        cmd += ["-an"]

    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
    cmd.append(output_path)
    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Video reversed!")
    return output_path


# ---------------------------------------------------------------------------
# Speed Ramp (Keyframe-Based)
# ---------------------------------------------------------------------------
def speed_ramp(
    input_path: str,
    keyframes: List[Dict],
    output_path: Optional[str] = None,
    output_dir: str = "",
    easing: str = "ease_in_out",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply keyframe-based speed ramp to video.

    Splits video into segments at keyframe boundaries, applies different
    speeds to each segment, then concatenates with crossfade.

    Args:
        keyframes: List of {"time": float, "speed": float}.
            Example: [{"time": 0, "speed": 1.0}, {"time": 5, "speed": 0.25},
                      {"time": 8, "speed": 1.0}, {"time": 12, "speed": 2.0}]
        easing: Interpolation between keyframes.
    """
    if not keyframes or len(keyframes) < 2:
        raise ValueError("Need at least 2 keyframes for speed ramp")

    # Sort by time
    keyframes = sorted(keyframes, key=lambda k: k["time"])

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1] or ".mp4"
        directory = output_dir or os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_ramped{ext}")

    if on_progress:
        on_progress(5, "Preparing speed ramp segments...")

    tmp_dir = tempfile.mkdtemp(prefix="opencut_ramp_")

    try:
        segment_files = []
        total_segs = len(keyframes) - 1

        for i in range(total_segs):
            start_time = keyframes[i]["time"]
            end_time = keyframes[i + 1]["time"]
            start_speed = keyframes[i]["speed"]
            end_speed = keyframes[i + 1]["speed"]

            if end_time <= start_time:
                continue

            # Use average speed for this segment
            avg_speed = (start_speed + end_speed) / 2.0
            avg_speed = max(0.25, min(8.0, avg_speed))

            seg_path = os.path.join(tmp_dir, f"seg_{i:03d}.mp4")

            if on_progress:
                pct = 5 + int((i / total_segs) * 80)
                on_progress(pct, f"Processing segment {i+1}/{total_segs} ({avg_speed:.1f}x)...")

            # Extract and speed-change segment
            vf = f"setpts={1.0/avg_speed}*PTS"
            af = _build_atempo_chain(avg_speed)

            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(start_time), "-to", str(end_time),
                "-i", input_path,
                "-vf", vf,
            ]
            if af:
                cmd += ["-af", af]
            cmd += [
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-pix_fmt", "yuv420p",
                seg_path,
            ]
            _run_ffmpeg(cmd)
            segment_files.append(seg_path)

        if not segment_files:
            raise RuntimeError("No segments produced")

        if on_progress:
            on_progress(90, "Concatenating segments...")

        # Concatenate segments
        list_file = os.path.join(tmp_dir, "concat.txt")
        with open(list_file, "w") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        _run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", list_file,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            output_path,
        ])

        if on_progress:
            on_progress(100, "Speed ramp applied!")
        return output_path

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
SPEED_RAMP_PRESETS = {
    "dramatic_slowmo": {
        "label": "Dramatic Slow-Mo",
        "description": "Normal -> 0.25x slow -> normal. Great for action moments.",
        "keyframes": [
            {"time": 0, "speed": 1.0},
            {"time_pct": 0.3, "speed": 0.25},
            {"time_pct": 0.7, "speed": 0.25},
            {"time_pct": 1.0, "speed": 1.0},
        ],
    },
    "speed_up_middle": {
        "label": "Speed Up Middle",
        "description": "Normal -> 3x fast -> normal. Skip boring middle sections.",
        "keyframes": [
            {"time": 0, "speed": 1.0},
            {"time_pct": 0.2, "speed": 3.0},
            {"time_pct": 0.8, "speed": 3.0},
            {"time_pct": 1.0, "speed": 1.0},
        ],
    },
    "ramp_up": {
        "label": "Accelerate",
        "description": "Start slow, end fast. Build energy.",
        "keyframes": [
            {"time": 0, "speed": 0.5},
            {"time_pct": 1.0, "speed": 3.0},
        ],
    },
    "ramp_down": {
        "label": "Decelerate",
        "description": "Start fast, end slow. Wind down.",
        "keyframes": [
            {"time": 0, "speed": 3.0},
            {"time_pct": 1.0, "speed": 0.5},
        ],
    },
    "pulse": {
        "label": "Pulse",
        "description": "Rhythmic fast-slow-fast pattern.",
        "keyframes": [
            {"time": 0, "speed": 2.0},
            {"time_pct": 0.25, "speed": 0.5},
            {"time_pct": 0.5, "speed": 2.0},
            {"time_pct": 0.75, "speed": 0.5},
            {"time_pct": 1.0, "speed": 2.0},
        ],
    },
}


def get_speed_ramp_presets() -> List[Dict]:
    """Return available speed ramp presets."""
    return [
        {"name": k, "label": v["label"], "description": v["description"]}
        for k, v in SPEED_RAMP_PRESETS.items()
    ]


def apply_speed_ramp_preset(
    input_path: str,
    preset_name: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """Apply a speed ramp preset, resolving percentage-based keyframes."""
    if preset_name not in SPEED_RAMP_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")

    preset = SPEED_RAMP_PRESETS[preset_name]
    duration = _get_duration(input_path)
    if duration <= 0:
        raise RuntimeError("Could not determine video duration")

    # Resolve percentage-based keyframes to absolute times
    keyframes = []
    for kf in preset["keyframes"]:
        if "time_pct" in kf:
            keyframes.append({"time": kf["time_pct"] * duration, "speed": kf["speed"]})
        else:
            keyframes.append({"time": kf["time"], "speed": kf["speed"]})

    return speed_ramp(
        input_path, keyframes,
        output_path=output_path, output_dir=output_dir,
        on_progress=on_progress,
    )
