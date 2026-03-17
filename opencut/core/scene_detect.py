"""
OpenCut Scene Detection

Detects scene boundaries in video using FFmpeg's scene change detection.
Also provides chapter marker generation for YouTube descriptions.

Uses FFmpeg only - no additional dependencies required.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


@dataclass
class SceneBoundary:
    """A detected scene change boundary."""
    time: float
    frame: int = 0
    score: float = 0.0  # Scene change score (0.0 - 1.0)
    label: str = ""


@dataclass
class SceneInfo:
    """Complete scene detection results."""
    boundaries: List[SceneBoundary] = field(default_factory=list)
    total_scenes: int = 0
    duration: float = 0.0
    avg_scene_length: float = 0.0


def detect_scenes(
    input_path: str,
    threshold: float = 0.3,
    min_scene_length: float = 2.0,
    on_progress: Optional[Callable] = None,
) -> SceneInfo:
    """
    Detect scene boundaries in a video file.

    Uses FFmpeg's scene detection filter to find frame-level
    changes that indicate scene transitions.

    Args:
        input_path: Source video file.
        threshold: Scene change threshold (0.0-1.0). Lower = more sensitive.
        min_scene_length: Minimum scene duration in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        SceneInfo with detected boundaries.
    """
    if on_progress:
        on_progress(10, "Analyzing video for scene changes...")

    # Get video duration first
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        input_path,
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    duration = 0.0
    try:
        probe_data = json.loads(probe_result.stdout)
        duration = float(probe_data.get("format", {}).get("duration", 0.0))
    except (json.JSONDecodeError, ValueError):
        pass

    if on_progress:
        on_progress(20, "Running scene detection filter...")

    # Use FFmpeg select filter to detect scene changes
    # This outputs timestamps where scene changes occur
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info",
        "-i", input_path,
        "-vf", f"select='gt(scene,{max(0.0, min(1.0, float(threshold)))})',showinfo",
        "-f", "null", "-",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if on_progress:
        on_progress(70, "Parsing scene boundaries...")

    # Parse showinfo output for timestamps
    boundaries = [SceneBoundary(time=0.0, frame=0, score=1.0, label="Start")]

    _pts_re = re.compile(r"pts_time:\s*([\d.]+)")
    _frame_re = re.compile(r"\bn:\s*(\d+)")

    for line in result.stderr.splitlines():
        if "showinfo" not in line:
            continue
        pts_match = _pts_re.search(line)
        if not pts_match:
            continue
        try:
            time_val = float(pts_match.group(1))

            # Check minimum spacing from last boundary
            if boundaries and (time_val - boundaries[-1].time) < min_scene_length:
                continue

            # Extract frame number if available
            frame_num = 0
            frame_match = _frame_re.search(line)
            if frame_match:
                frame_num = int(frame_match.group(1))

            boundaries.append(SceneBoundary(
                time=time_val,
                frame=frame_num,
                score=threshold,
            ))

        except (ValueError, IndexError):
            continue

    if on_progress:
        on_progress(90, "Finalizing scene analysis...")

    total_scenes = len(boundaries)
    avg_scene = duration / total_scenes if total_scenes > 0 else duration

    # Label scenes sequentially
    for i, b in enumerate(boundaries):
        if not b.label:
            b.label = f"Scene {i + 1}"

    info = SceneInfo(
        boundaries=boundaries,
        total_scenes=total_scenes,
        duration=duration,
        avg_scene_length=avg_scene,
    )

    if on_progress:
        on_progress(100, f"Found {total_scenes} scenes")

    return info


def generate_chapter_markers(
    scenes: SceneInfo,
    format: str = "youtube",
) -> str:
    """
    Generate chapter markers from scene boundaries.

    Args:
        scenes: SceneInfo from detect_scenes().
        format: Output format ("youtube", "markdown", "json").

    Returns:
        Formatted chapter markers string.
    """
    if format == "youtube":
        # YouTube chapter format: "00:00 Title"
        lines = []
        for i, boundary in enumerate(scenes.boundaries):
            t = boundary.time
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)

            if h > 0:
                timestamp = f"{h:d}:{m:02d}:{s:02d}"
            else:
                timestamp = f"{m:d}:{s:02d}"

            label = boundary.label or f"Scene {i + 1}"
            lines.append(f"{timestamp} {label}")

        return "\n".join(lines)

    elif format == "markdown":
        lines = ["# Chapters", ""]
        for i, boundary in enumerate(scenes.boundaries):
            t = boundary.time
            m = int(t // 60)
            s = int(t % 60)
            label = boundary.label or f"Scene {i + 1}"
            lines.append(f"- **{m:02d}:{s:02d}** - {label}")
        return "\n".join(lines)

    elif format == "json":
        chapters = []
        for i, boundary in enumerate(scenes.boundaries):
            chapters.append({
                "time": round(boundary.time, 3),
                "label": boundary.label or f"Scene {i + 1}",
                "frame": boundary.frame,
            })
        return json.dumps(chapters, indent=2)

    else:
        raise ValueError(f"Unknown chapter format: {format}")


def generate_speed_ramp(
    duration: float,
    preset: str = "dramatic_pause",
) -> List[Dict]:
    """
    Generate speed ramp keyframes for a clip.

    Args:
        duration: Total clip duration in seconds.
        preset: Speed ramp preset name.

    Returns:
        List of keyframe dicts: {"time": float, "speed": float}
    """
    SPEED_PRESETS = {
        "dramatic_pause": [
            {"pos": 0.0, "speed": 1.0},
            {"pos": 0.4, "speed": 1.0},
            {"pos": 0.45, "speed": 0.3},
            {"pos": 0.55, "speed": 0.3},
            {"pos": 0.6, "speed": 1.0},
            {"pos": 1.0, "speed": 1.0},
        ],
        "smooth_ramp_up": [
            {"pos": 0.0, "speed": 0.5},
            {"pos": 0.3, "speed": 0.7},
            {"pos": 0.7, "speed": 1.0},
            {"pos": 1.0, "speed": 1.5},
        ],
        "smooth_ramp_down": [
            {"pos": 0.0, "speed": 1.5},
            {"pos": 0.3, "speed": 1.0},
            {"pos": 0.7, "speed": 0.7},
            {"pos": 1.0, "speed": 0.5},
        ],
        "punch_in": [
            {"pos": 0.0, "speed": 0.5},
            {"pos": 0.15, "speed": 0.3},
            {"pos": 0.2, "speed": 2.0},
            {"pos": 0.5, "speed": 1.0},
            {"pos": 1.0, "speed": 1.0},
        ],
        "bullet_time": [
            {"pos": 0.0, "speed": 1.0},
            {"pos": 0.3, "speed": 1.0},
            {"pos": 0.35, "speed": 0.15},
            {"pos": 0.65, "speed": 0.15},
            {"pos": 0.7, "speed": 1.0},
            {"pos": 1.0, "speed": 1.0},
        ],
        "heartbeat": [
            {"pos": 0.0, "speed": 1.0},
            {"pos": 0.2, "speed": 0.4},
            {"pos": 0.3, "speed": 1.2},
            {"pos": 0.5, "speed": 0.4},
            {"pos": 0.6, "speed": 1.2},
            {"pos": 0.8, "speed": 0.4},
            {"pos": 0.9, "speed": 1.0},
            {"pos": 1.0, "speed": 1.0},
        ],
        "flash_forward": [
            {"pos": 0.0, "speed": 1.0},
            {"pos": 0.1, "speed": 3.0},
            {"pos": 0.3, "speed": 3.0},
            {"pos": 0.4, "speed": 0.5},
            {"pos": 0.9, "speed": 0.5},
            {"pos": 1.0, "speed": 1.0},
        ],
        "rewind": [
            {"pos": 0.0, "speed": 1.0},
            {"pos": 0.4, "speed": 1.0},
            {"pos": 0.5, "speed": -2.0},
            {"pos": 0.7, "speed": -2.0},
            {"pos": 0.8, "speed": 1.0},
            {"pos": 1.0, "speed": 1.0},
        ],
    }

    preset_data = SPEED_PRESETS.get(preset, SPEED_PRESETS["dramatic_pause"])

    keyframes = []
    for kf in preset_data:
        keyframes.append({
            "time": round(kf["pos"] * duration, 3),
            "speed": kf["speed"],
        })

    return keyframes


# ---------------------------------------------------------------------------
# ML Scene Detection (TransNetV2)
# ---------------------------------------------------------------------------
def detect_scenes_ml(
    input_path: str,
    threshold: float = 0.5,
    min_scene_length: float = 2.0,
    on_progress: Optional[Callable] = None,
) -> SceneInfo:
    """
    Detect scene boundaries using TransNetV2 neural network.

    More accurate than FFmpeg's threshold filter, especially for gradual
    transitions, fades, and complex cuts.

    Args:
        input_path: Source video file.
        threshold: Prediction threshold (0.0-1.0). Lower = more sensitive.
        min_scene_length: Minimum scene duration in seconds.
        on_progress: Progress callback(pct, msg).

    Returns:
        SceneInfo with detected boundaries.

    Requires: pip install transnetv2
    """
    try:
        from transnetv2 import TransNetV2
    except ImportError:
        raise RuntimeError(
            "TransNetV2 not installed. Install with: pip install transnetv2"
        )

    try:
        import numpy as np
    except ImportError:
        raise RuntimeError("numpy required. Install with: pip install numpy")

    if on_progress:
        on_progress(5, "Loading TransNetV2 model...")

    # Get video info
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        "-select_streams", "v:0",
        input_path,
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    duration = 0.0
    fps = 25.0
    try:
        probe_data = json.loads(probe_result.stdout)
        fmt = probe_data.get("format", {})
        duration = float(fmt.get("duration", 0.0))
        streams = probe_data.get("streams", [])
        if streams:
            fps_str = streams[0].get("r_frame_rate", "25/1")
            parts = fps_str.split("/")
            if len(parts) == 2 and float(parts[1]) > 0:
                fps = float(parts[0]) / float(parts[1])
            else:
                fps = 25.0
    except (json.JSONDecodeError, ValueError, IndexError):
        pass

    if on_progress:
        on_progress(10, "Extracting frames for analysis...")

    # Extract frames as raw RGB using FFmpeg
    # TransNetV2 expects 48x27 resolution frames
    frame_w, frame_h = 48, 27
    extract_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vf", f"scale={frame_w}:{frame_h}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-",
    ]

    try:
        result = subprocess.run(
            extract_cmd, capture_output=True,
            timeout=max(300, int(duration * 2)),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Frame extraction timed out")

    if result.returncode != 0:
        raise RuntimeError(f"Frame extraction failed: {result.stderr.decode(errors='replace')[-300:]}")

    raw_frames = result.stdout
    frame_size = frame_w * frame_h * 3
    n_frames = len(raw_frames) // frame_size

    if n_frames < 2:
        return SceneInfo(
            boundaries=[SceneBoundary(time=0.0, frame=0, score=1.0, label="Start")],
            total_scenes=1,
            duration=duration,
            avg_scene_length=duration,
        )

    if on_progress:
        on_progress(30, f"Analyzing {n_frames} frames with TransNetV2...")

    # Convert to numpy array
    frames = np.frombuffer(raw_frames[:n_frames * frame_size], dtype=np.uint8)
    frames = frames.reshape(n_frames, frame_h, frame_w, 3)

    # Run TransNetV2
    model = TransNetV2()
    predictions, _ = model.predict_frames(frames)

    if on_progress:
        on_progress(80, "Processing predictions...")

    # Find scene boundaries from predictions
    boundaries = [SceneBoundary(time=0.0, frame=0, score=1.0, label="Start")]

    min_frames = int(min_scene_length * fps)
    last_boundary_frame = 0

    for i, score in enumerate(predictions):
        pred_score = float(score)
        if pred_score >= threshold:
            # Check minimum distance from last boundary
            if (i - last_boundary_frame) >= min_frames:
                time_val = i / fps
                boundaries.append(SceneBoundary(
                    time=round(time_val, 3),
                    frame=i,
                    score=round(pred_score, 4),
                ))
                last_boundary_frame = i

    # Label scenes
    for i, b in enumerate(boundaries):
        if not b.label:
            b.label = f"Scene {i + 1}"

    total_scenes = len(boundaries)
    avg_scene = duration / total_scenes if total_scenes > 0 else duration

    if on_progress:
        on_progress(100, f"Found {total_scenes} scenes (ML)")

    return SceneInfo(
        boundaries=boundaries,
        total_scenes=total_scenes,
        duration=duration,
        avg_scene_length=round(avg_scene, 2),
    )


SPEED_RAMP_PRESETS = [
    {"name": "dramatic_pause", "label": "Dramatic Pause", "description": "Slow down in the middle for impact"},
    {"name": "smooth_ramp_up", "label": "Smooth Ramp Up", "description": "Gradually accelerate"},
    {"name": "smooth_ramp_down", "label": "Smooth Ramp Down", "description": "Gradually decelerate"},
    {"name": "punch_in", "label": "Punch In", "description": "Slow buildup then fast hit"},
    {"name": "bullet_time", "label": "Bullet Time", "description": "Matrix-style slow motion"},
    {"name": "heartbeat", "label": "Heartbeat", "description": "Rhythmic speed pulsing"},
    {"name": "flash_forward", "label": "Flash Forward", "description": "Quick flash then slow reveal"},
    {"name": "rewind", "label": "Rewind", "description": "Reverse playback section"},
]
