"""
OpenCut Speed Ramp Engine

Applies variable-speed effects to video clips using FFmpeg.
Supports preset speed curves and custom Bezier-based keyframes.

Two output modes:
  1. Render: Produces a new video file with baked-in speed changes
  2. XML:    Generates FCP XML with velocity keyframes for Premiere import

Uses FFmpeg only - no additional dependencies required.
"""

import json
import logging
import math
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SpeedKeyframe:
    """A single keyframe in a speed ramp curve."""
    time: float       # Absolute time in seconds
    speed: float      # Speed multiplier (1.0 = normal, 0.5 = half, 2.0 = double)
    ease: str = "linear"  # "linear", "ease_in", "ease_out", "ease_in_out"


@dataclass
class SpeedRampResult:
    """Result of a speed ramp operation."""
    output_path: str = ""
    xml_path: str = ""
    original_duration: float = 0.0
    output_duration: float = 0.0
    segments_processed: int = 0
    preset_used: str = ""


# ---------------------------------------------------------------------------
# Presets (expanded from scene_detect.py)
# ---------------------------------------------------------------------------
SPEED_PRESETS = {
    "ramp_in": {
        "label": "Ramp In",
        "description": "Gradually accelerate from slow to normal",
        "keyframes": [
            {"pos": 0.0, "speed": 0.4},
            {"pos": 0.3, "speed": 0.6},
            {"pos": 0.6, "speed": 0.85},
            {"pos": 1.0, "speed": 1.0},
        ],
    },
    "ramp_out": {
        "label": "Ramp Out",
        "description": "Gradually decelerate from normal to slow",
        "keyframes": [
            {"pos": 0.0, "speed": 1.0},
            {"pos": 0.4, "speed": 0.85},
            {"pos": 0.7, "speed": 0.6},
            {"pos": 1.0, "speed": 0.4},
        ],
    },
    "pulse": {
        "label": "Pulse",
        "description": "Fast-slow-fast rhythmic speed change",
        "keyframes": [
            {"pos": 0.0, "speed": 1.5},
            {"pos": 0.25, "speed": 0.4},
            {"pos": 0.5, "speed": 1.5},
            {"pos": 0.75, "speed": 0.4},
            {"pos": 1.0, "speed": 1.5},
        ],
    },
    "heartbeat": {
        "label": "Heartbeat",
        "description": "Rhythmic pulse like a heartbeat",
        "keyframes": [
            {"pos": 0.0, "speed": 1.0},
            {"pos": 0.15, "speed": 0.3},
            {"pos": 0.25, "speed": 1.3},
            {"pos": 0.4, "speed": 0.3},
            {"pos": 0.5, "speed": 1.3},
            {"pos": 0.7, "speed": 1.0},
            {"pos": 1.0, "speed": 1.0},
        ],
    },
    "smooth_slow": {
        "label": "Smooth Slow-Mo",
        "description": "Gentle slow motion in the middle",
        "keyframes": [
            {"pos": 0.0, "speed": 1.0},
            {"pos": 0.3, "speed": 1.0},
            {"pos": 0.4, "speed": 0.3},
            {"pos": 0.6, "speed": 0.3},
            {"pos": 0.7, "speed": 1.0},
            {"pos": 1.0, "speed": 1.0},
        ],
    },
    "dramatic_pause": {
        "label": "Dramatic Pause",
        "description": "Slow down for impact in the center",
        "keyframes": [
            {"pos": 0.0, "speed": 1.0},
            {"pos": 0.4, "speed": 1.0},
            {"pos": 0.45, "speed": 0.2},
            {"pos": 0.55, "speed": 0.2},
            {"pos": 0.6, "speed": 1.0},
            {"pos": 1.0, "speed": 1.0},
        ],
    },
    "punch_in": {
        "label": "Punch In",
        "description": "Slow buildup then fast hit",
        "keyframes": [
            {"pos": 0.0, "speed": 0.5},
            {"pos": 0.15, "speed": 0.3},
            {"pos": 0.2, "speed": 2.5},
            {"pos": 0.5, "speed": 1.2},
            {"pos": 1.0, "speed": 1.0},
        ],
    },
    "bullet_time": {
        "label": "Bullet Time",
        "description": "Matrix-style extreme slow motion",
        "keyframes": [
            {"pos": 0.0, "speed": 1.0},
            {"pos": 0.3, "speed": 1.0},
            {"pos": 0.35, "speed": 0.1},
            {"pos": 0.65, "speed": 0.1},
            {"pos": 0.7, "speed": 1.0},
            {"pos": 1.0, "speed": 1.0},
        ],
    },
    "flash_forward": {
        "label": "Flash Forward",
        "description": "Quick flash then slow reveal",
        "keyframes": [
            {"pos": 0.0, "speed": 1.0},
            {"pos": 0.05, "speed": 4.0},
            {"pos": 0.3, "speed": 4.0},
            {"pos": 0.35, "speed": 0.5},
            {"pos": 0.9, "speed": 0.5},
            {"pos": 1.0, "speed": 1.0},
        ],
    },
    "boomerang": {
        "label": "Boomerang",
        "description": "Fast-slow-fast like a boomerang arc",
        "keyframes": [
            {"pos": 0.0, "speed": 2.0},
            {"pos": 0.2, "speed": 1.0},
            {"pos": 0.4, "speed": 0.3},
            {"pos": 0.5, "speed": 0.2},
            {"pos": 0.6, "speed": 0.3},
            {"pos": 0.8, "speed": 1.0},
            {"pos": 1.0, "speed": 2.0},
        ],
    },
}


def get_speed_presets() -> List[Dict]:
    """Return all available speed ramp presets as a list of dicts."""
    return [
        {
            "name": name,
            "label": data["label"],
            "description": data["description"],
            "keyframe_count": len(data["keyframes"]),
        }
        for name, data in SPEED_PRESETS.items()
    ]


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------
def _probe_duration(filepath: str) -> float:
    """Get media duration in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        data = json.loads(result.stdout)
        # Try format duration first, then first stream
        dur = data.get("format", {}).get("duration")
        if dur:
            return float(dur)
        for s in data.get("streams", []):
            if s.get("duration"):
                return float(s["duration"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return 0.0


def _probe_video_info(filepath: str) -> Dict:
    """Get video stream info (fps, width, height, codec)."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    info = {"fps": 30.0, "width": 1920, "height": 1080, "codec": "h264"}
    try:
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        info["width"] = int(stream.get("width", 1920))
        info["height"] = int(stream.get("height", 1080))
        info["codec"] = stream.get("codec_name", "h264")

        # Parse frame rate
        r_frame = stream.get("r_frame_rate", "30/1")
        if "/" in str(r_frame):
            num, den = r_frame.split("/")
            info["fps"] = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            info["fps"] = float(r_frame)
    except (json.JSONDecodeError, ValueError, TypeError, IndexError):
        pass
    return info


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------
def _interpolate_speed(keyframes: List[Dict], t_normalized: float) -> float:
    """
    Interpolate speed at a normalized position (0.0-1.0)
    using the keyframe curve. Linear interpolation between keyframes.
    """
    if not keyframes:
        return 1.0

    # Clamp
    t_normalized = max(0.0, min(1.0, t_normalized))

    # Find surrounding keyframes
    for i in range(len(keyframes) - 1):
        kf_a = keyframes[i]
        kf_b = keyframes[i + 1]
        if kf_a["pos"] <= t_normalized <= kf_b["pos"]:
            span = kf_b["pos"] - kf_a["pos"]
            if span <= 0:
                return kf_a["speed"]
            frac = (t_normalized - kf_a["pos"]) / span
            return kf_a["speed"] + frac * (kf_b["speed"] - kf_a["speed"])

    return keyframes[-1]["speed"]


def _build_segments(
    duration: float,
    keyframes: List[Dict],
    segment_count: int = 20,
) -> List[Dict]:
    """
    Break the clip into segments, each with an average speed.
    Returns list of {"start": float, "end": float, "speed": float}.
    """
    segments = []
    seg_len = duration / segment_count

    for i in range(segment_count):
        start = i * seg_len
        end = min((i + 1) * seg_len, duration)
        mid = (start + end) / 2.0
        t_norm = mid / duration if duration > 0 else 0.5
        speed = _interpolate_speed(keyframes, t_norm)
        speed = max(0.1, min(10.0, speed))  # Clamp to safe range
        segments.append({"start": round(start, 4), "end": round(end, 4), "speed": round(speed, 3)})

    return segments


# ---------------------------------------------------------------------------
# FFmpeg rendering
# ---------------------------------------------------------------------------
def apply_speed_ramp(
    input_path: str,
    preset: str = "smooth_slow",
    custom_keyframes: Optional[List[Dict]] = None,
    output_dir: str = "",
    segment_count: int = 20,
    quality: str = "medium",
    on_progress: Optional[Callable] = None,
) -> SpeedRampResult:
    """
    Render a video with variable speed applied via FFmpeg.

    Splits the clip into N segments, each with its interpolated speed,
    then concatenates them. This produces a rendered output video.

    Args:
        input_path:       Source video file.
        preset:           Speed preset name (ignored if custom_keyframes).
        custom_keyframes: Custom keyframes [{"pos": 0.0-1.0, "speed": float}].
        output_dir:       Output directory (defaults to input directory).
        segment_count:    Number of segments to split into (more = smoother).
        quality:          Encoding quality ("low", "medium", "high").
        on_progress:      Callback(pct, msg).

    Returns:
        SpeedRampResult with output path and metadata.
    """
    if on_progress:
        on_progress(5, "Analyzing source video...")

    duration = _probe_duration(input_path)
    if duration <= 0:
        raise ValueError("Could not determine video duration")

    video_info = _probe_video_info(input_path)

    # Get keyframes
    if custom_keyframes:
        keyframes = custom_keyframes
        preset_name = "custom"
    else:
        preset_data = SPEED_PRESETS.get(preset, SPEED_PRESETS["smooth_slow"])
        keyframes = preset_data["keyframes"]
        preset_name = preset

    # Build segments
    segments = _build_segments(duration, keyframes, segment_count)

    if on_progress:
        on_progress(10, f"Processing {len(segments)} speed segments...")

    # Output path
    if not output_dir:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base}_speedramp_{preset_name}.mp4")

    # Quality settings
    crf_map = {"low": "28", "medium": "23", "high": "18", "lossless": "0"}
    crf = crf_map.get(quality, "23")

    # Build FFmpeg filter_complex with trim+setpts+atempo for each segment
    # then concat them all together
    filter_parts = []
    concat_inputs = []

    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        speed = seg["speed"]

        # Video: trim and adjust PTS
        pts_factor = 1.0 / speed
        filter_parts.append(
            f"[0:v]trim=start={start}:end={end},setpts={pts_factor:.6f}*(PTS-STARTPTS)[v{i}]"
        )

        # Audio: trim and adjust tempo
        # atempo only accepts 0.5 to 100.0, chain for extremes
        atempo_chain = _build_atempo_chain(speed)
        filter_parts.append(
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS,{atempo_chain}[a{i}]"
        )

        concat_inputs.append(f"[v{i}][a{i}]")

        if on_progress:
            pct = 10 + int((i / len(segments)) * 70)
            on_progress(pct, f"Building segment {i + 1}/{len(segments)} (speed: {speed}x)")

    # Concat all segments
    concat_str = "".join(concat_inputs)
    filter_parts.append(f"{concat_str}concat=n={len(segments)}:v=1:a=1[outv][outa]")

    filter_complex = ";".join(filter_parts)

    if on_progress:
        on_progress(80, "Encoding output video...")

    # Build FFmpeg command
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-y",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-crf", crf, "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error(f"FFmpeg speed ramp error: {result.stderr[-1000:]}")
            raise RuntimeError(f"FFmpeg encoding failed: {result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Speed ramp encoding timed out (>60 minutes)")

    output_duration = _probe_duration(output_path)

    if on_progress:
        on_progress(100, "Speed ramp complete")

    return SpeedRampResult(
        output_path=output_path,
        original_duration=duration,
        output_duration=output_duration,
        segments_processed=len(segments),
        preset_used=preset_name,
    )


def _build_atempo_chain(speed: float) -> str:
    """
    Build an atempo filter chain for the given speed multiplier.
    atempo accepts 0.5 to 100.0, so we chain multiple for extremes.
    """
    if speed <= 0.1:
        speed = 0.1

    parts = []
    remaining = speed

    if remaining < 0.5:
        # Chain multiple atempo filters for slow speeds
        while remaining < 0.5:
            parts.append("atempo=0.5")
            remaining /= 0.5
        parts.append(f"atempo={remaining:.6f}")
    elif remaining > 100.0:
        parts.append("atempo=100.0")
    else:
        parts.append(f"atempo={remaining:.6f}")

    return ",".join(parts)


# ---------------------------------------------------------------------------
# FCP XML generation with velocity keyframes
# ---------------------------------------------------------------------------
def generate_speed_xml(
    input_path: str,
    preset: str = "smooth_slow",
    custom_keyframes: Optional[List[Dict]] = None,
    output_dir: str = "",
    sequence_name: str = "",
    on_progress: Optional[Callable] = None,
) -> SpeedRampResult:
    """
    Generate FCP XML with speed ramp velocity keyframes.

    This creates an XML that Premiere Pro can import, containing
    time remapping keyframes on the clip.

    Args:
        input_path:       Source video file.
        preset:           Speed preset name.
        custom_keyframes: Custom keyframes.
        output_dir:       Output directory.
        sequence_name:    Name for the sequence.
        on_progress:      Callback(pct, msg).

    Returns:
        SpeedRampResult with xml_path set.
    """
    if on_progress:
        on_progress(10, "Analyzing source video...")

    duration = _probe_duration(input_path)
    video_info = _probe_video_info(input_path)
    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]

    if duration <= 0:
        raise ValueError("Could not determine video duration")

    # Get keyframes
    if custom_keyframes:
        keyframes = custom_keyframes
        preset_name = "custom"
    else:
        preset_data = SPEED_PRESETS.get(preset, SPEED_PRESETS["smooth_slow"])
        keyframes = preset_data["keyframes"]
        preset_name = preset

    if on_progress:
        on_progress(30, "Generating velocity keyframes...")

    # Convert normalized keyframes to absolute time + frame counts
    total_frames = int(duration * fps)

    # Timebase for FCP XML
    timebase = int(round(fps))
    if abs(fps - 29.97) < 0.1:
        timebase = 30
        ntsc = "TRUE"
    elif abs(fps - 23.976) < 0.1:
        timebase = 24
        ntsc = "TRUE"
    elif abs(fps - 59.94) < 0.1:
        timebase = 60
        ntsc = "TRUE"
    else:
        ntsc = "FALSE"

    # Build velocity keyframe XML entries
    velocity_kf_xml = ""
    for kf in keyframes:
        frame_num = int(kf["pos"] * total_frames)
        speed_pct = kf["speed"] * 100.0  # FCP uses percentage
        velocity_kf_xml += f"""
                                <keyframe>
                                    <when>
                                        <timebase>{timebase}</timebase>
                                        <ntsc>{ntsc}</ntsc>
                                        <samplecount>{frame_num}</samplecount>
                                    </when>
                                    <value>{speed_pct:.1f}</value>
                                    <interpolation>
                                        <name>Linear</name>
                                    </interpolation>
                                </keyframe>"""

    # Build sequence name
    if not sequence_name:
        base = os.path.splitext(os.path.basename(input_path))[0]
        sequence_name = f"{base} - Speed Ramp ({preset_name})"

    # Build file reference
    file_path = input_path.replace("\\", "/")
    if not file_path.startswith("file://"):
        file_path = "file://localhost/" + file_path.lstrip("/")

    # Calculate output duration (approximate based on average speed)
    avg_speed = sum(kf["speed"] for kf in keyframes) / len(keyframes) if keyframes else 1.0
    out_duration_frames = int(total_frames / avg_speed) if avg_speed > 0 else total_frames

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="4">
    <sequence id="speedramp">
        <uuid>{uuid.uuid4()}</uuid>
        <name>{_xml_escape(sequence_name)}</name>
        <duration>{out_duration_frames}</duration>
        <rate>
            <timebase>{timebase}</timebase>
            <ntsc>{ntsc}</ntsc>
        </rate>
        <media>
            <video>
                <format>
                    <samplecharacteristics>
                        <width>{width}</width>
                        <height>{height}</height>
                        <anamorphic>FALSE</anamorphic>
                        <pixelaspectratio>square</pixelaspectratio>
                        <fielddominance>none</fielddominance>
                        <rate>
                            <timebase>{timebase}</timebase>
                            <ntsc>{ntsc}</ntsc>
                        </rate>
                    </samplecharacteristics>
                </format>
                <track>
                    <clipitem id="speedramp_clip">
                        <name>{_xml_escape(os.path.basename(input_path))}</name>
                        <duration>{total_frames}</duration>
                        <rate>
                            <timebase>{timebase}</timebase>
                            <ntsc>{ntsc}</ntsc>
                        </rate>
                        <start>0</start>
                        <end>{out_duration_frames}</end>
                        <in>0</in>
                        <out>{total_frames}</out>
                        <file id="file-1">
                            <name>{_xml_escape(os.path.basename(input_path))}</name>
                            <pathurl>{_xml_escape(file_path)}</pathurl>
                            <duration>{total_frames}</duration>
                            <rate>
                                <timebase>{timebase}</timebase>
                                <ntsc>{ntsc}</ntsc>
                            </rate>
                            <media>
                                <video>
                                    <samplecharacteristics>
                                        <width>{width}</width>
                                        <height>{height}</height>
                                    </samplecharacteristics>
                                </video>
                                <audio>
                                    <samplecharacteristics>
                                        <samplerate>48000</samplerate>
                                        <depth>16</depth>
                                    </samplecharacteristics>
                                </audio>
                            </media>
                        </file>
                        <filter>
                            <effect>
                                <name>Time Remap</name>
                                <effectid>timeremap</effectid>
                                <effecttype>motion</effecttype>
                                <parameter>
                                    <parameterid>speed</parameterid>
                                    <name>speed</name>
                                    <value>100</value>
                                    <keyframe>
                                        <when>
                                            <timebase>{timebase}</timebase>
                                            <ntsc>{ntsc}</ntsc>
                                            <samplecount>0</samplecount>
                                        </when>
                                        <value>100.0</value>
                                    </keyframe>{velocity_kf_xml}
                                </parameter>
                            </effect>
                        </filter>
                    </clipitem>
                </track>
            </video>
            <audio>
                <track>
                    <clipitem id="speedramp_audio">
                        <name>{_xml_escape(os.path.basename(input_path))}</name>
                        <duration>{total_frames}</duration>
                        <rate>
                            <timebase>{timebase}</timebase>
                            <ntsc>{ntsc}</ntsc>
                        </rate>
                        <start>0</start>
                        <end>{out_duration_frames}</end>
                        <in>0</in>
                        <out>{total_frames}</out>
                        <file id="file-1"/>
                    </clipitem>
                </track>
            </audio>
        </media>
    </sequence>
</xmeml>
"""

    # Write XML
    if not output_dir:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    xml_path = os.path.join(output_dir, f"{base}_speedramp_{preset_name}.xml")

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    if on_progress:
        on_progress(100, "Speed ramp XML generated")

    return SpeedRampResult(
        xml_path=xml_path,
        original_duration=duration,
        output_duration=duration / avg_speed if avg_speed > 0 else duration,
        segments_processed=len(keyframes),
        preset_used=preset_name,
    )


def _xml_escape(text: str) -> str:
    """Escape special characters for XML."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
