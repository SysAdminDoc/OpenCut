"""
OpenCut Smart Defaults - Clip Analysis & Optimal Parameter Suggestion

Analyze clip properties (resolution, fps, codec, audio levels, motion, etc.)
and suggest optimal parameters for any processing operation. Uses FFprobe
for metadata and FFmpeg signalstats/ebur128 for heuristic content detection.
"""

import json
import logging
import os
import re
import subprocess as _sp
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path

logger = logging.getLogger("opencut")


@dataclass
class ClipProfile:
    """Analyzed clip properties used for smart default suggestions."""
    avg_loudness_lufs: Optional[float] = None
    peak_db: Optional[float] = None
    resolution: int = 0
    fps: float = 0.0
    codec: str = ""
    duration_s: float = 0.0
    has_audio: bool = False
    has_video: bool = False
    is_static_camera: bool = False
    detected_content_type: str = "unknown"
    width: int = 0
    height: int = 0
    audio_channels: int = 0
    bitrate_kbps: int = 0
    pixel_format: str = ""
    has_alpha: bool = False
    rotation: int = 0
    sample_rate: int = 0


CONTENT_TYPES = [
    "interview",
    "music_video",
    "screen_recording",
    "drone",
    "vlog",
    "gaming",
    "tutorial",
    "presentation",
    "sports",
    "unknown",
]


def _probe_metadata(video_path: str) -> dict:
    """Probe video file for stream metadata using ffprobe.

    Returns dict with parsed format and stream info.
    """
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    try:
        result = _sp.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout.decode(errors="replace"))
    except Exception as exc:
        logger.debug("ffprobe metadata failed: %s", exc)
    return {}


def _measure_loudness(video_path: str) -> tuple:
    """Measure audio loudness using FFmpeg ebur128 filter.

    Returns (avg_lufs, peak_db) or (None, None) on failure.
    """
    cmd = [
        get_ffmpeg_path(),
        "-i", video_path,
        "-af", "ebur128=peak=true",
        "-f", "null",
        "-t", "60",  # sample first 60s to keep it fast
        "-",
    ]
    try:
        result = _sp.run(cmd, capture_output=True, timeout=120)
        stderr = result.stderr.decode(errors="replace")

        # Parse integrated loudness
        lufs = None
        peak = None

        lufs_match = re.search(r"I:\s*([-\d.]+)\s*LUFS", stderr)
        if lufs_match:
            lufs = float(lufs_match.group(1))

        peak_match = re.search(r"Peak:\s*([-\d.]+)\s*dBFS", stderr)
        if not peak_match:
            peak_match = re.search(r"True peak:\s*([-\d.]+)\s*dBFS", stderr)
        if peak_match:
            peak = float(peak_match.group(1))

        return (lufs, peak)
    except Exception as exc:
        logger.debug("Loudness measurement failed: %s", exc)
        return (None, None)


def _estimate_motion(video_path: str) -> float:
    """Estimate average motion level using FFmpeg freezedetect.

    Returns a value 0.0 (static) to 1.0 (high motion).
    Uses freeze frame detection as a proxy for camera motion.
    """
    cmd = [
        get_ffmpeg_path(),
        "-i", video_path,
        "-t", "30",  # sample first 30s
        "-vf", "freezedetect=n=-60dB:d=0.5",
        "-f", "null",
        "-",
    ]
    try:
        result = _sp.run(cmd, capture_output=True, timeout=60)
        stderr = result.stderr.decode(errors="replace")

        # Sum freeze durations from FFmpeg's freezedetect filter output —
        # the previous line counted ``freeze_start`` events but discarded
        # the result. Total freeze duration is what feeds the ratio below.
        freeze_duration_matches = re.findall(r"freeze_duration:\s*([\d.]+)", stderr)
        total_freeze = sum(float(d) for d in freeze_duration_matches)

        # If more than 50% is frozen, it's static
        sample_duration = 30.0
        freeze_ratio = min(total_freeze / sample_duration, 1.0)

        # Invert: high freeze ratio = low motion
        return round(1.0 - freeze_ratio, 3)
    except Exception as exc:
        logger.debug("Motion estimation failed: %s", exc)
        return 0.5  # unknown, assume moderate


def _detect_content_type(
    probe_data: dict,
    motion_level: float,
    has_audio: bool,
    duration_s: float,
    width: int,
    height: int,
    fps: float,
) -> str:
    """Heuristic content type detection.

    Uses metadata, motion level, and stream properties to classify content.
    """
    # Check for screen recording indicators
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "video":
            stream.get("codec_name", "").lower()
            break

    fmt = probe_data.get("format", {})
    fmt_tags = fmt.get("tags", {})

    # Screen recording: typically low motion, standard resolutions, certain codecs
    is_screen_res = (width in (1920, 2560, 3840, 1280) and
                     height in (1080, 1440, 2160, 720, 800, 900))
    if motion_level < 0.3 and is_screen_res and fps <= 30:
        # Low motion at standard screen res likely screen recording
        return "screen_recording"

    # Drone: check for GPS metadata or keywords in tags
    all_tags_str = " ".join(str(v) for v in fmt_tags.values()).lower()
    if any(kw in all_tags_str for kw in ("dji", "drone", "mavic", "phantom", "aerial")):
        return "drone"

    # Music video: typically high motion, has audio, shorter duration
    if motion_level > 0.7 and has_audio and 120 < duration_s < 600:
        return "music_video"

    # Interview: low-to-moderate motion, has audio, longer clips
    if motion_level < 0.4 and has_audio and duration_s > 60:
        return "interview"

    # Gaming: high framerate common
    if fps >= 60 and is_screen_res:
        return "gaming"

    # Tutorial: moderate motion, screen-ish resolution, has audio
    if motion_level < 0.5 and has_audio and is_screen_res and duration_s > 120:
        return "tutorial"

    # Presentation: very low motion, has audio
    if motion_level < 0.2 and has_audio and duration_s > 300:
        return "presentation"

    # Vlog: moderate motion, has audio, front-facing camera feel
    if 0.3 <= motion_level <= 0.6 and has_audio and duration_s > 60:
        return "vlog"

    # Sports: very high motion
    if motion_level > 0.8:
        return "sports"

    return "unknown"


def analyze_clip_properties(
    video_path: str,
    on_progress: Optional[Callable] = None,
) -> ClipProfile:
    """Analyze clip properties and return a ClipProfile.

    Probes metadata, measures loudness, estimates motion, and classifies
    content type using heuristics.

    Args:
        video_path: Path to the video file.
        on_progress: Progress callback(pct, msg).

    Returns:
        ClipProfile dataclass with all analyzed properties.

    Raises:
        FileNotFoundError: If video_path does not exist.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    profile = ClipProfile()

    if on_progress:
        on_progress(5, "Probing metadata...")

    # Step 1: FFprobe metadata
    probe_data = _probe_metadata(video_path)
    fmt = probe_data.get("format", {})

    profile.duration_s = float(fmt.get("duration", 0))
    profile.bitrate_kbps = int(float(fmt.get("bit_rate", 0)) / 1000)

    for stream in probe_data.get("streams", []):
        codec_type = stream.get("codec_type", "")
        if codec_type == "video" and not profile.has_video:
            profile.has_video = True
            profile.width = int(stream.get("width", 0))
            profile.height = int(stream.get("height", 0))
            profile.resolution = max(profile.width, profile.height)
            profile.codec = stream.get("codec_name", "")
            profile.pixel_format = stream.get("pix_fmt", "")
            profile.has_alpha = "a" in profile.pixel_format.lower()
            profile.rotation = abs(int(stream.get("rotation", 0)))

            # Parse FPS from r_frame_rate
            fps_str = stream.get("r_frame_rate", "0/1")
            try:
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    den_val = int(den)
                    if den_val > 0:
                        profile.fps = round(int(num) / den_val, 3)
                else:
                    profile.fps = float(fps_str)
            except (ValueError, ZeroDivisionError):
                profile.fps = 0.0

        elif codec_type == "audio" and not profile.has_audio:
            profile.has_audio = True
            profile.audio_channels = int(stream.get("channels", 0))
            profile.sample_rate = int(stream.get("sample_rate", 0))

    if on_progress:
        on_progress(30, "Measuring audio loudness...")

    # Step 2: Audio loudness (only if audio exists)
    if profile.has_audio:
        lufs, peak = _measure_loudness(video_path)
        profile.avg_loudness_lufs = lufs
        profile.peak_db = peak

    if on_progress:
        on_progress(60, "Estimating motion level...")

    # Step 3: Motion estimation (only if video exists)
    motion_level = 0.5
    if profile.has_video:
        motion_level = _estimate_motion(video_path)

    profile.is_static_camera = motion_level < 0.25

    if on_progress:
        on_progress(80, "Classifying content type...")

    # Step 4: Content type classification
    profile.detected_content_type = _detect_content_type(
        probe_data, motion_level, profile.has_audio,
        profile.duration_s, profile.width, profile.height, profile.fps,
    )

    if on_progress:
        on_progress(100, f"Analysis complete: {profile.detected_content_type}")

    return profile


# ---------------------------------------------------------------------------
# Smart Defaults Engine
# ---------------------------------------------------------------------------
def get_smart_defaults(
    operation: str,
    clip_profile: ClipProfile,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Return recommended parameters for an operation based on clip profile.

    Args:
        operation: The operation name (e.g. "normalize", "denoise", "export").
        clip_profile: Analyzed ClipProfile from analyze_clip_properties().
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict of recommended parameter values for the operation.
    """
    if on_progress:
        on_progress(10, f"Computing defaults for {operation}...")

    defaults = {}
    ct = clip_profile.detected_content_type

    if operation == "normalize":
        # Target loudness based on content type
        if ct == "music_video":
            defaults["target_lufs"] = -14.0
            defaults["true_peak"] = -1.0
        elif ct in ("interview", "vlog", "tutorial", "presentation"):
            defaults["target_lufs"] = -16.0
            defaults["true_peak"] = -1.5
        elif ct == "screen_recording":
            defaults["target_lufs"] = -18.0
            defaults["true_peak"] = -2.0
        else:
            defaults["target_lufs"] = -16.0
            defaults["true_peak"] = -1.0

        # If current loudness is already close, note it
        if clip_profile.avg_loudness_lufs is not None:
            defaults["current_lufs"] = clip_profile.avg_loudness_lufs
            diff = abs(clip_profile.avg_loudness_lufs - defaults["target_lufs"])
            defaults["adjustment_needed"] = diff > 1.0

    elif operation == "denoise":
        if ct == "screen_recording":
            defaults["strength"] = "light"
            defaults["method"] = "hqdn3d"
        elif ct == "interview":
            defaults["strength"] = "moderate"
            defaults["method"] = "hqdn3d"
        elif ct in ("drone", "sports"):
            defaults["strength"] = "moderate"
            defaults["method"] = "nlmeans"
        else:
            defaults["strength"] = "moderate"
            defaults["method"] = "hqdn3d"

    elif operation == "upscale":
        if clip_profile.resolution < 720:
            defaults["target_resolution"] = "1080p"
            defaults["factor"] = 2
            defaults["method"] = "lanczos"
        elif clip_profile.resolution < 1080:
            defaults["target_resolution"] = "1080p"
            defaults["factor"] = 2
            defaults["method"] = "lanczos"
        elif clip_profile.resolution < 2160:
            defaults["target_resolution"] = "4k"
            defaults["factor"] = 2
            defaults["method"] = "lanczos"
        else:
            defaults["target_resolution"] = "native"
            defaults["factor"] = 1
            defaults["method"] = "none"
            defaults["note"] = "Resolution already 4K or higher"

    elif operation == "export":
        # Codec and quality recommendations
        if ct == "screen_recording":
            defaults["codec"] = "h264"
            defaults["crf"] = 20
            defaults["preset"] = "slow"
            defaults["tune"] = "stillimage"
        elif ct == "music_video":
            defaults["codec"] = "h265"
            defaults["crf"] = 18
            defaults["preset"] = "slow"
            defaults["audio_bitrate"] = "320k"
        elif ct == "interview":
            defaults["codec"] = "h265"
            defaults["crf"] = 22
            defaults["preset"] = "medium"
            defaults["audio_bitrate"] = "192k"
        elif ct == "drone":
            defaults["codec"] = "h265"
            defaults["crf"] = 16
            defaults["preset"] = "slow"
            defaults["note"] = "Lower CRF to preserve aerial detail"
        elif ct == "gaming":
            defaults["codec"] = "h264"
            defaults["crf"] = 18
            defaults["preset"] = "fast"
            defaults["note"] = "H.264 for wider player compatibility at high FPS"
        else:
            defaults["codec"] = "h265"
            defaults["crf"] = 20
            defaults["preset"] = "medium"
            defaults["audio_bitrate"] = "192k"

        # Resolution passthrough
        defaults["width"] = clip_profile.width
        defaults["height"] = clip_profile.height
        defaults["fps"] = clip_profile.fps

    elif operation == "stabilize":
        if ct == "drone":
            defaults["smoothing"] = 30
            defaults["crop"] = "auto"
            defaults["method"] = "vidstab"
        elif ct == "vlog":
            defaults["smoothing"] = 15
            defaults["crop"] = "auto"
            defaults["method"] = "vidstab"
        elif ct in ("interview", "presentation"):
            defaults["smoothing"] = 5
            defaults["crop"] = "minimal"
            defaults["note"] = "Minimal stabilization for static content"
        elif ct == "sports":
            defaults["smoothing"] = 20
            defaults["crop"] = "auto"
            defaults["method"] = "vidstab"
        else:
            defaults["smoothing"] = 15
            defaults["crop"] = "auto"
            defaults["method"] = "vidstab"

    elif operation == "caption":
        if ct == "interview":
            defaults["model"] = "medium"
            defaults["language"] = "auto"
            defaults["style"] = "clean"
            defaults["speaker_labels"] = True
        elif ct == "music_video":
            defaults["model"] = "large-v3"
            defaults["language"] = "auto"
            defaults["style"] = "karaoke"
            defaults["speaker_labels"] = False
        elif ct == "screen_recording":
            defaults["model"] = "base"
            defaults["language"] = "en"
            defaults["style"] = "clean"
            defaults["speaker_labels"] = False
        elif ct == "tutorial":
            defaults["model"] = "medium"
            defaults["language"] = "auto"
            defaults["style"] = "clean"
            defaults["speaker_labels"] = False
        else:
            defaults["model"] = "medium"
            defaults["language"] = "auto"
            defaults["style"] = "default"
            defaults["speaker_labels"] = False

    elif operation == "color_grade":
        if ct == "drone":
            defaults["mood"] = "warm sunset"
            defaults["intensity"] = 0.7
        elif ct == "interview":
            defaults["mood"] = "natural"
            defaults["intensity"] = 0.4
        elif ct == "music_video":
            defaults["mood"] = "cinematic"
            defaults["intensity"] = 0.8
        elif ct == "vlog":
            defaults["mood"] = "warm sunset"
            defaults["intensity"] = 0.5
        else:
            defaults["mood"] = "natural"
            defaults["intensity"] = 0.5

    elif operation == "silence_remove":
        if ct == "interview":
            defaults["threshold_db"] = -35
            defaults["min_silence_ms"] = 800
            defaults["padding_ms"] = 200
        elif ct == "tutorial":
            defaults["threshold_db"] = -40
            defaults["min_silence_ms"] = 1000
            defaults["padding_ms"] = 300
        elif ct == "vlog":
            defaults["threshold_db"] = -35
            defaults["min_silence_ms"] = 600
            defaults["padding_ms"] = 150
        elif ct == "presentation":
            defaults["threshold_db"] = -40
            defaults["min_silence_ms"] = 1500
            defaults["padding_ms"] = 400
        else:
            defaults["threshold_db"] = -35
            defaults["min_silence_ms"] = 800
            defaults["padding_ms"] = 200

    elif operation == "thumbnail":
        if ct == "interview":
            defaults["strategy"] = "face_closeup"
            defaults["text_overlay"] = True
        elif ct == "music_video":
            defaults["strategy"] = "action_frame"
            defaults["text_overlay"] = False
        elif ct == "screen_recording":
            defaults["strategy"] = "key_frame"
            defaults["text_overlay"] = True
        elif ct == "drone":
            defaults["strategy"] = "scenic"
            defaults["text_overlay"] = False
        else:
            defaults["strategy"] = "auto"
            defaults["text_overlay"] = True

    elif operation == "reframe":
        defaults["source_aspect"] = (
            f"{clip_profile.width}:{clip_profile.height}"
            if clip_profile.width and clip_profile.height else "16:9"
        )
        if ct == "interview":
            defaults["target_aspect"] = "9:16"
            defaults["method"] = "face_track"
        elif ct == "screen_recording":
            defaults["target_aspect"] = "9:16"
            defaults["method"] = "cursor_track"
        else:
            defaults["target_aspect"] = "9:16"
            defaults["method"] = "saliency"

    else:
        defaults["note"] = f"No specific defaults for operation: {operation}"

    # Add clip metadata to response for context
    defaults["_clip_info"] = {
        "content_type": clip_profile.detected_content_type,
        "resolution": f"{clip_profile.width}x{clip_profile.height}",
        "fps": clip_profile.fps,
        "duration_s": clip_profile.duration_s,
        "codec": clip_profile.codec,
    }

    if on_progress:
        on_progress(100, f"Defaults ready for {operation}")

    return defaults
