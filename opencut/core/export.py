"""
OpenCut Export & Publish Engine

Platform-optimized video export with presets for YouTube, TikTok, Instagram,
Twitter/X, Podcast, and custom render settings.

Features:
  - Platform presets with automatic resolution, codec, bitrate, audio tuning
  - Custom render (codec, CRF/bitrate, resolution, audio codec)
  - Thumbnail extraction (timestamp, best-frame, or grid contact sheet)
  - Subtitle burn-in (hardcode SRT/VTT/ASS into video via FFmpeg)
  - Watermark overlay (image or text, configurable position/opacity)
  - GIF export (palette-optimized, configurable FPS/width)
  - Audio extraction (MP3, WAV, FLAC, AAC, Opus from video)

All features use FFmpeg only - no additional dependencies.
"""

import json
import logging
import math
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------
@dataclass
class ExportResult:
    """Result from any export operation."""
    output_path: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    file_size_mb: float = 0.0
    operation: str = ""
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------
def _probe_media(filepath: str) -> Dict:
    """Get media info via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        filepath,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return json.loads(r.stdout)
    except Exception as e:
        logger.warning(f"ffprobe failed: {e}")
    return {}


def _get_video_info(filepath: str) -> Dict:
    """Extract key video properties."""
    info = _probe_media(filepath)
    result = {"width": 0, "height": 0, "fps": 0.0, "duration": 0.0,
              "codec": "", "audio_codec": "", "audio_rate": 0}

    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video" and not result["width"]:
            result["width"] = int(stream.get("width", 0))
            result["height"] = int(stream.get("height", 0))
            result["codec"] = stream.get("codec_name", "")
            # Parse fps
            r_fps = stream.get("r_frame_rate", "0/1")
            try:
                num, den = r_fps.split("/")
                result["fps"] = float(num) / float(den) if float(den) else 0
            except Exception:
                result["fps"] = 0
        elif stream.get("codec_type") == "audio" and not result["audio_codec"]:
            result["audio_codec"] = stream.get("codec_name", "")
            result["audio_rate"] = int(stream.get("sample_rate", 0))

    fmt = info.get("format", {})
    result["duration"] = float(fmt.get("duration", 0))
    return result


def _run_ffmpeg(cmd: List[str], timeout: int = 7200) -> str:
    """Run FFmpeg command and return stderr output."""
    logger.debug(f"FFmpeg: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {r.stderr[-500:]}")
    return r.stderr


def _file_size_mb(path: str) -> float:
    """Get file size in MB."""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except Exception:
        return 0.0


def _resolve_output(input_path: str, output_dir: str, suffix: str, ext: str = ".mp4") -> str:
    """Generate output path."""
    if not output_dir:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"{base}_{suffix}{ext}")


# ---------------------------------------------------------------------------
# Platform Presets
# ---------------------------------------------------------------------------
PLATFORM_PRESETS = {
    "youtube_1080": {
        "label": "YouTube 1080p",
        "platform": "YouTube",
        "width": 1920, "height": 1080,
        "fps": None,  # keep original
        "codec": "libx264", "crf": "18", "bitrate": None,
        "audio_codec": "aac", "audio_bitrate": "192k", "audio_rate": 48000,
        "pixel_format": "yuv420p",
        "aspect": "16:9",
        "max_duration": None,
        "extra_flags": ["-movflags", "+faststart"],
    },
    "youtube_4k": {
        "label": "YouTube 4K",
        "platform": "YouTube",
        "width": 3840, "height": 2160,
        "fps": None,
        "codec": "libx264", "crf": "16", "bitrate": None,
        "audio_codec": "aac", "audio_bitrate": "256k", "audio_rate": 48000,
        "pixel_format": "yuv420p",
        "aspect": "16:9",
        "max_duration": None,
        "extra_flags": ["-movflags", "+faststart"],
    },
    "tiktok": {
        "label": "TikTok / Reels (9:16)",
        "platform": "TikTok",
        "width": 1080, "height": 1920,
        "fps": 30,
        "codec": "libx264", "crf": "20", "bitrate": None,
        "audio_codec": "aac", "audio_bitrate": "128k", "audio_rate": 44100,
        "pixel_format": "yuv420p",
        "aspect": "9:16",
        "max_duration": 600,
        "extra_flags": ["-movflags", "+faststart"],
    },
    "instagram_reels": {
        "label": "Instagram Reels (9:16)",
        "platform": "Instagram",
        "width": 1080, "height": 1920,
        "fps": 30,
        "codec": "libx264", "crf": "20", "bitrate": None,
        "audio_codec": "aac", "audio_bitrate": "128k", "audio_rate": 44100,
        "pixel_format": "yuv420p",
        "aspect": "9:16",
        "max_duration": 90,
        "extra_flags": ["-movflags", "+faststart"],
    },
    "instagram_feed": {
        "label": "Instagram Feed (4:5)",
        "platform": "Instagram",
        "width": 1080, "height": 1350,
        "fps": 30,
        "codec": "libx264", "crf": "20", "bitrate": None,
        "audio_codec": "aac", "audio_bitrate": "128k", "audio_rate": 44100,
        "pixel_format": "yuv420p",
        "aspect": "4:5",
        "max_duration": 60,
        "extra_flags": ["-movflags", "+faststart"],
    },
    "instagram_square": {
        "label": "Instagram Square (1:1)",
        "platform": "Instagram",
        "width": 1080, "height": 1080,
        "fps": 30,
        "codec": "libx264", "crf": "20", "bitrate": None,
        "audio_codec": "aac", "audio_bitrate": "128k", "audio_rate": 44100,
        "pixel_format": "yuv420p",
        "aspect": "1:1",
        "max_duration": 60,
        "extra_flags": ["-movflags", "+faststart"],
    },
    "twitter": {
        "label": "Twitter / X",
        "platform": "Twitter",
        "width": 1280, "height": 720,
        "fps": 30,
        "codec": "libx264", "crf": "22", "bitrate": None,
        "audio_codec": "aac", "audio_bitrate": "128k", "audio_rate": 44100,
        "pixel_format": "yuv420p",
        "aspect": "16:9",
        "max_duration": 140,
        "extra_flags": ["-movflags", "+faststart"],
    },
    "linkedin": {
        "label": "LinkedIn",
        "platform": "LinkedIn",
        "width": 1920, "height": 1080,
        "fps": 30,
        "codec": "libx264", "crf": "20", "bitrate": None,
        "audio_codec": "aac", "audio_bitrate": "192k", "audio_rate": 48000,
        "pixel_format": "yuv420p",
        "aspect": "16:9",
        "max_duration": 600,
        "extra_flags": ["-movflags", "+faststart"],
    },
    "podcast_video": {
        "label": "Podcast (Video)",
        "platform": "Podcast",
        "width": 1920, "height": 1080,
        "fps": 30,
        "codec": "libx264", "crf": "23", "bitrate": None,
        "audio_codec": "aac", "audio_bitrate": "192k", "audio_rate": 48000,
        "pixel_format": "yuv420p",
        "aspect": "16:9",
        "max_duration": None,
        "extra_flags": ["-movflags", "+faststart"],
    },
    "podcast_audio": {
        "label": "Podcast (Audio Only)",
        "platform": "Podcast",
        "width": 0, "height": 0,
        "fps": None,
        "codec": None, "crf": None, "bitrate": None,
        "audio_codec": "libmp3lame", "audio_bitrate": "192k", "audio_rate": 44100,
        "pixel_format": None,
        "aspect": None,
        "max_duration": None,
        "extra_flags": [],
        "audio_only": True,
    },
    "shorts": {
        "label": "YouTube Shorts (9:16)",
        "platform": "YouTube",
        "width": 1080, "height": 1920,
        "fps": 30,
        "codec": "libx264", "crf": "20", "bitrate": None,
        "audio_codec": "aac", "audio_bitrate": "128k", "audio_rate": 44100,
        "pixel_format": "yuv420p",
        "aspect": "9:16",
        "max_duration": 60,
        "extra_flags": ["-movflags", "+faststart"],
    },
}

# Available codecs for custom render
AVAILABLE_CODECS = {
    "video": {
        "libx264":  {"label": "H.264 (MP4)", "ext": ".mp4"},
        "libx265":  {"label": "H.265/HEVC (MP4)", "ext": ".mp4"},
        "libvpx-vp9": {"label": "VP9 (WebM)", "ext": ".webm"},
        "prores_ks": {"label": "ProRes (MOV)", "ext": ".mov"},
        "copy":     {"label": "Copy (no re-encode)", "ext": None},
    },
    "audio": {
        "aac":         {"label": "AAC", "ext": ".m4a"},
        "libmp3lame":  {"label": "MP3", "ext": ".mp3"},
        "flac":        {"label": "FLAC", "ext": ".flac"},
        "pcm_s16le":   {"label": "WAV (16-bit)", "ext": ".wav"},
        "libopus":     {"label": "Opus (WebM)", "ext": ".webm"},
        "copy":        {"label": "Copy (no re-encode)", "ext": None},
    },
}

RESOLUTION_PRESETS = {
    "720p":  {"width": 1280, "height": 720},
    "1080p": {"width": 1920, "height": 1080},
    "1440p": {"width": 2560, "height": 1440},
    "4k":    {"width": 3840, "height": 2160},
}


# ---------------------------------------------------------------------------
# Platform Render
# ---------------------------------------------------------------------------
def render_platform(
    input_path: str,
    preset: str,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ExportResult:
    """
    Render video using a platform-specific preset.

    Args:
        input_path: Source video file.
        preset:     Key from PLATFORM_PRESETS.
        output_dir: Output directory.
        on_progress: Callback(pct, msg).
    """
    if preset not in PLATFORM_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PLATFORM_PRESETS.keys())}")

    p = PLATFORM_PRESETS[preset]
    if on_progress:
        on_progress(5, f"Preparing {p['label']} export...")

    info = _get_video_info(input_path)
    is_audio_only = p.get("audio_only", False)

    # Determine output extension
    if is_audio_only:
        ext = AVAILABLE_CODECS["audio"].get(p["audio_codec"], {}).get("ext", ".mp3")
    else:
        ext = ".mp4"
    out_path = _resolve_output(input_path, output_dir, f"export_{preset}", ext)

    # Build FFmpeg command
    cmd = ["ffmpeg", "-y", "-i", input_path]

    if is_audio_only:
        cmd += ["-vn"]
    else:
        # Video filters: scale + pad for target aspect
        tw, th = p["width"], p["height"]
        vf_parts = []

        # Scale to fit within target, maintaining aspect ratio
        vf_parts.append(f"scale={tw}:{th}:force_original_aspect_ratio=decrease")
        # Pad to exact target dimensions (letterbox/pillarbox)
        vf_parts.append(f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black")

        if p.get("fps"):
            vf_parts.append(f"fps={p['fps']}")

        cmd += ["-vf", ",".join(vf_parts)]
        cmd += ["-c:v", p["codec"]]

        if p.get("crf"):
            cmd += ["-crf", p["crf"]]
        if p.get("bitrate"):
            cmd += ["-b:v", p["bitrate"]]
        if p.get("pixel_format"):
            cmd += ["-pix_fmt", p["pixel_format"]]

    # Audio settings
    cmd += ["-c:a", p["audio_codec"]]
    if p.get("audio_bitrate"):
        cmd += ["-b:a", p["audio_bitrate"]]
    if p.get("audio_rate"):
        cmd += ["-ar", str(p["audio_rate"])]

    # Max duration
    if p.get("max_duration") and info["duration"] > p["max_duration"]:
        cmd += ["-t", str(p["max_duration"])]
        if on_progress:
            on_progress(15, f"Trimming to {p['max_duration']}s max ({p['platform']} limit)")

    # Extra flags
    cmd += p.get("extra_flags", [])
    cmd.append(out_path)

    if on_progress:
        on_progress(20, f"Encoding for {p['platform']}...")

    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Export complete")

    out_info = _get_video_info(out_path) if not is_audio_only else {"duration": info["duration"]}
    return ExportResult(
        output_path=out_path,
        duration=out_info.get("duration", 0),
        width=p.get("width", 0),
        height=p.get("height", 0),
        file_size_mb=_file_size_mb(out_path),
        operation="platform_render",
        details={"preset": preset, "platform": p["platform"], "label": p["label"]},
    )


# ---------------------------------------------------------------------------
# Custom Render
# ---------------------------------------------------------------------------
def render_custom(
    input_path: str,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    resolution: str = "",
    width: int = 0,
    height: int = 0,
    crf: int = 23,
    video_bitrate: str = "",
    audio_bitrate: str = "192k",
    fps: float = 0,
    pixel_format: str = "yuv420p",
    container: str = "mp4",
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ExportResult:
    """
    Render with fully custom settings.
    """
    if on_progress:
        on_progress(5, "Preparing custom render...")

    info = _get_video_info(input_path)

    # Resolve resolution
    if resolution and resolution in RESOLUTION_PRESETS:
        rp = RESOLUTION_PRESETS[resolution]
        width, height = rp["width"], rp["height"]

    # Determine extension
    ext_map = {
        "mp4": ".mp4", "mov": ".mov", "webm": ".webm",
        "mkv": ".mkv", "avi": ".avi",
    }
    ext = ext_map.get(container, ".mp4")

    # Override ext based on video codec if needed
    if video_codec == "libvpx-vp9":
        ext = ".webm"
    elif video_codec == "prores_ks":
        ext = ".mov"

    out_path = _resolve_output(input_path, output_dir, "custom", ext)

    cmd = ["ffmpeg", "-y", "-i", input_path]

    # Video
    if video_codec == "copy":
        cmd += ["-c:v", "copy"]
    else:
        vf_parts = []
        if width and height:
            vf_parts.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease")
            # Ensure even dimensions
            vf_parts.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
        elif width:
            vf_parts.append(f"scale={width}:-2")
        elif height:
            vf_parts.append(f"scale=-2:{height}")

        if fps > 0:
            vf_parts.append(f"fps={fps}")

        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]

        cmd += ["-c:v", video_codec]

        if video_codec in ("libx264", "libx265"):
            if video_bitrate:
                cmd += ["-b:v", video_bitrate]
            else:
                cmd += ["-crf", str(crf)]
            if pixel_format:
                cmd += ["-pix_fmt", pixel_format]
        elif video_codec == "prores_ks":
            cmd += ["-profile:v", "3"]  # ProRes HQ

    # Audio
    if audio_codec == "copy":
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", audio_codec]
        if audio_bitrate:
            cmd += ["-b:a", audio_bitrate]

    cmd += ["-movflags", "+faststart"]
    cmd.append(out_path)

    if on_progress:
        on_progress(20, f"Rendering with {video_codec}...")

    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Render complete")

    out_info = _get_video_info(out_path)
    return ExportResult(
        output_path=out_path,
        duration=out_info.get("duration", 0),
        width=out_info.get("width", 0),
        height=out_info.get("height", 0),
        file_size_mb=_file_size_mb(out_path),
        operation="custom_render",
        details={"video_codec": video_codec, "audio_codec": audio_codec},
    )


# ---------------------------------------------------------------------------
# Thumbnail Extraction
# ---------------------------------------------------------------------------
def extract_thumbnail(
    input_path: str,
    timestamp: float = -1,
    mode: str = "single",
    count: int = 1,
    width: int = 1920,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ExportResult:
    """
    Extract thumbnail(s) from video.

    Args:
        input_path: Source video.
        timestamp:  Specific timestamp in seconds (-1 = auto-detect best frame).
        mode:       'single' (one frame), 'grid' (contact sheet), 'multi' (several frames).
        count:      Number of frames for 'multi' mode.
        width:      Output image width.
        output_dir: Output directory.
        on_progress: Callback(pct, msg).
    """
    if on_progress:
        on_progress(5, "Analyzing video...")

    info = _get_video_info(input_path)
    duration = info["duration"]
    if duration <= 0:
        raise ValueError("Cannot extract thumbnail: video has no duration")

    if not output_dir:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    if mode == "grid":
        # Contact sheet: 3x3 grid of frames
        out_path = os.path.join(output_dir, f"{base}_contact_sheet.jpg")
        grid_count = 9
        interval = duration / (grid_count + 1)

        # Extract individual frames to temp dir
        tmp_dir = tempfile.mkdtemp(prefix="opencut_thumb_")
        try:
            for i in range(grid_count):
                ts = interval * (i + 1)
                frame_path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
                cmd = [
                    "ffmpeg", "-y", "-ss", str(ts), "-i", input_path,
                    "-vframes", "1", "-vf", f"scale={width // 3}:-1",
                    "-q:v", "2", frame_path,
                ]
                subprocess.run(cmd, capture_output=True, timeout=30)
                if on_progress:
                    on_progress(10 + int(60 * (i + 1) / grid_count), f"Frame {i + 1}/{grid_count}")

            # Tile into grid using FFmpeg
            inputs = []
            filter_parts = []
            for i in range(grid_count):
                fp = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
                if os.path.isfile(fp):
                    inputs += ["-i", fp]

            # Build xstack filter for 3x3
            actual_count = min(grid_count, len([1 for i in range(grid_count)
                                                if os.path.isfile(os.path.join(tmp_dir, f"frame_{i:03d}.jpg"))]))
            if actual_count >= 9:
                layout = "|".join([
                    "0_0", "w0_0", "w0+w1_0",
                    "0_h0", "w0_h0", "w0+w1_h0",
                    "0_h0+h3", "w0_h0+h3", "w0+w1_h0+h3",
                ])
                cmd = ["ffmpeg", "-y"] + inputs + [
                    "-filter_complex", f"xstack=inputs=9:layout={layout}",
                    "-q:v", "2", out_path,
                ]
                _run_ffmpeg(cmd)
            else:
                # Fallback: just use first frame
                import shutil
                first = os.path.join(tmp_dir, "frame_000.jpg")
                if os.path.isfile(first):
                    shutil.copy2(first, out_path)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if on_progress:
            on_progress(100, "Contact sheet created")

        return ExportResult(
            output_path=out_path,
            width=width, height=0,
            file_size_mb=_file_size_mb(out_path),
            operation="thumbnail_grid",
            details={"frames": grid_count, "mode": "grid"},
        )

    elif mode == "multi":
        # Multiple evenly-spaced frames
        out_paths = []
        interval = duration / (count + 1)
        for i in range(count):
            ts = interval * (i + 1)
            frame_out = os.path.join(output_dir, f"{base}_thumb_{i + 1:03d}.jpg")
            cmd = [
                "ffmpeg", "-y", "-ss", str(ts), "-i", input_path,
                "-vframes", "1", "-vf", f"scale={width}:-1",
                "-q:v", "2", frame_out,
            ]
            _run_ffmpeg(cmd)
            out_paths.append(frame_out)
            if on_progress:
                on_progress(10 + int(80 * (i + 1) / count), f"Frame {i + 1}/{count}")

        if on_progress:
            on_progress(100, f"Extracted {count} thumbnails")

        return ExportResult(
            output_path=out_paths[0] if out_paths else "",
            width=width,
            file_size_mb=sum(_file_size_mb(p) for p in out_paths),
            operation="thumbnail_multi",
            details={"frames": count, "paths": out_paths, "mode": "multi"},
        )

    else:
        # Single frame
        out_path = os.path.join(output_dir, f"{base}_thumbnail.jpg")

        if timestamp < 0:
            # Auto-detect: use thumbnail filter to find "interesting" frame
            # Strategy: grab frame at ~10% of duration (skip intros)
            # Then try scene-change detection for a visually interesting frame
            try:
                # Use thumbnail filter to pick the most representative frame
                # from a sample of frames around the 20-40% mark
                sample_start = duration * 0.15
                sample_len = min(duration * 0.3, 30)
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(sample_start),
                    "-t", str(sample_len),
                    "-i", input_path,
                    "-vf", f"thumbnail=300,scale={width}:-1",
                    "-vframes", "1",
                    "-q:v", "2", out_path,
                ]
                _run_ffmpeg(cmd)
            except Exception:
                # Fallback: frame at 25%
                timestamp = duration * 0.25
                cmd = [
                    "ffmpeg", "-y", "-ss", str(timestamp), "-i", input_path,
                    "-vframes", "1", "-vf", f"scale={width}:-1",
                    "-q:v", "2", out_path,
                ]
                _run_ffmpeg(cmd)
        else:
            timestamp = min(timestamp, duration - 0.1)
            cmd = [
                "ffmpeg", "-y", "-ss", str(timestamp), "-i", input_path,
                "-vframes", "1", "-vf", f"scale={width}:-1",
                "-q:v", "2", out_path,
            ]
            _run_ffmpeg(cmd)

        if on_progress:
            on_progress(100, "Thumbnail extracted")

        return ExportResult(
            output_path=out_path,
            width=width,
            file_size_mb=_file_size_mb(out_path),
            operation="thumbnail_single",
            details={"timestamp": timestamp, "mode": "single"},
        )


# ---------------------------------------------------------------------------
# Subtitle Burn-In
# ---------------------------------------------------------------------------
def burn_subtitles(
    input_path: str,
    subtitle_path: str,
    font_size: int = 24,
    font_name: str = "Arial",
    font_color: str = "white",
    outline_color: str = "black",
    outline_width: int = 2,
    position: str = "bottom",
    margin_v: int = 30,
    quality: str = "medium",
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ExportResult:
    """
    Burn (hardcode) subtitles into video.

    Args:
        input_path:     Source video.
        subtitle_path:  Path to .srt, .vtt, or .ass file.
        font_size:      Font size in pixels.
        font_name:      Font family name.
        font_color:     Font color (FFmpeg color name or hex).
        outline_color:  Outline/border color.
        outline_width:  Outline thickness.
        position:       'bottom', 'top', 'center'.
        margin_v:       Vertical margin in pixels.
        quality:        'low', 'medium', 'high'.
        output_dir:     Output directory.
        on_progress:    Callback(pct, msg).
    """
    if not os.path.isfile(subtitle_path):
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")

    if on_progress:
        on_progress(5, "Preparing subtitle burn-in...")

    ext = os.path.splitext(subtitle_path)[1].lower()
    crf = {"low": "28", "medium": "23", "high": "18"}.get(quality, "23")

    out_path = _resolve_output(input_path, output_dir, "hardcoded")

    # Escape path for FFmpeg filter (Windows paths need special handling)
    sub_escaped = subtitle_path.replace("\\", "/").replace(":", "\\:")

    if ext == ".ass":
        # ASS files have their own styling
        vf = f"ass='{sub_escaped}'"
    else:
        # SRT/VTT use subtitles filter with styling
        # Map position to alignment (SSA-style)
        alignment_map = {"bottom": 2, "top": 6, "center": 10}
        alignment = alignment_map.get(position, 2)

        # Build force_style string
        style_parts = [
            f"FontName={font_name}",
            f"FontSize={font_size}",
            f"PrimaryColour=&H00FFFFFF",  # White
            f"OutlineColour=&H00000000",  # Black outline
            f"BorderStyle=1",
            f"Outline={outline_width}",
            f"Shadow=1",
            f"Alignment={alignment}",
            f"MarginV={margin_v}",
        ]

        # Color mapping
        color_map = {
            "white": "&H00FFFFFF", "yellow": "&H0000FFFF",
            "green": "&H0000FF00", "cyan": "&H00FFFF00",
            "red": "&H000000FF",
        }
        if font_color in color_map:
            style_parts[2] = f"PrimaryColour={color_map[font_color]}"
        if outline_color in color_map:
            style_parts[3] = f"OutlineColour={color_map[outline_color]}"

        force_style = ",".join(style_parts)
        vf = f"subtitles='{sub_escaped}':force_style='{force_style}'"

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", crf,
        "-c:a", "copy",
        "-movflags", "+faststart",
        out_path,
    ]

    if on_progress:
        on_progress(20, "Burning subtitles...")

    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Subtitles burned in")

    out_info = _get_video_info(out_path)
    return ExportResult(
        output_path=out_path,
        duration=out_info.get("duration", 0),
        width=out_info.get("width", 0),
        height=out_info.get("height", 0),
        file_size_mb=_file_size_mb(out_path),
        operation="burn_subtitles",
        details={
            "subtitle_file": subtitle_path,
            "font_size": font_size,
            "position": position,
        },
    )


# ---------------------------------------------------------------------------
# Watermark
# ---------------------------------------------------------------------------
def add_watermark(
    input_path: str,
    watermark_type: str = "text",
    text: str = "",
    image_path: str = "",
    position: str = "bottom-right",
    opacity: float = 0.5,
    font_size: int = 24,
    font_color: str = "white",
    margin: int = 20,
    scale: float = 0.15,
    quality: str = "medium",
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ExportResult:
    """
    Add watermark (text or image) to video.

    Args:
        input_path:     Source video.
        watermark_type: 'text' or 'image'.
        text:           Watermark text (if type='text').
        image_path:     Path to watermark image (if type='image').
        position:       'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'.
        opacity:        Watermark opacity 0.0-1.0.
        font_size:      Font size for text watermarks.
        font_color:     Font color for text watermarks.
        margin:         Margin from edges in pixels.
        scale:          Scale factor for image watermarks (relative to video width).
        quality:        Encoding quality.
        output_dir:     Output directory.
        on_progress:    Callback(pct, msg).
    """
    if on_progress:
        on_progress(5, "Adding watermark...")

    crf = {"low": "28", "medium": "23", "high": "18"}.get(quality, "23")
    out_path = _resolve_output(input_path, output_dir, "watermarked")

    # Position coordinates
    pos_map = {
        "top-left":     (str(margin), str(margin)),
        "top-right":    (f"W-w-{margin}", str(margin)),
        "bottom-left":  (str(margin), f"H-h-{margin}"),
        "bottom-right": (f"W-w-{margin}", f"H-h-{margin}"),
        "center":       ("(W-w)/2", "(H-h)/2"),
    }
    px, py = pos_map.get(position, pos_map["bottom-right"])

    if watermark_type == "image" and image_path and os.path.isfile(image_path):
        info = _get_video_info(input_path)
        wm_width = int(info["width"] * scale) if info["width"] else 200

        vf = (
            f"[1:v]scale={wm_width}:-1,format=rgba,"
            f"colorchannelmixer=aa={opacity}[wm];"
            f"[0:v][wm]overlay={px}:{py}"
        )
        cmd = [
            "ffmpeg", "-y", "-i", input_path, "-i", image_path,
            "-filter_complex", vf,
            "-c:v", "libx264", "-crf", crf,
            "-c:a", "copy",
            "-movflags", "+faststart",
            out_path,
        ]
    else:
        # Text watermark
        if not text:
            text = "WATERMARK"
        text_escaped = text.replace("'", "\\'").replace(":", "\\:")

        color_hex = font_color
        if font_color == "white":
            color_hex = "ffffff"
        elif font_color == "black":
            color_hex = "000000"

        alpha_hex = f"{int(opacity * 255):02x}"

        vf = (
            f"drawtext=text='{text_escaped}':"
            f"fontsize={font_size}:fontcolor={color_hex}@{opacity}:"
            f"x={px}:y={py}:"
            f"shadowcolor=black@0.3:shadowx=2:shadowy=2"
        )
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", crf,
            "-c:a", "copy",
            "-movflags", "+faststart",
            out_path,
        ]

    if on_progress:
        on_progress(20, "Encoding with watermark...")

    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Watermark applied")

    out_info = _get_video_info(out_path)
    return ExportResult(
        output_path=out_path,
        duration=out_info.get("duration", 0),
        width=out_info.get("width", 0),
        height=out_info.get("height", 0),
        file_size_mb=_file_size_mb(out_path),
        operation="watermark",
        details={"type": watermark_type, "position": position, "opacity": opacity},
    )


# ---------------------------------------------------------------------------
# GIF Export
# ---------------------------------------------------------------------------
def export_gif(
    input_path: str,
    start: float = 0,
    duration: float = 5,
    width: int = 480,
    fps: int = 15,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ExportResult:
    """
    Export video segment as optimized GIF.

    Uses FFmpeg two-pass palette generation for high-quality GIFs.
    """
    if on_progress:
        on_progress(5, "Generating GIF palette...")

    out_path = _resolve_output(input_path, output_dir, "animated", ".gif")

    # Two-pass GIF: first generate palette, then use it
    palette_path = out_path + ".palette.png"

    try:
        # Pass 1: Generate palette
        filters = f"fps={fps},scale={width}:-1:flags=lanczos"
        cmd1 = [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(duration),
            "-i", input_path,
            "-vf", f"{filters},palettegen=stats_mode=diff",
            palette_path,
        ]
        _run_ffmpeg(cmd1)

        if on_progress:
            on_progress(40, "Encoding GIF...")

        # Pass 2: Generate GIF using palette
        cmd2 = [
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(duration),
            "-i", input_path, "-i", palette_path,
            "-filter_complex", f"{filters}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
            out_path,
        ]
        _run_ffmpeg(cmd2)

    finally:
        if os.path.isfile(palette_path):
            os.remove(palette_path)

    if on_progress:
        on_progress(100, "GIF exported")

    return ExportResult(
        output_path=out_path,
        duration=duration,
        width=width,
        file_size_mb=_file_size_mb(out_path),
        operation="gif_export",
        details={"start": start, "duration": duration, "fps": fps, "width": width},
    )


# ---------------------------------------------------------------------------
# Audio Extraction
# ---------------------------------------------------------------------------
def extract_audio(
    input_path: str,
    codec: str = "libmp3lame",
    bitrate: str = "192k",
    sample_rate: int = 44100,
    normalize: bool = False,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ExportResult:
    """
    Extract audio track from video.

    Args:
        input_path:  Source video.
        codec:       Audio codec (libmp3lame, aac, flac, pcm_s16le, libopus).
        bitrate:     Audio bitrate.
        sample_rate: Sample rate in Hz.
        normalize:   Apply loudness normalization (-16 LUFS).
        output_dir:  Output directory.
        on_progress: Callback(pct, msg).
    """
    if on_progress:
        on_progress(5, "Extracting audio...")

    ext_map = {
        "libmp3lame": ".mp3", "aac": ".m4a", "flac": ".flac",
        "pcm_s16le": ".wav", "libopus": ".opus",
    }
    ext = ext_map.get(codec, ".mp3")
    out_path = _resolve_output(input_path, output_dir, "audio", ext)

    cmd = ["ffmpeg", "-y", "-i", input_path, "-vn"]

    af_parts = []
    if normalize:
        af_parts.append("loudnorm=I=-16:TP=-1:LRA=11")

    if af_parts:
        cmd += ["-af", ",".join(af_parts)]

    cmd += ["-c:a", codec]

    if codec not in ("pcm_s16le", "flac"):
        cmd += ["-b:a", bitrate]

    cmd += ["-ar", str(sample_rate)]
    cmd.append(out_path)

    if on_progress:
        on_progress(20, f"Encoding {ext.upper().strip('.')}...")

    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Audio extracted")

    info = _probe_media(out_path)
    duration = float(info.get("format", {}).get("duration", 0))
    return ExportResult(
        output_path=out_path,
        duration=duration,
        file_size_mb=_file_size_mb(out_path),
        operation="audio_extract",
        details={"codec": codec, "bitrate": bitrate, "normalize": normalize},
    )


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------
def get_export_capabilities() -> Dict:
    """Return all export features and configuration options."""
    return {
        "platform_presets": {
            k: {"label": v["label"], "platform": v["platform"],
                "resolution": f"{v['width']}x{v['height']}" if v.get("width") else "audio",
                "aspect": v.get("aspect", ""),
                "max_duration": v.get("max_duration")}
            for k, v in PLATFORM_PRESETS.items()
        },
        "video_codecs": {k: v["label"] for k, v in AVAILABLE_CODECS["video"].items()},
        "audio_codecs": {k: v["label"] for k, v in AVAILABLE_CODECS["audio"].items()},
        "resolutions": {k: f"{v['width']}x{v['height']}" for k, v in RESOLUTION_PRESETS.items()},
        "thumbnail_modes": ["single", "multi", "grid"],
        "watermark_positions": ["top-left", "top-right", "bottom-left", "bottom-right", "center"],
    }
