"""
OpenCut Export Presets Module v0.7.2

Pre-configured export profiles for common platforms:
- YouTube (1080p, 4K, Shorts)
- TikTok / Instagram Reels (9:16 vertical)
- Instagram Feed / Story
- Twitter/X
- LinkedIn
- Podcast (audio-only)
- Archive (ProRes, DNxHR)
- Custom resolution/codec

Each preset defines resolution, codec, bitrate, aspect ratio, and
platform-specific constraints (max duration, file size, etc.).
"""

import logging
import os
import subprocess
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


EXPORT_PRESETS = {
    # === YouTube ===
    "youtube_1080p": {
        "label": "YouTube 1080p",
        "description": "Standard HD upload for YouTube",
        "category": "youtube",
        "width": 1920, "height": 1080,
        "codec": "libx264", "crf": 18, "preset": "medium",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p", "maxrate": "12M", "bufsize": "24M",
        "ext": ".mp4",
    },
    "youtube_4k": {
        "label": "YouTube 4K",
        "description": "Ultra HD upload for YouTube",
        "category": "youtube",
        "width": 3840, "height": 2160,
        "codec": "libx264", "crf": 18, "preset": "slow",
        "audio_codec": "aac", "audio_bitrate": "320k",
        "pix_fmt": "yuv420p", "maxrate": "40M", "bufsize": "80M",
        "ext": ".mp4",
    },
    "youtube_shorts": {
        "label": "YouTube Shorts",
        "description": "9:16 vertical for YouTube Shorts (max 60s)",
        "category": "youtube",
        "width": 1080, "height": 1920,
        "codec": "libx264", "crf": 20, "preset": "medium",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p", "maxrate": "8M", "bufsize": "16M",
        "ext": ".mp4", "max_duration": 60,
    },
    # === TikTok ===
    "tiktok": {
        "label": "TikTok",
        "description": "Vertical 9:16 for TikTok (max 10min)",
        "category": "social",
        "width": 1080, "height": 1920,
        "codec": "libx264", "crf": 20, "preset": "medium",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p", "maxrate": "6M", "bufsize": "12M",
        "ext": ".mp4", "max_duration": 600,
    },
    # === Instagram ===
    "instagram_feed": {
        "label": "Instagram Feed",
        "description": "Square 1:1 for Instagram feed (max 60s)",
        "category": "social",
        "width": 1080, "height": 1080,
        "codec": "libx264", "crf": 20, "preset": "medium",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p", "maxrate": "5M", "bufsize": "10M",
        "ext": ".mp4", "max_duration": 60,
    },
    "instagram_reels": {
        "label": "Instagram Reels",
        "description": "Vertical 9:16 for Instagram Reels (max 90s)",
        "category": "social",
        "width": 1080, "height": 1920,
        "codec": "libx264", "crf": 20, "preset": "medium",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p", "maxrate": "6M", "bufsize": "12M",
        "ext": ".mp4", "max_duration": 90,
    },
    "instagram_story": {
        "label": "Instagram Story",
        "description": "Vertical 9:16 for Stories (max 15s per clip)",
        "category": "social",
        "width": 1080, "height": 1920,
        "codec": "libx264", "crf": 20, "preset": "fast",
        "audio_codec": "aac", "audio_bitrate": "128k",
        "pix_fmt": "yuv420p", "maxrate": "5M", "bufsize": "10M",
        "ext": ".mp4", "max_duration": 15,
    },
    # === Twitter/X ===
    "twitter": {
        "label": "Twitter / X",
        "description": "16:9 for Twitter/X (max 2min 20s, 512MB)",
        "category": "social",
        "width": 1280, "height": 720,
        "codec": "libx264", "crf": 22, "preset": "medium",
        "audio_codec": "aac", "audio_bitrate": "128k",
        "pix_fmt": "yuv420p", "maxrate": "5M", "bufsize": "10M",
        "ext": ".mp4", "max_duration": 140, "max_size_mb": 512,
    },
    # === LinkedIn ===
    "linkedin": {
        "label": "LinkedIn",
        "description": "Professional 16:9 for LinkedIn (max 10min)",
        "category": "social",
        "width": 1920, "height": 1080,
        "codec": "libx264", "crf": 20, "preset": "medium",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p", "maxrate": "10M", "bufsize": "20M",
        "ext": ".mp4", "max_duration": 600, "max_size_mb": 5120,
    },
    # === Podcast / Audio ===
    "podcast_mp3": {
        "label": "Podcast (MP3)",
        "description": "High-quality MP3 for podcast distribution",
        "category": "audio",
        "audio_codec": "libmp3lame", "audio_bitrate": "192k",
        "audio_only": True, "ext": ".mp3",
    },
    "podcast_aac": {
        "label": "Podcast (AAC)",
        "description": "AAC for Apple Podcasts / Spotify",
        "category": "audio",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "audio_only": True, "ext": ".m4a",
    },
    "podcast_wav": {
        "label": "Podcast Master (WAV)",
        "description": "Uncompressed WAV master file",
        "category": "audio",
        "audio_codec": "pcm_s24le", "audio_bitrate": "",
        "audio_only": True, "ext": ".wav",
    },
    # === Archive / Professional ===
    "prores_422": {
        "label": "ProRes 422",
        "description": "Apple ProRes 422 for post-production",
        "category": "archive",
        "width": 0, "height": 0,  # Keep original
        "codec": "prores_ks", "profile": "2",
        "audio_codec": "pcm_s16le",
        "pix_fmt": "yuv422p10le",
        "ext": ".mov",
    },
    "prores_4444": {
        "label": "ProRes 4444",
        "description": "Apple ProRes 4444 with alpha channel",
        "category": "archive",
        "width": 0, "height": 0,
        "codec": "prores_ks", "profile": "4",
        "audio_codec": "pcm_s16le",
        "pix_fmt": "yuva444p10le",
        "ext": ".mov",
    },
    "dnxhr_hq": {
        "label": "DNxHR HQ",
        "description": "Avid DNxHR HQ for professional editing",
        "category": "archive",
        "width": 0, "height": 0,
        "codec": "dnxhd", "profile": "dnxhr_hq",
        "audio_codec": "pcm_s16le",
        "pix_fmt": "yuv422p",
        "ext": ".mxf",
    },
    # === Web ===
    "web_h264": {
        "label": "Web (H.264)",
        "description": "Optimized H.264 for web embedding",
        "category": "web",
        "width": 1920, "height": 1080,
        "codec": "libx264", "crf": 23, "preset": "medium",
        "audio_codec": "aac", "audio_bitrate": "128k",
        "pix_fmt": "yuv420p",
        "movflags": "+faststart",
        "ext": ".mp4",
    },
    "web_vp9": {
        "label": "Web (VP9/WebM)",
        "description": "VP9 for modern browsers, smaller files",
        "category": "web",
        "width": 1920, "height": 1080,
        "codec": "libvpx-vp9", "crf": 30, "b_v": "0",
        "audio_codec": "libopus", "audio_bitrate": "128k",
        "pix_fmt": "yuv420p",
        "ext": ".webm",
    },
    # === GIF ===
    "gif_high": {
        "label": "GIF (High Quality)",
        "description": "Animated GIF with palette optimization",
        "category": "web",
        "width": 640, "height": 0,  # Scale proportionally
        "fps": 15,
        "ext": ".gif",
    },
}


def get_export_presets() -> List[Dict]:
    """Return all export presets for UI display."""
    return [
        {
            "name": k,
            "label": v["label"],
            "description": v["description"],
            "category": v["category"],
        }
        for k, v in EXPORT_PRESETS.items()
    ]


def get_preset_categories() -> List[Dict]:
    """Return preset categories."""
    cats = {}
    for v in EXPORT_PRESETS.values():
        c = v["category"]
        if c not in cats:
            cats[c] = {"name": c, "label": c.title(), "count": 0}
        cats[c]["count"] += 1
    return list(cats.values())


def _run_ffmpeg(cmd: List[str], timeout: int = 7200) -> str:
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode(errors='replace')[-500:]}")
    return result.stderr.decode(errors="replace")


def _get_video_info(filepath: str) -> Dict:
    import json
    cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "json", filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    try:
        data = json.loads(result.stdout.decode())
        stream = data["streams"][0]
        fps_parts = stream.get("r_frame_rate", "30/1").split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
        return {
            "width": int(stream.get("width", 1920)),
            "height": int(stream.get("height", 1080)),
            "fps": fps,
            "duration": float(stream.get("duration", 0)),
        }
    except Exception:
        return {"width": 1920, "height": 1080, "fps": 30.0, "duration": 0}


def export_with_preset(
    input_path: str,
    preset_name: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Export video using a preset profile.
    """
    if preset_name not in EXPORT_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")

    preset = EXPORT_PRESETS[preset_name]

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        ext = preset.get("ext", ".mp4")
        directory = output_dir or os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_{preset_name}{ext}")

    if on_progress:
        on_progress(5, f"Exporting with {preset['label']}...")

    info = _get_video_info(input_path)

    # Special handling for GIF
    if preset.get("ext") == ".gif":
        return _export_gif(input_path, output_path, preset, info, on_progress)

    # Build FFmpeg command
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", input_path]

    # Audio-only export
    if preset.get("audio_only"):
        cmd += ["-vn"]
        if preset.get("audio_codec"):
            cmd += ["-acodec", preset["audio_codec"]]
        if preset.get("audio_bitrate"):
            cmd += ["-b:a", preset["audio_bitrate"]]
        cmd.append(output_path)
        _run_ffmpeg(cmd)
        if on_progress:
            on_progress(100, f"Audio exported ({preset['label']})")
        return output_path

    # Video filters for scaling
    vf_parts = []
    target_w = preset.get("width", 0)
    target_h = preset.get("height", 0)

    if target_w > 0 and target_h > 0:
        # Scale and pad to exact dimensions
        vf_parts.append(
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
        )
    elif target_w > 0:
        vf_parts.append(f"scale={target_w}:-2")
    elif target_h > 0:
        vf_parts.append(f"scale=-2:{target_h}")

    if preset.get("fps"):
        vf_parts.append(f"fps={preset['fps']}")

    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    # Video codec
    codec = preset.get("codec", "libx264")
    cmd += ["-c:v", codec]

    if preset.get("crf") is not None:
        cmd += ["-crf", str(preset["crf"])]
    if preset.get("preset"):
        cmd += ["-preset", preset["preset"]]
    if preset.get("profile"):
        if codec == "prores_ks":
            cmd += ["-profile:v", preset["profile"]]
        elif codec == "dnxhd":
            cmd += ["-profile:v", preset["profile"]]
    if preset.get("pix_fmt"):
        cmd += ["-pix_fmt", preset["pix_fmt"]]
    if preset.get("maxrate"):
        cmd += ["-maxrate", preset["maxrate"]]
    if preset.get("bufsize"):
        cmd += ["-bufsize", preset["bufsize"]]
    if preset.get("b_v"):
        cmd += ["-b:v", preset["b_v"]]
    if preset.get("movflags"):
        cmd += ["-movflags", preset["movflags"]]

    # Audio codec
    if preset.get("audio_codec"):
        cmd += ["-c:a", preset["audio_codec"]]
    if preset.get("audio_bitrate"):
        cmd += ["-b:a", preset["audio_bitrate"]]

    # Duration limit
    if preset.get("max_duration"):
        cmd += ["-t", str(preset["max_duration"])]

    cmd.append(output_path)

    if on_progress:
        on_progress(10, "Encoding...")

    _run_ffmpeg(cmd)

    # Check file size limit
    if preset.get("max_size_mb"):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        if size_mb > preset["max_size_mb"]:
            logger.warning(
                f"Export exceeds {preset['max_size_mb']}MB limit: {size_mb:.0f}MB"
            )

    if on_progress:
        on_progress(100, f"Exported ({preset['label']})")
    return output_path


def _export_gif(
    input_path: str, output_path: str,
    preset: Dict, info: Dict,
    on_progress: Optional[Callable] = None,
) -> str:
    """Export as optimized GIF using two-pass palette generation."""
    import tempfile

    palette = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    w = preset.get("width", 480)
    fps = preset.get("fps", 15)

    try:
        # Pass 1: Generate palette
        if on_progress:
            on_progress(20, "Generating color palette...")
        _run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            "-vf", f"fps={fps},scale={w}:-1:flags=lanczos,palettegen=stats_mode=diff",
            palette,
        ])

        # Pass 2: Apply palette
        if on_progress:
            on_progress(50, "Encoding GIF...")
        _run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path, "-i", palette,
            "-lavfi", f"fps={fps},scale={w}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
            output_path,
        ])

        if on_progress:
            on_progress(100, "GIF exported")
        return output_path
    finally:
        if os.path.exists(palette):
            os.unlink(palette)
