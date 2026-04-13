"""
OpenCut Export Presets Module v0.8.0

Pre-configured export profiles for common platforms:
- YouTube (1080p, 4K, Shorts)
- TikTok / Instagram Reels (9:16 vertical)
- Instagram Feed / Story
- Twitter/X
- LinkedIn
- Podcast (audio-only)
- Archive (ProRes, DNxHR)
- Custom resolution/codec
- Hardware-accelerated (NVENC, QSV, AMF, VideoToolbox)

Each preset defines resolution, codec, bitrate, aspect ratio, and
platform-specific constraints (max duration, file size, etc.).

Presets with ``hw_accel=True`` route through the hardware-accelerated
encoding pipeline with automatic software fallback.
"""

import logging
import os
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, run_ffmpeg

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
    # === Archive / Professional — ProRes ===
    "prores_proxy": {
        "label": "ProRes Proxy",
        "description": "Apple ProRes Proxy — lightweight offline editing",
        "category": "archive",
        "width": 0, "height": 0,
        "codec": "prores_ks", "profile": "0",
        "audio_codec": "pcm_s16le",
        "pix_fmt": "yuv422p10le",
        "ext": ".mov",
    },
    "prores_lt": {
        "label": "ProRes LT",
        "description": "Apple ProRes LT — reduced data rate for editing",
        "category": "archive",
        "width": 0, "height": 0,
        "codec": "prores_ks", "profile": "1",
        "audio_codec": "pcm_s16le",
        "pix_fmt": "yuv422p10le",
        "ext": ".mov",
    },
    "prores_422": {
        "label": "ProRes 422",
        "description": "Apple ProRes 422 for post-production",
        "category": "archive",
        "width": 0, "height": 0,
        "codec": "prores_ks", "profile": "2",
        "audio_codec": "pcm_s16le",
        "pix_fmt": "yuv422p10le",
        "ext": ".mov",
    },
    "prores_422hq": {
        "label": "ProRes 422 HQ",
        "description": "Apple ProRes 422 HQ — high quality finishing",
        "category": "archive",
        "width": 0, "height": 0,
        "codec": "prores_ks", "profile": "3",
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
    "prores_4444xq": {
        "label": "ProRes 4444 XQ",
        "description": "Apple ProRes 4444 XQ — highest quality with alpha",
        "category": "archive",
        "width": 0, "height": 0,
        "codec": "prores_ks", "profile": "5",
        "audio_codec": "pcm_s16le",
        "pix_fmt": "yuva444p10le",
        "ext": ".mov",
    },
    # === Archive / Professional — DNxHR ===
    "dnxhr_lb": {
        "label": "DNxHR LB",
        "description": "Avid DNxHR LB — low bandwidth proxy editing",
        "category": "archive",
        "width": 0, "height": 0,
        "codec": "dnxhd", "profile": "dnxhr_lb",
        "audio_codec": "pcm_s16le",
        "pix_fmt": "yuv422p",
        "ext": ".mov",
    },
    "dnxhr_sq": {
        "label": "DNxHR SQ",
        "description": "Avid DNxHR SQ — standard quality editing",
        "category": "archive",
        "width": 0, "height": 0,
        "codec": "dnxhd", "profile": "dnxhr_sq",
        "audio_codec": "pcm_s16le",
        "pix_fmt": "yuv422p",
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
    "dnxhr_hqx": {
        "label": "DNxHR HQX",
        "description": "Avid DNxHR HQX — 12-bit HDR grading workflows",
        "category": "archive",
        "width": 0, "height": 0,
        "codec": "dnxhd", "profile": "dnxhr_hqx",
        "audio_codec": "pcm_s16le",
        "pix_fmt": "yuv422p10le",
        "ext": ".mov",
    },
    "dnxhr_444": {
        "label": "DNxHR 444",
        "description": "Avid DNxHR 444 — full chroma mastering",
        "category": "archive",
        "width": 0, "height": 0,
        "codec": "dnxhd", "profile": "dnxhr_444",
        "audio_codec": "pcm_s16le",
        "pix_fmt": "yuv444p10le",
        "ext": ".mov",
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
    # === AV1 / HEVC ===
    "av1_1080p": {
        "label": "AV1 1080p",
        "description": "AV1 codec — 40% smaller files than H.264 at same quality",
        "category": "web",
        "width": 1920, "height": 1080,
        "codec": "libsvtav1", "crf": 28, "preset": 6,
        "audio_codec": "libopus", "audio_bitrate": "128k",
        "pix_fmt": "yuv420p10le",
        "ext": ".mp4",
    },
    "av1_4k": {
        "label": "AV1 4K",
        "description": "AV1 4K — excellent quality-to-size ratio for archival",
        "category": "web",
        "width": 3840, "height": 2160,
        "codec": "libsvtav1", "crf": 30, "preset": 4,
        "audio_codec": "libopus", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p10le",
        "ext": ".mp4",
    },
    "av1_youtube": {
        "label": "AV1 YouTube",
        "description": "AV1 optimized for YouTube — smaller uploads with premium quality",
        "category": "av1",
        "width": 1920, "height": 1080,
        "codec": "libsvtav1", "crf": 25, "preset": 6,
        "audio_codec": "libopus", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p10le",
        "movflags": "+faststart",
        "ext": ".mp4",
    },
    "av1_archive": {
        "label": "AV1 Archive",
        "description": "AV1 for long-term archival — high quality, small files",
        "category": "av1",
        "width": 0, "height": 0,
        "codec": "libsvtav1", "crf": 18, "preset": 4,
        "audio_codec": "libopus", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p10le",
        "ext": ".mp4",
    },
    "av1_hw_fast": {
        "label": "AV1 HW Fast",
        "description": "AV1 hardware-accelerated fast encode (NVENC/QSV with fallback)",
        "category": "hw_accel",
        "width": 1920, "height": 1080,
        "codec": "av1", "quality": "speed", "hw_type": "auto",
        "audio_codec": "libopus", "audio_bitrate": "128k",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
        "hw_accel": True,
    },
    "hevc_1080p": {
        "label": "HEVC 1080p",
        "description": "H.265/HEVC — 30% smaller than H.264, wide device support",
        "category": "web",
        "width": 1920, "height": 1080,
        "codec": "libx265", "crf": 23, "preset": "medium",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
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
    # === Hardware-Accelerated ===
    "h264_hw_fast": {
        "label": "H.264 HW Accelerated (Fast)",
        "description": "GPU-accelerated H.264 optimized for speed (NVENC/QSV/AMF)",
        "category": "hw_accel",
        "width": 1920, "height": 1080,
        "codec": "h264", "quality": "speed", "hw_type": "auto",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
        "hw_accel": True,
    },
    "h264_hw_quality": {
        "label": "H.264 HW Accelerated (Quality)",
        "description": "GPU-accelerated H.264 optimized for quality (NVENC/QSV/AMF)",
        "category": "hw_accel",
        "width": 1920, "height": 1080,
        "codec": "h264", "quality": "quality", "hw_type": "auto",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
        "hw_accel": True,
    },
    "hevc_hw_fast": {
        "label": "HEVC HW Accelerated (Fast)",
        "description": "GPU-accelerated H.265/HEVC optimized for speed",
        "category": "hw_accel",
        "width": 1920, "height": 1080,
        "codec": "hevc", "quality": "speed", "hw_type": "auto",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
        "hw_accel": True,
    },
    "hevc_hw_quality": {
        "label": "HEVC HW Accelerated (Quality)",
        "description": "GPU-accelerated H.265/HEVC optimized for quality",
        "category": "hw_accel",
        "width": 1920, "height": 1080,
        "codec": "hevc", "quality": "quality", "hw_type": "auto",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
        "hw_accel": True,
    },
    "av1_hw": {
        "label": "AV1 HW Accelerated",
        "description": "GPU-accelerated AV1 encoding (NVENC/QSV/AMF with fallback)",
        "category": "hw_accel",
        "width": 1920, "height": 1080,
        "codec": "av1", "quality": "balanced", "hw_type": "auto",
        "audio_codec": "libopus", "audio_bitrate": "128k",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
        "hw_accel": True,
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

def export_with_preset(
    input_path: str,
    preset_name: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
    hw_accel: Optional[bool] = None,
) -> str:
    """
    Export video using a preset profile.

    Args:
        input_path: Source video file path.
        preset_name: Key from EXPORT_PRESETS.
        output_path: Explicit output path (auto-generated if None).
        output_dir: Directory for auto-generated output path.
        on_progress: Callback(percent, message) for progress updates.
        hw_accel: Override hardware acceleration. If None, uses the
            preset's ``hw_accel`` field. If True and the preset uses a
            software H.264/HEVC/AV1 codec, the codec is replaced with
            the best available HW encoder. If False, forces software.
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

    # Determine if HW acceleration should be used
    use_hw = hw_accel if hw_accel is not None else preset.get("hw_accel", False)

    # --- HW-accelerated preset path ---
    if use_hw and not preset.get("audio_only") and preset.get("ext") != ".gif":
        return _export_hw_accel(input_path, output_path, preset, on_progress)

    # Special handling for GIF (no video info needed)
    if preset.get("ext") == ".gif":
        return _export_gif(input_path, output_path, preset, on_progress)

    # Build FFmpeg command
    cmd = [get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-i", input_path]

    # Audio-only export
    if preset.get("audio_only"):
        cmd += ["-vn"]
        if preset.get("audio_codec"):
            cmd += ["-acodec", preset["audio_codec"]]
        if preset.get("audio_bitrate"):
            cmd += ["-b:a", preset["audio_bitrate"]]
        cmd.append(output_path)
        run_ffmpeg(cmd, timeout=7200)
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
    if preset.get("preset") is not None:
        cmd += ["-preset", str(preset["preset"])]
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

    run_ffmpeg(cmd, timeout=7200)

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


def _export_hw_accel(
    input_path: str,
    output_path: str,
    preset: Dict,
    on_progress: Optional[Callable] = None,
) -> str:
    """Export using hardware-accelerated encoding with software fallback.

    For presets with ``hw_accel=True``, this replaces the software codec
    with the best available HW encoder. It also handles presets that use
    standard software codecs (libx264/libx265/libsvtav1) when the caller
    explicitly requests ``hw_accel=True`` via the ``export_with_preset``
    parameter.
    """
    from opencut.core.hw_accel import get_hw_encode_args

    # Map preset codec to hw_accel codec name
    preset_codec = preset.get("codec", "libx264")
    _SW_TO_CODEC = {
        "libx264": "h264", "h264": "h264",
        "libx265": "hevc", "hevc": "hevc",
        "libsvtav1": "av1", "av1": "av1",
    }
    codec = _SW_TO_CODEC.get(preset_codec, "h264")
    quality = preset.get("quality", "balanced")
    hw_type = preset.get("hw_type", "auto")

    if on_progress:
        on_progress(8, "Detecting hardware encoders...")

    # Get HW encoding args (handles detection + fallback internally)
    hw_args = get_hw_encode_args(codec=codec, quality=quality, hw_type=hw_type)

    # Build the full FFmpeg command with video filters from the preset
    cmd = [get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-i", input_path]

    # Video filters for scaling (same logic as software path)
    vf_parts = []
    target_w = preset.get("width", 0)
    target_h = preset.get("height", 0)

    if target_w > 0 and target_h > 0:
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

    # HW encoder args (codec + quality flags)
    cmd.extend(hw_args)

    # Pixel format
    if preset.get("pix_fmt"):
        cmd += ["-pix_fmt", preset["pix_fmt"]]

    # Bitrate limits (if the preset specifies them)
    if preset.get("maxrate"):
        cmd += ["-maxrate", preset["maxrate"]]
    if preset.get("bufsize"):
        cmd += ["-bufsize", preset["bufsize"]]

    # Faststart for MP4
    if preset.get("ext", ".mp4") == ".mp4":
        cmd += ["-movflags", "+faststart"]

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
        # Figure out which encoder we ended up with
        encoder_name = "unknown"
        for i, arg in enumerate(hw_args):
            if arg == "-c:v" and i + 1 < len(hw_args):
                encoder_name = hw_args[i + 1]
                break
        on_progress(10, f"Encoding with {encoder_name}...")

    run_ffmpeg(cmd, timeout=7200)

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
    preset: Dict,
    on_progress: Optional[Callable] = None,
) -> str:
    """Export as optimized GIF using two-pass palette generation."""
    import tempfile

    _ntf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    palette = _ntf.name
    _ntf.close()
    w = preset.get("width", 480)
    fps = preset.get("fps", 15)

    try:
        # Pass 1: Generate palette
        if on_progress:
            on_progress(20, "Generating color palette...")
        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            "-vf", f"fps={fps},scale={w}:-1:flags=lanczos,palettegen=stats_mode=diff",
            palette,
        ], timeout=7200)

        # Pass 2: Apply palette
        if on_progress:
            on_progress(50, "Encoding GIF...")
        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path, "-i", palette,
            "-lavfi", f"fps={fps},scale={w}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
            output_path,
        ], timeout=7200)

        if on_progress:
            on_progress(100, "GIF exported")
        return output_path
    finally:
        if os.path.exists(palette):
            os.unlink(palette)
