"""
Multi-Platform Batch Publish (5.1)

Export and publish videos to YouTube, TikTok, Instagram, and Twitter
simultaneously with per-platform format/resolution presets.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import FFmpegCmd, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Platform presets
# ---------------------------------------------------------------------------
PLATFORM_PRESETS: Dict[str, dict] = {
    "youtube": {
        "max_width": 3840, "max_height": 2160,
        "aspect": "16:9", "max_duration": 43200,
        "format": "mp4", "codec": "libx264",
        "audio_codec": "aac", "audio_bitrate": "256k",
        "crf": 18, "preset": "fast",
    },
    "tiktok": {
        "max_width": 1080, "max_height": 1920,
        "aspect": "9:16", "max_duration": 600,
        "format": "mp4", "codec": "libx264",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "crf": 23, "preset": "fast",
    },
    "instagram": {
        "max_width": 1080, "max_height": 1350,
        "aspect": "4:5", "max_duration": 3600,
        "format": "mp4", "codec": "libx264",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "crf": 23, "preset": "fast",
    },
    "twitter": {
        "max_width": 1920, "max_height": 1200,
        "aspect": "16:9", "max_duration": 140,
        "format": "mp4", "codec": "libx264",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "crf": 23, "preset": "fast",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PublishConfig:
    """Configuration for a single platform publish."""
    platform: str
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    privacy: str = "private"  # "public", "private", "unlisted"
    thumbnail_path: str = ""
    custom_width: int = 0
    custom_height: int = 0
    trim_start: float = 0.0
    trim_end: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PublishQueueItem:
    """An item in the publish queue."""
    platform: str
    config: PublishConfig
    export_path: str = ""
    status: str = "pending"  # "pending", "exporting", "exported", "publishing", "done", "error"
    error: str = ""
    publish_url: str = ""


@dataclass
class PublishQueue:
    """Queue of platform publishes."""
    video_path: str
    items: List[PublishQueueItem] = field(default_factory=list)
    completed: int = 0
    total: int = 0


# ---------------------------------------------------------------------------
# Queue Creation
# ---------------------------------------------------------------------------
def create_publish_queue(
    video_path: str,
    platforms: List[str],
    config: Optional[Dict[str, PublishConfig]] = None,
    on_progress: Optional[Callable] = None,
) -> PublishQueue:
    """Create a publish queue for multiple platforms.

    Args:
        video_path: Source video file path.
        platforms: List of platform names (youtube, tiktok, instagram, twitter).
        config: Optional per-platform PublishConfig overrides.
        on_progress: Optional callback(pct, msg).

    Returns:
        PublishQueue with items for each platform.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    valid_platforms = set(PLATFORM_PRESETS.keys())
    for p in platforms:
        if p.lower() not in valid_platforms:
            raise ValueError(
                f"Unknown platform: {p}. Supported: {sorted(valid_platforms)}"
            )

    if on_progress:
        on_progress(10, f"Creating publish queue for {len(platforms)} platforms...")

    config = config or {}
    items = []
    for p in platforms:
        p_lower = p.lower()
        p_config = config.get(p_lower, PublishConfig(platform=p_lower))
        if not p_config.platform:
            p_config.platform = p_lower
        items.append(PublishQueueItem(platform=p_lower, config=p_config))

    queue = PublishQueue(
        video_path=video_path,
        items=items,
        total=len(items),
    )

    if on_progress:
        on_progress(100, f"Queue created with {len(items)} items")

    return queue


# ---------------------------------------------------------------------------
# Per-Platform Export
# ---------------------------------------------------------------------------
def _compute_scale(src_w: int, src_h: int, max_w: int, max_h: int,
                   target_aspect: str) -> str:
    """Compute FFmpeg scale+pad filter for target dimensions."""
    # Parse target aspect ratio
    parts = target_aspect.split(":")
    ar_w, ar_h = int(parts[0]), int(parts[1])

    # Calculate target dimensions within max bounds
    target_w = min(src_w, max_w)
    target_h = min(src_h, max_h)

    # Adjust to target aspect ratio
    if ar_w > ar_h:
        target_h = min(target_h, int(target_w * ar_h / ar_w))
    else:
        target_w = min(target_w, int(target_h * ar_w / ar_h))

    # Ensure even dimensions
    target_w = target_w - (target_w % 2)
    target_h = target_h - (target_h % 2)

    if target_w <= 0:
        target_w = 2
    if target_h <= 0:
        target_h = 2

    return (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black"
    )


def export_for_platform(
    video_path: str,
    platform: str,
    output_dir: str = "",
    config: Optional[PublishConfig] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """Export a video formatted for a specific platform.

    Args:
        video_path: Source video file path.
        platform: Target platform name.
        output_dir: Output directory (default: same as source).
        config: Optional PublishConfig with trim/size overrides.
        on_progress: Optional callback(pct, msg).

    Returns:
        Path to the exported file.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    platform = platform.lower()
    preset = PLATFORM_PRESETS.get(platform)
    if not preset:
        raise ValueError(f"Unknown platform: {platform}")

    if on_progress:
        on_progress(5, f"Preparing {platform} export...")

    info = get_video_info(video_path)
    src_w, src_h = info["width"], info["height"]

    config = config or PublishConfig(platform=platform)
    max_w = config.custom_width or preset["max_width"]
    max_h = config.custom_height or preset["max_height"]

    out_dir = output_dir or os.path.dirname(os.path.abspath(video_path))
    os.makedirs(out_dir, exist_ok=True)
    out_path = output_path(video_path, platform, out_dir)

    scale_filter = _compute_scale(src_w, src_h, max_w, max_h, preset["aspect"])

    if on_progress:
        on_progress(15, f"Encoding for {platform}...")

    cmd = (FFmpegCmd()
           .input(video_path)
           .video_filter(scale_filter)
           .video_codec(preset["codec"], crf=preset["crf"], preset=preset["preset"])
           .audio_codec(preset["audio_codec"], bitrate=preset["audio_bitrate"])
           .faststart()
           .output(out_path))

    if config.trim_start > 0:
        cmd.seek(start=config.trim_start)
    if config.trim_end > 0:
        cmd.seek(end=config.trim_end)

    run_ffmpeg(cmd.build())

    if on_progress:
        on_progress(100, f"Exported for {platform}")

    return out_path


# ---------------------------------------------------------------------------
# Publish (stub — real API integration is platform-specific)
# ---------------------------------------------------------------------------
def publish_to_platform(
    video_path: str,
    platform: str,
    credentials: Optional[Dict[str, str]] = None,
    config: Optional[PublishConfig] = None,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Publish an exported video to a platform.

    This is a stub that validates inputs and simulates publishing.
    Real implementations would use platform-specific APIs.

    Args:
        video_path: Path to the platform-formatted video.
        platform: Target platform.
        credentials: API credentials (api_key, oauth_token, etc.).
        config: Publish configuration (title, description, etc.).
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with publish result including status and URL.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    platform = platform.lower()
    if platform not in PLATFORM_PRESETS:
        raise ValueError(f"Unknown platform: {platform}")

    credentials = credentials or {}
    config = config or PublishConfig(platform=platform)

    if on_progress:
        on_progress(10, f"Validating {platform} credentials...")

    # Validate file size and duration
    info = get_video_info(video_path)
    max_dur = PLATFORM_PRESETS[platform]["max_duration"]
    if info["duration"] > max_dur:
        raise ValueError(
            f"Video duration ({info['duration']:.0f}s) exceeds "
            f"{platform} maximum ({max_dur}s)"
        )

    if on_progress:
        on_progress(50, f"Publishing to {platform}...")

    # Stub: in production, call platform API here
    result = {
        "platform": platform,
        "status": "pending_upload",
        "title": config.title or os.path.basename(video_path),
        "video_path": video_path,
        "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 1),
        "message": (
            f"Video prepared for {platform} upload. "
            "Connect platform API credentials to enable automatic upload."
        ),
    }

    if on_progress:
        on_progress(100, f"Publish to {platform} complete")

    return result
