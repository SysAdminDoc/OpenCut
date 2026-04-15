"""
OpenCut Multi-Platform Auto-Publish Preparation

Generate platform-optimized export packages for YouTube, TikTok,
Instagram (Reels/Stories/Feed), Twitter/X, LinkedIn, Facebook,
Vimeo, and Podcast (RSS). Per-platform resolution, codec, file size,
thumbnail, metadata limits, and caption file format.
"""

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Platform specifications
# ---------------------------------------------------------------------------
PLATFORM_SPECS: Dict[str, dict] = {
    "youtube": {
        "label": "YouTube",
        "max_width": 3840, "max_height": 2160,
        "aspect_ratios": ["16:9", "9:16", "1:1", "4:3"],
        "preferred_aspect": "16:9",
        "codec": "libx264", "audio_codec": "aac",
        "audio_bitrate": "256k", "crf": 18, "preset": "slow",
        "max_file_size_mb": 256000,
        "max_duration_sec": 43200,
        "title_max_chars": 100,
        "description_max_chars": 5000,
        "hashtag_max": 15,
        "caption_format": "srt",
        "thumbnail_width": 1280, "thumbnail_height": 720,
        "container": "mp4",
    },
    "tiktok": {
        "label": "TikTok",
        "max_width": 1080, "max_height": 1920,
        "aspect_ratios": ["9:16"],
        "preferred_aspect": "9:16",
        "codec": "libx264", "audio_codec": "aac",
        "audio_bitrate": "192k", "crf": 23, "preset": "fast",
        "max_file_size_mb": 287,
        "max_duration_sec": 600,
        "title_max_chars": 150,
        "description_max_chars": 2200,
        "hashtag_max": 100,
        "caption_format": "srt",
        "thumbnail_width": 1080, "thumbnail_height": 1920,
        "container": "mp4",
    },
    "instagram_reels": {
        "label": "Instagram Reels",
        "max_width": 1080, "max_height": 1920,
        "aspect_ratios": ["9:16"],
        "preferred_aspect": "9:16",
        "codec": "libx264", "audio_codec": "aac",
        "audio_bitrate": "192k", "crf": 23, "preset": "fast",
        "max_file_size_mb": 650,
        "max_duration_sec": 5400,
        "title_max_chars": 0,
        "description_max_chars": 2200,
        "hashtag_max": 30,
        "caption_format": "srt",
        "thumbnail_width": 1080, "thumbnail_height": 1920,
        "container": "mp4",
    },
    "instagram_stories": {
        "label": "Instagram Stories",
        "max_width": 1080, "max_height": 1920,
        "aspect_ratios": ["9:16"],
        "preferred_aspect": "9:16",
        "codec": "libx264", "audio_codec": "aac",
        "audio_bitrate": "192k", "crf": 23, "preset": "fast",
        "max_file_size_mb": 250,
        "max_duration_sec": 60,
        "title_max_chars": 0,
        "description_max_chars": 0,
        "hashtag_max": 0,
        "caption_format": "srt",
        "thumbnail_width": 1080, "thumbnail_height": 1920,
        "container": "mp4",
    },
    "instagram_feed": {
        "label": "Instagram Feed",
        "max_width": 1080, "max_height": 1350,
        "aspect_ratios": ["4:5", "1:1", "16:9"],
        "preferred_aspect": "4:5",
        "codec": "libx264", "audio_codec": "aac",
        "audio_bitrate": "192k", "crf": 23, "preset": "fast",
        "max_file_size_mb": 650,
        "max_duration_sec": 3600,
        "title_max_chars": 0,
        "description_max_chars": 2200,
        "hashtag_max": 30,
        "caption_format": "srt",
        "thumbnail_width": 1080, "thumbnail_height": 1080,
        "container": "mp4",
    },
    "twitter": {
        "label": "Twitter / X",
        "max_width": 1920, "max_height": 1200,
        "aspect_ratios": ["16:9", "1:1"],
        "preferred_aspect": "16:9",
        "codec": "libx264", "audio_codec": "aac",
        "audio_bitrate": "192k", "crf": 23, "preset": "fast",
        "max_file_size_mb": 512,
        "max_duration_sec": 140,
        "title_max_chars": 0,
        "description_max_chars": 280,
        "hashtag_max": 10,
        "caption_format": "srt",
        "thumbnail_width": 1200, "thumbnail_height": 675,
        "container": "mp4",
    },
    "linkedin": {
        "label": "LinkedIn",
        "max_width": 1920, "max_height": 1080,
        "aspect_ratios": ["16:9", "1:1", "9:16"],
        "preferred_aspect": "16:9",
        "codec": "libx264", "audio_codec": "aac",
        "audio_bitrate": "192k", "crf": 23, "preset": "fast",
        "max_file_size_mb": 5120,
        "max_duration_sec": 600,
        "title_max_chars": 150,
        "description_max_chars": 3000,
        "hashtag_max": 5,
        "caption_format": "srt",
        "thumbnail_width": 1200, "thumbnail_height": 627,
        "container": "mp4",
    },
    "facebook": {
        "label": "Facebook",
        "max_width": 1920, "max_height": 1080,
        "aspect_ratios": ["16:9", "9:16", "1:1", "4:5"],
        "preferred_aspect": "16:9",
        "codec": "libx264", "audio_codec": "aac",
        "audio_bitrate": "192k", "crf": 23, "preset": "fast",
        "max_file_size_mb": 10240,
        "max_duration_sec": 14400,
        "title_max_chars": 255,
        "description_max_chars": 63206,
        "hashtag_max": 30,
        "caption_format": "srt",
        "thumbnail_width": 1200, "thumbnail_height": 630,
        "container": "mp4",
    },
    "vimeo": {
        "label": "Vimeo",
        "max_width": 7680, "max_height": 4320,
        "aspect_ratios": ["16:9", "4:3", "1:1"],
        "preferred_aspect": "16:9",
        "codec": "libx264", "audio_codec": "aac",
        "audio_bitrate": "320k", "crf": 18, "preset": "slow",
        "max_file_size_mb": 256000,
        "max_duration_sec": 0,
        "title_max_chars": 128,
        "description_max_chars": 5000,
        "hashtag_max": 20,
        "caption_format": "srt",
        "thumbnail_width": 1920, "thumbnail_height": 1080,
        "container": "mp4",
    },
    "podcast": {
        "label": "Podcast (RSS)",
        "max_width": 0, "max_height": 0,
        "aspect_ratios": [],
        "preferred_aspect": "",
        "codec": "libmp3lame", "audio_codec": "libmp3lame",
        "audio_bitrate": "192k", "crf": 0, "preset": "",
        "max_file_size_mb": 200,
        "max_duration_sec": 0,
        "title_max_chars": 255,
        "description_max_chars": 4000,
        "hashtag_max": 0,
        "caption_format": "txt",
        "thumbnail_width": 3000, "thumbnail_height": 3000,
        "container": "mp3",
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PublishMetadata:
    """Metadata for a platform publish."""
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    category: str = ""
    language: str = "en"
    privacy: str = "private"
    scheduled_time: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "hashtags": list(self.hashtags),
            "category": self.category,
            "language": self.language,
            "privacy": self.privacy,
            "scheduled_time": self.scheduled_time,
        }


@dataclass
class ValidationError:
    """A single validation error."""
    field: str
    message: str
    severity: str = "error"  # error, warning

    def to_dict(self) -> dict:
        return {"field": self.field, "message": self.message, "severity": self.severity}


@dataclass
class PublishPackage:
    """A prepared publish package for a single platform."""
    platform: str = ""
    video_path: str = ""
    thumbnail_path: str = ""
    caption_path: str = ""
    metadata: Dict = field(default_factory=dict)
    validation_errors: List[ValidationError] = field(default_factory=list)
    file_size_mb: float = 0.0
    resolution: str = ""
    duration_sec: float = 0.0
    is_valid: bool = True

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "video_path": self.video_path,
            "thumbnail_path": self.thumbnail_path,
            "caption_path": self.caption_path,
            "metadata": self.metadata,
            "validation_errors": [e.to_dict() for e in self.validation_errors],
            "file_size_mb": round(self.file_size_mb, 2),
            "resolution": self.resolution,
            "duration_sec": round(self.duration_sec, 2),
            "is_valid": self.is_valid,
        }


@dataclass
class PublishManifest:
    """Manifest containing all platform packages for a single source."""
    source_path: str = ""
    packages: List[PublishPackage] = field(default_factory=list)
    output_dir: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "packages": [p.to_dict() for p in self.packages],
            "output_dir": self.output_dir,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_metadata(platform: str, metadata: dict) -> List[ValidationError]:
    """Validate metadata against platform-specific constraints."""
    errors: List[ValidationError] = []
    spec = PLATFORM_SPECS.get(platform)
    if not spec:
        errors.append(ValidationError("platform", f"Unknown platform: {platform}"))
        return errors

    title = metadata.get("title", "")
    description = metadata.get("description", "")
    hashtags = metadata.get("hashtags", [])

    if spec["title_max_chars"] > 0 and len(title) > spec["title_max_chars"]:
        errors.append(ValidationError(
            "title",
            f"Title exceeds {spec['title_max_chars']} characters "
            f"(got {len(title)})",
        ))

    if spec["description_max_chars"] > 0 and len(description) > spec["description_max_chars"]:
        errors.append(ValidationError(
            "description",
            f"Description exceeds {spec['description_max_chars']} characters "
            f"(got {len(description)})",
        ))

    if spec["hashtag_max"] > 0 and len(hashtags) > spec["hashtag_max"]:
        errors.append(ValidationError(
            "hashtags",
            f"Too many hashtags for {spec['label']}: max {spec['hashtag_max']}, "
            f"got {len(hashtags)}",
        ))

    # Validate hashtag format
    for tag in hashtags:
        if not isinstance(tag, str):
            continue
        cleaned = tag.lstrip("#")
        if not re.match(r'^[A-Za-z0-9_]+$', cleaned):
            errors.append(ValidationError(
                "hashtags",
                f"Invalid hashtag format: {tag}",
                severity="warning",
            ))

    return errors


def validate_video_for_platform(platform: str, video_info: dict) -> List[ValidationError]:
    """Validate video specs against platform requirements."""
    errors: List[ValidationError] = []
    spec = PLATFORM_SPECS.get(platform)
    if not spec:
        return errors

    width = video_info.get("width", 0)
    height = video_info.get("height", 0)
    duration = video_info.get("duration", 0)

    if spec["max_width"] > 0 and width > spec["max_width"]:
        errors.append(ValidationError(
            "resolution",
            f"Width {width} exceeds max {spec['max_width']} for {spec['label']}",
            severity="warning",
        ))

    if spec["max_height"] > 0 and height > spec["max_height"]:
        errors.append(ValidationError(
            "resolution",
            f"Height {height} exceeds max {spec['max_height']} for {spec['label']}",
            severity="warning",
        ))

    if spec["max_duration_sec"] > 0 and duration > spec["max_duration_sec"]:
        errors.append(ValidationError(
            "duration",
            f"Duration {duration:.0f}s exceeds max {spec['max_duration_sec']}s "
            f"for {spec['label']}",
        ))

    return errors


# ---------------------------------------------------------------------------
# Resolution / aspect ratio fitting
# ---------------------------------------------------------------------------
def _compute_target_resolution(spec: dict, src_width: int, src_height: int) -> tuple:
    """Compute target width/height fitting the platform spec."""
    max_w = spec.get("max_width", 0)
    max_h = spec.get("max_height", 0)

    if max_w <= 0 or max_h <= 0:
        # Audio-only platform (podcast)
        return 0, 0

    target_w = min(src_width, max_w)
    target_h = min(src_height, max_h)

    # Maintain aspect ratio
    if src_width > 0 and src_height > 0:
        src_ratio = src_width / src_height
        target_ratio = target_w / max(target_h, 1)

        if target_ratio > src_ratio:
            target_w = int(target_h * src_ratio)
        else:
            target_h = int(target_w / src_ratio)

    # Ensure even dimensions for codec compat
    target_w = target_w - (target_w % 2)
    target_h = target_h - (target_h % 2)

    return max(target_w, 2), max(target_h, 2)


def _format_hashtags(hashtags: List[str], platform: str) -> str:
    """Format hashtags for a specific platform."""
    formatted = []
    for tag in hashtags:
        tag = tag.strip()
        if not tag.startswith("#"):
            tag = f"#{tag}"
        formatted.append(tag)
    return " ".join(formatted)


# ---------------------------------------------------------------------------
# Export / render for platform
# ---------------------------------------------------------------------------
def _render_for_platform(input_path: str, platform: str, output_dir: str,
                         on_progress: Optional[Callable] = None) -> str:
    """Render a platform-optimized version of the input video."""
    spec = PLATFORM_SPECS[platform]
    info = get_video_info(input_path)

    base = os.path.splitext(os.path.basename(input_path))[0]
    ext = f".{spec['container']}"
    out_name = f"{base}_{platform}{ext}"
    out_path = os.path.join(output_dir, out_name)

    if spec["container"] == "mp3":
        # Podcast: audio-only export
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", input_path,
            "-vn",
            "-c:a", spec["audio_codec"],
            "-b:a", spec["audio_bitrate"],
            out_path,
        ]
        if on_progress:
            on_progress(30)
        run_ffmpeg(cmd)
        if on_progress:
            on_progress(90)
        return out_path

    # Video export
    target_w, target_h = _compute_target_resolution(spec, info["width"], info["height"])

    builder = FFmpegCmd()
    builder.input(input_path)

    # Scale if needed
    if target_w != info["width"] or target_h != info["height"]:
        builder.video_filter(f"scale={target_w}:{target_h}:flags=lanczos")

    builder.video_codec(spec["codec"], crf=spec["crf"], preset=spec["preset"])
    builder.audio_codec(spec["audio_codec"], bitrate=spec["audio_bitrate"])
    builder.faststart()
    builder.output(out_path)

    cmd = builder.build()

    if on_progress:
        on_progress(20)

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(90)

    return out_path


def _generate_thumbnail(input_path: str, platform: str,
                        output_dir: str) -> str:
    """Extract a thumbnail at the 10% mark of the video."""
    spec = PLATFORM_SPECS[platform]
    if spec["thumbnail_width"] <= 0:
        return ""

    info = get_video_info(input_path)
    seek_time = max(info.get("duration", 10) * 0.1, 1.0)

    base = os.path.splitext(os.path.basename(input_path))[0]
    thumb_path = os.path.join(output_dir, f"{base}_{platform}_thumb.jpg")

    tw = spec["thumbnail_width"]
    th = spec["thumbnail_height"]

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-ss", str(seek_time),
        "-i", input_path,
        "-vf", f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
               f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:black",
        "-vframes", "1",
        "-q:v", "2",
        thumb_path,
    ]

    try:
        run_ffmpeg(cmd)
        return thumb_path
    except Exception as e:
        logger.warning("Failed to generate thumbnail for %s: %s", platform, e)
        return ""


# ---------------------------------------------------------------------------
# Package preparation
# ---------------------------------------------------------------------------
def prepare_publish_package(input_path: str, platform: str,
                            metadata: Optional[dict] = None,
                            output_dir: str = "",
                            caption_path: str = "",
                            on_progress: Optional[Callable] = None) -> PublishPackage:
    """
    Prepare a publish package for a single platform.

    Args:
        input_path: Path to source video.
        platform: Target platform key.
        metadata: Dict with title, description, hashtags, tags, etc.
        output_dir: Directory for output files.
        caption_path: Optional path to caption/subtitle file.
        on_progress: Callback taking int (0-100).

    Returns:
        PublishPackage with rendered video, thumbnail, and validated metadata.
    """
    if platform not in PLATFORM_SPECS:
        raise ValueError(f"Unsupported platform: {platform}. "
                         f"Supported: {', '.join(sorted(PLATFORM_SPECS.keys()))}")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    meta = dict(metadata or {})
    spec = PLATFORM_SPECS[platform]

    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_path), "publish")
    os.makedirs(output_dir, exist_ok=True)

    package = PublishPackage(platform=platform, metadata=meta)

    if on_progress:
        on_progress(5)

    # Validate metadata
    meta_errors = validate_metadata(platform, meta)
    video_info = get_video_info(input_path)
    video_errors = validate_video_for_platform(platform, video_info)
    package.validation_errors = meta_errors + video_errors

    # Check for blocking errors
    blocking = [e for e in package.validation_errors if e.severity == "error"]
    if blocking:
        package.is_valid = False

    if on_progress:
        on_progress(10)

    # Render platform-optimized video
    try:
        video_path = _render_for_platform(input_path, platform, output_dir,
                                          on_progress=on_progress)
        package.video_path = video_path

        # Get output file info
        if os.path.isfile(video_path):
            size_bytes = os.path.getsize(video_path)
            package.file_size_mb = size_bytes / (1024 * 1024)
            out_info = get_video_info(video_path)
            package.resolution = f"{out_info['width']}x{out_info['height']}"
            package.duration_sec = out_info.get("duration", 0)

            # Check file size
            if spec["max_file_size_mb"] > 0 and package.file_size_mb > spec["max_file_size_mb"]:
                package.validation_errors.append(ValidationError(
                    "file_size",
                    f"File size {package.file_size_mb:.1f}MB exceeds "
                    f"max {spec['max_file_size_mb']}MB for {spec['label']}",
                ))
                package.is_valid = False

    except Exception as e:
        logger.error("Render for %s failed: %s", platform, e)
        package.validation_errors.append(ValidationError(
            "render", f"Render failed: {e}"))
        package.is_valid = False

    if on_progress:
        on_progress(85)

    # Generate thumbnail
    try:
        thumb = _generate_thumbnail(input_path, platform, output_dir)
        package.thumbnail_path = thumb
    except Exception as e:
        logger.debug("Thumbnail generation failed for %s: %s", platform, e)

    # Copy caption file if provided
    if caption_path and os.path.isfile(caption_path):
        base = os.path.splitext(os.path.basename(input_path))[0]
        cap_ext = spec.get("caption_format", "srt")
        dest_cap = os.path.join(output_dir, f"{base}_{platform}.{cap_ext}")
        try:
            shutil.copy2(caption_path, dest_cap)
            package.caption_path = dest_cap
        except Exception as e:
            logger.debug("Caption copy failed: %s", e)

    # Format metadata for platform
    if "hashtags" in meta:
        meta["hashtags_formatted"] = _format_hashtags(meta["hashtags"], platform)

    if on_progress:
        on_progress(100)

    return package


def batch_prepare(input_path: str, platforms: List[str],
                  metadata: Optional[dict] = None,
                  output_dir: str = "",
                  caption_path: str = "",
                  on_progress: Optional[Callable] = None) -> PublishManifest:
    """
    Prepare publish packages for multiple platforms from a single source.

    Returns a PublishManifest with all platform packages.
    """
    import datetime

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not platforms:
        raise ValueError("No platforms specified")

    if not output_dir:
        output_dir = os.path.join(os.path.dirname(input_path), "publish")
    os.makedirs(output_dir, exist_ok=True)

    manifest = PublishManifest(
        source_path=input_path,
        output_dir=output_dir,
        created_at=datetime.datetime.now().isoformat(),
    )

    total = len(platforms)
    for i, platform in enumerate(platforms):
        def _inner_progress(pct):
            if on_progress:
                base_pct = int((i / max(total, 1)) * 100)
                step = int((1 / max(total, 1)) * 100)
                overall = base_pct + int(pct * step / 100)
                on_progress(min(overall, 99))

        try:
            pkg = prepare_publish_package(
                input_path, platform, metadata, output_dir,
                caption_path, on_progress=_inner_progress,
            )
            manifest.packages.append(pkg)
        except Exception as e:
            logger.error("Failed to prepare package for %s: %s", platform, e)
            pkg = PublishPackage(
                platform=platform, is_valid=False,
                validation_errors=[ValidationError("prepare", str(e))],
            )
            manifest.packages.append(pkg)

    # Save manifest
    manifest_path = os.path.join(output_dir, "publish_manifest.json")
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
    except Exception as e:
        logger.warning("Failed to save publish manifest: %s", e)

    if on_progress:
        on_progress(100)

    return manifest


# ---------------------------------------------------------------------------
# Utility: list platforms
# ---------------------------------------------------------------------------
def list_platforms() -> List[dict]:
    """Return list of supported platforms with their specs."""
    result = []
    for key, spec in sorted(PLATFORM_SPECS.items()):
        result.append({
            "key": key,
            "label": spec["label"],
            "max_resolution": f"{spec['max_width']}x{spec['max_height']}" if spec["max_width"] > 0 else "audio-only",
            "preferred_aspect": spec["preferred_aspect"],
            "max_duration_sec": spec["max_duration_sec"],
            "max_file_size_mb": spec["max_file_size_mb"],
            "title_max_chars": spec["title_max_chars"],
            "description_max_chars": spec["description_max_chars"],
            "hashtag_max": spec["hashtag_max"],
            "caption_format": spec["caption_format"],
            "container": spec["container"],
        })
    return result
