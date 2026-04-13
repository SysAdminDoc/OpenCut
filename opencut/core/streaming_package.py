"""
OpenCut HLS / DASH Streaming Package v1.0.0

Multi-quality adaptive streaming package generation:
  - HLS: multi-bitrate .m3u8 manifests with .ts segments
  - DASH: .mpd manifests with segmented .m4s files
  - Automatic rendition selection based on source resolution
  - ZIP packaging for easy upload
"""

import logging
import os
import zipfile
from dataclasses import dataclass
from typing import Callable, List, Optional

from opencut.helpers import (
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Rendition Definitions
# ---------------------------------------------------------------------------

@dataclass
class Rendition:
    """A single quality rendition for adaptive streaming."""
    name: str
    width: int
    height: int
    video_bitrate: str
    audio_bitrate: str
    profile: str = "main"
    level: str = "3.1"
    maxrate: str = ""
    bufsize: str = ""

    def __post_init__(self):
        if not self.maxrate:
            self.maxrate = self.video_bitrate
        if not self.bufsize:
            # Buffer size typically 2x bitrate
            bps = int(self.video_bitrate.rstrip("kKmM"))
            unit = self.video_bitrate[-1]
            self.bufsize = f"{bps * 2}{unit}"


# Pre-defined renditions
ALL_RENDITIONS = {
    "240p": Rendition("240p", 426, 240, "400k", "64k", "baseline", "2.1"),
    "360p": Rendition("360p", 640, 360, "800k", "96k", "main", "3.0"),
    "480p": Rendition("480p", 854, 480, "1400k", "128k", "main", "3.1"),
    "720p": Rendition("720p", 1280, 720, "2800k", "128k", "high", "3.1"),
    "1080p": Rendition("1080p", 1920, 1080, "5000k", "192k", "high", "4.0"),
    "1440p": Rendition("1440p", 2560, 1440, "8000k", "192k", "high", "5.0"),
    "2160p": Rendition("2160p", 3840, 2160, "15000k", "256k", "high", "5.1"),
}


def get_rendition_configs(
    source_resolution: Optional[tuple] = None,
    source_path: Optional[str] = None,
) -> List[Rendition]:
    """Get appropriate rendition configs based on source resolution.

    Only includes renditions at or below the source resolution.

    Args:
        source_resolution: Tuple (width, height) of source video.
        source_path: Path to source video (probed if resolution not given).

    Returns:
        List of Rendition objects from lowest to highest quality.
    """
    if source_resolution is None and source_path:
        info = get_video_info(source_path)
        source_resolution = (info["width"], info["height"])

    if source_resolution is None:
        source_resolution = (1920, 1080)

    src_h = source_resolution[1]

    # Select renditions at or below source resolution
    selected = []
    for name, rend in sorted(ALL_RENDITIONS.items(), key=lambda kv: kv[1].height):
        if rend.height <= src_h:
            selected.append(rend)

    # Always include at least 240p
    if not selected:
        selected = [ALL_RENDITIONS["240p"]]

    return selected


# ---------------------------------------------------------------------------
# HLS Package
# ---------------------------------------------------------------------------

def create_hls_package(
    video_path: str,
    output_dir: str = "",
    renditions: Optional[List[str]] = None,
    segment_duration: int = 6,
    include_zip: bool = True,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Generate an HLS streaming package with multiple quality renditions.

    Creates a master .m3u8 playlist with variant streams at different
    quality levels, each with its own .m3u8 and .ts segments.

    Args:
        video_path: Path to the source video.
        output_dir: Output directory. Created if not existing.
        renditions: List of rendition names (e.g. ["480p", "720p", "1080p"]).
            If None, auto-selected based on source resolution.
        segment_duration: HLS segment duration in seconds.
        include_zip: Create a zip archive of the package.
        on_progress: Optional callback (percent, message).

    Returns:
        Dict with output_dir, master_playlist, renditions info, zip_path.

    Raises:
        FileNotFoundError: If video_path does not exist.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input file not found: {video_path}")

    get_video_info(video_path)

    if not output_dir:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), f"{base}_hls")

    os.makedirs(output_dir, exist_ok=True)

    # Determine renditions
    if renditions:
        rend_list = [ALL_RENDITIONS[r] for r in renditions if r in ALL_RENDITIONS]
        if not rend_list:
            rend_list = get_rendition_configs(source_path=video_path)
    else:
        rend_list = get_rendition_configs(source_path=video_path)

    if on_progress:
        on_progress(5, f"Creating HLS package ({len(rend_list)} renditions)...")

    segment_duration = max(2, min(10, int(segment_duration)))
    created_renditions = []

    for i, rend in enumerate(rend_list):
        if on_progress:
            pct = 10 + int((i / len(rend_list)) * 70)
            on_progress(pct, f"Encoding {rend.name} rendition...")

        rend_dir = os.path.join(output_dir, rend.name)
        os.makedirs(rend_dir, exist_ok=True)

        playlist_path = os.path.join(rend_dir, "playlist.m3u8")
        segment_pattern = os.path.join(rend_dir, "segment_%03d.ts")

        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-b:v", rend.video_bitrate,
            "-maxrate", rend.maxrate,
            "-bufsize", rend.bufsize,
            "-profile:v", rend.profile,
            "-level", rend.level,
            "-vf", f"scale={rend.width}:{rend.height}:force_original_aspect_ratio=decrease,"
                   f"pad={rend.width}:{rend.height}:(ow-iw)/2:(oh-ih)/2",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", rend.audio_bitrate,
            "-ar", "44100",
            "-f", "hls",
            "-hls_time", str(segment_duration),
            "-hls_playlist_type", "vod",
            "-hls_segment_filename", segment_pattern,
            playlist_path,
        ]

        try:
            run_ffmpeg(cmd, timeout=7200)
            # Count segments
            segments = [f for f in os.listdir(rend_dir) if f.endswith(".ts")]
            created_renditions.append({
                "name": rend.name,
                "resolution": f"{rend.width}x{rend.height}",
                "video_bitrate": rend.video_bitrate,
                "audio_bitrate": rend.audio_bitrate,
                "playlist": f"{rend.name}/playlist.m3u8",
                "segments": len(segments),
            })
        except RuntimeError as e:
            logger.warning("Failed to create %s rendition: %s", rend.name, e)
            continue

    if not created_renditions:
        raise RuntimeError("Failed to create any HLS renditions.")

    if on_progress:
        on_progress(85, "Writing master playlist...")

    # Write master playlist
    master_path = os.path.join(output_dir, "master.m3u8")
    lines = ["#EXTM3U", "#EXT-X-VERSION:3", ""]

    for cr in created_renditions:
        rend = ALL_RENDITIONS.get(cr["name"])
        if rend:
            bw = int(rend.video_bitrate.rstrip("kKmM")) * 1000
            lines.append(
                f"#EXT-X-STREAM-INF:BANDWIDTH={bw},"
                f"RESOLUTION={rend.width}x{rend.height},"
                f"NAME=\"{rend.name}\""
            )
            lines.append(cr["playlist"])
            lines.append("")

    with open(master_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    result = {
        "output_dir": output_dir,
        "master_playlist": master_path,
        "renditions": created_renditions,
        "segment_duration": segment_duration,
    }

    # Create zip if requested
    if include_zip:
        if on_progress:
            on_progress(90, "Creating zip archive...")

        zip_path = output_dir.rstrip("/\\") + ".zip"
        _create_zip(output_dir, zip_path)
        result["zip_path"] = zip_path
        result["zip_size_mb"] = round(os.path.getsize(zip_path) / (1024 * 1024), 2)

    if on_progress:
        on_progress(100, f"HLS package complete ({len(created_renditions)} renditions).")

    return result


# ---------------------------------------------------------------------------
# DASH Package
# ---------------------------------------------------------------------------

def create_dash_package(
    video_path: str,
    output_dir: str = "",
    renditions: Optional[List[str]] = None,
    segment_duration: int = 4,
    include_zip: bool = True,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Generate a DASH streaming package with .mpd manifest.

    Args:
        video_path: Path to the source video.
        output_dir: Output directory. Created if not existing.
        renditions: List of rendition names. Auto-selected if None.
        segment_duration: DASH segment duration in seconds.
        include_zip: Create a zip archive of the package.
        on_progress: Optional callback (percent, message).

    Returns:
        Dict with output_dir, manifest path, renditions info, zip_path.

    Raises:
        FileNotFoundError: If video_path does not exist.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input file not found: {video_path}")

    get_video_info(video_path)

    if not output_dir:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), f"{base}_dash")

    os.makedirs(output_dir, exist_ok=True)

    if renditions:
        rend_list = [ALL_RENDITIONS[r] for r in renditions if r in ALL_RENDITIONS]
        if not rend_list:
            rend_list = get_rendition_configs(source_path=video_path)
    else:
        rend_list = get_rendition_configs(source_path=video_path)

    if on_progress:
        on_progress(5, f"Creating DASH package ({len(rend_list)} renditions)...")

    segment_duration = max(2, min(10, int(segment_duration)))
    manifest_path = os.path.join(output_dir, "manifest.mpd")

    # Build a multi-output FFmpeg command for DASH
    # DASH muxer handles multi-rendition in one pass with -adaptation_sets
    cmd = [get_ffmpeg_path(), "-hide_banner", "-y", "-i", video_path]

    map_args = []
    codec_args = []

    for i, rend in enumerate(rend_list):
        # Map video stream for each rendition
        map_args.extend(["-map", "0:v:0", "-map", "0:a:0"])

        # Video settings per rendition
        i * 2  # video stream index in output
        i * 2 + 1  # audio stream index in output
        codec_args.extend([
            f"-c:v:{i}", "libx264",
            f"-b:v:{i}", rend.video_bitrate,
            f"-maxrate:v:{i}", rend.maxrate,
            f"-bufsize:v:{i}", rend.bufsize,
            f"-profile:v:{i}", rend.profile,
            f"-filter:v:{i}",
            f"scale={rend.width}:{rend.height}:force_original_aspect_ratio=decrease,"
            f"pad={rend.width}:{rend.height}:(ow-iw)/2:(oh-ih)/2",
            f"-pix_fmt:v:{i}", "yuv420p",
            f"-c:a:{i}", "aac",
            f"-b:a:{i}", rend.audio_bitrate,
            f"-ar:a:{i}", "44100",
        ])

    # Build adaptation sets string
    video_streams = ",".join(str(i * 2) for i in range(len(rend_list)))
    audio_streams = ",".join(str(i * 2 + 1) for i in range(len(rend_list)))

    cmd.extend(map_args)
    cmd.extend(codec_args)
    cmd.extend([
        "-f", "dash",
        "-seg_duration", str(segment_duration),
        "-use_template", "1",
        "-use_timeline", "1",
        "-adaptation_sets", f"id=0,streams={video_streams} id=1,streams={audio_streams}",
        manifest_path,
    ])

    if on_progress:
        on_progress(20, "Encoding DASH renditions...")

    try:
        run_ffmpeg(cmd, timeout=7200)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to create DASH package: {e}")

    if on_progress:
        on_progress(85, "DASH encoding complete.")

    # Gather rendition info
    created_renditions = []
    for rend in rend_list:
        created_renditions.append({
            "name": rend.name,
            "resolution": f"{rend.width}x{rend.height}",
            "video_bitrate": rend.video_bitrate,
            "audio_bitrate": rend.audio_bitrate,
        })

    result = {
        "output_dir": output_dir,
        "manifest": manifest_path,
        "renditions": created_renditions,
        "segment_duration": segment_duration,
    }

    if include_zip:
        if on_progress:
            on_progress(90, "Creating zip archive...")
        zip_path = output_dir.rstrip("/\\") + ".zip"
        _create_zip(output_dir, zip_path)
        result["zip_path"] = zip_path
        result["zip_size_mb"] = round(os.path.getsize(zip_path) / (1024 * 1024), 2)

    if on_progress:
        on_progress(100, f"DASH package complete ({len(created_renditions)} renditions).")

    return result


# ---------------------------------------------------------------------------
# Zip Helper
# ---------------------------------------------------------------------------

def _create_zip(source_dir: str, zip_path: str):
    """Create a zip archive from a directory."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        base_name = os.path.basename(source_dir)
        for root, dirs, files in os.walk(source_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                arcname = os.path.join(
                    base_name,
                    os.path.relpath(full_path, source_dir)
                )
                zf.write(full_path, arcname)
