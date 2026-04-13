"""
OpenCut Soft Subtitle Embedding

Embed SRT/ASS/VTT subtitle files as soft (selectable) subtitle tracks
inside MP4 or MKV containers.  Video and audio streams are copied without
re-encoding -- only the subtitle streams are muxed in.

Also provides subtitle track listing via ffprobe.

Uses FFmpeg only -- no additional dependencies required.
"""

import json
import logging
import os
import subprocess
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg

logger = logging.getLogger("opencut")

# Subtitle codec selection per container
_CODEC_MAP = {
    "mp4": "mov_text",
    "m4v": "mov_text",
    "mkv": {
        ".srt": "srt",
        ".ass": "ass",
        ".ssa": "ass",
        ".vtt": "webvtt",
    },
    "webm": "webvtt",
}


def _subtitle_codec(container: str, sub_path: str) -> str:
    """Choose the correct subtitle codec for the container and file type."""
    container = container.lower().lstrip(".")
    mapping = _CODEC_MAP.get(container, "mov_text")
    if isinstance(mapping, dict):
        ext = os.path.splitext(sub_path)[1].lower()
        return mapping.get(ext, "srt")
    return mapping


def embed_subtitles(
    input_path: str,
    subtitle_paths: List[str],
    languages: Optional[List[str]] = None,
    output_path_override: Optional[str] = None,
    container: str = "mp4",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Embed one or more subtitle files as soft subtitle tracks.

    Args:
        input_path: Source video file.
        subtitle_paths: List of SRT / ASS / VTT file paths.
        languages: ISO 639 language codes matching subtitle_paths
                   (e.g. ["eng", "spa"]).  Defaults to "und" (undefined).
        output_path_override: Explicit output path; auto-generated if None.
        container: Target container format ("mp4" or "mkv").
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with "output_path" and "tracks_added".
    """
    if not subtitle_paths:
        raise ValueError("At least one subtitle file is required")

    for sp in subtitle_paths:
        if not os.path.isfile(sp):
            raise FileNotFoundError(f"Subtitle file not found: {sp}")

    if languages and len(languages) != len(subtitle_paths):
        raise ValueError(
            f"languages list length ({len(languages)}) must match "
            f"subtitle_paths length ({len(subtitle_paths)})"
        )

    if on_progress:
        on_progress(5, "Preparing subtitle embedding...")

    # Determine output container extension
    container = container.lower().lstrip(".")
    if container not in ("mp4", "mkv", "m4v", "webm"):
        container = "mp4"
    ext = f".{container}"

    # Build output path
    if output_path_override:
        out = output_path_override
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        out = os.path.join(
            os.path.dirname(input_path),
            f"{base}_subtitled{ext}",
        )

    # Build FFmpeg command
    cmd = [get_ffmpeg_path(), "-hide_banner", "-y"]

    # Input 0: video file
    cmd.extend(["-i", input_path])

    # Inputs 1..N: subtitle files
    for sp in subtitle_paths:
        cmd.extend(["-i", sp])

    # Map all video and audio from input 0
    cmd.extend(["-map", "0:v", "-map", "0:a?"])

    # Map subtitle stream from each subtitle input
    for idx in range(len(subtitle_paths)):
        cmd.extend(["-map", f"{idx + 1}:0"])

    # Stream copy video and audio
    cmd.extend(["-c:v", "copy", "-c:a", "copy"])

    # Subtitle codec
    codec = _subtitle_codec(container, subtitle_paths[0])
    cmd.extend(["-c:s", codec])

    # Language metadata for each subtitle stream
    langs = languages or ["und"] * len(subtitle_paths)
    for idx, lang in enumerate(langs):
        cmd.extend([f"-metadata:s:s:{idx}", f"language={lang}"])

    if on_progress:
        on_progress(20, "Muxing subtitles...")

    cmd.append(out)
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Subtitles embedded")

    return {
        "output_path": out,
        "tracks_added": len(subtitle_paths),
        "container": container,
        "codec": codec,
    }


def list_subtitle_tracks(input_path: str) -> List[Dict]:
    """
    List subtitle tracks present in a media file.

    Returns:
        List of dicts with keys: index, codec, language, title.
    """
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-select_streams", "s",
        "-show_entries", "stream=index,codec_name",
        "-show_entries", "stream_tags=language,title",
        "-of", "json",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        logger.warning("ffprobe subtitle probe failed for %s", input_path)
        return []

    try:
        data = json.loads(result.stdout.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    tracks = []
    for stream in data.get("streams", []):
        tags = stream.get("tags", {})
        tracks.append({
            "index": stream.get("index", 0),
            "codec": stream.get("codec_name", "unknown"),
            "language": tags.get("language", "und"),
            "title": tags.get("title", ""),
        })
    return tracks
