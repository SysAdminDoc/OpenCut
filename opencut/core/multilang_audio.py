"""
OpenCut Multi-Language Audio Track Management

Mux multiple audio streams with language metadata via FFmpeg
``-map`` + ``-metadata:s:a:N language=XXX``.

Features:
- Add/remove/label audio tracks
- Export multi-track files or per-language files
- List existing audio tracks with metadata
- Reorder audio tracks
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    get_ffmpeg_path,
    get_ffprobe_path,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ISO 639-2/B language codes for FFmpeg metadata
LANGUAGE_CODES = {
    "en": "eng", "es": "spa", "fr": "fre", "de": "ger", "it": "ita",
    "pt": "por", "ru": "rus", "zh": "chi", "ja": "jpn", "ko": "kor",
    "ar": "ara", "hi": "hin", "tr": "tur", "vi": "vie", "th": "tha",
    "pl": "pol", "nl": "dut", "sv": "swe", "da": "dan", "no": "nor",
    "fi": "fin", "cs": "cze", "hu": "hun", "ro": "rum", "el": "gre",
    "he": "heb", "fa": "per", "uk": "ukr", "bg": "bul", "hr": "hrv",
    "sr": "srp", "sk": "slo", "sl": "slv", "ca": "cat", "gl": "glg",
}


@dataclass
class AudioTrackInfo:
    """Information about a single audio track."""
    index: int = 0
    stream_index: int = 0
    language: str = ""
    language_code: str = ""
    codec: str = ""
    channels: int = 0
    sample_rate: int = 0
    bitrate: int = 0
    title: str = ""
    is_default: bool = False


@dataclass
class MultiLangResult:
    """Result of multi-language audio operations."""
    output_path: str = ""
    tracks: List[Dict] = field(default_factory=list)
    total_tracks: int = 0
    operation: str = ""


def list_audio_tracks(input_path: str) -> dict:
    """
    List all audio tracks in a media file with their metadata.

    Args:
        input_path: Path to the media file.

    Returns:
        dict with list of audio tracks and their properties.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-show_streams", "-select_streams", "a",
        "-show_entries",
        "stream=index,codec_name,channels,sample_rate,bit_rate,"
        "tags",
        "-of", "json",
        input_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(result.stdout)
    except Exception as exc:
        raise RuntimeError(f"Failed to probe audio tracks: {exc}")

    tracks = []
    for i, stream in enumerate(data.get("streams", [])):
        tags = stream.get("tags", {})
        lang = tags.get("language", "und")
        tracks.append({
            "index": i,
            "stream_index": stream.get("index", 0),
            "language": lang,
            "codec": stream.get("codec_name", "unknown"),
            "channels": int(stream.get("channels", 0)),
            "sample_rate": int(stream.get("sample_rate", 0)),
            "bitrate": int(stream.get("bit_rate", 0) or 0),
            "title": tags.get("title", ""),
            "is_default": i == 0,
        })

    return {
        "input_path": input_path,
        "total_tracks": len(tracks),
        "tracks": tracks,
    }


def add_audio_tracks(
    video_path: str,
    audio_tracks: List[Dict],
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Add audio tracks to a video file with language metadata.

    Args:
        video_path: Source video file.
        audio_tracks: List of dicts with 'path', 'language', optional 'title'.
        output_dir: Output directory.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path and track listing.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not audio_tracks:
        raise ValueError("No audio tracks provided")

    if on_progress:
        on_progress(10, f"Adding {len(audio_tracks)} audio tracks...")

    cmd = [get_ffmpeg_path(), "-hide_banner", "-y"]

    # Input: video file
    cmd.extend(["-i", video_path])

    # Input: each audio file
    for track in audio_tracks:
        audio_path = track.get("path", "").strip()
        if not audio_path or not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio track not found: {audio_path}")
        cmd.extend(["-i", audio_path])

    # Map video stream
    cmd.extend(["-map", "0:v:0"])

    # Map existing audio from video (if any)
    cmd.extend(["-map", "0:a?"])

    # Map each new audio input
    # Probe existing tracks to get starting index
    existing = list_audio_tracks(video_path)
    existing_count = existing["total_tracks"]

    for i in range(len(audio_tracks)):
        cmd.extend(["-map", f"{i + 1}:a:0"])

    # Copy video codec
    cmd.extend(["-c:v", "copy"])

    # Encode audio
    cmd.extend(["-c:a", "aac", "-b:a", "192k"])

    # Set language metadata for new tracks
    for i, track in enumerate(audio_tracks):
        track_idx = existing_count + i
        lang = track.get("language", "und").strip().lower()
        lang_code = LANGUAGE_CODES.get(lang, lang)

        cmd.extend([f"-metadata:s:a:{track_idx}", f"language={lang_code}"])

        title = track.get("title", "").strip()
        if title:
            cmd.extend([f"-metadata:s:a:{track_idx}", f"title={title}"])

    # Faststart for MP4
    cmd.extend(["-movflags", "+faststart"])

    out_dir = output_dir or os.path.dirname(video_path)
    out_file = output_path(video_path, "multilang", out_dir)
    cmd.append(out_file)

    if on_progress:
        on_progress(30, "Muxing audio tracks...")

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(90, "Verifying output...")

    # Verify output tracks
    output_tracks = list_audio_tracks(out_file)

    if on_progress:
        on_progress(100, f"Added {len(audio_tracks)} audio tracks")

    return {
        "output_path": out_file,
        "operation": "add_tracks",
        "tracks_added": len(audio_tracks),
        "total_tracks": output_tracks["total_tracks"],
        "tracks": output_tracks["tracks"],
    }


def remove_audio_tracks(
    input_path: str,
    track_indices: List[int],
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Remove specific audio tracks by index.

    Args:
        input_path: Source media file.
        track_indices: List of audio track indices to remove (0-based).
        output_dir: Output directory.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path and remaining tracks.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not track_indices:
        raise ValueError("No track indices specified for removal")

    existing = list_audio_tracks(input_path)
    total = existing["total_tracks"]

    # Validate indices
    remove_set = set(track_indices)
    keep_indices = [i for i in range(total) if i not in remove_set]

    if not keep_indices:
        raise ValueError("Cannot remove all audio tracks")

    if on_progress:
        on_progress(10, f"Removing {len(remove_set)} tracks...")

    cmd = [get_ffmpeg_path(), "-hide_banner", "-y", "-i", input_path]

    # Map video
    cmd.extend(["-map", "0:v:0?"])

    # Map only kept audio tracks
    for idx in keep_indices:
        cmd.extend(["-map", f"0:a:{idx}"])

    # Copy all codecs
    cmd.extend(["-c", "copy"])
    cmd.extend(["-movflags", "+faststart"])

    out_dir = output_dir or os.path.dirname(input_path)
    out_file = output_path(input_path, "tracks_removed", out_dir)
    cmd.append(out_file)

    run_ffmpeg(cmd)

    output_tracks = list_audio_tracks(out_file)

    if on_progress:
        on_progress(100, f"Removed {len(remove_set)} tracks")

    return {
        "output_path": out_file,
        "operation": "remove_tracks",
        "tracks_removed": len(remove_set),
        "total_tracks": output_tracks["total_tracks"],
        "tracks": output_tracks["tracks"],
    }


def label_audio_tracks(
    input_path: str,
    labels: List[Dict],
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Update language and title metadata for audio tracks.

    Args:
        input_path: Source media file.
        labels: List of dicts with 'index', 'language', optional 'title'.
        output_dir: Output directory.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path and updated tracks.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not labels:
        raise ValueError("No labels provided")

    if on_progress:
        on_progress(10, "Labeling audio tracks...")

    cmd = [get_ffmpeg_path(), "-hide_banner", "-y", "-i", input_path]

    # Copy everything
    cmd.extend(["-map", "0", "-c", "copy"])

    for label in labels:
        idx = int(label.get("index", 0))
        lang = label.get("language", "").strip().lower()
        title = label.get("title", "").strip()

        if lang:
            lang_code = LANGUAGE_CODES.get(lang, lang)
            cmd.extend([f"-metadata:s:a:{idx}", f"language={lang_code}"])
        if title:
            cmd.extend([f"-metadata:s:a:{idx}", f"title={title}"])

    cmd.extend(["-movflags", "+faststart"])

    out_dir = output_dir or os.path.dirname(input_path)
    out_file = output_path(input_path, "labeled", out_dir)
    cmd.append(out_file)

    run_ffmpeg(cmd)

    output_tracks = list_audio_tracks(out_file)

    if on_progress:
        on_progress(100, "Track labels updated")

    return {
        "output_path": out_file,
        "operation": "label_tracks",
        "labels_applied": len(labels),
        "total_tracks": output_tracks["total_tracks"],
        "tracks": output_tracks["tracks"],
    }


def export_single_language(
    input_path: str,
    track_index: int = 0,
    output_dir: str = "",
    audio_only: bool = False,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Export a single audio track (with or without video).

    Args:
        input_path: Source media file.
        track_index: Audio track index to export (0-based).
        output_dir: Output directory.
        audio_only: If True, export audio only (no video).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path and track info.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    existing = list_audio_tracks(input_path)
    if track_index >= existing["total_tracks"]:
        raise ValueError(
            f"Track index {track_index} out of range "
            f"(file has {existing['total_tracks']} audio tracks)"
        )

    track_info = existing["tracks"][track_index]
    lang = track_info.get("language", "und")

    if on_progress:
        on_progress(10, f"Exporting track {track_index} ({lang})...")

    cmd = [get_ffmpeg_path(), "-hide_banner", "-y", "-i", input_path]

    if audio_only:
        cmd.extend(["-map", f"0:a:{track_index}"])
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        ext = ".m4a"
    else:
        cmd.extend(["-map", "0:v:0?", "-map", f"0:a:{track_index}"])
        cmd.extend(["-c", "copy"])
        cmd.extend(["-movflags", "+faststart"])
        ext = os.path.splitext(input_path)[1] or ".mp4"

    out_dir = output_dir or os.path.dirname(input_path)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_file = os.path.join(out_dir, f"{base}_{lang}{ext}")

    cmd.append(out_file)

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, f"Exported track {track_index}")

    return {
        "output_path": out_file,
        "operation": "export_single",
        "track_index": track_index,
        "language": lang,
        "audio_only": audio_only,
    }


def manage_tracks(
    input_path: str,
    operation: str,
    audio_tracks: Optional[List[Dict]] = None,
    track_indices: Optional[List[int]] = None,
    labels: Optional[List[Dict]] = None,
    track_index: int = 0,
    audio_only: bool = False,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Unified track management endpoint.

    Args:
        input_path: Source media file.
        operation: One of 'list', 'add', 'remove', 'label', 'export'.
        audio_tracks: For 'add' - list of tracks to add.
        track_indices: For 'remove' - indices to remove.
        labels: For 'label' - metadata updates.
        track_index: For 'export' - track to export.
        audio_only: For 'export' - audio-only export.
        output_dir: Output directory.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with operation results.
    """
    operation = operation.strip().lower()

    if operation == "list":
        return list_audio_tracks(input_path)
    elif operation == "add":
        return add_audio_tracks(input_path, audio_tracks or [], output_dir, on_progress)
    elif operation == "remove":
        return remove_audio_tracks(input_path, track_indices or [], output_dir, on_progress)
    elif operation == "label":
        return label_audio_tracks(input_path, labels or [], output_dir, on_progress)
    elif operation == "export":
        return export_single_language(
            input_path, track_index, output_dir, audio_only, on_progress
        )
    else:
        raise ValueError(
            f"Unknown operation: {operation}. "
            f"Supported: list, add, remove, label, export"
        )
