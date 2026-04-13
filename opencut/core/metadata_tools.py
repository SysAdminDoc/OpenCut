"""
OpenCut Metadata Preservation & Stripping Controls v1.0.0

Read, strip, selectively preserve, and copy metadata on media files
using FFmpeg/FFprobe. Supports modes:
- strip_all:     Remove all metadata tags
- preserve_all:  Keep all metadata tags unchanged
- selective:     Keep or strip specific fields
- legal:         Strip privacy-sensitive fields (GPS, serial numbers)
                 while preserving timestamps and technical info

All functions follow the core module pattern: dict results,
optional on_progress callback, FFmpeg-based processing.
"""

import json as _json
import logging
import os
import subprocess as _sp
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg

logger = logging.getLogger("opencut")

# Metadata fields considered privacy-sensitive (stripped in "legal" mode)
_PRIVACY_FIELDS = frozenset({
    "location", "location-eng", "com.apple.quicktime.location.iso6709",
    "gps", "gps_latitude", "gps_longitude", "gps_altitude",
    "com.apple.quicktime.gps.horizontal_accuracy",
    "com.apple.quicktime.gps.speed",
    "serial_number", "serialnumber", "serial",
    "camera_serial", "bodyserialnumber",
    "make_model_serial",
    "artist", "author", "owner",
    "copyright", "rights",
    "comment", "description", "synopsis",
    "software", "handler_name", "encoder",
    "com.apple.quicktime.model",
    "com.apple.quicktime.make",
    "com.apple.quicktime.software",
    "device_manufacturer", "device_model",
})

# Fields preserved even in "legal" mode
_LEGAL_PRESERVE_FIELDS = frozenset({
    "creation_time", "date", "duration", "timecode",
    "major_brand", "minor_version", "compatible_brands",
    "codec_name", "codec_long_name", "codec_type",
    "width", "height", "display_aspect_ratio", "sample_aspect_ratio",
    "r_frame_rate", "avg_frame_rate", "bit_rate",
    "sample_rate", "channels", "channel_layout",
    "pix_fmt", "color_range", "color_space",
    "title",
})

VALID_MODES = frozenset({"strip_all", "preserve_all", "selective", "legal"})


# ---------------------------------------------------------------------------
# Metadata Reading
# ---------------------------------------------------------------------------

def get_metadata(input_path: str) -> Dict:
    """
    Read all metadata from a media file via ffprobe.

    Returns a dict with keys:
      - format_tags:  Dict of container-level metadata tags
      - streams:      List of dicts, each with stream info and tags
      - format:       Dict of format-level info (duration, size, etc.)
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        input_path,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(f"ffprobe failed: {stderr[-500:]}")

    data = _json.loads(result.stdout.decode(errors="replace"))

    format_info = data.get("format", {})
    format_tags = format_info.get("tags", {})
    streams = []
    for s in data.get("streams", []):
        streams.append({
            "index": s.get("index", 0),
            "codec_name": s.get("codec_name", ""),
            "codec_type": s.get("codec_type", ""),
            "width": s.get("width"),
            "height": s.get("height"),
            "duration": s.get("duration"),
            "bit_rate": s.get("bit_rate"),
            "sample_rate": s.get("sample_rate"),
            "channels": s.get("channels"),
            "tags": s.get("tags", {}),
        })

    return {
        "format_tags": format_tags,
        "streams": streams,
        "format": {
            "filename": format_info.get("filename", ""),
            "format_name": format_info.get("format_name", ""),
            "format_long_name": format_info.get("format_long_name", ""),
            "duration": format_info.get("duration", "0"),
            "size": format_info.get("size", "0"),
            "bit_rate": format_info.get("bit_rate", "0"),
            "nb_streams": format_info.get("nb_streams", 0),
        },
    }


# ---------------------------------------------------------------------------
# Metadata Stripping
# ---------------------------------------------------------------------------

def strip_metadata(
    input_path: str,
    output_path: Optional[str] = None,
    preserve_fields: Optional[List[str]] = None,
    strip_fields: Optional[List[str]] = None,
    mode: str = "strip_all",
    on_progress: Optional[Callable] = None,
) -> Dict:
    """
    Strip or selectively preserve metadata from a media file.

    Modes:
      - strip_all:     Remove all metadata (default)
      - preserve_all:  Copy file without touching metadata
      - selective:     Use preserve_fields/strip_fields lists
      - legal:         Strip GPS, serial numbers, device info;
                       keep timestamps and technical metadata

    Args:
        input_path:      Source media file.
        output_path:     Destination path (auto-generated if None).
        preserve_fields: Fields to keep (selective mode).
        strip_fields:    Fields to remove (selective mode).
        mode:            One of: strip_all, preserve_all, selective, legal.
        on_progress:     Optional callback(percent, message).

    Returns:
        Dict with keys: output_path, file_size, mode, stripped_count, preserved_count.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {', '.join(sorted(VALID_MODES))}")

    preserve_fields = preserve_fields or []
    strip_fields = strip_fields or []

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1] or ".mp4"
        directory = os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_stripped{ext}")

    if on_progress:
        on_progress(5, f"Starting metadata {mode}...")

    # Read existing metadata for counting
    existing_meta = get_metadata(input_path)
    all_tags = dict(existing_meta.get("format_tags", {}))
    for stream in existing_meta.get("streams", []):
        all_tags.update(stream.get("tags", {}))

    total_tags = len(all_tags)
    stripped_count = 0
    preserved_count = 0

    if mode == "preserve_all":
        # Stream copy with all metadata preserved
        if on_progress:
            on_progress(20, "Copying with all metadata preserved...")
        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            "-c", "copy",
            "-map_metadata", "0",
            output_path,
        ], timeout=7200)
        preserved_count = total_tags
        stripped_count = 0

    elif mode == "strip_all":
        # Stream copy with all metadata removed
        if on_progress:
            on_progress(20, "Stripping all metadata...")
        run_ffmpeg([
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            "-c", "copy",
            "-map_metadata", "-1",
            "-fflags", "+bitexact",
            "-flags:v", "+bitexact",
            "-flags:a", "+bitexact",
            output_path,
        ], timeout=7200)
        stripped_count = total_tags
        preserved_count = 0

    elif mode == "selective":
        if on_progress:
            on_progress(20, "Applying selective metadata filter...")

        # Determine which tags to keep vs strip
        preserve_set = {f.lower() for f in preserve_fields}
        strip_set = {f.lower() for f in strip_fields}

        # Build metadata arguments
        # Start by stripping all, then re-add what we want
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            "-c", "copy",
            "-map_metadata", "-1",
        ]

        for key, value in all_tags.items():
            key_lower = key.lower()
            keep = False
            if preserve_set and key_lower in preserve_set:
                keep = True
            elif strip_set and key_lower not in strip_set:
                keep = True
            elif not preserve_set and not strip_set:
                keep = False  # No filters = strip all

            if keep:
                cmd += ["-metadata", f"{key}={value}"]
                preserved_count += 1
            else:
                stripped_count += 1

        cmd.append(output_path)
        run_ffmpeg(cmd, timeout=7200)

    elif mode == "legal":
        if on_progress:
            on_progress(20, "Applying legal metadata filter...")

        # Strip privacy-sensitive fields, preserve everything else
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            "-c", "copy",
            "-map_metadata", "-1",
        ]

        for key, value in all_tags.items():
            key_lower = key.lower()
            if key_lower in _PRIVACY_FIELDS:
                stripped_count += 1
            else:
                cmd += ["-metadata", f"{key}={value}"]
                preserved_count += 1

        cmd.append(output_path)
        run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(90, "Metadata operation complete")

    file_size = 0
    try:
        file_size = os.path.getsize(output_path)
    except OSError:
        pass

    if on_progress:
        on_progress(100, f"Done ({stripped_count} stripped, {preserved_count} preserved)")

    return {
        "output_path": output_path,
        "file_size": file_size,
        "mode": mode,
        "stripped_count": stripped_count,
        "preserved_count": preserved_count,
    }


# ---------------------------------------------------------------------------
# Copy with Metadata Overrides
# ---------------------------------------------------------------------------

def copy_with_metadata(
    input_path: str,
    output_path: str,
    metadata_overrides: Optional[Dict[str, str]] = None,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """
    Copy a media file with custom metadata field overrides.

    Preserves all existing metadata, then applies overrides on top.
    Pass a value of "" (empty string) to remove a specific field.

    Args:
        input_path:         Source media file.
        output_path:        Destination path.
        metadata_overrides: Dict of field_name -> value to set/override.
        on_progress:        Optional callback(percent, message).

    Returns:
        Dict with keys: output_path, file_size, fields_modified.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not output_path:
        raise ValueError("output_path is required")

    metadata_overrides = metadata_overrides or {}

    if on_progress:
        on_progress(5, "Copying file with metadata overrides...")

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-c", "copy",
        "-map_metadata", "0",
    ]

    fields_modified = []
    for key, value in metadata_overrides.items():
        if not key or not isinstance(key, str):
            continue
        # Sanitize key: allow only alphanumeric, underscore, dot, dash
        safe_key = "".join(c for c in key if c.isalnum() or c in "._-")
        if not safe_key:
            continue
        cmd += ["-metadata", f"{safe_key}={value}"]
        fields_modified.append(safe_key)

    cmd.append(output_path)

    if on_progress:
        on_progress(30, f"Writing {len(fields_modified)} metadata fields...")

    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(90, "Copy complete")

    file_size = 0
    try:
        file_size = os.path.getsize(output_path)
    except OSError:
        pass

    if on_progress:
        on_progress(100, "Done")

    return {
        "output_path": output_path,
        "file_size": file_size,
        "fields_modified": fields_modified,
    }
