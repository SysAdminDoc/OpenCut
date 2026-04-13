"""
OpenCut Batch Metadata Editor Module v1.0.0

Read/edit metadata across multiple files in spreadsheet style.
Uses ffprobe for reading and ffmpeg for writing metadata tags.
"""

import csv
import json
import logging
import os
import subprocess
import time
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Standard metadata fields
# ---------------------------------------------------------------------------

_STANDARD_FIELDS = [
    "title", "artist", "album", "album_artist", "composer",
    "genre", "date", "track", "comment", "copyright",
    "description", "synopsis", "show", "episode_id",
    "network", "encoder",
]


# ---------------------------------------------------------------------------
# Read metadata
# ---------------------------------------------------------------------------

def read_batch_metadata(file_paths: List[str]) -> List[dict]:
    """Read metadata from multiple media files.

    Args:
        file_paths: List of media file paths.

    Returns:
        List of dicts, each with ``file_path``, ``filename``,
        ``metadata`` (tag dict), ``format`` info, and ``error`` if any.
    """
    if not file_paths:
        raise ValueError("No file paths provided")

    results = []

    for fpath in file_paths:
        entry = {
            "file_path": fpath,
            "filename": os.path.basename(fpath),
            "metadata": {},
            "format": {},
            "error": "",
        }

        if not os.path.isfile(fpath):
            entry["error"] = "File not found"
            results.append(entry)
            continue

        try:
            cmd = [
                get_ffprobe_path(), "-v", "quiet",
                "-print_format", "json",
                "-show_format", fpath,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                entry["error"] = f"ffprobe failed (rc={result.returncode})"
                results.append(entry)
                continue

            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            tags = fmt.get("tags", {})

            # Normalize tag keys to lowercase
            entry["metadata"] = {k.lower(): v for k, v in tags.items()}
            entry["format"] = {
                "format_name": fmt.get("format_name", ""),
                "duration": float(fmt.get("duration", 0)),
                "size": int(fmt.get("size", 0)),
                "bit_rate": int(fmt.get("bit_rate", 0)),
            }

        except Exception as e:
            entry["error"] = str(e)
            logger.error("Failed to read metadata for %s: %s", fpath, e)

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Write metadata
# ---------------------------------------------------------------------------

def write_batch_metadata(
    file_paths: List[str],
    metadata_updates: Dict[str, Dict[str, str]],
    on_progress: Optional[Callable] = None,
) -> List[dict]:
    """Write metadata to multiple media files.

    Uses ffmpeg to copy the file with updated metadata tags.
    The original is replaced with the tagged version.

    Args:
        file_paths: List of media file paths.
        metadata_updates: Dict mapping file path -> {tag: value} updates.
            A special key ``"*"`` applies to all files.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        List of dicts with ``file_path``, ``status``, ``updated_tags``, ``error``.
    """
    if not file_paths:
        raise ValueError("No file paths provided")
    if not metadata_updates:
        raise ValueError("No metadata updates provided")

    results = []
    global_updates = metadata_updates.get("*", {})

    if on_progress:
        on_progress(5, f"Writing metadata to {len(file_paths)} files...")

    for i, fpath in enumerate(file_paths):
        entry = {
            "file_path": fpath,
            "status": "pending",
            "updated_tags": {},
            "error": "",
        }

        if not os.path.isfile(fpath):
            entry["status"] = "failed"
            entry["error"] = "File not found"
            results.append(entry)
            continue

        # Merge global and per-file updates
        tags = dict(global_updates)
        per_file = metadata_updates.get(fpath, {})
        tags.update(per_file)

        if not tags:
            entry["status"] = "skipped"
            results.append(entry)
            continue

        try:
            # Build ffmpeg command: copy streams, update metadata
            base, ext = os.path.splitext(fpath)
            temp_path = f"{base}_meta_tmp{ext}"

            cmd = [
                get_ffmpeg_path(), "-hide_banner", "-y",
                "-i", fpath,
                "-c", "copy",
            ]

            for key, value in tags.items():
                cmd.extend(["-metadata", f"{key}={value}"])

            cmd.append(temp_path)

            run_ffmpeg(cmd)

            # Replace original with tagged version
            os.replace(temp_path, fpath)

            entry["status"] = "updated"
            entry["updated_tags"] = tags

        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
            logger.error("Failed to write metadata for %s: %s", fpath, e)
            # Clean up temp file on failure
            temp_path = f"{base}_meta_tmp{ext}"
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        results.append(entry)

        if on_progress:
            pct = min(int(((i + 1) / len(file_paths)) * 90) + 5, 95)
            on_progress(pct, f"Updated {i + 1}/{len(file_paths)}")

    if on_progress:
        updated = sum(1 for r in results if r["status"] == "updated")
        on_progress(100, f"Metadata update complete: {updated}/{len(file_paths)}")

    return results


# ---------------------------------------------------------------------------
# Apply metadata template
# ---------------------------------------------------------------------------

def apply_metadata_template(
    file_paths: List[str],
    template: Dict[str, str],
    on_progress: Optional[Callable] = None,
) -> List[dict]:
    """Apply a metadata template to multiple files.

    A template is a dict of tag -> value pairs applied uniformly.
    Supports placeholders: {filename}, {index}, {date}.

    Args:
        file_paths: List of media file paths.
        template: Dict of tag -> value pairs.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        List of result dicts (same format as write_batch_metadata).
    """
    if not file_paths:
        raise ValueError("No file paths provided")
    if not template:
        raise ValueError("No template provided")

    # Build per-file metadata updates with placeholder expansion
    updates = {}
    for i, fpath in enumerate(file_paths):
        basename = os.path.splitext(os.path.basename(fpath))[0]
        file_tags = {}
        for key, value in template.items():
            expanded = value
            expanded = expanded.replace("{filename}", basename)
            expanded = expanded.replace("{index}", str(i + 1))
            expanded = expanded.replace("{date}", time.strftime("%Y-%m-%d"))
            file_tags[key] = expanded
        updates[fpath] = file_tags

    return write_batch_metadata(file_paths, updates, on_progress=on_progress)


# ---------------------------------------------------------------------------
# Export metadata to CSV
# ---------------------------------------------------------------------------

def export_metadata_csv(
    file_paths: List[str],
    output_path: str,
) -> str:
    """Export metadata from multiple files to a CSV spreadsheet.

    Args:
        file_paths: List of media file paths.
        output_path: Path for the output CSV file.

    Returns:
        Path to the written CSV file.
    """
    if not file_paths:
        raise ValueError("No file paths provided")

    # Read all metadata first
    metadata_list = read_batch_metadata(file_paths)

    # Collect all unique tag keys
    all_keys = set()
    for entry in metadata_list:
        all_keys.update(entry["metadata"].keys())

    # Sort keys: standard fields first, then alphabetical
    ordered_keys = []
    for sf in _STANDARD_FIELDS:
        if sf in all_keys:
            ordered_keys.append(sf)
            all_keys.discard(sf)
    ordered_keys.extend(sorted(all_keys))

    # Write CSV
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not output_path.endswith(".csv"):
        output_path = os.path.splitext(output_path)[0] + ".csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["file_path", "filename"] + ordered_keys
        writer.writerow(header)

        for entry in metadata_list:
            row = [entry["file_path"], entry["filename"]]
            for key in ordered_keys:
                row.append(entry["metadata"].get(key, ""))
            writer.writerow(row)

    return output_path
