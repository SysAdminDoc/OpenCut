"""
Post-Production Deliverables Generator

Generates VFX sheets, ADR lists, music cue sheets, and asset lists
from Premiere Pro sequence data. Exports as CSV or plain text.
"""

import csv
import logging
import os
from typing import List

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Time formatting helpers
# ---------------------------------------------------------------------------

def _seconds_to_tc(seconds: float, fps: float = 24.0) -> str:
    """
    Convert seconds to SMPTE timecode string HH:MM:SS:FF.

    Args:
        seconds: Time in seconds.
        fps: Frames per second (default 24).

    Returns:
        Timecode string in HH:MM:SS:FF format.
    """
    seconds = max(0.0, seconds)
    fps = max(1.0, fps)
    total_frames = int(round(seconds * fps))
    fps_int = int(round(fps))
    ff = total_frames % fps_int
    total_secs = total_frames // fps_int
    ss = total_secs % 60
    total_mins = total_secs // 60
    mm = total_mins % 60
    hh = total_mins // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _seconds_to_readable(seconds: float) -> str:
    """
    Convert seconds to H:MM:SS string.

    Args:
        seconds: Time in seconds.

    Returns:
        Human-readable duration string.
    """
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# CSV writing helper
# ---------------------------------------------------------------------------

def _write_csv(output_path: str, fieldnames: List[str], rows: List[dict]) -> str:
    """Write a CSV file and return the output path."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), output_path)
    return output_path


# ---------------------------------------------------------------------------
# VFX Sheet
# ---------------------------------------------------------------------------

def generate_vfx_sheet(sequence_data: dict, output_path: str) -> dict:
    """
    Generate a VFX shot sheet CSV from sequence data.

    Columns: Shot#, ClipName, TimeIn, TimeOut, Duration, Track, Effects, Notes

    Args:
        sequence_data: Sequence dict with video_tracks, audio_tracks, markers, etc.
        output_path: Destination CSV file path.

    Returns:
        dict with "output" (str path) and "rows" (int count of data rows written).
    """
    fieldnames = ["Shot#", "ClipName", "TimeIn", "TimeOut", "Duration", "Track", "Effects", "Notes"]
    rows = []
    shot_num = 1

    for track in sequence_data.get("video_tracks", []):
        track_index = track.get("index", 0)
        for clip in track.get("clips", []):
            effects = clip.get("effects", [])
            # Only include clips that have effects or mark as potential VFX
            start = clip.get("start", 0.0)
            end = clip.get("end", 0.0)
            duration = end - start
            rows.append({
                "Shot#": shot_num,
                "ClipName": clip.get("name", ""),
                "TimeIn": _seconds_to_tc(start),
                "TimeOut": _seconds_to_tc(end),
                "Duration": _seconds_to_tc(duration),
                "Track": f"V{track_index + 1}",
                "Effects": "; ".join(effects) if effects else "",
                "Notes": "",
            })
            shot_num += 1

    if not rows:
        logger.warning("No video clips found in sequence data for VFX sheet")

    row_count = len(rows)
    return {"output": _write_csv(output_path, fieldnames, rows), "rows": row_count}


# ---------------------------------------------------------------------------
# ADR List
# ---------------------------------------------------------------------------

def generate_adr_list(sequence_data: dict, output_path: str) -> dict:
    """
    Generate an ADR (Automated Dialogue Replacement) list CSV.

    Columns: Line#, Character, Timecode, Duration, DialogueLine, Scene, Notes

    Audio clips on track 0 (dialogue track) are treated as ADR candidates.
    Clip names are used as the dialogue line text. Character is inferred from
    clip name if it contains a dash separator (e.g. "John - Hello there").

    Args:
        sequence_data: Sequence dict.
        output_path: Destination CSV file path.

    Returns:
        dict with "output" (str path) and "rows" (int count of data rows written).
    """
    fieldnames = ["Line#", "Character", "Timecode", "Duration", "DialogueLine", "Scene", "Notes"]
    rows = []
    line_num = 1

    for track in sequence_data.get("audio_tracks", []):
        track_index = track.get("index", 0)
        # Primary dialogue typically on track 0 or 1
        if track_index > 1:
            continue
        for clip in track.get("clips", []):
            name = clip.get("name", "")
            start = clip.get("start", 0.0)
            end = clip.get("end", 0.0)
            duration = end - start

            # Try to split character from dialogue: "CharName - line text"
            character = ""
            dialogue_line = name
            if " - " in name:
                parts = name.split(" - ", 1)
                character = parts[0].strip()
                dialogue_line = parts[1].strip()

            rows.append({
                "Line#": line_num,
                "Character": character,
                "Timecode": _seconds_to_tc(start),
                "Duration": _seconds_to_tc(duration),
                "DialogueLine": dialogue_line,
                "Scene": "",
                "Notes": "",
            })
            line_num += 1

    if not rows:
        logger.warning("No dialogue clips found for ADR list")

    row_count = len(rows)
    return {"output": _write_csv(output_path, fieldnames, rows), "rows": row_count}


# ---------------------------------------------------------------------------
# Music Cue Sheet
# ---------------------------------------------------------------------------

def _looks_like_music(clip_name: str, video_clip_names: set) -> bool:
    """
    Heuristic: a clip is music if its name doesn't match any video clip name
    and doesn't contain common dialogue keywords.
    """
    name_lower = clip_name.lower()
    dialogue_keywords = {"take", "cam", "camera", "interview", "sync", "dialogue", "dialog", "mic", "vo", "voiceover"}
    for kw in dialogue_keywords:
        if kw in name_lower:
            return False
    # If it matches a video clip name it's likely sync audio
    if clip_name in video_clip_names:
        return False
    return True


def generate_music_cue_sheet(sequence_data: dict, output_path: str) -> dict:
    """
    Generate a music cue sheet CSV.

    Columns: Cue#, Title, Composer, Start, End, Duration, Usage, Notes

    Detects music clips by checking audio tracks 2+ and filtering out clips
    that share names with video clips (likely sync/dialogue audio).

    Args:
        sequence_data: Sequence dict.
        output_path: Destination CSV file path.

    Returns:
        dict with "output" (str path) and "rows" (int count of data rows written).
    """
    fieldnames = ["Cue#", "Title", "Composer", "Start", "End", "Duration", "Usage", "Notes"]
    rows = []
    cue_num = 1

    # Collect video clip names for filtering
    video_clip_names = set()
    for track in sequence_data.get("video_tracks", []):
        for clip in track.get("clips", []):
            video_clip_names.add(clip.get("name", ""))

    for track in sequence_data.get("audio_tracks", []):
        track_index = track.get("index", 0)
        # Music is typically on track 2+ (0-indexed)
        if track_index < 2:
            continue
        for clip in track.get("clips", []):
            clip_name = clip.get("name", "")
            if not _looks_like_music(clip_name, video_clip_names):
                continue
            start = clip.get("start", 0.0)
            end = clip.get("end", 0.0)
            duration = end - start
            rows.append({
                "Cue#": cue_num,
                "Title": clip_name,
                "Composer": "",
                "Start": _seconds_to_tc(start),
                "End": _seconds_to_tc(end),
                "Duration": _seconds_to_tc(duration),
                "Usage": "Background",
                "Notes": "",
            })
            cue_num += 1

    if not rows:
        logger.info("No music cues detected in sequence (tracks 2+)")

    row_count = len(rows)
    return {"output": _write_csv(output_path, fieldnames, rows), "rows": row_count}


# ---------------------------------------------------------------------------
# Asset List
# ---------------------------------------------------------------------------

def generate_asset_list(sequence_data: dict, output_path: str) -> dict:
    """
    Generate a comprehensive media asset list CSV.

    Columns: #, AssetName, FilePath, Type, Duration, FirstUsed, LastUsed, UseCount

    Args:
        sequence_data: Sequence dict.
        output_path: Destination CSV file path.

    Returns:
        dict with "output" (str path) and "rows" (int count of data rows written).
    """
    fieldnames = ["#", "AssetName", "FilePath", "Type", "Duration", "FirstUsed", "LastUsed", "UseCount"]

    # Collect all clips across all tracks
    asset_map: dict = {}  # filepath -> aggregated stats

    def _register(clip: dict, asset_type: str):
        path = clip.get("path", "") or clip.get("name", "")
        name = clip.get("name", "") or os.path.basename(path)
        start = clip.get("start", 0.0)
        end = clip.get("end", 0.0)
        duration = end - start

        if path not in asset_map:
            asset_map[path] = {
                "name": name,
                "path": path,
                "type": asset_type,
                "duration": duration,
                "first_used": start,
                "last_used": end,
                "use_count": 1,
            }
        else:
            entry = asset_map[path]
            entry["first_used"] = min(entry["first_used"], start)
            entry["last_used"] = max(entry["last_used"], end)
            entry["use_count"] += 1

    for track in sequence_data.get("video_tracks", []):
        for clip in track.get("clips", []):
            _register(clip, "Video")

    for track in sequence_data.get("audio_tracks", []):
        for clip in track.get("clips", []):
            _register(clip, "Audio")

    rows = []
    for asset_num, entry in enumerate(
        sorted(asset_map.values(), key=lambda e: e["first_used"]), start=1
    ):
        rows.append({
            "#": asset_num,
            "AssetName": entry["name"],
            "FilePath": entry["path"],
            "Type": entry["type"],
            "Duration": _seconds_to_readable(entry["duration"]),
            "FirstUsed": _seconds_to_tc(entry["first_used"]),
            "LastUsed": _seconds_to_tc(entry["last_used"]),
            "UseCount": entry["use_count"],
        })

    row_count = len(rows)
    return {"output": _write_csv(output_path, fieldnames, rows), "rows": row_count}
