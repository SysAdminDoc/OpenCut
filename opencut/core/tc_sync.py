"""
OpenCut Timecode-Based Sync Module (44.3)

Synchronize multiple video/audio sources by matching embedded timecodes
for frame-accurate multi-camera alignment and timeline generation.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    get_video_info,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SyncSource:
    """A media source with timecode information."""
    filepath: str = ""
    label: str = ""
    start_tc: str = ""
    end_tc: str = ""
    fps: float = 0.0
    start_frame: int = 0
    end_frame: int = 0
    duration: float = 0.0
    tc_source: str = ""  # "ltc", "vitc", "embedded", "filename"


@dataclass
class SyncResult:
    """Result of timecode-based synchronization."""
    synced_sources: List[Dict] = field(default_factory=list)
    common_range: Dict = field(default_factory=dict)
    offsets: Dict = field(default_factory=dict)
    timeline_path: str = ""
    total_sources: int = 0
    synced_count: int = 0


# ---------------------------------------------------------------------------
# Timecode utilities
# ---------------------------------------------------------------------------

def _tc_to_frames(tc: str, fps: float) -> int:
    """Convert SMPTE timecode string to frame number."""
    if not tc:
        return 0
    parts = tc.replace(";", ":").split(":")
    if len(parts) != 4:
        return 0
    try:
        hh, mm, ss, ff = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return 0
    fps_int = round(fps)
    return hh * fps_int * 3600 + mm * fps_int * 60 + ss * fps_int + ff


def _frames_to_tc(frame_num: int, fps: float) -> str:
    """Convert frame number to SMPTE timecode string."""
    if fps <= 0:
        return "00:00:00:00"
    fps_int = round(fps)
    ff = frame_num % fps_int
    ss = (frame_num // fps_int) % 60
    mm = (frame_num // (fps_int * 60)) % 60
    hh = frame_num // (fps_int * 3600)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _frames_to_seconds(frames: int, fps: float) -> float:
    """Convert frame count to seconds."""
    return frames / max(fps, 0.001)


def _seconds_to_tc(seconds: float, fps: float) -> str:
    """Convert seconds to timecode string."""
    frame_num = int(seconds * fps)
    return _frames_to_tc(frame_num, fps)


# ---------------------------------------------------------------------------
# Source timecode extraction
# ---------------------------------------------------------------------------

def _extract_source_timecodes(
    filepath: str,
    fps: float = 0.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Extract timecode information from a media source.

    Tries embedded metadata timecodes first, then falls back to
    LTC/VITC extraction.

    Args:
        filepath: Path to media file.
        fps: Expected fps (auto-detect if 0).
        on_progress: Optional callback.

    Returns:
        Dict with start_tc, end_tc, fps, source info.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")

    info = get_video_info(filepath)
    if fps <= 0:
        fps = info.get("fps", 25.0) or 25.0
    duration = info.get("duration", 0.0) or 0.0

    # Try embedded timecode from ffprobe metadata
    try:
        import subprocess

        from opencut.helpers import get_ffprobe_path
        ffprobe = get_ffprobe_path()
        cmd = [
            ffprobe, "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            filepath,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            probe = json.loads(result.stdout)

            # Check format tags for timecode
            tags = probe.get("format", {}).get("tags", {})
            tc = tags.get("timecode", "") or tags.get("TIMECODE", "")

            # Check stream tags
            if not tc:
                for stream in probe.get("streams", []):
                    stags = stream.get("tags", {})
                    tc = stags.get("timecode", "") or stags.get("TIMECODE", "")
                    if tc:
                        break

            # Check for timecode stream
            if not tc:
                for stream in probe.get("streams", []):
                    if stream.get("codec_type") == "data" and "timecode" in str(stream.get("codec_tag_string", "")).lower():
                        tc = stream.get("tags", {}).get("timecode", "")
                        if tc:
                            break

            if tc and ":" in tc:
                start_frame = _tc_to_frames(tc, fps)
                total_frames = int(duration * fps)
                end_frame = start_frame + total_frames
                return {
                    "start_tc": tc,
                    "end_tc": _frames_to_tc(end_frame, fps),
                    "fps": fps,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration": duration,
                    "tc_source": "embedded",
                }
    except Exception as e:
        logger.debug("Embedded timecode extraction failed: %s", e)

    # Try LTC extraction
    try:
        from opencut.core.ltc_vitc import extract_ltc
        ltc_result = extract_ltc(filepath, fps=fps)
        if ltc_result.get("total_frames", 0) > 0:
            return {
                "start_tc": ltc_result["start_tc"],
                "end_tc": ltc_result["end_tc"],
                "fps": fps,
                "start_frame": _tc_to_frames(ltc_result["start_tc"], fps),
                "end_frame": _tc_to_frames(ltc_result["end_tc"], fps),
                "duration": duration,
                "tc_source": "ltc",
            }
    except Exception as e:
        logger.debug("LTC extraction failed for %s: %s", filepath, e)

    # Try VITC extraction
    try:
        from opencut.core.ltc_vitc import extract_vitc
        vitc_result = extract_vitc(filepath, fps=fps)
        if vitc_result.get("total_frames", 0) > 0:
            return {
                "start_tc": vitc_result["start_tc"],
                "end_tc": vitc_result["end_tc"],
                "fps": fps,
                "start_frame": _tc_to_frames(vitc_result["start_tc"], fps),
                "end_frame": _tc_to_frames(vitc_result["end_tc"], fps),
                "duration": duration,
                "tc_source": "vitc",
            }
    except Exception as e:
        logger.debug("VITC extraction failed for %s: %s", filepath, e)

    # Fallback: assume timecode starts at 00:00:00:00
    total_frames = int(duration * fps)
    return {
        "start_tc": "00:00:00:00",
        "end_tc": _frames_to_tc(total_frames, fps),
        "fps": fps,
        "start_frame": 0,
        "end_frame": total_frames,
        "duration": duration,
        "tc_source": "assumed",
    }


# ---------------------------------------------------------------------------
# Common timecode range
# ---------------------------------------------------------------------------

def find_common_timecode_range(
    timecodes: List[dict],
) -> dict:
    """Find the overlapping timecode range across multiple sources.

    Args:
        timecodes: List of dicts with start_frame, end_frame, fps.

    Returns:
        Dict with common start_frame, end_frame, start_tc, end_tc,
        duration_frames, duration_seconds.
    """
    if not timecodes:
        return {
            "start_frame": 0,
            "end_frame": 0,
            "start_tc": "00:00:00:00",
            "end_tc": "00:00:00:00",
            "duration_frames": 0,
            "duration_seconds": 0.0,
            "valid": False,
        }

    fps = timecodes[0].get("fps", 25.0)

    # Find the latest start and earliest end
    max_start = max(tc.get("start_frame", 0) for tc in timecodes)
    min_end = min(tc.get("end_frame", 0) for tc in timecodes)

    if min_end <= max_start:
        return {
            "start_frame": max_start,
            "end_frame": max_start,
            "start_tc": _frames_to_tc(max_start, fps),
            "end_tc": _frames_to_tc(max_start, fps),
            "duration_frames": 0,
            "duration_seconds": 0.0,
            "valid": False,
        }

    duration_frames = min_end - max_start
    duration_seconds = _frames_to_seconds(duration_frames, fps)

    return {
        "start_frame": max_start,
        "end_frame": min_end,
        "start_tc": _frames_to_tc(max_start, fps),
        "end_tc": _frames_to_tc(min_end, fps),
        "duration_frames": duration_frames,
        "duration_seconds": round(duration_seconds, 3),
        "valid": True,
    }


# ---------------------------------------------------------------------------
# Offset computation
# ---------------------------------------------------------------------------

def compute_tc_offsets(
    sources: List[dict],
) -> dict:
    """Compute frame offsets for each source relative to the earliest start.

    Args:
        sources: List of source dicts with start_frame, end_frame, filepath.

    Returns:
        Dict mapping filepath -> offset_frames and offset_seconds.
    """
    if not sources:
        return {}

    # Reference point: earliest start timecode
    min_start = min(s.get("start_frame", 0) for s in sources)
    fps = sources[0].get("fps", 25.0)

    offsets = {}
    for src in sources:
        fp = src.get("filepath", src.get("label", ""))
        start_frame = src.get("start_frame", 0)
        offset_frames = start_frame - min_start
        offset_seconds = _frames_to_seconds(offset_frames, fps)

        offsets[fp] = {
            "offset_frames": offset_frames,
            "offset_seconds": round(offset_seconds, 4),
            "start_tc": src.get("start_tc", ""),
            "end_tc": src.get("end_tc", ""),
        }

    return offsets


# ---------------------------------------------------------------------------
# Timeline generation
# ---------------------------------------------------------------------------

def generate_synced_timeline(
    sources: List[dict],
    offsets: dict,
    output_path: str,
    format: str = "json",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Generate a synchronized timeline/EDL from multiple sources.

    Args:
        sources: List of source info dicts.
        offsets: Offset dict from compute_tc_offsets.
        output_path: Path for the output timeline file.
        format: Output format - 'json' or 'edl'.
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with timeline_path, format, source_count.
    """
    format = format.lower().strip()
    if format not in ("json", "edl"):
        raise ValueError(f"Unsupported timeline format: {format}. Use 'json' or 'edl'.")

    if on_progress:
        on_progress(10, "Building synchronized timeline")

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fps = sources[0].get("fps", 25.0) if sources else 25.0

    if format == "json":
        timeline = _build_json_timeline(sources, offsets, fps)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2)
    elif format == "edl":
        edl_content = _build_edl_timeline(sources, offsets, fps)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(edl_content)

    if on_progress:
        on_progress(100, "Timeline generated")

    return {
        "timeline_path": output_path,
        "format": format,
        "source_count": len(sources),
        "offsets": offsets,
    }


def _build_json_timeline(
    sources: List[dict],
    offsets: dict,
    fps: float,
) -> dict:
    """Build a JSON timeline structure."""
    tracks = []

    for idx, src in enumerate(sources):
        fp = src.get("filepath", src.get("label", ""))
        src_offset = offsets.get(fp, {})

        track = {
            "track_index": idx,
            "label": src.get("label", os.path.basename(fp)),
            "filepath": fp,
            "start_tc": src.get("start_tc", "00:00:00:00"),
            "end_tc": src.get("end_tc", "00:00:00:00"),
            "offset_frames": src_offset.get("offset_frames", 0),
            "offset_seconds": src_offset.get("offset_seconds", 0.0),
            "duration": src.get("duration", 0.0),
            "tc_source": src.get("tc_source", ""),
        }
        tracks.append(track)

    # Compute common range
    common = find_common_timecode_range(sources)

    return {
        "timeline_version": "1.0",
        "generator": "OpenCut TC Sync",
        "fps": fps,
        "tracks": tracks,
        "common_range": common,
        "total_sources": len(sources),
    }


def _build_edl_timeline(
    sources: List[dict],
    offsets: dict,
    fps: float,
) -> str:
    """Build an EDL (Edit Decision List) string."""
    lines = []
    lines.append("TITLE: OpenCut TC Sync")
    lines.append("FCM: NON-DROP FRAME")
    lines.append("")

    for idx, src in enumerate(sources):
        fp = src.get("filepath", src.get("label", ""))
        src_offset = offsets.get(fp, {})
        offset_frames = src_offset.get("offset_frames", 0)

        start_tc = src.get("start_tc", "00:00:00:00")
        end_tc = src.get("end_tc", "00:00:00:00")

        # Record start/end in timeline (accounting for offset)
        rec_start = _frames_to_tc(offset_frames, fps)
        end_frame = _tc_to_frames(end_tc, fps) - _tc_to_frames(start_tc, fps) + offset_frames
        rec_end = _frames_to_tc(end_frame, fps)

        edit_num = idx + 1
        reel = os.path.splitext(os.path.basename(fp))[0][:8].ljust(8)

        lines.append(
            f"{edit_num:03d}  {reel} V     C        "
            f"{start_tc} {end_tc} {rec_start} {rec_end}"
        )
        lines.append(f"* FROM CLIP NAME: {os.path.basename(fp)}")
        lines.append(f"* SOURCE: {fp}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main sync function
# ---------------------------------------------------------------------------

def sync_by_timecode(
    sources: List[str],
    fps: float = 0.0,
    output: Optional[str] = None,
    timeline_format: str = "json",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Synchronize multiple media sources using embedded timecodes.

    Args:
        sources: List of file paths to synchronize.
        fps: Expected fps (auto-detect if 0).
        output: Output path for the sync timeline.
        timeline_format: Format for the timeline file ('json' or 'edl').
        on_progress: Optional callback(pct, msg).

    Returns:
        SyncResult-like dict with synced sources, offsets, and timeline.
    """
    if not sources:
        raise ValueError("No source files provided")

    for src in sources:
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Source file not found: {src}")

    if on_progress:
        on_progress(5, f"Synchronizing {len(sources)} sources by timecode")

    # Extract timecodes from each source
    source_infos = []
    for idx, src_path in enumerate(sources):
        if on_progress:
            pct = 5 + int(50 * idx / len(sources))
            on_progress(pct, f"Extracting timecodes from source {idx + 1}/{len(sources)}")

        tc_info = _extract_source_timecodes(src_path, fps=fps)
        tc_info["filepath"] = src_path
        tc_info["label"] = os.path.basename(src_path)
        source_infos.append(tc_info)

    if on_progress:
        on_progress(60, "Computing offsets")

    # Compute offsets
    offsets = compute_tc_offsets(source_infos)

    # Find common range
    common_range = find_common_timecode_range(source_infos)

    if on_progress:
        on_progress(70, "Generating timeline")

    # Generate timeline
    if output is None:
        ext = ".json" if timeline_format == "json" else ".edl"
        output = _output_path(sources[0], "_tc_sync", ext)

    timeline_result = generate_synced_timeline(
        source_infos, offsets, output,
        format=timeline_format,
        on_progress=lambda p, m: on_progress(70 + p * 0.3, m) if on_progress else None,
    )

    if on_progress:
        on_progress(100, "Timecode sync complete")

    return {
        "synced_sources": source_infos,
        "common_range": common_range,
        "offsets": offsets,
        "timeline_path": timeline_result["timeline_path"],
        "timeline_format": timeline_format,
        "total_sources": len(sources),
        "synced_count": sum(1 for s in source_infos if s.get("tc_source") != "assumed"),
    }
