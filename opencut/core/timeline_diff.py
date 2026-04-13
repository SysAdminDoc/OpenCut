"""
Timeline Diff / Comparison.

Compare two edit snapshots and highlight changes with
structured diff output and visual diff rendering.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import FFmpegCmd, run_ffmpeg

logger = logging.getLogger("opencut")


@dataclass
class TimelineChange:
    """A single change between two timeline snapshots."""
    change_type: str           # added | removed | modified | moved | trimmed
    track: int = 0
    clip_id: str = ""
    clip_name: str = ""
    old_start: Optional[float] = None
    new_start: Optional[float] = None
    old_end: Optional[float] = None
    new_end: Optional[float] = None
    old_properties: Dict[str, Any] = field(default_factory=dict)
    new_properties: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class TimelineDiff:
    """Complete diff between two timeline snapshots."""
    snapshot_a_name: str = ""
    snapshot_b_name: str = ""
    changes: List[TimelineChange] = field(default_factory=list)
    added_count: int = 0
    removed_count: int = 0
    modified_count: int = 0
    moved_count: int = 0
    total_changes: int = 0
    summary: str = ""


def _normalize_clip(clip: dict) -> dict:
    """Normalize a clip dict for comparison."""
    return {
        "id": clip.get("id", clip.get("clip_id", "")),
        "name": clip.get("name", clip.get("clip_name", "")),
        "track": clip.get("track", 0),
        "start": float(clip.get("start", 0)),
        "end": float(clip.get("end", 0)),
        "source": clip.get("source", ""),
        "effects": clip.get("effects", []),
        "volume": clip.get("volume", 1.0),
        "opacity": clip.get("opacity", 1.0),
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def diff_timelines(
    snapshot_a: Dict,
    snapshot_b: Dict,
    on_progress: Optional[Callable] = None,
) -> TimelineDiff:
    """Compare two timeline snapshots and produce a structured diff.

    Args:
        snapshot_a: First timeline snapshot (dict with 'clips' list).
        snapshot_b: Second timeline snapshot (dict with 'clips' list).

    Returns:
        TimelineDiff with all detected changes.
    """
    if on_progress:
        on_progress(10, "Parsing snapshots")

    clips_a = {_normalize_clip(c)["id"]: _normalize_clip(c)
               for c in snapshot_a.get("clips", [])}
    clips_b = {_normalize_clip(c)["id"]: _normalize_clip(c)
               for c in snapshot_b.get("clips", [])}

    ids_a = set(clips_a.keys())
    ids_b = set(clips_b.keys())

    changes = []

    if on_progress:
        on_progress(30, "Finding added/removed clips")

    # Added clips
    for cid in ids_b - ids_a:
        clip = clips_b[cid]
        changes.append(TimelineChange(
            change_type="added",
            track=clip["track"],
            clip_id=cid,
            clip_name=clip["name"],
            new_start=clip["start"],
            new_end=clip["end"],
            new_properties=clip,
            description=f"Added '{clip['name']}' at {clip['start']:.2f}s",
        ))

    # Removed clips
    for cid in ids_a - ids_b:
        clip = clips_a[cid]
        changes.append(TimelineChange(
            change_type="removed",
            track=clip["track"],
            clip_id=cid,
            clip_name=clip["name"],
            old_start=clip["start"],
            old_end=clip["end"],
            old_properties=clip,
            description=f"Removed '{clip['name']}' from {clip['start']:.2f}s",
        ))

    if on_progress:
        on_progress(60, "Comparing modified clips")

    # Modified / moved clips
    for cid in ids_a & ids_b:
        old = clips_a[cid]
        new = clips_b[cid]

        if old == new:
            continue

        # Determine change type
        if old["track"] != new["track"] or abs(old["start"] - new["start"]) > 0.01:
            ctype = "moved"
        elif abs(old["end"] - new["end"]) > 0.01 or abs(old["start"] - new["start"]) > 0.01:
            ctype = "trimmed"
        else:
            ctype = "modified"

        desc_parts = []
        if old["start"] != new["start"]:
            desc_parts.append(f"start {old['start']:.2f}s -> {new['start']:.2f}s")
        if old["end"] != new["end"]:
            desc_parts.append(f"end {old['end']:.2f}s -> {new['end']:.2f}s")
        if old["track"] != new["track"]:
            desc_parts.append(f"track {old['track']} -> {new['track']}")
        if old.get("effects") != new.get("effects"):
            desc_parts.append("effects changed")

        changes.append(TimelineChange(
            change_type=ctype,
            track=new["track"],
            clip_id=cid,
            clip_name=new["name"],
            old_start=old["start"],
            new_start=new["start"],
            old_end=old["end"],
            new_end=new["end"],
            old_properties=old,
            new_properties=new,
            description=f"{ctype.title()} '{new['name']}': {', '.join(desc_parts)}",
        ))

    if on_progress:
        on_progress(90, "Compiling diff")

    added = sum(1 for c in changes if c.change_type == "added")
    removed = sum(1 for c in changes if c.change_type == "removed")
    modified = sum(1 for c in changes if c.change_type in ("modified", "trimmed"))
    moved = sum(1 for c in changes if c.change_type == "moved")

    summary_parts = []
    if added:
        summary_parts.append(f"{added} added")
    if removed:
        summary_parts.append(f"{removed} removed")
    if modified:
        summary_parts.append(f"{modified} modified")
    if moved:
        summary_parts.append(f"{moved} moved")

    diff = TimelineDiff(
        snapshot_a_name=snapshot_a.get("name", "Snapshot A"),
        snapshot_b_name=snapshot_b.get("name", "Snapshot B"),
        changes=changes,
        added_count=added,
        removed_count=removed,
        modified_count=modified,
        moved_count=moved,
        total_changes=len(changes),
        summary=", ".join(summary_parts) if summary_parts else "No changes",
    )

    if on_progress:
        on_progress(100, "Diff complete")

    return diff


def render_diff_visual(
    diff: TimelineDiff,
    output_path: str,
    width: int = 1920,
    height: int = 200,
    on_progress: Optional[Callable] = None,
) -> str:
    """Render a visual representation of timeline changes.

    Generates an image showing changed regions highlighted on a timeline bar.

    Args:
        diff: TimelineDiff to visualize.
        output_path: Path for the output image.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Path to the rendered diff image.
    """
    if on_progress:
        on_progress(20, "Generating diff visualization")

    # Build a filter that draws colored regions on a blank canvas
    color_map = {
        "added": "0x00FF0080",
        "removed": "0xFF000080",
        "modified": "0xFFFF0080",
        "moved": "0x0088FF80",
        "trimmed": "0xFF880080",
    }

    # Find time range
    all_times = []
    for c in diff.changes:
        if c.old_start is not None:
            all_times.append(c.old_start)
        if c.new_start is not None:
            all_times.append(c.new_start)
        if c.old_end is not None:
            all_times.append(c.old_end)
        if c.new_end is not None:
            all_times.append(c.new_end)

    if not all_times:
        # Create empty diff image
        cmd = (
            FFmpegCmd()
            .pre_input("-f", "lavfi")
            .input(f"color=c=0x333333:s={width}x{height}:d=1")
            .frames(1)
            .output(output_path)
            .build()
        )
        run_ffmpeg(cmd, timeout=10)
        return output_path

    max_time = max(all_times) if all_times else 1.0
    if max_time <= 0:
        max_time = 1.0

    # Build drawbox filter for each change
    boxes = []
    for c in diff.changes:
        start = c.new_start if c.new_start is not None else (c.old_start or 0)
        end = c.new_end if c.new_end is not None else (c.old_end or start + 0.5)
        x = int(start / max_time * width)
        w = max(int((end - start) / max_time * width), 2)
        color = color_map.get(c.change_type, "0xFFFFFF80")
        boxes.append(f"drawbox=x={x}:y=0:w={w}:h={height}:color={color}:t=fill")

    vf = "color=c=0x333333:s={w}x{h}:d=1,{boxes}".format(
        w=width, h=height, boxes=",".join(boxes) if boxes else "null"
    )

    cmd = [
        "-f", "lavfi", "-i", vf.split(",")[0] + ":d=1",
        "-vf", ",".join(vf.split(",")[1:]) if "," in vf else "null",
        "-frames:v", "1", "-y", output_path,
    ]

    # Simpler approach: use lavfi source with drawbox chain
    filter_str = f"color=c=0x333333:s={width}x{height}:d=1"
    for box in boxes:
        filter_str += f",{box}"

    cmd = (
        FFmpegCmd()
        .pre_input("-f", "lavfi")
        .input(filter_str)
        .frames(1)
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd, timeout=15)

    if on_progress:
        on_progress(100, "Diff visualization complete")

    return output_path


def export_diff_report(
    diff: TimelineDiff,
    format: str = "json",
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """Export a diff report in various formats.

    Args:
        diff: TimelineDiff to export.
        format: Output format ('json', 'text', 'html').
        output_path: Output file path (auto-generated if None).

    Returns:
        Path to the exported report.
    """
    if format not in ("json", "text", "html"):
        raise ValueError(f"Unsupported format: {format}. Use json, text, or html.")

    if output_path is None:
        import tempfile
        ext = {"json": ".json", "text": ".txt", "html": ".html"}[format]
        output_path = os.path.join(tempfile.gettempdir(), f"timeline_diff{ext}")

    if on_progress:
        on_progress(30, f"Generating {format} report")

    if format == "json":
        report = {
            "snapshot_a": diff.snapshot_a_name,
            "snapshot_b": diff.snapshot_b_name,
            "summary": diff.summary,
            "total_changes": diff.total_changes,
            "added": diff.added_count,
            "removed": diff.removed_count,
            "modified": diff.modified_count,
            "moved": diff.moved_count,
            "changes": [asdict(c) for c in diff.changes],
        }
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    elif format == "text":
        lines = [
            f"Timeline Diff: {diff.snapshot_a_name} vs {diff.snapshot_b_name}",
            f"Summary: {diff.summary}",
            f"Total changes: {diff.total_changes}",
            "",
        ]
        for c in diff.changes:
            lines.append(f"  [{c.change_type.upper():10s}] {c.description}")
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    elif format == "html":
        rows = ""
        for c in diff.changes:
            color = {
                "added": "#22c55e", "removed": "#ef4444",
                "modified": "#eab308", "moved": "#3b82f6",
                "trimmed": "#f97316",
            }.get(c.change_type, "#888")
            rows += (
                f'<tr style="border-left:4px solid {color}">'
                f'<td>{c.change_type}</td>'
                f'<td>{c.clip_name}</td>'
                f'<td>{c.description}</td></tr>\n'
            )
        html = (
            f"<html><head><title>Timeline Diff</title></head><body>"
            f"<h1>{diff.snapshot_a_name} vs {diff.snapshot_b_name}</h1>"
            f"<p>{diff.summary}</p>"
            f"<table border='1' cellpadding='4'>"
            f"<tr><th>Type</th><th>Clip</th><th>Description</th></tr>"
            f"{rows}</table></body></html>"
        )
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)

    if on_progress:
        on_progress(100, "Report exported")

    return output_path
