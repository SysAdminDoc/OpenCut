"""
OpenCut EDL / AAF Import & Export

Generate CMX3600 EDL from cut list data, parse EDL back into cut list,
and export a simplified AAF stub as structured data.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

# CMX3600 timecode pattern  HH:MM:SS:FF
_TC_PATTERN = re.compile(r"(\d{2}):(\d{2}):(\d{2}):(\d{2})")


@dataclass
class CutEntry:
    """A single cut/edit event in a cut list."""
    reel: str = "AX"
    channel: str = "V"
    transition: str = "C"
    source_in: str = "00:00:00:00"
    source_out: str = "00:00:00:00"
    record_in: str = "00:00:00:00"
    record_out: str = "00:00:00:00"
    clip_name: str = ""
    source_file: str = ""


@dataclass
class EDLResult:
    """Result of an EDL export."""
    output_path: str
    event_count: int


@dataclass
class EDLImportResult:
    """Result of an EDL import."""
    cuts: List[Dict[str, Any]]
    title: str
    event_count: int


def _seconds_to_tc(seconds: float, fps: float = 30.0) -> str:
    """Convert seconds to timecode string HH:MM:SS:FF."""
    if seconds < 0:
        seconds = 0.0
    total_frames = int(round(seconds * fps))
    ff = total_frames % int(fps)
    total_seconds = total_frames // int(fps)
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _tc_to_seconds(tc: str, fps: float = 30.0) -> float:
    """Convert timecode string HH:MM:SS:FF to seconds."""
    m = _TC_PATTERN.match(tc)
    if not m:
        raise ValueError(f"Invalid timecode: {tc}")
    hh, mm, ss, ff = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    total_frames = hh * 3600 * fps + mm * 60 * fps + ss * fps + ff
    return total_frames / fps


def export_edl(
    cuts: List[Dict[str, Any]],
    output_path: str,
    title: str = "OpenCut Export",
    fps: float = 30.0,
    on_progress: Optional[Callable] = None,
) -> EDLResult:
    """Export a cut list as CMX3600 EDL.

    Args:
        cuts: List of cut dicts with keys: reel, channel, transition,
              source_in, source_out, record_in, record_out, clip_name.
              Times can be timecodes (HH:MM:SS:FF) or seconds (float).
        output_path: Path for the output .edl file.
        title: EDL title line.
        fps: Frame rate for timecode conversion.
        on_progress: Optional progress callback.

    Returns:
        EDLResult with output path and event count.
    """
    if on_progress:
        on_progress(5, "Preparing EDL export...")

    if not cuts:
        raise ValueError("Cut list is empty")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]

    for i, cut in enumerate(cuts):
        event_num = i + 1

        if on_progress:
            pct = 10 + int((i / len(cuts)) * 80)
            on_progress(pct, f"Writing event {event_num}/{len(cuts)}...")

        reel = cut.get("reel", "AX")
        channel = cut.get("channel", "V")
        transition = cut.get("transition", "C")

        # Convert seconds to timecodes if needed
        src_in = cut.get("source_in", "00:00:00:00")
        src_out = cut.get("source_out", "00:00:00:00")
        rec_in = cut.get("record_in", "00:00:00:00")
        rec_out = cut.get("record_out", "00:00:00:00")

        if isinstance(src_in, (int, float)):
            src_in = _seconds_to_tc(src_in, fps)
        if isinstance(src_out, (int, float)):
            src_out = _seconds_to_tc(src_out, fps)
        if isinstance(rec_in, (int, float)):
            rec_in = _seconds_to_tc(rec_in, fps)
        if isinstance(rec_out, (int, float)):
            rec_out = _seconds_to_tc(rec_out, fps)

        # CMX3600 event line format:
        # NNN  REEL  CHAN  TRANS  SRC_IN  SRC_OUT  REC_IN  REC_OUT
        event_line = (
            f"{event_num:03d}  {reel:<8s} {channel:<4s} {transition:<8s} "
            f"{src_in} {src_out} {rec_in} {rec_out}"
        )
        lines.append(event_line)

        # Optional clip name comment
        clip_name = cut.get("clip_name", "")
        if clip_name:
            lines.append(f"* FROM CLIP NAME: {clip_name}")

        source_file = cut.get("source_file", "")
        if source_file:
            lines.append(f"* SOURCE FILE: {source_file}")

        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if on_progress:
        on_progress(100, "EDL export complete")

    logger.info("Exported EDL with %d events to %s", len(cuts), output_path)
    return EDLResult(output_path=output_path, event_count=len(cuts))


def import_edl(
    edl_path: str,
    fps: float = 30.0,
    on_progress: Optional[Callable] = None,
) -> EDLImportResult:
    """Import a CMX3600 EDL file into a cut list.

    Args:
        edl_path: Path to the .edl file.
        fps: Frame rate for timecode interpretation.
        on_progress: Optional progress callback.

    Returns:
        EDLImportResult with cuts list, title, and event count.
    """
    if on_progress:
        on_progress(5, "Reading EDL file...")

    if not os.path.isfile(edl_path):
        raise FileNotFoundError(f"EDL file not found: {edl_path}")

    with open(edl_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.strip().split("\n")

    if on_progress:
        on_progress(20, "Parsing EDL events...")

    title = "Untitled"
    cuts: List[Dict[str, Any]] = []
    current_cut: Optional[Dict[str, Any]] = None

    # EDL event line pattern: NNN  REEL  CHAN  TRANS  TC TC TC TC
    event_pattern = re.compile(
        r"^\s*(\d{3})\s+"           # event number
        r"(\S+)\s+"                  # reel
        r"([VBAB12]+\d*)\s+"        # channel
        r"(\S+)\s+"                  # transition type
        r"(\d{2}:\d{2}:\d{2}:\d{2})\s+"  # source in
        r"(\d{2}:\d{2}:\d{2}:\d{2})\s+"  # source out
        r"(\d{2}:\d{2}:\d{2}:\d{2})\s+"  # record in
        r"(\d{2}:\d{2}:\d{2}:\d{2})"     # record out
    )

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("TITLE:"):
            title = stripped[6:].strip()
            continue

        if stripped.startswith("FCM:"):
            continue

        m = event_pattern.match(stripped)
        if m:
            # Save previous cut
            if current_cut is not None:
                cuts.append(current_cut)

            current_cut = {
                "event_num": int(m.group(1)),
                "reel": m.group(2),
                "channel": m.group(3),
                "transition": m.group(4),
                "source_in": m.group(5),
                "source_out": m.group(6),
                "record_in": m.group(7),
                "record_out": m.group(8),
                "source_in_seconds": _tc_to_seconds(m.group(5), fps),
                "source_out_seconds": _tc_to_seconds(m.group(6), fps),
                "record_in_seconds": _tc_to_seconds(m.group(7), fps),
                "record_out_seconds": _tc_to_seconds(m.group(8), fps),
                "clip_name": "",
                "source_file": "",
            }
            continue

        # Parse comment lines associated with current event
        if current_cut is not None:
            if stripped.startswith("* FROM CLIP NAME:"):
                current_cut["clip_name"] = stripped[17:].strip()
            elif stripped.startswith("* SOURCE FILE:"):
                current_cut["source_file"] = stripped[14:].strip()

    # Don't forget the last event
    if current_cut is not None:
        cuts.append(current_cut)

    if on_progress:
        on_progress(100, "EDL import complete")

    logger.info("Imported EDL '%s' with %d events from %s", title, len(cuts), edl_path)
    return EDLImportResult(cuts=cuts, title=title, event_count=len(cuts))


def export_aaf_stub(
    cuts: List[Dict[str, Any]],
    output_path: str,
    title: str = "OpenCut Export",
    fps: float = 30.0,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Export a simplified AAF stub as structured JSON data.

    This creates a JSON file representing the AAF structure without
    requiring a full AAF library.  Useful for interchange with tools
    that can read structured timeline data.

    Args:
        cuts: List of cut dicts (same format as export_edl).
        output_path: Path for the output .json file.
        title: Composition title.
        fps: Frame rate.
        on_progress: Optional progress callback.

    Returns:
        Dict with output_path, event_count, and structure summary.
    """
    if on_progress:
        on_progress(5, "Preparing AAF stub export...")

    if not cuts:
        raise ValueError("Cut list is empty")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Build AAF-like structure
    mob_slots = []
    for i, cut in enumerate(cuts):
        if on_progress:
            pct = 10 + int((i / len(cuts)) * 80)
            on_progress(pct, f"Processing event {i + 1}/{len(cuts)}...")

        src_in = cut.get("source_in", 0)
        src_out = cut.get("source_out", 0)
        if isinstance(src_in, str):
            src_in = _tc_to_seconds(src_in, fps)
        if isinstance(src_out, str):
            src_out = _tc_to_seconds(src_out, fps)

        mob_slots.append({
            "slot_id": i + 1,
            "segment": {
                "type": "SourceClip",
                "start_time": src_in,
                "length": src_out - src_in,
                "source_mob_id": cut.get("reel", f"mob_{i}"),
                "clip_name": cut.get("clip_name", ""),
                "source_file": cut.get("source_file", ""),
                "channel": cut.get("channel", "V"),
            },
        })

    aaf_data = {
        "format": "aaf_stub_v1",
        "title": title,
        "fps": fps,
        "composition": {
            "type": "CompositionMob",
            "name": title,
            "mob_slots": mob_slots,
        },
        "source_mobs": list({
            cut.get("reel", f"mob_{i}")
            for i, cut in enumerate(cuts)
        }),
        "event_count": len(cuts),
    }

    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aaf_data, f, indent=2)

    if on_progress:
        on_progress(100, "AAF stub export complete")

    logger.info("Exported AAF stub with %d events to %s", len(cuts), output_path)
    return {
        "output_path": output_path,
        "event_count": len(cuts),
        "format": "aaf_stub_v1",
    }
