"""
OpenCut FCPXML Export v1.0.0

Generate Final Cut Pro XML (FCPXML 1.11) project files:
  - Timeline sequences with clips, cuts, markers
  - Frame-accurate timecode references
  - Resource and asset management
  - Compatible with Final Cut Pro 10.6+
"""

import logging
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union
from xml.dom import minidom

from opencut.helpers import get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------

FCPXML_VERSION = "1.11"
FCPXML_XMLNS = ""  # FCPXML uses no namespace prefix


@dataclass
class FCPXMLMarker:
    """A marker in the FCPXML timeline."""
    name: str = ""
    start: float = 0.0          # seconds
    duration: float = 0.0       # seconds (0 = point marker)
    note: str = ""
    marker_type: str = "standard"  # "standard", "todo", "chapter"


@dataclass
class FCPXMLClip:
    """A clip reference in an FCPXML sequence."""
    name: str = ""
    source_path: str = ""
    start: float = 0.0          # timeline position in seconds
    duration: float = 0.0       # clip duration in seconds
    source_start: float = 0.0   # source in-point in seconds
    source_duration: float = 0.0  # source extent in seconds
    audio_only: bool = False
    video_only: bool = False
    markers: List[FCPXMLMarker] = field(default_factory=list)
    enabled: bool = True


@dataclass
class FCPXMLSequence:
    """An FCPXML sequence / timeline."""
    name: str = ""
    duration: float = 0.0       # total duration in seconds
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    audio_rate: int = 48000
    audio_channels: int = 2
    clips: List[FCPXMLClip] = field(default_factory=list)
    markers: List[FCPXMLMarker] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Timecode Helpers
# ---------------------------------------------------------------------------

def _seconds_to_rational(seconds: float, fps: float = 30.0) -> str:
    """Convert seconds to FCPXML rational time (e.g. '3003/1001s' for 29.97).

    FCPXML uses rational frame counts as its time base.
    """
    if fps <= 0:
        fps = 30.0

    # Common frame rates and their rational bases
    fps_map = {
        23.976: (24000, 1001),
        23.98: (24000, 1001),
        24.0: (24, 1),
        25.0: (25, 1),
        29.97: (30000, 1001),
        30.0: (30, 1),
        50.0: (50, 1),
        59.94: (60000, 1001),
        60.0: (60, 1),
    }

    # Find closest known rate
    closest_rate = min(fps_map.keys(), key=lambda r: abs(r - fps))
    if abs(closest_rate - fps) < 0.05:
        num, den = fps_map[closest_rate]
    else:
        num, den = int(round(fps)), 1

    # Convert seconds to frame count * frame duration
    total_frames = round(seconds * num / den)
    # Express as rational: frames * den / num seconds
    rational_num = total_frames * den
    rational_den = num

    # Simplify
    g = math.gcd(rational_num, rational_den) if rational_num > 0 else 1
    rational_num //= g
    rational_den //= g

    return f"{rational_num}/{rational_den}s"


def _fps_to_frameDuration(fps: float) -> str:
    """Convert FPS to FCPXML frameDuration attribute (e.g. '1001/30000s')."""
    if fps <= 0:
        fps = 30.0

    fps_durations = {
        23.976: "1001/24000s",
        23.98: "1001/24000s",
        24.0: "100/2400s",
        25.0: "100/2500s",
        29.97: "1001/30000s",
        30.0: "100/3000s",
        50.0: "100/5000s",
        59.94: "1001/60000s",
        60.0: "100/6000s",
    }

    closest = min(fps_durations.keys(), key=lambda r: abs(r - fps))
    if abs(closest - fps) < 0.05:
        return fps_durations[closest]

    # Fallback
    den = int(round(fps * 100))
    return f"100/{den}s"


# ---------------------------------------------------------------------------
# Clip Creation
# ---------------------------------------------------------------------------

def create_fcpxml_clip(
    source_path: str,
    name: str = "",
    start: float = 0.0,
    duration: float = 0.0,
    source_start: float = 0.0,
    source_duration: float = 0.0,
    markers: Optional[List[dict]] = None,
    on_progress: Optional[Callable] = None,
) -> FCPXMLClip:
    """Create an FCPXML clip reference from a source media file.

    If duration is not specified, probes the media file for its duration.

    Args:
        source_path: Path to the source media file.
        name: Clip name. Defaults to filename without extension.
        start: Timeline position in seconds.
        duration: Clip duration in seconds (0 = use full source).
        source_start: Source in-point in seconds.
        source_duration: Source out-extent in seconds (0 = same as duration).
        markers: Optional list of marker dicts with name, start, note keys.
        on_progress: Optional callback (percent, message).

    Returns:
        FCPXMLClip object.

    Raises:
        FileNotFoundError: If source_path does not exist.
    """
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    if on_progress:
        on_progress(10, f"Creating clip from {os.path.basename(source_path)}...")

    if not name:
        name = os.path.splitext(os.path.basename(source_path))[0]

    # Probe source for duration if not provided
    if duration <= 0:
        info = get_video_info(source_path)
        duration = info["duration"]
        if duration <= 0:
            duration = 10.0  # fallback

    if source_duration <= 0:
        source_duration = duration

    clip = FCPXMLClip(
        name=name,
        source_path=source_path,
        start=start,
        duration=duration,
        source_start=source_start,
        source_duration=source_duration,
    )

    # Add markers
    if markers:
        for m in markers:
            if isinstance(m, dict):
                clip.markers.append(FCPXMLMarker(
                    name=str(m.get("name", "")),
                    start=float(m.get("start", 0)),
                    duration=float(m.get("duration", 0)),
                    note=str(m.get("note", "")),
                    marker_type=str(m.get("type", "standard")),
                ))

    if on_progress:
        on_progress(100, "Clip created.")

    return clip


# ---------------------------------------------------------------------------
# Sequence Creation
# ---------------------------------------------------------------------------

def create_fcpxml_sequence(
    clips: List[Union[FCPXMLClip, dict]],
    name: str = "OpenCut Sequence",
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    markers: Optional[List[dict]] = None,
    on_progress: Optional[Callable] = None,
) -> FCPXMLSequence:
    """Create an FCPXML sequence from a list of clips.

    Clips are placed sequentially on the timeline unless explicit start
    times are provided.

    Args:
        clips: List of FCPXMLClip objects or dicts with clip info.
        name: Sequence name.
        width: Sequence frame width.
        height: Sequence frame height.
        fps: Frames per second.
        markers: Optional list of sequence-level marker dicts.
        on_progress: Optional callback (percent, message).

    Returns:
        FCPXMLSequence object.

    Raises:
        ValueError: If clips list is empty.
    """
    if not clips:
        raise ValueError("At least one clip is required to create a sequence.")

    if on_progress:
        on_progress(10, "Building sequence...")

    resolved_clips: List[FCPXMLClip] = []
    timeline_cursor = 0.0

    for i, clip_data in enumerate(clips):
        if isinstance(clip_data, FCPXMLClip):
            clip = clip_data
        elif isinstance(clip_data, dict):
            clip = FCPXMLClip(
                name=str(clip_data.get("name", f"Clip {i+1}")),
                source_path=str(clip_data.get("source_path", "")),
                start=float(clip_data.get("start", -1)),
                duration=float(clip_data.get("duration", 10.0)),
                source_start=float(clip_data.get("source_start", 0)),
                source_duration=float(clip_data.get("source_duration", 0)),
            )
            # Parse markers
            for m in clip_data.get("markers", []):
                clip.markers.append(FCPXMLMarker(
                    name=str(m.get("name", "")),
                    start=float(m.get("start", 0)),
                    note=str(m.get("note", "")),
                ))
        else:
            continue

        # Auto-position clips sequentially if no explicit start
        if clip.start < 0:
            clip.start = timeline_cursor
        if clip.source_duration <= 0:
            clip.source_duration = clip.duration

        timeline_cursor = clip.start + clip.duration
        resolved_clips.append(clip)

        if on_progress:
            pct = 10 + int((i / len(clips)) * 50)
            on_progress(pct, f"Added clip {i+1}/{len(clips)}...")

    total_duration = max(
        (c.start + c.duration for c in resolved_clips), default=0.0
    )

    seq = FCPXMLSequence(
        name=name,
        duration=total_duration,
        width=width,
        height=height,
        fps=fps,
        clips=resolved_clips,
    )

    # Add sequence-level markers
    if markers:
        for m in markers:
            if isinstance(m, dict):
                seq.markers.append(FCPXMLMarker(
                    name=str(m.get("name", "")),
                    start=float(m.get("start", 0)),
                    duration=float(m.get("duration", 0)),
                    note=str(m.get("note", "")),
                    marker_type=str(m.get("type", "standard")),
                ))

    if on_progress:
        on_progress(80, "Sequence built.")

    return seq


# ---------------------------------------------------------------------------
# FCPXML Export
# ---------------------------------------------------------------------------

def export_fcpxml(
    sequence: Union[FCPXMLSequence, dict],
    output_path: str,
    project_name: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Export a sequence as FCPXML 1.11 XML file.

    Args:
        sequence: FCPXMLSequence object or dict with sequence data.
        output_path: Output .fcpxml file path.
        project_name: Project name in FCPXML metadata.
        on_progress: Optional callback (percent, message).

    Returns:
        Dict with output_path, clip count, duration, format version.

    Raises:
        ValueError: If sequence has no clips.
    """
    if on_progress:
        on_progress(5, "Preparing FCPXML export...")

    # Handle dict input
    if isinstance(sequence, dict):
        clips_data = sequence.get("clips", [])
        seq = create_fcpxml_sequence(
            clips=clips_data,
            name=sequence.get("name", "OpenCut Sequence"),
            width=int(sequence.get("width", 1920)),
            height=int(sequence.get("height", 1080)),
            fps=float(sequence.get("fps", 30.0)),
            markers=sequence.get("markers"),
        )
    else:
        seq = sequence

    if not seq.clips:
        raise ValueError("Sequence has no clips to export.")

    if not project_name:
        project_name = seq.name

    if on_progress:
        on_progress(15, "Building FCPXML document...")

    # Build FCPXML document
    fcpxml = ET.Element("fcpxml", attrib={"version": FCPXML_VERSION})

    # Resources element
    resources = ET.SubElement(fcpxml, "resources")

    # Format resource
    format_id = "r1"
    frame_dur = _fps_to_frameDuration(seq.fps)
    ET.SubElement(resources, "format", attrib={
        "id": format_id,
        "name": f"FFVideoFormat{seq.height}p{int(round(seq.fps))}",
        "frameDuration": frame_dur,
        "width": str(seq.width),
        "height": str(seq.height),
    })

    # Asset resources for each unique source file
    asset_map: Dict[str, str] = {}  # source_path -> asset_id
    asset_counter = 1

    for clip in seq.clips:
        src = clip.source_path
        if src and src not in asset_map:
            asset_counter += 1
            asset_id = f"r{asset_counter}"
            asset_map[src] = asset_id

            asset_elem = ET.SubElement(resources, "asset", attrib={
                "id": asset_id,
                "name": clip.name,
                "start": "0s",
                "duration": _seconds_to_rational(clip.source_duration, seq.fps),
                "format": format_id,
                "hasVideo": "1",
                "hasAudio": "1",
            })
            ET.SubElement(asset_elem, "media-rep", attrib={
                "kind": "original-media",
                "src": _path_to_file_url(src),
            })

    if on_progress:
        on_progress(40, "Building timeline structure...")

    # Library > Event > Project > Sequence
    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", attrib={
        "name": project_name,
    })
    project = ET.SubElement(event, "project", attrib={
        "name": project_name,
    })

    # Sequence (spine container)
    seq_duration = _seconds_to_rational(seq.duration, seq.fps)
    sequence_elem = ET.SubElement(project, "sequence", attrib={
        "duration": seq_duration,
        "format": format_id,
        "tcStart": "0s",
        "tcFormat": "NDF",
    })

    spine = ET.SubElement(sequence_elem, "spine")

    # Add clips to spine
    for i, clip in enumerate(seq.clips):
        if on_progress:
            pct = 40 + int((i / len(seq.clips)) * 40)
            on_progress(pct, f"Writing clip {i+1}/{len(seq.clips)}...")

        asset_id = asset_map.get(clip.source_path, format_id)

        clip_attribs = {
            "name": clip.name,
            "offset": _seconds_to_rational(clip.start, seq.fps),
            "duration": _seconds_to_rational(clip.duration, seq.fps),
            "start": _seconds_to_rational(clip.source_start, seq.fps),
            "ref": asset_id,
        }
        if not clip.enabled:
            clip_attribs["enabled"] = "0"

        if clip.audio_only:
            clip_elem = ET.SubElement(spine, "asset-clip", attrib=clip_attribs)
            clip_elem.set("audioRole", "dialogue")
        elif clip.video_only:
            clip_elem = ET.SubElement(spine, "asset-clip", attrib=clip_attribs)
            clip_elem.set("videoRole", "video")
        else:
            clip_elem = ET.SubElement(spine, "asset-clip", attrib=clip_attribs)

        # Add clip markers
        for marker in clip.markers:
            marker_attribs = {
                "start": _seconds_to_rational(marker.start, seq.fps),
                "duration": _seconds_to_rational(max(marker.duration, 1.0 / seq.fps), seq.fps),
                "value": marker.name,
            }

            if marker.marker_type == "todo":
                m_elem = ET.SubElement(clip_elem, "todo-marker", attrib=marker_attribs)
            elif marker.marker_type == "chapter":
                m_elem = ET.SubElement(clip_elem, "chapter-marker", attrib=marker_attribs)
            else:
                m_elem = ET.SubElement(clip_elem, "marker", attrib=marker_attribs)

            if marker.note:
                m_elem.set("note", marker.note)

    # Add sequence-level markers to the sequence element
    for marker in seq.markers:
        marker_attribs = {
            "start": _seconds_to_rational(marker.start, seq.fps),
            "duration": _seconds_to_rational(max(marker.duration, 1.0 / seq.fps), seq.fps),
            "value": marker.name,
        }
        if marker.marker_type == "chapter":
            m_elem = ET.SubElement(sequence_elem, "chapter-marker", attrib=marker_attribs)
        else:
            m_elem = ET.SubElement(sequence_elem, "marker", attrib=marker_attribs)
        if marker.note:
            m_elem.set("note", marker.note)

    if on_progress:
        on_progress(85, "Writing FCPXML file...")

    # Serialize to XML with pretty printing
    xml_str = ET.tostring(fcpxml, encoding="unicode", xml_declaration=False)
    declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    doctype = '<!DOCTYPE fcpxml>\n'

    # Pretty print
    try:
        dom = minidom.parseString(xml_str)
        pretty = dom.toprettyxml(indent="  ", encoding=None)
        # Remove the minidom declaration (we add our own)
        if pretty.startswith("<?xml"):
            pretty = pretty.split("\n", 1)[1] if "\n" in pretty else pretty
    except Exception:
        pretty = xml_str

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(declaration)
        f.write(doctype)
        f.write(pretty)

    if on_progress:
        on_progress(100, "FCPXML export complete.")

    return {
        "output_path": output_path,
        "format": f"fcpxml-{FCPXML_VERSION}",
        "project_name": project_name,
        "clip_count": len(seq.clips),
        "duration": seq.duration,
        "resolution": f"{seq.width}x{seq.height}",
        "fps": seq.fps,
        "marker_count": len(seq.markers) + sum(len(c.markers) for c in seq.clips),
        "file_size_bytes": os.path.getsize(output_path),
    }


# ---------------------------------------------------------------------------
# Path Helper
# ---------------------------------------------------------------------------

def _path_to_file_url(filepath: str) -> str:
    """Convert a local file path to a file:// URL for FCPXML."""
    abs_path = os.path.abspath(filepath)
    # Normalize separators
    abs_path = abs_path.replace("\\", "/")
    # Ensure leading slash for Windows paths (C:/foo -> /C:/foo)
    if abs_path[0] != "/":
        abs_path = "/" + abs_path
    return f"file://{abs_path}"
