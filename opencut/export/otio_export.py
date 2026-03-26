"""
OpenTimelineIO (OTIO) export.

Generates OTIO files that can be imported into any NLE that supports the format:
- Adobe Premiere Pro (via OTIO adapter or FCP XML conversion)
- DaVinci Resolve (native OTIO support)
- Final Cut Pro (via FCPXML adapter)
- Avid Media Composer (via AAF adapter)
- And many more

OTIO is the Academy Software Foundation's universal timeline interchange format.

Requires: pip install opentimelineio
"""

import logging
import os
from typing import List, Optional

from ..core.silence import TimeSegment

logger = logging.getLogger("opencut")


def check_otio_available() -> bool:
    """Check if OpenTimelineIO is installed."""
    try:
        import opentimelineio  # noqa: F401
        return True
    except ImportError:
        return False


def export_otio(
    filepath: str,
    speech_segments: List[TimeSegment],
    output_path: str,
    sequence_name: str = "OpenCut Edit",
    framerate: float = 24.0,
) -> str:
    """
    Export an edited timeline as an OTIO file.

    Creates an OTIO timeline with clips corresponding to the speech
    segments, effectively removing all silences. Compatible with any
    NLE that supports OpenTimelineIO.

    Args:
        filepath: Path to the original media file.
        speech_segments: List of speech segments to keep.
        output_path: Path for the output .otio file.
        sequence_name: Name for the timeline/sequence.
        framerate: Timeline frame rate.

    Returns:
        Path to the written OTIO file.
    """
    try:
        import opentimelineio as otio
    except ImportError:
        raise ImportError(
            "OpenTimelineIO not installed. Install with: pip install opentimelineio"
        )

    # Validate input
    if not speech_segments:
        raise ValueError("No segments to export. Run silence removal or filler detection first.")

    # Create the timeline
    timeline = otio.schema.Timeline(name=sequence_name)
    timeline.global_start_time = otio.opentime.RationalTime(0, framerate)

    # Get the media reference URL (shared across all clips)
    media_url = _file_to_url(filepath)

    # Helper to create a clip from a segment
    def _make_clip(seg, idx, suffix=""):
        start_time = otio.opentime.RationalTime.from_seconds(seg.start, framerate)
        end_time = otio.opentime.RationalTime.from_seconds(seg.end, framerate)
        duration = end_time - start_time
        label = seg.label if seg.label and seg.label != "speech" else f"Segment {idx + 1}"
        return otio.schema.Clip(
            name=f"{label}{suffix}",
            media_reference=otio.schema.ExternalReference(target_url=media_url),
            source_range=otio.opentime.TimeRange(start_time=start_time, duration=duration),
        )

    # Create video track
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    for i, segment in enumerate(speech_segments):
        track.append(_make_clip(segment, i))
    timeline.tracks.append(track)

    # Create matching audio track
    audio_track = otio.schema.Track(name="A1", kind=otio.schema.TrackKind.Audio)
    for i, segment in enumerate(speech_segments):
        audio_track.append(_make_clip(segment, i, " (audio)"))
    timeline.tracks.append(audio_track)

    # Write the OTIO file
    otio.adapters.write_to_file(timeline, output_path)
    logger.info("Exported OTIO timeline: %s (%d clips)", output_path, len(speech_segments))

    return output_path


def export_otio_from_cuts(
    filepath: str,
    cuts: List[dict],
    output_path: str,
    sequence_name: str = "OpenCut Edit",
    framerate: float = 24.0,
    total_duration: float = 0.0,
) -> str:
    """
    Export an OTIO timeline from cut regions (regions to REMOVE).

    Inverts cuts to get kept regions, then exports as OTIO.

    Args:
        filepath: Path to the original media file.
        cuts: List of dicts with 'start' and 'end' keys (regions to remove).
        output_path: Path for the output .otio file.
        sequence_name: Name for the timeline.
        framerate: Timeline frame rate.
        total_duration: Total duration of the source media.

    Returns:
        Path to the written OTIO file.
    """
    if total_duration <= 0:
        from ..utils.media import probe
        info = probe(filepath)
        total_duration = info.duration

    # Sort cuts by start time
    sorted_cuts = sorted(cuts, key=lambda c: float(c.get("start", 0)))

    # Invert: get kept regions
    kept = []
    pos = 0.0

    for cut in sorted_cuts:
        cut_start = float(cut.get("start", 0))
        cut_end = float(cut.get("end", 0))

        if cut_start > pos:
            kept.append(TimeSegment(start=pos, end=cut_start, label="speech"))
        pos = max(pos, cut_end)

    if pos < total_duration:
        kept.append(TimeSegment(start=pos, end=total_duration, label="speech"))

    return export_otio(filepath, kept, output_path, sequence_name, framerate)


def export_otio_markers(
    filepath: str,
    markers: List[dict],
    output_path: str,
    sequence_name: str = "OpenCut Markers",
    framerate: float = 24.0,
    total_duration: float = 0.0,
) -> str:
    """
    Export an OTIO timeline with markers (chapters, beat markers, etc.).

    Args:
        filepath: Path to the original media file.
        markers: List of dicts with 'time', 'name', optional 'color'.
        output_path: Path for the output .otio file.
        sequence_name: Name for the timeline.
        framerate: Timeline frame rate.
        total_duration: Total duration of the source media.

    Returns:
        Path to the written OTIO file.
    """
    try:
        import opentimelineio as otio
    except ImportError:
        raise ImportError("OpenTimelineIO not installed. Install with: pip install opentimelineio")

    if total_duration <= 0:
        from ..utils.media import probe
        info = probe(filepath)
        total_duration = info.duration

    timeline = otio.schema.Timeline(name=sequence_name)
    timeline.global_start_time = otio.opentime.RationalTime(0, framerate)

    # Single clip spanning the full media
    media_ref = otio.schema.ExternalReference(target_url=_file_to_url(filepath))

    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)

    full_clip = otio.schema.Clip(
        name=os.path.basename(filepath),
        media_reference=media_ref,
        source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, framerate),
            duration=otio.opentime.RationalTime.from_seconds(total_duration, framerate),
        ),
    )

    # Add markers to the clip
    for m in markers:
        marker_time = otio.opentime.RationalTime.from_seconds(
            float(m.get("time", 0)), framerate
        )
        marker = otio.schema.Marker(
            name=m.get("name", "Marker"),
            marked_range=otio.opentime.TimeRange(
                start_time=marker_time,
                duration=otio.opentime.RationalTime(1, framerate),
            ),
            color=_map_color(m.get("color", "GREEN")),
        )
        full_clip.markers.append(marker)

    track.append(full_clip)
    timeline.tracks.append(track)

    otio.adapters.write_to_file(timeline, output_path)
    logger.info("Exported OTIO with %d markers: %s", len(markers), output_path)

    return output_path


def _file_to_url(filepath: str) -> str:
    """Convert a file path to a file:// URL (with fallback for network paths)."""
    try:
        import pathlib
        return pathlib.Path(filepath).as_uri()
    except (ValueError, OSError):
        # Fallback for network paths, UNC paths, or special characters
        # that pathlib can't convert to URI
        import urllib.parse
        abs_path = os.path.abspath(filepath)
        return "file:///" + urllib.parse.quote(abs_path.replace("\\", "/"), safe="/:")


def _map_color(color_name: str) -> str:
    """Map color names to OTIO marker color constants."""
    try:
        import opentimelineio as otio
        color_map = {
            "red": otio.schema.MarkerColor.RED,
            "green": otio.schema.MarkerColor.GREEN,
            "blue": otio.schema.MarkerColor.BLUE,
            "yellow": otio.schema.MarkerColor.YELLOW,
            "cyan": otio.schema.MarkerColor.CYAN,
            "magenta": otio.schema.MarkerColor.MAGENTA,
            "pink": otio.schema.MarkerColor.PINK,
            "orange": otio.schema.MarkerColor.ORANGE,
            "purple": otio.schema.MarkerColor.PURPLE,
            "white": otio.schema.MarkerColor.WHITE,
            "black": otio.schema.MarkerColor.BLACK,
        }
        return color_map.get(color_name.lower(), otio.schema.MarkerColor.GREEN)
    except (ImportError, AttributeError):
        return "GREEN"
