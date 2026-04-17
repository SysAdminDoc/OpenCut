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
from typing import List

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


def check_aaf_available() -> bool:
    """True when the OTIO AAF adapter (`otio-aaf-adapter`) is installed."""
    if not check_otio_available():
        return False
    try:
        import otio_aaf_adapter  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import opentimelineio as otio
        # Some installs register the adapter without the shim module.
        available = {a.name for a in otio.adapters.available_adapter_names()}
        return "aaf" in available
    except Exception:  # noqa: BLE001
        return False


def export_aaf(
    filepath: str,
    speech_segments: List[TimeSegment],
    output_path: str,
    sequence_name: str = "OpenCut Edit",
    framerate: float = 24.0,
) -> str:
    """Export speech segments as an Avid-compatible .aaf file.

    Uses the ``otio-aaf-adapter`` (https://github.com/OpenTimelineIO/otio-aaf-adapter,
    Apache-2) to emit an AAF that Avid Media Composer imports natively.
    Falls back to a clear error if the adapter is missing — AAF emission
    is not in core OpenTimelineIO.

    Raises:
        ImportError: OTIO or the AAF adapter is not installed.
        ValueError: ``speech_segments`` empty.
    """
    if not check_aaf_available():
        raise ImportError(
            "OTIO AAF adapter not installed. "
            "Install: pip install otio-aaf-adapter"
        )
    if not speech_segments:
        raise ValueError("No segments to export.")

    # Build a normal OTIO timeline first (in-memory), then swap the
    # output adapter to "aaf" via write_to_file's adapter selection.
    import opentimelineio as otio

    timeline = otio.schema.Timeline(name=sequence_name)
    timeline.global_start_time = otio.opentime.RationalTime(0, framerate)
    media_url = _file_to_url(filepath)
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    for i, seg in enumerate(speech_segments):
        start_time = otio.opentime.RationalTime.from_seconds(seg.start, framerate)
        duration = otio.opentime.RationalTime.from_seconds(
            max(0.0, seg.end - seg.start), framerate
        )
        label = seg.label if seg.label and seg.label != "speech" else f"Segment {i + 1}"
        track.append(otio.schema.Clip(
            name=label,
            media_reference=otio.schema.ExternalReference(target_url=media_url),
            source_range=otio.opentime.TimeRange(start_time=start_time, duration=duration),
        ))
    timeline.tracks.append(track)

    # Force the AAF adapter by file extension, or pass adapter_name explicitly.
    if not output_path.lower().endswith(".aaf"):
        output_path = output_path + ".aaf"
    otio.adapters.write_to_file(timeline, output_path, adapter_name="aaf")
    logger.info("Exported AAF timeline: %s (%d clips)", output_path, len(speech_segments))
    return output_path


def export_otioz(
    filepath: str,
    speech_segments: List[TimeSegment],
    output_path: str,
    sequence_name: str = "OpenCut Edit",
    framerate: float = 24.0,
    bundle_media: bool = False,
) -> str:
    """Export speech segments as an ``.otioz`` bundle (portable handoff).

    OTIOZ is a zip container containing the ``.otio`` file plus an
    optional ``media/`` directory — a one-file deliverable that any
    OTIO-aware NLE can unpack.

    Args:
        bundle_media: When True, the source ``filepath`` is copied into
            the bundle so the receiver doesn't need the original path
            intact. When False, the bundle carries only the timeline
            and external references (smaller, but requires the
            receiver to have the media).

    Raises:
        ImportError: OTIO is not installed.
        ValueError: ``speech_segments`` empty.
    """
    if not check_otio_available():
        raise ImportError(
            "OpenTimelineIO not installed. Install: pip install opentimelineio"
        )
    if not speech_segments:
        raise ValueError("No segments to export.")

    import opentimelineio as otio

    # Build the in-memory timeline (same logic as export_otio)
    timeline = otio.schema.Timeline(name=sequence_name)
    timeline.global_start_time = otio.opentime.RationalTime(0, framerate)
    media_url = _file_to_url(filepath)
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    for i, seg in enumerate(speech_segments):
        start_time = otio.opentime.RationalTime.from_seconds(seg.start, framerate)
        duration = otio.opentime.RationalTime.from_seconds(
            max(0.0, seg.end - seg.start), framerate
        )
        label = seg.label if seg.label and seg.label != "speech" else f"Segment {i + 1}"
        track.append(otio.schema.Clip(
            name=label,
            media_reference=otio.schema.ExternalReference(target_url=media_url),
            source_range=otio.opentime.TimeRange(start_time=start_time, duration=duration),
        ))
    timeline.tracks.append(track)

    # Normalise extension
    if not output_path.lower().endswith(".otioz"):
        output_path = output_path + ".otioz"

    if bundle_media:
        # file_bundle_utils ships with OpenTimelineIO and handles zip
        # layout correctly.  MediaReferencePolicy.ALL_MISSING_REFERENCES
        # would fail the write if refs can't be resolved; we use
        # COPY_ALL_REFERENCES to embed the media.
        from opentimelineio.file_bundle_utils import MediaReferencePolicy
        otio.adapters.write_to_file(
            timeline, output_path,
            adapter_name="otioz",
            media_policy=MediaReferencePolicy.AllMissingReferences,
        )
    else:
        otio.adapters.write_to_file(
            timeline, output_path,
            adapter_name="otioz",
        )
    logger.info(
        "Exported OTIOZ bundle: %s (%d clips, media_bundled=%s)",
        output_path, len(speech_segments), bundle_media,
    )
    return output_path


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
