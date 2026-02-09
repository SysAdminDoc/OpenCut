"""
Premiere Pro XML export.

Generates FCP 7 XML interchange format that can be imported into
Adobe Premiere Pro and other NLEs that support this format
(DaVinci Resolve, Vegas Pro, etc.).
"""

import os
import uuid
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Optional

from ..core.silence import TimeSegment
from ..core.zoom import ZoomEvent
from ..utils.config import ExportConfig
from ..utils.media import MediaInfo, probe


def export_premiere_xml(
    filepath: str,
    speech_segments: List[TimeSegment],
    output_path: str,
    config: Optional[ExportConfig] = None,
    zoom_events: Optional[List[ZoomEvent]] = None,
) -> str:
    """
    Export an edited timeline as Premiere Pro XML.

    Creates an FCP 7 XML file with clips corresponding to the speech
    segments, effectively removing all silences.

    Args:
        filepath: Path to the original media file.
        speech_segments: List of speech segments to keep.
        output_path: Path for the output XML file.
        config: Export configuration.
        zoom_events: Optional zoom keyframes to include.

    Returns:
        Path to the generated XML file.
    """
    if config is None:
        config = ExportConfig()

    # Probe media for metadata
    info = probe(filepath)

    # Build XML document
    xmeml = ET.Element("xmeml", version="4")
    sequence = _build_sequence(xmeml, info, speech_segments, config, zoom_events)

    # Write formatted XML
    xml_str = _prettify(xmeml)

    # Prepend DOCTYPE
    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + xml_str

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    return output_path


def _build_sequence(
    parent: ET.Element,
    info: MediaInfo,
    segments: List[TimeSegment],
    config: ExportConfig,
    zoom_events: Optional[List[ZoomEvent]] = None,
) -> ET.Element:
    """Build the <sequence> element with all tracks and clips."""
    seq = ET.SubElement(parent, "sequence", id="sequence-1")

    seq_uuid = str(uuid.uuid4())
    _add_text(seq, "uuid", seq_uuid)
    _add_text(seq, "name", config.sequence_name)

    # Calculate total output duration in frames
    video = info.video
    if video is None:
        # Audio-only file -- use defaults
        fps = 29.97
        timebase = 30
        ntsc = True
        width, height = 1920, 1080
    else:
        fps = video.fps
        timebase = video.effective_timebase
        ntsc = video.ntsc
        width = video.width
        height = video.height

    total_output_frames = sum(
        _seconds_to_frames(seg.duration, fps) for seg in segments
    )
    total_source_frames = _seconds_to_frames(info.duration, fps)

    _add_text(seq, "duration", str(total_output_frames))

    # Rate
    _add_rate(seq, timebase, ntsc)

    # Timecode
    tc = ET.SubElement(seq, "timecode")
    _add_rate(tc, timebase, ntsc)
    _add_text(tc, "string", "00:00:00:00")
    _add_text(tc, "frame", "0")
    _add_text(tc, "displayformat", "NDF")

    # Media container
    media = ET.SubElement(seq, "media")

    # Use a single file ID for all tracks (video + audio).
    # The file element is defined fully ONCE in the first clipitem,
    # and all subsequent clipitems reference it by the same ID.
    file_id = "file-1"
    masterclip_id = "masterclip-1"
    file_defined = [False]  # mutable so _add_clip_items can toggle it

    # ----- Video Track -----
    if config.include_video and info.has_video:
        video_elem = ET.SubElement(media, "video")

        # Video format
        fmt = ET.SubElement(video_elem, "format")
        sc = ET.SubElement(fmt, "samplecharacteristics")
        _add_rate(sc, timebase, ntsc)
        _add_text(sc, "width", str(width))
        _add_text(sc, "height", str(height))
        _add_text(sc, "anamorphic", "FALSE")
        _add_text(sc, "pixelaspectratio", "square")
        _add_text(sc, "fielddominance", "none")

        # Video track with clips
        track = ET.SubElement(video_elem, "track")
        _add_text(track, "locked", "FALSE")
        _add_text(track, "enabled", "TRUE")

        _add_clip_items(
            track, info, segments, fps, timebase, ntsc,
            total_source_frames, stream_type="video",
            file_id=file_id, masterclip_id=masterclip_id,
            file_defined=file_defined,
            zoom_events=zoom_events,
        )

    # ----- Audio Track(s) -----
    if config.include_audio and info.has_audio:
        audio_elem = ET.SubElement(media, "audio")

        # Audio format
        fmt = ET.SubElement(audio_elem, "format")
        sc = ET.SubElement(fmt, "samplecharacteristics")
        _add_text(sc, "depth", str(info.audio.bit_depth))
        _add_text(sc, "samplerate", str(info.audio.sample_rate))

        # Audio output configuration
        outputs = ET.SubElement(audio_elem, "outputs")
        group = ET.SubElement(outputs, "group")
        _add_text(group, "index", "1")
        _add_text(group, "numchannels", str(min(info.audio.channels, 2)))
        _add_text(group, "downmix", "0")
        ch = ET.SubElement(group, "channel")
        _add_text(ch, "index", "1")

        # Create one audio track per channel (up to stereo)
        num_audio_tracks = min(info.audio.channels, 2)
        for ch_idx in range(num_audio_tracks):
            track = ET.SubElement(audio_elem, "track")
            _add_text(track, "locked", "FALSE")
            _add_text(track, "enabled", "TRUE")
            _add_text(track, "outputchannelindex", str(ch_idx + 1))

            _add_clip_items(
                track, info, segments, fps, timebase, ntsc,
                total_source_frames, stream_type="audio",
                channel_index=ch_idx,
                file_id=file_id, masterclip_id=masterclip_id,
                file_defined=file_defined,
            )

    return seq


def _add_clip_items(
    track: ET.Element,
    info: MediaInfo,
    segments: List[TimeSegment],
    fps: float,
    timebase: int,
    ntsc: bool,
    total_source_frames: int,
    stream_type: str = "video",
    channel_index: int = 0,
    file_id: str = "file-1",
    masterclip_id: str = "masterclip-1",
    file_defined: list = None,
    zoom_events: Optional[List[ZoomEvent]] = None,
):
    """Add clipitem elements for each speech segment."""
    if file_defined is None:
        file_defined = [False]

    timeline_pos = 0  # Current position on the output timeline (frames)

    for i, seg in enumerate(segments):
        clip_id = f"clipitem-{stream_type}-{channel_index}-{i + 1}"

        source_in = _seconds_to_frames(seg.start, fps)
        source_out = _seconds_to_frames(seg.end, fps)
        clip_duration = source_out - source_in

        clipitem = ET.SubElement(track, "clipitem", id=clip_id)
        _add_text(clipitem, "masterclipid", masterclip_id)
        _add_text(clipitem, "name", info.filename)

        # Timeline position
        _add_text(clipitem, "enabled", "TRUE")
        _add_text(clipitem, "start", str(timeline_pos))
        _add_text(clipitem, "end", str(timeline_pos + clip_duration))

        # Source in/out
        _add_text(clipitem, "in", str(source_in))
        _add_text(clipitem, "out", str(source_out))

        # File reference: define fully on first occurrence, reference-only thereafter
        if not file_defined[0]:
            _add_file_element(clipitem, file_id, info, timebase, ntsc, total_source_frames)
            file_defined[0] = True
        else:
            ET.SubElement(clipitem, "file", id=file_id)

        # Link video/audio
        if stream_type == "video":
            _add_link(clipitem, clip_id, f"clipitem-audio-0-{i + 1}", stream_type)
        else:
            _add_link(clipitem, clip_id, f"clipitem-video-0-{i + 1}", stream_type)

        # Source channel for audio
        if stream_type == "audio":
            src_ch = ET.SubElement(clipitem, "sourcetrack")
            _add_text(src_ch, "mediatype", "audio")
            _add_text(src_ch, "trackindex", str(channel_index + 1))

        # Add zoom keyframes if this is video and we have zoom events
        if stream_type == "video" and zoom_events:
            _add_zoom_keyframes(clipitem, seg, zoom_events, fps, info)

        timeline_pos += clip_duration


def _add_file_element(
    parent: ET.Element,
    file_id: str,
    info: MediaInfo,
    timebase: int,
    ntsc: bool,
    total_frames: int,
):
    """Add the full <file> element with media description."""
    file_elem = ET.SubElement(parent, "file", id=file_id)
    _add_text(file_elem, "name", info.filename)
    _add_text(file_elem, "pathurl", info.pathurl)

    _add_rate(file_elem, timebase, ntsc)
    _add_text(file_elem, "duration", str(total_frames))

    file_media = ET.SubElement(file_elem, "media")

    if info.has_video:
        vid = ET.SubElement(file_media, "video")
        sc = ET.SubElement(vid, "samplecharacteristics")
        _add_rate(sc, timebase, ntsc)
        _add_text(sc, "width", str(info.video.width))
        _add_text(sc, "height", str(info.video.height))
        _add_text(sc, "anamorphic", "FALSE")
        _add_text(sc, "pixelaspectratio", "square")
        _add_text(sc, "fielddominance", "none")

    if info.has_audio:
        aud = ET.SubElement(file_media, "audio")
        sc = ET.SubElement(aud, "samplecharacteristics")
        _add_text(sc, "depth", str(info.audio.bit_depth))
        _add_text(sc, "samplerate", str(info.audio.sample_rate))


def _add_zoom_keyframes(
    clipitem: ET.Element,
    segment: TimeSegment,
    zoom_events: List[ZoomEvent],
    fps: float,
    info: MediaInfo,
):
    """Add motion/scale keyframes for zoom effects on a clip."""
    # Find zoom events that overlap with this segment
    relevant = []
    for event in zoom_events:
        if event.start < segment.end and event.end > segment.start:
            relevant.append(event)

    if not relevant:
        return

    # Add effect (Motion) with keyframes
    effect = ET.SubElement(clipitem, "effect")
    _add_text(effect, "name", "Basic Motion")
    _add_text(effect, "effectid", "basic")
    _add_text(effect, "effecttype", "motion")

    # Scale parameter with keyframes
    param = ET.SubElement(effect, "parameter")
    _add_text(param, "parameterid", "scale")
    _add_text(param, "name", "Scale")
    _add_text(param, "value", "100")

    for event in relevant:
        for kf in event.to_keyframes():
            # Convert absolute time to clip-relative time
            rel_time = kf.time - segment.start
            if 0 <= rel_time <= segment.duration:
                frame = _seconds_to_frames(rel_time, fps)
                keyframe = ET.SubElement(param, "keyframe")
                _add_text(keyframe, "when", str(frame))
                _add_text(keyframe, "value", str(int(kf.scale * 100)))


def _add_link(clipitem: ET.Element, this_id: str, linked_id: str, stream_type: str):
    """Add a link element to connect video and audio clips."""
    link = ET.SubElement(clipitem, "link")
    _add_text(link, "linkclipref", this_id)
    _add_text(link, "mediatype", stream_type)
    _add_text(link, "trackindex", "1")
    _add_text(link, "clipindex", "1")

    link2 = ET.SubElement(clipitem, "link")
    _add_text(link2, "linkclipref", linked_id)
    other_type = "audio" if stream_type == "video" else "video"
    _add_text(link2, "mediatype", other_type)
    _add_text(link2, "trackindex", "1")
    _add_text(link2, "clipindex", "1")


def _add_rate(parent: ET.Element, timebase: int, ntsc: bool):
    """Add a <rate> element."""
    rate = ET.SubElement(parent, "rate")
    _add_text(rate, "timebase", str(timebase))
    _add_text(rate, "ntsc", "TRUE" if ntsc else "FALSE")


def _add_text(parent: ET.Element, tag: str, text: str):
    """Add a simple text element."""
    elem = ET.SubElement(parent, tag)
    elem.text = str(text)
    return elem


def _seconds_to_frames(seconds: float, fps: float) -> int:
    """Convert seconds to frame count."""
    return round(seconds * fps)


def _prettify(elem: ET.Element) -> str:
    """Pretty-print an XML element."""
    rough_string = ET.tostring(elem, encoding="unicode")
    parsed = minidom.parseString(rough_string)
    pretty = parsed.toprettyxml(indent="  ")
    # Remove the XML declaration minidom adds (we add our own with DOCTYPE)
    lines = pretty.split("\n")
    if lines and lines[0].startswith("<?xml"):
        lines = lines[1:]
    return "\n".join(lines)
