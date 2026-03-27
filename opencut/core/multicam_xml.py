"""
OpenCut Multicam XML Export

Generates Final Cut Pro XML (compatible with Premiere Pro import)
from multicam diarization cut data.
"""

import logging
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

logger = logging.getLogger("opencut")


def _seconds_to_frames(seconds, fps=29.97):
    """Convert seconds to frame count."""
    return int(round(seconds * fps))


def _indent_xml(elem):
    """Pretty-print XML element."""
    rough = ET.tostring(elem, encoding="unicode")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ", encoding=None)


def generate_multicam_xml(
    cuts,
    source_files,
    sequence_name="OpenCut Multicam",
    fps=29.97,
    width=1920,
    height=1080,
    sample_rate=48000,
    output_path=None,
):
    """Generate Premiere-compatible FCP XML from multicam cut data.

    Args:
        cuts: list of dicts with keys:
            - start (float): start time in seconds
            - end (float): end time in seconds
            - speaker (str): speaker identifier
            - track (int): target video track (1-based)
        source_files: dict mapping speaker/track to file path, e.g.
            {"SPEAKER_00": "/path/to/cam1.mp4", "SPEAKER_01": "/path/to/cam2.mp4"}
            OR list of file paths (indexed by track number - 1)
        sequence_name: name for the sequence
        fps: frames per second
        width: frame width
        height: frame height
        sample_rate: audio sample rate
        output_path: if provided, write XML to this file path

    Returns:
        dict with keys:
            - xml (str): the XML string
            - output (str): file path if written, or None
            - cuts_count (int): number of cuts in the sequence
            - duration (float): total duration in seconds
    """
    # Normalize source_files to a speaker->path mapping
    if isinstance(source_files, list):
        file_map = {}
        for i, fp in enumerate(source_files):
            file_map[f"SPEAKER_{i:02d}"] = fp
    else:
        file_map = dict(source_files)

    # Calculate total duration from cuts
    total_duration = 0
    if cuts:
        total_duration = max(c.get("end", 0) for c in cuts)
    total_frames = _seconds_to_frames(total_duration, fps)

    # Determine timebase (integer fps)
    timebase = int(round(fps))
    ntsc = abs(fps - timebase) > 0.001  # True for 29.97, 23.976, etc.

    # Build XML structure
    xmeml = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(xmeml, "sequence")
    ET.SubElement(sequence, "name").text = sequence_name
    ET.SubElement(sequence, "duration").text = str(total_frames)

    rate = ET.SubElement(sequence, "rate")
    ET.SubElement(rate, "timebase").text = str(timebase)
    ET.SubElement(rate, "ntsc").text = "TRUE" if ntsc else "FALSE"

    media = ET.SubElement(sequence, "media")

    # --- Video tracks ---
    video_section = ET.SubElement(media, "video")

    fmt = ET.SubElement(video_section, "format")
    sc = ET.SubElement(fmt, "samplecharacteristics")
    ET.SubElement(sc, "width").text = str(width)
    ET.SubElement(sc, "height").text = str(height)
    v_rate = ET.SubElement(sc, "rate")
    ET.SubElement(v_rate, "timebase").text = str(timebase)
    ET.SubElement(v_rate, "ntsc").text = "TRUE" if ntsc else "FALSE"

    # Group cuts by track
    tracks = {}
    for cut in cuts:
        track_num = cut.get("track", 1)
        if track_num not in tracks:
            tracks[track_num] = []
        tracks[track_num].append(cut)

    # Sort cuts within each track
    for track_num in tracks:
        tracks[track_num].sort(key=lambda c: c.get("start", 0))

    # Collect all unique speakers/tracks for file references
    all_speakers = set()
    for cut in cuts:
        all_speakers.add(cut.get("speaker", "SPEAKER_00"))

    # Create video tracks
    max_track = max(tracks.keys()) if tracks else 1
    for track_num in range(1, max_track + 1):
        track_elem = ET.SubElement(video_section, "track")

        track_cuts = tracks.get(track_num, [])
        for cut in track_cuts:
            speaker = cut.get("speaker", "SPEAKER_00")
            source_path = file_map.get(speaker, "")
            start_sec = cut.get("start", 0)
            end_sec = cut.get("end", 0)

            start_frame = _seconds_to_frames(start_sec, fps)
            end_frame = _seconds_to_frames(end_sec, fps)
            duration_frames = end_frame - start_frame

            clip_item = ET.SubElement(track_elem, "clipitem")
            clip_name = f"{speaker}_{start_frame}-{end_frame}"
            ET.SubElement(clip_item, "name").text = clip_name
            ET.SubElement(clip_item, "duration").text = str(duration_frames)

            c_rate = ET.SubElement(clip_item, "rate")
            ET.SubElement(c_rate, "timebase").text = str(timebase)
            ET.SubElement(c_rate, "ntsc").text = "TRUE" if ntsc else "FALSE"

            ET.SubElement(clip_item, "start").text = str(start_frame)
            ET.SubElement(clip_item, "end").text = str(end_frame)

            # In/out points (source clip timecodes)
            ET.SubElement(clip_item, "in").text = str(start_frame)
            ET.SubElement(clip_item, "out").text = str(end_frame)

            if source_path:
                file_elem = ET.SubElement(clip_item, "file")
                ET.SubElement(file_elem, "name").text = os.path.basename(source_path)
                ET.SubElement(file_elem, "pathurl").text = _path_to_url(source_path)

                media_elem = ET.SubElement(file_elem, "media")
                vid = ET.SubElement(media_elem, "video")
                vid_sc = ET.SubElement(vid, "samplecharacteristics")
                ET.SubElement(vid_sc, "width").text = str(width)
                ET.SubElement(vid_sc, "height").text = str(height)

    # --- Audio tracks ---
    audio_section = ET.SubElement(media, "audio")

    a_fmt = ET.SubElement(audio_section, "format")
    a_sc = ET.SubElement(a_fmt, "samplecharacteristics")
    ET.SubElement(a_sc, "depth").text = "16"
    ET.SubElement(a_sc, "samplerate").text = str(sample_rate)

    # Mirror audio from the active video cuts (linked audio)
    audio_track = ET.SubElement(audio_section, "track")
    for cut in sorted(cuts, key=lambda c: c.get("start", 0)):
        speaker = cut.get("speaker", "SPEAKER_00")
        source_path = file_map.get(speaker, "")
        start_frame = _seconds_to_frames(cut.get("start", 0), fps)
        end_frame = _seconds_to_frames(cut.get("end", 0), fps)
        duration_frames = end_frame - start_frame

        clip_item = ET.SubElement(audio_track, "clipitem")
        ET.SubElement(clip_item, "name").text = f"{speaker}_audio"
        ET.SubElement(clip_item, "duration").text = str(duration_frames)

        c_rate = ET.SubElement(clip_item, "rate")
        ET.SubElement(c_rate, "timebase").text = str(timebase)
        ET.SubElement(c_rate, "ntsc").text = "TRUE" if ntsc else "FALSE"

        ET.SubElement(clip_item, "start").text = str(start_frame)
        ET.SubElement(clip_item, "end").text = str(end_frame)
        ET.SubElement(clip_item, "in").text = str(start_frame)
        ET.SubElement(clip_item, "out").text = str(end_frame)

        if source_path:
            file_elem = ET.SubElement(clip_item, "file")
            ET.SubElement(file_elem, "name").text = os.path.basename(source_path)
            ET.SubElement(file_elem, "pathurl").text = _path_to_url(source_path)

    # Generate XML string
    xml_str = _indent_xml(xmeml)

    # Write to file if requested
    output_file = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
        output_file = output_path
        logger.info("Multicam XML written: %s (%d cuts)", output_path, len(cuts))

    return {
        "xml": xml_str,
        "output": output_file,
        "cuts_count": len(cuts),
        "duration": total_duration,
    }


def _path_to_url(filepath):
    """Convert a local file path to a file:// URL for FCP XML."""
    import urllib.parse
    # Normalize path separators
    path = filepath.replace("\\", "/")
    if not path.startswith("/"):
        # Windows drive letter (C:/...)
        path = "/" + path
    # URI-encode spaces and special chars (but preserve / and :)
    return "file://localhost" + urllib.parse.quote(path, safe="/:")
