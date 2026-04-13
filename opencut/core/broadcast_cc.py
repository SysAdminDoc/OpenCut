"""
OpenCut Broadcast Closed Caption Export v1.0.0

Export captions in broadcast-standard formats:
  - EBU-TT XML (European Broadcasting Union Timed Text)
  - TTML/IMSC1 (Internet Media Subtitles and Captions)
  - CEA-608/708 embedded via FFmpeg
"""

import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union
from xml.dom import minidom

from opencut.helpers import (
    FFmpegCmd,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Caption Data Types
# ---------------------------------------------------------------------------

@dataclass
class Caption:
    """A single caption/subtitle entry."""
    start: float = 0.0       # Start time in seconds
    end: float = 0.0         # End time in seconds
    text: str = ""
    speaker: str = ""
    style: str = ""           # "italic", "bold", etc.
    position: str = ""        # "top", "bottom", etc.
    align: str = "center"


@dataclass
class CaptionData:
    """Collection of captions with metadata."""
    captions: List[Caption] = field(default_factory=list)
    language: str = "en"
    title: str = ""
    frame_rate: float = 30.0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_captions(data: Union[list, dict, str]) -> CaptionData:
    """Parse caption data from various input formats.

    Accepts:
      - List of dicts with start/end/text keys
      - Dict with 'captions' key containing a list
      - SRT-format string
      - Path to .srt or .json file

    Returns:
        CaptionData object.
    """
    result = CaptionData()

    if isinstance(data, str):
        # Check if it's a file path
        if os.path.isfile(data):
            ext = os.path.splitext(data)[1].lower()
            with open(data, "r", encoding="utf-8") as f:
                content = f.read()
            if ext == ".json":
                return _parse_captions(json.loads(content))
            elif ext in (".srt", ".txt"):
                return _parse_srt(content)
            else:
                return _parse_srt(content)  # try SRT parsing
        else:
            return _parse_srt(data)

    if isinstance(data, dict):
        result.language = data.get("language", "en")
        result.title = data.get("title", "")
        result.frame_rate = float(data.get("frame_rate", 30.0))
        captions_raw = data.get("captions", data.get("segments", []))
        if isinstance(captions_raw, list):
            data = captions_raw
        else:
            return result

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                result.captions.append(Caption(
                    start=float(item.get("start", item.get("start_time", 0))),
                    end=float(item.get("end", item.get("end_time", 0))),
                    text=str(item.get("text", item.get("content", ""))),
                    speaker=str(item.get("speaker", "")),
                    style=str(item.get("style", "")),
                    position=str(item.get("position", "")),
                    align=str(item.get("align", "center")),
                ))

    return result


def _parse_srt(srt_text: str) -> CaptionData:
    """Parse SRT subtitle format into CaptionData."""
    result = CaptionData()

    # Split on double newlines (subtitle blocks)
    blocks = re.split(r"\n\s*\n", srt_text.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        # Line 1: index (skip)
        # Line 2: timestamps
        # Lines 3+: text
        time_line = lines[1].strip()
        time_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            time_line
        )
        if not time_match:
            continue

        g = time_match.groups()
        start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
        end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000

        text = "\n".join(lines[2:]).strip()

        result.captions.append(Caption(start=start, end=end, text=text))

    return result


# ---------------------------------------------------------------------------
# Time Formatting
# ---------------------------------------------------------------------------

def _secs_to_timecode(seconds: float, frame_rate: float = 30.0) -> str:
    """Convert seconds to HH:MM:SS:FF timecode."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    f = int((seconds % 1) * frame_rate)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def _secs_to_media_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm media time."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# EBU-TT Export
# ---------------------------------------------------------------------------

def export_ebu_tt(
    captions: Union[list, dict, str],
    output_path: str,
    language: str = "en",
    title: str = "",
    frame_rate: float = 30.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Export captions as EBU-TT (EBU Timed Text) XML.

    EBU-TT is the European Broadcasting Union profile of TTML used
    in broadcast and online delivery.

    Args:
        captions: Caption data (list, dict, SRT string, or file path).
        output_path: Output .xml file path.
        language: Caption language code (e.g. "en", "de").
        title: Program title.
        frame_rate: Video frame rate for timecode conversion.
        on_progress: Optional callback (percent, message).

    Returns:
        Dict with output_path, caption count, format info.

    Raises:
        ValueError: If no valid captions found.
    """
    if on_progress:
        on_progress(10, "Parsing caption data...")

    cap_data = _parse_captions(captions)
    if not cap_data.captions:
        raise ValueError("No valid captions found in input data.")

    language = language or cap_data.language or "en"
    title = title or cap_data.title or "Untitled"
    frame_rate = frame_rate or cap_data.frame_rate or 30.0

    if on_progress:
        on_progress(30, f"Generating EBU-TT XML ({len(cap_data.captions)} captions)...")

    # Build EBU-TT XML
    EBUTT_NS = "urn:ebu:tt:metadata"
    TT_NS = "http://www.w3.org/ns/ttml"
    TTP_NS = "http://www.w3.org/ns/ttml#parameter"
    TTS_NS = "http://www.w3.org/ns/ttml#styling"
    XML_NS = "http://www.w3.org/XML/1998/namespace"

    # Register namespaces
    ET.register_namespace("", TT_NS)
    ET.register_namespace("ttp", TTP_NS)
    ET.register_namespace("tts", TTS_NS)
    ET.register_namespace("ebuttm", EBUTT_NS)
    ET.register_namespace("xml", XML_NS)

    tt = ET.Element(f"{{{TT_NS}}}tt", attrib={
        f"{{{XML_NS}}}lang": language,
        f"{{{TTP_NS}}}timeBase": "media",
        f"{{{TTP_NS}}}frameRate": str(int(frame_rate)),
        f"{{{TTP_NS}}}cellResolution": "50 30",
    })

    # Head
    head = ET.SubElement(tt, "head")
    metadata = ET.SubElement(head, f"{{{EBUTT_NS}}}documentMetadata")
    ET.SubElement(metadata, f"{{{EBUTT_NS}}}documentEbuttVersion").text = "v1.0"
    if title:
        ET.SubElement(metadata, f"{{{EBUTT_NS}}}documentOriginalProgrammeTitle").text = title

    # Styling
    styling = ET.SubElement(head, "styling")
    ET.SubElement(styling, "style", attrib={
        f"{{{XML_NS}}}id": "defaultStyle",
        f"{{{TTS_NS}}}fontFamily": "monospaceSansSerif",
        f"{{{TTS_NS}}}fontSize": "1c",
        f"{{{TTS_NS}}}textAlign": "center",
        f"{{{TTS_NS}}}color": "white",
        f"{{{TTS_NS}}}backgroundColor": "transparent",
    })

    # Layout
    layout = ET.SubElement(head, "layout")
    ET.SubElement(layout, "region", attrib={
        f"{{{XML_NS}}}id": "bottom",
        f"{{{TTS_NS}}}origin": "10% 80%",
        f"{{{TTS_NS}}}extent": "80% 15%",
        f"{{{TTS_NS}}}displayAlign": "after",
        f"{{{TTS_NS}}}writingMode": "lrtb",
    })
    ET.SubElement(layout, "region", attrib={
        f"{{{XML_NS}}}id": "top",
        f"{{{TTS_NS}}}origin": "10% 5%",
        f"{{{TTS_NS}}}extent": "80% 15%",
        f"{{{TTS_NS}}}displayAlign": "before",
        f"{{{TTS_NS}}}writingMode": "lrtb",
    })

    # Body
    body = ET.SubElement(tt, "body")
    div = ET.SubElement(body, "div", attrib={
        "style": "defaultStyle",
    })

    for i, cap in enumerate(cap_data.captions):
        begin = _secs_to_media_time(cap.start)
        end = _secs_to_media_time(cap.end)
        region = "top" if cap.position == "top" else "bottom"

        p_attrib = {
            f"{{{XML_NS}}}id": f"sub{i+1}",
            "begin": begin,
            "end": end,
            "region": region,
        }
        p = ET.SubElement(div, "p", attrib=p_attrib)

        # Handle multi-line text with <br/>
        text_lines = cap.text.split("\n")
        for j, line in enumerate(text_lines):
            if cap.speaker:
                span = ET.SubElement(p, "span")
                span.text = f"[{cap.speaker}] {line}"
            else:
                if j == 0:
                    p.text = line
                else:
                    br = ET.SubElement(p, "br")
                    br.tail = line

    if on_progress:
        on_progress(70, "Writing EBU-TT file...")

    # Write with pretty printing
    xml_str = ET.tostring(tt, encoding="unicode", xml_declaration=False)
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ", encoding="UTF-8")

    with open(output_path, "wb") as f:
        f.write(pretty)

    if on_progress:
        on_progress(100, "EBU-TT export complete.")

    return {
        "output_path": output_path,
        "format": "ebu-tt",
        "caption_count": len(cap_data.captions),
        "language": language,
        "file_size_bytes": os.path.getsize(output_path),
    }


# ---------------------------------------------------------------------------
# TTML / IMSC1 Export
# ---------------------------------------------------------------------------

def export_ttml(
    captions: Union[list, dict, str],
    output_path: str,
    language: str = "en",
    title: str = "",
    profile: str = "imsc1",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Export captions as TTML (Timed Text Markup Language) / IMSC1.

    IMSC1 is the profile of TTML required by many streaming platforms.

    Args:
        captions: Caption data (list, dict, SRT string, or file path).
        output_path: Output .ttml or .xml file path.
        language: Caption language code.
        title: Program title for metadata.
        profile: TTML profile — "imsc1" or "ttml".
        on_progress: Optional callback (percent, message).

    Returns:
        Dict with output_path, caption count, format info.
    """
    if on_progress:
        on_progress(10, "Parsing caption data...")

    cap_data = _parse_captions(captions)
    if not cap_data.captions:
        raise ValueError("No valid captions found in input data.")

    language = language or cap_data.language or "en"

    if on_progress:
        on_progress(30, f"Generating TTML/IMSC1 XML ({len(cap_data.captions)} captions)...")

    TT_NS = "http://www.w3.org/ns/ttml"
    TTP_NS = "http://www.w3.org/ns/ttml#parameter"
    TTS_NS = "http://www.w3.org/ns/ttml#styling"
    ITTP_NS = "http://www.w3.org/ns/ttml/profile/imsc1#parameter"
    XML_NS = "http://www.w3.org/XML/1998/namespace"

    ET.register_namespace("", TT_NS)
    ET.register_namespace("ttp", TTP_NS)
    ET.register_namespace("tts", TTS_NS)
    ET.register_namespace("ittp", ITTP_NS)

    attribs = {
        "xmlns": TT_NS,
        "xmlns:ttp": TTP_NS,
        "xmlns:tts": TTS_NS,
        f"{{{XML_NS}}}lang": language,
        f"{{{TTP_NS}}}timeBase": "media",
    }

    if profile == "imsc1":
        attribs["xmlns:ittp"] = ITTP_NS
        attribs[f"{{{TTP_NS}}}profile"] = "http://www.w3.org/ns/ttml/profile/imsc1/text"

    tt = ET.Element("tt", attrib=attribs)

    # Head with styling
    head = ET.SubElement(tt, "head")

    styling = ET.SubElement(head, "styling")
    ET.SubElement(styling, "style", attrib={
        f"{{{XML_NS}}}id": "default",
        f"{{{TTS_NS}}}fontFamily": "proportionalSansSerif",
        f"{{{TTS_NS}}}fontSize": "100%",
        f"{{{TTS_NS}}}textAlign": "center",
        f"{{{TTS_NS}}}color": "white",
        f"{{{TTS_NS}}}backgroundColor": "rgba(0,0,0,0.8)",
    })

    if any(c.style == "italic" for c in cap_data.captions):
        ET.SubElement(styling, "style", attrib={
            f"{{{XML_NS}}}id": "italic",
            f"{{{TTS_NS}}}fontStyle": "italic",
        })

    # Layout
    layout = ET.SubElement(head, "layout")
    ET.SubElement(layout, "region", attrib={
        f"{{{XML_NS}}}id": "bottom",
        f"{{{TTS_NS}}}origin": "10% 80%",
        f"{{{TTS_NS}}}extent": "80% 15%",
        f"{{{TTS_NS}}}displayAlign": "after",
    })
    ET.SubElement(layout, "region", attrib={
        f"{{{XML_NS}}}id": "top",
        f"{{{TTS_NS}}}origin": "10% 5%",
        f"{{{TTS_NS}}}extent": "80% 15%",
        f"{{{TTS_NS}}}displayAlign": "before",
    })

    # Body
    body = ET.SubElement(tt, "body")
    div = ET.SubElement(body, "div", attrib={"style": "default"})

    for i, cap in enumerate(cap_data.captions):
        begin = _secs_to_media_time(cap.start)
        end = _secs_to_media_time(cap.end)
        region = "top" if cap.position == "top" else "bottom"

        p_attrib = {
            f"{{{XML_NS}}}id": f"c{i+1}",
            "begin": begin,
            "end": end,
            "region": region,
        }
        if cap.style == "italic":
            p_attrib["style"] = "italic"

        p = ET.SubElement(div, "p", attrib=p_attrib)

        text_lines = cap.text.split("\n")
        for j, line in enumerate(text_lines):
            if j == 0:
                p.text = line
            else:
                br = ET.SubElement(p, "br")
                br.tail = line

    if on_progress:
        on_progress(70, "Writing TTML file...")

    xml_str = ET.tostring(tt, encoding="unicode", xml_declaration=False)
    declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(declaration)
        f.write(xml_str)

    if on_progress:
        on_progress(100, "TTML/IMSC1 export complete.")

    return {
        "output_path": output_path,
        "format": f"ttml-{profile}",
        "caption_count": len(cap_data.captions),
        "language": language,
        "file_size_bytes": os.path.getsize(output_path),
    }


# ---------------------------------------------------------------------------
# CEA-608/708 Embedding
# ---------------------------------------------------------------------------

def _captions_to_srt(cap_data: CaptionData) -> str:
    """Convert CaptionData to SRT string for FFmpeg embedding."""
    lines = []
    for i, cap in enumerate(cap_data.captions):
        start = _secs_to_media_time(cap.start).replace(".", ",")
        end = _secs_to_media_time(cap.end).replace(".", ",")
        lines.append(str(i + 1))
        lines.append(f"{start} --> {end}")
        lines.append(cap.text)
        lines.append("")
    return "\n".join(lines)


def embed_cea608(
    video_path: str,
    captions: Union[list, dict, str],
    output_path: str = "",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Embed CEA-608/708 closed captions into a video file via FFmpeg.

    Converts caption data to SRT format, then uses FFmpeg to embed
    as a subtitle stream (mov_text for MP4, or SRT for MKV).

    Args:
        video_path: Path to the input video.
        captions: Caption data (list, dict, SRT string, or file path).
        output_path: Output video path. Auto-generated if empty.
        on_progress: Optional callback (percent, message).

    Returns:
        Dict with output_path and embedding info.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If no valid captions found.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if on_progress:
        on_progress(10, "Parsing caption data...")

    cap_data = _parse_captions(captions)
    if not cap_data.captions:
        raise ValueError("No valid captions found in input data.")

    if not output_path:
        from opencut.helpers import output_path as _output_path
        output_path = _output_path(video_path, "cc")

    if on_progress:
        on_progress(20, "Preparing subtitle data...")

    # Write captions to temporary SRT file
    import tempfile
    srt_content = _captions_to_srt(cap_data)
    srt_file = tempfile.NamedTemporaryFile(
        suffix=".srt", mode="w", encoding="utf-8", delete=False
    )
    try:
        srt_file.write(srt_content)
        srt_file.close()

        if on_progress:
            on_progress(40, "Embedding closed captions...")

        ext = os.path.splitext(output_path)[1].lower()
        sub_codec = "mov_text" if ext in (".mp4", ".m4v", ".mov") else "srt"

        cmd = (FFmpegCmd()
               .input(video_path)
               .input(srt_file.name)
               .map("0:v", "0:a?", "1:s")
               .video_codec("copy", pix_fmt=None)
               .audio_codec("copy")
               .option("c:s", sub_codec)
               .option("metadata:s:s:0", f"language={cap_data.language}")
               .faststart()
               .output(output_path)
               .build())

        try:
            run_ffmpeg(cmd, timeout=3600)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to embed captions: {e}")

    finally:
        try:
            os.unlink(srt_file.name)
        except OSError:
            pass

    if on_progress:
        on_progress(100, "Closed captions embedded.")

    return {
        "output_path": output_path,
        "caption_count": len(cap_data.captions),
        "subtitle_codec": sub_codec,
        "language": cap_data.language,
    }
