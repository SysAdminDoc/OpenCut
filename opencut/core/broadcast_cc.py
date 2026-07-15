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
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

from opencut.core.caption_interchange import (
    CaptionDocument as InterchangeDocument,
)
from opencut.core.caption_interchange import (
    document_from_items,
    export_caption_document,
    normalize_profile,
)
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
    language: str = ""
    writing_mode: str = ""
    direction: str = ""


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
                    language=str(item.get("language", "")),
                    writing_mode=str(item.get("writing_mode", item.get("writingMode", ""))),
                    direction=str(item.get("direction", "")),
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


def _to_interchange_document(
    cap_data: CaptionData,
    *,
    language: str,
    title: str,
    frame_rate: float,
) -> InterchangeDocument:
    items = []
    for index, caption in enumerate(cap_data.captions, start=1):
        text = caption.text
        if caption.speaker:
            lines = text.split("\n")
            text = "\n".join(
                f"[{caption.speaker}] {line}" if line else f"[{caption.speaker}]"
                for line in lines
            )
        items.append(
            {
                "id": f"c{index}",
                "start": caption.start,
                "end": caption.end,
                "text": text,
                "position": caption.position,
                "style": caption.style if caption.style in {"italic", "bold"} else "default",
                "language": caption.language,
                "writing_mode": caption.writing_mode,
                "direction": caption.direction,
            }
        )
    return document_from_items(
        items,
        language=language,
        title=title,
        frame_rate=frame_rate,
    )


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

    document = _to_interchange_document(
        cap_data,
        language=language,
        title=title,
        frame_rate=frame_rate,
    )
    document.styles["default"].properties.update(
        {
            "fontFamily": "monospaceSansSerif",
            "fontSize": "1c",
            "backgroundColor": "transparent",
        }
    )
    if on_progress:
        on_progress(70, "Validating and writing EBU-TT file...")
    report = export_caption_document(document, output_path, profile="ebu_tt")

    if on_progress:
        on_progress(100, "EBU-TT export complete.")

    return {
        "output_path": output_path,
        "format": "ebu-tt",
        "caption_count": len(cap_data.captions),
        "language": language,
        "file_size_bytes": os.path.getsize(output_path),
        "conformance": report.to_dict(),
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
        profile: "ttml", explicit legacy "imsc1", or "imsc1.3".
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

    normalized_profile = normalize_profile(profile)
    if normalized_profile == "ebu_tt":
        raise ValueError("Use export_ebu_tt() for the EBU-TT profile.")
    document = _to_interchange_document(
        cap_data,
        language=language,
        title=title,
        frame_rate=cap_data.frame_rate,
    )
    if on_progress:
        on_progress(70, "Validating and writing TTML file...")
    report = export_caption_document(
        document,
        output_path,
        profile=normalized_profile,
    )

    if on_progress:
        on_progress(100, "TTML/IMSC1 export complete.")

    return {
        "output_path": output_path,
        "format": f"ttml-{normalized_profile}",
        "caption_count": len(cap_data.captions),
        "language": language,
        "file_size_bytes": os.path.getsize(output_path),
        "conformance": report.to_dict(),
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
