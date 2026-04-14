"""
OpenCut Essential Graphics Caption Export

Generate captions in formats compatible with Premiere Pro:
- Premiere-compatible XML sequences with text layers
- SRT for Premiere caption track import
- After Effects-compatible JSON for text layer automation
- Style+data JSON for CEP panel ExtendScript integration

Since MOGRT is a binary format, we export structured data that
Premiere and After Effects can consume via their scripting APIs.
Inspired by AutoCut's native Premiere integration.
"""

import json
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MOGRTExportResult:
    """Result of an Essential Graphics export."""
    output_path: str = ""
    format: str = ""
    caption_count: int = 0
    duration: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# SRT export
# ---------------------------------------------------------------------------
def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp ``HH:MM:SS,mmm``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_as_srt_for_premiere(
    captions: List[Dict],
    out_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Export captions as SRT for Premiere Pro caption track import.

    Args:
        captions: List of ``{"text": str, "start": float, "end": float}``.
        out_path: Output file path. Auto-generated if omitted.
        on_progress: Progress callback ``(pct, msg)``.

    Returns:
        Path to the generated SRT file.
    """
    if not captions:
        raise ValueError("captions must contain at least one entry")

    if not out_path:
        out_path = os.path.join(
            os.path.expanduser("~"), ".opencut", "export_captions.srt"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if on_progress:
        on_progress(10, "Generating SRT file...")

    lines: List[str] = []
    total = len(captions)
    for i, cap in enumerate(captions):
        start = _format_srt_time(float(cap.get("start", 0)))
        end = _format_srt_time(float(cap.get("end", 0)))
        text = cap.get("text", "").strip()
        lines.append(f"{i + 1}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

        if on_progress and total > 0:
            on_progress(10 + int(80 * (i + 1) / total), f"Caption {i + 1}/{total}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if on_progress:
        on_progress(100, "SRT export complete")

    return out_path


# ---------------------------------------------------------------------------
# Premiere Pro XML export
# ---------------------------------------------------------------------------
def generate_premiere_caption_xml(
    captions: List[Dict],
    style: Optional[Dict] = None,
    fps: float = 30.0,
    width: int = 1920,
    height: int = 1080,
) -> str:
    """
    Generate Premiere Pro XML with caption markers/text layers.

    Args:
        captions: List of ``{"text": str, "start": float, "end": float}``.
        style: Optional style dict with ``font``, ``size``, ``color`` keys.
        fps: Timeline frame rate.
        width: Sequence width.
        height: Sequence height.

    Returns:
        XML string.
    """
    if not captions:
        raise ValueError("captions must contain at least one entry")

    style = style or {}
    font = style.get("font", "Arial")
    font_size = style.get("size", 48)
    font_color = style.get("color", "#FFFFFF")

    # Compute total duration from captions
    max_end = max(float(c.get("end", 0)) for c in captions)
    total_frames = int(max_end * fps) + int(fps)  # pad 1 second

    root = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(root, "sequence")
    ET.SubElement(sequence, "name").text = "OpenCut Captions"
    ET.SubElement(sequence, "duration").text = str(total_frames)

    rate_elem = ET.SubElement(sequence, "rate")
    ET.SubElement(rate_elem, "timebase").text = str(int(fps))
    ET.SubElement(rate_elem, "ntsc").text = "FALSE"

    media = ET.SubElement(sequence, "media")
    video = ET.SubElement(media, "video")
    video_format = ET.SubElement(video, "format")
    sample_chars = ET.SubElement(video_format, "samplecharacteristics")
    ET.SubElement(sample_chars, "width").text = str(width)
    ET.SubElement(sample_chars, "height").text = str(height)

    track = ET.SubElement(video, "track")

    for i, cap in enumerate(captions):
        text = cap.get("text", "")
        start_frames = int(float(cap.get("start", 0)) * fps)
        end_frames = int(float(cap.get("end", 0)) * fps)

        clip_item = ET.SubElement(track, "clipitem", id=f"caption_{i + 1}")
        ET.SubElement(clip_item, "name").text = f"Caption {i + 1}"
        ET.SubElement(clip_item, "start").text = str(start_frames)
        ET.SubElement(clip_item, "end").text = str(end_frames)
        ET.SubElement(clip_item, "in").text = "0"
        ET.SubElement(clip_item, "out").text = str(end_frames - start_frames)

        # Add text metadata via effect
        effect = ET.SubElement(clip_item, "effect")
        ET.SubElement(effect, "name").text = "Text"
        ET.SubElement(effect, "effectid").text = "Text"
        ET.SubElement(effect, "effecttype").text = "generator"

        # Text parameter
        param_text = ET.SubElement(effect, "parameter")
        ET.SubElement(param_text, "parameterid").text = "str"
        ET.SubElement(param_text, "name").text = "Text"
        ET.SubElement(param_text, "value").text = text

        # Font parameter
        param_font = ET.SubElement(effect, "parameter")
        ET.SubElement(param_font, "parameterid").text = "font"
        ET.SubElement(param_font, "name").text = "Font"
        ET.SubElement(param_font, "value").text = font

        # Font size parameter
        param_size = ET.SubElement(effect, "parameter")
        ET.SubElement(param_size, "parameterid").text = "fontsize"
        ET.SubElement(param_size, "name").text = "Font Size"
        ET.SubElement(param_size, "value").text = str(font_size)

        # Color parameter
        param_color = ET.SubElement(effect, "parameter")
        ET.SubElement(param_color, "parameterid").text = "fontcolor"
        ET.SubElement(param_color, "name").text = "Font Color"
        ET.SubElement(param_color, "value").text = font_color

    # Markers
    markers = ET.SubElement(sequence, "markers")
    for i, cap in enumerate(captions):
        marker = ET.SubElement(markers, "marker")
        ET.SubElement(marker, "name").text = cap.get("text", "")[:50]
        ET.SubElement(marker, "in").text = str(
            int(float(cap.get("start", 0)) * fps)
        )
        ET.SubElement(marker, "out").text = str(
            int(float(cap.get("end", 0)) * fps)
        )

    return ET.tostring(root, encoding="unicode", xml_declaration=True)


# ---------------------------------------------------------------------------
# MOGRT-equivalent export (structured JSON for CEP panel)
# ---------------------------------------------------------------------------
def export_as_mogrt_data(
    captions: List[Dict],
    style: Optional[Dict] = None,
    out_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> MOGRTExportResult:
    """
    Export caption data as a structured JSON file for Premiere CEP panel
    or ExtendScript import as Essential Graphics.

    Args:
        captions: List of ``{"text": str, "start": float, "end": float}``.
        style: Optional style dict.
        out_path: Output path (auto-generated if omitted).
        on_progress: Progress callback ``(pct, msg)``.

    Returns:
        :class:`MOGRTExportResult`.
    """
    if not captions:
        raise ValueError("captions must contain at least one entry")

    style = style or {}
    if not out_path:
        out_path = os.path.join(
            os.path.expanduser("~"), ".opencut", "essential_graphics.json"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if on_progress:
        on_progress(10, "Building Essential Graphics data...")

    max_end = max(float(c.get("end", 0)) for c in captions)

    export_data = {
        "format": "opencut_essential_graphics",
        "version": "1.0",
        "type": "caption_sequence",
        "style": {
            "font_family": style.get("font", "Arial"),
            "font_size": style.get("size", 48),
            "font_color": style.get("color", "#FFFFFF"),
            "background_color": style.get("background", ""),
            "position": style.get("position", "bottom"),
            "animation": style.get("animation", "none"),
        },
        "captions": [],
        "total_duration": max_end,
        "caption_count": len(captions),
    }

    total = len(captions)
    for i, cap in enumerate(captions):
        entry = {
            "index": i + 1,
            "text": cap.get("text", ""),
            "start": float(cap.get("start", 0)),
            "end": float(cap.get("end", 0)),
            "duration": float(cap.get("end", 0)) - float(cap.get("start", 0)),
        }
        # Include per-word timing if available
        if "words" in cap:
            entry["words"] = cap["words"]
        export_data["captions"].append(entry)

        if on_progress and total > 0:
            on_progress(
                10 + int(60 * (i + 1) / total),
                f"Processing caption {i + 1}/{total}",
            )

    if on_progress:
        on_progress(80, "Writing export file...")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    if on_progress:
        on_progress(100, "Essential Graphics export complete")

    return MOGRTExportResult(
        output_path=out_path,
        format="opencut_essential_graphics_json",
        caption_count=len(captions),
        duration=max_end,
    )


# ---------------------------------------------------------------------------
# After Effects JSON export
# ---------------------------------------------------------------------------
def export_as_ae_json(
    captions: List[Dict],
    style: Optional[Dict] = None,
    out_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Generate an After Effects-compatible JSON mapping captions to
    text layers with timing.

    Args:
        captions: List of ``{"text": str, "start": float, "end": float}``.
        style: Optional style dict.
        out_path: Output path (auto-generated if omitted).
        on_progress: Progress callback ``(pct, msg)``.

    Returns:
        Path to the generated JSON file.
    """
    if not captions:
        raise ValueError("captions must contain at least one entry")

    style = style or {}
    if not out_path:
        out_path = os.path.join(
            os.path.expanduser("~"), ".opencut", "ae_captions.json"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if on_progress:
        on_progress(10, "Building After Effects caption data...")

    max_end = max(float(c.get("end", 0)) for c in captions)

    ae_data = {
        "format": "opencut_ae_captions",
        "version": "1.0",
        "composition": {
            "width": style.get("width", 1920),
            "height": style.get("height", 1080),
            "fps": style.get("fps", 30.0),
            "duration": max_end,
        },
        "text_layer_style": {
            "font": style.get("font", "Arial"),
            "fontSize": style.get("size", 48),
            "fillColor": style.get("color", "#FFFFFF"),
            "strokeColor": style.get("outline", ""),
            "strokeWidth": style.get("stroke_width", 0),
            "justification": "center",
        },
        "layers": [],
    }

    total = len(captions)
    for i, cap in enumerate(captions):
        layer = {
            "index": i + 1,
            "name": f"Caption_{i + 1}",
            "text": cap.get("text", ""),
            "inPoint": float(cap.get("start", 0)),
            "outPoint": float(cap.get("end", 0)),
            "position": [
                style.get("width", 1920) / 2,
                style.get("height", 1080) * 0.85,
            ],
        }
        ae_data["layers"].append(layer)

        if on_progress and total > 0:
            on_progress(
                10 + int(80 * (i + 1) / total),
                f"Layer {i + 1}/{total}",
            )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ae_data, f, indent=2, ensure_ascii=False)

    if on_progress:
        on_progress(100, "After Effects export complete")

    return out_path
