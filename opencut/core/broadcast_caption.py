"""
OpenCut Broadcast Caption Format Export

Generate broadcast-standard caption formats: CEA-608 (line 21), CEA-708
(DTVCC), EBU-TT (XML), TTML (W3C), IMSC1 (TTML subset for streaming),
and WebVTT with positioning.

Each format enforces format-specific constraints and validates output
against the relevant specification.
"""

import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_FORMATS = {
    "cea608": {
        "description": "CEA-608 (Line 21 Closed Captioning)",
        "max_chars_per_line": 32,
        "max_lines": 4,
        "max_rows": 15,
        "channels": 2,
    },
    "cea708": {
        "description": "CEA-708 (DTVCC - Digital TV Closed Captioning)",
        "max_chars_per_line": 42,
        "max_lines": 4,
        "service_channels": 8,
    },
    "ebu_tt": {
        "description": "EBU-TT (European Broadcasting Union Timed Text)",
        "max_chars_per_line": 40,
        "max_lines": 2,
    },
    "ttml": {
        "description": "TTML (W3C Timed Text Markup Language)",
        "max_chars_per_line": 42,
        "max_lines": 2,
    },
    "imsc1": {
        "description": "IMSC1 (Internet Media Subtitles and Captions)",
        "max_chars_per_line": 42,
        "max_lines": 2,
    },
    "webvtt_pos": {
        "description": "WebVTT with positioning metadata",
        "max_chars_per_line": 42,
        "max_lines": 2,
    },
}


@dataclass
class CaptionSegment:
    """A single caption/subtitle segment."""

    index: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    row: int = 15
    column: int = 0
    style: str = ""
    channel: int = 1


@dataclass
class CaptionExportResult:
    """Result of a broadcast caption export."""

    output_path: str = ""
    format: str = ""
    segments_exported: int = 0
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Timestamp formatting
# ---------------------------------------------------------------------------
def _seconds_to_smpte(s: float, fps: float = 30.0) -> str:
    """Convert seconds to SMPTE timecode HH:MM:SS:FF."""
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    frames = int((s % 1) * fps)
    return f"{h:02d}:{m:02d}:{sec:02d}:{frames:02d}"


def _seconds_to_ttml(s: float) -> str:
    """Convert seconds to TTML timestamp HH:MM:SS.mmm."""
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def _seconds_to_vtt(s: float) -> str:
    """Convert seconds to VTT timestamp HH:MM:SS.mmm."""
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


# ---------------------------------------------------------------------------
# CEA-608 encoding helpers
# ---------------------------------------------------------------------------
_CEA608_SPECIAL_CHARS = {
    "\u00ae": 0x1930,  # Registered sign
    "\u00b0": 0x1931,  # Degree sign
    "\u00bd": 0x1932,  # 1/2
    "\u00bf": 0x1933,  # Inverted question mark
    "\u2122": 0x1939,  # Trademark
    "\u00a2": 0x193a,  # Cent sign
    "\u00a3": 0x193b,  # Pound sign
    "\u266a": 0x193d,  # Music note
    "\u00e0": 0x1940,  # a grave
    "\u00e8": 0x1942,  # e grave
    "\u00e2": 0x1943,  # a circumflex
    "\u00ea": 0x1944,  # e circumflex
    "\u00ee": 0x1945,  # i circumflex
    "\u00f4": 0x1946,  # o circumflex
    "\u00fb": 0x1947,  # u circumflex
}


def _encode_cea608_char(char: str) -> int:
    """Encode a single character to CEA-608 byte pair.

    Returns 16-bit value representing the two-byte pair, or 0 for
    unsupported characters.
    """
    if char in _CEA608_SPECIAL_CHARS:
        return _CEA608_SPECIAL_CHARS[char]
    code = ord(char)
    # Basic ASCII printable range (0x20-0x7F) maps directly
    if 0x20 <= code <= 0x7F:
        return code
    return 0x20  # Space for unsupported chars


def _encode_cea608_text(text: str) -> List[int]:
    """Encode a text string to CEA-608 byte pair sequence."""
    result: List[int] = []
    for char in text:
        encoded = _encode_cea608_char(char)
        result.append(encoded)
    return result


def _truncate_lines(
    text: str, max_chars: int, max_lines: int,
) -> List[str]:
    """Split and truncate text to fit within line/char constraints."""
    raw_lines = text.split("\n")
    result: List[str] = []
    for line in raw_lines:
        if len(line) > max_chars:
            # Word-wrap long lines
            words = line.split()
            current = ""
            for word in words:
                test = f"{current} {word}".strip() if current else word
                if len(test) <= max_chars:
                    current = test
                else:
                    if current:
                        result.append(current)
                    current = word[:max_chars]
            if current:
                result.append(current)
        else:
            result.append(line)
        if len(result) >= max_lines:
            break
    return result[:max_lines]


# ---------------------------------------------------------------------------
# Format-specific validation
# ---------------------------------------------------------------------------
def _validate_segments(
    segments: List[CaptionSegment],
    fmt_config: Dict,
) -> List[str]:
    """Validate segments against format constraints. Returns error messages."""
    errors: List[str] = []
    max_chars = fmt_config.get("max_chars_per_line", 42)
    max_lines = fmt_config.get("max_lines", 4)

    for seg in segments:
        lines = seg.text.split("\n")
        if len(lines) > max_lines:
            errors.append(
                f"Segment {seg.index}: {len(lines)} lines "
                f"(max {max_lines})"
            )
        for i, line in enumerate(lines):
            if len(line) > max_chars:
                errors.append(
                    f"Segment {seg.index}, line {i + 1}: "
                    f"{len(line)} chars (max {max_chars})"
                )
        if seg.start >= seg.end:
            errors.append(
                f"Segment {seg.index}: invalid timing "
                f"({seg.start} >= {seg.end})"
            )
    return errors


# ---------------------------------------------------------------------------
# CEA-608 export
# ---------------------------------------------------------------------------
def export_cea608(
    segments: List[CaptionSegment],
    output_path: str,
    channel: int = 1,
    on_progress: Optional[Callable] = None,
) -> CaptionExportResult:
    """Export to CEA-608 format (SCC - Scenarist Closed Captions).

    Generates an SCC file with pop-on captions. Each segment is encoded
    as a sequence of control codes and character data.
    """
    fmt = SUPPORTED_FORMATS["cea608"]
    max_chars = fmt["max_chars_per_line"]
    max_lines_limit = fmt["max_lines"]
    errors = _validate_segments(segments, fmt)
    warnings: List[str] = []

    lines_out: List[str] = [
        "Scenarist_SCC V1.0",
        "",
        "",
    ]

    for i, seg in enumerate(segments):
        text_lines = _truncate_lines(seg.text, max_chars, max_lines_limit)
        if not text_lines:
            continue

        tc = _seconds_to_smpte(seg.start)
        # Build byte pairs: Resume Caption Loading + text + End of Caption
        pairs: List[str] = []
        # RCL (Resume Caption Loading) - channel 1
        pairs.append("9420")
        # Position cursor (row 15 for bottom)
        row_code = min(seg.row, 15)
        preamble = 0x1140 + (row_code % 16)
        pairs.append(f"{preamble:04x}")
        # Encode text
        for tl in text_lines:
            encoded = _encode_cea608_text(tl)
            for j in range(0, len(encoded), 2):
                if j + 1 < len(encoded):
                    pair = (encoded[j] << 8) | encoded[j + 1]
                else:
                    pair = (encoded[j] << 8) | 0x80
                pairs.append(f"{pair:04x}")
            # New line between caption lines
            if tl != text_lines[-1]:
                pairs.append("94ad")  # Carriage return

        # EOC (End of Caption) - display
        pairs.append("942f")
        lines_out.append(f"{tc}\t{' '.join(pairs)}")

        # Clear at end time
        tc_end = _seconds_to_smpte(seg.end)
        lines_out.append(f"{tc_end}\t942c")  # EDM (Erase Display)

        if on_progress:
            on_progress(int(((i + 1) / len(segments)) * 90))

    content = "\n".join(lines_out) + "\n"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    if on_progress:
        on_progress(100)

    return CaptionExportResult(
        output_path=output_path,
        format="cea608",
        segments_exported=len(segments),
        validation_errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# CEA-708 export
# ---------------------------------------------------------------------------
def export_cea708(
    segments: List[CaptionSegment],
    output_path: str,
    service_channel: int = 1,
    on_progress: Optional[Callable] = None,
) -> CaptionExportResult:
    """Export to CEA-708 format (MCC - MacCaption Closed Captions).

    Generates an MCC file with DTVCC-encoded captions.
    """
    fmt = SUPPORTED_FORMATS["cea708"]
    max_chars = fmt["max_chars_per_line"]
    max_lines_limit = fmt["max_lines"]
    errors = _validate_segments(segments, fmt)
    warnings: List[str] = []

    if service_channel < 1 or service_channel > 8:
        warnings.append(
            f"Service channel {service_channel} out of range (1-8), using 1"
        )
        service_channel = 1

    lines_out: List[str] = [
        "File Format=MacCaption_MCC V2.0",
        "",
        "////////////////////// Program //////////////////////",
        "",
        "UUID=OpenCut-708-export",
        "Creation Program=OpenCut",
        f"Creation Date={_get_date_str()}",
        f"Creation Time={_get_time_str()}",
        "",
        "////////////////////// Captions //////////////////////",
        "",
    ]

    for i, seg in enumerate(segments):
        text_lines = _truncate_lines(seg.text, max_chars, max_lines_limit)
        if not text_lines:
            continue

        tc = _seconds_to_smpte(seg.start)
        # Service block header
        svc_byte = 0x60 | (service_channel & 0x07)
        # Define window 0 with positioning
        pairs: List[str] = [f"{svc_byte:02x}"]
        # DefineWindow command (0x98 + window_id)
        pairs.append("98")
        # Window attributes (visible, row-locked, anchor)
        pairs.append("00")
        # Anchor vertical (bottom)
        pairs.append("4f")
        # Anchor horizontal (center)
        pairs.append("28")
        # Row/column count
        rows = min(len(text_lines), max_lines_limit)
        pairs.append(f"{rows:02x}")
        pairs.append(f"{max_chars:02x}")

        # Text data
        for tl in text_lines:
            for ch in tl:
                code = ord(ch)
                if 0x20 <= code <= 0x7F:
                    pairs.append(f"{code:02x}")
            if tl != text_lines[-1]:
                pairs.append("0d")  # Carriage return

        lines_out.append(f"{tc}\t{' '.join(pairs)}")

        # Clear at end time
        tc_end = _seconds_to_smpte(seg.end)
        lines_out.append(f"{tc_end}\t{svc_byte:02x} 89 00")  # ClearWindows

        if on_progress:
            on_progress(int(((i + 1) / len(segments)) * 90))

    content = "\n".join(lines_out) + "\n"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    if on_progress:
        on_progress(100)

    return CaptionExportResult(
        output_path=output_path,
        format="cea708",
        segments_exported=len(segments),
        validation_errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# EBU-TT export
# ---------------------------------------------------------------------------
def export_ebu_tt(
    segments: List[CaptionSegment],
    output_path: str,
    lang: str = "en",
    on_progress: Optional[Callable] = None,
) -> CaptionExportResult:
    """Export to EBU-TT (XML format for European broadcasting)."""
    fmt = SUPPORTED_FORMATS["ebu_tt"]
    max_chars = fmt["max_chars_per_line"]
    max_lines_limit = fmt["max_lines"]
    errors = _validate_segments(segments, fmt)
    warnings: List[str] = []

    ns = "http://www.w3.org/ns/ttml"
    ns_ttp = "http://www.w3.org/ns/ttml#parameter"
    ns_tts = "http://www.w3.org/ns/ttml#styling"
    ns_ebuttm = "urn:ebu:tt:metadata"

    root = ET.Element("tt")
    root.set("xmlns", ns)
    root.set("xmlns:ttp", ns_ttp)
    root.set("xmlns:tts", ns_tts)
    root.set("xmlns:ebuttm", ns_ebuttm)
    root.set("xml:lang", lang)
    root.set("ttp:timeBase", "media")

    # Head with styling
    head = ET.SubElement(root, "head")
    styling = ET.SubElement(head, "styling")
    style = ET.SubElement(styling, "style")
    style.set("xml:id", "defaultStyle")
    style.set("tts:fontFamily", "monospaceSansSerif")
    style.set("tts:fontSize", "80%")
    style.set("tts:color", "white")
    style.set("tts:backgroundColor", "black")
    style.set("tts:textAlign", "center")

    layout = ET.SubElement(head, "layout")
    region = ET.SubElement(layout, "region")
    region.set("xml:id", "bottom")
    region.set("tts:origin", "10% 80%")
    region.set("tts:extent", "80% 20%")
    region.set("tts:displayAlign", "after")

    # Metadata
    metadata = ET.SubElement(head, "metadata")
    doc_meta = ET.SubElement(metadata, "ebuttm:documentMetadata")
    ET.SubElement(doc_meta, "ebuttm:documentEbuttVersion").text = "v1.0"

    # Body
    body = ET.SubElement(root, "body")
    div = ET.SubElement(body, "div")
    div.set("style", "defaultStyle")

    for i, seg in enumerate(segments):
        text_lines = _truncate_lines(seg.text, max_chars, max_lines_limit)
        p = ET.SubElement(div, "p")
        p.set("xml:id", f"sub{seg.index}")
        p.set("begin", _seconds_to_ttml(seg.start))
        p.set("end", _seconds_to_ttml(seg.end))
        p.set("region", "bottom")
        for j, tl in enumerate(text_lines):
            if j > 0:
                ET.SubElement(p, "br")
            span = ET.SubElement(p, "span")
            span.text = tl

        if on_progress:
            on_progress(int(((i + 1) / len(segments)) * 90))

    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    if on_progress:
        on_progress(100)

    return CaptionExportResult(
        output_path=output_path,
        format="ebu_tt",
        segments_exported=len(segments),
        validation_errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# TTML export
# ---------------------------------------------------------------------------
def export_ttml(
    segments: List[CaptionSegment],
    output_path: str,
    lang: str = "en",
    on_progress: Optional[Callable] = None,
) -> CaptionExportResult:
    """Export to W3C TTML (Timed Text Markup Language)."""
    fmt = SUPPORTED_FORMATS["ttml"]
    max_chars = fmt["max_chars_per_line"]
    max_lines_limit = fmt["max_lines"]
    errors = _validate_segments(segments, fmt)
    warnings: List[str] = []

    ns = "http://www.w3.org/ns/ttml"
    ns_ttp = "http://www.w3.org/ns/ttml#parameter"
    ns_tts = "http://www.w3.org/ns/ttml#styling"

    root = ET.Element("tt")
    root.set("xmlns", ns)
    root.set("xmlns:ttp", ns_ttp)
    root.set("xmlns:tts", ns_tts)
    root.set("xml:lang", lang)
    root.set("ttp:timeBase", "media")

    head = ET.SubElement(root, "head")
    styling = ET.SubElement(head, "styling")
    style = ET.SubElement(styling, "style")
    style.set("xml:id", "default")
    style.set("tts:fontFamily", "sansSerif")
    style.set("tts:fontSize", "100%")
    style.set("tts:color", "white")
    style.set("tts:textAlign", "center")

    layout = ET.SubElement(head, "layout")
    region = ET.SubElement(layout, "region")
    region.set("xml:id", "bottom")
    region.set("tts:origin", "10% 80%")
    region.set("tts:extent", "80% 20%")
    region.set("tts:displayAlign", "after")

    body = ET.SubElement(root, "body")
    div = ET.SubElement(body, "div")

    for i, seg in enumerate(segments):
        text_lines = _truncate_lines(seg.text, max_chars, max_lines_limit)
        p = ET.SubElement(div, "p")
        p.set("xml:id", f"caption{seg.index}")
        p.set("begin", _seconds_to_ttml(seg.start))
        p.set("end", _seconds_to_ttml(seg.end))
        p.set("region", "bottom")
        p.set("style", "default")
        for j, tl in enumerate(text_lines):
            if j > 0:
                ET.SubElement(p, "br")
            span = ET.SubElement(p, "span")
            span.text = tl

        if on_progress:
            on_progress(int(((i + 1) / len(segments)) * 90))

    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    if on_progress:
        on_progress(100)

    return CaptionExportResult(
        output_path=output_path,
        format="ttml",
        segments_exported=len(segments),
        validation_errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# IMSC1 export
# ---------------------------------------------------------------------------
def export_imsc1(
    segments: List[CaptionSegment],
    output_path: str,
    lang: str = "en",
    on_progress: Optional[Callable] = None,
) -> CaptionExportResult:
    """Export to IMSC1 (subset of TTML for streaming).

    IMSC1 constrains TTML to a profile suitable for internet streaming,
    with specific restrictions on styling and layout.
    """
    fmt = SUPPORTED_FORMATS["imsc1"]
    max_chars = fmt["max_chars_per_line"]
    max_lines_limit = fmt["max_lines"]
    errors = _validate_segments(segments, fmt)
    warnings: List[str] = []

    ns = "http://www.w3.org/ns/ttml"
    ns_ttp = "http://www.w3.org/ns/ttml#parameter"
    ns_tts = "http://www.w3.org/ns/ttml#styling"
    ns_ittp = "http://www.w3.org/ns/ttml/profile/imsc1#parameter"

    root = ET.Element("tt")
    root.set("xmlns", ns)
    root.set("xmlns:ttp", ns_ttp)
    root.set("xmlns:tts", ns_tts)
    root.set("xmlns:ittp", ns_ittp)
    root.set("xml:lang", lang)
    root.set("ttp:timeBase", "media")
    root.set("ttp:cellResolution", "32 15")
    root.set("ittp:aspectRatio", "16 9")

    head = ET.SubElement(root, "head")

    # IMSC1 profile declaration
    styling = ET.SubElement(head, "styling")
    style = ET.SubElement(styling, "style")
    style.set("xml:id", "default")
    style.set("tts:fontFamily", "proportionalSansSerif")
    style.set("tts:fontSize", "100%")
    style.set("tts:color", "white")
    style.set("tts:backgroundColor", "rgba(0,0,0,0.8)")
    style.set("tts:textAlign", "center")

    layout = ET.SubElement(head, "layout")
    region = ET.SubElement(layout, "region")
    region.set("xml:id", "bottom")
    region.set("tts:origin", "10% 80%")
    region.set("tts:extent", "80% 20%")
    region.set("tts:displayAlign", "after")

    body = ET.SubElement(root, "body")
    div = ET.SubElement(body, "div")

    for i, seg in enumerate(segments):
        text_lines = _truncate_lines(seg.text, max_chars, max_lines_limit)
        p = ET.SubElement(div, "p")
        p.set("begin", _seconds_to_ttml(seg.start))
        p.set("end", _seconds_to_ttml(seg.end))
        p.set("region", "bottom")
        p.set("style", "default")
        for j, tl in enumerate(text_lines):
            if j > 0:
                ET.SubElement(p, "br")
            span = ET.SubElement(p, "span")
            span.text = tl

        if on_progress:
            on_progress(int(((i + 1) / len(segments)) * 90))

    tree = ET.ElementTree(root)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    if on_progress:
        on_progress(100)

    return CaptionExportResult(
        output_path=output_path,
        format="imsc1",
        segments_exported=len(segments),
        validation_errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# WebVTT with positioning export
# ---------------------------------------------------------------------------
def export_webvtt_positioned(
    segments: List[CaptionSegment],
    output_path: str,
    on_progress: Optional[Callable] = None,
) -> CaptionExportResult:
    """Export to WebVTT format with positioning cue settings."""
    fmt = SUPPORTED_FORMATS["webvtt_pos"]
    max_chars = fmt["max_chars_per_line"]
    max_lines_limit = fmt["max_lines"]
    errors = _validate_segments(segments, fmt)
    warnings: List[str] = []

    lines_out: List[str] = ["WEBVTT", ""]

    for i, seg in enumerate(segments):
        text_lines = _truncate_lines(seg.text, max_chars, max_lines_limit)
        start_ts = _seconds_to_vtt(seg.start)
        end_ts = _seconds_to_vtt(seg.end)

        # Build cue settings
        cue_settings = []
        # Position based on row
        if seg.row <= 4:
            cue_settings.append("line:10%")
            cue_settings.append("position:50%")
            cue_settings.append("align:center")
        elif seg.row >= 12:
            cue_settings.append("line:90%")
            cue_settings.append("position:50%")
            cue_settings.append("align:center")
        else:
            pct = int((seg.row / 15) * 100)
            cue_settings.append(f"line:{pct}%")
            cue_settings.append("position:50%")
            cue_settings.append("align:center")
        cue_settings.append("size:80%")

        settings_str = " ".join(cue_settings)
        timing_line = f"{start_ts} --> {end_ts} {settings_str}"
        lines_out.append(timing_line)
        lines_out.append("\n".join(text_lines))
        lines_out.append("")

        if on_progress:
            on_progress(int(((i + 1) / len(segments)) * 90))

    content = "\n".join(lines_out)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    if on_progress:
        on_progress(100)

    return CaptionExportResult(
        output_path=output_path,
        format="webvtt_pos",
        segments_exported=len(segments),
        validation_errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Unified export function
# ---------------------------------------------------------------------------
def export_broadcast(
    segments: List[CaptionSegment],
    output_path: str,
    fmt: str = "cea608",
    lang: str = "en",
    service_channel: int = 1,
    on_progress: Optional[Callable] = None,
) -> CaptionExportResult:
    """Export captions to any supported broadcast format.

    Args:
        segments: Caption segments to export.
        output_path: Output file path.
        fmt: Format name (cea608, cea708, ebu_tt, ttml, imsc1, webvtt_pos).
        lang: Language code for XML formats.
        service_channel: Service channel for CEA-708.
        on_progress: Progress callback.

    Returns:
        CaptionExportResult with export details.
    """
    fmt = fmt.lower().strip()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {fmt}. "
            f"Available: {', '.join(SUPPORTED_FORMATS.keys())}"
        )

    exporters = {
        "cea608": lambda: export_cea608(segments, output_path, on_progress=on_progress),
        "cea708": lambda: export_cea708(
            segments, output_path, service_channel, on_progress,
        ),
        "ebu_tt": lambda: export_ebu_tt(segments, output_path, lang, on_progress),
        "ttml": lambda: export_ttml(segments, output_path, lang, on_progress),
        "imsc1": lambda: export_imsc1(segments, output_path, lang, on_progress),
        "webvtt_pos": lambda: export_webvtt_positioned(
            segments, output_path, on_progress,
        ),
    }
    return exporters[fmt]()


def list_formats() -> List[Dict]:
    """Return list of supported broadcast caption formats."""
    result = []
    for name, cfg in SUPPORTED_FORMATS.items():
        result.append({
            "name": name,
            "description": cfg["description"],
            "max_chars_per_line": cfg.get("max_chars_per_line", 42),
            "max_lines": cfg.get("max_lines", 4),
        })
    return result


def segments_from_dicts(dicts: List[Dict]) -> List[CaptionSegment]:
    """Convert list of dicts to CaptionSegment objects."""
    result: List[CaptionSegment] = []
    for i, d in enumerate(dicts):
        result.append(CaptionSegment(
            index=i + 1,
            start=float(d.get("start", 0)),
            end=float(d.get("end", 0)),
            text=str(d.get("text", "")),
            row=int(d.get("row", 15)),
            channel=int(d.get("channel", 1)),
        ))
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_date_str() -> str:
    """Get current date as string."""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d")


def _get_time_str() -> str:
    """Get current time as string."""
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")
