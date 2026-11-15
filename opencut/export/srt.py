"""SRT and VTT subtitle export.

Generates standard subtitle files from transcription results. SRT output is
UTF-8 without a BOM by default; pass ``legacy_windows_bom=True`` only for
legacy Windows players that require a UTF-8 signature.
"""

from typing import List, Optional

from ..core.asr_provenance import provenance_to_dict
from ..core.caption_line_breaks import caption_lines, timed_caption_chunks
from ..core.captions import CaptionSegment, TranscriptionResult, caption_segment_to_dict

UTF8_NO_BOM = "utf-8"
UTF8_WITH_BOM = "utf-8-sig"
UTF8_BOM = b"\xef\xbb\xbf"


def export_srt(
    result: TranscriptionResult,
    output_path: str,
    max_line_length: int = 42,
    max_lines: int = 2,
    *,
    legacy_windows_bom: bool = False,
    encoding: Optional[str] = None,
) -> str:
    """
    Export transcription as SRT (SubRip) subtitle file.

    Args:
        result: Transcription result from Whisper.
        output_path: Output file path.
        max_line_length: Maximum characters per line.
        max_lines: Maximum lines per subtitle entry.
        legacy_windows_bom: If True, write UTF-8 with BOM for old Windows players.
        encoding: Optional explicit text encoding. Defaults to UTF-8 without BOM
            unless legacy_windows_bom is true.

    Returns:
        Path to the generated SRT file.
    """
    entries = _split_segments(result.segments, max_line_length, max_lines)

    lines = []
    for i, entry in enumerate(entries, 1):
        start_tc = _format_srt_time(entry.start)
        end_tc = _format_srt_time(entry.end)
        text = _wrap_text(entry.text.strip(), max_line_length)

        lines.append(f"{i}")
        lines.append(f"{start_tc} --> {end_tc}")
        lines.append(text)
        lines.append("")  # Blank line separator

    write_srt_text(output_path, "\n".join(lines), legacy_windows_bom=legacy_windows_bom, encoding=encoding)

    return output_path


def export_vtt(
    result: TranscriptionResult,
    output_path: str,
    max_line_length: int = 42,
    max_lines: int = 2,
) -> str:
    """
    Export transcription as WebVTT subtitle file.

    Args:
        result: Transcription result from Whisper.
        output_path: Output file path.
        max_line_length: Maximum characters per line.
        max_lines: Maximum lines per subtitle entry.

    Returns:
        Path to the generated VTT file.
    """
    entries = _split_segments(result.segments, max_line_length, max_lines)

    lines = ["WEBVTT", ""]
    for i, entry in enumerate(entries, 1):
        start_tc = _format_vtt_time(entry.start)
        end_tc = _format_vtt_time(entry.end)
        text = _wrap_text(entry.text.strip(), max_line_length)

        lines.append(f"{i}")
        lines.append(f"{start_tc} --> {end_tc}")
        lines.append(text)
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def export_json(
    result: TranscriptionResult,
    output_path: str,
) -> str:
    """
    Export transcription as JSON with full word-level data.

    Args:
        result: Transcription result from Whisper.
        output_path: Output file path.

    Returns:
        Path to the generated JSON file.
    """
    import json

    segments = list(getattr(result, "segments", []) or [])
    text = getattr(result, "text", " ".join(str(getattr(seg, "text", "")).strip() for seg in segments))
    word_count = getattr(result, "word_count", sum(len(getattr(seg, "words", []) or []) for seg in segments))

    data = {
        "language": getattr(result, "language", "en"),
        "duration": getattr(result, "duration", 0.0),
        "language_confidence": getattr(result, "language_confidence", 1.0),
        "text": text,
        "word_count": word_count,
        "segments": [
            caption_segment_to_dict(seg, include_words=True)
            for seg in segments
        ],
        "human_review_recommended": bool(getattr(result, "human_review_recommended", False)),
        "review_segment_count": int(getattr(result, "review_segment_count", 0)),
        "asr_provenance": provenance_to_dict(getattr(result, "provenance", None)),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path


def srt_text_encoding(*, legacy_windows_bom: bool = False, encoding: Optional[str] = None) -> str:
    """Resolve the SRT writer encoding for default/legacy output."""
    if encoding:
        normalised = encoding.strip().lower().replace("_", "-")
        if normalised in {UTF8_NO_BOM, "utf8"}:
            return UTF8_NO_BOM
        if normalised in {UTF8_WITH_BOM, "utf8-sig", "utf-8-bom"}:
            return UTF8_WITH_BOM
        raise ValueError("SRT export supports only utf-8 or utf-8-sig encodings")
    return UTF8_WITH_BOM if legacy_windows_bom else UTF8_NO_BOM


def write_srt_text(
    output_path: str,
    text: str,
    *,
    legacy_windows_bom: bool = False,
    encoding: Optional[str] = None,
) -> str:
    """Write SRT text with the repository's BOM policy."""
    resolved = srt_text_encoding(legacy_windows_bom=legacy_windows_bom, encoding=encoding)
    with open(output_path, "w", encoding=resolved, newline="\n") as f:
        f.write(text)
    return output_path


def has_utf8_bom(path: str) -> bool:
    """Return True when a file starts with a UTF-8 BOM."""
    with open(path, "rb") as f:
        return f.read(3) == UTF8_BOM


def export_ass(
    result: TranscriptionResult,
    output_path: str,
    style_name: str = "Default",
    font_name: str = "Arial",
    font_size: int = 48,
    primary_color: str = "&H00FFFFFF",
    highlight_color: str = "&H0000FFFF",
    outline_color: str = "&H00000000",
    back_color: str = "&H80000000",
    outline_width: int = 2,
    shadow_depth: int = 1,
    alignment: int = 2,
    margin_v: int = 60,
    video_width: int = 1920,
    video_height: int = 1080,
    karaoke: bool = True,
) -> str:
    """
    Export transcription as ASS (Advanced SubStation Alpha) subtitle file
    with optional word-by-word karaoke timing using \\kf tags.

    ASS color format: &HAABBGGRR (hex, alpha-blue-green-red)

    Args:
        result: Transcription result from Whisper.
        output_path: Output file path.
        style_name: Name for the subtitle style.
        font_name: Font face name.
        font_size: Font size in pixels.
        primary_color: ASS color for normal text.
        highlight_color: ASS color for karaoke highlight fill.
        outline_color: ASS color for text outline.
        back_color: ASS color for shadow/background.
        outline_width: Outline thickness.
        shadow_depth: Shadow distance.
        alignment: Numpad-style alignment (2=bottom center).
        margin_v: Vertical margin in pixels.
        video_width: Video width for PlayResX.
        video_height: Video height for PlayResY.
        karaoke: If True, use \\kf tags for word-by-word highlight.

    Returns:
        Path to the generated ASS file.
    """
    # ASS header
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {video_width}",
        f"PlayResY: {video_height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "Title: OpenCut Captions",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: {style_name},{font_name},{font_size},{primary_color},{highlight_color},{outline_color},{back_color},-1,0,0,0,100,100,0,0,1,{outline_width},{shadow_depth},{alignment},40,40,{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    entries = _split_segments(result.segments, 42, 2)

    for entry in entries:
        start_tc = _format_ass_time(entry.start)
        end_tc = _format_ass_time(entry.end)

        if karaoke and entry.words and len(entry.words) > 1:
            # Build karaoke line with \kf tags
            # \kf = smooth fill from left to right over duration
            karaoke_parts = []
            for w in entry.words:
                # Duration in centiseconds (ASS \kf unit)
                dur_cs = max(1, int((w.end - w.start) * 100))
                word_text = w.text.strip()
                if not word_text:
                    continue
                # Add space before word (except first)
                if karaoke_parts:
                    word_text = " " + word_text
                karaoke_parts.append(f"{{\\kf{dur_cs}}}{word_text}")

            text = "".join(karaoke_parts)
        else:
            text = entry.text.strip()

        lines.append(
            f"Dialogue: 0,{start_tc},{end_tc},{style_name},,0,0,0,,{text}"
        )

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    return output_path


def _sanitize_seconds(seconds) -> float:
    """Coerce *seconds* into a safe non-negative finite float for timecoding.

    Whisper occasionally emits ``None`` (clipped end boundaries) or very
    small negatives on certain backends. Without coercion those leak into
    the timecode formatters and produce invalid SRT/VTT/ASS output that
    downstream players silently drop.
    """
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return 0.0
    # NaN / ±inf → 0.0; negative → 0.0.
    if s != s or s in (float("inf"), float("-inf")) or s < 0:
        return 0.0
    return s


def _format_ass_time(seconds: float) -> str:
    """Format seconds as ASS timecode: H:MM:SS.cc (centiseconds)."""
    seconds = _sanitize_seconds(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def rgb_to_ass_color(r: int, g: int, b: int, a: int = 0) -> str:
    """
    Convert RGBA (0-255) to ASS color format &HAABBGGRR.

    ASS alpha is inverted vs standard RGBA:
      RGBA a=0   (transparent) -> ASS alpha=255 (transparent)
      RGBA a=255 (opaque)      -> ASS alpha=0   (opaque)

    The default a=0 means fully transparent in RGBA, which maps to
    ASS alpha=255.  Pass a=255 for fully opaque text.
    """
    ass_alpha = max(0, min(255, 255 - a))
    return f"&H{ass_alpha:02X}{b:02X}{g:02X}{r:02X}"


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timecode: HH:MM:SS,mmm."""
    seconds = _sanitize_seconds(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_time(seconds: float) -> str:
    """Format seconds as VTT timecode: HH:MM:SS.mmm."""
    seconds = _sanitize_seconds(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _derived_segment(seg: CaptionSegment, text: str, start: float, end: float, words=None) -> CaptionSegment:
    return CaptionSegment(
        text=text,
        start=start,
        end=end,
        words=list(words or []),
        speaker=getattr(seg, "speaker", None),
        language=getattr(seg, "language", None),
        language_confidence=getattr(seg, "language_confidence", 1.0),
        confidence=getattr(seg, "confidence", 1.0),
        human_review_recommended=getattr(seg, "human_review_recommended", False),
        review_reasons=list(getattr(seg, "review_reasons", []) or []),
    )


def _proportional_cues(
    seg: CaptionSegment,
    text: str,
    start: float,
    end: float,
    max_line_length: int,
    max_lines: int,
) -> List[CaptionSegment]:
    return [
        _derived_segment(seg, chunk, chunk_start, chunk_end)
        for chunk, chunk_start, chunk_end in timed_caption_chunks(text, start, end, max_line_length, max_lines)
    ]


def _split_segments(
    segments: List[CaptionSegment],
    max_line_length: int,
    max_lines: int,
) -> List[CaptionSegment]:
    """Split segments so every cue wraps into at most ``max_lines`` lines.

    No text is dropped. A segment that needs more lines is split at word
    boundaries, timed from its word timestamps when it has them and
    proportionally otherwise. A single word too long for one cue is split
    across cues within its own timing.
    """
    def fits(text: str) -> bool:
        return len(caption_lines(text, max_line_length)) <= max_lines

    def words_text(words) -> str:
        return " ".join(w.text.strip() for w in words if w.text.strip())

    result = []
    for seg in segments:
        text = seg.text.strip()
        if fits(text):
            result.append(seg)
            continue

        if not seg.words:
            result.extend(_proportional_cues(seg, text, seg.start, seg.end, max_line_length, max_lines))
            continue

        group = []
        for word in seg.words:
            if fits(words_text(group + [word])):
                group.append(word)
                continue
            if group:
                result.append(_derived_segment(seg, words_text(group), group[0].start, group[-1].end, group))
                group = []
            if fits(words_text([word])):
                group = [word]
            else:
                result.extend(_proportional_cues(
                    seg, word.text.strip(), word.start, word.end, max_line_length, max_lines,
                ))
        if group:
            result.append(_derived_segment(seg, words_text(group), group[0].start, group[-1].end, group))

    return result


def _wrap_text(text: str, max_line_length: int) -> str:
    """Wrap a cue that ``_split_segments`` already sized, keeping all of it."""
    return "\n".join(caption_lines(text, max_line_length))
