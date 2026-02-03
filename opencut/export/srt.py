"""
SRT and VTT subtitle export.

Generates standard subtitle files from transcription results.
"""

from typing import List, Optional

from ..core.captions import CaptionSegment, TranscriptionResult


def export_srt(
    result: TranscriptionResult,
    output_path: str,
    max_line_length: int = 42,
    max_lines: int = 2,
) -> str:
    """
    Export transcription as SRT (SubRip) subtitle file.

    Args:
        result: Transcription result from Whisper.
        output_path: Output file path.
        max_line_length: Maximum characters per line.
        max_lines: Maximum lines per subtitle entry.

    Returns:
        Path to the generated SRT file.
    """
    entries = _split_segments(result.segments, max_line_length, max_lines)

    lines = []
    for i, entry in enumerate(entries, 1):
        start_tc = _format_srt_time(entry.start)
        end_tc = _format_srt_time(entry.end)
        text = _wrap_text(entry.text.strip(), max_line_length, max_lines)

        lines.append(f"{i}")
        lines.append(f"{start_tc} --> {end_tc}")
        lines.append(text)
        lines.append("")  # Blank line separator

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

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
        text = _wrap_text(entry.text.strip(), max_line_length, max_lines)

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

    data = {
        "language": result.language,
        "duration": result.duration,
        "text": result.text,
        "word_count": result.word_count,
        "segments": [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "words": [
                    {
                        "text": w.text,
                        "start": w.start,
                        "end": w.end,
                        "confidence": w.confidence,
                    }
                    for w in seg.words
                ],
            }
            for seg in result.segments
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timecode: HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_time(seconds: float) -> str:
    """Format seconds as VTT timecode: HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _split_segments(
    segments: List[CaptionSegment],
    max_line_length: int,
    max_lines: int,
) -> List[CaptionSegment]:
    """
    Split long segments into shorter ones suitable for subtitle display.
    """
    result = []

    for seg in segments:
        text = seg.text.strip()

        # If it fits, keep as-is
        if len(text) <= max_line_length * max_lines:
            result.append(seg)
            continue

        # Split by words, respecting timestamps
        if seg.words:
            # Use word-level timestamps for precise splitting
            current_words = []
            current_text = ""

            for word in seg.words:
                test_text = (current_text + " " + word.text).strip()
                if len(test_text) > max_line_length * max_lines and current_words:
                    # Emit current group
                    result.append(CaptionSegment(
                        text=current_text.strip(),
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        words=list(current_words),
                    ))
                    current_words = [word]
                    current_text = word.text
                else:
                    current_words.append(word)
                    current_text = test_text

            # Emit remaining
            if current_words:
                result.append(CaptionSegment(
                    text=current_text.strip(),
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    words=list(current_words),
                ))
        else:
            # No word timestamps — split by character count with proportional timing
            words = text.split()
            total_chars = len(text)
            chunk_size = max_line_length * max_lines

            current_chunk = ""
            chunk_start_ratio = 0.0

            for word in words:
                test = (current_chunk + " " + word).strip()
                if len(test) > chunk_size and current_chunk:
                    # Calculate proportional timing
                    end_ratio = len(current_chunk) / total_chars
                    t_start = seg.start + (seg.duration * chunk_start_ratio)
                    t_end = seg.start + (seg.duration * end_ratio)

                    result.append(CaptionSegment(
                        text=current_chunk.strip(),
                        start=t_start,
                        end=t_end,
                    ))

                    chunk_start_ratio = end_ratio
                    current_chunk = word
                else:
                    current_chunk = test

            if current_chunk:
                t_start = seg.start + (seg.duration * chunk_start_ratio)
                result.append(CaptionSegment(
                    text=current_chunk.strip(),
                    start=t_start,
                    end=seg.end,
                ))

    return result


def _wrap_text(text: str, max_line_length: int, max_lines: int) -> str:
    """Wrap text to fit within line length constraints."""
    if len(text) <= max_line_length:
        return text

    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test = (current_line + " " + word).strip()
        if len(test) > max_line_length and current_line:
            lines.append(current_line)
            current_line = word
            if len(lines) >= max_lines:
                break
        else:
            current_line = test

    if current_line and len(lines) < max_lines:
        lines.append(current_line)

    return "\n".join(lines)
