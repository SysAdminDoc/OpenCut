"""
Transcript-Based Editing (1.1)

Text-to-timeline bidirectional map from WhisperX word-level timestamps.
Edit video by editing text: delete words -> cut list, rearrange -> new sequence.

Builds on PaperEditSelection from paper_edit.py but adds full bidirectional
mapping and word-level editing.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import FFmpegCmd, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class WordMapping:
    """A single word with bidirectional text<->timeline mapping."""
    index: int
    text: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: str = ""
    paragraph_index: int = 0
    is_deleted: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ParagraphMapping:
    """A paragraph/segment mapped between text and timeline."""
    index: int
    text: str
    start: float
    end: float
    word_start_index: int = 0
    word_end_index: int = 0
    speaker: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def word_count(self) -> int:
        return max(0, self.word_end_index - self.word_start_index)


@dataclass
class TimeRange:
    """A time range in the video."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TextEdit:
    """Describes a single edit operation on the transcript text."""
    edit_type: str  # "delete", "rearrange", "keep"
    word_indices: List[int] = field(default_factory=list)
    paragraph_indices: List[int] = field(default_factory=list)
    new_order: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.edit_type not in ("delete", "rearrange", "keep"):
            raise ValueError(
                f"Invalid edit_type: {self.edit_type}. "
                "Use 'delete', 'rearrange', or 'keep'"
            )


@dataclass
class CutSegment:
    """A segment to include in the final cut."""
    start: float
    end: float
    source_word_start: int = -1
    source_word_end: int = -1

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TranscriptMap:
    """Bidirectional map between transcript text and video timeline."""
    words: List[WordMapping] = field(default_factory=list)
    paragraphs: List[ParagraphMapping] = field(default_factory=list)
    total_duration: float = 0.0
    language: str = "en"
    source_file: str = ""

    @property
    def text(self) -> str:
        """Full transcript text from non-deleted words."""
        return " ".join(w.text for w in self.words if not w.is_deleted)

    @property
    def word_count(self) -> int:
        return sum(1 for w in self.words if not w.is_deleted)

    @property
    def paragraph_count(self) -> int:
        return len(self.paragraphs)

    def get_word(self, index: int) -> Optional[WordMapping]:
        """Get word by index."""
        if 0 <= index < len(self.words):
            return self.words[index]
        return None

    def get_paragraph(self, index: int) -> Optional[ParagraphMapping]:
        """Get paragraph by index."""
        if 0 <= index < len(self.paragraphs):
            return self.paragraphs[index]
        return None

    def get_active_words(self) -> List[WordMapping]:
        """Get all non-deleted words in order."""
        return [w for w in self.words if not w.is_deleted]

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        return {
            "words": [
                {
                    "index": w.index,
                    "text": w.text,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                    "speaker": w.speaker,
                    "paragraph_index": w.paragraph_index,
                    "is_deleted": w.is_deleted,
                }
                for w in self.words
            ],
            "paragraphs": [
                {
                    "index": p.index,
                    "text": p.text,
                    "start": p.start,
                    "end": p.end,
                    "word_start_index": p.word_start_index,
                    "word_end_index": p.word_end_index,
                    "speaker": p.speaker,
                }
                for p in self.paragraphs
            ],
            "total_duration": self.total_duration,
            "language": self.language,
            "source_file": self.source_file,
            "word_count": self.word_count,
            "paragraph_count": self.paragraph_count,
        }


@dataclass
class TranscriptEditResult:
    """Result of applying text edits to video."""
    output_path: str = ""
    duration: float = 0.0
    cut_count: int = 0
    removed_duration: float = 0.0
    edl_path: str = ""
    otio_path: str = ""


# ---------------------------------------------------------------------------
# Build Transcript Map
# ---------------------------------------------------------------------------
def build_transcript_map(
    transcript_json: dict,
    source_file: str = "",
    on_progress: Optional[Callable] = None,
) -> TranscriptMap:
    """Build a bidirectional text<->timeline map from WhisperX transcript data.

    Accepts transcript data in WhisperX format (segments with word-level
    timestamps) or simplified format (list of segments with words).

    Args:
        transcript_json: Transcript data dict. Expected formats:
            - WhisperX: {"segments": [{"text", "start", "end", "words": [...]}], ...}
            - Simple:   [{"text", "start", "end", "words": [...]}]
            - Words-only: {"words": [{"word"/"text", "start", "end"}]}
        source_file: Optional source video file path.
        on_progress: Progress callback(pct, msg).

    Returns:
        TranscriptMap with word-level bidirectional mapping.
    """
    if on_progress:
        on_progress(5, "Parsing transcript data...")

    # Normalize input format
    segments = _normalize_transcript(transcript_json)

    if not segments:
        raise ValueError("No segments found in transcript data")

    if on_progress:
        on_progress(20, f"Processing {len(segments)} segments...")

    words: List[WordMapping] = []
    paragraphs: List[ParagraphMapping] = []
    word_idx = 0

    for seg_idx, seg in enumerate(segments):
        seg_text = seg.get("text", "").strip()
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", 0))
        speaker = seg.get("speaker", "")
        seg_words = seg.get("words", [])

        para_word_start = word_idx

        if seg_words:
            # Use word-level timestamps
            for w in seg_words:
                w_text = w.get("word", w.get("text", "")).strip()
                if not w_text:
                    continue
                w_start = float(w.get("start", seg_start))
                w_end = float(w.get("end", w_start + 0.1))
                confidence = float(w.get("score", w.get("confidence", 1.0)))

                words.append(WordMapping(
                    index=word_idx,
                    text=w_text,
                    start=w_start,
                    end=w_end,
                    confidence=confidence,
                    speaker=speaker,
                    paragraph_index=seg_idx,
                ))
                word_idx += 1
        elif seg_text:
            # No word-level timestamps; split text and interpolate
            text_words = seg_text.split()
            if text_words:
                seg_dur = max(0.01, seg_end - seg_start)
                word_dur = seg_dur / len(text_words)
                for i, tw in enumerate(text_words):
                    w_start = seg_start + i * word_dur
                    w_end = w_start + word_dur
                    words.append(WordMapping(
                        index=word_idx,
                        text=tw,
                        start=w_start,
                        end=w_end,
                        confidence=0.5,  # interpolated
                        speaker=speaker,
                        paragraph_index=seg_idx,
                    ))
                    word_idx += 1

        paragraphs.append(ParagraphMapping(
            index=seg_idx,
            text=seg_text,
            start=seg_start,
            end=seg_end,
            word_start_index=para_word_start,
            word_end_index=word_idx,
            speaker=speaker,
        ))

        if on_progress and len(segments) > 1:
            pct = 20 + int((seg_idx / len(segments)) * 70)
            on_progress(pct, f"Mapped segment {seg_idx + 1}/{len(segments)}")

    # Calculate total duration
    total_duration = 0.0
    if words:
        total_duration = max(w.end for w in words)
    elif paragraphs:
        total_duration = max(p.end for p in paragraphs)

    language = "en"
    if isinstance(transcript_json, dict):
        language = transcript_json.get("language", "en")

    if on_progress:
        on_progress(100, f"Mapped {len(words)} words across {len(paragraphs)} paragraphs")

    return TranscriptMap(
        words=words,
        paragraphs=paragraphs,
        total_duration=total_duration,
        language=language,
        source_file=source_file,
    )


def _normalize_transcript(data) -> List[dict]:
    """Normalize various transcript formats to a list of segment dicts."""
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # WhisperX format
        if "segments" in data:
            return data["segments"]
        # Words-only format
        if "words" in data and isinstance(data["words"], list):
            # Group words into segments by gaps
            return _words_to_segments(data["words"])
        # Single segment
        if "text" in data and ("start" in data or "words" in data):
            return [data]

    return []


def _words_to_segments(words: list, gap_threshold: float = 1.0) -> List[dict]:
    """Group raw words into segments based on time gaps."""
    if not words:
        return []

    segments = []
    current_words = []
    current_start = None

    for w in words:
        w_start = float(w.get("start", 0))
        float(w.get("end", w_start + 0.1))
        w_text = w.get("word", w.get("text", "")).strip()
        if not w_text:
            continue

        if current_start is None:
            current_start = w_start
            current_words = [w]
        elif w_start - float(current_words[-1].get("end", 0)) > gap_threshold:
            # Gap detected, finalize current segment
            segments.append({
                "text": " ".join(
                    cw.get("word", cw.get("text", "")) for cw in current_words
                ).strip(),
                "start": current_start,
                "end": float(current_words[-1].get("end", 0)),
                "words": current_words,
            })
            current_start = w_start
            current_words = [w]
        else:
            current_words.append(w)

    if current_words:
        segments.append({
            "text": " ".join(
                cw.get("word", cw.get("text", "")) for cw in current_words
            ).strip(),
            "start": current_start,
            "end": float(current_words[-1].get("end", 0)),
            "words": current_words,
        })

    return segments


# ---------------------------------------------------------------------------
# Text Selection <-> Time Range
# ---------------------------------------------------------------------------
def text_selection_to_timerange(
    transcript_map: TranscriptMap,
    start_idx: int,
    end_idx: int,
) -> TimeRange:
    """Convert a text selection (word index range) to a video time range.

    Args:
        transcript_map: The transcript map.
        start_idx: Start word index (inclusive).
        end_idx: End word index (inclusive).

    Returns:
        TimeRange with start and end times.
    """
    if not transcript_map.words:
        raise ValueError("Transcript map has no words")

    start_idx = max(0, start_idx)
    end_idx = min(end_idx, len(transcript_map.words) - 1)

    if start_idx > end_idx:
        raise ValueError(f"start_idx ({start_idx}) > end_idx ({end_idx})")

    # Get time range from the word indices
    start_word = transcript_map.words[start_idx]
    end_word = transcript_map.words[end_idx]

    return TimeRange(start=start_word.start, end=end_word.end)


def timerange_to_text_selection(
    transcript_map: TranscriptMap,
    start_time: float,
    end_time: float,
) -> Tuple[int, int]:
    """Convert a video time range to text selection (word index range).

    Args:
        transcript_map: The transcript map.
        start_time: Start time in seconds.
        end_time: End time in seconds.

    Returns:
        Tuple of (start_word_index, end_word_index) inclusive.
    """
    if not transcript_map.words:
        raise ValueError("Transcript map has no words")

    if start_time > end_time:
        raise ValueError(f"start_time ({start_time}) > end_time ({end_time})")

    first_idx = None
    last_idx = None

    for w in transcript_map.words:
        if w.is_deleted:
            continue
        # Word overlaps with time range
        if w.end > start_time and w.start < end_time:
            if first_idx is None:
                first_idx = w.index
            last_idx = w.index

    if first_idx is None:
        raise ValueError(
            f"No words found in time range {start_time:.2f}-{end_time:.2f}"
        )

    return (first_idx, last_idx)


# ---------------------------------------------------------------------------
# Edit Operations
# ---------------------------------------------------------------------------
def delete_words(
    transcript_map: TranscriptMap,
    word_indices: List[int],
    on_progress: Optional[Callable] = None,
) -> List[CutSegment]:
    """Delete words from transcript map and return the resulting cut list.

    Marks words as deleted and computes contiguous keep-segments
    from the remaining words.

    Args:
        transcript_map: The transcript map (modified in-place).
        word_indices: Indices of words to delete.
        on_progress: Progress callback(pct, msg).

    Returns:
        List of CutSegment describing the video segments to keep.
    """
    if not word_indices:
        return _compute_cut_segments(transcript_map)

    if on_progress:
        on_progress(10, f"Deleting {len(word_indices)} words...")

    indices_set = set(word_indices)
    for idx in indices_set:
        if 0 <= idx < len(transcript_map.words):
            transcript_map.words[idx].is_deleted = True

    if on_progress:
        on_progress(60, "Computing cut segments...")

    segments = _compute_cut_segments(transcript_map)

    if on_progress:
        on_progress(100, f"Generated {len(segments)} cut segments")

    return segments


def rearrange_paragraphs(
    transcript_map: TranscriptMap,
    new_order: List[int],
    on_progress: Optional[Callable] = None,
) -> List[CutSegment]:
    """Rearrange paragraphs in the transcript and return new cut list.

    Args:
        transcript_map: The transcript map.
        new_order: New paragraph order (list of paragraph indices).
        on_progress: Progress callback(pct, msg).

    Returns:
        List of CutSegment in the new order.
    """
    if not new_order:
        raise ValueError("new_order cannot be empty")

    n_paras = len(transcript_map.paragraphs)
    for idx in new_order:
        if idx < 0 or idx >= n_paras:
            raise ValueError(
                f"Paragraph index {idx} out of range (0-{n_paras - 1})"
            )

    if on_progress:
        on_progress(10, f"Rearranging {len(new_order)} paragraphs...")

    segments = []
    for i, para_idx in enumerate(new_order):
        transcript_map.paragraphs[para_idx]

        # Get non-deleted words in this paragraph
        para_words = [
            w for w in transcript_map.words
            if w.paragraph_index == para_idx and not w.is_deleted
        ]

        if para_words:
            seg_start = para_words[0].start
            seg_end = para_words[-1].end
            segments.append(CutSegment(
                start=seg_start,
                end=seg_end,
                source_word_start=para_words[0].index,
                source_word_end=para_words[-1].index,
            ))

        if on_progress:
            pct = 10 + int((i / len(new_order)) * 80)
            on_progress(pct, f"Processed paragraph {i + 1}/{len(new_order)}")

    if on_progress:
        on_progress(100, f"Rearranged into {len(segments)} segments")

    return segments


def _compute_cut_segments(transcript_map: TranscriptMap) -> List[CutSegment]:
    """Compute contiguous keep-segments from non-deleted words."""
    active_words = transcript_map.get_active_words()
    if not active_words:
        return []

    segments = []
    seg_start = active_words[0].start
    seg_end = active_words[0].end
    seg_word_start = active_words[0].index
    prev_word = active_words[0]

    for w in active_words[1:]:
        # If there's a gap > 50ms between words, it's the same segment
        # A gap means a deleted word existed between them
        gap = w.start - prev_word.end
        if gap > 0.5:  # Significant gap = new segment
            segments.append(CutSegment(
                start=seg_start,
                end=seg_end,
                source_word_start=seg_word_start,
                source_word_end=prev_word.index,
            ))
            seg_start = w.start
            seg_word_start = w.index
        seg_end = w.end
        prev_word = w

    # Final segment
    segments.append(CutSegment(
        start=seg_start,
        end=seg_end,
        source_word_start=seg_word_start,
        source_word_end=prev_word.index,
    ))

    return segments


# ---------------------------------------------------------------------------
# Apply Text Edits to Video
# ---------------------------------------------------------------------------
def apply_text_edits(
    video_path: str,
    transcript_map: TranscriptMap,
    edits: List[TextEdit],
    out_path: str = "",
    on_progress: Optional[Callable] = None,
) -> TranscriptEditResult:
    """Apply text-based edits to a video file.

    Processes edits sequentially: deletions first, then rearrangements.
    Uses FFmpeg to cut and concatenate video segments.

    Args:
        video_path: Source video file.
        transcript_map: The transcript map.
        edits: List of TextEdit operations.
        out_path: Output file path (auto-generated if empty).
        on_progress: Progress callback(pct, msg).

    Returns:
        TranscriptEditResult with output path and stats.
    """
    if not edits:
        raise ValueError("No edits provided")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(5, "Processing edits...")

    # Apply deletions first
    for edit in edits:
        if edit.edit_type == "delete":
            delete_words(transcript_map, edit.word_indices)

    # Then compute segments (rearrange or keep)
    rearrange_edits = [e for e in edits if e.edit_type == "rearrange"]
    if rearrange_edits:
        # Use the last rearrange edit's order
        last_rearrange = rearrange_edits[-1]
        segments = rearrange_paragraphs(
            transcript_map, last_rearrange.new_order
        )
    else:
        segments = _compute_cut_segments(transcript_map)

    if not segments:
        raise ValueError("All content was deleted; no segments remain")

    if on_progress:
        on_progress(20, f"Assembling {len(segments)} segments...")

    # Calculate stats
    original_duration = transcript_map.total_duration
    kept_duration = sum(s.duration for s in segments)
    removed_duration = original_duration - kept_duration

    # Build output
    if not out_path:
        out_dir = os.path.dirname(os.path.abspath(video_path))
        out_path = output_path(video_path, "transcript_edit", out_dir)

    # Use FFmpeg to extract and concatenate segments
    _assemble_segments(video_path, segments, out_path, on_progress)

    if on_progress:
        on_progress(100, "Edit complete")

    return TranscriptEditResult(
        output_path=out_path,
        duration=kept_duration,
        cut_count=len(segments),
        removed_duration=removed_duration,
    )


def _assemble_segments(
    video_path: str,
    segments: List[CutSegment],
    out_path: str,
    on_progress: Optional[Callable] = None,
) -> None:
    """Assemble video from cut segments using FFmpeg concat."""
    tmp_dir = tempfile.mkdtemp(prefix="opencut_txedit_")
    segment_files = []
    total = len(segments)

    try:
        for i, seg in enumerate(segments):
            if on_progress:
                pct = 20 + int((i / total) * 55)
                on_progress(pct, f"Extracting segment {i + 1}/{total}...")

            seg_path = os.path.join(tmp_dir, f"seg_{i:04d}.mp4")
            cmd = (FFmpegCmd()
                   .input(video_path, ss=seg.start, to=seg.end)
                   .video_codec("libx264", crf=18, preset="fast")
                   .audio_codec("aac", bitrate="192k")
                   .faststart()
                   .output(seg_path)
                   .build())
            run_ffmpeg(cmd)
            segment_files.append(seg_path)

        if on_progress:
            on_progress(80, "Concatenating segments...")

        # Build concat file
        concat_path = os.path.join(tmp_dir, "concat.txt")
        with open(concat_path, "w", encoding="utf-8") as f:
            for seg_path in segment_files:
                safe = seg_path.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        # Concatenate
        cmd = (FFmpegCmd()
               .option("f", "concat")
               .option("safe", "0")
               .input(concat_path)
               .copy_streams()
               .faststart()
               .output(out_path)
               .build())
        run_ffmpeg(cmd)

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Export Edited Sequence (EDL / OTIO)
# ---------------------------------------------------------------------------
def export_edited_sequence(
    video_path: str,
    transcript_map: TranscriptMap,
    edits: List[TextEdit],
    format: str = "otio",
    out_path: str = "",
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Export the edited sequence as EDL or OTIO.

    Args:
        video_path: Source video file path.
        transcript_map: The transcript map.
        edits: List of TextEdit operations.
        format: Export format — "otio", "edl", or "json".
        out_path: Output file path (auto-generated if empty).
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path, format, and segment count.
    """
    format = format.lower().strip()
    if format not in ("otio", "edl", "json"):
        raise ValueError(f"Unsupported format: {format}. Use otio, edl, or json")

    if on_progress:
        on_progress(10, "Processing edits for export...")

    # Apply edits to get segments
    for edit in edits:
        if edit.edit_type == "delete":
            delete_words(transcript_map, edit.word_indices)

    rearrange_edits = [e for e in edits if e.edit_type == "rearrange"]
    if rearrange_edits:
        last_rearrange = rearrange_edits[-1]
        segments = rearrange_paragraphs(transcript_map, last_rearrange.new_order)
    else:
        segments = _compute_cut_segments(transcript_map)

    if not segments:
        raise ValueError("No segments to export")

    if on_progress:
        on_progress(50, f"Exporting {len(segments)} segments as {format}...")

    if not out_path:
        ext_map = {"otio": ".otio", "edl": ".edl", "json": ".json"}
        fd, out_path = tempfile.mkstemp(
            suffix=ext_map[format], prefix="transcript_edit_"
        )
        os.close(fd)

    filename = os.path.basename(video_path) if video_path else "source"

    if format == "edl":
        _export_edl(segments, filename, out_path)
    elif format == "otio":
        _export_otio(segments, filename, video_path, out_path)
    elif format == "json":
        _export_json(segments, filename, transcript_map, out_path)

    if on_progress:
        on_progress(100, f"Exported as {format}")

    return {
        "output_path": out_path,
        "format": format,
        "segment_count": len(segments),
    }


def _export_edl(
    segments: List[CutSegment],
    filename: str,
    out_path: str,
) -> None:
    """Export segments as CMX 3600 EDL."""
    lines = ["TITLE: Transcript Edit\nFCM: NON-DROP FRAME\n\n"]
    record_in = 0.0

    for i, seg in enumerate(segments):
        record_out = record_in + seg.duration
        lines.append(
            f"{i + 1:03d}  AX  V  C  "
            f"{_fmt_tc(seg.start)} {_fmt_tc(seg.end)} "
            f"{_fmt_tc(record_in)} {_fmt_tc(record_out)}\n"
        )
        lines.append(f"* FROM CLIP NAME: {filename}\n\n")
        record_in = record_out

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _export_otio(
    segments: List[CutSegment],
    filename: str,
    video_path: str,
    out_path: str,
) -> None:
    """Export segments as OpenTimelineIO JSON.

    Produces a valid OTIO file structure without requiring the
    opentimelineio package at runtime.
    """
    fps = 30.0
    if video_path and os.path.isfile(video_path):
        try:
            info = get_video_info(video_path)
            fps = float(info.get("fps", 30)) or 30.0
        except Exception:
            pass

    clips = []
    for i, seg in enumerate(segments):
        start_frames = int(seg.start * fps)
        dur_frames = int(seg.duration * fps)
        clips.append({
            "OTIO_SCHEMA": "Clip.2",
            "name": f"{filename} - Clip {i + 1}",
            "source_range": {
                "OTIO_SCHEMA": "TimeRange.1",
                "start_time": {
                    "OTIO_SCHEMA": "RationalTime.1",
                    "value": start_frames,
                    "rate": fps,
                },
                "duration": {
                    "OTIO_SCHEMA": "RationalTime.1",
                    "value": dur_frames,
                    "rate": fps,
                },
            },
            "media_references": {
                "DEFAULT_MEDIA": {
                    "OTIO_SCHEMA": "ExternalReference.1",
                    "target_url": video_path or filename,
                },
            },
        })

    total_dur_frames = sum(int(s.duration * fps) for s in segments)
    otio_data = {
        "OTIO_SCHEMA": "Timeline.1",
        "name": "Transcript Edit",
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "children": [{
                "OTIO_SCHEMA": "Track.1",
                "name": "V1",
                "kind": "Video",
                "children": clips,
                "source_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "value": 0,
                        "rate": fps,
                    },
                    "duration": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "value": total_dur_frames,
                        "rate": fps,
                    },
                },
            }],
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(otio_data, f, indent=2)


def _export_json(
    segments: List[CutSegment],
    filename: str,
    transcript_map: TranscriptMap,
    out_path: str,
) -> None:
    """Export segments as JSON."""
    data = {
        "source": filename,
        "total_segments": len(segments),
        "total_duration": sum(s.duration for s in segments),
        "segments": [
            {
                "index": i,
                "start": seg.start,
                "end": seg.end,
                "duration": seg.duration,
                "source_word_start": seg.source_word_start,
                "source_word_end": seg.source_word_end,
            }
            for i, seg in enumerate(segments)
        ],
        "transcript_map": transcript_map.to_dict(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _fmt_tc(seconds: float) -> str:
    """Format seconds as HH:MM:SS:FF (30fps)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    f = int((seconds % 1) * 30)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"
