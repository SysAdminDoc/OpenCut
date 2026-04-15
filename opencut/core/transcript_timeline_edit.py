"""
Transcript Timeline Edit — Edit video by editing text (Descript-style).

Parse word-level timestamps from WhisperX transcription into a bidirectional
text-timeline map.  Supports delete, rearrange, duplicate, and insert-pause
operations with undo stack.  Generates EDL/cut-list and OTIO-compatible JSON.

Builds on transcript_edit.py but adds operation-stack undo, paragraph
rearrangement, segment duplication, pause insertion, and multi-format export.
"""

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import FFmpegCmd, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TimelineWord:
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

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "speaker": self.speaker,
            "paragraph_index": self.paragraph_index,
            "is_deleted": self.is_deleted,
        }


@dataclass
class TimelineParagraph:
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

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "word_start_index": self.word_start_index,
            "word_end_index": self.word_end_index,
            "speaker": self.speaker,
        }


@dataclass
class CutEntry:
    """A single cut/segment in the resulting edit list."""
    source_start: float
    source_end: float
    dest_start: float = 0.0
    label: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.source_end - self.source_start)

    def to_dict(self) -> dict:
        return {
            "source_start": round(self.source_start, 4),
            "source_end": round(self.source_end, 4),
            "dest_start": round(self.dest_start, 4),
            "duration": round(self.duration, 4),
            "label": self.label,
        }


@dataclass
class EditOperation:
    """A reversible edit operation on the transcript."""
    op_type: str  # "delete_words", "rearrange_segments", "duplicate_segment", "insert_pause"
    params: Dict = field(default_factory=dict)
    snapshot_before: Optional[Dict] = None  # serialized state for undo

    def to_dict(self) -> dict:
        return {
            "op_type": self.op_type,
            "params": self.params,
        }


@dataclass
class TranscriptEdit:
    """Result of applying text edits to video timeline."""
    operations: List[Dict] = field(default_factory=list)
    resulting_cuts: List[Dict] = field(default_factory=list)
    total_duration_change: float = 0.0
    new_duration: float = 0.0
    original_duration: float = 0.0
    word_count_before: int = 0
    word_count_after: int = 0

    def to_dict(self) -> dict:
        return {
            "operations": self.operations,
            "resulting_cuts": self.resulting_cuts,
            "total_duration_change": round(self.total_duration_change, 4),
            "new_duration": round(self.new_duration, 4),
            "original_duration": round(self.original_duration, 4),
            "word_count_before": self.word_count_before,
            "word_count_after": self.word_count_after,
        }


# ---------------------------------------------------------------------------
# Timeline Map — central mutable state
# ---------------------------------------------------------------------------
class TimelineMap:
    """Bidirectional text<->timeline map with operation stack for undo."""

    def __init__(self):
        self.words: List[TimelineWord] = []
        self.paragraphs: List[TimelineParagraph] = []
        self.total_duration: float = 0.0
        self.language: str = "en"
        self.source_file: str = ""
        self._undo_stack: List[EditOperation] = []
        self._pauses: List[Dict] = []  # inserted pauses

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words if not w.is_deleted)

    @property
    def active_word_count(self) -> int:
        return sum(1 for w in self.words if not w.is_deleted)

    @property
    def paragraph_count(self) -> int:
        return len(self.paragraphs)

    @property
    def undo_depth(self) -> int:
        return len(self._undo_stack)

    def get_active_words(self) -> List[TimelineWord]:
        return [w for w in self.words if not w.is_deleted]

    def _snapshot(self) -> Dict:
        """Capture current state for undo."""
        return {
            "words": [copy.deepcopy(w) for w in self.words],
            "paragraphs": [copy.deepcopy(p) for p in self.paragraphs],
            "pauses": list(self._pauses),
            "total_duration": self.total_duration,
        }

    def _restore_snapshot(self, snapshot: Dict):
        """Restore state from snapshot."""
        self.words = snapshot["words"]
        self.paragraphs = snapshot["paragraphs"]
        self._pauses = snapshot["pauses"]
        self.total_duration = snapshot["total_duration"]

    def to_dict(self) -> dict:
        return {
            "words": [w.to_dict() for w in self.words],
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "total_duration": self.total_duration,
            "language": self.language,
            "source_file": self.source_file,
            "active_word_count": self.active_word_count,
            "paragraph_count": self.paragraph_count,
            "undo_depth": self.undo_depth,
            "pauses": list(self._pauses),
        }


# ---------------------------------------------------------------------------
# Parse transcript into TimelineMap
# ---------------------------------------------------------------------------
def parse_transcript(
    transcript_data: dict,
    source_file: str = "",
    on_progress: Optional[Callable] = None,
) -> TimelineMap:
    """Build a TimelineMap from WhisperX transcript data.

    Args:
        transcript_data: Transcript dict. Formats:
            - WhisperX: {"segments": [{"text", "start", "end", "words": [...]}]}
            - Simple list: [{"text", "start", "end", "words": [...]}]
            - Words-only: {"words": [{"word"/"text", "start", "end"}]}
        source_file: Optional source video file path.
        on_progress: Progress callback(pct).

    Returns:
        TimelineMap with word-level bidirectional mapping.
    """
    if on_progress:
        on_progress(5)

    segments = _normalize_segments(transcript_data)
    if not segments:
        raise ValueError("No segments found in transcript data")

    if on_progress:
        on_progress(15)

    tmap = TimelineMap()
    tmap.source_file = source_file
    if isinstance(transcript_data, dict):
        tmap.language = transcript_data.get("language", "en")

    word_idx = 0
    for seg_idx, seg in enumerate(segments):
        seg_text = seg.get("text", "").strip()
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", 0))
        speaker = seg.get("speaker", "")
        seg_words = seg.get("words", [])
        para_word_start = word_idx

        if seg_words:
            for w in seg_words:
                w_text = w.get("word", w.get("text", "")).strip()
                if not w_text:
                    continue
                w_start = float(w.get("start", seg_start))
                w_end = float(w.get("end", w_start + 0.1))
                confidence = float(w.get("score", w.get("confidence", 1.0)))
                tmap.words.append(TimelineWord(
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
            text_words = seg_text.split()
            if text_words:
                seg_dur = max(0.01, seg_end - seg_start)
                word_dur = seg_dur / len(text_words)
                for i, tw in enumerate(text_words):
                    w_start = seg_start + i * word_dur
                    w_end = w_start + word_dur
                    tmap.words.append(TimelineWord(
                        index=word_idx,
                        text=tw,
                        start=w_start,
                        end=w_end,
                        confidence=0.5,
                        speaker=speaker,
                        paragraph_index=seg_idx,
                    ))
                    word_idx += 1

        tmap.paragraphs.append(TimelineParagraph(
            index=seg_idx,
            text=seg_text,
            start=seg_start,
            end=seg_end,
            word_start_index=para_word_start,
            word_end_index=word_idx,
            speaker=speaker,
        ))

        if on_progress and len(segments) > 1:
            pct = 15 + int((seg_idx / len(segments)) * 75)
            on_progress(pct)

    if tmap.words:
        tmap.total_duration = max(w.end for w in tmap.words)
    elif tmap.paragraphs:
        tmap.total_duration = max(p.end for p in tmap.paragraphs)

    if on_progress:
        on_progress(100)

    return tmap


def _normalize_segments(data) -> List[dict]:
    """Normalize various transcript formats to list of segment dicts."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "segments" in data:
            return data["segments"]
        if "words" in data and isinstance(data["words"], list):
            return _group_words_to_segments(data["words"])
        if "text" in data and ("start" in data or "words" in data):
            return [data]
    return []


def _group_words_to_segments(
    words: list, gap_threshold: float = 1.0,
) -> List[dict]:
    """Group raw word dicts into segments by time gaps."""
    if not words:
        return []
    segments: List[dict] = []
    current_words: List[dict] = []
    current_start: Optional[float] = None

    for w in words:
        w_start = float(w.get("start", 0))
        w_text = w.get("word", w.get("text", "")).strip()
        if not w_text:
            continue
        if current_start is None:
            current_start = w_start
            current_words = [w]
        elif w_start - float(current_words[-1].get("end", 0)) > gap_threshold:
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
    if current_words and current_start is not None:
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
# Edit Operations
# ---------------------------------------------------------------------------
def delete_words(
    tmap: TimelineMap,
    word_indices: List[int],
    on_progress: Optional[Callable] = None,
) -> List[CutEntry]:
    """Delete words from timeline map, return resulting cut list.

    Marks specified words as deleted and computes contiguous keep-segments
    from the remaining active words.

    Args:
        tmap: The timeline map (modified in-place).
        word_indices: Word indices to delete.
        on_progress: Progress callback(pct).

    Returns:
        List of CutEntry describing segments to keep.
    """
    if not word_indices:
        return _build_cut_list(tmap)

    if on_progress:
        on_progress(10)

    # Snapshot for undo
    snap = tmap._snapshot()
    op = EditOperation(
        op_type="delete_words",
        params={"word_indices": list(word_indices)},
        snapshot_before=snap,
    )

    indices_set = set(word_indices)
    for w in tmap.words:
        if w.index in indices_set:
            w.is_deleted = True

    tmap._undo_stack.append(op)

    if on_progress:
        on_progress(80)

    cuts = _build_cut_list(tmap)

    if on_progress:
        on_progress(100)

    return cuts


def rearrange_segments(
    tmap: TimelineMap,
    new_order: List[int],
    on_progress: Optional[Callable] = None,
) -> List[CutEntry]:
    """Reorder paragraphs/segments in the timeline.

    Args:
        tmap: The timeline map (modified in-place).
        new_order: List of paragraph indices in desired output order.
        on_progress: Progress callback(pct).

    Returns:
        List of CutEntry describing segments in new order.
    """
    if on_progress:
        on_progress(10)

    valid_indices = set(range(len(tmap.paragraphs)))
    for idx in new_order:
        if idx not in valid_indices:
            raise ValueError(f"Invalid paragraph index: {idx}")

    snap = tmap._snapshot()
    op = EditOperation(
        op_type="rearrange_segments",
        params={"new_order": list(new_order)},
        snapshot_before=snap,
    )

    # Rebuild paragraphs and words in new order
    new_paragraphs: List[TimelineParagraph] = []
    new_words: List[TimelineWord] = []
    word_idx = 0
    running_time = 0.0

    for out_idx, para_idx in enumerate(new_order):
        old_para = tmap.paragraphs[para_idx]
        para_words = [
            w for w in tmap.words
            if w.paragraph_index == para_idx and not w.is_deleted
        ]

        para_start = running_time
        for w in para_words:
            offset = w.start - old_para.start
            new_w = TimelineWord(
                index=word_idx,
                text=w.text,
                start=para_start + offset,
                end=para_start + offset + w.duration,
                confidence=w.confidence,
                speaker=w.speaker,
                paragraph_index=out_idx,
                is_deleted=False,
            )
            new_words.append(new_w)
            word_idx += 1

        para_dur = old_para.duration
        new_paragraphs.append(TimelineParagraph(
            index=out_idx,
            text=old_para.text,
            start=para_start,
            end=para_start + para_dur,
            word_start_index=word_idx - len(para_words),
            word_end_index=word_idx,
            speaker=old_para.speaker,
        ))
        running_time += para_dur

        if on_progress and len(new_order) > 1:
            pct = 10 + int((out_idx / len(new_order)) * 80)
            on_progress(pct)

    tmap.words = new_words
    tmap.paragraphs = new_paragraphs
    tmap.total_duration = running_time
    tmap._undo_stack.append(op)

    if on_progress:
        on_progress(100)

    return _build_cut_list_from_order(tmap, new_order, snap)


def duplicate_segment(
    tmap: TimelineMap,
    paragraph_index: int,
    insert_after: Optional[int] = None,
    on_progress: Optional[Callable] = None,
) -> List[CutEntry]:
    """Duplicate a paragraph/segment.

    Args:
        tmap: The timeline map (modified in-place).
        paragraph_index: Index of paragraph to duplicate.
        insert_after: Position after which to insert. None = after original.
        on_progress: Progress callback(pct).

    Returns:
        List of CutEntry describing segments including the duplicate.
    """
    if paragraph_index < 0 or paragraph_index >= len(tmap.paragraphs):
        raise ValueError(f"Invalid paragraph index: {paragraph_index}")

    if on_progress:
        on_progress(10)

    snap = tmap._snapshot()
    op = EditOperation(
        op_type="duplicate_segment",
        params={"paragraph_index": paragraph_index, "insert_after": insert_after},
        snapshot_before=snap,
    )

    if insert_after is None:
        insert_after = paragraph_index

    source_para = tmap.paragraphs[paragraph_index]
    source_words = [
        w for w in tmap.words
        if w.paragraph_index == paragraph_index and not w.is_deleted
    ]

    # Build new order: all paragraphs with duplicate inserted
    new_order = list(range(len(tmap.paragraphs)))
    insert_pos = insert_after + 1
    new_order.insert(insert_pos, paragraph_index)

    if on_progress:
        on_progress(40)

    # Create duplicate paragraph and words
    new_para_idx = len(tmap.paragraphs)
    dup_words: List[TimelineWord] = []
    base_word_idx = len(tmap.words)

    for i, w in enumerate(source_words):
        dup_words.append(TimelineWord(
            index=base_word_idx + i,
            text=w.text,
            start=w.start,
            end=w.end,
            confidence=w.confidence,
            speaker=w.speaker,
            paragraph_index=new_para_idx,
            is_deleted=False,
        ))

    dup_para = TimelineParagraph(
        index=new_para_idx,
        text=source_para.text,
        start=source_para.start,
        end=source_para.end,
        word_start_index=base_word_idx,
        word_end_index=base_word_idx + len(dup_words),
        speaker=source_para.speaker,
    )

    tmap.words.extend(dup_words)
    tmap.paragraphs.append(dup_para)
    tmap._undo_stack.append(op)

    if on_progress:
        on_progress(80)

    # Build cut list: original order with duplicate inserted
    cuts = _build_ordered_cut_list(tmap, new_order)

    if on_progress:
        on_progress(100)

    return cuts


def insert_pause(
    tmap: TimelineMap,
    after_word_index: int,
    pause_duration: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> List[CutEntry]:
    """Insert a silence gap at the specified position.

    Args:
        tmap: The timeline map (modified in-place).
        after_word_index: Insert pause after this word index.
        pause_duration: Duration of silence in seconds.
        on_progress: Progress callback(pct).

    Returns:
        List of CutEntry including the pause gap.
    """
    if after_word_index < -1 or after_word_index >= len(tmap.words):
        raise ValueError(f"Invalid word index: {after_word_index}")
    if pause_duration <= 0:
        raise ValueError("Pause duration must be positive")

    if on_progress:
        on_progress(10)

    snap = tmap._snapshot()
    op = EditOperation(
        op_type="insert_pause",
        params={"after_word_index": after_word_index, "pause_duration": pause_duration},
        snapshot_before=snap,
    )

    # Determine pause insertion time
    if after_word_index == -1:
        pause_time = 0.0
    else:
        pause_time = tmap.words[after_word_index].end

    tmap._pauses.append({
        "after_word_index": after_word_index,
        "time": pause_time,
        "duration": pause_duration,
    })
    tmap._undo_stack.append(op)

    if on_progress:
        on_progress(60)

    cuts = _build_cut_list_with_pauses(tmap)

    if on_progress:
        on_progress(100)

    return cuts


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------
def undo(tmap: TimelineMap) -> bool:
    """Undo the last operation. Returns True if undo was performed."""
    if not tmap._undo_stack:
        return False

    op = tmap._undo_stack.pop()
    if op.snapshot_before is not None:
        tmap._restore_snapshot(op.snapshot_before)
    return True


# ---------------------------------------------------------------------------
# Cut List Builders
# ---------------------------------------------------------------------------
def _build_cut_list(tmap: TimelineMap) -> List[CutEntry]:
    """Build cut list from active (non-deleted) words as contiguous segments."""
    active = tmap.get_active_words()
    if not active:
        return []

    cuts: List[CutEntry] = []
    seg_start = active[0].start
    prev_end = active[0].end
    dest_time = 0.0

    for i in range(1, len(active)):
        w = active[i]
        gap = w.start - prev_end
        if gap > 0.15:  # gap threshold for segment break
            cut = CutEntry(
                source_start=seg_start,
                source_end=prev_end,
                dest_start=dest_time,
                label=f"segment_{len(cuts)}",
            )
            cuts.append(cut)
            dest_time += cut.duration
            seg_start = w.start
        prev_end = w.end

    # Final segment
    cuts.append(CutEntry(
        source_start=seg_start,
        source_end=prev_end,
        dest_start=dest_time,
        label=f"segment_{len(cuts)}",
    ))

    return cuts


def _build_cut_list_from_order(
    tmap: TimelineMap,
    new_order: List[int],
    original_snap: Dict,
) -> List[CutEntry]:
    """Build cut list from rearranged paragraph order using original timestamps."""
    orig_paragraphs = original_snap["paragraphs"]
    cuts: List[CutEntry] = []
    dest_time = 0.0

    for para_idx in new_order:
        p = orig_paragraphs[para_idx]
        cut = CutEntry(
            source_start=p.start,
            source_end=p.end,
            dest_start=dest_time,
            label=f"para_{para_idx}",
        )
        cuts.append(cut)
        dest_time += cut.duration

    return cuts


def _build_ordered_cut_list(
    tmap: TimelineMap,
    order: List[int],
) -> List[CutEntry]:
    """Build cut list from a paragraph order list."""
    cuts: List[CutEntry] = []
    dest_time = 0.0

    for para_idx in order:
        p = tmap.paragraphs[para_idx]
        cut = CutEntry(
            source_start=p.start,
            source_end=p.end,
            dest_start=dest_time,
            label=f"para_{para_idx}",
        )
        cuts.append(cut)
        dest_time += cut.duration

    return cuts


def _build_cut_list_with_pauses(tmap: TimelineMap) -> List[CutEntry]:
    """Build cut list incorporating pause insertions."""
    base_cuts = _build_cut_list(tmap)
    if not tmap._pauses:
        return base_cuts

    # Sort pauses by insertion time
    sorted_pauses = sorted(tmap._pauses, key=lambda p: p["time"])

    result: List[CutEntry] = []
    dest_time = 0.0
    pause_idx = 0

    for cut in base_cuts:
        # Insert any pauses that fall before or within this cut
        while (
            pause_idx < len(sorted_pauses)
            and sorted_pauses[pause_idx]["time"] <= cut.source_end
        ):
            pause = sorted_pauses[pause_idx]
            pause_time = pause["time"]

            if pause_time >= cut.source_start:
                # Split cut at pause point
                if pause_time > cut.source_start:
                    pre = CutEntry(
                        source_start=cut.source_start,
                        source_end=pause_time,
                        dest_start=dest_time,
                        label=f"{cut.label}_pre",
                    )
                    result.append(pre)
                    dest_time += pre.duration

                # Pause gap (silence — no source)
                result.append(CutEntry(
                    source_start=-1,  # sentinel for silence
                    source_end=-1,
                    dest_start=dest_time,
                    label=f"pause_{pause_idx}",
                ))
                dest_time += pause["duration"]
                cut = CutEntry(
                    source_start=pause_time,
                    source_end=cut.source_end,
                    dest_start=dest_time,
                    label=f"{cut.label}_post",
                )
            pause_idx += 1

        if cut.duration > 0:
            cut.dest_start = dest_time
            result.append(cut)
            dest_time += cut.duration

    return result


# ---------------------------------------------------------------------------
# EDL Export
# ---------------------------------------------------------------------------
def export_edl(
    cuts: List[CutEntry],
    title: str = "OpenCut Edit",
    fps: float = 30.0,
) -> str:
    """Export cut list as CMX 3600 EDL string.

    Args:
        cuts: List of CutEntry from an edit operation.
        title: EDL title.
        fps: Frame rate for timecode conversion.

    Returns:
        EDL content string.
    """
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    edit_num = 1

    for cut in cuts:
        if cut.source_start < 0:
            # Silence/pause entry
            tc_in = _seconds_to_tc(cut.dest_start, fps)
            tc_out = _seconds_to_tc(cut.dest_start + 1.0, fps)
            lines.append(
                f"{edit_num:03d}  BL       V     C        "
                f"{tc_in} {tc_out} {tc_in} {tc_out}"
            )
        else:
            src_in = _seconds_to_tc(cut.source_start, fps)
            src_out = _seconds_to_tc(cut.source_end, fps)
            rec_in = _seconds_to_tc(cut.dest_start, fps)
            rec_out = _seconds_to_tc(cut.dest_start + cut.duration, fps)
            lines.append(
                f"{edit_num:03d}  AX       AA/V  C        "
                f"{src_in} {src_out} {rec_in} {rec_out}"
            )
        edit_num += 1

    lines.append("")
    return "\n".join(lines)


def _seconds_to_tc(seconds: float, fps: float = 30.0) -> str:
    """Convert seconds to SMPTE timecode HH:MM:SS:FF."""
    seconds = max(0.0, seconds)
    total_frames = int(seconds * fps)
    ff = total_frames % int(fps)
    total_seconds = total_frames // int(fps)
    ss = total_seconds % 60
    mm = (total_seconds // 60) % 60
    hh = total_seconds // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


# ---------------------------------------------------------------------------
# OTIO-compatible JSON Export
# ---------------------------------------------------------------------------
def export_otio_json(
    cuts: List[CutEntry],
    source_file: str = "",
    fps: float = 30.0,
) -> dict:
    """Export cut list as OTIO-compatible JSON structure.

    Args:
        cuts: List of CutEntry from an edit operation.
        source_file: Source media file reference.
        fps: Frame rate.

    Returns:
        Dict representing OTIO timeline structure.
    """
    clips = []
    for i, cut in enumerate(cuts):
        if cut.source_start < 0:
            # Gap/pause
            clips.append({
                "OTIO_SCHEMA": "Gap.1",
                "name": cut.label or f"gap_{i}",
                "source_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": fps, "value": 0},
                    "duration": {"OTIO_SCHEMA": "RationalTime.1", "rate": fps, "value": int(1.0 * fps)},
                },
            })
        else:
            clips.append({
                "OTIO_SCHEMA": "Clip.2",
                "name": cut.label or f"clip_{i}",
                "media_reference": {
                    "OTIO_SCHEMA": "ExternalReference.1",
                    "target_url": source_file,
                },
                "source_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "rate": fps,
                        "value": int(cut.source_start * fps),
                    },
                    "duration": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "rate": fps,
                        "value": int(cut.duration * fps),
                    },
                },
            })

    return {
        "OTIO_SCHEMA": "Timeline.1",
        "name": "OpenCut Edit",
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "children": [{
                "OTIO_SCHEMA": "Track.1",
                "name": "V1",
                "kind": "Video",
                "children": clips,
            }],
        },
    }


# ---------------------------------------------------------------------------
# Apply edits to video via FFmpeg
# ---------------------------------------------------------------------------
def apply_edits(
    source_file: str,
    cuts: List[CutEntry],
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> TranscriptEdit:
    """Apply cut list to source video, producing edited output.

    Args:
        source_file: Path to source video.
        cuts: List of CutEntry from edit operations.
        output_dir: Output directory (default: same as source).
        on_progress: Progress callback(pct).

    Returns:
        TranscriptEdit with output path and statistics.
    """
    if not cuts:
        raise ValueError("No cuts to apply")
    if not os.path.isfile(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")

    if on_progress:
        on_progress(5)

    info = get_video_info(source_file)
    fps = info.get("fps", 30.0)
    original_duration = info.get("duration", 0.0)

    # Filter to actual media cuts (skip silence markers)
    media_cuts = [c for c in cuts if c.source_start >= 0]
    if not media_cuts:
        raise ValueError("No media segments in cut list")

    out_path = output_path(source_file, "edited", output_dir)

    if on_progress:
        on_progress(10)

    if len(media_cuts) == 1:
        # Simple trim
        cut = media_cuts[0]
        cmd = (FFmpegCmd()
               .input(source_file, ss=str(cut.source_start), to=str(cut.source_end))
               .video_codec("libx264", crf=18, preset="fast")
               .audio_codec("aac", bitrate="192k")
               .faststart()
               .output(out_path)
               .build())
        run_ffmpeg(cmd)
    else:
        # Concat multiple segments
        _concat_segments(source_file, media_cuts, out_path, on_progress)

    if on_progress:
        on_progress(90)

    # Compute stats
    new_dur = sum(c.duration for c in media_cuts)
    duration_change = new_dur - original_duration

    # Export EDL and OTIO sidecar files
    edl_path = out_path.rsplit(".", 1)[0] + ".edl"
    edl_content = export_edl(cuts, fps=fps)
    with open(edl_path, "w", encoding="utf-8") as f:
        f.write(edl_content)

    otio_path = out_path.rsplit(".", 1)[0] + ".otio.json"
    otio_data = export_otio_json(cuts, source_file=source_file, fps=fps)
    with open(otio_path, "w", encoding="utf-8") as f:
        json.dump(otio_data, f, indent=2)

    if on_progress:
        on_progress(100)

    return TranscriptEdit(
        operations=[],
        resulting_cuts=[c.to_dict() for c in cuts],
        total_duration_change=duration_change,
        new_duration=new_dur,
        original_duration=original_duration,
        word_count_before=0,
        word_count_after=0,
    )


def _concat_segments(
    source_file: str,
    cuts: List[CutEntry],
    out_path: str,
    on_progress: Optional[Callable] = None,
):
    """Concatenate multiple segments using FFmpeg filter_complex."""
    # Build filter_complex string
    n = len(cuts)
    filter_parts = []
    for i, cut in enumerate(cuts):
        filter_parts.append(
            f"[0:v]trim=start={cut.source_start:.4f}:end={cut.source_end:.4f},"
            f"setpts=PTS-STARTPTS[v{i}];"
        )
        filter_parts.append(
            f"[0:a]atrim=start={cut.source_start:.4f}:end={cut.source_end:.4f},"
            f"asetpts=PTS-STARTPTS[a{i}];"
        )

    v_streams = "".join(f"[v{i}]" for i in range(n))
    a_streams = "".join(f"[a{i}]" for i in range(n))
    filter_parts.append(f"{v_streams}concat=n={n}:v=1:a=0[outv];")
    filter_parts.append(f"{a_streams}concat=n={n}:v=0:a=1[outa]")

    fc = "".join(filter_parts)

    cmd = (FFmpegCmd()
           .input(source_file)
           .filter_complex(fc, maps=["[outv]", "[outa]"])
           .video_codec("libx264", crf=18, preset="fast")
           .audio_codec("aac", bitrate="192k")
           .faststart()
           .output(out_path)
           .build())
    run_ffmpeg(cmd)


# ---------------------------------------------------------------------------
# Preview — return cut list without applying
# ---------------------------------------------------------------------------
def preview_edits(
    tmap: TimelineMap,
    operations: List[Dict],
    on_progress: Optional[Callable] = None,
) -> TranscriptEdit:
    """Preview edit operations without modifying original or rendering video.

    Args:
        tmap: The timeline map.
        operations: List of operation dicts, each with "type" and params.
        on_progress: Progress callback(pct).

    Returns:
        TranscriptEdit with preview cut list (no output_path).
    """
    if on_progress:
        on_progress(5)

    # Work on a deep copy to avoid mutating original
    preview_map = TimelineMap()
    preview_map.words = [copy.deepcopy(w) for w in tmap.words]
    preview_map.paragraphs = [copy.deepcopy(p) for p in tmap.paragraphs]
    preview_map.total_duration = tmap.total_duration
    preview_map._pauses = list(tmap._pauses)

    word_count_before = preview_map.active_word_count
    original_duration = preview_map.total_duration

    all_cuts: List[CutEntry] = []

    for i, op in enumerate(operations):
        op_type = op.get("type", "")
        if op_type == "delete_words":
            indices = op.get("word_indices", [])
            all_cuts = delete_words(preview_map, indices)
        elif op_type == "rearrange_segments":
            order = op.get("new_order", [])
            all_cuts = rearrange_segments(preview_map, order)
        elif op_type == "duplicate_segment":
            para_idx = op.get("paragraph_index", 0)
            after = op.get("insert_after")
            all_cuts = duplicate_segment(preview_map, para_idx, after)
        elif op_type == "insert_pause":
            word_idx = op.get("after_word_index", 0)
            dur = op.get("pause_duration", 1.0)
            all_cuts = insert_pause(preview_map, word_idx, dur)

        if on_progress and len(operations) > 1:
            pct = 5 + int(((i + 1) / len(operations)) * 90)
            on_progress(pct)

    if not all_cuts:
        all_cuts = _build_cut_list(preview_map)

    new_dur = sum(c.duration for c in all_cuts if c.source_start >= 0)
    duration_change = new_dur - original_duration

    if on_progress:
        on_progress(100)

    return TranscriptEdit(
        operations=[op for op in operations],
        resulting_cuts=[c.to_dict() for c in all_cuts],
        total_duration_change=duration_change,
        new_duration=new_dur,
        original_duration=original_duration,
        word_count_before=word_count_before,
        word_count_after=preview_map.active_word_count,
    )
