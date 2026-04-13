"""
Paper Edit from Transcript (14.1)

Display transcript, select passages, reorder, and auto-assemble
a video sequence from the selected transcript segments.
"""

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
class PaperEditSelection:
    """A selected passage from the transcript."""
    start: float
    end: float
    text: str = ""
    label: str = ""
    order: int = 0
    speaker: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class PaperEdit:
    """A complete paper edit — ordered list of selected transcript passages."""
    selections: List[PaperEditSelection] = field(default_factory=list)
    source_transcript: str = ""
    total_duration: float = 0.0
    gap_seconds: float = 0.0

    def reorder(self, new_order: List[int]) -> None:
        """Reorder selections by index list."""
        if sorted(new_order) != list(range(len(self.selections))):
            raise ValueError("new_order must be a permutation of selection indices")
        self.selections = [self.selections[i] for i in new_order]
        for i, sel in enumerate(self.selections):
            sel.order = i
        self._recalc()

    def _recalc(self) -> None:
        self.total_duration = sum(s.duration for s in self.selections)


# ---------------------------------------------------------------------------
# Create Paper Edit
# ---------------------------------------------------------------------------
def create_paper_edit(
    transcript: List[dict],
    selections: List[dict],
    on_progress: Optional[Callable] = None,
) -> PaperEdit:
    """Create a paper edit from transcript and selected passages.

    Args:
        transcript: Full transcript segments [{'text', 'start', 'end', ...}].
        selections: Selected passages [{'start', 'end', 'text'?, 'label'?, 'speaker'?}].
        on_progress: Optional callback(pct, msg).

    Returns:
        PaperEdit object with ordered selections.
    """
    if not selections:
        raise ValueError("At least one selection is required")

    if on_progress:
        on_progress(10, "Building paper edit...")

    # Build a text lookup from transcript for enrichment
    transcript_text = {}
    for seg in transcript:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        transcript_text[(start, end)] = seg

    paper_selections = []
    for i, sel in enumerate(selections):
        start = float(sel.get("start", 0))
        end = float(sel.get("end", 0))
        if end <= start:
            raise ValueError(f"Selection {i}: end ({end}) must be after start ({start})")

        # Try to find matching transcript text
        text = sel.get("text", "")
        speaker = sel.get("speaker", "")
        if not text:
            # Find overlapping transcript segments
            parts = []
            for seg in transcript:
                s_start = float(seg.get("start", 0))
                s_end = float(seg.get("end", 0))
                if s_start < end and s_end > start:
                    parts.append(seg.get("text", ""))
                    if not speaker:
                        speaker = seg.get("speaker", "")
            text = " ".join(parts)

        paper_selections.append(PaperEditSelection(
            start=start,
            end=end,
            text=text.strip(),
            label=sel.get("label", f"Selection {i + 1}"),
            order=i,
            speaker=speaker,
        ))

    if on_progress:
        on_progress(50, f"Created {len(paper_selections)} selections")

    edit = PaperEdit(
        selections=paper_selections,
        source_transcript=json.dumps(transcript[:5]) if transcript else "",
    )
    edit._recalc()

    if on_progress:
        on_progress(100, f"Paper edit ready: {edit.total_duration:.1f}s total")

    return edit


# ---------------------------------------------------------------------------
# Assemble Video from Paper Edit
# ---------------------------------------------------------------------------
def assemble_from_paper_edit(
    video_path: str,
    paper_edit: PaperEdit,
    output_path_str: str = "",
    gap_seconds: float = 0.0,
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Assemble a video from paper edit selections using FFmpeg concat.

    Args:
        video_path: Source video file.
        paper_edit: PaperEdit with ordered selections.
        output_path_str: Output file path (auto-generated if empty).
        gap_seconds: Gap between selections in seconds (black frames).
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with output_path, duration, segment_count.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not paper_edit.selections:
        raise ValueError("Paper edit has no selections")

    if on_progress:
        on_progress(5, "Preparing assembly...")

    get_video_info(video_path)
    out_dir = os.path.dirname(os.path.abspath(video_path))

    if not output_path_str:
        output_path_str = output_path(video_path, "paper_edit", out_dir)

    # Extract each selection as a temporary segment
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="opencut_paper_")
    segment_files = []
    total = len(paper_edit.selections)

    try:
        for i, sel in enumerate(paper_edit.selections):
            if on_progress:
                pct = 10 + int((i / total) * 60)
                on_progress(pct, f"Extracting segment {i + 1}/{total}...")

            seg_path = os.path.join(tmp_dir, f"seg_{i:04d}.mp4")
            cmd = (FFmpegCmd()
                   .input(video_path, ss=sel.start, to=sel.end)
                   .video_codec("libx264", crf=18, preset="fast")
                   .audio_codec("aac", bitrate="192k")
                   .faststart()
                   .output(seg_path)
                   .build())
            run_ffmpeg(cmd)
            segment_files.append(seg_path)

        if on_progress:
            on_progress(75, "Concatenating segments...")

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
               .output(output_path_str)
               .build())
        run_ffmpeg(cmd)

    finally:
        # Clean up temp segments
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    if on_progress:
        on_progress(100, "Assembly complete")

    return {
        "output_path": output_path_str,
        "duration": paper_edit.total_duration,
        "segment_count": len(paper_edit.selections),
    }


# ---------------------------------------------------------------------------
# Export Paper Edit (EDL/JSON/text)
# ---------------------------------------------------------------------------
def export_paper_edit(
    paper_edit: PaperEdit,
    format: str = "json",
    output_path_str: str = "",
    on_progress: Optional[Callable] = None,
) -> Dict:
    """Export a paper edit to various formats.

    Args:
        paper_edit: PaperEdit to export.
        format: Export format — "json", "txt", "edl".
        output_path_str: Output file path.
        on_progress: Optional callback(pct, msg).

    Returns:
        Dict with output_path and format.
    """
    if not paper_edit.selections:
        raise ValueError("Paper edit has no selections")

    format = format.lower()
    if format not in ("json", "txt", "edl"):
        raise ValueError(f"Unsupported format: {format}. Use json, txt, or edl")

    if on_progress:
        on_progress(10, f"Exporting paper edit as {format}...")

    if not output_path_str:
        import tempfile
        ext = {"json": ".json", "txt": ".txt", "edl": ".edl"}[format]
        fd, output_path_str = tempfile.mkstemp(suffix=ext, prefix="paper_edit_")
        os.close(fd)

    if format == "json":
        data = {
            "total_duration": paper_edit.total_duration,
            "selection_count": len(paper_edit.selections),
            "selections": [
                {
                    "order": s.order,
                    "start": s.start,
                    "end": s.end,
                    "duration": s.duration,
                    "text": s.text,
                    "label": s.label,
                    "speaker": s.speaker,
                }
                for s in paper_edit.selections
            ],
        }
        with open(output_path_str, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    elif format == "txt":
        lines = [f"Paper Edit — {len(paper_edit.selections)} selections\n"]
        lines.append(f"Total duration: {paper_edit.total_duration:.1f}s\n")
        lines.append("-" * 60 + "\n")
        for s in paper_edit.selections:
            lines.append(
                f"[{s.order + 1}] {_fmt_tc(s.start)} → {_fmt_tc(s.end)} "
                f"({s.duration:.1f}s)\n"
            )
            if s.speaker:
                lines.append(f"    Speaker: {s.speaker}\n")
            if s.text:
                lines.append(f"    {s.text[:120]}\n")
            lines.append("\n")
        with open(output_path_str, "w", encoding="utf-8") as f:
            f.writelines(lines)

    elif format == "edl":
        lines = ["TITLE: Paper Edit\nFCM: NON-DROP FRAME\n\n"]
        record_in = 0.0
        for i, s in enumerate(paper_edit.selections):
            record_out = record_in + s.duration
            lines.append(
                f"{i + 1:03d}  AX  V  C  "
                f"{_fmt_tc(s.start)} {_fmt_tc(s.end)} "
                f"{_fmt_tc(record_in)} {_fmt_tc(record_out)}\n"
            )
            if s.text:
                lines.append(f"* {s.text[:80]}\n")
            lines.append("\n")
            record_in = record_out
        with open(output_path_str, "w", encoding="utf-8") as f:
            f.writelines(lines)

    if on_progress:
        on_progress(100, f"Exported as {format}")

    return {
        "output_path": output_path_str,
        "format": format,
        "selection_count": len(paper_edit.selections),
    }


def _fmt_tc(seconds: float) -> str:
    """Format seconds as HH:MM:SS:FF (30fps)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    f = int((seconds % 1) * 30)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"
