"""
OpenCut Shot-Change-Aware Subtitle Timing

Snaps subtitle boundaries to scene cuts so no subtitle straddles a
visual transition, improving readability.

Uses FFmpeg scene detection and SRT parsing - no additional dependencies.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from opencut.helpers import get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# SRT Parsing Utilities (shared logic with caption_compliance)
# ---------------------------------------------------------------------------
_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")


def _parse_srt_time(s: str) -> float:
    m = _TIME_RE.search(s)
    if not m:
        return 0.0
    h, mi, sec, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mi * 60 + sec + ms / 1000.0


def _format_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    remainder = seconds - h * 3600
    m = int(remainder // 60)
    s = remainder - m * 60
    sec = int(s)
    ms = int(round((s - sec) * 1000))
    if ms >= 1000:
        sec += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


@dataclass
class _Subtitle:
    index: int
    start: float
    end: float
    text: str


def _parse_srt(srt_path: str) -> List[_Subtitle]:
    with open(srt_path, "r", encoding="utf-8-sig") as f:
        content = f.read()
    blocks = re.split(r"\n\s*\n", content.strip())
    subs: List[_Subtitle] = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        timing_idx = -1
        for i, line in enumerate(lines):
            if "-->" in line:
                timing_idx = i
                break
        if timing_idx < 0:
            continue
        parts = lines[timing_idx].split("-->")
        if len(parts) != 2:
            continue
        start = _parse_srt_time(parts[0].strip())
        end = _parse_srt_time(parts[1].strip())
        text = "\n".join(ln.strip() for ln in lines[timing_idx + 1:] if ln.strip())
        try:
            idx = int(lines[0].strip())
        except ValueError:
            idx = len(subs) + 1
        subs.append(_Subtitle(index=idx, start=start, end=end, text=text))
    return subs


def _write_srt(subs: List[_Subtitle], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, sub in enumerate(subs, 1):
            f.write(f"{i}\n")
            f.write(f"{_format_srt_time(sub.start)} --> {_format_srt_time(sub.end)}\n")
            f.write(f"{sub.text}\n\n")


# ---------------------------------------------------------------------------
# Snap Subtitles to Cuts
# ---------------------------------------------------------------------------
def snap_subtitles_to_cuts(
    srt_path: str,
    cut_times: List[float],
    min_gap_frames: int = 2,
    fps: float = 24.0,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Snap subtitle boundaries to scene change cut points.

    For each subtitle that spans a cut, either:
    - Split it at the cut point with a min_gap
    - Adjust timing so no subtitle starts within min_gap_frames of a cut

    Args:
        srt_path: Path to the .srt file.
        cut_times: List of scene change timestamps in seconds.
        min_gap_frames: Minimum gap in frames between subtitle and cut.
        fps: Frame rate of the video.
        output_path: Output file path (defaults to input with _snapped suffix).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path and adjustments_made count.
    """
    if on_progress:
        on_progress(5, "Parsing subtitles...")

    subtitles = _parse_srt(srt_path)
    if not output_path:
        base, ext = os.path.splitext(srt_path)
        output_path = f"{base}_snapped{ext}"

    if not cut_times:
        # No cuts - just copy through
        _write_srt(subtitles, output_path)
        if on_progress:
            on_progress(100, "No cuts provided, subtitles unchanged")
        return {"output_path": output_path, "adjustments_made": 0}

    sorted_cuts = sorted(set(cut_times))
    gap_seconds = min_gap_frames / fps
    adjustments = 0
    result_subs: List[_Subtitle] = []

    total = len(subtitles)

    if on_progress:
        on_progress(10, f"Processing {total} subtitles against {len(sorted_cuts)} cuts...")

    for si, sub in enumerate(subtitles):
        # Find cuts that fall within this subtitle's time range
        spanning_cuts = [
            c for c in sorted_cuts
            if sub.start < c < sub.end
        ]

        if not spanning_cuts:
            # No cuts within this subtitle - check if start is too close to a cut
            for c in sorted_cuts:
                # If subtitle starts within gap_seconds of a cut, push it after
                if 0 < (sub.start - c) < gap_seconds:
                    sub.start = c + gap_seconds
                    adjustments += 1
                    break
                # If subtitle ends within gap_seconds before a cut, pull it back
                if 0 < (c - sub.end) < gap_seconds:
                    sub.end = c - gap_seconds
                    adjustments += 1
                    break
            result_subs.append(sub)
        else:
            # Subtitle spans one or more cuts - split it
            segments = []
            current_start = sub.start
            sub.text.split()
            total_chars = len(sub.text.replace("\n", " "))

            for cut in spanning_cuts:
                seg_end = cut - gap_seconds
                if seg_end > current_start + 0.05:
                    # Estimate text proportion based on time proportion
                    time_ratio = (seg_end - current_start) / (sub.end - sub.start)
                    char_split = max(1, int(total_chars * time_ratio))

                    # Find word boundary near the split point
                    text_so_far = sub.text.replace("\n", " ")
                    split_text = text_so_far[:char_split]
                    # Adjust to nearest word boundary
                    last_space = split_text.rfind(" ")
                    if last_space > 0:
                        split_text = split_text[:last_space]

                    segments.append((current_start, seg_end, split_text.strip()))
                current_start = cut + gap_seconds

            # Remaining segment after last cut
            if current_start < sub.end - 0.05:
                remaining_text = sub.text.replace("\n", " ")
                # Get text not yet assigned
                assigned_text = " ".join(s[2] for s in segments)
                if assigned_text and remaining_text.startswith(assigned_text):
                    remaining_text = remaining_text[len(assigned_text):].strip()
                elif segments:
                    # Approximate: use latter portion
                    ratio = len(segments) / (len(segments) + 1)
                    char_pos = int(len(remaining_text) * ratio)
                    remaining_text = remaining_text[char_pos:].strip()

                if remaining_text:
                    segments.append((current_start, sub.end, remaining_text))

            if segments:
                for seg_start, seg_end, seg_text in segments:
                    if seg_text:
                        result_subs.append(_Subtitle(
                            index=sub.index,
                            start=round(seg_start, 3),
                            end=round(seg_end, 3),
                            text=seg_text,
                        ))
                adjustments += 1
            else:
                # Could not split meaningfully - keep original
                result_subs.append(sub)

        if on_progress:
            pct = 10 + int(80 * (si + 1) / total)
            on_progress(pct, f"Processed subtitle {si + 1}/{total}")

    if on_progress:
        on_progress(95, "Writing adjusted subtitles...")

    _write_srt(result_subs, output_path)

    if on_progress:
        on_progress(100, f"Made {adjustments} adjustments")

    return {
        "output_path": output_path,
        "adjustments_made": adjustments,
    }


def auto_snap_subtitles(
    srt_path: str,
    video_path: str,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Automatically detect scene cuts and snap subtitles to them.

    Runs scene detection internally, then snaps subtitles.

    Args:
        srt_path: Path to the .srt file.
        video_path: Path to the video file.
        output_path: Output file path.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, adjustments_made, and cuts_detected.
    """
    if on_progress:
        on_progress(5, "Detecting scene cuts...")

    # Run scene detection
    from opencut.core.scene_detect import detect_scenes

    def _scene_progress(pct, msg=""):
        if on_progress:
            on_progress(5 + int(pct * 0.4), f"Scene detection: {msg}")

    scene_info = detect_scenes(
        video_path, threshold=0.3, min_scene_length=1.0,
        on_progress=_scene_progress,
    )

    cut_times = [b.time for b in scene_info.boundaries if b.time > 0]

    if on_progress:
        on_progress(50, f"Found {len(cut_times)} scene cuts, snapping subtitles...")

    # Get video FPS
    info = get_video_info(video_path)
    fps = info.get("fps", 24.0)

    def _snap_progress(pct, msg=""):
        if on_progress:
            on_progress(50 + int(pct * 0.5), msg)

    result = snap_subtitles_to_cuts(
        srt_path=srt_path,
        cut_times=cut_times,
        min_gap_frames=2,
        fps=fps,
        output_path=output_path,
        on_progress=_snap_progress,
    )

    result["cuts_detected"] = len(cut_times)

    if on_progress:
        on_progress(100, f"Snapped to {len(cut_times)} cuts, {result['adjustments_made']} adjustments")

    return result
