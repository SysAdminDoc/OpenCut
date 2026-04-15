"""
OpenCut Shot-Change-Aware Subtitle Timing

Post-process subtitle/caption data to ensure no subtitle spans a scene cut.
Splits subtitles at cut points, enforces minimum gaps between subtitle edges
and cuts, merges short fragments, and enforces minimum display duration.

Supports configurable timing profiles: netflix, bbc, fcc, custom.
Exports as SRT, VTT, and ASS formats.
"""

import copy
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Timing Profiles
# ---------------------------------------------------------------------------
TIMING_PROFILES: Dict[str, Dict] = {
    "netflix": {
        "description": "Netflix Timed Text Style Guide",
        "max_chars_per_line": 42,
        "max_lines": 2,
        "min_duration": 0.833,
        "max_duration": 7.0,
        "min_gap_frames": 2,
        "gap_ms": 83,
        "fps": 24,
        "max_cps": 20,
    },
    "bbc": {
        "description": "BBC Subtitle Guidelines",
        "max_chars_per_line": 37,
        "max_lines": 2,
        "min_duration": 1.0,
        "max_duration": 7.0,
        "min_gap_frames": 2,
        "gap_ms": 80,
        "fps": 25,
        "max_cps": 20,
    },
    "fcc": {
        "description": "FCC Closed Captioning Requirements",
        "max_chars_per_line": 32,
        "max_lines": 4,
        "min_duration": 1.0,
        "max_duration": 8.0,
        "min_gap_frames": 2,
        "gap_ms": 67,
        "fps": 30,
        "max_cps": 25,
    },
    "custom": {
        "description": "Custom profile with user-defined settings",
        "max_chars_per_line": 42,
        "max_lines": 2,
        "min_duration": 1.0,
        "max_duration": 7.0,
        "min_gap_frames": 2,
        "gap_ms": 83,
        "fps": 24,
        "max_cps": 20,
    },
}


@dataclass
class SubtitleSegment:
    """A single subtitle segment with timing and text."""

    index: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    position: Optional[str] = None
    style: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def char_count(self) -> int:
        return len(self.text.replace("\n", "").replace(" ", ""))

    def cps(self) -> float:
        d = self.duration
        if d <= 0:
            return 0.0
        return self.char_count() / d


@dataclass
class ShotAwareResult:
    """Result of shot-change-aware subtitle timing adjustment."""

    adjusted_subtitles: List[SubtitleSegment] = field(default_factory=list)
    splits_made: int = 0
    gaps_enforced: int = 0
    violations_fixed: int = 0
    profile_used: str = ""
    total_segments: int = 0
    merge_count: int = 0
    line_wraps: int = 0


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------
def list_profiles() -> List[Dict]:
    """Return list of available timing profiles with metadata."""
    result = []
    for name, cfg in TIMING_PROFILES.items():
        result.append({
            "name": name,
            "description": cfg["description"],
            "max_chars_per_line": cfg["max_chars_per_line"],
            "max_lines": cfg["max_lines"],
            "min_duration": cfg["min_duration"],
            "max_duration": cfg["max_duration"],
            "fps": cfg["fps"],
            "max_cps": cfg["max_cps"],
        })
    return result


def get_profile(name: str) -> Dict:
    """Retrieve a timing profile by name. Raises ValueError if unknown."""
    name = name.lower().strip()
    if name not in TIMING_PROFILES:
        raise ValueError(
            f"Unknown profile: {name}. "
            f"Available: {', '.join(TIMING_PROFILES.keys())}"
        )
    return dict(TIMING_PROFILES[name])


# ---------------------------------------------------------------------------
# Line wrapping
# ---------------------------------------------------------------------------
def _wrap_text(text: str, max_chars: int, max_lines: int) -> str:
    """Wrap subtitle text to fit within character and line constraints."""
    if not text:
        return text
    existing_lines = text.split("\n")
    all_words: List[str] = []
    for line in existing_lines:
        all_words.extend(line.split())
    if not all_words:
        return text
    lines: List[str] = []
    current_line = ""
    for word in all_words:
        test = f"{current_line} {word}".strip() if current_line else word
        if len(test) <= max_chars:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    if len(lines) > max_lines:
        merged = " ".join(lines)
        lines = []
        chunk_size = max(1, len(merged) // max_lines)
        pos = 0
        for i in range(max_lines):
            if i == max_lines - 1:
                lines.append(merged[pos:].strip())
            else:
                end = pos + chunk_size
                space_pos = merged.rfind(" ", pos, end + 10)
                if space_pos > pos:
                    end = space_pos
                lines.append(merged[pos:end].strip())
                pos = end
        lines = [ln for ln in lines if ln]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SRT / VTT / ASS parsing
# ---------------------------------------------------------------------------
def _ts_to_seconds(ts: str) -> float:
    """Parse HH:MM:SS,mmm or HH:MM:SS.mmm to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts)


def _seconds_to_srt(s: float) -> str:
    """Convert seconds to SRT timestamp HH:MM:SS,mmm."""
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")


def _seconds_to_vtt(s: float) -> str:
    """Convert seconds to VTT timestamp HH:MM:SS.mmm."""
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def _seconds_to_ass_ts(s: float) -> str:
    """Convert seconds to ASS timestamp H:MM:SS.cc."""
    if s < 0:
        s = 0.0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    cs = int((sec - int(sec)) * 100)
    return f"{h}:{m:02d}:{int(sec):02d}.{cs:02d}"


def parse_srt(content: str) -> List[SubtitleSegment]:
    """Parse SRT content into SubtitleSegment list."""
    segments: List[SubtitleSegment] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        time_line = None
        text_start = 0
        for i, line in enumerate(lines):
            if "-->" in line:
                time_line = line
                text_start = i + 1
                break
        if not time_line:
            continue
        parts = time_line.split("-->")
        if len(parts) != 2:
            continue
        start = _ts_to_seconds(parts[0])
        end = _ts_to_seconds(parts[1].split()[0])
        text = "\n".join(lines[text_start:]).strip()
        segments.append(SubtitleSegment(
            index=len(segments) + 1,
            start=start,
            end=end,
            text=text,
        ))
    return segments


def parse_vtt(content: str) -> List[SubtitleSegment]:
    """Parse WebVTT content into SubtitleSegment list."""
    lines_all = content.strip().split("\n")
    start_idx = 0
    for i, line in enumerate(lines_all):
        if line.strip().upper().startswith("WEBVTT"):
            start_idx = i + 1
            break
    body = "\n".join(lines_all[start_idx:])
    blocks = re.split(r"\n\s*\n", body.strip())
    segments: List[SubtitleSegment] = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        time_line = None
        text_start = 0
        for i, line in enumerate(lines):
            if "-->" in line:
                time_line = line
                text_start = i + 1
                break
        if not time_line:
            continue
        parts = time_line.split("-->")
        if len(parts) != 2:
            continue
        start = _ts_to_seconds(parts[0])
        end = _ts_to_seconds(parts[1].split()[0])
        text = "\n".join(lines[text_start:]).strip()
        segments.append(SubtitleSegment(
            index=len(segments) + 1,
            start=start,
            end=end,
            text=text,
        ))
    return segments


# ---------------------------------------------------------------------------
# Core logic: shot-aware adjustment
# ---------------------------------------------------------------------------
def _split_at_cuts(
    segments: List[SubtitleSegment],
    cuts: List[float],
    gap_sec: float,
) -> tuple:
    """Split subtitles that span cut points. Returns (new_segments, split_count)."""
    if not cuts:
        return segments, 0
    sorted_cuts = sorted(cuts)
    result: List[SubtitleSegment] = []
    split_count = 0
    for seg in segments:
        crossing_cuts = [c for c in sorted_cuts if seg.start < c < seg.end]
        if not crossing_cuts:
            result.append(copy.deepcopy(seg))
            continue
        boundaries = [seg.start] + crossing_cuts + [seg.end]
        words = seg.text.split()
        total_dur = seg.end - seg.start
        num_parts = len(boundaries) - 1
        for i in range(num_parts):
            part_start = boundaries[i]
            part_end = boundaries[i + 1]
            if i > 0:
                part_start += gap_sec
            if i < num_parts - 1:
                part_end -= gap_sec
            if part_end <= part_start:
                part_end = part_start + 0.1
            frac_start = (boundaries[i] - seg.start) / total_dur if total_dur > 0 else 0
            frac_end = (boundaries[i + 1] - seg.start) / total_dur if total_dur > 0 else 1
            word_start = int(frac_start * len(words))
            word_end = int(frac_end * len(words))
            if i == num_parts - 1:
                word_end = len(words)
            part_text = " ".join(words[word_start:word_end]).strip()
            if not part_text and words:
                part_text = words[min(word_start, len(words) - 1)]
            result.append(SubtitleSegment(
                index=0,
                start=part_start,
                end=part_end,
                text=part_text,
                position=seg.position,
                style=seg.style,
            ))
            split_count += 1
        if num_parts > 0:
            split_count -= 1
    return result, split_count


def _enforce_cut_gaps(
    segments: List[SubtitleSegment],
    cuts: List[float],
    gap_sec: float,
) -> tuple:
    """Ensure minimum gap between subtitle edges and cuts."""
    if not cuts:
        return segments, 0
    sorted_cuts = sorted(cuts)
    gap_count = 0
    for seg in segments:
        for cut in sorted_cuts:
            if abs(seg.end - cut) < gap_sec and seg.end <= cut:
                seg.end = cut - gap_sec
                gap_count += 1
            elif abs(seg.start - cut) < gap_sec and seg.start >= cut:
                seg.start = cut + gap_sec
                gap_count += 1
        if seg.end <= seg.start:
            seg.end = seg.start + 0.1
    return segments, gap_count


def _merge_short_fragments(
    segments: List[SubtitleSegment],
    min_duration: float,
) -> tuple:
    """Merge fragments shorter than min_duration with neighbors."""
    if not segments:
        return segments, 0
    merge_count = 0
    merged: List[SubtitleSegment] = []
    i = 0
    while i < len(segments):
        seg = copy.deepcopy(segments[i])
        if seg.duration < min_duration * 0.5 and merged:
            prev = merged[-1]
            gap = seg.start - prev.end
            if gap < min_duration:
                prev.end = seg.end
                prev.text = f"{prev.text} {seg.text}".strip()
                merge_count += 1
                i += 1
                continue
        merged.append(seg)
        i += 1
    return merged, merge_count


def _enforce_min_duration(
    segments: List[SubtitleSegment],
    min_duration: float,
) -> tuple:
    """Ensure all subtitles meet minimum display duration."""
    fix_count = 0
    for seg in segments:
        if seg.duration < min_duration:
            seg.end = seg.start + min_duration
            fix_count += 1
    return segments, fix_count


def _enforce_max_duration(
    segments: List[SubtitleSegment],
    max_duration: float,
) -> tuple:
    """Split subtitles exceeding maximum duration."""
    if max_duration <= 0:
        return segments, 0
    result: List[SubtitleSegment] = []
    split_count = 0
    for seg in segments:
        if seg.duration <= max_duration:
            result.append(seg)
            continue
        words = seg.text.split()
        n_parts = max(2, int(seg.duration / max_duration) + 1)
        part_dur = seg.duration / n_parts
        words_per_part = max(1, len(words) // n_parts)
        for p in range(n_parts):
            p_start = seg.start + p * part_dur
            p_end = seg.start + (p + 1) * part_dur
            w_start = p * words_per_part
            w_end = (p + 1) * words_per_part if p < n_parts - 1 else len(words)
            p_text = " ".join(words[w_start:w_end]).strip()
            if not p_text:
                continue
            result.append(SubtitleSegment(
                index=0, start=p_start, end=p_end, text=p_text,
                position=seg.position, style=seg.style,
            ))
            split_count += 1
        split_count -= 1
    return result, split_count


def _enforce_line_limits(
    segments: List[SubtitleSegment],
    max_chars: int,
    max_lines: int,
) -> int:
    """Wrap subtitle text to fit within character/line constraints."""
    wrap_count = 0
    for seg in segments:
        original = seg.text
        seg.text = _wrap_text(seg.text, max_chars, max_lines)
        if seg.text != original:
            wrap_count += 1
    return wrap_count


def _reindex(segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
    """Re-assign sequential indices to segments."""
    for i, seg in enumerate(segments):
        seg.index = i + 1
    return segments


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------
def process_shot_aware(
    subtitles: List[SubtitleSegment],
    cuts: List[float],
    profile: str = "netflix",
    custom_settings: Optional[Dict] = None,
    on_progress: Optional[Callable] = None,
) -> ShotAwareResult:
    """Apply shot-change-aware timing adjustments to subtitles.

    Args:
        subtitles: Input subtitle segments.
        cuts: List of cut/scene-change timestamps in seconds.
        profile: Timing profile name (netflix, bbc, fcc, custom).
        custom_settings: Override profile settings when profile is 'custom'.
        on_progress: Progress callback (percentage int).

    Returns:
        ShotAwareResult with adjusted subtitles and statistics.
    """
    if not subtitles:
        return ShotAwareResult(profile_used=profile)

    cfg = get_profile(profile)
    if profile == "custom" and custom_settings:
        cfg.update(custom_settings)

    gap_sec = cfg["gap_ms"] / 1000.0
    min_dur = cfg["min_duration"]
    max_dur = cfg["max_duration"]
    max_chars = cfg["max_chars_per_line"]
    max_lines = cfg["max_lines"]

    if on_progress:
        on_progress(10)

    working = [copy.deepcopy(s) for s in subtitles]

    # Step 1: Split at cut points
    working, splits = _split_at_cuts(working, cuts, gap_sec)
    if on_progress:
        on_progress(30)

    # Step 2: Enforce gaps near cuts
    working, gaps = _enforce_cut_gaps(working, cuts, gap_sec)
    if on_progress:
        on_progress(45)

    # Step 3: Merge short fragments
    working, merges = _merge_short_fragments(working, min_dur)
    if on_progress:
        on_progress(55)

    # Step 4: Enforce minimum duration
    working, min_fixes = _enforce_min_duration(working, min_dur)
    if on_progress:
        on_progress(65)

    # Step 5: Enforce maximum duration
    working, max_splits = _enforce_max_duration(working, max_dur)
    if on_progress:
        on_progress(75)

    # Step 6: Line wrapping
    wrap_count = _enforce_line_limits(working, max_chars, max_lines)
    if on_progress:
        on_progress(90)

    # Re-index
    working = _reindex(working)

    if on_progress:
        on_progress(100)

    return ShotAwareResult(
        adjusted_subtitles=working,
        splits_made=splits + max_splits,
        gaps_enforced=gaps,
        violations_fixed=min_fixes,
        profile_used=profile,
        total_segments=len(working),
        merge_count=merges,
        line_wraps=wrap_count,
    )


# ---------------------------------------------------------------------------
# Convenience: process from raw dicts
# ---------------------------------------------------------------------------
def process_shot_aware_dicts(
    subtitle_dicts: List[Dict],
    cut_times: List[float],
    profile: str = "netflix",
    custom_settings: Optional[Dict] = None,
    on_progress: Optional[Callable] = None,
) -> ShotAwareResult:
    """Process subtitle dicts (start, end, text) with shot-aware timing.

    Each dict should have 'start', 'end', 'text' keys. Returns ShotAwareResult.
    """
    segments = []
    for i, d in enumerate(subtitle_dicts):
        segments.append(SubtitleSegment(
            index=i + 1,
            start=float(d.get("start", 0)),
            end=float(d.get("end", 0)),
            text=str(d.get("text", "")),
        ))
    return process_shot_aware(segments, cut_times, profile, custom_settings, on_progress)


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------
def export_srt(segments: List[SubtitleSegment]) -> str:
    """Export subtitle segments as SRT string."""
    lines: List[str] = []
    for seg in segments:
        lines.append(str(seg.index))
        lines.append(f"{_seconds_to_srt(seg.start)} --> {_seconds_to_srt(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def export_vtt(segments: List[SubtitleSegment]) -> str:
    """Export subtitle segments as WebVTT string."""
    lines: List[str] = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_seconds_to_vtt(seg.start)} --> {_seconds_to_vtt(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def export_ass(
    segments: List[SubtitleSegment],
    title: str = "OpenCut Subtitles",
    video_width: int = 1920,
    video_height: int = 1080,
) -> str:
    """Export subtitle segments as ASS (Advanced SubStation Alpha) string."""
    header = (
        "[Script Info]\n"
        f"Title: {title}\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_width}\n"
        f"PlayResY: {video_height}\n"
        "WrapStyle: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,"
        "&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,20,20,40,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    events: List[str] = []
    for seg in segments:
        start = _seconds_to_ass_ts(seg.start)
        end = _seconds_to_ass_ts(seg.end)
        text = seg.text.replace("\n", "\\N")
        if seg.position:
            text = f"{seg.position}{text}"
        events.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
        )
    return header + "\n".join(events) + "\n"


def export_to_file(
    segments: List[SubtitleSegment],
    output_path: str,
    fmt: str = "srt",
    **kwargs,
) -> str:
    """Export segments to file. Returns output path.

    Args:
        segments: Subtitle segments to export.
        output_path: Output file path.
        fmt: Format (srt, vtt, ass).
        **kwargs: Additional args for ASS export (title, video_width, video_height).
    """
    fmt = fmt.lower().strip()
    if fmt == "srt":
        content = export_srt(segments)
    elif fmt == "vtt":
        content = export_vtt(segments)
    elif fmt == "ass":
        content = export_ass(segments, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use srt, vtt, or ass.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Exported %d subtitles to %s (%s)", len(segments), output_path, fmt)
    return output_path
