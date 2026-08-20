"""
OpenCut Caption Burn-in Module v0.8.0

Burn subtitles/captions directly into video pixels:
- ASS/SRT/VTT subtitle overlay via FFmpeg subtitles filter
- Styled burn-in using OpenCut caption styles
- Position control (top, center, bottom)
- Font embedding for consistent rendering

Burns subtitles permanently into the video stream (hardcoded).
Use this when the target player doesn't support soft subs.
"""

import logging
import os
import re as _re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from opencut.helpers import (
    FFmpegCmd,
    escape_filter_path,
    get_ffmpeg_path,
    get_ffprobe_path,
    get_video_info,
    run_ffmpeg,
    write_concat_list,
)

logger = logging.getLogger("opencut")

# A caption edit rarely moves a cue by a whole frame, so ranges are padded
# before being snapped outward to keyframes. Without the pad, a cue whose end
# moved by a few milliseconds could produce a zero-length changed range.
CHANGE_RANGE_PAD_SECONDS = 0.04
#: Below this share of the file left untouched, segmenting and concatenating
#: costs more than simply re-encoding the whole thing.
INCREMENTAL_MIN_COPY_RATIO = 0.15
#: Keyframe cadence forced on burn-in output so a later caption edit has
#: cut points to stream-copy around.
KEYFRAME_INTERVAL_SECONDS = 2.0

# ---------------------------------------------------------------------------
# Incremental re-burn planning
# ---------------------------------------------------------------------------


def _cue_key(segment: Dict) -> Tuple[float, float, str]:
    return (
        round(float(segment.get("start") or 0.0), 3),
        round(float(segment.get("end") or 0.0), 3),
        str(segment.get("text") or "").strip(),
    )


def _merge_ranges(ranges: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def caption_change_ranges(
    old_segments: Sequence[Dict],
    new_segments: Sequence[Dict],
    pad: float = CHANGE_RANGE_PAD_SECONDS,
) -> List[Tuple[float, float]]:
    """Time ranges whose burned pixels differ between two caption sets.

    A cue that is identical in both sets contributes nothing. Anything added,
    removed, retimed, or reworded contributes the span it occupies in *both*
    sets, because the old pixels have to be painted over and the new ones
    drawn. Ranges are padded and merged so adjacent edits become one segment.
    """
    old_by_key = {}
    for segment in old_segments or ():
        old_by_key.setdefault(_cue_key(segment), []).append(segment)
    new_by_key = {}
    for segment in new_segments or ():
        new_by_key.setdefault(_cue_key(segment), []).append(segment)

    changed: List[Tuple[float, float]] = []
    for key, entries in old_by_key.items():
        surplus = len(entries) - len(new_by_key.get(key, ()))
        for segment in entries[: max(surplus, 0)]:
            changed.append((float(segment.get("start") or 0.0), float(segment.get("end") or 0.0)))
    for key, entries in new_by_key.items():
        surplus = len(entries) - len(old_by_key.get(key, ()))
        for segment in entries[: max(surplus, 0)]:
            changed.append((float(segment.get("start") or 0.0), float(segment.get("end") or 0.0)))

    padded = []
    for start, end in changed:
        low, high = min(start, end), max(start, end)
        padded.append((max(0.0, low - pad), high + pad))
    return _merge_ranges(padded)


@dataclass
class IncrementalBurninPlan:
    """Which parts of a re-burn are re-encoded and which are copied."""

    duration: float
    encode_ranges: List[Tuple[float, float]] = field(default_factory=list)
    copy_ranges: List[Tuple[float, float]] = field(default_factory=list)
    fallback_reason: str = ""

    @property
    def incremental(self) -> bool:
        return not self.fallback_reason and bool(self.encode_ranges)

    @property
    def encode_duration(self) -> float:
        return sum(end - start for start, end in self.encode_ranges)

    @property
    def copy_duration(self) -> float:
        return sum(end - start for start, end in self.copy_ranges)

    def as_dict(self) -> Dict:
        return {
            "incremental": self.incremental,
            "duration": round(self.duration, 3),
            "encode_ranges": [[round(s, 3), round(e, 3)] for s, e in self.encode_ranges],
            "copy_ranges": [[round(s, 3), round(e, 3)] for s, e in self.copy_ranges],
            "encode_duration": round(self.encode_duration, 3),
            "copy_duration": round(self.copy_duration, 3),
            "fallback_reason": self.fallback_reason,
        }


def build_incremental_plan(
    duration: float,
    change_ranges: Sequence[Tuple[float, float]],
    keyframes: Sequence[float],
    min_copy_ratio: float = INCREMENTAL_MIN_COPY_RATIO,
) -> IncrementalBurninPlan:
    """Snap changed ranges outward to keyframes and derive the copy gaps.

    Pure so the boundary arithmetic is testable without media. Every refusal
    sets ``fallback_reason`` rather than returning a plan that cannot be
    honoured, so callers report why they re-encoded the whole file.
    """
    if duration <= 0:
        return IncrementalBurninPlan(duration=0.0, fallback_reason="source duration is unknown")
    if not change_ranges:
        return IncrementalBurninPlan(duration=duration, fallback_reason="captions are unchanged")
    if len(keyframes) < 2:
        return IncrementalBurninPlan(
            duration=duration,
            fallback_reason="source has too few keyframes to cut on",
        )

    ordered = sorted(float(k) for k in keyframes)
    snapped: List[Tuple[float, float]] = []
    for start, end in change_ranges:
        before = [k for k in ordered if k <= start]
        after = [k for k in ordered if k >= end]
        low = max(before) if before else 0.0
        high = min(after) if after else duration
        snapped.append((max(0.0, low), min(duration, high)))
    encode_ranges = _merge_ranges(snapped)

    copy_ranges: List[Tuple[float, float]] = []
    cursor = 0.0
    for start, end in encode_ranges:
        if start - cursor > 1e-6:
            copy_ranges.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor > 1e-6:
        copy_ranges.append((cursor, duration))

    plan = IncrementalBurninPlan(
        duration=duration,
        encode_ranges=encode_ranges,
        copy_ranges=copy_ranges,
    )
    if plan.copy_duration / duration < min_copy_ratio:
        plan.fallback_reason = (
            f"only {plan.copy_duration / duration:.0%} of the file is unchanged; "
            "a whole-file render is cheaper than segmenting"
        )
    return plan


# ---------------------------------------------------------------------------
# Burn-in from subtitle file
# ---------------------------------------------------------------------------
def build_subtitle_filter(
    subtitle_path: str,
    font_size: int = 0,
    margin_bottom: int = 0,
    force_style: str = "",
) -> str:
    """Build the ``-vf`` value that draws *subtitle_path* onto the video.

    Escape path for the FFmpeg subtitles/ass filter. FFmpeg parses filter
    option values in two passes (filtergraph then per-filter options), so a
    Windows drive-letter colon must be escaped as ``\\:`` even inside single
    quotes — otherwise the option parser reads ``C`` as the filename and
    fails. escape_filter_path() centralises the verified two-level escaping
    (drive colons, apostrophes, spaces). See tests/test_ffmpeg_escaping.py.
    """
    escaped_sub = escape_filter_path(subtitle_path)
    ext_lower = os.path.splitext(subtitle_path)[1].lower()

    if ext_lower in (".ass", ".ssa"):
        # ASS subtitles: use ass filter (respects all ASS styling)
        return f"ass='{escaped_sub}'"

    # SRT/VTT: use subtitles filter with optional force_style
    style_parts = []
    if font_size > 0:
        style_parts.append(f"FontSize={font_size}")
    if margin_bottom > 0:
        style_parts.append(f"MarginV={margin_bottom}")
    if force_style:
        style_parts.append(force_style)

    if style_parts:
        style_str = ",".join(style_parts)
        return f"subtitles='{escaped_sub}':force_style='{style_str}'"
    return f"subtitles='{escaped_sub}'"


def burnin_subtitles(
    video_path: str,
    subtitle_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    font_size: int = 0,
    margin_bottom: int = 0,
    force_style: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Burn a subtitle file (SRT/ASS/VTT) into video.

    Args:
        subtitle_path: Path to .srt, .ass, or .vtt file.
        font_size: Override font size (0 = use subtitle file's default).
        margin_bottom: Override bottom margin in pixels.
        force_style: FFmpeg force_style override string for SRT/VTT.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        ext = os.path.splitext(video_path)[1] or ".mp4"
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_subtitled{ext}")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.isfile(subtitle_path):
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")

    if on_progress:
        on_progress(10, "Burning subtitles into video...")

    vf = build_subtitle_filter(subtitle_path, font_size, margin_bottom, force_style)

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        # A regular keyframe cadence is what makes a later caption edit
        # re-encodable in pieces: x264's default keyint can leave a short
        # render with a single keyframe, and stream-copy can only cut on one.
        "-force_key_frames", f"expr:gte(t,n_forced*{KEYFRAME_INTERVAL_SECONDS:g})",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path,
    ]
    run_ffmpeg(cmd, timeout=7200, stderr_cap=2000)

    if not os.path.isfile(output_path):
        raise RuntimeError(f"FFmpeg succeeded but output file not created: {output_path}")

    if on_progress:
        on_progress(100, "Subtitles burned in!")
    return output_path


def _keyframe_times(video_path: str) -> List[float]:
    """Keyframe timestamps, or an empty list when they cannot be read.

    Imported lazily: smart_render pulls in the media-profile machinery, and a
    plain whole-file burn-in has no reason to pay for it.
    """
    try:
        from opencut.core.smart_render import _get_keyframes

        return _get_keyframes(video_path)
    except Exception as exc:  # pragma: no cover - probe failures fall back
        logger.warning("keyframe scan failed, incremental burn-in unavailable: %s", exc)
        return []


def _segment_duration(segment_path: str, fps: float) -> float:
    """Video duration of a rendered segment, from its frame count.

    Container duration is the wrong measure here: an MPEG-TS segment holding
    102 video frames at 10fps (10.2s) reports 10.5s because the audio runs
    past the last video frame. Advancing the cursor by that figure skips real
    video, so count the frames the segment actually carries.
    """
    if fps <= 0:
        return 0.0
    try:
        result = subprocess.run(
            [
                get_ffprobe_path(), "-v", "error", "-select_streams", "v:0",
                "-count_packets", "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0", segment_path,
            ],
            capture_output=True, text=True, timeout=120, check=False,
        )
        frames = int((result.stdout or "0").strip().split(",")[0] or 0)
    except Exception:
        return 0.0
    return frames / fps if frames > 0 else 0.0


def plan_incremental_reburn(
    video_path: str,
    old_segments: Sequence[Dict],
    new_segments: Sequence[Dict],
    previous_render: str = "",
    duration: float = 0.0,
) -> IncrementalBurninPlan:
    """Decide whether a re-burn can reuse the previous render."""
    if not previous_render or not os.path.isfile(previous_render):
        return IncrementalBurninPlan(
            duration=duration,
            fallback_reason="no previous render to copy unchanged regions from",
        )
    if not duration:
        try:
            duration = float(get_video_info(video_path).get("duration") or 0.0)
        except Exception:
            duration = 0.0
    changes = caption_change_ranges(old_segments, new_segments)
    # Cut points are constrained by the file the unchanged regions are
    # stream-copied out of, which is the previous render, not the source. Its
    # GOP structure is the encoder's, and planning against the source's
    # keyframes starts a copy segment mid-GOP and silently loses frames.
    return build_incremental_plan(duration, changes, _keyframe_times(previous_render))


def reburn_subtitles_incremental(
    video_path: str,
    previous_render: str,
    subtitle_path: str,
    plan: IncrementalBurninPlan,
    output_path: str,
    font_size: int = 0,
    margin_bottom: int = 0,
    force_style: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """Re-burn only the segments a caption edit touched.

    Changed segments are re-encoded from *video_path* with the new subtitle
    file; unchanged segments are stream-copied out of *previous_render*, so
    they stay bit-identical to what the user already reviewed. Output seeking
    keeps the subtitle filter on the source timeline, so the subtitle file
    needs no per-segment retiming (verified against a fixture whose only cue
    falls inside one segment).
    """
    if not plan.incremental:
        raise ValueError(f"plan is not incremental: {plan.fallback_reason or 'no changed ranges'}")

    vf = build_subtitle_filter(subtitle_path, font_size, margin_bottom, force_style)
    try:
        fps = float(get_video_info(video_path).get("fps") or 0.0)
    except Exception:
        fps = 0.0
    ordered = sorted(
        [("copy", start, end) for start, end in plan.copy_ranges]
        + [("encode", start, end) for start, end in plan.encode_ranges],
        key=lambda item: item[1],
    )

    with tempfile.TemporaryDirectory(prefix="opencut_reburn_") as temp_dir:
        segment_files = []
        # A stream-copied segment does not stop exactly where asked: cutting
        # 0-10s of a 10fps render yields 10.2s, and no epsilon on -t fixes it
        # reliably. So the next segment starts where the previous one actually
        # ended, measured, rather than where the plan said it would. Without
        # this the overshoot is emitted twice and the file grows on every edit.
        cursor = 0.0
        for index, (mode, start, end) in enumerate(ordered):
            start = max(start, cursor)
            if end - start <= 1e-3:
                continue
            segment_path = os.path.join(temp_dir, f"seg_{index:04d}.ts")
            builder = FFmpegCmd()
            if mode == "copy":
                # Input seeking with an explicit duration. Output seeking with
                # -c copy silently returns short segments (a 4s range came back
                # as 2s), because the packet accounting is done against input
                # timestamps after the copy path has already dropped frames.
                builder.pre_input("-ss", str(start))
                builder.pre_input("-t", str(end - start))
                builder.input(previous_render).map("0:v:0", "0:a?")
                builder.copy_streams()
            else:
                # Output seeking, so the subtitles filter still sees the source
                # timeline and the subtitle file needs no per-segment retiming.
                builder.input(video_path)
                builder.seek(start=str(start), end=str(end))
                builder.map("0:v:0", "0:a?")
                builder.video_filter(vf)
                builder.video_codec("libx264", crf=18, preset="medium")
                builder.option(
                    "force_key_frames",
                    f"expr:gte(t,n_forced*{KEYFRAME_INTERVAL_SECONDS:g})",
                )
                builder.audio_codec("aac", bitrate="192k")
            command = (
                builder.option("avoid_negative_ts", "make_zero")
                .option("reset_timestamps", "1")
                .format("mpegts")
                .output(segment_path)
                .build()
            )
            run_ffmpeg(command, timeout=3600)
            segment_files.append(segment_path)
            produced = _segment_duration(segment_path, fps)
            cursor = start + (produced if produced > 0 else (end - start))
            if on_progress:
                percent = 15 + int(70 * (index + 1) / max(len(ordered), 1))
                on_progress(percent, f"Segment {index + 1}/{len(ordered)} ({mode})")

        concat_path = os.path.join(temp_dir, "concat.txt")
        write_concat_list(segment_files, concat_path)
        concat_builder = (
            FFmpegCmd()
            .pre_input("-f", "concat")
            .pre_input("-safe", "0")
            .input(concat_path)
            .map("0:v?", "0:a?")
            .copy_streams()
        )
        if os.path.splitext(output_path)[1].lower() in {".mp4", ".m4v", ".mov"}:
            concat_builder.faststart()
        run_ffmpeg(concat_builder.output(output_path).build(), timeout=900)

    if not os.path.isfile(output_path):
        raise RuntimeError(f"FFmpeg succeeded but output file not created: {output_path}")

    # A stream-copy cut lands on a packet boundary, not an exact timestamp, so
    # each copy segment can be a frame or two long or short. Bounded drift is
    # acceptable; unbounded drift means the segments did not line up, and a
    # timeline of the wrong length is worse than a slow render. Fail here so
    # the caller re-renders the whole file and says why.
    tolerance = max(0.5, (2.0 / fps if fps > 0 else 0.1) * max(len(plan.copy_ranges), 1))
    # Frame count again, not container duration: the audio tail runs past the
    # last video frame and would read as drift that isn't there.
    produced_duration = _segment_duration(output_path, fps)
    drift = abs(produced_duration - plan.duration)
    if produced_duration <= 0 or drift > tolerance:
        os.unlink(output_path)
        raise RuntimeError(
            f"re-burned timeline is {produced_duration:.2f}s against a "
            f"{plan.duration:.2f}s source (drift {drift:.2f}s > {tolerance:.2f}s tolerance)"
        )

    if on_progress:
        on_progress(100, "Captions re-burned!")
    return output_path


# ---------------------------------------------------------------------------
# Burn-in from caption segments (generates ASS then burns)
# ---------------------------------------------------------------------------
BURNIN_STYLES = {
    "default": {
        "label": "Default White",
        "fontname": "Arial",
        "fontsize": 48,
        "primary_color": "&H00FFFFFF",     # white
        "outline_color": "&H00000000",     # black
        "outline": 3,
        "shadow": 1,
        "alignment": 2,
        "margin_v": 40,
    },
    "bold_yellow": {
        "label": "Bold Yellow",
        "fontname": "Impact",
        "fontsize": 56,
        "primary_color": "&H0000FFFF",     # yellow (BGR)
        "outline_color": "&H00000000",
        "outline": 4,
        "shadow": 2,
        "alignment": 2,
        "margin_v": 50,
    },
    "boxed_dark": {
        "label": "Dark Box",
        "fontname": "Arial",
        "fontsize": 44,
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "back_color": "&H80000000",        # semi-transparent black
        "borderstyle": 3,                  # opaque box
        "outline": 2,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 35,
    },
    "neon_cyan": {
        "label": "Neon Cyan",
        "fontname": "Arial Bold",
        "fontsize": 50,
        "primary_color": "&H00FFFF00",     # cyan (BGR)
        "outline_color": "&H00800000",
        "outline": 3,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 45,
    },
    "cinematic_serif": {
        "label": "Cinematic Serif",
        "fontname": "Georgia",
        "fontsize": 42,
        "primary_color": "&H00D2D2DC",     # warm white
        "outline_color": "&H00000000",
        "outline": 0,
        "shadow": 3,
        "alignment": 2,
        "margin_v": 60,
    },
    "top_center": {
        "label": "Top Center",
        "fontname": "Arial",
        "fontsize": 44,
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "outline": 3,
        "shadow": 1,
        "alignment": 8,                    # top center
        "margin_v": 30,
    },
    "rtl_arabic": {
        "label": "RTL Arabic",
        "fontname": "Arial",
        "fontsize": 48,
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "outline": 3,
        "shadow": 1,
        "alignment": 6,                    # bottom right (natural for RTL)
        "margin_v": 40,
    },
}


def burnin_segments(
    video_path: str,
    segments: List[Dict],
    output_path: Optional[str] = None,
    output_dir: str = "",
    style: str = "default",
    on_progress: Optional[Callable] = None,
    previous_render: str = "",
    previous_segments: Optional[Sequence[Dict]] = None,
    render_report: Optional[Dict] = None,
) -> str:
    """
    Burn caption segments directly into video.

    Takes raw segments (with start, end, text) and generates
    a temporary ASS file, then burns it into the video.

    Args:
        segments: List of {"start": float, "end": float, "text": str}.
        style: Burn-in style preset name from BURNIN_STYLES.
        previous_render: An earlier burn-in of the same source. When supplied
            with *previous_segments*, only the regions a caption edit touched
            are re-encoded and the rest is copied from it.
        previous_segments: The caption list that produced *previous_render*.
        render_report: Optional dict updated in place with the plan that ran,
            so callers can report whether the fast path was taken and why not.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        ext = os.path.splitext(video_path)[1] or ".mp4"
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_captioned{ext}")

    if not segments:
        raise ValueError("No caption segments provided")

    style_cfg = BURNIN_STYLES.get(style, BURNIN_STYLES["default"])
    info = get_video_info(video_path)

    if on_progress:
        on_progress(5, "Generating subtitle file...")

    # Generate temporary ASS file
    tmp_ass = tempfile.NamedTemporaryFile(suffix=".ass", mode="w",
                                          encoding="utf-8", delete=False)
    try:
        _write_ass_file(tmp_ass, segments, style_cfg, info)
        tmp_ass.close()

        plan = plan_incremental_reburn(
            video_path,
            previous_segments or (),
            segments,
            previous_render=previous_render,
            duration=float(info.get("duration") or 0.0),
        )
        if render_report is not None:
            render_report.update(plan.as_dict())

        if plan.incremental:
            if on_progress:
                on_progress(10, "Re-burning changed caption regions...")
            try:
                return reburn_subtitles_incremental(
                    video_path,
                    previous_render,
                    tmp_ass.name,
                    plan,
                    output_path,
                    on_progress=on_progress,
                )
            except Exception as exc:
                # A segmented render that cannot be concatenated is a quality
                # problem, not a correctness one: fall back and say so rather
                # than hand back a file assembled from mismatched segments.
                logger.warning("incremental burn-in failed, re-rendering whole file: %s", exc)
                if render_report is not None:
                    render_report["incremental"] = False
                    render_report["fallback_reason"] = f"segment render failed: {exc}"

        if on_progress:
            on_progress(10, "Burning captions into video...")

        result = burnin_subtitles(
            video_path, tmp_ass.name,
            output_path=output_path,
            on_progress=on_progress,
        )
        return result
    finally:
        if os.path.exists(tmp_ass.name):
            os.unlink(tmp_ass.name)


def _write_ass_file(f, segments: List[Dict], style: Dict, info: Dict):
    """Write an ASS subtitle file with styled events."""
    w, h = info["width"], info["height"]
    font = style.get("fontname", "Arial")
    size = style.get("fontsize", 48)
    pc = style.get("primary_color", "&H00FFFFFF")
    oc = style.get("outline_color", "&H00000000")
    bc = style.get("back_color", "&H00000000")
    outline = style.get("outline", 3)
    shadow = style.get("shadow", 1)
    align = style.get("alignment", 2)
    margin_v = style.get("margin_v", 40)
    borderstyle = style.get("borderstyle", 1)

    f.write("[Script Info]\n")
    f.write("Title: OpenCut Burn-in\n")
    f.write("ScriptType: v4.00+\n")
    f.write(f"PlayResX: {w}\n")
    f.write(f"PlayResY: {h}\n")
    f.write("ScaledBorderAndShadow: yes\n\n")

    f.write("[V4+ Styles]\n")
    f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
    f.write(f"Style: Default,{font},{size},{pc},{pc},{oc},{bc},-1,0,0,0,100,100,0,0,{borderstyle},{outline},{shadow},{align},20,20,{margin_v},1\n\n")

    f.write("[Events]\n")
    f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

    for seg in segments:
        start = _format_ass_time(seg.get("start", 0))
        end = _format_ass_time(seg.get("end", 0))
        text = seg.get("text", "").strip()
        # Strip backslashes from source text to prevent ASS override injection
        text = text.replace("\\", "")
        text = text.replace("\n", "\\N")
        text = _re.sub(r'\{[^}]*\}', '', text)
        f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")


def _format_ass_time(seconds: float) -> str:
    """Format seconds to ASS time format H:MM:SS.CC"""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ---------------------------------------------------------------------------
# Available styles
# ---------------------------------------------------------------------------
def get_burnin_styles() -> List[Dict]:
    """Return available burn-in styles."""
    return [
        {"name": k, "label": v["label"]}
        for k, v in BURNIN_STYLES.items()
    ]
