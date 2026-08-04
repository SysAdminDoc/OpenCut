"""Text-assisted subtitle timing resynchronisation.

The resync workflow keeps subtitle text intact and estimates a monotonic timing
transform from matching subtitle cues to timestamped ASR segments.  Matching
the cue text first makes the transform useful for both a constant offset and
clock drift without requiring a second copy of the audio-alignment stack.

The public workflow is deliberately split into preview and write operations:
``resync_subtitles`` never writes a file, while ``write_resynced_srt`` is only
called after the user has reviewed the returned preview.
"""

from __future__ import annotations

import difflib
import html
import math
import os
import re
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from opencut.export.srt import write_srt_text

MAX_SRT_BYTES = 10 * 1024 * 1024
MAX_CUES = 10_000
DEFAULT_FPS = 30.0
DEFAULT_MATCH_THRESHOLD = 0.72
MAX_REFERENCE_LOOKAHEAD = 64
_TIMESTAMP_RE = re.compile(
    r"(?P<hours>\d+):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)(?:[,.](?P<millis>\d{1,3}))?"
)
_TAG_RE = re.compile(r"<[^>]*>")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    """A parsed SRT cue with stable source ordering."""

    index: int
    start: float
    end: float
    text: str


def _parse_timestamp(value: str) -> float:
    match = _TIMESTAMP_RE.search(str(value or ""))
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    millis = (match.group("millis") or "").ljust(3, "0")
    return (
        int(match.group("hours")) * 3600
        + int(match.group("minutes")) * 60
        + int(match.group("seconds"))
        + int(millis or "0") / 1000.0
    )


def _format_timestamp(seconds: float) -> str:
    """Format a non-negative float as an SRT timestamp."""
    if not math.isfinite(float(seconds)):
        raise ValueError("Subtitle timestamps must be finite")
    total_millis = max(0, int(round(float(seconds) * 1000.0)))
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_part, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{millis:03d}"


def parse_srt_text(text: str) -> list[SubtitleCue]:
    """Parse SRT text into cues, rejecting malformed timed entries."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("SRT input is empty")

    blocks = re.split(r"\r?\n[ \t]*\r?\n", text.strip())
    cues: list[SubtitleCue] = []
    for block_number, block in enumerate(blocks, 1):
        lines = block.splitlines()
        timing_index = next(
            (index for index, line in enumerate(lines) if "-->" in line),
            None,
        )
        if timing_index is None:
            raise ValueError(f"SRT block {block_number} has no timing line")

        timing = lines[timing_index].split("-->", 1)
        if len(timing) != 2:
            raise ValueError(f"SRT block {block_number} has an invalid timing line")
        start = _parse_timestamp(timing[0].strip())
        end = _parse_timestamp(timing[1].strip())
        if end <= start:
            raise ValueError(f"SRT block {block_number} ends before it starts")

        cue_text = "\n".join(lines[timing_index + 1:]).strip()
        if not cue_text:
            raise ValueError(f"SRT block {block_number} has no subtitle text")
        try:
            index = int(lines[0].strip())
        except (IndexError, ValueError):
            index = len(cues) + 1
        cues.append(SubtitleCue(index=index, start=start, end=end, text=cue_text))

    if not cues:
        raise ValueError("SRT input contains no cues")
    if len(cues) > MAX_CUES:
        raise ValueError(f"SRT input exceeds the {MAX_CUES} cue limit")
    return cues


def parse_srt_file(path: str | os.PathLike[str]) -> list[SubtitleCue]:
    """Read and parse an SRT file using the standard OpenCut UTF-8 policy."""
    source = Path(path)
    payload = source.read_bytes()
    if len(payload) > MAX_SRT_BYTES:
        raise ValueError(f"SRT input exceeds the {MAX_SRT_BYTES} byte limit")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Existing subtitle libraries often contain Windows-1252 SRT files.
        # Keep the fallback narrow and local; the output is always UTF-8.
        text = payload.decode("cp1252")
    return parse_srt_text(text)


def render_srt_text(cues: Sequence[Mapping[str, Any] | SubtitleCue]) -> str:
    """Render transformed cues as deterministic UTF-8-ready SRT text."""
    lines: list[str] = []
    for output_index, cue in enumerate(cues, 1):
        if isinstance(cue, Mapping):
            start = cue.get("start", 0.0)
            end = cue.get("end", 0.0)
            text = cue.get("text", "")
        else:
            start, end, text = cue.start, cue.end, cue.text
        start_value = max(0.0, float(start))
        end_value = max(start_value + 0.001, float(end))
        lines.extend(
            [
                str(output_index),
                f"{_format_timestamp(start_value)} --> {_format_timestamp(end_value)}",
                str(text).strip(),
                "",
            ]
        )
    if not lines:
        raise ValueError("Cannot render an empty subtitle result")
    return "\n".join(lines)


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _coerce_reference_segments(
    segments: Sequence[Any] | Mapping[str, Any],
) -> list[SubtitleCue]:
    if isinstance(segments, Mapping):
        segments = segments.get("segments", [])
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise ValueError("reference_segments must be a list of timestamped segments")
    if len(segments) > MAX_CUES:
        raise ValueError(f"reference_segments exceeds the {MAX_CUES} segment limit")

    result: list[SubtitleCue] = []
    for index, item in enumerate(segments, 1):
        text = str(_field(item, "text", "") or "").strip()
        if not text:
            continue
        try:
            start = float(_field(item, "start"))
            end = float(_field(item, "end"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        result.append(SubtitleCue(index=index, start=start, end=end, text=text))
    if not result:
        raise ValueError("reference_segments contains no usable timestamped text")
    return result


def _normalise_text(text: str) -> str:
    value = html.unescape(str(text or ""))
    value = _TAG_RE.sub(" ", value)
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char if char.isalnum() else " " for char in value).strip()


def _text_similarity(left: str, right: str) -> float:
    left_normalised = _normalise_text(left)
    right_normalised = _normalise_text(right)
    if not left_normalised or not right_normalised:
        return 0.0
    if left_normalised == right_normalised:
        return 1.0
    sequence_score = difflib.SequenceMatcher(
        None, left_normalised, right_normalised, autojunk=False
    ).ratio()
    left_tokens = set(left_normalised.split())
    right_tokens = set(right_normalised.split())
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    return max(sequence_score, overlap)


def _match_cues(
    source_cues: Sequence[SubtitleCue],
    reference_cues: Sequence[SubtitleCue],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Greedily match text in source order while keeping references monotonic."""
    matches: list[dict[str, Any]] = []
    reference_cursor = 0
    for source in source_cues:
        best: tuple[float, int] | None = None
        last_candidate = min(
            len(reference_cues), reference_cursor + MAX_REFERENCE_LOOKAHEAD
        )
        for reference_index in range(reference_cursor, last_candidate):
            score = _text_similarity(source.text, reference_cues[reference_index].text)
            if best is None or score > best[0]:
                best = (score, reference_index)
            if score == 1.0:
                break
        if best is None or best[0] < threshold:
            continue
        score, reference_index = best
        reference = reference_cues[reference_index]
        matches.append(
            {
                "subtitle_index": source.index,
                "reference_index": reference.index,
                "source_start": source.start,
                "source_end": source.end,
                "reference_start": reference.start,
                "reference_end": reference.end,
                "score": round(score, 4),
                "text": source.text,
            }
        )
        reference_cursor = reference_index + 1
    return matches


def _fit_affine(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        raise ValueError("At least one timing point is required")
    if len(points) < 2:
        return 1.0, points[0][1] - points[0][0]
    x_mean = statistics.fmean(point[0] for point in points)
    y_mean = statistics.fmean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator <= 1e-12:
        return 1.0, statistics.median(y - x for x, y in points)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    return slope, y_mean - slope * x_mean


def _fit_transform(
    matches: Sequence[Mapping[str, Any]],
    *,
    fps: float,
) -> tuple[float, float, float]:
    """Return ``(rate, offset, max_boundary_error)`` for matched cues."""
    if len(matches) == 1:
        match = matches[0]
        rate = 1.0
        offset = statistics.median(
            (
                float(match["reference_start"]) - float(match["source_start"]),
                float(match["reference_end"]) - float(match["source_end"]),
            )
        )
    else:
        points = [
            (float(match[source_key]), float(match[reference_key]))
            for match in matches
            for source_key, reference_key in (
                ("source_start", "reference_start"),
                ("source_end", "reference_end"),
            )
        ]
        rate, offset = _fit_affine(points)
        tolerance = max(0.08, 4.0 / fps)
        inliers = [
            point
            for point in points
            if abs(point[1] - (rate * point[0] + offset)) <= tolerance
        ]
        if len(inliers) >= 4 and len(inliers) < len(points):
            rate, offset = _fit_affine(inliers)

    if not math.isfinite(rate) or not math.isfinite(offset):
        raise ValueError("Could not estimate a finite subtitle timing transform")
    if rate < 0.5 or rate > 2.0:
        raise ValueError(
            f"Estimated subtitle clock rate {rate:.4f} is outside the safe 0.5–2.0 range"
        )
    errors = [
        abs(float(match[reference_key]) - (rate * float(match[source_key]) + offset))
        for match in matches
        for source_key, reference_key in (
            ("source_start", "reference_start"),
            ("source_end", "reference_end"),
        )
    ]
    return rate, offset, max(errors or [0.0])


def _load_reference_segments(
    video_path: str,
    *,
    model: str,
    language: str | None,
    transcriber: Callable[[str], Any] | None,
) -> list[SubtitleCue]:
    if transcriber is not None:
        result = transcriber(video_path)
    else:
        from opencut.core.captions import transcribe
        from opencut.utils.config import CaptionConfig

        result = transcribe(
            video_path,
            config=CaptionConfig(
                model=model,
                language=language,
                word_timestamps=False,
            ),
        )
    return _coerce_reference_segments(
        result if isinstance(result, Mapping) else getattr(result, "segments", result)
    )


def resync_subtitles(
    srt_path: str | os.PathLike[str],
    *,
    reference_segments: Sequence[Any] | Mapping[str, Any] | None = None,
    video_path: str | os.PathLike[str] | None = None,
    fps: float = DEFAULT_FPS,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    model: str = "base",
    language: str | None = None,
    transcriber: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Build a no-write subtitle resync preview.

    ``reference_segments`` is the timestamped text source used for matching.
    When omitted, ``video_path`` is transcribed with the existing OpenCut ASR
    contract.  A single match produces a constant offset; two or more matches
    fit an affine clock transform so accumulated drift is corrected as well.
    """
    try:
        fps_value = float(fps)
        threshold_value = float(match_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps and match_threshold must be numeric") from exc
    if not math.isfinite(fps_value) or not 1.0 <= fps_value <= 120.0:
        raise ValueError("fps must be between 1 and 120")
    if not math.isfinite(threshold_value) or not 0.5 <= threshold_value <= 1.0:
        raise ValueError("match_threshold must be between 0.5 and 1.0")

    source_path = os.path.realpath(os.fspath(srt_path))
    source_cues = parse_srt_file(source_path)
    if reference_segments is None:
        if not video_path:
            raise ValueError("Provide video_path or reference_segments")
        reference_cues = _load_reference_segments(
            os.fspath(video_path),
            model=str(model or "base"),
            language=language,
            transcriber=transcriber,
        )
        reference_source = "transcription"
    else:
        reference_cues = _coerce_reference_segments(reference_segments)
        reference_source = "provided"

    matches = _match_cues(
        source_cues,
        reference_cues,
        threshold=threshold_value,
    )
    if not matches:
        raise ValueError(
            "No subtitle cues matched the reference transcript; check the language or source file"
        )
    rate, offset, max_boundary_error = _fit_transform(matches, fps=fps_value)

    matched_indices = {int(match["subtitle_index"]) for match in matches}
    transformed_cues: list[dict[str, Any]] = []
    for cue in source_cues:
        transformed_cues.append(
            {
                "index": cue.index,
                "text": cue.text,
                "source_start": round(cue.start, 6),
                "source_end": round(cue.end, 6),
                "start": round(max(0.0, rate * cue.start + offset), 6),
                "end": round(max(0.0, rate * cue.end + offset), 6),
                "matched": cue.index in matched_indices,
            }
        )

    frame_duration = 1.0 / fps_value
    return {
        "source_path": source_path,
        "reference_source": reference_source,
        "fps": fps_value,
        "frame_duration": frame_duration,
        "match_threshold": threshold_value,
        "source_cue_count": len(source_cues),
        "reference_segment_count": len(reference_cues),
        "matched_count": len(matches),
        "unmatched_count": len(source_cues) - len(matches),
        "fit_mode": "constant" if len(matches) == 1 else "affine",
        "rate": round(rate, 9),
        "offset_seconds": round(offset, 9),
        "drift_ppm": round((rate - 1.0) * 1_000_000.0, 3),
        "max_boundary_error": round(max_boundary_error, 9),
        "within_one_frame": max_boundary_error <= frame_duration + 1e-9,
        "matches": matches,
        "cues": transformed_cues,
        "preview_srt": render_srt_text(transformed_cues),
    }


def write_resynced_srt(
    preview: Mapping[str, Any],
    output_path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a previously generated preview after explicit apply approval."""
    cues = preview.get("cues")
    if not isinstance(cues, Sequence) or isinstance(cues, (str, bytes)) or not cues:
        raise ValueError("Preview does not contain transformed subtitle cues")
    destination = os.path.realpath(os.fspath(output_path))
    source_path = os.path.realpath(os.fspath(preview.get("source_path", "")))
    if destination == source_path and not overwrite:
        raise ValueError("Refusing to overwrite the source SRT without overwrite=true")
    if os.path.exists(destination) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}; pass overwrite=true to replace it"
        )
    parent = os.path.dirname(destination)
    if parent and not os.path.isdir(parent):
        raise ValueError(f"Output directory does not exist: {parent}")

    text = render_srt_text(cues)
    write_srt_text(destination, text)
    return {
        "output_path": destination,
        "cue_count": len(cues),
        "bytes_written": os.path.getsize(destination),
    }


__all__ = [
    "SubtitleCue",
    "parse_srt_file",
    "parse_srt_text",
    "render_srt_text",
    "resync_subtitles",
    "write_resynced_srt",
]
