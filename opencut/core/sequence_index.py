"""
Sequence Index panel backend (RESEARCH_FEATURE_PLAN_2026-05-25 Q7 / F273).

Adobe Premiere 26.x ships a "Sequence Index" panel — a spreadsheet view
of every clip in the active timeline with timecode, duration, track,
effects, and (when available) a transcript excerpt. This module
implements the OpenCut equivalent: a sequence-data normaliser plus a
search/filter/sort surface that the CEP and UXP panels can render as a
sortable table.

Input shape matches the JSON returned by ``host/index.jsx::ocGetSequenceInfo()``:

  {
    "name": "Sequence 01",
    "duration": 305.5,
    "fps": 24.0,
    "width": 1920,
    "height": 1080,
    "videoTracks": [
      {"index": 0, "clips": [
        {"name": "...", "path": "...", "start": 0.0, "end": 12.5, "effects": [...]},
      ]}
    ],
    "audioTracks": [
      {"index": 0, "clips": [{"name": "...", "path": "...", "start": ..., "end": ...}]}
    ],
    "markers": [{"time": 4.2, "name": "intro", "type": "comment", "color": 0}]
  }

Output is a flat list of ``IndexRow`` objects (one per clip), each
augmented with timecode strings, duration, track type/index, and any
transcript excerpt overlapping the clip's window.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------
@dataclass
class IndexRow:
    track_type: str               # "video" | "audio"
    track_index: int              # 0-based
    clip_index: int               # position within track
    name: str
    path: str
    start_s: float
    end_s: float
    duration_s: float
    timecode_in: str              # "HH:MM:SS:FF"
    timecode_out: str
    effects: List[str] = field(default_factory=list)
    rating: int = 0               # 0..5 (panel-side; 0 = unrated)
    tags: List[str] = field(default_factory=list)
    transcript_excerpt: str = ""  # joined text of overlapping transcript segments

    def to_dict(self) -> dict:
        return {
            "track_type": self.track_type,
            "track_index": self.track_index,
            "clip_index": self.clip_index,
            "name": self.name,
            "path": self.path,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "timecode_in": self.timecode_in,
            "timecode_out": self.timecode_out,
            "effects": list(self.effects),
            "rating": int(self.rating),
            "tags": list(self.tags),
            "transcript_excerpt": self.transcript_excerpt,
        }


@dataclass
class SequenceIndexResult:
    sequence_name: str = ""
    fps: float = 24.0
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    rows: List[IndexRow] = field(default_factory=list)
    marker_count: int = 0
    total_rows: int = 0

    # Flask jsonify protocol.
    def __getitem__(self, key: str) -> Any:
        if key == "rows":
            return [r.to_dict() for r in self.rows]
        return getattr(self, key)

    def keys(self):
        return (
            "sequence_name", "fps", "duration_s", "width", "height",
            "rows", "marker_count", "total_rows",
        )

    def __contains__(self, key: str) -> bool:
        return key in self.keys()


# ---------------------------------------------------------------------------
# Public availability check (matches the `check_X_available()` convention)
# ---------------------------------------------------------------------------
def check_sequence_index_available() -> bool:
    """Always True — pure-Python, no optional deps."""
    return True


INSTALL_HINT = "Sequence Index is pure stdlib; nothing to install."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seconds_to_timecode(seconds: float, fps: float) -> str:
    """Format seconds as ``HH:MM:SS:FF`` (drop-frame not handled — caller
    can adjust offline)."""
    if seconds is None or seconds < 0 or not isinstance(seconds, (int, float)):
        seconds = 0.0
    if fps <= 0:
        fps = 24.0
    total_frames = int(round(seconds * fps))
    fpr = int(round(fps))
    if fpr <= 0:
        fpr = 24
    hh, rem = divmod(total_frames, fpr * 3600)
    mm, rem = divmod(rem, fpr * 60)
    ss, ff = divmod(rem, fpr)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _coerce_seq(payload: Any) -> dict:
    """Normalize the sequence JSON shape so partial / malformed payloads
    still produce a usable index. Always returns every key — callers
    can rely on ``seq["fps"]`` etc. without further None-checking.
    """
    if not isinstance(payload, dict):
        payload = {}
    return {
        "name": str(payload.get("name") or ""),
        "duration": _safe_float(payload.get("duration"), 0.0),
        "fps": _safe_float(payload.get("fps"), 24.0),
        "width": _safe_int(payload.get("width"), 0),
        "height": _safe_int(payload.get("height"), 0),
        "videoTracks": payload.get("videoTracks") if isinstance(payload.get("videoTracks"), list) else [],
        "audioTracks": payload.get("audioTracks") if isinstance(payload.get("audioTracks"), list) else [],
        "markers": payload.get("markers") if isinstance(payload.get("markers"), list) else [],
    }


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _transcript_excerpt_for(
    clip_start: float,
    clip_end: float,
    transcript_segments: Optional[List[dict]],
    excerpt_chars: int = 240,
) -> str:
    """Concatenate the text of every transcript segment that overlaps the
    clip's window. Caller can cap the length via ``excerpt_chars``."""
    if not transcript_segments:
        return ""
    parts: List[str] = []
    for seg in transcript_segments:
        if not isinstance(seg, dict):
            continue
        s = _safe_float(seg.get("start"), 0.0)
        e = _safe_float(seg.get("end"), s)
        if e < clip_start or s > clip_end:
            continue
        text = str(seg.get("text") or "").strip()
        if text:
            parts.append(text)
    joined = " ".join(parts).strip()
    if excerpt_chars > 0 and len(joined) > excerpt_chars:
        joined = joined[: excerpt_chars - 1].rstrip() + "…"  # ellipsis
    return joined


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_index(
    sequence_payload: dict,
    transcript_segments: Optional[List[dict]] = None,
    ratings: Optional[dict] = None,
    tags: Optional[dict] = None,
    excerpt_chars: int = 240,
) -> SequenceIndexResult:
    """Convert a Premiere sequence JSON blob to a flat row list.

    Args:
        sequence_payload: JSON shape from ``ocGetSequenceInfo`` (see module
            docstring). Tolerant of missing fields.
        transcript_segments: Optional ``[{start, end, text}, ...]`` covering
            the source-timeline window. Each clip gets a ``transcript_excerpt``
            built from overlapping segments.
        ratings: Optional ``{clip_path: int}`` — 0..5 star ratings keyed by
            clip ``path``. Defaults to 0.
        tags: Optional ``{clip_path: [str, ...]}`` — free-form tags per clip.
        excerpt_chars: Cap transcript excerpt length per clip (0 = no cap).

    Returns:
        :class:`SequenceIndexResult`.
    """
    seq = _coerce_seq(sequence_payload)
    fps = seq["fps"] or 24.0
    ratings = ratings or {}
    tags = tags or {}

    rows: List[IndexRow] = []

    for vt in seq["videoTracks"]:
        if not isinstance(vt, dict):
            continue
        ti = _safe_int(vt.get("index"), 0)
        for ci, clip in enumerate(vt.get("clips") or []):
            if not isinstance(clip, dict):
                continue
            start = _safe_float(clip.get("start"), 0.0)
            end = _safe_float(clip.get("end"), start)
            path = str(clip.get("path") or "")
            rows.append(IndexRow(
                track_type="video",
                track_index=ti,
                clip_index=ci,
                name=str(clip.get("name") or ""),
                path=path,
                start_s=start,
                end_s=end,
                duration_s=max(0.0, end - start),
                timecode_in=_seconds_to_timecode(start, fps),
                timecode_out=_seconds_to_timecode(end, fps),
                effects=[str(x) for x in (clip.get("effects") or []) if x],
                rating=_safe_int(ratings.get(path), 0),
                tags=list(tags.get(path) or []),
                transcript_excerpt=_transcript_excerpt_for(start, end, transcript_segments, excerpt_chars),
            ))

    for at in seq["audioTracks"]:
        if not isinstance(at, dict):
            continue
        ti = _safe_int(at.get("index"), 0)
        for ci, clip in enumerate(at.get("clips") or []):
            if not isinstance(clip, dict):
                continue
            start = _safe_float(clip.get("start"), 0.0)
            end = _safe_float(clip.get("end"), start)
            path = str(clip.get("path") or "")
            rows.append(IndexRow(
                track_type="audio",
                track_index=ti,
                clip_index=ci,
                name=str(clip.get("name") or ""),
                path=path,
                start_s=start,
                end_s=end,
                duration_s=max(0.0, end - start),
                timecode_in=_seconds_to_timecode(start, fps),
                timecode_out=_seconds_to_timecode(end, fps),
                effects=[],  # audio clips don't ship 'effects' in the JSX payload
                rating=_safe_int(ratings.get(path), 0),
                tags=list(tags.get(path) or []),
                transcript_excerpt=_transcript_excerpt_for(start, end, transcript_segments, excerpt_chars),
            ))

    return SequenceIndexResult(
        sequence_name=seq["name"],
        fps=fps,
        duration_s=seq["duration"],
        width=seq["width"],
        height=seq["height"],
        rows=rows,
        marker_count=len(seq["markers"]),
        total_rows=len(rows),
    )


# Sort + filter on top of a built index (so the panel can re-sort without
# re-walking the sequence).
SORT_KEYS = frozenset({
    "track_type", "track_index", "clip_index", "name", "path",
    "start_s", "end_s", "duration_s", "rating",
})


def sort_rows(rows: List[IndexRow], key: str, descending: bool = False) -> List[IndexRow]:
    """Stable sort by one of :data:`SORT_KEYS`.

    Raises ``ValueError`` on unknown keys so the frontend can't drift.
    """
    if key not in SORT_KEYS:
        raise ValueError(f"Unknown sort key '{key}'. Valid: {sorted(SORT_KEYS)}")
    return sorted(rows, key=lambda r: getattr(r, key), reverse=descending)


def filter_rows(
    rows: List[IndexRow],
    query: str = "",
    track_type: Optional[str] = None,
    min_rating: int = 0,
    has_effects: Optional[bool] = None,
) -> List[IndexRow]:
    """Free-text + faceted filter.

    Args:
        rows: Rows from :func:`build_index`.
        query: Case-insensitive substring matched against name / path /
            transcript_excerpt / tags / effects.
        track_type: ``"video"`` | ``"audio"`` | None for both.
        min_rating: Drop rows with rating below this.
        has_effects: True = only clips with effects; False = only without.
    """
    q = (query or "").strip().lower()
    out: List[IndexRow] = []
    for r in rows:
        if track_type and r.track_type != track_type:
            continue
        if r.rating < min_rating:
            continue
        if has_effects is True and not r.effects:
            continue
        if has_effects is False and r.effects:
            continue
        if q:
            haystack = " ".join([
                r.name.lower(),
                r.path.lower(),
                r.transcript_excerpt.lower(),
                " ".join(t.lower() for t in r.tags),
                " ".join(e.lower() for e in r.effects),
            ])
            if q not in haystack:
                continue
        out.append(r)
    return out


__all__ = [
    "IndexRow",
    "SequenceIndexResult",
    "SORT_KEYS",
    "INSTALL_HINT",
    "check_sequence_index_available",
    "build_index",
    "sort_rows",
    "filter_rows",
]
