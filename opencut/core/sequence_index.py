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

import hashlib
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
    locator_id: str = ""          # stable timeline-instance key for ratings/tags
    host_locators: dict[str, Any] = field(default_factory=dict)

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
            "locator_id": self.locator_id,
            "host_locators": dict(self.host_locators),
        }


@dataclass
class SequenceIndexResult:
    sequence_name: str = ""
    sequence_guid: str = ""
    fps: float = 24.0
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    rows: List[IndexRow] = field(default_factory=list)
    markers: List[dict[str, Any]] = field(default_factory=list)
    marker_count: int = 0
    total_rows: int = 0

    # Flask jsonify protocol.
    def __getitem__(self, key: str) -> Any:
        if key == "rows":
            return [r.to_dict() for r in self.rows]
        if key == "markers":
            return [dict(marker) for marker in self.markers]
        return getattr(self, key)

    def keys(self):
        return (
            "sequence_name", "sequence_guid", "fps", "duration_s", "width", "height",
            "rows", "markers", "marker_count", "total_rows",
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
    video_tracks = payload.get("videoTracks")
    if not isinstance(video_tracks, list):
        video_tracks = payload.get("video_tracks")
    audio_tracks = payload.get("audioTracks")
    if not isinstance(audio_tracks, list):
        audio_tracks = payload.get("audio_tracks")
    return {
        "name": str(payload.get("name") or ""),
        "sequence_guid": str(
            payload.get("sequence_guid")
            or payload.get("sequenceGuid")
            or payload.get("guid")
            or payload.get("id")
            or ""
        ),
        "duration": _safe_float(payload.get("duration"), 0.0),
        "fps": _safe_float(payload.get("fps", payload.get("framerate")), 24.0),
        "width": _safe_int(payload.get("width"), 0),
        "height": _safe_int(payload.get("height"), 0),
        "videoTracks": video_tracks if isinstance(video_tracks, list) else [],
        "audioTracks": audio_tracks if isinstance(audio_tracks, list) else [],
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


def _host_locators(
    *,
    sequence_name: str,
    sequence_guid: str,
    track_type: str,
    track_index: int,
    clip_index: int,
    name: str,
    path: str,
    start_s: float,
    end_s: float,
    track_item_id: str = "",
    project_item_id: str = "",
) -> dict[str, Any]:
    return {
        "schema": "opencut.sequence_index_locator",
        "sequence_guid": sequence_guid,
        "sequence_name": sequence_name,
        "track_type": track_type,
        "track_index": track_index,
        "clip_index": clip_index,
        "name": name,
        "path": path,
        "start_s": start_s,
        "end_s": end_s,
        "track_item_id": track_item_id,
        "project_item_id": project_item_id,
    }


def _locator_id(locators: dict[str, Any]) -> str:
    parts = [
        str(locators.get("sequence_guid", "")),
        str(locators.get("sequence_name", "")),
        str(locators.get("track_type", "")),
        str(locators.get("track_index", 0)),
        str(locators.get("clip_index", 0)),
        f"{float(locators.get('start_s', 0.0) or 0.0):.3f}",
        f"{float(locators.get('end_s', 0.0) or 0.0):.3f}",
        str(locators.get("path", "")),
        str(locators.get("name", "")),
    ]
    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=8).hexdigest()
    return f"seqidx:{digest}"


def _metadata_value(mapping: dict, locator_id: str, path: str, default: Any) -> Any:
    if locator_id and locator_id in mapping:
        return mapping[locator_id]
    if path and path in mapping:
        return mapping[path]
    return default


def _marker_host_locators(seq: dict[str, Any], marker: dict[str, Any], index: int, time_s: float) -> dict[str, Any]:
    locators = dict(marker.get("host_locators") or {}) if isinstance(marker.get("host_locators"), dict) else {}
    marker_id = marker.get("marker_id") or marker.get("id") or marker.get("guid") or marker.get("nodeId")
    marker_type = str(marker.get("type") or marker.get("marker_type") or "")
    locators.update({
        "schema": "opencut.sequence_marker_locator",
        "sequence_guid": seq["sequence_guid"],
        "sequence_name": seq["name"],
        "marker_index": index,
        "marker_id": str(marker_id or ""),
        "marker_time_s": time_s,
        "marker_name": str(marker.get("name") or marker.get("label") or ""),
        "marker_type": marker_type,
    })
    return locators


def _normalise_markers(seq: dict[str, Any]) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for index, marker in enumerate(seq["markers"]):
        if not isinstance(marker, dict):
            continue
        time_s = _safe_float(marker.get("time", marker.get("start", marker.get("start_s"))), 0.0)
        markers.append({
            "time": time_s,
            "name": str(marker.get("name") or marker.get("label") or ""),
            "type": str(marker.get("type") or marker.get("marker_type") or ""),
            "color": marker.get("color", marker.get("colorIndex")),
            "duration": _safe_float(marker.get("duration", marker.get("duration_s")), 0.0),
            "comment": str(marker.get("comment") or marker.get("comments") or ""),
            "ticks": marker.get("ticks"),
            "host_locators": _marker_host_locators(seq, marker, index, time_s),
        })
    return markers


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
        ratings: Optional ``{locator_id|clip_path: int}`` — 0..5 star ratings
            keyed by sequence locator first, with clip ``path`` as fallback.
        tags: Optional ``{locator_id|clip_path: [str, ...]}`` — free-form tags.
        excerpt_chars: Cap transcript excerpt length per clip (0 = no cap).

    Returns:
        :class:`SequenceIndexResult`.
    """
    seq = _coerce_seq(sequence_payload)
    fps = seq["fps"] or 24.0
    ratings = ratings or {}
    tags = tags or {}

    rows: List[IndexRow] = []
    markers = _normalise_markers(seq)

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
            name = str(clip.get("name") or "")
            track_item_id = str(clip.get("nodeId") or clip.get("node_id") or clip.get("id") or clip.get("guid") or "")
            project_item_id = str(
                clip.get("projectItemId") or clip.get("project_item_id") or clip.get("projectNodeId") or ""
            )
            locators = _host_locators(
                sequence_name=seq["name"],
                sequence_guid=seq["sequence_guid"],
                track_type="video",
                track_index=ti,
                clip_index=ci,
                name=name,
                path=path,
                start_s=start,
                end_s=end,
                track_item_id=track_item_id,
                project_item_id=project_item_id,
            )
            locator_id = _locator_id(locators)
            rows.append(IndexRow(
                track_type="video",
                track_index=ti,
                clip_index=ci,
                name=name,
                path=path,
                start_s=start,
                end_s=end,
                duration_s=max(0.0, end - start),
                timecode_in=_seconds_to_timecode(start, fps),
                timecode_out=_seconds_to_timecode(end, fps),
                effects=[str(x) for x in (clip.get("effects") or []) if x],
                rating=_safe_int(_metadata_value(ratings, locator_id, path, 0), 0),
                tags=list(_metadata_value(tags, locator_id, path, []) or []),
                transcript_excerpt=_transcript_excerpt_for(start, end, transcript_segments, excerpt_chars),
                locator_id=locator_id,
                host_locators=locators,
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
            name = str(clip.get("name") or "")
            track_item_id = str(clip.get("nodeId") or clip.get("node_id") or clip.get("id") or clip.get("guid") or "")
            project_item_id = str(
                clip.get("projectItemId") or clip.get("project_item_id") or clip.get("projectNodeId") or ""
            )
            locators = _host_locators(
                sequence_name=seq["name"],
                sequence_guid=seq["sequence_guid"],
                track_type="audio",
                track_index=ti,
                clip_index=ci,
                name=name,
                path=path,
                start_s=start,
                end_s=end,
                track_item_id=track_item_id,
                project_item_id=project_item_id,
            )
            locator_id = _locator_id(locators)
            rows.append(IndexRow(
                track_type="audio",
                track_index=ti,
                clip_index=ci,
                name=name,
                path=path,
                start_s=start,
                end_s=end,
                duration_s=max(0.0, end - start),
                timecode_in=_seconds_to_timecode(start, fps),
                timecode_out=_seconds_to_timecode(end, fps),
                effects=[],  # audio clips don't ship 'effects' in the JSX payload
                rating=_safe_int(_metadata_value(ratings, locator_id, path, 0), 0),
                tags=list(_metadata_value(tags, locator_id, path, []) or []),
                transcript_excerpt=_transcript_excerpt_for(start, end, transcript_segments, excerpt_chars),
                locator_id=locator_id,
                host_locators=locators,
            ))

    return SequenceIndexResult(
        sequence_name=seq["name"],
        sequence_guid=seq["sequence_guid"],
        fps=fps,
        duration_s=seq["duration"],
        width=seq["width"],
        height=seq["height"],
        rows=rows,
        markers=markers,
        marker_count=len(markers),
        total_rows=len(rows),
    )


# Sort + filter on top of a built index (so the panel can re-sort without
# re-walking the sequence).
SORT_KEYS = frozenset({
    "track_type", "track_index", "clip_index", "name", "path", "locator_id",
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
                r.locator_id.lower(),
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
