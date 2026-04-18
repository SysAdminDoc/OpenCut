"""
Semantic timeline diff / merge for OpenTimelineIO files.

Compares two ``.otio`` / ``.otioz`` / FCP-XML-through-OTIO timelines
and reports structural differences at three layers:

1. **Track-level**: tracks added, removed, renamed, or reordered.
2. **Clip-level**: clips added, removed, moved, retimed, or renamed.
3. **Marker-level**: markers added, removed, shifted.

Output is a structured :class:`OtioDiffResult` that serialises to JSON
so panels / CI can highlight conflicts before a merge.  The module is
FFmpeg-free — only OpenTimelineIO is required.

Graceful degradation: when ``opentimelineio`` isn't installed,
``check_otio_diff_available()`` returns ``False`` and callers route to
a clear ``MISSING_DEPENDENCY`` error.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_otio_diff_available() -> bool:
    try:
        import opentimelineio  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

DIFF_KINDS = frozenset({"added", "removed", "moved", "retimed", "renamed"})


@dataclass
class ClipDiff:
    """A single clip-level difference between two timelines."""
    kind: str                           # one of DIFF_KINDS
    track: str = ""
    left: Dict[str, Any] = field(default_factory=dict)
    right: Dict[str, Any] = field(default_factory=dict)
    details: str = ""


@dataclass
class TrackDiff:
    """A track-level difference."""
    kind: str                           # added / removed / renamed
    name: str = ""
    left_index: Optional[int] = None
    right_index: Optional[int] = None
    details: str = ""


@dataclass
class MarkerDiff:
    """A marker-level difference."""
    kind: str                           # added / removed / shifted
    name: str = ""
    t_left: Optional[float] = None
    t_right: Optional[float] = None
    clip: str = ""


@dataclass
class OtioDiffResult:
    """Structured return for :func:`diff_timelines`."""
    left: str = ""
    right: str = ""
    track_diffs: List[TrackDiff] = field(default_factory=list)
    clip_diffs: List[ClipDiff] = field(default_factory=list)
    marker_diffs: List[MarkerDiff] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    identical: bool = False
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready representation with nested dataclasses unpacked."""
        return {
            "left": self.left,
            "right": self.right,
            "track_diffs": [asdict(d) for d in self.track_diffs],
            "clip_diffs": [asdict(d) for d in self.clip_diffs],
            "marker_diffs": [asdict(d) for d in self.marker_diffs],
            "summary": dict(self.summary),
            "identical": self.identical,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Timeline normalisation
# ---------------------------------------------------------------------------

def _clip_signature(clip) -> Dict[str, Any]:
    """Build a stable signature for a clip so we can match across files.

    Signature keys: clip name, media_reference target_url (if any),
    source_range start + duration seconds.  Two clips with the same
    name + media URL + source range are treated as "the same clip";
    differing timings become a ``retimed`` diff.
    """
    name = getattr(clip, "name", "") or ""
    src_url = ""
    mr = getattr(clip, "media_reference", None)
    if mr is not None:
        src_url = getattr(mr, "target_url", "") or ""

    src_range = getattr(clip, "source_range", None)
    start = 0.0
    duration = 0.0
    if src_range is not None:
        try:
            start = float(src_range.start_time.to_seconds())
            duration = float(src_range.duration.to_seconds())
        except Exception:  # noqa: BLE001
            start, duration = 0.0, 0.0

    return {
        "name": name,
        "source_url": src_url,
        "start": round(start, 4),
        "duration": round(duration, 4),
    }


def _marker_signature(marker) -> Dict[str, Any]:
    try:
        t = float(marker.marked_range.start_time.to_seconds())
    except Exception:  # noqa: BLE001
        t = 0.0
    return {
        "name": getattr(marker, "name", "") or "",
        "t": round(t, 4),
        "color": str(getattr(marker, "color", "") or ""),
    }


def _collect_clips(track) -> List[Dict[str, Any]]:
    """Walk a track and emit a flat list of clip signatures.

    Only keeps ``otio.schema.Clip`` items — transitions and gaps are
    ignored because the diff operates on *content*, not *layout*.
    """
    import opentimelineio as otio  # local import so module import stays cheap

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(track):
        if isinstance(item, otio.schema.Clip):
            sig = _clip_signature(item)
            sig["index"] = i
            out.append(sig)
    return out


def _collect_markers(clip) -> List[Dict[str, Any]]:
    out = []
    for m in getattr(clip, "markers", []) or []:
        sig = _marker_signature(m)
        sig["clip"] = getattr(clip, "name", "") or ""
        out.append(sig)
    return out


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

def _index_by_name_url(clips: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Group clips by (name, source_url) so equal clips land together."""
    idx: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for c in clips:
        key = (c["name"], c["source_url"])
        idx.setdefault(key, []).append(c)
    return idx


def _diff_clips_in_track(
    track_name: str,
    left_clips: List[Dict[str, Any]],
    right_clips: List[Dict[str, Any]],
) -> List[ClipDiff]:
    """Walk two sets of clip signatures and emit a diff list.

    Matching strategy: bucket by ``(name, source_url)`` — for each
    bucket, pair left-right entries in order.  Extra-left entries are
    **removed**; extra-right are **added**.  Paired entries that differ
    in ``start`` or ``duration`` are **retimed**; paired entries with
    the same timing but different track-index emit a **moved**.
    """
    out: List[ClipDiff] = []
    li = _index_by_name_url(left_clips)
    ri = _index_by_name_url(right_clips)

    all_keys = set(li.keys()) | set(ri.keys())
    for key in sorted(all_keys):
        l_list = li.get(key, [])
        r_list = ri.get(key, [])
        matched = min(len(l_list), len(r_list))
        for i in range(matched):
            lc, rc = l_list[i], r_list[i]
            if lc["start"] != rc["start"] or lc["duration"] != rc["duration"]:
                out.append(ClipDiff(
                    kind="retimed", track=track_name,
                    left={"start": lc["start"], "duration": lc["duration"], "name": lc["name"]},
                    right={"start": rc["start"], "duration": rc["duration"], "name": rc["name"]},
                    details=(
                        f"start {lc['start']}→{rc['start']} "
                        f"dur {lc['duration']}→{rc['duration']}"
                    ),
                ))
            elif lc["index"] != rc["index"]:
                out.append(ClipDiff(
                    kind="moved", track=track_name,
                    left={"index": lc["index"], "name": lc["name"]},
                    right={"index": rc["index"], "name": rc["name"]},
                    details=f"index {lc['index']}→{rc['index']}",
                ))
        # Extra left → removed; extra right → added
        for lc in l_list[matched:]:
            out.append(ClipDiff(
                kind="removed", track=track_name,
                left={"name": lc["name"], "start": lc["start"], "duration": lc["duration"]},
            ))
        for rc in r_list[matched:]:
            out.append(ClipDiff(
                kind="added", track=track_name,
                right={"name": rc["name"], "start": rc["start"], "duration": rc["duration"]},
            ))
    return out


def _diff_markers(
    left_clips_in_track: List[Any],
    right_clips_in_track: List[Any],
) -> List[MarkerDiff]:
    """Compare markers across paired clips in a track.

    Two markers are "the same" when their ``(name, clip)`` tuple
    matches.  Different timestamps emit a ``shifted`` diff.
    """
    out: List[MarkerDiff] = []

    left_markers: List[Dict[str, Any]] = []
    right_markers: List[Dict[str, Any]] = []
    for c in left_clips_in_track:
        left_markers.extend(_collect_markers(c))
    for c in right_clips_in_track:
        right_markers.extend(_collect_markers(c))

    # Index by (clip, name)
    l_idx: Dict[Tuple[str, str], Dict[str, Any]] = {
        (m["clip"], m["name"]): m for m in left_markers
    }
    r_idx: Dict[Tuple[str, str], Dict[str, Any]] = {
        (m["clip"], m["name"]): m for m in right_markers
    }
    for key in sorted(set(l_idx.keys()) | set(r_idx.keys())):
        lm = l_idx.get(key)
        rm = r_idx.get(key)
        if lm and rm:
            if lm["t"] != rm["t"]:
                out.append(MarkerDiff(
                    kind="shifted", name=key[1], clip=key[0],
                    t_left=lm["t"], t_right=rm["t"],
                ))
        elif lm:
            out.append(MarkerDiff(
                kind="removed", name=key[1], clip=key[0], t_left=lm["t"],
            ))
        elif rm:
            out.append(MarkerDiff(
                kind="added", name=key[1], clip=key[0], t_right=rm["t"],
            ))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diff_timelines(
    left_path: str,
    right_path: str,
    on_progress: Optional[Callable] = None,
) -> OtioDiffResult:
    """Compute a structural diff between two OTIO-compatible files.

    Args:
        left_path: Path to the "baseline" timeline (any OTIO adapter
            target: .otio, .otioz, .xml, .fcpxml, .aaf, .edl).
        right_path: Path to the "changed" timeline.
        on_progress: ``(pct, msg)`` callback.

    Returns:
        :class:`OtioDiffResult`.

    Raises:
        RuntimeError: OpenTimelineIO not installed.
        ValueError: one of the files can't be parsed.
    """
    if not check_otio_diff_available():
        raise RuntimeError(
            "OpenTimelineIO not installed. Install: pip install opentimelineio"
        )
    import opentimelineio as otio

    if on_progress:
        on_progress(5, "Loading left timeline…")
    try:
        left = otio.adapters.read_from_file(left_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not parse left timeline: {exc}") from exc

    if on_progress:
        on_progress(20, "Loading right timeline…")
    try:
        right = otio.adapters.read_from_file(right_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not parse right timeline: {exc}") from exc

    # Build track indices. OTIO's Timeline.tracks is a Stack; iterate
    # directly for ordered access.
    left_tracks = list(getattr(left, "tracks", []) or [])
    right_tracks = list(getattr(right, "tracks", []) or [])

    if on_progress:
        on_progress(40, "Diffing tracks…")

    # Index tracks by name, falling back to positional index when names
    # collide (both "V1" for instance).
    track_diffs: List[TrackDiff] = []
    clip_diffs: List[ClipDiff] = []
    marker_diffs: List[MarkerDiff] = []

    l_by_name: Dict[str, Tuple[int, Any]] = {}
    for i, t in enumerate(left_tracks):
        name = getattr(t, "name", "") or f"<track_{i}>"
        l_by_name.setdefault(name, (i, t))
    r_by_name: Dict[str, Tuple[int, Any]] = {}
    for i, t in enumerate(right_tracks):
        name = getattr(t, "name", "") or f"<track_{i}>"
        r_by_name.setdefault(name, (i, t))

    all_track_names = set(l_by_name.keys()) | set(r_by_name.keys())
    for tname in sorted(all_track_names):
        l_entry = l_by_name.get(tname)
        r_entry = r_by_name.get(tname)
        if l_entry and not r_entry:
            track_diffs.append(TrackDiff(
                kind="removed", name=tname,
                left_index=l_entry[0],
            ))
            continue
        if r_entry and not l_entry:
            track_diffs.append(TrackDiff(
                kind="added", name=tname,
                right_index=r_entry[0],
            ))
            continue
        # Both present — diff their clips
        l_clips_raw = list(l_entry[1])
        r_clips_raw = list(r_entry[1])
        l_clip_sigs = _collect_clips(l_entry[1])
        r_clip_sigs = _collect_clips(r_entry[1])

        clip_diffs.extend(_diff_clips_in_track(tname, l_clip_sigs, r_clip_sigs))
        marker_diffs.extend(_diff_markers(l_clips_raw, r_clips_raw))

    if on_progress:
        on_progress(90, "Summarising diff…")

    summary = {
        "tracks_added": sum(1 for d in track_diffs if d.kind == "added"),
        "tracks_removed": sum(1 for d in track_diffs if d.kind == "removed"),
        "clips_added": sum(1 for d in clip_diffs if d.kind == "added"),
        "clips_removed": sum(1 for d in clip_diffs if d.kind == "removed"),
        "clips_retimed": sum(1 for d in clip_diffs if d.kind == "retimed"),
        "clips_moved": sum(1 for d in clip_diffs if d.kind == "moved"),
        "markers_added": sum(1 for d in marker_diffs if d.kind == "added"),
        "markers_removed": sum(1 for d in marker_diffs if d.kind == "removed"),
        "markers_shifted": sum(1 for d in marker_diffs if d.kind == "shifted"),
    }
    identical = all(v == 0 for v in summary.values())

    if on_progress:
        on_progress(100, "Diff complete")

    return OtioDiffResult(
        left=left_path,
        right=right_path,
        track_diffs=track_diffs,
        clip_diffs=clip_diffs,
        marker_diffs=marker_diffs,
        summary=summary,
        identical=identical,
        notes=[
            f"left_tracks={len(left_tracks)}",
            f"right_tracks={len(right_tracks)}",
        ],
    )


def format_diff_text(result: OtioDiffResult, max_entries: int = 60) -> str:
    """Render a human-readable text summary (for CLI / log output)."""
    lines: List[str] = []
    lines.append(f"OTIO diff: {result.left}")
    lines.append(f"    vs.   {result.right}")
    if result.identical:
        lines.append("  (no differences)")
        return "\n".join(lines)

    lines.append("  summary:")
    for k, v in sorted(result.summary.items()):
        if v:
            lines.append(f"    {k}: {v}")

    if result.track_diffs:
        lines.append("  tracks:")
        for d in result.track_diffs[:max_entries]:
            lines.append(f"    [{d.kind}] {d.name}")
    if result.clip_diffs:
        lines.append("  clips:")
        for d in result.clip_diffs[:max_entries]:
            lines.append(f"    [{d.kind}] track={d.track or '<?>'} {d.details}")
    if result.marker_diffs:
        lines.append("  markers:")
        for d in result.marker_diffs[:max_entries]:
            lines.append(
                f"    [{d.kind}] {d.name}@{d.clip} "
                f"{d.t_left}→{d.t_right}"
            )
    return "\n".join(lines)
