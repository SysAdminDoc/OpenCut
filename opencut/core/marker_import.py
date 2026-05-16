"""Marker import — CSV, EDL, and Premiere CSV variants (F102).

Three input formats are supported. All produce the same normalised
``Marker`` dataclass so the existing JSX bridge can apply them in a
single pass.

``csv``
    Generic CSV with columns ``timecode,name,duration,color,comment``.
    Timecode accepts ``HH:MM:SS:FF`` (drop-frame ok), ``HH:MM:SS.fff``,
    or float seconds. Color values map onto Premiere's eight marker
    colors (case-insensitive).

``premiere_csv``
    Premiere's native marker CSV export — same shape as ``csv`` but
    with localised column headers ("Marker Name", "In", "Out", ...).
    We normalise the header row before parsing.

``edl``
    CMX 3600 EDL. We only read the ``M:`` (manual marker) and ``*``
    comment lines; cut events are intentionally ignored — markers in
    OpenCut go onto a sequence, they are not cut events.

The parsers are pure functions: no IO outside of opening the input
file. The route layer is responsible for path validation, async
queueing, and forwarding to the JSX bridge.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("opencut")


# Premiere's marker colours. The CSV format usually carries the literal
# colour name; we accept synonyms (case-insensitive) so any reasonable
# input round-trips.
_PREMIERE_COLORS = {
    "green": "green",
    "red": "red",
    "orange": "orange",
    "yellow": "yellow",
    "purple": "purple",
    "magenta": "purple",
    "cyan": "cyan",
    "blue": "cyan",
    "white": "white",
    "rose": "rose",
    "pink": "rose",
    "": "green",
}


@dataclass
class Marker:
    """Normalised marker entry."""

    name: str = ""
    start_seconds: float = 0.0
    duration_seconds: float = 0.0
    color: str = "green"
    comment: str = ""
    chapter: bool = False

    def as_dict(self) -> dict:
        payload = asdict(self)
        return payload


@dataclass
class MarkerImportResult:
    """Result of parsing a marker file."""

    format: str
    markers: List[Marker] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "format": self.format,
            "count": len(self.markers),
            "markers": [m.as_dict() for m in self.markers],
            "warnings": list(self.warnings),
            "rejected": list(self.rejected),
        }


# ---------------------------------------------------------------------------
# Timecode parsing
# ---------------------------------------------------------------------------

_TC_TC = re.compile(r"^(\d{1,2}):([0-5]?\d):([0-5]?\d)[:;](\d{1,3})$")
_TC_DOT = re.compile(r"^(\d{1,2}):([0-5]?\d):([0-5]?\d)(?:[.,](\d{1,6}))?$")
_TC_FLOAT = re.compile(r"^-?\d+(\.\d+)?$")


def parse_timecode(value: str, fps: float = 30.0) -> Optional[float]:
    """Convert a timecode/seconds string into floating-point seconds.

    Returns ``None`` when the value is unparseable so callers can drop
    the row with a structured warning. ``fps`` is only used by the
    ``HH:MM:SS:FF`` branch — float seconds and ``HH:MM:SS.fff`` ignore
    it.
    """
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None

    m = _TC_TC.match(token)
    if m:
        h, mi, s, frames = (int(p) for p in m.groups())
        if fps <= 0:
            return None
        return h * 3600 + mi * 60 + s + (frames / fps)

    m = _TC_DOT.match(token)
    if m:
        h, mi, s, frac = m.groups()
        seconds = int(h) * 3600 + int(mi) * 60 + int(s)
        if frac is not None:
            seconds += float("0." + frac)
        return float(seconds)

    if _TC_FLOAT.match(token):
        try:
            return float(token)
        except ValueError:
            return None

    return None


def _normalise_color(raw: str) -> str:
    return _PREMIERE_COLORS.get(str(raw or "").strip().lower(), "green")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_CSV_FIELD_ALIASES = {
    "timecode": ("timecode", "in", "in point", "marker in", "time", "start", "start time", "start timecode"),
    "out": ("out", "out point", "marker out", "end", "end time", "end timecode"),
    "name": ("name", "marker name", "title"),
    "duration": ("duration", "length"),
    "color": ("color", "colour", "marker color", "marker colour"),
    "comment": ("comment", "comments", "note", "description"),
    "chapter": ("chapter", "is chapter", "chapter marker"),
}


def _build_alias_map(header: Sequence[str]) -> dict:
    """Normalise CSV header tokens onto canonical field names."""
    lookup: dict = {}
    for idx, col in enumerate(header):
        token = (col or "").strip().lower()
        for canonical, synonyms in _CSV_FIELD_ALIASES.items():
            if token in synonyms:
                lookup[canonical] = idx
                break
    return lookup


def parse_csv(text: str, *, fps: float = 30.0) -> MarkerImportResult:
    """Parse generic / Premiere CSV marker exports."""
    result = MarkerImportResult(format="csv")
    if not text.strip():
        result.warnings.append("empty CSV input")
        return result

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        result.warnings.append("empty CSV input")
        return result

    header_idx = 0
    alias_map = _build_alias_map(rows[0])
    if "timecode" not in alias_map:
        # Maybe the first row is data rather than a header. Look for a
        # row with a plausible header before giving up.
        for idx, row in enumerate(rows[:5]):
            am = _build_alias_map(row)
            if "timecode" in am:
                alias_map = am
                header_idx = idx
                break
    if "timecode" not in alias_map:
        result.warnings.append(
            "no recognised timecode column (expected one of: " + ", ".join(_CSV_FIELD_ALIASES["timecode"]) + ")"
        )
        return result

    name_idx = alias_map.get("name")
    duration_idx = alias_map.get("duration")
    out_idx = alias_map.get("out")
    color_idx = alias_map.get("color")
    comment_idx = alias_map.get("comment")
    chapter_idx = alias_map.get("chapter")

    for row_no, row in enumerate(rows[header_idx + 1:], start=header_idx + 2):
        if not row or all(not (c or "").strip() for c in row):
            continue
        tc_raw = row[alias_map["timecode"]] if alias_map["timecode"] < len(row) else ""
        start_seconds = parse_timecode(tc_raw, fps=fps)
        if start_seconds is None:
            result.rejected.append(f"row {row_no}: unparseable timecode {tc_raw!r}")
            continue

        duration = 0.0
        if duration_idx is not None and duration_idx < len(row):
            duration = parse_timecode(row[duration_idx], fps=fps) or 0.0
        elif out_idx is not None and out_idx < len(row):
            out_seconds = parse_timecode(row[out_idx], fps=fps)
            if out_seconds is not None and out_seconds > start_seconds:
                duration = out_seconds - start_seconds

        marker = Marker(
            name=(row[name_idx].strip() if name_idx is not None and name_idx < len(row) else f"Marker {row_no}"),
            start_seconds=float(start_seconds),
            duration_seconds=max(0.0, float(duration)),
            color=_normalise_color(row[color_idx] if color_idx is not None and color_idx < len(row) else ""),
            comment=(row[comment_idx].strip() if comment_idx is not None and comment_idx < len(row) else ""),
            chapter=(
                (row[chapter_idx].strip().lower() in {"true", "1", "yes", "chapter"})
                if chapter_idx is not None and chapter_idx < len(row)
                else False
            ),
        )
        result.markers.append(marker)
    return result


def parse_premiere_csv(text: str, *, fps: float = 30.0) -> MarkerImportResult:
    """Premiere's native CSV is a shape of CSV — re-use the generic parser."""
    result = parse_csv(text, fps=fps)
    result.format = "premiere_csv"
    return result


# ---------------------------------------------------------------------------
# EDL (CMX 3600 — marker subset)
# ---------------------------------------------------------------------------

_EDL_MARKER_RE = re.compile(
    r"^\s*M:\s*(?P<reel>\S+)\s+(?P<color>\S+)\s+(?P<tc>\d{2}:\d{2}:\d{2}[:;]\d{2})(?:\s+(?P<name>.+))?\s*$"
)
_EDL_COMMENT_NAME_RE = re.compile(r"^\*\s*(?:LOC|FROM CLIP NAME|MARKER):?\s*(.+)$")
_EDL_FCM_RE = re.compile(r"^\s*FCM:\s*(NON-DROP FRAME|DROP FRAME)\s*$", re.IGNORECASE)


def parse_edl(text: str, *, fps: float = 30.0) -> MarkerImportResult:
    """Parse the marker subset of a CMX 3600 EDL."""
    result = MarkerImportResult(format="edl")
    if not text.strip():
        result.warnings.append("empty EDL input")
        return result

    pending_comment = ""
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if _EDL_FCM_RE.match(line):
            continue
        marker_match = _EDL_MARKER_RE.match(line)
        if marker_match:
            tc = parse_timecode(marker_match.group("tc"), fps=fps)
            if tc is None:
                result.rejected.append(f"line {line_no}: bad timecode {marker_match.group('tc')!r}")
                continue
            name = (marker_match.group("name") or "").strip() or pending_comment or "Marker"
            result.markers.append(
                Marker(
                    name=name,
                    start_seconds=float(tc),
                    duration_seconds=0.0,
                    color=_normalise_color(marker_match.group("color")),
                    comment=pending_comment,
                )
            )
            pending_comment = ""
            continue

        comment_match = _EDL_COMMENT_NAME_RE.match(line)
        if comment_match:
            pending_comment = comment_match.group(1).strip()
            continue

        # Cut events and other lines reset the pending comment so it
        # doesn't bleed into the next marker.
        if line.strip() and not line.startswith("*"):
            pending_comment = ""

    if not result.markers:
        result.warnings.append("no markers found in EDL (only manual `M:` rows are imported)")
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_format(filename: str, body: str = "") -> str:
    """Best-effort format detection from filename or first 1 KB of body."""
    name = (filename or "").lower()
    if name.endswith(".edl"):
        return "edl"
    if name.endswith(".csv"):
        return "csv"
    head = body[:1024]
    if "M:" in head and any(line.strip().endswith("V") for line in head.splitlines()[:5]):
        return "edl"
    return "csv"


def import_markers(
    *,
    text: Optional[str] = None,
    path: Optional[str] = None,
    fps: float = 30.0,
    format: Optional[str] = None,
) -> MarkerImportResult:
    """Public entry: dispatch to the right parser."""
    if (text is None) == (path is None):
        raise ValueError("supply exactly one of text / path")
    if text is None:
        with open(path, "r", encoding="utf-8-sig") as fh:
            text = fh.read()

    chosen = (format or detect_format(path or "", text)).lower()
    if chosen == "edl":
        return parse_edl(text, fps=fps)
    if chosen == "premiere_csv":
        return parse_premiere_csv(text, fps=fps)
    if chosen == "csv":
        return parse_csv(text, fps=fps)
    raise ValueError(f"unknown marker format: {chosen!r}")
