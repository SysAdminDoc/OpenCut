"""Portable review bundle export (F105).

A review bundle is a single ``.zip`` archive that captures everything a
reviewer needs to evaluate a job without round-tripping back to the
panel: the rendered media (or proxy), a captions track, a marker list,
a one-page HTML summary, and a structured ``manifest.json``. The bundle
is *local-first* — no cloud accounts, no upload URLs, no auth tokens —
which is the OpenCut alternative to Frame.io / Wipster style cloud
review surfaces.

Key design choices:

* **Zip, not tarball.** Most macOS / Windows reviewers double-click to
  open. Zip is the lingua franca.
* **Deterministic ordering.** Files inside the zip are added in
  alphabetical order so the same input always produces the same hash.
* **No PII in the manifest.** Source paths are reduced to their
  basename. The full project path is intentionally absent.
* **Optional media.** The caller picks whether to embed the rendered
  file (``include_media=True``) or skip it for size — the manifest
  always records the SHA-256 of the original so the reviewer can
  cross-check externally.

The module returns a :class:`ReviewBundleResult` so route handlers can
emit a structured response (path, sha256, contained files) without
parsing the zip back out.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from xml.sax.saxutils import escape

from opencut.core.marker_metadata import denormalise_color, normalise_color
from opencut.openapi_registry import openapi_response_schema

logger = logging.getLogger("opencut")

BUNDLE_VERSION = 4
DEFAULT_OTIO_RATE = 30.0
OTIO_MARKERS_BASENAME = "markers.otio"
ANNOTATIONS_INDEX_BASENAME = "annotations/index.json"
THREADS_BASENAME = "review_threads.json"
PREMIERE_MARKERS_BASENAME = "premiere_markers.csv"
EDL_MARKERS_BASENAME = "review_markers.edl"
DEFAULT_ANNOTATION_WIDTH = 1920
DEFAULT_ANNOTATION_HEIGHT = 1080
DRAWING_ANNOTATION_TYPES = frozenset({"drawing_rect", "drawing_circle", "drawing_arrow"})
CLOSED_REVIEW_STATUSES = frozenset({"resolved", "wontfix"})


@dataclass
class BundleEntry:
    """A single file inside the bundle."""

    arcname: str
    sha256: str
    bytes: int
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
@openapi_response_schema(
    "/review/bundle",
    extra_properties={"version": {"type": "integer"}},
)
class ReviewBundleResult:
    output_path: str
    bundle_sha256: str
    total_bytes: int
    entries: List[BundleEntry] = field(default_factory=list)
    manifest_path: str = "manifest.json"
    summary_path: str = "summary.html"
    otio_markers_path: str = ""
    annotations_path: str = ""
    annotation_count: int = 0
    threads_path: str = ""
    thread_count: int = 0
    open_thread_count: int = 0
    completion_status: str = ""
    premiere_markers_path: str = ""
    edl_markers_path: str = ""
    marker_export_count: int = 0
    generated_at: float = field(default_factory=time.time)
    job_label: str = ""

    def as_dict(self) -> dict:
        return {
            "version": BUNDLE_VERSION,
            "output_path": self.output_path,
            "bundle_sha256": self.bundle_sha256,
            "total_bytes": self.total_bytes,
            "manifest_path": self.manifest_path,
            "summary_path": self.summary_path,
            "otio_markers_path": self.otio_markers_path,
            "annotations_path": self.annotations_path,
            "annotation_count": self.annotation_count,
            "threads_path": self.threads_path,
            "thread_count": self.thread_count,
            "open_thread_count": self.open_thread_count,
            "completion_status": self.completion_status,
            "premiere_markers_path": self.premiere_markers_path,
            "edl_markers_path": self.edl_markers_path,
            "marker_export_count": self.marker_export_count,
            "generated_at": self.generated_at,
            "job_label": self.job_label,
            "entries": [e.as_dict() for e in self.entries],
        }


def _sha256_of(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _sanitise_arcname(name: str) -> str:
    """Force POSIX separators and strip parent traversal."""
    safe = name.replace("\\", "/").lstrip("/")
    parts = [p for p in safe.split("/") if p not in {"", ".", ".."}]
    return "/".join(parts) or "asset.bin"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if result != result or result in (float("inf"), float("-inf")):
        return default
    return result


def _first_present(payload: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _marker_rows(markers_payload: Any) -> List[Dict[str, Any]]:
    """Return marker/comment rows from the common payload shapes."""
    if markers_payload is None:
        return []
    if isinstance(markers_payload, list):
        return [m for m in markers_payload if isinstance(m, dict)]
    if not isinstance(markers_payload, dict):
        return []
    for key in ("markers", "comments", "items"):
        value = markers_payload.get(key)
        if isinstance(value, list):
            return [m for m in value if isinstance(m, dict)]
    if any(key in markers_payload for key in ("time", "timestamp", "timestamp_sec", "start_seconds", "text", "name")):
        return [markers_payload]
    return []


def _marker_start_seconds(marker: Dict[str, Any], framerate: float) -> float:
    raw = _first_present(marker, ("start_seconds", "timestamp_sec", "time", "timestamp", "start", "in_seconds", "in"))
    if raw is not None:
        return max(0.0, _safe_float(raw))
    frame = _first_present(marker, ("frame_number", "frame"))
    if frame is not None and framerate > 0:
        return max(0.0, _safe_float(frame) / framerate)
    return 0.0


def _marker_duration_seconds(marker: Dict[str, Any], start_seconds: float) -> float:
    raw = _first_present(marker, ("duration_seconds", "duration", "length_seconds", "length"))
    if raw is not None:
        return max(0.0, _safe_float(raw))
    end_raw = _first_present(marker, ("end_seconds", "end", "out_seconds", "out"))
    if end_raw is not None:
        return max(0.0, _safe_float(end_raw) - start_seconds)
    return 0.0


def _marker_name(marker: Dict[str, Any], idx: int) -> str:
    raw = _first_present(marker, ("name", "title", "label"))
    if raw is None:
        raw = _first_present(marker, ("text", "comment", "notes"))
    name = str(raw or "").strip()
    if not name:
        return f"Marker {idx + 1}"
    return name[:96]


def _marker_comment(marker: Dict[str, Any]) -> str:
    raw = _first_present(marker, ("comment", "notes", "text", "description"))
    return str(raw or "").strip()


def _marker_metadata(marker: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in (
        "id",
        "parent_id",
        "status",
        "author",
        "annotation_type",
        "tags",
        "chapter",
        "frame_number",
        "source",
        "type",
    ):
        value = marker.get(key)
        if value not in (None, ""):
            metadata[key] = value
    annotation_data = marker.get("annotation_data")
    if isinstance(annotation_data, dict) and annotation_data:
        metadata["annotation_data"] = annotation_data
    return metadata


def _marker_id(marker: Dict[str, Any], idx: int) -> str:
    raw = _first_present(marker, ("id", "comment_id", "marker_id"))
    value = str(raw or "").strip()
    return value or f"comment-{idx + 1}"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "done", "complete", "completed", "resolved"}
    return False


def _review_status(marker: Dict[str, Any]) -> str:
    raw = str(_first_present(marker, ("status", "state", "review_status")) or "").strip().lower()
    status = raw.replace("-", "_").replace(" ", "_")
    if status in {"resolved", "resolve", "done", "closed", "complete", "completed", "approved", "accepted"}:
        return "resolved"
    if status in {"wontfix", "wont_fix", "won't_fix", "rejected", "declined", "not_applicable"}:
        return "wontfix"
    if _truthy(_first_present(marker, ("completed", "complete", "is_complete", "is_resolved", "resolved"))):
        return "resolved"
    return "open"


def _comment_tags(marker: Dict[str, Any]) -> List[str]:
    tags = marker.get("tags")
    if isinstance(tags, list):
        return [str(tag).strip() for tag in tags if str(tag).strip()]
    if isinstance(tags, str):
        return [tag.strip() for tag in re.split(r"[;,]", tags) if tag.strip()]
    return []


def normalise_review_markers(markers_payload: Any, *, framerate: float = DEFAULT_OTIO_RATE) -> List[Dict[str, Any]]:
    """Normalize review marker/comment payloads into the OpenCut canonical shape."""
    rate = max(1.0, _safe_float(framerate, DEFAULT_OTIO_RATE))
    normalised: List[Dict[str, Any]] = []
    for idx, marker in enumerate(_marker_rows(markers_payload)):
        start_seconds = _marker_start_seconds(marker, rate)
        duration_seconds = _marker_duration_seconds(marker, start_seconds)
        color = normalise_color(str(marker.get("color") or "green"), host=str(marker.get("source") or "premiere"))
        normalised.append(
            {
                "name": _marker_name(marker, idx),
                "start_seconds": start_seconds,
                "duration_seconds": duration_seconds,
                "color": color,
                "comment": _marker_comment(marker),
                "metadata": _marker_metadata(marker),
            }
        )
    return sorted(normalised, key=lambda item: (item["start_seconds"], item["name"]))


def _otio_rational(seconds: float, framerate: float) -> Dict[str, Any]:
    return {
        "OTIO_SCHEMA": "RationalTime.1",
        "rate": framerate,
        "value": int(round(max(0.0, seconds) * framerate)),
    }


def _otio_time_range(start_seconds: float, duration_seconds: float, framerate: float) -> Dict[str, Any]:
    return {
        "OTIO_SCHEMA": "TimeRange.1",
        "start_time": _otio_rational(start_seconds, framerate),
        "duration": _otio_rational(duration_seconds, framerate),
    }


def _otio_marker(marker: Dict[str, Any], framerate: float) -> Dict[str, Any]:
    opencut_metadata = {
        "schema": "review-marker.v1",
        "start_seconds": marker["start_seconds"],
        "duration_seconds": marker["duration_seconds"],
        "comment": marker["comment"],
        **marker["metadata"],
    }
    return {
        "OTIO_SCHEMA": "Marker.2",
        "name": marker["name"],
        "marked_range": _otio_time_range(
            marker["start_seconds"],
            marker["duration_seconds"],
            framerate,
        ),
        "color": denormalise_color(marker["color"], "otio"),
        "metadata": {"opencut": opencut_metadata},
    }


def build_review_markers_otio(
    markers_payload: Any,
    *,
    job_label: str = "",
    media_basename: str = "",
    framerate: float = DEFAULT_OTIO_RATE,
    duration_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Build a minimal OTIO timeline carrying review comments as Marker objects."""
    rate = max(1.0, _safe_float(framerate, DEFAULT_OTIO_RATE))
    markers = normalise_review_markers(markers_payload, framerate=rate)
    marker_end = max((m["start_seconds"] + m["duration_seconds"] for m in markers), default=0.0)
    timeline_duration = max(_safe_float(duration_seconds), marker_end + (1.0 / rate), 1.0 / rate)
    otio_markers = [_otio_marker(marker, rate) for marker in markers]
    return {
        "OTIO_SCHEMA": "Timeline.1",
        "name": job_label or "OpenCut Review Markers",
        "metadata": {
            "opencut": {
                "schema": "review-bundle-markers.v1",
                "bundle_version": BUNDLE_VERSION,
                "marker_count": len(markers),
                "media_basename": media_basename,
            }
        },
        "global_start_time": _otio_rational(0.0, rate),
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "name": "tracks",
            "children": [
                {
                    "OTIO_SCHEMA": "Track.1",
                    "name": "Review Markers",
                    "kind": "Video",
                    "children": [
                        {
                            "OTIO_SCHEMA": "Gap.1",
                            "name": "Review marker range",
                            "source_range": _otio_time_range(0.0, timeline_duration, rate),
                            "markers": otio_markers,
                        }
                    ],
                }
            ],
        },
    }


def _seconds_to_timecode(seconds: float, framerate: float) -> str:
    fps = max(1, int(round(_safe_float(framerate, DEFAULT_OTIO_RATE))))
    total_frames = max(0, int(round(_safe_float(seconds) * fps)))
    frames = total_frames % fps
    total_seconds = total_frames // fps
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"


def _single_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _review_marker_description(marker: Dict[str, Any], *, multiline: bool = True) -> str:
    metadata = marker.get("metadata") if isinstance(marker.get("metadata"), dict) else {}
    parts: List[str] = []
    comment = str(marker.get("comment") or "").strip()
    if comment:
        parts.append(comment)
    for key, label in (
        ("status", "Status"),
        ("author", "Author"),
        ("id", "Comment ID"),
        ("parent_id", "Parent ID"),
        ("annotation_type", "Annotation"),
    ):
        value = metadata.get(key)
        if value not in (None, ""):
            parts.append(f"{label}: {value}")
    if not parts:
        return ""
    separator = "\n" if multiline else " | "
    return separator.join(_single_line(part) for part in parts if _single_line(part))


def build_premiere_marker_csv(markers_payload: Any, *, framerate: float = DEFAULT_OTIO_RATE) -> str:
    """Export review comments as Premiere-importable marker CSV text."""
    rate = max(1.0, _safe_float(framerate, DEFAULT_OTIO_RATE))
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["Marker Name", "Description", "In", "Out", "Duration", "Marker Color"])
    for marker in normalise_review_markers(markers_payload, framerate=rate):
        start = marker["start_seconds"]
        duration = marker["duration_seconds"]
        writer.writerow(
            [
                marker["name"],
                _review_marker_description(marker, multiline=True),
                _seconds_to_timecode(start, rate),
                _seconds_to_timecode(start + duration, rate),
                _seconds_to_timecode(duration, rate),
                denormalise_color(marker["color"], "premiere"),
            ]
        )
    return output.getvalue()


def build_review_markers_edl(
    markers_payload: Any,
    *,
    job_label: str = "",
    framerate: float = DEFAULT_OTIO_RATE,
) -> str:
    """Export review comments as a CMX3600 marker-only EDL subset."""
    rate = max(1.0, _safe_float(framerate, DEFAULT_OTIO_RATE))
    title = _single_line(job_label) or "OpenCut Review Markers"
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    for marker in normalise_review_markers(markers_payload, framerate=rate):
        description = _review_marker_description(marker, multiline=False)
        if description:
            lines.append(f"* LOC: {description}")
        color = denormalise_color(marker["color"], "premiere").upper()
        lines.append(f"M: AX     {color:<8} {_seconds_to_timecode(marker['start_seconds'], rate)} {marker['name']}")
    return "\n".join(lines).rstrip() + "\n"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, float(default))))
    except (TypeError, ValueError, OverflowError):
        return default


def _slug(value: str, default: str = "annotation") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-._")
    return slug[:48] or default


def _svg_escape(value: Any) -> str:
    return escape(str(value or ""), {"\"": "&quot;"})


def _annotation_dimension(value: Any, default: int) -> int:
    return max(1, min(16384, _safe_int(value, default)))


def _annotation_color(value: Any, fallback: str = "green") -> str:
    raw = str(value or "").strip()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", raw):
        return raw.lower()
    palette = {
        "black": "#111827",
        "blue": "#2563eb",
        "cyan": "#0891b2",
        "green": "#16a34a",
        "magenta": "#c026d3",
        "orange": "#ea580c",
        "pink": "#db2777",
        "purple": "#7c3aed",
        "red": "#dc2626",
        "rose": "#e11d48",
        "white": "#f8fafc",
        "yellow": "#ca8a04",
    }
    canonical = normalise_color(raw or fallback)
    return palette.get(canonical, palette["green"])


def _annotation_stroke_width(data: Dict[str, Any]) -> float:
    width = _safe_float(_first_present(data, ("stroke_width", "strokeWidth", "width_px")), 4.0)
    return max(1.0, min(64.0, width))


def _annotation_value(data: Dict[str, Any], keys: Sequence[str], default: float = 0.0) -> float:
    raw = _first_present(data, keys)
    return _safe_float(raw, default)


def _annotation_coord(data: Dict[str, Any], keys: Sequence[str], limit: int, default: float = 0.0) -> float:
    value = _annotation_value(data, keys, default)
    normalized = data.get("normalized") is True or str(data.get("coordinate_space", "")).lower() == "normalized"
    if normalized:
        value *= limit
    return max(0.0, min(float(limit), value))


def _annotation_label(marker: Dict[str, Any]) -> str:
    return str(_first_present(marker, ("text", "comment", "notes", "name", "title", "label")) or "").strip()


def _svg_doc(*, width: int, height: int, annotation_id: str, title: str, body: str) -> bytes:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title-{annotation_id} desc-{annotation_id}">\n'
        f'  <title id="title-{annotation_id}">{_svg_escape(title)}</title>\n'
        f'  <desc id="desc-{annotation_id}">OpenCut review drawing annotation</desc>\n'
        f"{body}\n"
        "</svg>\n"
    )
    return svg.encode("utf-8")


def _rect_svg(marker: Dict[str, Any], width: int, height: int, annotation_id: str) -> bytes:
    data = marker.get("annotation_data") if isinstance(marker.get("annotation_data"), dict) else {}
    x = _annotation_coord(data, ("x", "left"), width)
    y = _annotation_coord(data, ("y", "top"), height)
    w = _annotation_value(data, ("w", "width"), 0.0)
    h = _annotation_value(data, ("h", "height"), 0.0)
    if not w and "right" in data:
        w = _annotation_coord(data, ("right",), width) - x
    if not h and "bottom" in data:
        h = _annotation_coord(data, ("bottom",), height) - y
    if data.get("normalized") is True or str(data.get("coordinate_space", "")).lower() == "normalized":
        w *= width
        h *= height
    w = max(1.0, min(float(width) - x, w))
    h = max(1.0, min(float(height) - y, h))
    stroke = _annotation_color(data.get("stroke") or data.get("color") or marker.get("color"))
    stroke_width = _annotation_stroke_width(data)
    body = (
        f'  <rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'fill="none" stroke="{stroke}" stroke-width="{stroke_width:.2f}" />'
    )
    return _svg_doc(width=width, height=height, annotation_id=annotation_id, title=_annotation_label(marker), body=body)


def _circle_svg(marker: Dict[str, Any], width: int, height: int, annotation_id: str) -> bytes:
    data = marker.get("annotation_data") if isinstance(marker.get("annotation_data"), dict) else {}
    cx = _annotation_coord(data, ("cx", "x"), width, width / 2)
    cy = _annotation_coord(data, ("cy", "y"), height, height / 2)
    r = _annotation_value(data, ("r", "radius"), 48.0)
    if data.get("normalized") is True or str(data.get("coordinate_space", "")).lower() == "normalized":
        r *= min(width, height)
    r = max(1.0, min(r, cx, cy, width - cx, height - cy))
    stroke = _annotation_color(data.get("stroke") or data.get("color") or marker.get("color"))
    stroke_width = _annotation_stroke_width(data)
    body = (
        f'  <circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" '
        f'fill="none" stroke="{stroke}" stroke-width="{stroke_width:.2f}" />'
    )
    return _svg_doc(width=width, height=height, annotation_id=annotation_id, title=_annotation_label(marker), body=body)


def _arrow_svg(marker: Dict[str, Any], width: int, height: int, annotation_id: str) -> bytes:
    data = marker.get("annotation_data") if isinstance(marker.get("annotation_data"), dict) else {}
    x1 = _annotation_coord(data, ("x1", "start_x", "from_x"), width)
    y1 = _annotation_coord(data, ("y1", "start_y", "from_y"), height)
    x2 = _annotation_coord(data, ("x2", "end_x", "to_x"), width, width / 2)
    y2 = _annotation_coord(data, ("y2", "end_y", "to_y"), height, height / 2)
    stroke = _annotation_color(data.get("stroke") or data.get("color") or marker.get("color"))
    stroke_width = _annotation_stroke_width(data)
    body = (
        "  <defs>\n"
        f'    <marker id="arrow-{annotation_id}" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        'orient="auto" markerUnits="strokeWidth">\n'
        f'      <path d="M0,0 L0,6 L9,3 z" fill="{stroke}" />\n'
        "    </marker>\n"
        "  </defs>\n"
        f'  <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" stroke-linecap="round" '
        f'marker-end="url(#arrow-{annotation_id})" />'
    )
    return _svg_doc(width=width, height=height, annotation_id=annotation_id, title=_annotation_label(marker), body=body)


def build_review_annotation_svgs(
    markers_payload: Any,
    *,
    framerate: float = DEFAULT_OTIO_RATE,
    width: int = DEFAULT_ANNOTATION_WIDTH,
    height: int = DEFAULT_ANNOTATION_HEIGHT,
) -> tuple[Dict[str, Any], List[tuple[str, bytes]]]:
    """Build SVG overlay files for drawing annotations in a marker payload."""
    rate = max(1.0, _safe_float(framerate, DEFAULT_OTIO_RATE))
    canvas_width = _annotation_dimension(width, DEFAULT_ANNOTATION_WIDTH)
    canvas_height = _annotation_dimension(height, DEFAULT_ANNOTATION_HEIGHT)
    index: Dict[str, Any] = {
        "schema": "opencut.review-annotations.v1",
        "framerate": rate,
        "width": canvas_width,
        "height": canvas_height,
        "annotations": [],
    }
    files: List[tuple[str, bytes]] = []
    renderers = {
        "drawing_rect": _rect_svg,
        "drawing_circle": _circle_svg,
        "drawing_arrow": _arrow_svg,
    }

    for idx, marker in enumerate(_marker_rows(markers_payload)):
        annotation_type = str(marker.get("annotation_type") or marker.get("type") or "").strip()
        if annotation_type not in DRAWING_ANNOTATION_TYPES:
            continue
        start_seconds = _marker_start_seconds(marker, rate)
        frame_number = _safe_int(marker.get("frame_number"), int(round(start_seconds * rate)))
        annotation_id = _slug(str(marker.get("id") or f"{annotation_type}-{idx + 1}"), f"annotation-{idx + 1}")
        arcname = f"annotations/{frame_number:08d}_{annotation_id}.svg"
        svg = renderers[annotation_type](marker, canvas_width, canvas_height, annotation_id)
        files.append((arcname, svg))
        index["annotations"].append(
            {
                "id": str(marker.get("id") or annotation_id),
                "annotation_type": annotation_type,
                "timestamp_sec": start_seconds,
                "frame_number": frame_number,
                "duration_seconds": _marker_duration_seconds(marker, start_seconds),
                "svg": arcname,
                "text": _annotation_label(marker),
                "status": str(marker.get("status") or "open"),
            }
        )

    return index, files


def build_review_threads(markers_payload: Any, *, framerate: float = DEFAULT_OTIO_RATE) -> Dict[str, Any]:
    """Build a stable threaded-comment sidecar with review completion status."""
    rate = max(1.0, _safe_float(framerate, DEFAULT_OTIO_RATE))
    comments: List[Dict[str, Any]] = []
    used_ids: set[str] = set()

    for idx, marker in enumerate(_marker_rows(markers_payload)):
        start_seconds = _marker_start_seconds(marker, rate)
        raw_id = _marker_id(marker, idx)
        comment_id = raw_id
        if comment_id in used_ids:
            comment_id = f"{comment_id}-{idx + 1}"
        used_ids.add(comment_id)
        parent_id = str(_first_present(marker, ("parent_id", "parentId", "reply_to", "replyTo")) or "").strip()
        frame_number = _safe_int(marker.get("frame_number"), int(round(start_seconds * rate)))
        comment = {
            "id": comment_id,
            "parent_id": parent_id,
            "timestamp_sec": start_seconds,
            "frame_number": max(0, frame_number),
            "duration_seconds": _marker_duration_seconds(marker, start_seconds),
            "author": str(marker.get("author") or marker.get("owner") or "").strip(),
            "text": _annotation_label(marker) or _marker_comment(marker),
            "status": _review_status(marker),
            "annotation_type": str(marker.get("annotation_type") or marker.get("type") or "text").strip() or "text",
            "tags": _comment_tags(marker),
            "created_at": _safe_float(marker.get("created_at") or marker.get("inserted_at")),
            "updated_at": _safe_float(marker.get("updated_at")),
        }
        comments.append(comment)

    by_id = {comment["id"]: comment for comment in comments}
    for comment in comments:
        parent_id = comment.get("parent_id") or ""
        if parent_id and parent_id not in by_id:
            comment["orphaned_parent_id"] = parent_id
    children: Dict[str, List[Dict[str, Any]]] = {}
    roots: List[Dict[str, Any]] = []
    for comment in comments:
        parent_id = comment.get("parent_id") or ""
        if parent_id and parent_id in by_id and parent_id != comment["id"]:
            children.setdefault(parent_id, []).append(comment)
        else:
            roots.append(comment)

    def sort_key(comment: Dict[str, Any]) -> tuple[float, float, str]:
        return (
            _safe_float(comment.get("timestamp_sec")),
            _safe_float(comment.get("created_at")),
            str(comment.get("id") or ""),
        )

    def public_comment(comment: Dict[str, Any], visited: set[str]) -> Dict[str, Any]:
        comment_id = str(comment["id"])
        next_visited = set(visited)
        next_visited.add(comment_id)
        replies = [
            public_comment(child, next_visited)
            for child in sorted(children.get(comment_id, []), key=sort_key)
            if child["id"] not in next_visited
        ]
        payload = {
            key: comment[key]
            for key in (
                "id",
                "parent_id",
                "timestamp_sec",
                "frame_number",
                "duration_seconds",
                "author",
                "text",
                "status",
                "annotation_type",
                "tags",
                "created_at",
                "updated_at",
            )
        }
        if "orphaned_parent_id" in comment:
            payload["orphaned_parent_id"] = comment["orphaned_parent_id"]
        payload["reply_count"] = len(replies)
        payload["replies"] = replies
        return payload

    def flatten_thread(comment: Dict[str, Any], visited: set[str]) -> List[Dict[str, Any]]:
        if comment["id"] in visited:
            return []
        next_visited = set(visited)
        next_visited.add(comment["id"])
        items = [comment]
        for child in children.get(comment["id"], []):
            items.extend(flatten_thread(child, next_visited))
        return items

    threads: List[Dict[str, Any]] = []
    for root in sorted(roots, key=sort_key):
        thread_comments = flatten_thread(root, set())
        is_complete = all(comment["status"] in CLOSED_REVIEW_STATUSES for comment in thread_comments)
        public = public_comment(root, set())
        public["completion_status"] = "complete" if is_complete else "changes_requested"
        public["comment_count"] = len(thread_comments)
        threads.append(public)

    status_counts: Dict[str, int] = {}
    for comment in comments:
        status = str(comment["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    open_thread_count = sum(1 for thread in threads if thread["completion_status"] != "complete")
    completion_status = "empty" if not comments else ("complete" if open_thread_count == 0 else "changes_requested")

    return {
        "schema": "opencut.review-threads.v1",
        "bundle_version": BUNDLE_VERSION,
        "framerate": rate,
        "completion_status": completion_status,
        "thread_count": len(threads),
        "open_thread_count": open_thread_count,
        "completed_thread_count": len(threads) - open_thread_count,
        "comment_count": len(comments),
        "reply_count": max(0, len(comments) - len(threads)),
        "status_counts": {
            "open": status_counts.get("open", 0),
            "resolved": status_counts.get("resolved", 0),
            "wontfix": status_counts.get("wontfix", 0),
        },
        "threads": threads,
    }


def _render_summary_html(
    *,
    job_label: str,
    media_basename: str,
    captions_basename: str,
    markers_basename: str,
    otio_markers_basename: str,
    threads_basename: str,
    thread_count: int,
    open_thread_count: int,
    completion_status: str,
    premiere_markers_basename: str,
    edl_markers_basename: str,
    marker_export_count: int,
    annotations_basename: str,
    annotation_count: int,
    entries: Sequence[BundleEntry],
    notes: str,
) -> str:
    """Render the one-page HTML review summary."""
    rows = []
    for entry in entries:
        rows.append(
            "      <tr>"
            f"<td><code>{entry.arcname}</code></td>"
            f"<td>{entry.bytes:,}</td>"
            f"<td><code>{entry.sha256[:12]}…</code></td>"
            f"<td>{entry.note}</td>"
            "</tr>"
        )
    rows_html = "\n".join(rows) if rows else "      <tr><td colspan=\"4\">No entries</td></tr>"
    notes_block = notes.strip() or "(none)"

    return (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        '  <meta charset="utf-8" />\n'
        f"  <title>OpenCut review — {job_label or media_basename}</title>\n"
        "  <style>\n"
        "    body { font-family: -apple-system, system-ui, sans-serif; max-width: 880px; margin: 2rem auto; padding: 0 1rem; color: #1f2933; }\n"
        "    h1 { font-size: 1.4rem; margin-bottom: 0.2rem; }\n"
        "    h2 { margin-top: 2rem; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.06em; color: #52606d; }\n"
        "    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }\n"
        "    th, td { padding: 0.45rem 0.6rem; border-bottom: 1px solid #e4e7eb; text-align: left; }\n"
        "    code { background: #f5f7fa; padding: 0 0.2rem; border-radius: 3px; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>OpenCut review bundle</h1>\n"
        f"  <p><strong>Job:</strong> {job_label or '(unspecified)'}<br>"
        f"<strong>Media:</strong> <code>{media_basename or '(none)'}</code><br>"
        f"<strong>Captions:</strong> <code>{captions_basename or '(none)'}</code><br>"
        f"<strong>Markers:</strong> <code>{markers_basename or '(none)'}</code><br>"
        f"<strong>OTIO markers:</strong> <code>{otio_markers_basename or '(none)'}</code><br>"
        f"<strong>Threads:</strong> <code>{threads_basename or '(none)'}</code> "
        f"({thread_count} threads, {open_thread_count} open, {completion_status or 'n/a'})<br>"
        f"<strong>Premiere markers:</strong> <code>{premiere_markers_basename or '(none)'}</code><br>"
        f"<strong>EDL markers:</strong> <code>{edl_markers_basename or '(none)'}</code> "
        f"({marker_export_count} markers)<br>"
        f"<strong>Annotations:</strong> <code>{annotations_basename or '(none)'}</code> ({annotation_count})</p>\n"
        "  <h2>Notes</h2>\n"
        f"  <pre>{notes_block}</pre>\n"
        "  <h2>Contents</h2>\n"
        "  <table>\n"
        "    <thead><tr><th>arcname</th><th>bytes</th><th>sha-256 (head)</th><th>note</th></tr></thead>\n"
        "    <tbody>\n"
        f"{rows_html}\n"
        "    </tbody>\n"
        "  </table>\n"
        "</body>\n"
        "</html>\n"
    )


def build_review_bundle(
    *,
    output_path: str | os.PathLike,
    job_label: str = "",
    media_path: Optional[str] = None,
    captions_path: Optional[str] = None,
    markers_payload: Optional[dict] = None,
    notes: str = "",
    extra_files: Optional[List[str]] = None,
    include_media: bool = True,
    framerate: float = DEFAULT_OTIO_RATE,
    duration_seconds: float = 0.0,
    annotation_width: int = DEFAULT_ANNOTATION_WIDTH,
    annotation_height: int = DEFAULT_ANNOTATION_HEIGHT,
) -> ReviewBundleResult:
    """Create the review bundle on disk.

    ``markers_payload`` is written as ``markers.json`` inside the zip
    so the bundle is self-describing — reviewers don't need to know
    which CSV/EDL variant the team uses.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    media_basename = ""
    captions_basename = ""
    markers_basename = ""
    otio_markers_basename = ""
    annotations_basename = ""
    annotation_count = 0
    threads_basename = ""
    thread_count = 0
    open_thread_count = 0
    completion_status = ""
    premiere_markers_basename = ""
    edl_markers_basename = ""
    marker_export_count = 0
    pre_entries: List[BundleEntry] = []
    queued_files: List[tuple] = []  # (arcname, source_path, note)

    if media_path:
        media = Path(media_path)
        if not media.exists():
            raise FileNotFoundError(media_path)
        media_basename = media.name
        if include_media:
            queued_files.append((f"media/{media.name}", media, "rendered media"))
        else:
            # Even when omitted we still want the manifest to record the
            # source hash so reviewers can verify against the original.
            pre_entries.append(
                BundleEntry(
                    arcname=f"media/{media.name}",
                    sha256=_sha256_of(media),
                    bytes=media.stat().st_size,
                    note="media omitted (include_media=False); hash refers to source file",
                )
            )

    if captions_path:
        captions = Path(captions_path)
        if not captions.exists():
            raise FileNotFoundError(captions_path)
        captions_basename = captions.name
        queued_files.append((f"captions/{captions.name}", captions, "subtitle track"))

    if extra_files:
        for raw in extra_files:
            p = Path(raw)
            if not p.exists():
                raise FileNotFoundError(raw)
            queued_files.append((f"extras/{p.name}", p, "additional file"))

    if markers_payload is not None:
        markers_basename = "markers.json"
        otio_markers_basename = OTIO_MARKERS_BASENAME
        threads_basename = THREADS_BASENAME
        premiere_markers_basename = PREMIERE_MARKERS_BASENAME
        edl_markers_basename = EDL_MARKERS_BASENAME
        annotations_basename = ANNOTATIONS_INDEX_BASENAME

    # Deterministic ordering — alphabetical by arcname.
    queued_files.sort(key=lambda item: item[0])

    bundle = ReviewBundleResult(
        output_path=str(out),
        bundle_sha256="",
        total_bytes=0,
        job_label=job_label,
    )

    # Build a stable manifest before writing so the summary can reference
    # the SHA-256 of every file. We first read each file once to compute
    # the hash, then write everything into the zip in a second pass.
    entries: List[BundleEntry] = list(pre_entries)
    for arcname, src, note in queued_files:
        entries.append(
            BundleEntry(
                arcname=arcname,
                sha256=_sha256_of(src),
                bytes=src.stat().st_size,
                note=note,
            )
        )

    if markers_payload is not None:
        markers_text = json.dumps(markers_payload, indent=2, sort_keys=True).encode("utf-8")
        otio_payload = build_review_markers_otio(
            markers_payload,
            job_label=job_label,
            media_basename=media_basename,
            framerate=framerate,
            duration_seconds=duration_seconds,
        )
        otio_markers_text = json.dumps(otio_payload, indent=2, sort_keys=True).encode("utf-8")
        threads_payload = build_review_threads(markers_payload, framerate=framerate)
        thread_count = int(threads_payload["thread_count"])
        open_thread_count = int(threads_payload["open_thread_count"])
        completion_status = str(threads_payload["completion_status"])
        threads_text = json.dumps(threads_payload, indent=2, sort_keys=True).encode("utf-8")
        marker_export_count = len(normalise_review_markers(markers_payload, framerate=framerate))
        premiere_markers_text = build_premiere_marker_csv(markers_payload, framerate=framerate).encode("utf-8")
        edl_markers_text = build_review_markers_edl(
            markers_payload,
            job_label=job_label,
            framerate=framerate,
        ).encode("utf-8")
        annotations_index, annotation_files = build_review_annotation_svgs(
            markers_payload,
            framerate=framerate,
            width=annotation_width,
            height=annotation_height,
        )
        annotation_count = len(annotation_files)
        if not annotation_count:
            annotations_basename = ""
        annotations_index_text = json.dumps(annotations_index, indent=2, sort_keys=True).encode("utf-8")
        entries.append(
            BundleEntry(
                arcname="markers.json",
                sha256=hashlib.sha256(markers_text).hexdigest(),
                bytes=len(markers_text),
                note="marker list",
            )
        )
        entries.append(
            BundleEntry(
                arcname=OTIO_MARKERS_BASENAME,
                sha256=hashlib.sha256(otio_markers_text).hexdigest(),
                bytes=len(otio_markers_text),
                note="OpenTimelineIO marker timeline",
            )
        )
        entries.append(
            BundleEntry(
                arcname=THREADS_BASENAME,
                sha256=hashlib.sha256(threads_text).hexdigest(),
                bytes=len(threads_text),
                note="threaded comments and review completion status",
            )
        )
        entries.append(
            BundleEntry(
                arcname=PREMIERE_MARKERS_BASENAME,
                sha256=hashlib.sha256(premiere_markers_text).hexdigest(),
                bytes=len(premiere_markers_text),
                note="Premiere marker CSV export",
            )
        )
        entries.append(
            BundleEntry(
                arcname=EDL_MARKERS_BASENAME,
                sha256=hashlib.sha256(edl_markers_text).hexdigest(),
                bytes=len(edl_markers_text),
                note="CMX3600 marker EDL export",
            )
        )
        if annotation_count:
            entries.append(
                BundleEntry(
                    arcname=ANNOTATIONS_INDEX_BASENAME,
                    sha256=hashlib.sha256(annotations_index_text).hexdigest(),
                    bytes=len(annotations_index_text),
                    note="SVG drawing annotation index",
                )
            )
            for arcname, svg_text in annotation_files:
                entries.append(
                    BundleEntry(
                        arcname=arcname,
                        sha256=hashlib.sha256(svg_text).hexdigest(),
                        bytes=len(svg_text),
                        note="SVG drawing annotation",
                    )
                )

    summary_html = _render_summary_html(
        job_label=job_label,
        media_basename=media_basename,
        captions_basename=captions_basename,
        markers_basename=markers_basename,
        otio_markers_basename=otio_markers_basename,
        threads_basename=threads_basename,
        thread_count=thread_count,
        open_thread_count=open_thread_count,
        completion_status=completion_status,
        premiere_markers_basename=premiere_markers_basename,
        edl_markers_basename=edl_markers_basename,
        marker_export_count=marker_export_count,
        annotations_basename=annotations_basename,
        annotation_count=annotation_count,
        entries=entries,
        notes=notes,
    ).encode("utf-8")
    entries.append(
        BundleEntry(
            arcname="summary.html",
            sha256=hashlib.sha256(summary_html).hexdigest(),
            bytes=len(summary_html),
            note="one-page review summary",
        )
    )

    manifest_payload = {
        "version": BUNDLE_VERSION,
        "generated_at": time.time(),
        "job_label": job_label,
        "media_basename": media_basename,
        "captions_basename": captions_basename,
        "markers_basename": markers_basename,
        "otio_markers_basename": otio_markers_basename,
        "threads_basename": threads_basename,
        "thread_count": thread_count,
        "open_thread_count": open_thread_count,
        "completion_status": completion_status,
        "premiere_markers_basename": premiere_markers_basename,
        "edl_markers_basename": edl_markers_basename,
        "marker_export_count": marker_export_count,
        "annotations_basename": annotations_basename,
        "annotation_count": annotation_count,
        "annotation_width": _annotation_dimension(annotation_width, DEFAULT_ANNOTATION_WIDTH),
        "annotation_height": _annotation_dimension(annotation_height, DEFAULT_ANNOTATION_HEIGHT),
        "framerate": max(1.0, _safe_float(framerate, DEFAULT_OTIO_RATE)),
        "duration_seconds": max(0.0, _safe_float(duration_seconds)),
        "notes": notes,
        "entries": [e.as_dict() for e in entries],
    }
    manifest_text = json.dumps(manifest_payload, indent=2, sort_keys=True).encode("utf-8")
    entries.append(
        BundleEntry(
            arcname="manifest.json",
            sha256=hashlib.sha256(manifest_text).hexdigest(),
            bytes=len(manifest_text),
            note="machine-readable bundle manifest",
        )
    )

    # Write the zip deterministically.
    queue_sorted = sorted(queued_files, key=lambda item: item[0])
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if markers_payload is not None:
            zi = zipfile.ZipInfo(filename="markers.json", date_time=(2024, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, markers_text)
            zi = zipfile.ZipInfo(filename=OTIO_MARKERS_BASENAME, date_time=(2024, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, otio_markers_text)
            zi = zipfile.ZipInfo(filename=THREADS_BASENAME, date_time=(2024, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, threads_text)
            zi = zipfile.ZipInfo(filename=PREMIERE_MARKERS_BASENAME, date_time=(2024, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, premiere_markers_text)
            zi = zipfile.ZipInfo(filename=EDL_MARKERS_BASENAME, date_time=(2024, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, edl_markers_text)
            if annotation_count:
                zi = zipfile.ZipInfo(filename=ANNOTATIONS_INDEX_BASENAME, date_time=(2024, 1, 1, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zi, annotations_index_text)
                for arcname, svg_text in annotation_files:
                    zi = zipfile.ZipInfo(filename=arcname, date_time=(2024, 1, 1, 0, 0, 0))
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    zf.writestr(zi, svg_text)
        for arcname, src, _note in queue_sorted:
            zi = zipfile.ZipInfo(filename=_sanitise_arcname(arcname), date_time=(2024, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, src.read_bytes())
        # Summary + manifest go last so they show up at the top in most
        # zip viewers (they sort by directory order).
        zi = zipfile.ZipInfo(filename="summary.html", date_time=(2024, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, summary_html)
        zi = zipfile.ZipInfo(filename="manifest.json", date_time=(2024, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, manifest_text)

    bundle.bundle_sha256 = _sha256_of(out)
    bundle.total_bytes = out.stat().st_size
    bundle.entries = entries
    bundle.otio_markers_path = otio_markers_basename
    bundle.annotations_path = annotations_basename
    bundle.annotation_count = annotation_count
    bundle.threads_path = threads_basename
    bundle.thread_count = thread_count
    bundle.open_thread_count = open_thread_count
    bundle.completion_status = completion_status
    bundle.premiere_markers_path = premiere_markers_basename
    bundle.edl_markers_path = edl_markers_basename
    bundle.marker_export_count = marker_export_count
    return bundle
