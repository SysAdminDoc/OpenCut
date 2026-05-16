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

import hashlib
import json
import logging
import os
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger("opencut")

BUNDLE_VERSION = 1


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
class ReviewBundleResult:
    output_path: str
    bundle_sha256: str
    total_bytes: int
    entries: List[BundleEntry] = field(default_factory=list)
    manifest_path: str = "manifest.json"
    summary_path: str = "summary.html"
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


def _render_summary_html(
    *,
    job_label: str,
    media_basename: str,
    captions_basename: str,
    markers_basename: str,
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
        f"<strong>Markers:</strong> <code>{markers_basename or '(none)'}</code></p>\n"
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
        entries.append(
            BundleEntry(
                arcname="markers.json",
                sha256=hashlib.sha256(markers_text).hexdigest(),
                bytes=len(markers_text),
                note="marker list",
            )
        )

    summary_html = _render_summary_html(
        job_label=job_label,
        media_basename=media_basename,
        captions_basename=captions_basename,
        markers_basename=markers_basename,
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
    return bundle
