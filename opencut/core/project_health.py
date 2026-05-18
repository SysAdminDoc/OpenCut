"""Local project + media health report (F011 / v4.2).

`opencut.core.capability_profile` (F106) answers "can this machine run a
render?". This module answers the complementary question: "given the
project on disk, will it actually finish?". The two together cover the
release-trust gap that competitor issue signal flags repeatedly —
broken imports, stale media paths, missing render output dirs.

A project health report is the answer to a single, focused question:
**"If I start a render right now, what's the most likely failure?"**

Inputs:

* ``project_root`` — the directory holding the project artefacts
  (rendered media, sidecars, captions, marker exports, etc.).
* ``media_paths`` — optional explicit list of source media to verify.

Outputs:

* a deterministic JSON payload listing every check (missing file,
  unreadable media, suspiciously short renders, stale captions sidecar
  against media mtime, write-failure on output dir).

The module is stdlib-only and never blocks on a network call.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from opencut.openapi_registry import openapi_response_schema

logger = logging.getLogger("opencut")

_MEDIA_EXTS = {
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mxf",
    ".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff",
    ".cube", ".lut",
}

_PROJECT_DOC_EXTS = {".srt", ".vtt", ".otio", ".otioz", ".aaf", ".edl", ".xml", ".json", ".csv"}


@dataclass
class HealthCheck:
    rule: str
    severity: str  # "info" | "warning" | "error"
    path: str
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
@openapi_response_schema("/system/project-health")
class HealthReport:
    project_root: str
    media_count: int = 0
    sidecar_count: int = 0
    free_bytes: int = 0
    free_mb: int = 0
    total_bytes: int = 0
    checks: List[HealthCheck] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        payload = asdict(self)
        return payload

    def error_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == "error")

    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == "warning")


def _is_media(path: Path) -> bool:
    return path.suffix.lower() in _MEDIA_EXTS


def _is_sidecar(path: Path) -> bool:
    return path.suffix.lower() in _PROJECT_DOC_EXTS


def _scan_directory(root: Path, limit: int = 1024) -> tuple:
    media: List[Path] = []
    sidecars: List[Path] = []
    for parent, dirnames, filenames in os.walk(root):
        # Don't descend into common non-project directories.
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", "node_modules", ".git"}]
        for name in filenames:
            p = Path(parent) / name
            if _is_media(p):
                media.append(p)
            elif _is_sidecar(p):
                sidecars.append(p)
            if len(media) + len(sidecars) >= limit:
                return media, sidecars
    return media, sidecars


def _free_space(path: Path) -> tuple:
    try:
        usage = os.statvfs(path) if hasattr(os, "statvfs") else None
        if usage is not None:
            free = usage.f_bavail * usage.f_frsize
            total = usage.f_blocks * usage.f_frsize
            return free, total
    except OSError:
        pass
    try:
        import shutil

        usage = shutil.disk_usage(path)
        return usage.free, usage.total
    except OSError:
        return 0, 0


def _check_writable(path: Path) -> Optional[str]:
    """Return an error message when ``path`` cannot accept a write, else None."""
    if not path.exists():
        return "directory does not exist"
    if not path.is_dir():
        return "path exists but is not a directory"
    test_file = path / f".opencut_health_{os.getpid()}.tmp"
    try:
        test_file.write_bytes(b"")
        test_file.unlink()
    except OSError as exc:
        return f"cannot write inside directory: {exc}"
    return None


def _stale_sidecars(sidecars: Sequence[Path], media: Sequence[Path]) -> List[HealthCheck]:
    out: List[HealthCheck] = []
    media_by_stem = {m.stem: m for m in media}
    for sidecar in sidecars:
        stem = sidecar.stem
        # Strip trailing language tags from caption files: ``clip.en.srt``.
        if "." in stem:
            stem_no_lang = stem.rsplit(".", 1)[0]
            if stem_no_lang in media_by_stem:
                media_path = media_by_stem[stem_no_lang]
                try:
                    if sidecar.stat().st_mtime + 1 < media_path.stat().st_mtime:
                        out.append(
                            HealthCheck(
                                rule="stale_sidecar",
                                severity="warning",
                                path=str(sidecar),
                                message=(
                                    f"sidecar predates {media_path.name} "
                                    f"(media re-rendered after the sidecar was generated)"
                                ),
                            )
                        )
                except OSError:
                    pass
    return out


def build_report(
    project_root: str | os.PathLike,
    *,
    media_paths: Optional[Iterable[str]] = None,
    min_free_mb: int = 2048,
) -> HealthReport:
    """Run the project health checks against ``project_root``."""
    root = Path(project_root)
    report = HealthReport(project_root=str(root))

    if not root.exists():
        report.checks.append(
            HealthCheck(
                rule="project_root_missing",
                severity="error",
                path=str(root),
                message="project root does not exist",
            )
        )
        return report
    if not root.is_dir():
        report.checks.append(
            HealthCheck(
                rule="project_root_not_directory",
                severity="error",
                path=str(root),
                message="project root path is not a directory",
            )
        )
        return report

    media_from_scan, sidecars = _scan_directory(root)
    media_from_explicit = [Path(p) for p in (media_paths or [])]

    # Verify explicit paths actually exist.
    for path in media_from_explicit:
        if not path.exists():
            report.checks.append(
                HealthCheck(
                    rule="media_missing",
                    severity="error",
                    path=str(path),
                    message="explicit media path does not exist",
                )
            )
        elif path.stat().st_size == 0:
            report.checks.append(
                HealthCheck(
                    rule="media_empty",
                    severity="error",
                    path=str(path),
                    message="media file is zero bytes — likely a failed render",
                )
            )

    all_media = sorted(set(media_from_scan + media_from_explicit), key=lambda p: str(p))
    report.media_count = len(all_media)
    report.sidecar_count = len(sidecars)

    # Suspiciously small media files (< 4 KB) are usually placeholders.
    for path in all_media:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0:
            # Already flagged above for explicit paths; flag the scan ones now.
            if path not in media_from_explicit:
                report.checks.append(
                    HealthCheck(
                        rule="media_empty",
                        severity="error",
                        path=str(path),
                        message="media file is zero bytes",
                    )
                )
        elif size < 4 * 1024:
            report.checks.append(
                HealthCheck(
                    rule="media_suspiciously_small",
                    severity="warning",
                    path=str(path),
                    message=f"media file is only {size} bytes; likely truncated",
                )
            )

    # Stale sidecars (mtime older than the media they describe).
    report.checks.extend(_stale_sidecars(sidecars, all_media))

    # Free space in the project root.
    free, total = _free_space(root)
    report.free_bytes = free
    report.total_bytes = total
    report.free_mb = free // (1024 * 1024) if free else 0
    if free and free < min_free_mb * 1024 * 1024:
        report.checks.append(
            HealthCheck(
                rule="low_free_space",
                severity="warning",
                path=str(root),
                message=f"only {report.free_mb} MB free; renders may fail",
            )
        )

    # Writeability — required for any export.
    err = _check_writable(root)
    if err:
        report.checks.append(
            HealthCheck(
                rule="project_root_unwritable",
                severity="error",
                path=str(root),
                message=err,
            )
        )

    return report
