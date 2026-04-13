"""
OpenCut Storage Tiering Module v1.0.0

Track file access times, move unused media to archive path,
restore on demand. Stub files left behind for transparent restore.
"""

import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ArchiveFileResult:
    """Result for a single archived or restored file."""
    original_path: str
    archive_path: str = ""
    status: str = "pending"  # pending, archived, restored, skipped, failed
    size_bytes: int = 0
    idle_days: float = 0.0
    error: str = ""


@dataclass
class ArchiveScanResult:
    """Result for scanning files eligible for archival."""
    total_scanned: int = 0
    eligible_count: int = 0
    eligible_size_bytes: int = 0
    files: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArchiveResult:
    """Result for a batch archive operation."""
    total: int = 0
    archived: int = 0
    failed: int = 0
    skipped: int = 0
    freed_bytes: int = 0
    results: List[ArchiveFileResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Manifest management
# ---------------------------------------------------------------------------

_MANIFEST_NAME = ".opencut_archive_manifest.json"


def _load_manifest(archive_path: str) -> dict:
    """Load the archive manifest file."""
    manifest_path = os.path.join(archive_path, _MANIFEST_NAME)
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load manifest: %s", e)
    return {"version": 1, "entries": {}}


def _save_manifest(archive_path: str, manifest: dict):
    """Save the archive manifest file."""
    os.makedirs(archive_path, exist_ok=True)
    manifest_path = os.path.join(archive_path, _MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def get_archive_manifest(archive_path: str = "") -> dict:
    """Get the current archive manifest.

    Args:
        archive_path: Path to archive directory. If empty, returns
            empty manifest structure.

    Returns:
        Dict with ``version`` and ``entries`` mapping original paths
        to archive metadata.
    """
    if not archive_path or not os.path.isdir(archive_path):
        return {"version": 1, "entries": {}}
    return _load_manifest(archive_path)


# ---------------------------------------------------------------------------
# Stub files
# ---------------------------------------------------------------------------

_STUB_EXT = ".opencut_stub"


def _create_stub(original_path: str, archive_dest: str):
    """Create a stub file at the original location pointing to archive."""
    stub_path = original_path + _STUB_EXT
    stub_data = {
        "type": "opencut_archive_stub",
        "original_path": original_path,
        "archive_path": archive_dest,
        "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(stub_path, "w", encoding="utf-8") as f:
        json.dump(stub_data, f, indent=2)
    return stub_path


def _read_stub(stub_path: str) -> dict:
    """Read a stub file and return its metadata."""
    with open(stub_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Scan for archival candidates
# ---------------------------------------------------------------------------

_MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".mxf", ".prores",
    ".wav", ".mp3", ".aac", ".flac", ".ogg",
    ".mts", ".m2ts", ".ts", ".webm", ".r3d",
    ".braw", ".ari", ".dpx", ".exr", ".tif", ".tiff",
    ".png", ".jpg", ".jpeg",
}


def scan_for_archival(
    project_dir: str,
    idle_days: float = 30.0,
    on_progress: Optional[Callable] = None,
) -> ArchiveScanResult:
    """Scan a project directory for files idle beyond threshold.

    Args:
        project_dir: Project directory to scan.
        idle_days: Number of days since last access to consider idle.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        :class:`ArchiveScanResult` with eligible files.
    """
    if not os.path.isdir(project_dir):
        raise FileNotFoundError(f"Directory not found: {project_dir}")

    result = ArchiveScanResult()
    now = time.time()
    threshold_seconds = idle_days * 86400

    if on_progress:
        on_progress(5, f"Scanning {project_dir} for idle files...")

    all_files = []
    for root, dirs, files in os.walk(project_dir):
        # Skip hidden directories and archive stubs
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            full_path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext in _MEDIA_EXTENSIONS:
                all_files.append(full_path)

    result.total_scanned = len(all_files)

    for i, fpath in enumerate(all_files):
        try:
            stat = os.stat(fpath)
            # Use the most recent of atime and mtime
            last_access = max(stat.st_atime, stat.st_mtime)
            idle_secs = now - last_access
            file_idle_days = idle_secs / 86400

            if idle_secs >= threshold_seconds:
                result.eligible_count += 1
                result.eligible_size_bytes += stat.st_size
                result.files.append({
                    "path": fpath,
                    "size_bytes": stat.st_size,
                    "idle_days": round(file_idle_days, 1),
                    "last_accessed": time.strftime(
                        "%Y-%m-%dT%H:%M:%S", time.localtime(last_access)
                    ),
                })
        except OSError as e:
            logger.debug("Could not stat %s: %s", fpath, e)

        if on_progress and (i + 1) % 50 == 0:
            pct = min(int(((i + 1) / len(all_files)) * 90) + 5, 95)
            on_progress(pct, f"Scanned {i + 1}/{len(all_files)} files")

    if on_progress:
        on_progress(100, f"Scan complete: {result.eligible_count} files eligible")

    return result


# ---------------------------------------------------------------------------
# Archive files
# ---------------------------------------------------------------------------

def archive_files(
    file_list: List[str],
    archive_path: str,
    on_progress: Optional[Callable] = None,
) -> ArchiveResult:
    """Archive a list of files to the archive directory.

    Moves files to archive_path, creates stub files at original locations,
    and updates the archive manifest.

    Args:
        file_list: List of file paths to archive.
        archive_path: Destination archive directory.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        :class:`ArchiveResult` with per-file status.
    """
    if not file_list:
        raise ValueError("No files provided for archival")

    os.makedirs(archive_path, exist_ok=True)
    manifest = _load_manifest(archive_path)
    result = ArchiveResult(total=len(file_list))

    if on_progress:
        on_progress(5, f"Archiving {len(file_list)} files...")

    for i, src_path in enumerate(file_list):
        file_result = ArchiveFileResult(original_path=src_path)

        try:
            if not os.path.isfile(src_path):
                file_result.status = "skipped"
                file_result.error = "File not found"
                result.skipped += 1
                result.results.append(file_result)
                continue

            stat = os.stat(src_path)
            file_result.size_bytes = stat.st_size

            # Compute idle days
            now = time.time()
            last_access = max(stat.st_atime, stat.st_mtime)
            file_result.idle_days = round((now - last_access) / 86400, 1)

            # Preserve relative path structure in archive
            rel_name = os.path.basename(src_path)
            dest_path = os.path.join(archive_path, rel_name)

            # Handle name collisions
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(rel_name)
                counter = 2
                while os.path.exists(dest_path):
                    dest_path = os.path.join(archive_path, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.move(src_path, dest_path)
            file_result.archive_path = dest_path
            file_result.status = "archived"

            # Create stub at original location
            _create_stub(src_path, dest_path)

            # Update manifest
            manifest["entries"][src_path] = {
                "archive_path": dest_path,
                "size_bytes": stat.st_size,
                "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "idle_days": file_result.idle_days,
            }

            result.archived += 1
            result.freed_bytes += stat.st_size

        except Exception as e:
            file_result.status = "failed"
            file_result.error = str(e)
            result.failed += 1
            logger.error("Archive failed for %s: %s", src_path, e)

        result.results.append(file_result)

        if on_progress:
            pct = min(int(((i + 1) / len(file_list)) * 90) + 5, 95)
            on_progress(pct, f"Archived {i + 1}/{len(file_list)}")

    _save_manifest(archive_path, manifest)

    if on_progress:
        freed_mb = round(result.freed_bytes / (1024 * 1024), 1)
        on_progress(100, f"Archive complete: {result.archived} files, {freed_mb} MB freed")

    return result


# ---------------------------------------------------------------------------
# Restore file
# ---------------------------------------------------------------------------

def restore_file(
    stub_path: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Restore a file from archive using its stub.

    Args:
        stub_path: Path to the .opencut_stub file, or the original file
            path (stub extension appended automatically).
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        Dict with ``original_path``, ``archive_path``, ``status``.
    """
    # Accept either stub path or original path
    if not stub_path.endswith(_STUB_EXT):
        stub_path = stub_path + _STUB_EXT

    if not os.path.isfile(stub_path):
        raise FileNotFoundError(f"Stub file not found: {stub_path}")

    if on_progress:
        on_progress(10, "Reading stub file...")

    stub_data = _read_stub(stub_path)
    archive_src = stub_data.get("archive_path", "")
    original_dest = stub_data.get("original_path", "")

    if not archive_src or not os.path.isfile(archive_src):
        raise FileNotFoundError(f"Archived file not found: {archive_src}")

    if on_progress:
        on_progress(30, f"Restoring {os.path.basename(original_dest)}...")

    # Restore the file
    os.makedirs(os.path.dirname(original_dest) or ".", exist_ok=True)
    shutil.move(archive_src, original_dest)

    # Remove the stub
    try:
        os.unlink(stub_path)
    except OSError:
        pass

    # Update manifest
    archive_dir = os.path.dirname(archive_src)
    manifest = _load_manifest(archive_dir)
    manifest["entries"].pop(original_dest, None)
    _save_manifest(archive_dir, manifest)

    if on_progress:
        on_progress(100, f"Restored {os.path.basename(original_dest)}")

    return {
        "original_path": original_dest,
        "archive_path": archive_src,
        "status": "restored",
    }
