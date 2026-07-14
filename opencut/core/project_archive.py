"""
OpenCut Project Archival / Package

Collect all source media, outputs, workflows, and presets into an
archive directory.  Compress to zip.  Restore from archive.
"""

import json
import logging
import os
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from opencut.core import archive_safety

logger = logging.getLogger("opencut")

_MANIFEST_NAME = "opencut_manifest.json"
_MANIFEST_VERSION = 1


@dataclass
class ArchiveResult:
    """Result of creating a project archive."""
    archive_path: str
    total_files: int
    total_bytes: int
    manifest_included: bool


@dataclass
class RestoreResult:
    """Result of restoring a project archive."""
    dest_path: str
    files_restored: int
    manifest: Dict[str, Any]


def create_archive(
    project_data: Dict[str, Any],
    output_path: str,
    on_progress: Optional[Callable] = None,
) -> ArchiveResult:
    """Create a project archive as a zip file.

    Args:
        project_data: Dict with keys:
            - 'name': Project name.
            - 'source_files': List of file paths (source media).
            - 'output_files': List of file paths (rendered outputs).
            - 'workflows': List of workflow dicts (saved as JSON).
            - 'presets': List of preset dicts (saved as JSON).
            - 'metadata': Optional dict of extra project metadata.
        output_path: Path for the output .zip file.
        on_progress: Optional progress callback.

    Returns:
        ArchiveResult with archive path, file count, and byte size.
    """
    if on_progress:
        on_progress(5, "Preparing archive...")

    name = project_data.get("name", "Untitled Project")
    source_files = project_data.get("source_files", [])
    output_files = project_data.get("output_files", [])
    workflows = project_data.get("workflows", [])
    presets = project_data.get("presets", [])
    metadata = project_data.get("metadata", {})

    # Collect all files to archive
    all_files: List[str] = []
    for f in source_files + output_files:
        if isinstance(f, str) and os.path.isfile(f):
            all_files.append(f)

    if not all_files and not workflows and not presets:
        raise ValueError("Nothing to archive: no files, workflows, or presets provided")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Build manifest
    manifest = {
        "version": _MANIFEST_VERSION,
        "name": name,
        "created": time.time(),
        "source_files": [],
        "output_files": [],
        "workflows": workflows,
        "presets": presets,
        "metadata": metadata,
    }

    total_files = 0
    total_bytes = 0

    if on_progress:
        on_progress(10, "Creating zip archive...")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Archive source media
        for i, fpath in enumerate(source_files):
            if not isinstance(fpath, str) or not os.path.isfile(fpath):
                continue
            arcname = os.path.join("sources", os.path.basename(fpath))
            zf.write(fpath, arcname)
            fsize = os.path.getsize(fpath)
            manifest["source_files"].append({
                "original_path": fpath,
                "archive_path": arcname,
                "size": fsize,
            })
            total_files += 1
            total_bytes += fsize

            if on_progress and all_files:
                pct = 10 + int(((i + 1) / len(all_files)) * 60)
                on_progress(pct, f"Archiving source {i + 1}...")

        # Archive output files
        for i, fpath in enumerate(output_files):
            if not isinstance(fpath, str) or not os.path.isfile(fpath):
                continue
            arcname = os.path.join("outputs", os.path.basename(fpath))
            zf.write(fpath, arcname)
            fsize = os.path.getsize(fpath)
            manifest["output_files"].append({
                "original_path": fpath,
                "archive_path": arcname,
                "size": fsize,
            })
            total_files += 1
            total_bytes += fsize

        # Archive workflows as JSON
        if workflows:
            workflow_json = json.dumps(workflows, indent=2)
            zf.writestr("workflows.json", workflow_json)
            total_files += 1

        # Archive presets as JSON
        if presets:
            presets_json = json.dumps(presets, indent=2)
            zf.writestr("presets.json", presets_json)
            total_files += 1

        if on_progress:
            on_progress(85, "Writing manifest...")

        # Write manifest
        manifest_json = json.dumps(manifest, indent=2)
        zf.writestr(_MANIFEST_NAME, manifest_json)

    if on_progress:
        on_progress(100, "Archive created")

    archive_size = os.path.getsize(output_path)
    logger.info(
        "Created archive '%s': %d files, %d bytes (compressed %d bytes)",
        name, total_files, total_bytes, archive_size,
    )
    return ArchiveResult(
        archive_path=output_path,
        total_files=total_files,
        total_bytes=total_bytes,
        manifest_included=True,
    )


def restore_archive(
    archive_path: str,
    dest_path: str,
    on_progress: Optional[Callable] = None,
) -> RestoreResult:
    """Restore a project archive from a zip file.

    Args:
        archive_path: Path to the .zip archive.
        dest_path: Destination directory for extracted files.
        on_progress: Optional progress callback.

    Returns:
        RestoreResult with destination path, file count, and manifest.
    """
    if on_progress:
        on_progress(5, "Opening archive...")

    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Not a valid zip file: {archive_path}")

    # Read the manifest with a bounded read before touching the destination.
    manifest = {}
    with zipfile.ZipFile(archive_path, "r") as zf:
        if _MANIFEST_NAME in zf.namelist():
            manifest_data = archive_safety.safe_read_member(zf, _MANIFEST_NAME)
            manifest = json.loads(manifest_data.decode("utf-8"))

    if on_progress:
        on_progress(10, "Extracting items...")

    def _cb(done: int, total: int) -> None:
        if on_progress:
            pct = 10 + int((done / max(1, total)) * 85)
            on_progress(pct, f"Extracting {done}/{total}...")

    # Validate every member, extract to a staging dir, then promote atomically:
    # a rejected or failed restore leaves no partial destination. Project
    # archives may legitimately carry large media, so the ceilings are generous
    # while still bounding member count and compression-bomb expansion.
    files_restored = archive_safety.safe_extract_all(
        archive_path,
        dest_path,
        max_members=50000,
        max_total_bytes=4 * 1024 * 1024 * 1024,
        max_member_bytes=2 * 1024 * 1024 * 1024,
        on_member=_cb,
    )

    if on_progress:
        on_progress(100, "Archive restored")

    logger.info("Restored archive to %s: %d files", dest_path, files_restored)
    return RestoreResult(
        dest_path=dest_path,
        files_restored=files_restored,
        manifest=manifest,
    )


def list_archive_contents(archive_path: str) -> Dict[str, Any]:
    """List the contents of a project archive without extracting.

    Args:
        archive_path: Path to the .zip archive.

    Returns:
        Dict with manifest data and file list.
    """
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Not a valid zip file: {archive_path}")

    manifest = {}
    file_list = []

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()

        if _MANIFEST_NAME in zf.namelist():
            manifest_data = zf.read(_MANIFEST_NAME)
            manifest = json.loads(manifest_data.decode("utf-8"))

        for info in members:
            file_list.append({
                "name": info.filename,
                "size": info.file_size,
                "compressed_size": info.compress_size,
                "is_dir": info.is_dir(),
            })

    return {
        "archive_path": archive_path,
        "archive_size": os.path.getsize(archive_path),
        "manifest": manifest,
        "files": file_list,
        "total_files": len([f for f in file_list if not f["is_dir"]]),
    }
