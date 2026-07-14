"""Shared bounded, path-safe handling for untrusted ZIP archives.

Project restore, Lottie inspection, and the plugin installer all consume ZIP
data that can come from outside the machine. This module centralizes the
defenses each of them needs: path-traversal / absolute-path / special-entry
rejection, configurable member-count and expanded-byte ceilings (compression-
bomb protection), bounded single-member reads, and staged-then-atomic
extraction so a rejected or failed restore never leaves a partial destination.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
import zipfile
from typing import Callable, List, Optional, Tuple

# Generous but bounded defaults. Callers override per surface (project restore
# allows more/larger members than a plugin bundle).
DEFAULT_MAX_MEMBERS = 5000
DEFAULT_MAX_TOTAL_BYTES = 512 * 1024 * 1024  # expanded
DEFAULT_MAX_MEMBER_BYTES = 256 * 1024 * 1024

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class ArchiveError(ValueError):
    """Raised when an archive violates a safety constraint."""


def normalize_member(name: str) -> str:
    """Return a safe relative path for *name*, or ``""`` for root/current dir.

    Rejects absolute paths, Windows drive paths, and any entry that escapes the
    extraction root via ``..`` traversal.
    """
    if not isinstance(name, str) or not name:
        raise ArchiveError("Archive contains an invalid entry name")
    if name.startswith(("/", "\\")) or _WINDOWS_DRIVE_RE.match(name):
        raise ArchiveError(f"Archive member uses an absolute path: {name}")

    normalized = os.path.normpath(name.replace("\\", "/")).replace("\\", "/")
    if normalized in {"", "."}:
        return ""
    if normalized == ".." or normalized.startswith("../") or "/../" in f"/{normalized}/":
        raise ArchiveError(f"Archive member escapes target directory: {name}")
    return normalized


def is_special_member(info: zipfile.ZipInfo) -> bool:
    """True if *info* is a symlink or other non-regular/non-directory entry.

    Symlinks in an archive can redirect a later write outside the destination
    even after path normalization, so they are rejected outright.
    """
    mode = info.external_attr >> 16
    if not mode:
        return False
    fmt = stat.S_IFMT(mode)
    if fmt == 0:
        # Permission bits present but no file-type bits (common for members
        # written via ZipFile.writestr) — treat as a regular file.
        return False
    if fmt == stat.S_IFLNK:
        return True
    # Regular files and directories are allowed; block/char/fifo/socket are not.
    return fmt not in (stat.S_IFREG, stat.S_IFDIR)


def resolve_within(base_dir: str, relative: str) -> str:
    """Join *relative* onto *base_dir*, guaranteeing the result stays inside."""
    base = os.path.abspath(base_dir)
    target = os.path.abspath(os.path.join(base, relative))
    if target != base and not target.startswith(base + os.sep):
        raise ArchiveError(f"Archive member escapes target directory: {relative}")
    return target


def inspect_members(
    zf: zipfile.ZipFile,
    *,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
) -> List[Tuple[zipfile.ZipInfo, str]]:
    """Validate every member up front and return ``(info, normalized_name)``.

    Raises :class:`ArchiveError` on too many members, an oversized member, an
    oversized expanded total, a special entry, or an unsafe path — before any
    bytes are written to disk.
    """
    infos = zf.infolist()
    if len(infos) > max_members:
        raise ArchiveError(
            f"Archive contains too many members ({len(infos)} > {max_members})"
        )

    total = 0
    result: List[Tuple[zipfile.ZipInfo, str]] = []
    for info in infos:
        if is_special_member(info):
            raise ArchiveError(f"Archive contains a disallowed special entry: {info.filename}")
        size = max(0, int(info.file_size))
        if size > max_member_bytes:
            raise ArchiveError(
                f"Archive member is too large ({size} > {max_member_bytes} bytes): {info.filename}"
            )
        if not info.is_dir():
            total += size
            if total > max_total_bytes:
                raise ArchiveError(
                    f"Archive expands beyond the {max_total_bytes}-byte ceiling (possible zip bomb)"
                )
        result.append((info, normalize_member(info.filename)))
    return result


def safe_read_member(
    zf: zipfile.ZipFile, name: str, *, max_bytes: int = DEFAULT_MAX_MEMBER_BYTES
) -> bytes:
    """Read a single archive member with a hard byte ceiling.

    Guards against an entry whose real expanded size exceeds its declared
    ``file_size`` by reading at most ``max_bytes + 1`` bytes and rejecting
    overflow, so a crafted member cannot exhaust memory.
    """
    info = zf.getinfo(name)
    if is_special_member(info):
        raise ArchiveError(f"Archive member is a disallowed special entry: {name}")
    if int(info.file_size) > max_bytes:
        raise ArchiveError(f"Archive member {name} exceeds the {max_bytes}-byte read limit")
    with zf.open(info, "r") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ArchiveError(f"Archive member {name} exceeds the {max_bytes}-byte read limit")
    return data


def _promote(staging: str, dest_path: str) -> None:
    """Atomically move *staging* into *dest_path* (replacing any existing dir)."""
    dest = os.path.abspath(dest_path)
    if os.path.exists(dest):
        backup = f"{dest}.ocbak_{os.path.basename(staging)}"
        os.replace(dest, backup)
        try:
            os.replace(staging, dest)
        except Exception:
            os.replace(backup, dest)  # roll back
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        os.replace(staging, dest)


def safe_extract_all(
    archive_path: str,
    dest_path: str,
    *,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    on_member: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Extract *archive_path* into *dest_path* transactionally.

    All members are validated first, then extracted into a staging directory on
    the same filesystem, then promoted atomically. On any failure the staging
    directory is removed and the destination is left untouched. Returns the
    number of files written.
    """
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise ArchiveError(f"Not a valid zip file: {archive_path}")

    parent = os.path.dirname(os.path.abspath(dest_path)) or "."
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".octmp_restore_", dir=parent)
    files_written = 0
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = inspect_members(
                zf,
                max_members=max_members,
                max_total_bytes=max_total_bytes,
                max_member_bytes=max_member_bytes,
            )
            total = len(members)
            for info, normalized in members:
                if not normalized:
                    continue
                target = resolve_within(staging, normalized)
                if info.is_dir() or info.filename.endswith("/"):
                    os.makedirs(target, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(info, "r") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                files_written += 1
                if on_member is not None:
                    on_member(files_written, total)
        _promote(staging, dest_path)
        staging = None
        return files_written
    finally:
        if staging is not None and os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
