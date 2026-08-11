"""FTS5 memory-safety floor for locally-opened SQLite databases (F309).

CVE-2026-11822 is an out-of-bounds read in ``fts5LeafSeek()`` and a heap
overflow in ``fts5ChunkIterate()``, reachable by running a ``MATCH`` query
against a *crafted database file*. It is fixed in SQLite 3.53.2.

OpenCut runs FTS5 over the footage index and the federated media index. Both
are normally created by this install, which makes them trusted input — the
realistic attack is a restored, copied, or shared index file. So the policy is
narrow on purpose:

* every index this install creates is stamped with a provenance marker;
* on a runtime below the floor, opening an index **without** that marker is
  refused with a typed error;
* a self-created index on any runtime, and any index on a patched runtime,
  behave exactly as before.

Refusing every unpatched runtime outright would break offline users whose
Python ships an older SQLite while giving them nothing: their own index is not
the attack vector.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

logger = logging.getLogger("opencut")

#: First SQLite release carrying the CVE-2026-11822 FTS5 fixes.
FTS5_SAFE_VERSION: tuple[int, int, int] = (3, 53, 2)

#: PRAGMA slot used to stamp indexes this install created.
PROVENANCE_PRAGMA = "application_id"

#: ASCII "OCUT" — identifies a database created by OpenCut.
OPENCUT_APPLICATION_ID = 0x4F435554

CVE_REFERENCE = "CVE-2026-11822"


class UntrustedFts5DatabaseError(RuntimeError):
    """Raised when a foreign FTS5 index is opened on a vulnerable runtime."""

    code = "UNTRUSTED_FTS5_DATABASE"

    def __init__(self, path: str, runtime_version: str):
        self.path = path
        self.runtime_version = runtime_version
        super().__init__(
            f"Refusing to open {path!r}: it was not created by this OpenCut "
            f"install, and SQLite {runtime_version} predates "
            f"{'.'.join(map(str, FTS5_SAFE_VERSION))}, which fixes "
            f"{CVE_REFERENCE} (FTS5 memory corruption from a crafted database "
            "file). Upgrade Python's bundled SQLite, or delete the index so "
            "OpenCut can rebuild it locally."
        )


def runtime_version() -> str:
    """Return the SQLite library version this process is linked against."""
    return sqlite3.sqlite_version


def runtime_version_tuple() -> tuple[int, ...]:
    parts = []
    for chunk in sqlite3.sqlite_version.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def fts5_runtime_is_patched() -> bool:
    """True when the linked SQLite carries the CVE-2026-11822 fixes."""
    return runtime_version_tuple() >= FTS5_SAFE_VERSION


def stamp_local_provenance(conn: sqlite3.Connection) -> None:
    """Mark a database as created by this install. Safe to call repeatedly."""
    try:
        conn.execute(f"PRAGMA {PROVENANCE_PRAGMA} = {OPENCUT_APPLICATION_ID}")
    except sqlite3.Error as exc:
        logger.warning("could not stamp SQLite provenance: %s", exc)


def has_local_provenance(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute(f"PRAGMA {PROVENANCE_PRAGMA}").fetchone()
    except sqlite3.Error as exc:
        logger.warning("could not read SQLite provenance: %s", exc)
        return False
    return bool(row) and int(row[0]) == OPENCUT_APPLICATION_ID


def is_inside_user_data_dir(path: str) -> bool:
    """True when *path* lives under the OpenCut user-data directory.

    That directory is created and owned by this install, so an index inside it
    is ours by construction — including indexes written before provenance
    stamping existed. Without this, adding the guard would refuse every
    already-installed user's index on an unpatched runtime.
    """
    try:
        import os

        from opencut.user_data import OPENCUT_DIR

        base = os.path.realpath(OPENCUT_DIR)
        target = os.path.realpath(path)
        return os.path.commonpath([base, target]) == base
    except Exception:  # noqa: BLE001 - an unresolvable path is not "ours"
        return False


def ensure_fts5_database_trusted(
    conn: sqlite3.Connection,
    path: str,
    *,
    created_here: bool = False,
) -> None:
    """Refuse a foreign FTS5 index on a runtime below the floor.

    Trusted, in order: a database this call just created; any database on a
    patched runtime; one already stamped by this install; and one living inside
    the OpenCut user-data directory. Everything else — a restored, copied, or
    downloaded index pointed at from elsewhere — is refused while the runtime
    predates the CVE-2026-11822 fixes.
    """
    if created_here:
        stamp_local_provenance(conn)
        return
    if fts5_runtime_is_patched():
        # Patched runtime: stamp opportunistically so a later downgrade still
        # recognises our own index.
        stamp_local_provenance(conn)
        return
    if has_local_provenance(conn):
        return
    if is_inside_user_data_dir(path):
        # Pre-existing local index from before stamping; adopt it.
        stamp_local_provenance(conn)
        return
    raise UntrustedFts5DatabaseError(path, runtime_version())


def fts5_safety_report() -> dict:
    """Diagnostic payload for system status and release smoke."""
    patched = fts5_runtime_is_patched()
    return {
        "sqlite_version": runtime_version(),
        "fts5_floor": ".".join(map(str, FTS5_SAFE_VERSION)),
        "fts5_runtime_patched": patched,
        "cve": CVE_REFERENCE,
        "policy": (
            "all indexes opened normally"
            if patched
            else "indexes not created by this install are refused"
        ),
    }


def resolve_optional_version(value: Optional[str]) -> tuple[int, ...]:
    """Parse a dotted version string for tests and external callers."""
    parts = []
    for chunk in (value or "0").split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])
