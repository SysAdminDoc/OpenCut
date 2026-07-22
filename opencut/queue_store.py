"""Versioned, atomic persistence for the legacy job queue."""

from __future__ import annotations

import copy
import time

from opencut.user_data import read_user_file, write_user_file

QUEUE_FILE = "job_queue.json"
QUEUE_SCHEMA_VERSION = 1


class QueueDocumentError(ValueError):
    """Raised when a queue document cannot be safely loaded or imported."""


class QueueSchemaVersionError(QueueDocumentError):
    """Raised for an unsupported schema version.

    Kept distinct from structural corruption: a *newer* schema likely holds
    real work written by a newer OpenCut, so startup fails closed instead of
    quarantining the file.
    """


def build_document(entries: list[dict]) -> dict:
    """Return a detached, versioned queue document."""
    return {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "exported_at": time.time(),
        "entries": copy.deepcopy(entries),
    }


def parse_document(data, *, allow_legacy: bool = False) -> tuple[list[dict], bool]:
    """Validate a queue document and return ``(entries, migrated)``.

    The on-disk reader accepts the never-published legacy list shape so a
    pre-release queue file is not stranded. Public imports remain explicitly
    versioned.
    """
    if allow_legacy and isinstance(data, list):
        return copy.deepcopy(data), True
    if not isinstance(data, dict):
        raise QueueDocumentError("Queue document must be a JSON object")
    version = data.get("schema_version")
    if isinstance(version, int) and not isinstance(version, bool) and version != QUEUE_SCHEMA_VERSION:
        raise QueueSchemaVersionError(
            f"Unsupported queue schema version: {version!r}; "
            f"expected {QUEUE_SCHEMA_VERSION}"
        )
    if isinstance(version, bool) or not isinstance(version, int):
        raise QueueDocumentError(
            f"Unsupported queue schema version: {version!r}; "
            f"expected {QUEUE_SCHEMA_VERSION}"
        )
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise QueueDocumentError("Queue document entries must be a JSON array")
    return copy.deepcopy(entries), False


def load_queue() -> tuple[list[dict], bool]:
    """Load the persisted queue, returning ``(entries, migrated)``."""
    missing = object()
    data = read_user_file(QUEUE_FILE, default=missing)
    if data is missing:
        return [], False
    return parse_document(data, allow_legacy=True)


def save_queue(entries: list[dict]) -> None:
    """Atomically replace the persisted queue document."""
    write_user_file(QUEUE_FILE, build_document(entries))
