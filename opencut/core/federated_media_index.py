"""Offline, move-aware federation for OpenCut media search.

The application historically kept transcript search, multimodal metadata,
and visual sidecars in separate stores.  This module is the small library
registry that makes those stores queryable as one bounded, local-only index.
It deliberately imports existing metadata and sidecar embeddings; it never
loads a visual model or downloads anything while scanning or searching.

The database is a versioned user-local manifest.  Absolute paths are retained
for reconciliation, but public records omit them unless the caller explicitly
opts into ``include_paths``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from collections.abc import Callable, Iterable
from typing import Any

from opencut.core.sqlite_safety import ensure_fts5_database_trusted
from opencut.local_db_migrations import migrate_user_version

logger = logging.getLogger("opencut")

_DB_PATH = os.path.join(
    os.path.expanduser("~"), ".opencut", "federated_media_index.db"
)

SCHEMA_VERSION = 1
MANIFEST_VERSION = 1
DEFAULT_RETENTION_DAYS = 30
MAX_ROOTS = 32
MAX_FILES_PER_SCAN = 5000
MAX_SCAN_BYTES = 50 * 1024 * 1024 * 1024
MAX_QUERY_LENGTH = 500
MAX_RESULTS = 100

MEDIA_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
        ".mpeg",
        ".mpg",
        ".ts",
        ".mts",
        ".m2ts",
        ".3gp",
        ".mp3",
        ".wav",
        ".flac",
        ".aac",
        ".ogg",
        ".m4a",
        ".wma",
        ".aiff",
        ".aif",
    }
)
VIDEO_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
        ".mpeg",
        ".mpg",
        ".ts",
        ".mts",
        ".m2ts",
        ".3gp",
    }
)
MODALITIES = ("text", "ocr", "audio", "visual")
MODALITY_ALIASES = {
    "transcript": "text",
    "transcripts": "text",
    "audio_tags": "audio",
    "audio-tags": "audio",
    "image": "visual",
}
MODALITY_STATES = (
    "available",
    "unindexed",
    "missing_model",
    "stale",
    "schema_incompatible",
    "error",
)
_SKIP_DIRS = frozenset(
    {".opencut", ".opencut_index", ".git", "node_modules", "__pycache__"}
)


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    # F309: see opencut.core.sqlite_safety — a foreign index is untrusted
    # input for FTS5 MATCH on a runtime predating the CVE-2026-11822 fixes.
    created_here = not os.path.exists(_DB_PATH)
    conn = sqlite3.connect(_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        ensure_fts5_database_trusted(conn, _DB_PATH, created_here=created_here)
    except Exception:
        conn.close()
        raise
    return conn


def _create_schema_v1(conn: sqlite3.Connection) -> None:
    statements = (
        """
        CREATE TABLE IF NOT EXISTS roots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            added_at REAL NOT NULL,
            last_scan_at REAL NOT NULL DEFAULT 0,
            last_scan_complete INTEGER NOT NULL DEFAULT 0,
            retention_days INTEGER NOT NULL DEFAULT 30,
            max_files INTEGER NOT NULL DEFAULT 5000,
            max_bytes INTEGER NOT NULL DEFAULT 53687091200
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            root_id INTEGER NOT NULL REFERENCES roots(id) ON DELETE CASCADE,
            relative_path TEXT NOT NULL,
            source_signature TEXT NOT NULL,
            file_size INTEGER NOT NULL DEFAULT 0,
            file_mtime_ns INTEGER NOT NULL DEFAULT 0,
            duration REAL NOT NULL DEFAULT 0,
            media_type TEXT NOT NULL DEFAULT 'unknown',
            status TEXT NOT NULL DEFAULT 'active',
            first_seen_at REAL NOT NULL,
            last_seen_at REAL NOT NULL,
            missing_since REAL,
            last_error TEXT NOT NULL DEFAULT '',
            UNIQUE(root_id, relative_path)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS modalities (
            media_id INTEGER NOT NULL REFERENCES media(id) ON DELETE CASCADE,
            modality TEXT NOT NULL,
            engine TEXT NOT NULL DEFAULT '',
            schema_version INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'unindexed',
            text TEXT NOT NULL DEFAULT '',
            timestamps_json TEXT NOT NULL DEFAULT '[]',
            thumbnail_path TEXT NOT NULL DEFAULT '',
            capability_json TEXT NOT NULL DEFAULT '{}',
            updated_at REAL NOT NULL,
            PRIMARY KEY(media_id, modality)
        )
        """,
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(
            media_id UNINDEXED,
            content
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_media_root_status
            ON media(root_id, status);
        """.replace(";", ""),
        """
        CREATE INDEX IF NOT EXISTS idx_media_signature
            ON media(root_id, source_signature);
        """.replace(";", ""),
        """
        CREATE INDEX IF NOT EXISTS idx_modalities_state
            ON modalities(modality, state);
        """.replace(";", ""),
    )
    for statement in statements:
        conn.execute(statement)


def init_db() -> None:
    """Create or migrate the local federation database."""
    with _connect() as conn:
        migrate_user_version(
            conn,
            store_name="federated media index",
            target_version=SCHEMA_VERSION,
            migrations={1: _create_schema_v1},
        )


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    return max(minimum, min(maximum, result))


def _canonical_root(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("root path is required")
    resolved = os.path.realpath(os.path.abspath(path.strip()))
    if not os.path.isdir(resolved):
        raise ValueError(f"root is not a directory: {path}")
    return resolved


def _root_ids(value: Iterable[Any] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        value = [value]
    result = []
    for item in value:
        try:
            root_id = int(item)
        except (TypeError, ValueError):
            raise ValueError("root_ids must contain integers") from None
        if root_id < 1:
            raise ValueError("root_ids must contain positive integers")
        if root_id not in result:
            result.append(root_id)
    return result


def _root_filter(root_ids: list[int] | None, *, enabled_only: bool = True) -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    if enabled_only:
        clauses.append("r.enabled = 1")
    if root_ids:
        clauses.append("r.id IN (" + ",".join("?" for _ in root_ids) + ")")
        params.extend(root_ids)
    return (" AND ".join(clauses) or "1=1", params)


def _root_record(conn: sqlite3.Connection, row: sqlite3.Row, *, include_path: bool) -> dict:
    counts = conn.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active,
               SUM(CASE WHEN status = 'missing' THEN 1 ELSE 0 END) AS missing,
               SUM(CASE WHEN status = 'deleted' THEN 1 ELSE 0 END) AS deleted
        FROM media WHERE root_id = ?
        """,
        (row["id"],),
    ).fetchone()
    record = {
        "root_id": int(row["id"]),
        "label": row["label"] or os.path.basename(row["path"]) or row["path"],
        "enabled": bool(row["enabled"]),
        "added_at": float(row["added_at"] or 0),
        "last_scan_at": float(row["last_scan_at"] or 0),
        "last_scan_complete": bool(row["last_scan_complete"]),
        "retention_days": int(row["retention_days"]),
        "max_files": int(row["max_files"]),
        "max_bytes": int(row["max_bytes"]),
        "file_counts": {
            "total": int(counts["total"] or 0),
            "active": int(counts["active"] or 0),
            "missing": int(counts["missing"] or 0),
            "deleted": int(counts["deleted"] or 0),
        },
    }
    if include_path:
        record["path"] = row["path"]
    return record


def add_root(
    path: str,
    label: str = "",
    *,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    max_files: int = MAX_FILES_PER_SCAN,
    max_bytes: int = MAX_SCAN_BYTES,
) -> dict:
    """Register or re-enable one existing local project root."""
    resolved = _canonical_root(path)
    label = str(label or "").strip()[:120]
    retention_days = _bounded_int(retention_days, DEFAULT_RETENTION_DAYS, 1, 3650)
    max_files = _bounded_int(max_files, MAX_FILES_PER_SCAN, 1, MAX_FILES_PER_SCAN)
    max_bytes = _bounded_int(max_bytes, MAX_SCAN_BYTES, 1, MAX_SCAN_BYTES)
    now = time.time()
    init_db()
    with _connect() as conn:
        existing = conn.execute("SELECT id FROM roots WHERE path = ?", (resolved,)).fetchone()
        if existing is None:
            root_count = int(conn.execute("SELECT COUNT(*) FROM roots").fetchone()[0])
            if root_count >= MAX_ROOTS:
                raise ValueError(f"maximum configured roots reached (max {MAX_ROOTS})")
        conn.execute(
            """
            INSERT INTO roots(
                path, label, enabled, added_at, retention_days, max_files, max_bytes
            ) VALUES (?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                label = excluded.label,
                enabled = 1,
                retention_days = excluded.retention_days,
                max_files = excluded.max_files,
                max_bytes = excluded.max_bytes
            """,
            (resolved, label, now, retention_days, max_files, max_bytes),
        )
        row = conn.execute("SELECT * FROM roots WHERE path = ?", (resolved,)).fetchone()
        return _root_record(conn, row, include_path=True)


def list_roots(*, include_paths: bool = False, include_disabled: bool = False) -> list[dict]:
    """Return configured roots, redacting absolute paths by default."""
    init_db()
    with _connect() as conn:
        where = "" if include_disabled else "WHERE enabled = 1"
        rows = conn.execute(f"SELECT * FROM roots {where} ORDER BY id").fetchall()
        return [
            _root_record(conn, row, include_path=include_paths) for row in rows
        ]


def remove_root(root_id: int, *, purge: bool = False) -> dict:
    """Disable a root, or explicitly purge its indexed records."""
    try:
        root_id = int(root_id)
    except (TypeError, ValueError):
        raise ValueError("root_id must be an integer") from None
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM roots WHERE id = ?", (root_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown root_id: {root_id}")
        if purge:
            conn.execute("DELETE FROM roots WHERE id = ?", (root_id,))
        else:
            conn.execute("UPDATE roots SET enabled = 0 WHERE id = ?", (root_id,))
        return {
            "root_id": root_id,
            "disabled": not purge,
            "purged": bool(purge),
            "file_count": int(
                conn.execute("SELECT COUNT(*) FROM media WHERE root_id = ?", (root_id,)).fetchone()[0]
            ) if not purge else 0,
        }


def _safe_json(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return default
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return default


def _normalise_segments(segments: Any, duration: float, fallback_text: str = "") -> list[dict]:
    result = []
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            start = max(0.0, _finite_float(segment.get("start"), 0.0))
            end = max(start, _finite_float(segment.get("end"), start))
            if duration > 0:
                start = min(start, duration)
                end = min(max(start, end), duration)
            result.append({"start": round(start, 6), "end": round(end, 6), "text": text})
    if not result and str(fallback_text or "").strip():
        end = max(0.0, duration)
        result.append({"start": 0.0, "end": round(end, 6), "text": str(fallback_text).strip()})
    return result


def _mtime_matches(record: dict | None, mtime: float) -> bool:
    if not record:
        return False
    indexed = _finite_float(record.get("file_mtime", record.get("mtime")), -1.0)
    return indexed >= 0 and abs(indexed - mtime) <= 0.01


def _legacy_entry(index: dict, path: str) -> dict | None:
    if not isinstance(index, dict):
        return None
    entry = index.get(path)
    if isinstance(entry, dict):
        return entry
    wanted = os.path.normcase(os.path.realpath(path))
    for candidate, value in index.items():
        if not isinstance(candidate, str) or not isinstance(value, dict):
            continue
        if os.path.normcase(os.path.realpath(candidate)) == wanted:
            return value
    return None


def _load_legacy_metadata(path: str, mtime: float) -> dict:
    metadata = {"db": None, "json": None}
    try:
        from opencut.core.footage_index_db import get_indexed_file

        metadata["db"] = get_indexed_file(path)
    except Exception as exc:  # noqa: BLE001 - an optional legacy store
        logger.debug("Could not import footage DB row for %s: %s", path, exc)
    try:
        from opencut.core import footage_search

        metadata["json"] = _legacy_entry(footage_search.load_index(), path)
    except Exception as exc:  # noqa: BLE001 - an optional legacy store
        logger.debug("Could not import footage JSON row for %s: %s", path, exc)
    return metadata


def _video_duration(path: str, media_type: str) -> float:
    if media_type not in {"video", "audio"}:
        return 0.0
    try:
        from opencut.helpers import get_video_info

        return max(0.0, _finite_float(get_video_info(path).get("duration"), 0.0))
    except Exception as exc:  # noqa: BLE001 - media probing is best-effort
        logger.debug("Could not probe duration for %s: %s", path, exc)
        return 0.0


def _media_type(path: str) -> str:
    return "video" if os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS else "audio"


def _safe_thumbnail(path: Any, root_path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        return ""
    candidate = os.path.realpath(path.strip())
    root = os.path.realpath(root_path)
    try:
        if os.path.commonpath([candidate, root]) != root or not os.path.isfile(candidate):
            return ""
    except ValueError:
        return ""
    return candidate


def _visual_metadata(root_path: str, path: str, duration: float) -> dict | None:
    """Read a current compatible sidecar without loading a model."""
    try:
        from opencut.core.semantic_video_search import (
            DEFAULT_ENGINE,
            SEARCH_ENGINES,
            load_sidecar_embeddings,
        )

        engine_info = SEARCH_ENGINES[DEFAULT_ENGINE]
        data = load_sidecar_embeddings(root_path, path, DEFAULT_ENGINE)
    except Exception as exc:  # noqa: BLE001 - optional visual dependency
        logger.debug("Could not read visual sidecar for %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        return None

    raw_timestamps = data.get("timestamps", data.get("frame_timestamps", []))
    timestamps = []
    if isinstance(raw_timestamps, (list, tuple)):
        for timestamp in raw_timestamps:
            if isinstance(timestamp, dict):
                start = _finite_float(timestamp.get("start", timestamp.get("timestamp")), 0.0)
                end = _finite_float(timestamp.get("end", start), start)
            else:
                start = _finite_float(timestamp, 0.0)
                end = start
            start = max(0.0, start)
            end = max(start, end)
            if duration > 0:
                start = min(start, duration)
                end = min(max(start, end), duration)
            timestamps.append({"start": round(start, 6), "end": round(end, 6)})

    thumbnail = _safe_thumbnail(data.get("thumbnail_path"), root_path)
    if not thumbnail:
        frame_paths = data.get("thumbnail_paths", data.get("frame_paths", []))
        if isinstance(frame_paths, (list, tuple)):
            for frame_path in frame_paths:
                thumbnail = _safe_thumbnail(frame_path, root_path)
                if thumbnail:
                    break
    return {
        "engine": DEFAULT_ENGINE,
        "schema_version": int(engine_info.get("schema_version", 1)),
        "timestamps": timestamps,
        "thumbnail_path": thumbnail,
        "capability": {
            "network_required": False,
            "model_loaded": False,
            "sidecar_available": True,
            "thumbnail_available": bool(thumbnail),
            "timestamps_available": bool(timestamps),
        },
    }


def _modality_row(conn: sqlite3.Connection, media_id: int, modality: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM modalities WHERE media_id = ? AND modality = ?",
        (media_id, modality),
    ).fetchone()


def _capability(row: sqlite3.Row | None) -> dict:
    if row is None:
        return {
            "state": "unindexed",
            "network_required": False,
            "timestamps_available": False,
            "thumbnail_available": False,
        }
    capability = _safe_json(row["capability_json"], {})
    if not isinstance(capability, dict):
        capability = {}
    capability.setdefault("network_required", False)
    capability["state"] = row["state"]
    if row["engine"]:
        capability.setdefault("engine", row["engine"])
    if row["schema_version"]:
        capability.setdefault("schema_version", int(row["schema_version"]))
    capability.setdefault("timestamps_available", bool(row["timestamps_json"] not in ("", "[]")))
    capability.setdefault("thumbnail_available", bool(row["thumbnail_path"]))
    return capability


def _write_modality(
    conn: sqlite3.Connection,
    media_id: int,
    modality: str,
    *,
    state: str,
    text: str = "",
    timestamps: list[dict] | None = None,
    engine: str = "",
    schema_version: int = 0,
    thumbnail_path: str = "",
    capability: dict | None = None,
) -> None:
    if state not in MODALITY_STATES:
        state = "error"
    timestamps = timestamps if isinstance(timestamps, list) else []
    capability = dict(capability or {})
    capability.setdefault("network_required", False)
    capability["state"] = state
    now = time.time()
    conn.execute(
        """
        INSERT INTO modalities(
            media_id, modality, engine, schema_version, state, text,
            timestamps_json, thumbnail_path, capability_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(media_id, modality) DO UPDATE SET
            engine = excluded.engine,
            schema_version = excluded.schema_version,
            state = excluded.state,
            text = excluded.text,
            timestamps_json = excluded.timestamps_json,
            thumbnail_path = excluded.thumbnail_path,
            capability_json = excluded.capability_json,
            updated_at = excluded.updated_at
        """,
        (
            media_id,
            modality,
            str(engine or ""),
            int(schema_version or 0),
            state,
            str(text or ""),
            json.dumps(timestamps, ensure_ascii=False),
            str(thumbnail_path or ""),
            json.dumps(capability, ensure_ascii=False, sort_keys=True),
            now,
        ),
    )


def _refresh_fts(conn: sqlite3.Connection, media_id: int) -> None:
    rows = conn.execute(
        "SELECT text FROM modalities WHERE media_id = ? AND modality IN ('text','ocr','audio')",
        (media_id,),
    ).fetchall()
    content = " ".join(str(row["text"] or "") for row in rows).strip()
    conn.execute("DELETE FROM media_fts WHERE media_id = ?", (str(media_id),))
    if content:
        conn.execute("INSERT INTO media_fts(media_id, content) VALUES (?, ?)", (str(media_id), content))


def _mark_modalities_stale(conn: sqlite3.Connection, media_id: int) -> None:
    rows = conn.execute("SELECT * FROM modalities WHERE media_id = ?", (media_id,)).fetchall()
    for row in rows:
        capability = _capability(row)
        capability["state"] = "stale"
        _write_modality(
            conn,
            media_id,
            row["modality"],
            state="stale",
            text=row["text"] or "",
            timestamps=_safe_json(row["timestamps_json"], []),
            engine=row["engine"] or "",
            schema_version=row["schema_version"] or 0,
            thumbnail_path=row["thumbnail_path"] or "",
            capability=capability,
        )


def _sync_legacy_modalities(
    conn: sqlite3.Connection,
    media_id: int,
    path: str,
    duration: float,
    mtime: float,
    legacy: dict,
    *,
    content_changed: bool,
) -> None:
    db_row = legacy.get("db") if isinstance(legacy, dict) else None
    json_row = legacy.get("json") if isinstance(legacy, dict) else None
    db_fresh = _mtime_matches(db_row, mtime)
    json_fresh = _mtime_matches(json_row, mtime)

    transcript = ""
    timestamps = []
    source = ""
    if db_fresh and db_row:
        transcript = str(db_row.get("transcript") or "").strip()
        source = "legacy_sqlite"
    if json_fresh and json_row:
        json_text = str(json_row.get("full_text") or "").strip()
        if json_text and not transcript:
            transcript = json_text
        timestamps = _normalise_segments(json_row.get("segments"), duration, transcript)
        source = source or "legacy_json"
    if db_fresh and db_row and not timestamps:
        timestamps = _normalise_segments([], duration, transcript)

    previous = {
        modality: _modality_row(conn, media_id, modality)
        for modality in ("text", "ocr", "audio")
    }
    if transcript:
        _write_modality(
            conn,
            media_id,
            "text",
            state="available",
            text=transcript,
            timestamps=timestamps,
            capability={
                "network_required": False,
                "source": source or "legacy",
                "timestamps_available": bool(timestamps),
            },
        )
    else:
        old = previous["text"]
        state = "stale" if (content_changed or (old and old["state"] == "stale")) else "unindexed"
        _write_modality(
            conn,
            media_id,
            "text",
            state=state,
            text=(old["text"] if old and state == "stale" else ""),
            timestamps=_safe_json(old["timestamps_json"], []) if old and state == "stale" else [],
            engine=old["engine"] if old else "",
            schema_version=old["schema_version"] if old else 0,
            capability={"network_required": False, "source": "legacy"},
        )

    for modality, key in (("ocr", "ocr_text"), ("audio", "audio_tags")):
        value = str(db_row.get(key) or "").strip() if db_fresh and db_row else ""
        old = previous[modality]
        if value:
            _write_modality(
                conn,
                media_id,
                modality,
                state="available",
                text=value,
                capability={"network_required": False, "source": "legacy_sqlite"},
            )
        else:
            state = "stale" if (content_changed or (old and old["state"] == "stale")) else "unindexed"
            _write_modality(
                conn,
                media_id,
                modality,
                state=state,
                text=(old["text"] if old and state == "stale" else ""),
                timestamps=_safe_json(old["timestamps_json"], []) if old and state == "stale" else [],
                engine=old["engine"] if old else "",
                schema_version=old["schema_version"] if old else 0,
                capability={"network_required": False, "source": "legacy_sqlite"},
            )


def _sync_visual_modality(
    conn: sqlite3.Connection,
    media_id: int,
    root_path: str,
    path: str,
    duration: float,
    *,
    content_changed: bool,
) -> None:
    old = _modality_row(conn, media_id, "visual")
    visual = _visual_metadata(root_path, path, duration)
    if visual:
        _write_modality(
            conn,
            media_id,
            "visual",
            state="available",
            engine=visual["engine"],
            schema_version=visual["schema_version"],
            timestamps=visual["timestamps"],
            thumbnail_path=visual["thumbnail_path"],
            capability=visual["capability"],
        )
        return

    try:
        from opencut.core.semantic_video_search import DEFAULT_ENGINE, SEARCH_ENGINES

        current_engine = DEFAULT_ENGINE
        current_schema = int(SEARCH_ENGINES[DEFAULT_ENGINE].get("schema_version", 1))
    except Exception:  # pragma: no cover - registry is a required module
        current_engine, current_schema = "clip-vit-b32", 1
    incompatible = bool(
        old
        and old["engine"]
        and (old["engine"] != current_engine or int(old["schema_version"] or 0) != current_schema)
    )
    if incompatible:
        state = "schema_incompatible"
    elif content_changed or (old and old["state"] == "stale"):
        state = "stale"
    else:
        state = "unindexed"
    _write_modality(
        conn,
        media_id,
        "visual",
        state=state,
        engine=old["engine"] if old and incompatible else current_engine,
        schema_version=old["schema_version"] if old and incompatible else current_schema,
        timestamps=_safe_json(old["timestamps_json"], []) if old and state in {"stale", "schema_incompatible"} else [],
        thumbnail_path=old["thumbnail_path"] if old and state in {"stale", "schema_incompatible"} else "",
        capability={
            "network_required": False,
            "sidecar_available": False,
            "model_loaded": False,
            "model_required": True,
            "reason": "visual sidecar is not available",
        },
    )


def _scan_one_root(
    conn: sqlite3.Connection,
    root: sqlite3.Row,
    *,
    max_files: int,
    max_bytes: int,
    on_progress: Callable[[int, str], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> dict:
    root_id = int(root["id"])
    root_path = root["path"]
    now = time.time()
    files_seen: set[str] = set()
    scanned = indexed = unchanged = relinked = bytes_scanned = 0
    errors = []
    complete = True
    cancelled = False

    try:
        from opencut.core.semantic_video_search import content_signature
    except Exception as exc:  # pragma: no cover - required dependency
        raise RuntimeError(f"content signature unavailable: {exc}") from exc
    try:
        from opencut.core import footage_search

        legacy_index = footage_search.load_index()
    except Exception:
        legacy_index = {}

    for current_dir, dirs, filenames in os.walk(root_path, topdown=True, followlinks=False):
        dirs[:] = sorted(
            directory
            for directory in dirs
            if directory not in _SKIP_DIRS and not directory.startswith(".")
        )
        for filename in sorted(filenames):
            if is_cancelled and is_cancelled():
                cancelled = True
                complete = False
                break
            path = os.path.join(current_dir, filename)
            if os.path.splitext(filename)[1].lower() not in MEDIA_EXTENSIONS:
                continue
            try:
                stat = os.stat(path)
                if not os.path.isfile(path):
                    continue
                if scanned >= max_files or bytes_scanned + stat.st_size > max_bytes:
                    complete = False
                    break
                relative_path = os.path.relpath(path, root_path)
                signature = content_signature(path)
                media_type = _media_type(path)
                mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
                duration = _video_duration(path, media_type)
                files_seen.add(relative_path)
                scanned += 1
                bytes_scanned += int(stat.st_size)
                current = conn.execute(
                    "SELECT * FROM media WHERE root_id = ? AND relative_path = ?",
                    (root_id, relative_path),
                ).fetchone()
                content_changed = bool(current and current["source_signature"] != signature)
                media_id = None
                if current is None:
                    moved = conn.execute(
                        """
                        SELECT * FROM media
                        WHERE root_id = ? AND source_signature = ? AND relative_path != ?
                          AND status IN ('missing', 'active')
                        ORDER BY CASE status WHEN 'missing' THEN 0 ELSE 1 END, id
                        LIMIT 1
                        """,
                        (root_id, signature, relative_path),
                    ).fetchone()
                    if moved is not None:
                        old_path = os.path.join(root_path, moved["relative_path"])
                        if moved["status"] == "missing" or not os.path.isfile(old_path):
                            conn.execute(
                                """
                                UPDATE media SET relative_path = ?, file_size = ?,
                                    file_mtime_ns = ?, duration = ?, media_type = ?,
                                    status = 'active', last_seen_at = ?, missing_since = NULL,
                                    last_error = ''
                                WHERE id = ?
                                """,
                                (
                                    relative_path,
                                    int(stat.st_size),
                                    mtime_ns,
                                    duration,
                                    media_type,
                                    now,
                                    int(moved["id"]),
                                ),
                            )
                            media_id = int(moved["id"])
                            relinked += 1
                if media_id is None and current is None:
                    cursor = conn.execute(
                        """
                        INSERT INTO media(
                            root_id, relative_path, source_signature, file_size,
                            file_mtime_ns, duration, media_type, status,
                            first_seen_at, last_seen_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                        """,
                        (
                            root_id,
                            relative_path,
                            signature,
                            int(stat.st_size),
                            mtime_ns,
                            duration,
                            media_type,
                            now,
                            now,
                        ),
                    )
                    media_id = int(cursor.lastrowid)
                    indexed += 1
                elif media_id is None:
                    media_id = int(current["id"])
                    if content_changed:
                        _mark_modalities_stale(conn, media_id)
                        indexed += 1
                    else:
                        unchanged += 1
                    conn.execute(
                        """
                        UPDATE media SET source_signature = ?, file_size = ?,
                            file_mtime_ns = ?, duration = ?, media_type = ?,
                            status = 'active', last_seen_at = ?, missing_since = NULL,
                            last_error = ''
                        WHERE id = ?
                        """,
                        (
                            signature,
                            int(stat.st_size),
                            mtime_ns,
                            duration,
                            media_type,
                            now,
                            media_id,
                        ),
                    )

                legacy = {
                    "db": None,
                    "json": _legacy_entry(legacy_index, path),
                }
                try:
                    from opencut.core.footage_index_db import get_indexed_file

                    legacy["db"] = get_indexed_file(path)
                except Exception:
                    pass
                _sync_legacy_modalities(
                    conn,
                    media_id,
                    path,
                    duration,
                    stat.st_mtime,
                    legacy,
                    content_changed=content_changed,
                )
                _sync_visual_modality(
                    conn,
                    media_id,
                    root_path,
                    path,
                    duration,
                    content_changed=content_changed,
                )
                _refresh_fts(conn, media_id)
                if on_progress:
                    try:
                        on_progress(min(99, int((scanned / max_files) * 100)), relative_path)
                    except Exception:
                        pass
            except Exception as exc:  # noqa: BLE001 - one bad media must not abort a scan
                relative_path = os.path.relpath(path, root_path)
                errors.append({"relative_path": relative_path, "error": str(exc)[:500]})
                logger.warning("Federated scan failed for %s: %s", path, exc)
        if cancelled or not complete:
            break

    if complete and not cancelled:
        placeholders = ",".join("?" for _ in files_seen) or "''"
        params: list[Any] = [now, root_id]
        # Reconcile only active rows that were not seen. A complete walk is
        # required before this update so a resource cap can never create
        # false missing records.
        if files_seen:
            params.extend(sorted(files_seen))
            conn.execute(
                f"""
                UPDATE media SET status = 'missing',
                    missing_since = COALESCE(missing_since, ?)
                WHERE root_id = ? AND status = 'active'
                  AND relative_path NOT IN ({placeholders})
                """,
                params,
            )
        else:
            conn.execute(
                """
                UPDATE media SET status = 'missing',
                    missing_since = COALESCE(missing_since, ?)
                WHERE root_id = ? AND status = 'active'
                """,
                (now, root_id),
            )

    conn.execute(
        "UPDATE roots SET last_scan_at = ?, last_scan_complete = ? WHERE id = ?",
        (now, int(complete and not cancelled), root_id),
    )
    return {
        "root_id": root_id,
        "scanned": scanned,
        "indexed": indexed,
        "unchanged": unchanged,
        "relinked": relinked,
        "bytes_scanned": bytes_scanned,
        "complete": bool(complete and not cancelled),
        "cancelled": cancelled,
        "errors": errors,
    }


def _prune_conn(
    conn: sqlite3.Connection,
    *,
    root_ids: list[int] | None,
    retention_days: int | None,
    dry_run: bool,
) -> dict:
    clauses = ["m.status = 'missing'"]
    params: list[Any] = []
    if root_ids:
        clauses.append("m.root_id IN (" + ",".join("?" for _ in root_ids) + ")")
        params.extend(root_ids)
    rows = conn.execute(
        f"""
        SELECT m.id, m.root_id, m.relative_path, m.missing_since,
               r.retention_days
        FROM media m JOIN roots r ON r.id = m.root_id
        WHERE {' AND '.join(clauses)}
        ORDER BY m.root_id, m.relative_path
        """,
        params,
    ).fetchall()
    now = time.time()
    candidates = []
    for row in rows:
        days = _bounded_int(
            retention_days if retention_days is not None else row["retention_days"],
            DEFAULT_RETENTION_DAYS,
            1,
            3650,
        )
        missing_since = _finite_float(row["missing_since"], now)
        if now - missing_since >= days * 86400:
            candidates.append(
                {
                    "media_id": int(row["id"]),
                    "root_id": int(row["root_id"]),
                    "relative_path": row["relative_path"],
                    "retention_days": days,
                }
            )
    if not dry_run:
        for candidate in candidates:
            conn.execute("DELETE FROM media_fts WHERE media_id = ?", (str(candidate["media_id"]),))
            conn.execute("DELETE FROM modalities WHERE media_id = ?", (candidate["media_id"],))
            conn.execute(
                "UPDATE media SET status = 'deleted', last_error = 'retention_expired' WHERE id = ?",
                (candidate["media_id"],),
            )
    return {
        "dry_run": bool(dry_run),
        "pruned": len(candidates),
        "entries": candidates,
        "retention_days": retention_days,
    }


def scan_roots(
    root_ids: Iterable[Any] | None = None,
    *,
    max_files: int | None = None,
    max_bytes: int | None = None,
    on_progress: Callable[[int, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Incrementally scan enabled roots under deterministic resource caps."""
    selected = _root_ids(root_ids)
    init_db()
    with _connect() as conn:
        where, params = _root_filter(selected, enabled_only=True)
        roots = conn.execute(f"SELECT * FROM roots r WHERE {where} ORDER BY r.id", params).fetchall()
        results = []
        for root in roots:
            root_max_files = _bounded_int(
                max_files if max_files is not None else root["max_files"],
                int(root["max_files"]),
                1,
                MAX_FILES_PER_SCAN,
            )
            root_max_bytes = _bounded_int(
                max_bytes if max_bytes is not None else root["max_bytes"],
                int(root["max_bytes"]),
                1,
                MAX_SCAN_BYTES,
            )
            results.append(
                _scan_one_root(
                    conn,
                    root,
                    max_files=root_max_files,
                    max_bytes=root_max_bytes,
                    on_progress=on_progress,
                    is_cancelled=is_cancelled,
                )
            )
            if results[-1]["cancelled"]:
                break
        prune = _prune_conn(conn, root_ids=[int(root["id"]) for root in roots], retention_days=None, dry_run=False)
        return {
            "manifest_version": MANIFEST_VERSION,
            "schema_version": SCHEMA_VERSION,
            "roots": results,
            "root_count": len(results),
            "complete": bool(results) and all(item["complete"] for item in results),
            "cancelled": any(item["cancelled"] for item in results),
            "pruned": int(prune["pruned"]),
            "limits": {
                "max_files": max_files if max_files is not None else MAX_FILES_PER_SCAN,
                "max_bytes": max_bytes if max_bytes is not None else MAX_SCAN_BYTES,
            },
        }


def _fts_query(query: str) -> str:
    words = [word.replace('"', '""') for word in query.split() if word.strip()]
    return " ".join(f'"{word}"' for word in words)


def _normalise_modalities(modalities: Iterable[Any] | None) -> list[str]:
    if modalities is None:
        return list(MODALITIES)
    if isinstance(modalities, str):
        modalities = [part.strip() for part in modalities.split(",")]
    result = []
    for modality in modalities:
        name = MODALITY_ALIASES.get(str(modality).strip().lower(), str(modality).strip().lower())
        if name not in MODALITIES:
            raise ValueError(f"unsupported modality: {modality}")
        if name not in result:
            result.append(name)
    return result or list(MODALITIES)


def _text_matches(query: str, text: str) -> bool:
    query = query.strip().casefold()
    text = str(text or "").casefold()
    if not query:
        return True
    return all(word in text for word in query.split())


def _result_record(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    query: str,
    modalities: list[str],
    include_paths: bool,
    score: float,
) -> dict | None:
    modality_rows = {
        modality: _modality_row(conn, int(row["id"]), modality)
        for modality in MODALITIES
    }
    matched = []
    timestamps = []
    thumbnail = ""
    for modality in modalities:
        item = modality_rows[modality]
        if item is None or item["state"] not in {"available", "stale"}:
            continue
        if query and modality != "visual" and not _text_matches(query, item["text"]):
            continue
        if query and modality == "visual":
            # Visual embeddings are intentionally not queried here.  A visual
            # sidecar can be listed by an empty query, while a text query
            # reports capability state instead of pretending a model exists.
            continue
        matched.append(modality)
        parsed = _safe_json(item["timestamps_json"], [])
        if isinstance(parsed, list):
            timestamps.extend(parsed)
        if modality == "visual" and item["thumbnail_path"] and include_paths:
            thumbnail = item["thumbnail_path"]
    if query and not matched:
        return None
    if not query and not matched:
        return None
    deduped_timestamps = []
    seen_timestamps = set()
    for timestamp in timestamps:
        if not isinstance(timestamp, dict):
            continue
        start = round(max(0.0, _finite_float(timestamp.get("start"), 0.0)), 6)
        end = round(max(start, _finite_float(timestamp.get("end"), start)), 6)
        key = (start, end, str(timestamp.get("text") or ""))
        if key not in seen_timestamps:
            seen_timestamps.add(key)
            deduped_timestamps.append({"start": start, "end": end, **({"text": str(timestamp.get("text"))} if timestamp.get("text") else {})})
    result = {
        "root_id": int(row["root_id"]),
        "root_label": row["root_label"] or "",
        "relative_path": row["relative_path"],
        "source_signature": row["source_signature"],
        "status": row["status"],
        "media_type": row["media_type"],
        "duration": float(row["duration"] or 0),
        "file_size": int(row["file_size"] or 0),
        "score": round(float(score), 6),
        "matched_modalities": matched,
        "timestamps": deduped_timestamps,
        "capabilities": {
            modality: _capability(modality_rows[modality]) for modality in modalities
        },
    }
    if include_paths:
        result["path"] = os.path.realpath(os.path.join(row["root_path"], row["relative_path"]))
        if thumbnail:
            result["thumbnail_path"] = thumbnail
    return result


def search(
    query: str = "",
    *,
    modalities: Iterable[Any] | None = None,
    root_ids: Iterable[Any] | None = None,
    limit: int = 50,
    include_paths: bool = False,
    include_stale: bool = False,
) -> dict:
    """Search federated local metadata with redacted, normalized results."""
    query = str(query or "").strip()
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"query too long (max {MAX_QUERY_LENGTH} characters)")
    limit = _bounded_int(limit, 50, 1, MAX_RESULTS)
    selected_modalities = _normalise_modalities(modalities)
    selected_roots = _root_ids(root_ids)
    init_db()
    with _connect() as conn:
        root_clause, params = _root_filter(selected_roots, enabled_only=True)
        status_clause = "m.status != 'deleted'" if include_stale else "m.status = 'active'"
        sql = f"""
            SELECT m.*, r.path AS root_path, r.label AS root_label
            FROM media m JOIN roots r ON r.id = m.root_id
            WHERE {root_clause} AND {status_clause}
        """
        rows = conn.execute(sql, params).fetchall()
        candidates = []
        fts_ids = set()
        if query:
            fts = _fts_query(query)
            if fts:
                try:
                    fts_rows = conn.execute(
                        "SELECT media_id, rank FROM media_fts WHERE media_fts MATCH ? ORDER BY rank LIMIT ?",
                        (fts, limit * 4),
                    ).fetchall()
                    fts_ids = {int(row["media_id"]) for row in fts_rows if str(row["media_id"]).isdigit()}
                    rank_by_id = {int(row["media_id"]): row["rank"] for row in fts_rows if str(row["media_id"]).isdigit()}
                except sqlite3.OperationalError:
                    rank_by_id = {}
            else:
                rank_by_id = {}
        else:
            rank_by_id = {}

        for row in rows:
            if query and fts_ids and int(row["id"]) not in fts_ids:
                continue
            score = 1.0
            if query and int(row["id"]) in rank_by_id:
                score = 1.0 / (1.0 + abs(_finite_float(rank_by_id[int(row["id"])], 0.0)))
            result = _result_record(
                conn,
                row,
                query=query,
                modalities=selected_modalities,
                include_paths=include_paths,
                score=score,
            )
            if result is not None:
                candidates.append(result)
        candidates.sort(key=lambda item: (-item["score"], item["root_id"], item["relative_path"]))
        return {
            "manifest_version": MANIFEST_VERSION,
            "schema_version": SCHEMA_VERSION,
            "query": query,
            "modalities": selected_modalities,
            "results": candidates[:limit],
            "count": min(len(candidates), limit),
            "include_paths": bool(include_paths),
            "network_required": False,
        }


def status(*, root_ids: Iterable[Any] | None = None, include_paths: bool = False) -> dict:
    """Return manifest, counts, and capability state without absolute paths."""
    selected = _root_ids(root_ids)
    init_db()
    with _connect() as conn:
        where, params = _root_filter(selected, enabled_only=False)
        roots = conn.execute(f"SELECT * FROM roots r WHERE {where} ORDER BY r.id", params).fetchall()
        media_where = ""
        media_params: list[Any] = []
        if selected:
            media_where = "WHERE root_id IN (" + ",".join("?" for _ in selected) + ")"
            media_params = selected
        media_counts = {
            row["status"]: int(row["count"])
            for row in conn.execute(
                f"SELECT status, COUNT(*) AS count FROM media {media_where} GROUP BY status",
                media_params,
            ).fetchall()
        }
        modality_where = ""
        modality_params: list[Any] = []
        if selected:
            modality_where = "WHERE media_id IN (SELECT id FROM media WHERE root_id IN (" + ",".join("?" for _ in selected) + "))"
            modality_params = selected
        modality_counts = {
            f"{row['modality']}:{row['state']}": int(row["count"])
            for row in conn.execute(
                f"SELECT modality, state, COUNT(*) AS count FROM modalities {modality_where} GROUP BY modality, state",
                modality_params,
            ).fetchall()
        }
        return {
            "manifest_version": MANIFEST_VERSION,
            "schema_version": SCHEMA_VERSION,
            "network_required": False,
            "limits": {
                "max_roots": MAX_ROOTS,
                "max_files_per_scan": MAX_FILES_PER_SCAN,
                "max_scan_bytes": MAX_SCAN_BYTES,
                "max_query_length": MAX_QUERY_LENGTH,
                "max_results": MAX_RESULTS,
            },
            "roots": [
                _root_record(conn, row, include_path=include_paths) for row in roots
            ],
            "media_counts": media_counts,
            "modality_counts": modality_counts,
            "capabilities": {
                modality: {
                    "state": "indexed_or_unindexed",
                    "network_required": False,
                    "model_loaded": False,
                }
                for modality in MODALITIES
            },
        }


def prune_missing(
    *,
    retention_days: int | None = None,
    root_ids: Iterable[Any] | None = None,
    dry_run: bool = False,
) -> dict:
    """Apply or preview retention for media missing from a complete scan."""
    selected = _root_ids(root_ids)
    if retention_days is not None:
        retention_days = _bounded_int(retention_days, DEFAULT_RETENTION_DAYS, 1, 3650)
    init_db()
    with _connect() as conn:
        result = _prune_conn(
            conn,
            root_ids=selected,
            retention_days=retention_days,
            dry_run=bool(dry_run),
        )
        result.update({"manifest_version": MANIFEST_VERSION, "schema_version": SCHEMA_VERSION})
        return result


# Names used by callers that describe the operation as indexing rather than scanning.
index_roots = scan_roots
