"""Persistent content-addressed transcript cache."""

from __future__ import annotations

import hashlib
import json
import os
import string
import tempfile
import threading
import time
from importlib import metadata
from typing import Any, Dict, Optional, Tuple

from opencut.user_data import OPENCUT_DIR

CACHE_SCHEMA_VERSION = 1
CACHE_ENV = "OPENCUT_TRANSCRIPT_CACHE"
CACHE_DIR_ENV = "OPENCUT_TRANSCRIPT_CACHE_DIR"
DEFAULT_CHUNK_SIZE = 1024 * 1024

_RUNTIME_STATS = {
    "hits": 0,
    "misses": 0,
    "writes": 0,
    "corrupt_entries": 0,
}
_LOCK = threading.RLock()


def cache_enabled() -> bool:
    """Return whether the persistent transcript cache should be used."""
    value = os.environ.get(CACHE_ENV, "")
    return value.strip().lower() not in {"0", "false", "no", "off"}


def cache_dir() -> str:
    """Return the transcript cache directory."""
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return os.path.realpath(os.path.expanduser(override))
    return os.path.join(OPENCUT_DIR, "transcript_cache")


def _entry_path(key: str) -> str:
    if not _is_safe_key(key):
        raise ValueError("Invalid transcript cache key")
    return os.path.join(cache_dir(), f"{key}.json")


def cache_entry_path(key: str) -> str:
    """Return the on-disk path for a safe cache key."""
    return _entry_path(key)


def _is_safe_key(key: str) -> bool:
    return (
        isinstance(key, str)
        and len(key) == 64
        and all(ch in string.hexdigits for ch in key)
    )


def _json_dumps(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _backend_version(backend: str) -> str:
    package_names = {
        "faster-whisper": "faster-whisper",
        "openai-whisper": "openai-whisper",
        "whisperx": "whisperx",
    }
    package = package_names.get(backend)
    if not package:
        return "unknown"
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def source_digest(filepath: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Dict[str, Any]:
    """Return source file identity for content-addressed cache keys."""
    digest = hashlib.sha256()
    total = 0
    with open(filepath, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    try:
        stat = os.stat(filepath)
        mtime_ns = int(stat.st_mtime_ns)
    except OSError:
        mtime_ns = 0
    return {
        "hash_algorithm": "sha256",
        "source_sha256": digest.hexdigest(),
        "source_size_bytes": total,
        "source_mtime_ns": mtime_ns,
    }


def build_cache_key(
    filepath: str,
    *,
    backend: str,
    config: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build a stable cache key and metadata payload for a transcription."""
    source = source_digest(filepath)
    settings = {
        "model": getattr(config, "model", "base"),
        "language": getattr(config, "language", None) or "",
        "word_timestamps": bool(getattr(config, "word_timestamps", True)),
        "translate": bool(getattr(config, "translate", False)),
        "diarize": bool(getattr(config, "diarize", False)),
        "min_speakers": getattr(config, "min_speakers", None),
        "max_speakers": getattr(config, "max_speakers", None),
        "vad_filter": True,
    }
    metadata_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source": source,
        "backend": backend,
        "backend_version": _backend_version(backend),
        "settings": settings,
        "extra": extra or {},
    }
    key = hashlib.sha256(_json_dumps(metadata_payload).encode("utf-8")).hexdigest()
    metadata = {
        **metadata_payload,
        "key": key,
        "source_path_was": os.path.realpath(filepath),
    }
    return key, metadata


def load_transcript(key: str) -> Optional[Dict[str, Any]]:
    """Load a cached transcript entry, or None on miss/corruption."""
    if not cache_enabled():
        return None
    path = _entry_path(key)
    with _LOCK:
        if not os.path.isfile(path):
            _RUNTIME_STATS["misses"] += 1
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                entry = json.load(handle)
            if (
                not isinstance(entry, dict)
                or entry.get("schema_version") != CACHE_SCHEMA_VERSION
                or entry.get("key") != key
                or not isinstance(entry.get("result"), dict)
            ):
                raise ValueError("invalid transcript cache entry")
        except Exception:
            _RUNTIME_STATS["misses"] += 1
            _RUNTIME_STATS["corrupt_entries"] += 1
            _quarantine(path)
            return None
        _RUNTIME_STATS["hits"] += 1
        return entry


def store_transcript(
    key: str,
    metadata_payload: Dict[str, Any],
    result_payload: Dict[str, Any],
) -> str:
    """Atomically write a transcript cache entry."""
    if not cache_enabled():
        return ""
    path = _entry_path(key)
    entry = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "key": key,
        "created_ts": time.time(),
        "metadata": metadata_payload,
        "result": result_payload,
    }
    directory = os.path.dirname(path)
    with _LOCK:
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=directory,
            prefix=f"{key}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(entry, handle, indent=2, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        _RUNTIME_STATS["writes"] += 1
    return path


def cache_stats() -> Dict[str, Any]:
    """Return cache inventory and since-boot hit counters."""
    directory = cache_dir()
    entries = 0
    bytes_total = 0
    if os.path.isdir(directory):
        for name in os.listdir(directory):
            if not name.endswith(".json"):
                continue
            path = os.path.join(directory, name)
            try:
                if os.path.isfile(path):
                    entries += 1
                    bytes_total += os.path.getsize(path)
            except OSError:
                continue
    with _LOCK:
        runtime = dict(_RUNTIME_STATS)
    return {
        "enabled": cache_enabled(),
        "cache_dir": directory,
        "entries": entries,
        "bytes": bytes_total,
        **runtime,
    }


def clear_cache() -> Dict[str, Any]:
    """Delete all transcript cache entries."""
    directory = cache_dir()
    removed_entries = 0
    removed_bytes = 0
    with _LOCK:
        if os.path.isdir(directory):
            for name in os.listdir(directory):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(directory, name)
                try:
                    size = os.path.getsize(path)
                    os.unlink(path)
                    removed_entries += 1
                    removed_bytes += size
                except OSError:
                    continue
    return {
        "removed_entries": removed_entries,
        "removed_bytes": removed_bytes,
        "cache_dir": directory,
    }


def reset_runtime_stats() -> None:
    """Reset hit/miss counters. Intended for focused tests."""
    with _LOCK:
        for key in _RUNTIME_STATS:
            _RUNTIME_STATS[key] = 0


def _quarantine(path: str) -> None:
    try:
        target = f"{path}.corrupt-{int(time.time())}"
        os.replace(path, target)
    except OSError:
        try:
            os.unlink(path)
        except OSError:
            pass
