"""
Render Cache with Dependency Tracking.

Content-addressable cache for render outputs with automatic
invalidation when upstream parameters change.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

CACHE_DIR = os.path.join(OPENCUT_DIR, "render_cache")
CACHE_INDEX = os.path.join(CACHE_DIR, "index.json")


@dataclass
class CacheEntry:
    """A single entry in the render cache."""
    cache_key: str
    input_hash: str
    operation: str
    params_hash: str
    output_path: str
    file_size: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0
    dependencies: List[str] = field(default_factory=list)


def _compute_hash(*parts: Any) -> str:
    """Compute a content-addressable hash from parts."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _cache_key(input_hash: str, operation: str, params: dict) -> str:
    """Generate a unique cache key."""
    params_hash = _compute_hash(params)
    return _compute_hash(input_hash, operation, params_hash)


def _load_index() -> Dict[str, dict]:
    """Load the cache index from disk."""
    if os.path.exists(CACHE_INDEX):
        try:
            with open(CACHE_INDEX, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt cache index, starting fresh")
    return {}


def _save_index(index: Dict[str, dict]) -> None:
    """Persist the cache index to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_INDEX, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)


def _cache_root() -> str:
    return os.path.realpath(os.path.abspath(CACHE_DIR))


def _is_under_cache_dir(path: str) -> bool:
    try:
        resolved = os.path.realpath(os.path.abspath(path))
        root = _cache_root()
        common = os.path.commonpath([root, resolved])
        return os.path.normcase(common) == os.path.normcase(root)
    except (TypeError, ValueError, OSError):
        return False


def _is_expected_cache_file(cache_key: str, path: str) -> bool:
    if not cache_key or not path or not _is_under_cache_dir(path):
        return False
    basename = os.path.basename(os.path.realpath(path))
    stem, _ext = os.path.splitext(basename)
    return stem == cache_key


def _safe_unlink_cached_file(cache_key: str, path: str) -> tuple[int, bool]:
    """Delete one cache file only when the index path is confined and expected."""
    if not _is_expected_cache_file(cache_key, path):
        logger.warning("Refusing unsafe render-cache path for %s: %s", cache_key, path)
        return 0, False
    if not os.path.isfile(path):
        return 0, True
    size = os.path.getsize(path)
    os.unlink(path)
    return size, True


def _cache_plan_entry(cache_key: str, data: dict, *, category: str) -> dict:
    """Build non-mutating metadata for one indexed cache entry."""
    path = str(data.get("output_path", ""))
    safe_path = _is_expected_cache_file(cache_key, path)
    return {
        "id": cache_key,
        "path": path,
        "category": category,
        "root": CACHE_DIR,
        "type": "file",
        "bytes": int(data.get("file_size", 0) or 0),
        "exists": safe_path and os.path.isfile(path),
        "safe_path": safe_path,
        "input_hash": str(data.get("input_hash", "")),
        "operation": str(data.get("operation", "")),
        "last_accessed": data.get("last_accessed", 0),
        "reversible": False,
    }


def _downstream_cache_keys(index: Dict[str, dict], input_hash: str, operation: str) -> list[str]:
    """Return the seed cache key plus dependent keys in deterministic walk order."""
    seed_key = None
    for key, data in index.items():
        if data.get("input_hash") == input_hash and data.get("operation") == operation:
            seed_key = key
            break

    if seed_key is None:
        return []

    to_remove = {seed_key}
    ordered = [seed_key]
    queue = [seed_key]
    while queue:
        current = queue.pop(0)
        for key, data in index.items():
            if key not in to_remove and current in data.get("dependencies", []):
                to_remove.add(key)
                ordered.append(key)
                queue.append(key)
    return ordered


def _file_content_hash(filepath: str) -> str:
    """Compute a hash of file contents for input tracking."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    h = hashlib.sha256()
    with open(filepath, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:32]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def cache_result(
    input_hash: str,
    operation: str,
    params: dict,
    output_path: str,
    dependencies: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> CacheEntry:
    """Store a render result in the cache.

    Args:
        input_hash: Content hash of the input.
        operation: Operation name (e.g., 'encode', 'filter').
        params: Operation parameters dict.
        output_path: Path to the output file to cache.
        dependencies: List of upstream cache keys this depends on.

    Returns:
        CacheEntry for the stored item.
    """
    if not os.path.isfile(output_path):
        raise FileNotFoundError(f"Output file not found: {output_path}")

    if on_progress:
        on_progress(20, "Computing cache key")

    key = _cache_key(input_hash, operation, params)
    params_hash = _compute_hash(params)

    # Copy to cache storage
    os.makedirs(CACHE_DIR, exist_ok=True)
    ext = os.path.splitext(output_path)[1]
    cached_path = os.path.join(CACHE_DIR, f"{key}{ext}")

    if on_progress:
        on_progress(50, "Copying to cache")

    shutil.copy2(output_path, cached_path)
    file_size = os.path.getsize(cached_path)

    entry = CacheEntry(
        cache_key=key,
        input_hash=input_hash,
        operation=operation,
        params_hash=params_hash,
        output_path=cached_path,
        file_size=file_size,
        dependencies=dependencies or [],
    )

    if on_progress:
        on_progress(80, "Updating index")

    index = _load_index()
    index[key] = {
        "cache_key": key,
        "input_hash": input_hash,
        "operation": operation,
        "params_hash": params_hash,
        "output_path": cached_path,
        "file_size": file_size,
        "created_at": entry.created_at,
        "last_accessed": entry.last_accessed,
        "hit_count": 0,
        "dependencies": entry.dependencies,
    }
    _save_index(index)

    if on_progress:
        on_progress(100, "Cached successfully")

    logger.info("Cached result: %s (%s/%s, %d bytes)", key, operation, input_hash[:8], file_size)
    return entry


def get_cached(
    input_hash: str,
    operation: str,
    params: dict,
    on_progress: Optional[Callable] = None,
) -> Optional[CacheEntry]:
    """Look up a cached render result.

    Args:
        input_hash: Content hash of the input.
        operation: Operation name.
        params: Operation parameters.

    Returns:
        CacheEntry if found and valid, None otherwise.
    """
    key = _cache_key(input_hash, operation, params)
    index = _load_index()

    if key not in index:
        return None

    data = index[key]
    cached_path = data.get("output_path", "")

    if not _is_expected_cache_file(key, cached_path):
        logger.warning("Cache miss (unsafe index path): %s", key)
        del index[key]
        _save_index(index)
        return None

    # Validate the cached file still exists
    if not os.path.isfile(cached_path):
        logger.info("Cache miss (file deleted): %s", key)
        del index[key]
        _save_index(index)
        return None

    # Update access stats
    data["last_accessed"] = time.time()
    data["hit_count"] = data.get("hit_count", 0) + 1
    _save_index(index)

    entry = CacheEntry(
        cache_key=data["cache_key"],
        input_hash=data["input_hash"],
        operation=data["operation"],
        params_hash=data["params_hash"],
        output_path=data["output_path"],
        file_size=data.get("file_size", 0),
        created_at=data.get("created_at", 0),
        last_accessed=data["last_accessed"],
        hit_count=data["hit_count"],
        dependencies=data.get("dependencies", []),
    )

    logger.debug("Cache hit: %s (hits=%d)", key, entry.hit_count)
    return entry


def invalidate_downstream(
    input_hash: str,
    operation: str,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Invalidate all cache entries that depend on a given input+operation.

    Args:
        input_hash: Content hash of the changed input.
        operation: Operation that changed.

    Returns:
        Dict with 'invalidated' count and 'freed_bytes'.
    """
    if on_progress:
        on_progress(20, "Scanning dependencies")

    index = _load_index()
    to_remove = _downstream_cache_keys(index, input_hash, operation)
    if not to_remove:
        return {"invalidated": 0, "freed_bytes": 0}

    if on_progress:
        on_progress(50, "Tracing dependency chain")

    if on_progress:
        on_progress(70, f"Removing {len(to_remove)} entries")

    freed = 0
    invalid_entries = 0
    missing_entries = 0
    for key in to_remove:
        data = index.get(key, {})
        path = data.get("output_path", "")
        existed = _is_expected_cache_file(key, path) and os.path.isfile(path)
        size, safe = _safe_unlink_cached_file(key, path)
        if not safe:
            invalid_entries += 1
        elif not existed:
            missing_entries += 1
        freed += size
        del index[key]

    _save_index(index)

    if on_progress:
        on_progress(100, "Invalidation complete")

    logger.info("Invalidated %d cache entries, freed %d bytes", len(to_remove), freed)
    return {
        "invalidated": len(to_remove),
        "freed_bytes": freed,
        "invalid_entries": invalid_entries,
        "missing_entries": missing_entries,
    }


def get_cache_stats(
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Get cache statistics.

    Returns:
        Dict with entry count, total size, hit rates, etc.
    """
    index = _load_index()
    total_size = 0
    total_hits = 0
    operations: Dict[str, int] = {}
    oldest = float("inf")
    newest = 0.0

    for data in index.values():
        total_size += data.get("file_size", 0)
        total_hits += data.get("hit_count", 0)
        op = data.get("operation", "unknown")
        operations[op] = operations.get(op, 0) + 1
        created = data.get("created_at", 0)
        if created < oldest:
            oldest = created
        if created > newest:
            newest = created

    return {
        "entry_count": len(index),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "total_hits": total_hits,
        "operations": operations,
        "oldest_entry": oldest if oldest != float("inf") else None,
        "newest_entry": newest if newest > 0 else None,
    }


def cleanup_cache(
    max_size_gb: float = 5.0,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Clean up the cache to stay under the size limit.

    Uses LRU eviction: removes least-recently-accessed entries first.

    Args:
        max_size_gb: Maximum cache size in gigabytes.

    Returns:
        Dict with removed count and freed bytes.
    """
    max_bytes = int(max_size_gb * 1024 * 1024 * 1024)

    if on_progress:
        on_progress(20, "Calculating cache size")

    index = _load_index()
    total_size = sum(int(d.get("file_size", 0) or 0) for d in index.values())

    if total_size <= max_bytes:
        if on_progress:
            on_progress(100, "Cache within limits")
        return {"removed": 0, "freed_bytes": 0, "current_size": total_size}

    if on_progress:
        on_progress(40, "Evicting LRU entries")

    # Sort by last_accessed (oldest first)
    entries = sorted(index.items(), key=lambda kv: kv[1].get("last_accessed", 0))

    removed = 0
    freed = 0
    invalid_entries = 0
    missing_entries = 0
    for key, data in entries:
        if total_size <= max_bytes:
            break
        path = data.get("output_path", "")
        fsize = data.get("file_size", 0)
        existed = _is_expected_cache_file(key, path) and os.path.isfile(path)
        actual_size, safe = _safe_unlink_cached_file(key, path)
        if not safe:
            invalid_entries += 1
        elif not existed:
            missing_entries += 1
        del index[key]
        total_size -= fsize
        freed += actual_size
        removed += 1

    _save_index(index)

    if on_progress:
        on_progress(100, f"Removed {removed} entries")

    logger.info("Cache cleanup: removed %d entries, freed %.1f MB",
                removed, freed / (1024 * 1024))
    return {
        "removed": removed,
        "freed_bytes": freed,
        "current_size": total_size,
        "invalid_entries": invalid_entries,
        "missing_entries": missing_entries,
    }


def cleanup_cache_plan(max_size_gb: float = 5.0) -> Dict[str, Any]:
    """Preview LRU render-cache entries that cleanup would remove."""
    max_bytes = int(max_size_gb * 1024 * 1024 * 1024)
    index = _load_index()
    total_size = sum(int(d.get("file_size", 0) or 0) for d in index.values())
    current_size = total_size
    entries = []

    if current_size > max_bytes:
        for key, data in sorted(index.items(), key=lambda kv: kv[1].get("last_accessed", 0)):
            if current_size <= max_bytes:
                break
            entries.append(_cache_plan_entry(key, data, category="render-cache-cleanup"))
            current_size -= int(data.get("file_size", 0) or 0)

    estimated_bytes = sum(int(entry["bytes"]) for entry in entries)
    return {
        "entries": entries,
        "removed": len(entries),
        "estimated_freed_bytes": estimated_bytes,
        "current_size": current_size,
        "total_size": total_size,
        "max_size_bytes": max_bytes,
    }


def invalidate_downstream_plan(input_hash: str, operation: str) -> Dict[str, Any]:
    """Preview render-cache entries invalidation would remove."""
    index = _load_index()
    keys = _downstream_cache_keys(index, input_hash, operation)
    entries = [
        _cache_plan_entry(key, index.get(key, {}), category="render-cache-invalidate")
        for key in keys
    ]
    estimated_bytes = sum(int(entry["bytes"]) for entry in entries)
    invalid_entries = sum(1 for entry in entries if not entry["safe_path"])
    missing_entries = sum(1 for entry in entries if entry["safe_path"] and not entry["exists"])
    return {
        "entries": entries,
        "invalidated": len(entries),
        "estimated_freed_bytes": estimated_bytes,
        "invalid_entries": invalid_entries,
        "missing_entries": missing_entries,
        "input_hash": str(input_hash),
        "operation": str(operation),
    }
