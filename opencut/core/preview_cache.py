"""
OpenCut Preview Cache — Intelligent Preview Render Cache Manager

LRU cache with configurable size limits for preview images.  Stores
preview files in ``~/.opencut/preview_cache/`` with metadata tracking
for hit/miss statistics, TTL-based expiry, and background cleanup.

Thread-safe via a single reentrant lock.  Supports invalidation by
source file, by effect name, or full flush.
"""

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Optional

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------
PREVIEW_CACHE_DIR = os.path.join(OPENCUT_DIR, "preview_cache")
CACHE_METADATA_FILE = os.path.join(PREVIEW_CACHE_DIR, "metadata.json")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class CacheEntry:
    """A single cached preview entry."""
    cache_key: str = ""
    file_path: str = ""
    source_path: str = ""
    source_mtime: float = 0.0
    effect_name: str = ""
    params_json: str = "{}"
    file_size: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0
    ttl: float = 3600.0  # seconds

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if this entry has expired based on TTL."""
        if now is None:
            now = time.time()
        return (now - self.created_at) > self.ttl

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CacheEntry":
        return CacheEntry(**{k: v for k, v in d.items()
                             if k in CacheEntry.__dataclass_fields__})


@dataclass
class CacheStats:
    """Cache statistics."""
    hit_count: int = 0
    miss_count: int = 0
    hit_ratio: float = 0.0
    total_size_mb: float = 0.0
    entry_count: int = 0
    max_size_mb: float = 0.0
    oldest_entry_age_s: float = 0.0
    newest_entry_age_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Cache Manager
# ---------------------------------------------------------------------------
class PreviewCacheManager:
    """Thread-safe LRU preview cache with disk persistence."""

    def __init__(self, cache_dir: str = PREVIEW_CACHE_DIR,
                 max_size_mb: float = 512.0,
                 default_ttl: float = 3600.0,
                 cleanup_interval: float = 300.0):
        self._cache_dir = cache_dir
        self._max_size_mb = max_size_mb
        self._max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._lock = threading.Lock()
        self._entries: Dict[str, CacheEntry] = {}
        self._hit_count = 0
        self._miss_count = 0
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        os.makedirs(self._cache_dir, exist_ok=True)
        self._load_metadata()
        self._start_cleanup()

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------
    @staticmethod
    def make_key(file_path: str, mtime: float, effect_name: str,
                 params_json: str) -> str:
        """Generate a cache key from source file + effect + params."""
        raw = json.dumps({
            "path": file_path,
            "mtime": mtime,
            "effect": effect_name,
            "params": params_json,
        }, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def get(self, key: str) -> Optional[str]:
        """Look up a cached preview by key.

        Returns the file path if found and valid, else None.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._miss_count += 1
                return None

            # Check expiry
            if entry.is_expired():
                self._remove_entry(key)
                self._miss_count += 1
                return None

            # Check file exists
            if not os.path.isfile(entry.file_path):
                self._remove_entry(key)
                self._miss_count += 1
                return None

            entry.last_accessed = time.time()
            entry.hit_count += 1
            self._hit_count += 1
            return entry.file_path

    def put(self, key: str, file_path: str, source_path: str = "",
            effect_name: str = "", params_json: str = "{}",
            ttl: Optional[float] = None) -> None:
        """Store a preview file in the cache."""
        if not os.path.isfile(file_path):
            logger.warning("Cannot cache non-existent file: %s", file_path)
            return

        ttl = ttl if ttl is not None else self._default_ttl

        # Copy file to cache directory if not already there
        cache_file = file_path
        if not file_path.startswith(self._cache_dir):
            ext = os.path.splitext(file_path)[1] or ".jpg"
            cache_file = os.path.join(self._cache_dir, f"{key}{ext}")
            try:
                shutil.copy2(file_path, cache_file)
            except OSError as e:
                logger.warning("Failed to copy to cache: %s", e)
                return

        file_size = 0
        try:
            file_size = os.path.getsize(cache_file)
        except OSError:
            pass

        mtime = 0.0
        if source_path:
            try:
                mtime = os.path.getmtime(source_path)
            except OSError:
                pass

        entry = CacheEntry(
            cache_key=key,
            file_path=cache_file,
            source_path=source_path,
            source_mtime=mtime,
            effect_name=effect_name,
            params_json=params_json,
            file_size=file_size,
            created_at=time.time(),
            last_accessed=time.time(),
            hit_count=0,
            ttl=ttl,
        )

        with self._lock:
            self._entries[key] = entry
            self._evict_lru()
            self._save_metadata_unlocked()

    def remove(self, key: str) -> bool:
        """Remove a specific entry by key."""
        with self._lock:
            return self._remove_entry(key)

    def _remove_entry(self, key: str) -> bool:
        """Remove entry (caller must hold lock)."""
        entry = self._entries.pop(key, None)
        if entry is None:
            return False
        try:
            if os.path.isfile(entry.file_path):
                os.unlink(entry.file_path)
        except OSError:
            pass
        return True

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------
    def invalidate_by_file(self, source_path: str) -> int:
        """Remove all cache entries for a specific source file."""
        count = 0
        with self._lock:
            keys_to_remove = [
                k for k, e in self._entries.items()
                if e.source_path == source_path
            ]
            for k in keys_to_remove:
                if self._remove_entry(k):
                    count += 1
            if count > 0:
                self._save_metadata_unlocked()
        return count

    def invalidate_by_effect(self, effect_name: str) -> int:
        """Remove all cache entries for a specific effect."""
        count = 0
        with self._lock:
            keys_to_remove = [
                k for k, e in self._entries.items()
                if e.effect_name == effect_name
            ]
            for k in keys_to_remove:
                if self._remove_entry(k):
                    count += 1
            if count > 0:
                self._save_metadata_unlocked()
        return count

    def flush(self) -> int:
        """Remove all cache entries and delete cache directory contents."""
        count = 0
        with self._lock:
            count = len(self._entries)
            for entry in self._entries.values():
                try:
                    if os.path.isfile(entry.file_path):
                        os.unlink(entry.file_path)
                except OSError:
                    pass
            self._entries.clear()
            self._hit_count = 0
            self._miss_count = 0
            self._save_metadata_unlocked()
        return count

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------
    def _evict_lru(self) -> None:
        """Evict least-recently-used entries until under size limit.
        Caller must hold lock.
        """
        total = sum(e.file_size for e in self._entries.values())
        while total > self._max_size_bytes and self._entries:
            # Find LRU entry
            lru_key = min(self._entries,
                          key=lambda k: self._entries[k].last_accessed)
            entry = self._entries.pop(lru_key)
            total -= entry.file_size
            try:
                if os.path.isfile(entry.file_path):
                    os.unlink(entry.file_path)
            except OSError:
                pass
            logger.debug("Evicted cache entry: %s", lru_key)

    # ------------------------------------------------------------------
    # TTL cleanup
    # ------------------------------------------------------------------
    def _cleanup_expired(self) -> int:
        """Remove expired entries.  Returns count removed."""
        count = 0
        now = time.time()
        with self._lock:
            expired = [k for k, e in self._entries.items()
                       if e.is_expired(now)]
            for k in expired:
                if self._remove_entry(k):
                    count += 1
            if count > 0:
                self._save_metadata_unlocked()
        return count

    def _start_cleanup(self) -> None:
        """Start a background daemon thread for periodic TTL cleanup."""
        if self._cleanup_thread is not None:
            return

        def _worker():
            while not self._shutdown.wait(timeout=self._cleanup_interval):
                try:
                    removed = self._cleanup_expired()
                    if removed > 0:
                        logger.debug("Cache cleanup: removed %d expired entries", removed)
                except Exception as e:
                    logger.debug("Cache cleanup error: %s", e)

        self._cleanup_thread = threading.Thread(
            target=_worker, daemon=True, name="opencut-preview-cache-cleanup",
        )
        self._cleanup_thread.start()

    def shutdown(self) -> None:
        """Signal the cleanup thread to stop."""
        self._shutdown.set()

    # ------------------------------------------------------------------
    # Cache warming
    # ------------------------------------------------------------------
    def warm_cache(
        self,
        video_path: str,
        effect_name: str,
        params: Optional[dict] = None,
        num_frames: int = 10,
        on_progress: Optional[Callable] = None,
    ) -> int:
        """Pre-generate and cache previews for a video at regular intervals.

        Returns the number of frames cached.
        """
        from opencut.core.live_preview import generate_live_preview  # noqa: F811

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        from opencut.helpers import get_video_info  # noqa: F811
        info = get_video_info(video_path)
        duration = info.get("duration", 0)
        if duration <= 0:
            duration = 10.0

        params = params or {}
        num_frames = max(1, min(num_frames, 100))
        step = duration / num_frames
        timestamps = [i * step for i in range(num_frames)]

        cached = 0
        total = len(timestamps)
        for idx, ts in enumerate(timestamps):
            try:
                result = generate_live_preview(
                    video_path=video_path,
                    effect=effect_name,
                    params=params,
                    timestamp=ts,
                    use_cache=False,
                    output_dir=self._cache_dir,
                )
                if result.preview_path and os.path.isfile(result.preview_path):
                    mtime = 0.0
                    try:
                        mtime = os.path.getmtime(video_path)
                    except OSError:
                        pass
                    key = self.make_key(video_path, mtime, effect_name,
                                        json.dumps(params, sort_keys=True))
                    # Append timestamp to key for uniqueness
                    key = hashlib.sha256(
                        f"{key}:{ts}".encode()
                    ).hexdigest()[:24]
                    self.put(key, result.preview_path,
                             source_path=video_path,
                             effect_name=effect_name,
                             params_json=json.dumps(params, sort_keys=True))
                    cached += 1
            except Exception as e:
                logger.warning("Cache warm failed at ts=%.2f: %s", ts, e)

            if on_progress:
                pct = int((idx + 1) / total * 100)
                on_progress(pct)

        return cached

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def stats(self) -> CacheStats:
        """Return current cache statistics."""
        with self._lock:
            total_size = sum(e.file_size for e in self._entries.values())
            total_requests = self._hit_count + self._miss_count
            hit_ratio = (self._hit_count / total_requests
                         if total_requests > 0 else 0.0)

            now = time.time()
            ages = [(now - e.created_at) for e in self._entries.values()]
            oldest = max(ages) if ages else 0.0
            newest = min(ages) if ages else 0.0

            return CacheStats(
                hit_count=self._hit_count,
                miss_count=self._miss_count,
                hit_ratio=round(hit_ratio, 4),
                total_size_mb=round(total_size / (1024 * 1024), 2),
                entry_count=len(self._entries),
                max_size_mb=self._max_size_mb,
                oldest_entry_age_s=round(oldest, 1),
                newest_entry_age_s=round(newest, 1),
            )

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------
    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        if not os.path.isfile(CACHE_METADATA_FILE):
            return
        try:
            with open(CACHE_METADATA_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            entries = data.get("entries", {})
            for k, v in entries.items():
                entry = CacheEntry.from_dict(v)
                # Verify file still exists
                if os.path.isfile(entry.file_path):
                    self._entries[k] = entry
            self._hit_count = data.get("hit_count", 0)
            self._miss_count = data.get("miss_count", 0)
            logger.debug("Loaded %d cache entries from metadata",
                         len(self._entries))
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning("Failed to load cache metadata: %s", e)

    def _save_metadata_unlocked(self) -> None:
        """Persist cache metadata to disk.  Caller must hold lock."""
        data = {
            "entries": {k: e.to_dict() for k, e in self._entries.items()},
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
        }
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            with open(CACHE_METADATA_FILE, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError as e:
            logger.warning("Failed to save cache metadata: %s", e)

    def save_metadata(self) -> None:
        """Public method to persist metadata (acquires lock)."""
        with self._lock:
            self._save_metadata_unlocked()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_default_manager: Optional[PreviewCacheManager] = None
_manager_lock = threading.Lock()


def get_cache_manager(max_size_mb: float = 512.0,
                      default_ttl: float = 3600.0) -> PreviewCacheManager:
    """Return (or create) the default cache manager singleton."""
    global _default_manager
    with _manager_lock:
        if _default_manager is None:
            _default_manager = PreviewCacheManager(
                max_size_mb=max_size_mb,
                default_ttl=default_ttl,
            )
        return _default_manager


def reset_cache_manager() -> None:
    """Shut down and reset the singleton (for testing)."""
    global _default_manager
    with _manager_lock:
        if _default_manager is not None:
            _default_manager.shutdown()
        _default_manager = None


def cache_stats() -> dict:
    """Return cache stats as a dict."""
    mgr = get_cache_manager()
    return mgr.stats().to_dict()


def cache_flush() -> int:
    """Flush all cache entries."""
    mgr = get_cache_manager()
    return mgr.flush()


def cache_invalidate_file(source_path: str) -> int:
    """Invalidate cache entries for a source file."""
    mgr = get_cache_manager()
    return mgr.invalidate_by_file(source_path)


def cache_invalidate_effect(effect_name: str) -> int:
    """Invalidate cache entries for an effect."""
    mgr = get_cache_manager()
    return mgr.invalidate_by_effect(effect_name)
