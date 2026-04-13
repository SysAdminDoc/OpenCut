"""
OpenCut Watch Folder / Hot Folder

Monitor directories for new media files and trigger callbacks automatically.
Uses polling (os.path.getmtime) with no external dependencies.
"""

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR, _ensure_opencut_dir, get_video_info

logger = logging.getLogger("opencut")

# Supported media extensions
DEFAULT_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mxf", ".ts",
                      ".wav", ".mp3", ".flac", ".aac", ".m4a"}

_PROCESSED_PATH = os.path.join(OPENCUT_DIR, "watch_processed.json")
_CONFIGS_PATH = os.path.join(OPENCUT_DIR, "watch_folders.json")

_active_watches: Dict[str, "WatcherHandle"] = {}
_watches_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class WatchFolderConfig:
    """Configuration for a single watch folder."""
    folder_path: str
    workflow_name: str = ""
    output_dir: str = ""
    file_extensions: Optional[List[str]] = None
    poll_interval_sec: float = 5.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self):
        if self.file_extensions is None:
            self.file_extensions = sorted(DEFAULT_EXTENSIONS)


class WatcherHandle:
    """Handle returned by start_watch() -- call .stop() to terminate."""

    def __init__(self, config: WatchFolderConfig, thread: threading.Thread):
        self.config = config
        self._thread = thread
        self._stopped = threading.Event()

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def active(self) -> bool:
        return not self._stopped.is_set() and self._thread.is_alive()

    def stop(self):
        self._stopped.set()


# ---------------------------------------------------------------------------
# Processed file tracking
# ---------------------------------------------------------------------------
_processed_lock = threading.Lock()


def _load_processed() -> Dict[str, float]:
    """Load processed files dict {path: timestamp}."""
    try:
        with open(_PROCESSED_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_processed(processed: Dict[str, float]):
    """Persist processed files dict."""
    _ensure_opencut_dir()
    with open(_PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2)


def _mark_processed(filepath: str):
    """Mark a file as processed."""
    with _processed_lock:
        processed = _load_processed()
        processed[filepath] = time.time()
        _save_processed(processed)


def _is_processed(filepath: str) -> bool:
    """Check if a file has already been processed."""
    with _processed_lock:
        return filepath in _load_processed()


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------
def load_watch_configs() -> List[WatchFolderConfig]:
    """Load saved watch folder configurations from ~/.opencut/watch_folders.json."""
    try:
        with open(_CONFIGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        configs = []
        for item in data:
            configs.append(WatchFolderConfig(
                folder_path=item.get("folder_path", ""),
                workflow_name=item.get("workflow_name", ""),
                output_dir=item.get("output_dir", ""),
                file_extensions=item.get("file_extensions"),
                poll_interval_sec=item.get("poll_interval_sec", 5.0),
                id=item.get("id", uuid.uuid4().hex[:12]),
            ))
        return configs
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_watch_configs(configs: List[WatchFolderConfig]):
    """Persist watch folder configurations to ~/.opencut/watch_folders.json."""
    _ensure_opencut_dir()
    data = [asdict(c) for c in configs]
    with open(_CONFIGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Validate media file
# ---------------------------------------------------------------------------
def _validate_media(filepath: str) -> bool:
    """Quick validation that a file is a readable media file."""
    if not os.path.isfile(filepath):
        return False
    # Check minimum size (1KB) to skip partial writes
    try:
        if os.path.getsize(filepath) < 1024:
            return False
    except OSError:
        return False
    # Attempt ffprobe -- get_video_info returns defaults on failure,
    # but a duration of 0 on a video file means probe failed.
    try:
        info = get_video_info(filepath)
        return info.get("duration", 0) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------
def _poll_loop(handle: WatcherHandle, on_new_file: Optional[Callable]):
    """Background polling loop for a single watch folder."""
    config = handle.config
    extensions = set(config.file_extensions or DEFAULT_EXTENSIONS)
    seen_mtimes: Dict[str, float] = {}

    # Initial scan -- record existing files so we don't trigger on startup
    if os.path.isdir(config.folder_path):
        for name in os.listdir(config.folder_path):
            full = os.path.join(config.folder_path, name)
            if os.path.isfile(full):
                try:
                    seen_mtimes[full] = os.path.getmtime(full)
                except OSError:
                    pass

    logger.info("Watch folder started: %s (id=%s, interval=%.1fs)",
                config.folder_path, config.id, config.poll_interval_sec)

    while not handle._stopped.wait(timeout=config.poll_interval_sec):
        if not os.path.isdir(config.folder_path):
            logger.debug("Watch folder missing: %s", config.folder_path)
            continue

        try:
            entries = os.listdir(config.folder_path)
        except OSError as e:
            logger.debug("Cannot list watch folder %s: %s", config.folder_path, e)
            continue

        for name in entries:
            full = os.path.join(config.folder_path, name)
            if not os.path.isfile(full):
                continue

            ext = os.path.splitext(name)[1].lower()
            if ext not in extensions:
                continue

            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue

            prev_mtime = seen_mtimes.get(full)
            if prev_mtime is not None and mtime == prev_mtime:
                continue  # unchanged

            seen_mtimes[full] = mtime

            # Skip if already processed
            if _is_processed(full):
                continue

            # Wait briefly for file to finish writing
            time.sleep(0.5)
            try:
                new_mtime = os.path.getmtime(full)
                if new_mtime != mtime:
                    seen_mtimes[full] = new_mtime
                    continue  # still being written
            except OSError:
                continue

            # Validate media
            if not _validate_media(full):
                logger.debug("Watch folder: invalid media skipped: %s", full)
                continue

            logger.info("Watch folder: new file detected: %s", full)
            _mark_processed(full)

            if on_new_file:
                try:
                    on_new_file(full, config)
                except Exception as e:
                    logger.error("Watch folder callback error for %s: %s", full, e)

    logger.info("Watch folder stopped: %s (id=%s)", config.folder_path, config.id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def start_watch(config: WatchFolderConfig,
                on_new_file: Optional[Callable] = None) -> WatcherHandle:
    """Start monitoring a folder for new files.

    Args:
        config: Watch folder configuration.
        on_new_file: Callback ``(filepath, config)`` invoked for each new file.

    Returns:
        WatcherHandle with a ``.stop()`` method.
    """
    if not os.path.isdir(config.folder_path):
        raise ValueError(f"Folder does not exist: {config.folder_path}")

    thread = threading.Thread(
        target=lambda h: _poll_loop(h, on_new_file),
        args=(None,),  # placeholder, replaced below
        daemon=True,
        name=f"opencut-watch-{config.id}",
    )
    handle = WatcherHandle(config, thread)
    # Fix the lambda closure
    thread._target = lambda: _poll_loop(handle, on_new_file)
    thread._args = ()
    thread.start()

    with _watches_lock:
        _active_watches[config.id] = handle

    return handle


def stop_watch(handle: WatcherHandle) -> None:
    """Stop a running watcher."""
    handle.stop()
    with _watches_lock:
        _active_watches.pop(handle.id, None)


def list_active_watches() -> List[Dict]:
    """Return list of active watch configs as dicts."""
    with _watches_lock:
        result = []
        for wid, handle in list(_active_watches.items()):
            if handle.active:
                d = asdict(handle.config)
                d["active"] = True
                result.append(d)
            else:
                # Stale entry -- clean up
                _active_watches.pop(wid, None)
        return result


def stop_all_watches() -> int:
    """Stop all active watches. Returns count stopped."""
    with _watches_lock:
        count = 0
        for handle in _active_watches.values():
            handle.stop()
            count += 1
        _active_watches.clear()
        return count
