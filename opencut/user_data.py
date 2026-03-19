"""
OpenCut User Data

Thread-safe JSON file access for user settings files:
favorites, workflows, presets, whisper settings, job times.
"""

import json
import logging
import os
import tempfile
import threading

logger = logging.getLogger("opencut")

OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")

# Per-file locks to prevent concurrent read-modify-write corruption.
# Bounded: only OpenCut user-data files should be locked (< 20 files).
_MAX_FILE_LOCKS = 50
_file_locks = {}
_file_locks_guard = threading.Lock()


def _get_lock(filepath: str) -> threading.RLock:
    """Get or create a reentrant lock for a specific file path.

    Uses RLock so that nested calls (e.g. load_whisper_settings calling
    read_user_file) don't deadlock if they share the same file path.

    Normalizes the path so that different representations of the same
    file (e.g. forward-slash vs backslash, relative vs absolute) share
    a single lock on case-insensitive filesystems.
    """
    key = os.path.normcase(os.path.realpath(filepath))
    with _file_locks_guard:
        if key not in _file_locks:
            # Evict oldest entry if at capacity (safety bound)
            if len(_file_locks) >= _MAX_FILE_LOCKS:
                oldest = next(iter(_file_locks))
                del _file_locks[oldest]
            _file_locks[key] = threading.RLock()
        return _file_locks[key]


def read_user_file(filename: str, default=None):
    """
    Read a JSON file from the OpenCut user directory.
    Returns *default* (default: None) if the file doesn't exist or can't be parsed.
    """
    filepath = os.path.join(OPENCUT_DIR, filename)
    lock = _get_lock(filepath)
    with lock:
        try:
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("Could not read %s: %s", filename, e)
    return default


def write_user_file(filename: str, data):
    """
    Write JSON data to a file in the OpenCut user directory.
    Creates the directory if needed. Thread-safe per file.
    """
    filepath = os.path.join(OPENCUT_DIR, filename)
    lock = _get_lock(filepath)
    with lock:
        try:
            os.makedirs(OPENCUT_DIR, exist_ok=True)
            # Atomic write: write to temp file then rename to prevent corruption
            fd, tmp_path = tempfile.mkstemp(
                dir=OPENCUT_DIR, suffix=".tmp", prefix=filename + "."
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, filepath)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning("Could not write %s: %s", filename, e)
            raise


# ---------------------------------------------------------------------------
# Convenience wrappers for specific files
# ---------------------------------------------------------------------------

# Presets
def load_presets() -> dict:
    return read_user_file("user_presets.json", default={})

def save_presets(presets: dict):
    write_user_file("user_presets.json", presets)

# Favorites
def load_favorites() -> list:
    return read_user_file("favorites.json", default=[])

def save_favorites(favorites: list):
    write_user_file("favorites.json", favorites)

# Workflows
def load_workflows() -> list:
    return read_user_file("workflows.json", default=[])

def save_workflows(workflows: list):
    write_user_file("workflows.json", workflows)

# Whisper settings
_WHISPER_DEFAULTS = {"cpu_mode": False, "model": "base"}

def load_whisper_settings() -> dict:
    saved = read_user_file("whisper_settings.json", default={})
    result = dict(_WHISPER_DEFAULTS)
    if isinstance(saved, dict):
        result.update(saved)
    return result

def save_whisper_settings(settings: dict):
    write_user_file("whisper_settings.json", settings)
