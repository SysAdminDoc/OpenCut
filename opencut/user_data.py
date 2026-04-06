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
_WHISPER_DEFAULTS = {"cpu_mode": False, "model": "turbo"}

def load_whisper_settings() -> dict:
    saved = read_user_file("whisper_settings.json", default={})
    result = dict(_WHISPER_DEFAULTS)
    if isinstance(saved, dict):
        result.update(saved)
    return result

def save_whisper_settings(settings: dict):
    write_user_file("whisper_settings.json", settings)


# ---------------------------------------------------------------------------
# LLM Settings
# ---------------------------------------------------------------------------

def load_llm_settings() -> dict:
    """Load LLM provider/model/key settings."""
    defaults = {
        "provider": "ollama",
        "model": "llama3",
        "api_key": "",
        "base_url": "http://localhost:11434",
        "max_tokens": 2000,
        "temperature": 0.3,
    }
    saved = read_user_file("llm_settings.json", default={})
    if isinstance(saved, dict):
        defaults.update(saved)
    return defaults


def save_llm_settings(settings: dict) -> None:
    """Save LLM settings."""
    write_user_file("llm_settings.json", settings)


# ---------------------------------------------------------------------------
# Footage Index Settings
# ---------------------------------------------------------------------------

def load_footage_index_config() -> dict:
    """Load footage search index configuration."""
    defaults = {
        "index_path": "",  # empty = default ~/.opencut/footage_index.json
        "auto_index_on_load": False,
        "whisper_model": "base",
        "max_index_size_mb": 500,
    }
    saved = read_user_file("footage_index_config.json", default={})
    if isinstance(saved, dict):
        defaults.update(saved)
    return defaults


def save_footage_index_config(config: dict) -> None:
    """Save footage index configuration."""
    write_user_file("footage_index_config.json", config)


# ---------------------------------------------------------------------------
# Loudness Target
# ---------------------------------------------------------------------------

def load_loudness_target() -> dict:
    """Load loudness normalization preferences."""
    defaults = {
        "target_lufs": -14.0,
        "true_peak": -1.0,
        "lra_max": 11.0,
    }
    saved = read_user_file("loudness_settings.json", default={})
    if isinstance(saved, dict):
        defaults.update(saved)
    return defaults


def save_loudness_target(settings: dict) -> None:
    """Save loudness settings."""
    write_user_file("loudness_settings.json", settings)


# ---------------------------------------------------------------------------
# Color Profile Presets
# ---------------------------------------------------------------------------

def load_color_profiles() -> list:
    """Load saved color matching reference profiles."""
    saved = read_user_file("color_profiles.json", default=[])
    return saved if isinstance(saved, list) else []


def save_color_profiles(profiles: list) -> None:
    """Save color matching profiles."""
    write_user_file("color_profiles.json", profiles)


# ---------------------------------------------------------------------------
# Multicam Defaults
# ---------------------------------------------------------------------------

def load_multicam_config() -> dict:
    """Load multicam auto-switching configuration."""
    defaults = {
        "min_cut_duration": 1.0,
        "gap_tolerance": 0.5,
        "default_speaker_count": 2,
    }
    saved = read_user_file("multicam_config.json", default={})
    if isinstance(saved, dict):
        defaults.update(saved)
    return defaults


def save_multicam_config(config: dict) -> None:
    """Save multicam configuration."""
    write_user_file("multicam_config.json", config)


# ---------------------------------------------------------------------------
# Auto Zoom Presets
# ---------------------------------------------------------------------------

def load_auto_zoom_presets() -> dict:
    """Load auto zoom preferences."""
    defaults = {
        "zoom_amount": 1.15,
        "easing": "ease_in_out",
        "sample_rate": 2.0,
        "face_padding": 0.2,
    }
    saved = read_user_file("auto_zoom_presets.json", default={})
    if isinstance(saved, dict):
        defaults.update(saved)
    return defaults


def save_auto_zoom_presets(presets: dict) -> None:
    """Save auto zoom preferences."""
    write_user_file("auto_zoom_presets.json", presets)


# ---------------------------------------------------------------------------
# Chapter Generation Defaults
# ---------------------------------------------------------------------------

def load_chapter_defaults() -> dict:
    """Load chapter generation defaults."""
    defaults = {
        "max_chapters": 15,
        "min_chapter_duration": 30.0,
        "naming_style": "descriptive",  # "descriptive" | "numbered" | "timecode"
    }
    saved = read_user_file("chapter_defaults.json", default={})
    if isinstance(saved, dict):
        defaults.update(saved)
    return defaults


def save_chapter_defaults(defaults: dict) -> None:
    """Save chapter generation defaults."""
    write_user_file("chapter_defaults.json", defaults)
