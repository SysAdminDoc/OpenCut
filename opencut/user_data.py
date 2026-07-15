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
import time
import uuid

from opencut.credential_store import (
    load_and_migrate_secrets,
    persist_secret_changes,
    secret_id,
)

logger = logging.getLogger("opencut")

OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
USER_TOMBSTONES_FILE = "user_tombstones.json"
USER_TOMBSTONE_MAX_COUNT = 100
USER_TOMBSTONE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60

# Per-file locks to prevent concurrent read-modify-write corruption.
# Bounded: only OpenCut user-data files should be locked (< 20 files).
_file_locks = {}
_file_locks_guard = threading.Lock()


def _safe_user_filepath(filename: str) -> str:
    """Resolve a user-data filename and verify it stays within OPENCUT_DIR.

    Blocks path traversal (``../``), absolute paths, null bytes, and any
    filename that would resolve outside the OpenCut user directory.

    Raises ``ValueError`` on invalid input.
    """
    if not filename or not isinstance(filename, str):
        raise ValueError("Invalid user-data filename")
    if "\x00" in filename:
        raise ValueError("Null byte in filename")
    # Block absolute paths and traversal components
    if os.path.isabs(filename) or ".." in filename.replace("\\", "/").split("/"):
        raise ValueError("Invalid user-data filename")
    filepath = os.path.join(OPENCUT_DIR, filename)
    resolved = os.path.realpath(filepath)
    real_base = os.path.realpath(OPENCUT_DIR)
    # Case-insensitive prefix compare — ``realpath`` on Windows may return a
    # different case than the caller supplied for OPENCUT_DIR, which would
    # falsely reject legitimate filenames.
    cmp_resolved = os.path.normcase(resolved)
    cmp_base = os.path.normcase(real_base)
    if not (cmp_resolved == cmp_base or cmp_resolved.startswith(cmp_base + os.sep)):
        raise ValueError("Filename escapes user-data directory")
    return filepath


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


def _quarantine_corrupt_file(filepath: str) -> None:
    """Move unreadable JSON aside so future writes can recover cleanly."""
    if not os.path.isfile(filepath):
        return
    directory = os.path.dirname(filepath) or "."
    base = os.path.basename(filepath)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = os.path.join(directory, f"{base}.corrupt-{stamp}")
    counter = 1
    while os.path.exists(target):
        counter += 1
        target = os.path.join(directory, f"{base}.corrupt-{stamp}-{counter}")
    try:
        os.replace(filepath, target)
        logger.warning("Quarantined corrupt user-data file %s -> %s", filepath, target)
    except OSError as e:
        logger.warning("Could not quarantine corrupt file %s: %s", filepath, e)


def read_user_file(filename: str, default=None):
    """
    Read a JSON file from the OpenCut user directory.
    Returns *default* (default: None) if the file doesn't exist or can't be parsed.
    """
    try:
        filepath = _safe_user_filepath(filename)
    except ValueError:
        logger.warning("Rejected invalid user-data filename: %s", filename)
        return default
    lock = _get_lock(filepath)
    with lock:
        try:
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Could not parse %s: %s", filename, e)
            _quarantine_corrupt_file(filepath)
        except Exception as e:
            logger.warning("Could not read %s: %s", filename, e)
    return default


def write_user_file(filename: str, data):
    """
    Write JSON data to a file in the OpenCut user directory.
    Creates the directory if needed. Thread-safe per file.
    """
    filepath = _safe_user_filepath(filename)
    lock = _get_lock(filepath)
    with lock:
        try:
            # Ensure both the root and any nested subdir requested by the
            # caller exist before mkstemp tries to create the staging file.
            parent_dir = os.path.dirname(filepath) or OPENCUT_DIR
            os.makedirs(parent_dir, exist_ok=True)
            # mkstemp's ``prefix`` rejects path separators on Windows, so
            # base the prefix on the filename's leaf only. Stage the temp
            # file in the same directory as the final target to keep the
            # os.replace() atomic across filesystem partitions.
            tmp_prefix = os.path.basename(filename) + "."
            fd, tmp_path = tempfile.mkstemp(
                dir=parent_dir, suffix=".tmp", prefix=tmp_prefix
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
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


CONFIG_SCHEMAS: dict = {}
_MIGRATION_BACKUP_SUFFIX = ".migration-backup"


class ConfigSchemaVersionError(RuntimeError):
    """Raised when a JSON config uses a schema this runtime cannot open."""


def register_config_schema(
    filename: str,
    version: int,
    migrations: dict | None = None,
) -> None:
    """Register a config file's schema version and migration functions.

    Args:
        filename: The config filename (e.g. "whisper_settings.json").
        version: Current schema version (positive integer).
        migrations: Dict of ``{target_version: fn(data) -> data}`` that
            upgrade data from ``target_version - 1`` to ``target_version``.
    """
    _safe_user_filepath(filename)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Config schema version must be a positive integer")
    if migrations is not None and not isinstance(migrations, dict):
        raise TypeError("Config migrations must be a dict")
    CONFIG_SCHEMAS[filename] = {
        "version": version,
        "migrations": migrations or {},
    }


def _migration_backup_filename(filename: str) -> str:
    return f"{filename}{_MIGRATION_BACKUP_SUFFIX}"


def _delete_user_file(filename: str) -> None:
    filepath = _safe_user_filepath(filename)
    with _get_lock(filepath):
        try:
            os.unlink(filepath)
        except FileNotFoundError:
            pass


def _restore_migration_backup(filename: str, backup_filename: str, original: dict) -> bool:
    backup = read_user_file(backup_filename, default=None)
    restore_data = backup if isinstance(backup, dict) else original
    try:
        write_user_file(filename, restore_data)
    except Exception as exc:
        logger.warning(
            "Config migration restore failed for %s; error_type=%s; backup retained",
            filename,
            type(exc).__name__,
        )
        return False
    _delete_user_file(backup_filename)
    return True


def _recover_interrupted_migration(filename: str, default):
    """Restore a durable backup only when the primary is missing/corrupt."""
    missing = object()
    data = read_user_file(filename, default=missing)
    backup_filename = _migration_backup_filename(filename)
    backup = read_user_file(backup_filename, default=missing)
    if data is missing and isinstance(backup, dict):
        try:
            write_user_file(filename, backup)
            _delete_user_file(backup_filename)
            logger.warning(
                "Recovered interrupted config migration for %s from its backup",
                filename,
            )
            return backup
        except Exception as exc:
            logger.warning(
                "Config migration recovery failed for %s; error_type=%s; backup retained",
                filename,
                type(exc).__name__,
            )
            return default
    # When both files are valid, retain the backup until the caller determines
    # whether every target step committed. This is the normal crash window
    # between successful per-version promotions.
    return default if data is missing else data


def read_user_file_versioned(filename: str, default=None):
    """Read a config file and auto-migrate from older schema versions.

    If the file has no ``_schema_version`` field, it is treated as v0
    and all registered migrations are applied in order. Unknown keys
    are preserved (not stripped).
    """
    schema = CONFIG_SCHEMAS.get(filename)
    if schema is None:
        return read_user_file(filename, default=default)

    filepath = _safe_user_filepath(filename)
    with _get_lock(filepath):
        data = _recover_interrupted_migration(filename, default)
        if data is None or not isinstance(data, dict):
            return data

        current = data.get("_schema_version", 0)
        target = schema["version"]
        if isinstance(current, bool) or not isinstance(current, int) or current < 0:
            raise ConfigSchemaVersionError(
                f"{filename} has an invalid _schema_version; refusing migration"
            )
        if current > target:
            raise ConfigSchemaVersionError(
                f"{filename} schema {current} is newer than supported schema "
                f"{target}; refusing to downgrade or open an unknown schema"
            )
        if current == target:
            _delete_user_file(_migration_backup_filename(filename))
            return data

        backup_filename = _migration_backup_filename(filename)
        prior_backup = read_user_file(backup_filename, default=None)
        original = _json_snapshot(
            prior_backup if isinstance(prior_backup, dict) else data
        )
        if not isinstance(prior_backup, dict):
            try:
                write_user_file(backup_filename, original)
            except Exception as exc:
                logger.warning(
                    "Config migration backup failed for %s; error_type=%s; migration skipped",
                    filename,
                    type(exc).__name__,
                )
                return original

        working = _json_snapshot(data)
        migrations = schema.get("migrations") or {}
        for version in range(current + 1, target + 1):
            migration = migrations.get(version)
            try:
                candidate = _json_snapshot(working)
                if migration is not None:
                    candidate = migration(candidate)
                if not isinstance(candidate, dict):
                    raise TypeError("Config migration must return a JSON object")
                candidate["_schema_version"] = version
                write_user_file(filename, candidate)
                working = candidate
            except Exception as exc:
                restored = _restore_migration_backup(
                    filename, backup_filename, original
                )
                logger.warning(
                    "Config migration failed for %s v%d->v%d; "
                    "error_type=%s; original_restored=%s",
                    filename,
                    version - 1,
                    version,
                    type(exc).__name__,
                    restored,
                )
                return original

        _delete_user_file(backup_filename)
        return working


def _utc_iso(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _json_snapshot(value):
    return json.loads(json.dumps(value))


def _prune_user_tombstones(entries: list, *, now: float | None = None) -> list:
    now = time.time() if now is None else now
    cutoff = now - USER_TOMBSTONE_MAX_AGE_SECONDS
    kept = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        created_at = entry.get("created_at", 0)
        if not isinstance(created_at, (int, float)) or created_at < cutoff:
            continue
        kept.append(entry)
    kept.sort(key=lambda item: float(item.get("created_at", 0)))
    return kept[-USER_TOMBSTONE_MAX_COUNT:]


def create_user_tombstone(kind: str, key: str, value, *, source_file: str, action: str = "delete", metadata: dict | None = None) -> dict:
    """Persist a capped restorable snapshot before mutating user data."""
    if not kind or not isinstance(kind, str):
        raise ValueError("Tombstone kind is required")
    now = time.time()
    expires_at = now + USER_TOMBSTONE_MAX_AGE_SECONDS
    entry = {
        "id": f"{kind}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime(now))}-{uuid.uuid4().hex[:8]}",
        "kind": kind,
        "key": str(key or "default"),
        "action": action,
        "source_file": source_file,
        "value": _json_snapshot(value),
        "metadata": _json_snapshot(metadata or {}),
        "created_at": now,
        "created_at_iso": _utc_iso(now),
        "expires_at": expires_at,
        "expires_at_iso": _utc_iso(expires_at),
        "restore_route": "/settings/tombstones/restore",
    }
    entries = read_user_file(USER_TOMBSTONES_FILE, default=[])
    if not isinstance(entries, list):
        entries = []
    entries = _prune_user_tombstones(entries, now=now)
    entries.append(entry)
    entries = _prune_user_tombstones(entries, now=now)
    write_user_file(USER_TOMBSTONES_FILE, entries)
    return entry


def build_user_data_destructive_record(
    kind: str,
    key: str,
    value,
    *,
    source_file: str,
    route: str,
    action: str = "delete",
) -> dict:
    """Build non-mutating metadata for a user-data record delete/replace."""
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return {
        "id": str(key),
        "kind": str(kind),
        "key": str(key),
        "action": str(action),
        "path": _safe_user_filepath(source_file),
        "category": f"user-data-{kind}",
        "root": OPENCUT_DIR,
        "type": "record",
        "bytes": len(encoded),
        "route": str(route),
        "reversible": True,
        "restore_route": "/settings/tombstones/restore",
    }


def list_user_tombstones(kind: str | None = None) -> list[dict]:
    entries = read_user_file(USER_TOMBSTONES_FILE, default=[])
    if not isinstance(entries, list):
        entries = []
    pruned = _prune_user_tombstones(entries)
    if len(pruned) != len(entries):
        write_user_file(USER_TOMBSTONES_FILE, pruned)
    if kind:
        return [entry for entry in pruned if entry.get("kind") == kind]
    return pruned


def get_user_tombstone(tombstone_id: str) -> dict | None:
    for entry in list_user_tombstones():
        if entry.get("id") == tombstone_id:
            return entry
    return None


def mark_user_tombstone_restored(tombstone_id: str) -> dict | None:
    entries = list_user_tombstones()
    now = time.time()
    restored = None
    for entry in entries:
        if entry.get("id") == tombstone_id:
            entry["restored_at"] = now
            entry["restored_at_iso"] = _utc_iso(now)
            restored = entry
            break
    if restored is not None:
        write_user_file(USER_TOMBSTONES_FILE, entries)
    return restored


def summarize_user_tombstone(entry: dict, *, include_value: bool = False) -> dict:
    keys = (
        "id",
        "kind",
        "key",
        "action",
        "source_file",
        "metadata",
        "created_at",
        "created_at_iso",
        "expires_at",
        "expires_at_iso",
        "restored_at",
        "restored_at_iso",
        "restore_route",
    )
    summary = {key: entry[key] for key in keys if key in entry}
    if include_value:
        summary["value"] = entry.get("value")
    return summary


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
# Assistant dismissed suggestions (v1.10.2, feature O)
# Persisted per-sequence so "don't suggest chapters" sticks across reloads.
# ---------------------------------------------------------------------------
def load_assistant_dismissed(sequence_key: str = "default") -> list:
    all_data = read_user_file("assistant_dismissed.json", default={}) or {}
    if not isinstance(all_data, dict):
        return []
    ids = all_data.get(sequence_key) or []
    return [str(i) for i in ids if isinstance(i, (str, int))]


def save_assistant_dismissed(sequence_key: str, dismissed_ids: list) -> None:
    all_data = read_user_file("assistant_dismissed.json", default={}) or {}
    if not isinstance(all_data, dict):
        all_data = {}
    # Dedupe + cap so the file doesn't grow unbounded
    cleaned: list = []
    for i in (dismissed_ids or []):
        s = str(i)
        if s and s not in cleaned:
            cleaned.append(s)
    all_data[sequence_key or "default"] = cleaned[:200]
    write_user_file("assistant_dismissed.json", all_data)


# ---------------------------------------------------------------------------
# LLM Settings
# ---------------------------------------------------------------------------


def _load_secret_backed_settings(
    filename: str,
    defaults: dict,
    *,
    secret_field: str,
    namespace: str,
) -> dict:
    saved = read_user_file(filename, default={})
    if not isinstance(saved, dict):
        saved = {}
    result = dict(defaults)
    result.update({key: value for key, value in saved.items() if key in defaults})
    identifier = secret_id(namespace)

    def persist_sanitized() -> None:
        sanitized = dict(saved)
        value = str(sanitized.pop(secret_field, "") or "")
        sanitized[f"{secret_field}_set"] = bool(value)
        sanitized["_credential_storage"] = "os_vault"
        write_user_file(filename, sanitized)

    secrets = load_and_migrate_secrets(
        {secret_field: identifier},
        saved,
        persist_sanitized,
    )
    result[secret_field] = secrets[secret_field]
    return result


def _save_secret_backed_settings(
    filename: str,
    settings: dict,
    *,
    secret_field: str,
    namespace: str,
) -> None:
    value = str(settings.get(secret_field) or "")
    previous = read_user_file(filename, default={})
    if not isinstance(previous, dict):
        previous = {}

    def persist_metadata(secure: bool) -> None:
        data = dict(settings)
        if secure:
            data.pop(secret_field, None)
            data[f"{secret_field}_set"] = bool(value)
            data["_credential_storage"] = "os_vault"
        else:
            data[secret_field] = value
            data[f"{secret_field}_set"] = bool(value)
            data["_credential_storage"] = "plaintext-opt-in"
        write_user_file(filename, data)

    changes = {}
    if value or previous.get(secret_field) or previous.get(f"{secret_field}_set"):
        changes[secret_id(namespace)] = value or None
    persist_secret_changes(changes, persist_metadata)

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
    return _load_secret_backed_settings(
        "llm_settings.json",
        defaults,
        secret_field="api_key",
        namespace="llm/api-key",
    )


def save_llm_settings(settings: dict) -> None:
    """Save LLM settings."""
    current = load_llm_settings()
    if isinstance(settings, dict):
        current.update(settings)
    _save_secret_backed_settings(
        "llm_settings.json",
        current,
        secret_field="api_key",
        namespace="llm/api-key",
    )


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


# ---------------------------------------------------------------------------
# Optional Telemetry Settings
# ---------------------------------------------------------------------------

_TELEMETRY_DEFAULTS = {
    "provider": "aptabase",
    "enabled": False,
    "app_key": "",
    "base_url": "",
    "include_diagnostics": False,
}


def load_telemetry_settings() -> dict:
    """Load opt-in telemetry settings.

    Telemetry is disabled by default. The app key is loaded from the OS
    credential vault and route responses must mask it before returning settings.
    """
    return _load_secret_backed_settings(
        "telemetry_settings.json",
        _TELEMETRY_DEFAULTS,
        secret_field="app_key",
        namespace="telemetry/app-key",
    )


def save_telemetry_settings(settings: dict) -> None:
    """Save opt-in telemetry settings."""
    current = load_telemetry_settings()
    if isinstance(settings, dict):
        current.update({k: settings.get(k, current[k]) for k in _TELEMETRY_DEFAULTS})
    current["provider"] = "aptabase"
    current["enabled"] = bool(current.get("enabled"))
    current["include_diagnostics"] = bool(current.get("include_diagnostics"))
    current["app_key"] = str(current.get("app_key") or "").strip()[:200]
    current["base_url"] = str(current.get("base_url") or "").strip().rstrip("/")[:500]
    _save_secret_backed_settings(
        "telemetry_settings.json",
        current,
        secret_field="app_key",
        namespace="telemetry/app-key",
    )


def load_local_only_setting() -> dict:
    """Load local-only privacy mode setting."""
    saved = read_user_file("local_only.json", default={})
    if not isinstance(saved, dict):
        return {"enabled": False}
    return {"enabled": bool(saved.get("enabled", False))}


def save_local_only_setting(enabled: bool) -> None:
    """Save local-only privacy mode setting."""
    write_user_file("local_only.json", {"enabled": bool(enabled)})
