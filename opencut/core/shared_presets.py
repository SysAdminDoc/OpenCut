"""
OpenCut Shared Presets Library

Team-shared preset library with import/export, ratings, duplicate
detection, and merge conflict resolution.

Preset categories: color_grades, audio_chains, export_profiles,
workflow_templates, caption_styles.

Presets are stored as ``.opencut-preset`` files (JSON with metadata
header).  The index is maintained in ``~/.opencut/shared_presets/index.json``.

Duplicate detection uses parameter hashing.  Merge conflicts can be
resolved with keep_local, keep_remote, or keep_newest strategies.
"""

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("opencut")

_OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
_PRESETS_DIR = os.path.join(_OPENCUT_DIR, "shared_presets")
_INDEX_FILE = os.path.join(_PRESETS_DIR, "index.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRESET_CATEGORIES = frozenset({
    "color_grades", "audio_chains", "export_profiles",
    "workflow_templates", "caption_styles",
})

MERGE_STRATEGIES = frozenset({"keep_local", "keep_remote", "keep_newest"})


def _param_hash(parameters: dict) -> str:
    """Compute a deterministic hash of preset parameters for duplicate detection."""
    serialized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


@dataclass
class SharedPreset:
    """A single shared preset definition."""

    id: str = ""
    name: str = ""
    category: str = "export_profiles"
    author: str = ""
    version: int = 1
    created_at: float = 0.0
    updated_at: float = 0.0
    parameters: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    rating: float = 0.0
    rating_count: int = 0
    param_hash: str = ""
    source: str = "local"

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = time.time()
        if not self.updated_at:
            self.updated_at = self.created_at
        if self.category not in PRESET_CATEGORIES:
            self.category = "export_profiles"
        if not self.param_hash and self.parameters:
            self.param_hash = _param_hash(self.parameters)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SharedPreset":
        known = {
            "id", "name", "category", "author", "version", "created_at",
            "updated_at", "parameters", "tags", "description", "rating",
            "rating_count", "param_hash", "source",
        }
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_preset_file(self) -> str:
        """Serialize to .opencut-preset format (JSON with metadata header)."""
        data = {
            "_opencut_preset_version": 1,
            "metadata": {
                "id": self.id,
                "name": self.name,
                "category": self.category,
                "author": self.author,
                "version": self.version,
                "created_at": self.created_at,
                "description": self.description,
                "tags": self.tags,
            },
            "parameters": self.parameters,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_preset_file(cls, content: str) -> "SharedPreset":
        """Deserialize from .opencut-preset file content."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid preset file: {exc}") from exc

        meta = data.get("metadata", {})
        params = data.get("parameters", {})

        return cls(
            id=meta.get("id", ""),
            name=meta.get("name", "Untitled"),
            category=meta.get("category", "export_profiles"),
            author=meta.get("author", ""),
            version=meta.get("version", 1),
            created_at=meta.get("created_at", 0),
            description=meta.get("description", ""),
            tags=meta.get("tags", []),
            parameters=params,
        )


# ---------------------------------------------------------------------------
# PresetLibrary — manages the collection
# ---------------------------------------------------------------------------

class PresetLibrary:
    """Manages a collection of shared presets with persistent index."""

    def __init__(self):
        self._lock = threading.Lock()
        self._presets: Dict[str, SharedPreset] = {}
        self._load_index()

    def _load_index(self):
        """Load the preset index from disk."""
        if not os.path.isfile(_INDEX_FILE):
            return
        try:
            with open(_INDEX_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for entry in data.get("presets", []):
                preset = SharedPreset.from_dict(entry)
                self._presets[preset.id] = preset
            logger.debug("Loaded %d shared presets from index",
                         len(self._presets))
        except Exception as exc:
            logger.warning("Failed to load preset index: %s", exc)

    def _save_index(self):
        """Persist the preset index to disk."""
        os.makedirs(_PRESETS_DIR, exist_ok=True)
        data = {
            "saved_at": time.time(),
            "count": len(self._presets),
            "presets": [p.to_dict() for p in self._presets.values()],
        }
        tmp = _INDEX_FILE + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, _INDEX_FILE)
        except Exception as exc:
            logger.warning("Failed to save preset index: %s", exc)
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    # -- CRUD ---------------------------------------------------------------

    def add(self, preset: SharedPreset) -> SharedPreset:
        """Add a preset to the library."""
        if not preset.name or not preset.name.strip():
            raise ValueError("Preset name is required")
        preset.param_hash = _param_hash(preset.parameters) if preset.parameters else ""

        with self._lock:
            self._presets[preset.id] = preset
            self._save_index()
        logger.info("Added shared preset: %s (%s)", preset.name, preset.id)
        return preset

    def get(self, preset_id: str) -> Optional[SharedPreset]:
        """Get a preset by ID."""
        return self._presets.get(preset_id)

    def update(self, preset_id: str, **kwargs) -> SharedPreset:
        """Update a preset's fields."""
        with self._lock:
            preset = self._presets.get(preset_id)
            if preset is None:
                raise ValueError(f"Preset not found: {preset_id}")

            for key, value in kwargs.items():
                if hasattr(preset, key) and key not in ("id", "created_at"):
                    setattr(preset, key, value)

            preset.updated_at = time.time()
            if "parameters" in kwargs:
                preset.param_hash = _param_hash(preset.parameters)
            self._save_index()
        return preset

    def delete(self, preset_id: str) -> bool:
        """Delete a preset. Returns True if found and deleted."""
        with self._lock:
            if preset_id not in self._presets:
                return False
            del self._presets[preset_id]
            self._save_index()
        return True

    def list_presets(self, category: Optional[str] = None,
                     author: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     search: Optional[str] = None,
                     sort_by: str = "name") -> List[SharedPreset]:
        """List presets with optional filtering and sorting."""
        results = list(self._presets.values())

        if category:
            results = [p for p in results if p.category == category]
        if author:
            author_lower = author.lower()
            results = [p for p in results if p.author.lower() == author_lower]
        if tags:
            tag_set = set(t.lower() for t in tags)
            results = [p for p in results
                       if tag_set.intersection(t.lower() for t in p.tags)]
        if search:
            search_lower = search.lower()
            results = [p for p in results
                       if search_lower in p.name.lower()
                       or search_lower in p.description.lower()]

        if sort_by == "rating":
            results.sort(key=lambda p: p.rating, reverse=True)
        elif sort_by == "created_at":
            results.sort(key=lambda p: p.created_at, reverse=True)
        elif sort_by == "updated_at":
            results.sort(key=lambda p: p.updated_at, reverse=True)
        elif sort_by == "author":
            results.sort(key=lambda p: p.author.lower())
        else:
            results.sort(key=lambda p: p.name.lower())

        return results

    # -- rating -------------------------------------------------------------

    def rate_preset(self, preset_id: str, rating: float) -> SharedPreset:
        """Rate a preset (1-5). Computes running average."""
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")

        with self._lock:
            preset = self._presets.get(preset_id)
            if preset is None:
                raise ValueError(f"Preset not found: {preset_id}")

            # Running average
            total = preset.rating * preset.rating_count + rating
            preset.rating_count += 1
            preset.rating = round(total / preset.rating_count, 2)
            preset.updated_at = time.time()
            self._save_index()
        return preset

    # -- duplicate detection ------------------------------------------------

    def find_duplicates(self, preset: SharedPreset) -> List[SharedPreset]:
        """Find presets with matching parameter hashes."""
        if not preset.parameters:
            return []
        target_hash = _param_hash(preset.parameters)
        return [
            p for p in self._presets.values()
            if p.param_hash == target_hash and p.id != preset.id
        ]

    def check_duplicate(self, parameters: dict) -> Optional[SharedPreset]:
        """Check if a preset with identical parameters already exists."""
        if not parameters:
            return None
        target_hash = _param_hash(parameters)
        for p in self._presets.values():
            if p.param_hash == target_hash:
                return p
        return None

    # -- import / export ----------------------------------------------------

    def export_preset(self, preset_id: str) -> str:
        """Export a single preset as .opencut-preset content."""
        preset = self._presets.get(preset_id)
        if preset is None:
            raise ValueError(f"Preset not found: {preset_id}")
        return preset.to_preset_file()

    def export_preset_to_file(self, preset_id: str, output_path: str) -> str:
        """Export a preset to a .opencut-preset file."""
        content = self.export_preset(preset_id)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return output_path

    def import_preset(self, content: str, source: str = "import") -> SharedPreset:
        """Import a preset from .opencut-preset file content."""
        preset = SharedPreset.from_preset_file(content)
        preset.source = source
        preset.param_hash = _param_hash(preset.parameters) if preset.parameters else ""

        # Generate a new ID to avoid collision
        preset.id = uuid.uuid4().hex[:12]
        return self.add(preset)

    def import_from_file(self, filepath: str) -> SharedPreset:
        """Import a preset from a .opencut-preset file on disk."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Preset file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as fh:
            content = fh.read()
        return self.import_preset(content, source=os.path.basename(filepath))

    def batch_import(self, directory: str) -> dict:
        """Import all .opencut-preset files from a directory.

        Returns dict with imported count, skipped count, and errors.
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        imported = 0
        skipped = 0
        errors = []

        for fname in os.listdir(directory):
            if not fname.endswith(".opencut-preset"):
                continue
            fpath = os.path.join(directory, fname)
            try:
                preset = self.import_from_file(fpath)
                # Check for duplicates
                dupes = self.find_duplicates(preset)
                if dupes:
                    skipped += 1
                    # Remove the just-added duplicate
                    self.delete(preset.id)
                else:
                    imported += 1
            except Exception as exc:
                errors.append({"file": fname, "error": str(exc)})

        return {
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "total_scanned": imported + skipped + len(errors),
        }

    # -- merge conflict resolution ------------------------------------------

    def merge_preset(self, local_id: str, remote_preset: SharedPreset,
                     strategy: str = "keep_newest") -> SharedPreset:
        """Resolve a merge conflict between local and remote presets.

        Strategies:
            keep_local: Keep the local version unchanged.
            keep_remote: Replace local with remote.
            keep_newest: Keep whichever has the newer updated_at timestamp.
        """
        if strategy not in MERGE_STRATEGIES:
            raise ValueError(f"Invalid merge strategy: {strategy}. "
                             f"Must be one of: {sorted(MERGE_STRATEGIES)}")

        with self._lock:
            local = self._presets.get(local_id)
            if local is None:
                raise ValueError(f"Local preset not found: {local_id}")

            if strategy == "keep_local":
                return local

            if strategy == "keep_remote":
                remote_preset.id = local_id
                remote_preset.updated_at = time.time()
                remote_preset.param_hash = (
                    _param_hash(remote_preset.parameters)
                    if remote_preset.parameters else ""
                )
                self._presets[local_id] = remote_preset
                self._save_index()
                return remote_preset

            # keep_newest
            if remote_preset.updated_at > local.updated_at:
                remote_preset.id = local_id
                remote_preset.param_hash = (
                    _param_hash(remote_preset.parameters)
                    if remote_preset.parameters else ""
                )
                self._presets[local_id] = remote_preset
                self._save_index()
                return remote_preset

            return local

    # -- stats --------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary statistics for the preset library."""
        presets = list(self._presets.values())
        by_category: Dict[str, int] = {}
        by_author: Dict[str, int] = {}
        total_ratings = 0

        for p in presets:
            by_category[p.category] = by_category.get(p.category, 0) + 1
            if p.author:
                by_author[p.author] = by_author.get(p.author, 0) + 1
            total_ratings += p.rating_count

        return {
            "total": len(presets),
            "by_category": by_category,
            "by_author": by_author,
            "total_ratings": total_ratings,
            "categories": sorted(by_category.keys()),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_library: Optional[PresetLibrary] = None
_library_lock = threading.Lock()


def _get_library() -> PresetLibrary:
    """Get the singleton PresetLibrary instance."""
    global _library
    if _library is None:
        with _library_lock:
            if _library is None:
                _library = PresetLibrary()
    return _library


def reset_library():
    """Reset the singleton (for testing)."""
    global _library
    with _library_lock:
        _library = None


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def add_preset(name: str, category: str, parameters: dict,
               author: str = "", tags: Optional[List[str]] = None,
               description: str = "") -> dict:
    """Add a new shared preset. Returns preset dict."""
    preset = SharedPreset(
        name=name,
        category=category,
        parameters=parameters,
        author=author,
        tags=tags or [],
        description=description,
    )
    result = _get_library().add(preset)
    return result.to_dict()


def list_presets(**kwargs) -> List[dict]:
    """List shared presets with optional filters."""
    presets = _get_library().list_presets(**kwargs)
    return [p.to_dict() for p in presets]


def get_preset(preset_id: str) -> Optional[dict]:
    """Get a single preset by ID."""
    preset = _get_library().get(preset_id)
    return preset.to_dict() if preset else None


def delete_preset(preset_id: str) -> bool:
    """Delete a preset."""
    return _get_library().delete(preset_id)


def rate_preset(preset_id: str, rating: float) -> dict:
    """Rate a preset (1-5). Returns updated preset dict."""
    preset = _get_library().rate_preset(preset_id, rating)
    return preset.to_dict()


def export_preset(preset_id: str) -> str:
    """Export a preset as .opencut-preset content."""
    return _get_library().export_preset(preset_id)


def import_preset(content: str, source: str = "import") -> dict:
    """Import a preset from .opencut-preset content. Returns preset dict."""
    preset = _get_library().import_preset(content, source)
    return preset.to_dict()


def batch_import_presets(directory: str) -> dict:
    """Import all presets from a directory."""
    return _get_library().batch_import(directory)


def merge_preset(local_id: str, remote_content: str,
                 strategy: str = "keep_newest") -> dict:
    """Merge a remote preset with a local one."""
    remote = SharedPreset.from_preset_file(remote_content)
    result = _get_library().merge_preset(local_id, remote, strategy)
    return result.to_dict()


def get_library_stats() -> dict:
    """Get library statistics."""
    return _get_library().stats()
