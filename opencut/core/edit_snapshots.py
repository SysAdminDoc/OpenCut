"""
OpenCut Edit Decision Snapshots

Save named snapshots of editing state: job history, output files,
parameters.  Restore any snapshot.  Compare snapshots to see
differences.
"""

import copy
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

_SNAPSHOTS_DIR = os.path.join(OPENCUT_DIR, "snapshots")

_lock = threading.Lock()
# In-memory cache: project_id -> {snapshot_name -> SnapshotData}
_snapshot_cache: Dict[str, Dict[str, "SnapshotData"]] = {}


@dataclass
class SnapshotData:
    """A saved snapshot of editing state."""
    name: str
    project_id: str
    created: float = field(default_factory=time.time)
    job_history: List[Dict[str, Any]] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotData":
        return cls(
            name=data.get("name", ""),
            project_id=data.get("project_id", "default"),
            created=data.get("created", time.time()),
            job_history=data.get("job_history", []),
            output_files=data.get("output_files", []),
            parameters=data.get("parameters", {}),
            metadata=data.get("metadata", {}),
        )


def _project_dir(project_id: str) -> str:
    """Get the snapshots directory for a project."""
    # Sanitize project_id: only alphanumeric, dash, underscore
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_id)
    return os.path.join(_SNAPSHOTS_DIR, safe_id)


def _snapshot_path(project_id: str, name: str) -> str:
    """Get the file path for a snapshot."""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return os.path.join(_project_dir(project_id), f"{safe_name}.json")


def create_snapshot(
    name: str,
    project_state: Dict[str, Any],
    project_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Create a named snapshot of the current editing state.

    Args:
        name: Unique name for this snapshot.
        project_state: Dict with:
            - 'job_history': List of job dicts.
            - 'output_files': List of output file paths.
            - 'parameters': Dict of current editing parameters.
            - 'metadata': Optional extra metadata.
        project_id: Project identifier for scoping.
        on_progress: Optional progress callback.

    Returns:
        Dict with snapshot info.

    Raises:
        ValueError: If name is empty.
    """
    if on_progress:
        on_progress(10, "Creating snapshot...")

    if not name or not name.strip():
        raise ValueError("Snapshot name is required")
    name = name.strip()

    snapshot = SnapshotData(
        name=name,
        project_id=project_id,
        job_history=copy.deepcopy(project_state.get("job_history", [])),
        output_files=list(project_state.get("output_files", [])),
        parameters=copy.deepcopy(project_state.get("parameters", {})),
        metadata=copy.deepcopy(project_state.get("metadata", {})),
    )

    if on_progress:
        on_progress(40, "Saving snapshot to disk...")

    # Save to disk
    snap_dir = _project_dir(project_id)
    os.makedirs(snap_dir, exist_ok=True)
    snap_path = _snapshot_path(project_id, name)

    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(snapshot.to_dict(), f, indent=2)

    # Update cache
    with _lock:
        if project_id not in _snapshot_cache:
            _snapshot_cache[project_id] = {}
        _snapshot_cache[project_id][name] = snapshot

    if on_progress:
        on_progress(100, "Snapshot created")

    logger.info("Created snapshot '%s' for project '%s'", name, project_id)
    return {
        "name": name,
        "project_id": project_id,
        "created": snapshot.created,
        "job_count": len(snapshot.job_history),
        "output_count": len(snapshot.output_files),
        "param_count": len(snapshot.parameters),
    }


def restore_snapshot(
    name: str,
    project_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Restore a named snapshot, returning its state data.

    Args:
        name: Name of the snapshot to restore.
        project_id: Project identifier.
        on_progress: Optional progress callback.

    Returns:
        Dict with the full snapshot state data.

    Raises:
        FileNotFoundError: If the snapshot doesn't exist.
    """
    if on_progress:
        on_progress(10, "Loading snapshot...")

    snapshot = _load_snapshot(project_id, name)
    if snapshot is None:
        raise FileNotFoundError(f"Snapshot '{name}' not found for project '{project_id}'")

    if on_progress:
        on_progress(100, "Snapshot restored")

    logger.info("Restored snapshot '%s' for project '%s'", name, project_id)
    return {
        "name": snapshot.name,
        "project_id": snapshot.project_id,
        "created": snapshot.created,
        "job_history": snapshot.job_history,
        "output_files": snapshot.output_files,
        "parameters": snapshot.parameters,
        "metadata": snapshot.metadata,
    }


def list_snapshots(
    project_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """List all snapshots for a project.

    Args:
        project_id: Project identifier.
        on_progress: Optional progress callback.

    Returns:
        List of snapshot summary dicts, sorted by creation time.
    """
    if on_progress:
        on_progress(30, "Listing snapshots...")

    snap_dir = _project_dir(project_id)
    snapshots = []

    if os.path.isdir(snap_dir):
        for filename in os.listdir(snap_dir):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(snap_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                snap = SnapshotData.from_dict(data)
                snapshots.append({
                    "name": snap.name,
                    "project_id": snap.project_id,
                    "created": snap.created,
                    "job_count": len(snap.job_history),
                    "output_count": len(snap.output_files),
                    "param_count": len(snap.parameters),
                })
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to read snapshot %s: %s", filepath, e)
                continue

    # Sort by creation time
    snapshots.sort(key=lambda s: s["created"])

    if on_progress:
        on_progress(100, "Snapshot list ready")

    return snapshots


def compare_snapshots(
    name_a: str,
    name_b: str,
    project_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Compare two snapshots to find differences.

    Args:
        name_a: Name of the first snapshot.
        name_b: Name of the second snapshot.
        project_id: Project identifier.
        on_progress: Optional progress callback.

    Returns:
        Dict with comparison results: parameter diffs, file diffs,
        and job history diffs.

    Raises:
        FileNotFoundError: If either snapshot doesn't exist.
    """
    if on_progress:
        on_progress(10, "Loading snapshots for comparison...")

    snap_a = _load_snapshot(project_id, name_a)
    snap_b = _load_snapshot(project_id, name_b)

    if snap_a is None:
        raise FileNotFoundError(f"Snapshot '{name_a}' not found")
    if snap_b is None:
        raise FileNotFoundError(f"Snapshot '{name_b}' not found")

    if on_progress:
        on_progress(40, "Comparing snapshots...")

    # Compare parameters
    param_diffs = _compare_dicts(snap_a.parameters, snap_b.parameters)

    # Compare output files
    files_a = set(snap_a.output_files)
    files_b = set(snap_b.output_files)
    files_added = sorted(files_b - files_a)
    files_removed = sorted(files_a - files_b)
    files_common = sorted(files_a & files_b)

    # Compare job history
    jobs_a_count = len(snap_a.job_history)
    jobs_b_count = len(snap_b.job_history)

    # Find job types in each
    job_types_a = [j.get("type", "unknown") for j in snap_a.job_history]
    job_types_b = [j.get("type", "unknown") for j in snap_b.job_history]

    if on_progress:
        on_progress(100, "Comparison complete")

    return {
        "snapshot_a": name_a,
        "snapshot_b": name_b,
        "parameter_diffs": param_diffs,
        "files": {
            "added": files_added,
            "removed": files_removed,
            "common": files_common,
        },
        "jobs": {
            "count_a": jobs_a_count,
            "count_b": jobs_b_count,
            "types_a": job_types_a,
            "types_b": job_types_b,
        },
        "time_diff": snap_b.created - snap_a.created,
    }


def _load_snapshot(project_id: str, name: str) -> Optional[SnapshotData]:
    """Load a snapshot from cache or disk."""
    # Check cache
    with _lock:
        project_snaps = _snapshot_cache.get(project_id, {})
        if name in project_snaps:
            return project_snaps[name]

    # Load from disk
    snap_path = _snapshot_path(project_id, name)
    if not os.path.isfile(snap_path):
        return None

    try:
        with open(snap_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        snapshot = SnapshotData.from_dict(data)

        # Cache it
        with _lock:
            if project_id not in _snapshot_cache:
                _snapshot_cache[project_id] = {}
            _snapshot_cache[project_id][name] = snapshot

        return snapshot
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to load snapshot '%s': %s", name, e)
        return None


def _compare_dicts(
    dict_a: Dict[str, Any],
    dict_b: Dict[str, Any],
    prefix: str = "",
) -> List[Dict[str, Any]]:
    """Compare two dicts and return a list of differences."""
    diffs = []
    all_keys = sorted(set(list(dict_a.keys()) + list(dict_b.keys())))

    for key in all_keys:
        full_key = f"{prefix}.{key}" if prefix else key
        in_a = key in dict_a
        in_b = key in dict_b

        if in_a and not in_b:
            diffs.append({
                "key": full_key,
                "change": "removed",
                "old_value": dict_a[key],
                "new_value": None,
            })
        elif not in_a and in_b:
            diffs.append({
                "key": full_key,
                "change": "added",
                "old_value": None,
                "new_value": dict_b[key],
            })
        elif dict_a[key] != dict_b[key]:
            # Recurse into nested dicts
            if isinstance(dict_a[key], dict) and isinstance(dict_b[key], dict):
                diffs.extend(_compare_dicts(dict_a[key], dict_b[key], full_key))
            else:
                diffs.append({
                    "key": full_key,
                    "change": "modified",
                    "old_value": dict_a[key],
                    "new_value": dict_b[key],
                })

    return diffs
