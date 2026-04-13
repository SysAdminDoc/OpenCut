"""
Branching Edit Workflows.

Named branches for edit snapshots: create, switch, merge, and
visualize the branch graph for non-linear editing history.
"""

import copy
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

BRANCHES_DIR = os.path.join(OPENCUT_DIR, "branches")


@dataclass
class EditBranch:
    """A named branch holding a timeline snapshot."""
    name: str
    project_id: str
    snapshot: Dict[str, Any] = field(default_factory=dict)
    parent_branch: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    is_active: bool = False
    commit_count: int = 0


@dataclass
class MergeResult:
    """Result of merging two branches."""
    success: bool = True
    merged_snapshot: Dict[str, Any] = field(default_factory=dict)
    conflicts: List[Dict] = field(default_factory=list)
    auto_resolved: int = 0
    message: str = ""


def _project_path(project_id: str) -> str:
    path = os.path.join(BRANCHES_DIR, project_id)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, "branches.json")


def _load_branches(project_id: str) -> Dict[str, dict]:
    path = _project_path(project_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_branches(project_id: str, branches: Dict[str, dict]) -> None:
    path = _project_path(project_id)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(branches, fh, indent=2)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def create_branch(
    name: str,
    from_snapshot: Dict,
    project_id: str = "default",
    parent_branch: str = "",
    on_progress: Optional[Callable] = None,
) -> EditBranch:
    """Create a new named branch from a timeline snapshot.

    Args:
        name: Branch name (must be unique within project).
        from_snapshot: Timeline snapshot dict to branch from.
        project_id: Project identifier.
        parent_branch: Name of the parent branch.

    Returns:
        The created EditBranch.
    """
    if not name or not name.strip():
        raise ValueError("Branch name cannot be empty")
    name = name.strip()

    branches = _load_branches(project_id)
    if name in branches:
        raise ValueError(f"Branch '{name}' already exists")

    if on_progress:
        on_progress(30, f"Creating branch '{name}'")

    branch = EditBranch(
        name=name,
        project_id=project_id,
        snapshot=copy.deepcopy(from_snapshot),
        parent_branch=parent_branch,
        commit_count=1,
    )

    branches[name] = asdict(branch)
    _save_branches(project_id, branches)

    if on_progress:
        on_progress(100, "Branch created")

    logger.info("Created branch '%s' for project '%s'", name, project_id)
    return branch


def switch_branch(
    name: str,
    project_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> EditBranch:
    """Switch to a named branch (make it active).

    Args:
        name: Branch name to switch to.
        project_id: Project identifier.

    Returns:
        The activated EditBranch with its snapshot.
    """
    branches = _load_branches(project_id)
    if name not in branches:
        raise KeyError(f"Branch not found: '{name}'")

    if on_progress:
        on_progress(30, f"Switching to branch '{name}'")

    # Deactivate all branches, activate target
    for bname in branches:
        branches[bname]["is_active"] = (bname == name)

    branches[name]["updated_at"] = time.time()
    _save_branches(project_id, branches)

    data = branches[name]
    branch = EditBranch(**{k: v for k, v in data.items()})

    if on_progress:
        on_progress(100, f"Switched to '{name}'")

    logger.info("Switched to branch '%s'", name)
    return branch


def merge_branches(
    source: str,
    target: str,
    project_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> MergeResult:
    """Merge non-conflicting changes from source into target.

    Args:
        source: Source branch name (changes to merge from).
        target: Target branch name (branch to merge into).
        project_id: Project identifier.

    Returns:
        MergeResult with merged snapshot and any conflicts.
    """
    branches = _load_branches(project_id)
    if source not in branches:
        raise KeyError(f"Source branch not found: '{source}'")
    if target not in branches:
        raise KeyError(f"Target branch not found: '{target}'")

    if on_progress:
        on_progress(20, f"Merging '{source}' into '{target}'")

    src_snapshot = branches[source].get("snapshot", {})
    tgt_snapshot = branches[target].get("snapshot", {})

    src_clips = {c.get("id", c.get("clip_id", f"clip_{i}")): c
                 for i, c in enumerate(src_snapshot.get("clips", []))}
    tgt_clips = {c.get("id", c.get("clip_id", f"clip_{i}")): c
                 for i, c in enumerate(tgt_snapshot.get("clips", []))}

    merged_clips = copy.deepcopy(tgt_clips)
    conflicts = []
    auto_resolved = 0

    if on_progress:
        on_progress(50, "Resolving changes")

    # Add clips that exist in source but not in target
    for cid, clip in src_clips.items():
        if cid not in tgt_clips:
            merged_clips[cid] = copy.deepcopy(clip)
            auto_resolved += 1
        elif clip != tgt_clips[cid]:
            # Both branches modified the same clip -- conflict
            conflicts.append({
                "clip_id": cid,
                "source_version": clip,
                "target_version": tgt_clips[cid],
                "resolution": "kept_target",
            })

    if on_progress:
        on_progress(80, "Saving merged result")

    merged_snapshot = copy.deepcopy(tgt_snapshot)
    merged_snapshot["clips"] = list(merged_clips.values())

    branches[target]["snapshot"] = merged_snapshot
    branches[target]["updated_at"] = time.time()
    branches[target]["commit_count"] = branches[target].get("commit_count", 0) + 1
    _save_branches(project_id, branches)

    if on_progress:
        on_progress(100, "Merge complete")

    msg = f"Merged '{source}' into '{target}'"
    if conflicts:
        msg += f" with {len(conflicts)} conflict(s)"
    if auto_resolved:
        msg += f", {auto_resolved} auto-resolved"

    logger.info(msg)
    return MergeResult(
        success=len(conflicts) == 0,
        merged_snapshot=merged_snapshot,
        conflicts=conflicts,
        auto_resolved=auto_resolved,
        message=msg,
    )


def list_branches(
    project_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> List[EditBranch]:
    """List all branches for a project.

    Args:
        project_id: Project identifier.

    Returns:
        List of EditBranch (without full snapshots for efficiency).
    """
    branches = _load_branches(project_id)
    results = []
    for name, data in branches.items():
        results.append(EditBranch(
            name=data.get("name", name),
            project_id=data.get("project_id", project_id),
            parent_branch=data.get("parent_branch", ""),
            created_at=data.get("created_at", 0),
            updated_at=data.get("updated_at", 0),
            is_active=data.get("is_active", False),
            commit_count=data.get("commit_count", 0),
            snapshot={},  # omit snapshot for listing
        ))
    return results


def get_branch_graph(
    project_id: str = "default",
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Get the branch graph showing parent-child relationships.

    Args:
        project_id: Project identifier.

    Returns:
        Dict with 'nodes' and 'edges' for graph visualization.
    """
    branches = _load_branches(project_id)
    nodes = []
    edges = []

    for name, data in branches.items():
        nodes.append({
            "id": name,
            "label": name,
            "is_active": data.get("is_active", False),
            "commit_count": data.get("commit_count", 0),
            "created_at": data.get("created_at", 0),
        })
        parent = data.get("parent_branch", "")
        if parent and parent in branches:
            edges.append({
                "from": parent,
                "to": name,
            })

    return {
        "project_id": project_id,
        "nodes": nodes,
        "edges": edges,
        "branch_count": len(nodes),
    }
