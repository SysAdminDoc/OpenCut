"""
OpenCut Approval Workflow

Multi-stage approval pipeline for video projects.  Tracks a project
through:  draft -> internal_review -> client_review -> approved -> final.

Each stage transition is recorded with timestamp, actor, and notes.
Configurable required_approvers per stage, with auto-advance when all
approvers sign off.  Deadline tracking with overdue detection.

Persistent JSON storage in ``~/.opencut/approvals/``.
"""

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
_APPROVALS_DIR = os.path.join(_OPENCUT_DIR, "approvals")

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------
STAGES = ("draft", "internal_review", "client_review", "approved", "final")
STAGE_SET = frozenset(STAGES)
STAGE_ORDER = {stage: idx for idx, stage in enumerate(STAGES)}

# Actions
VALID_ACTIONS = frozenset({"approve", "reject", "request_changes"})


@dataclass
class StageTransition:
    """Record of a stage transition in the approval workflow."""
    from_stage: str = ""
    to_stage: str = ""
    action: str = ""
    actor: str = ""
    notes: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StageTransition":
        known = {"from_stage", "to_stage", "action", "actor", "notes", "timestamp"}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ApprovalWorkflow:
    """Tracks the approval state of a single project."""

    id: str = ""
    project_id: str = ""
    project_name: str = ""
    current_stage: str = "draft"
    created_at: float = 0.0
    updated_at: float = 0.0
    deadline: float = 0.0
    history: List[StageTransition] = field(default_factory=list)
    required_approvers: Dict[str, List[str]] = field(default_factory=dict)
    current_approvals: Dict[str, List[str]] = field(default_factory=dict)
    rejection_reason: str = ""
    change_requests: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = time.time()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> dict:
        d = asdict(self)
        d["history"] = [h if isinstance(h, dict) else h for h in d["history"]]
        d["is_overdue"] = self.is_overdue
        d["age_hours"] = self.age_hours
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ApprovalWorkflow":
        history_raw = d.pop("history", [])
        d.pop("is_overdue", None)
        d.pop("age_hours", None)
        known = {
            "id", "project_id", "project_name", "current_stage", "created_at",
            "updated_at", "deadline", "required_approvers", "current_approvals",
            "rejection_reason", "change_requests", "metadata",
        }
        filtered = {k: v for k, v in d.items() if k in known}
        wf = cls(**filtered)
        wf.history = [
            StageTransition.from_dict(h) if isinstance(h, dict) else h
            for h in history_raw
        ]
        return wf

    @property
    def is_overdue(self) -> bool:
        """Check if the workflow has passed its deadline."""
        if self.deadline <= 0:
            return False
        return time.time() > self.deadline and self.current_stage != "final"

    @property
    def age_hours(self) -> float:
        """Hours since workflow creation."""
        return round((time.time() - self.created_at) / 3600, 2)

    def _next_stage(self) -> Optional[str]:
        """Return the next stage in the pipeline, or None if at final."""
        idx = STAGE_ORDER.get(self.current_stage, -1)
        if idx < 0 or idx >= len(STAGES) - 1:
            return None
        return STAGES[idx + 1]

    def _prev_stage(self) -> Optional[str]:
        """Return the previous stage, or None if at draft."""
        idx = STAGE_ORDER.get(self.current_stage, -1)
        if idx <= 0:
            return None
        return STAGES[idx - 1]

    def _check_auto_advance(self) -> bool:
        """Check if all required approvers have approved the current stage."""
        required = self.required_approvers.get(self.current_stage, [])
        if not required:
            return False
        current = set(self.current_approvals.get(self.current_stage, []))
        return all(approver in current for approver in required)

    def approve(self, actor: str, notes: str = "") -> dict:
        """Record an approval for the current stage.

        Auto-advances to the next stage if all required approvers have signed off.
        Returns dict with action taken and new state.
        """
        if self.current_stage == "final":
            raise ValueError("Workflow is already in final stage")

        # Record approval
        stage_key = self.current_stage
        if stage_key not in self.current_approvals:
            self.current_approvals[stage_key] = []
        if actor not in self.current_approvals[stage_key]:
            self.current_approvals[stage_key].append(actor)

        self.rejection_reason = ""
        self.updated_at = time.time()

        result = {
            "action": "approve",
            "actor": actor,
            "stage": self.current_stage,
            "auto_advanced": False,
        }

        # Check if all required approvers have signed off
        if self._check_auto_advance():
            next_stage = self._next_stage()
            if next_stage:
                transition = StageTransition(
                    from_stage=self.current_stage,
                    to_stage=next_stage,
                    action="auto_advance",
                    actor=actor,
                    notes=notes or "All required approvers signed off",
                )
                self.history.append(transition)
                self.current_stage = next_stage
                result["auto_advanced"] = True
                result["new_stage"] = next_stage
        else:
            # Just an individual approval, no stage change
            transition = StageTransition(
                from_stage=self.current_stage,
                to_stage=self.current_stage,
                action="approve",
                actor=actor,
                notes=notes,
            )
            self.history.append(transition)

        result["current_stage"] = self.current_stage
        return result

    def reject(self, actor: str, reason: str = "") -> dict:
        """Reject the current stage, sending the project back to draft."""
        if self.current_stage == "draft":
            raise ValueError("Cannot reject a draft — it hasn't been submitted")

        prev_stage = "draft"
        transition = StageTransition(
            from_stage=self.current_stage,
            to_stage=prev_stage,
            action="reject",
            actor=actor,
            notes=reason,
        )
        self.history.append(transition)
        self.rejection_reason = reason
        old_stage = self.current_stage
        self.current_stage = prev_stage
        # Clear approvals for the rejected stage
        self.current_approvals.pop(old_stage, None)
        self.updated_at = time.time()

        return {
            "action": "reject",
            "actor": actor,
            "from_stage": old_stage,
            "current_stage": self.current_stage,
            "reason": reason,
        }

    def request_changes(self, actor: str, notes: str = "") -> dict:
        """Request changes on the current stage without rejecting entirely."""
        if self.current_stage == "draft":
            raise ValueError("Cannot request changes on a draft")

        transition = StageTransition(
            from_stage=self.current_stage,
            to_stage=self.current_stage,
            action="request_changes",
            actor=actor,
            notes=notes,
        )
        self.history.append(transition)
        if notes:
            self.change_requests.append(notes)
        self.updated_at = time.time()

        return {
            "action": "request_changes",
            "actor": actor,
            "stage": self.current_stage,
            "notes": notes,
        }

    def advance(self, actor: str, notes: str = "") -> dict:
        """Manually advance to the next stage (bypass approver requirement)."""
        next_stage = self._next_stage()
        if next_stage is None:
            raise ValueError("Already at final stage")

        transition = StageTransition(
            from_stage=self.current_stage,
            to_stage=next_stage,
            action="manual_advance",
            actor=actor,
            notes=notes,
        )
        self.history.append(transition)
        old_stage = self.current_stage
        self.current_stage = next_stage
        self.updated_at = time.time()

        return {
            "action": "advance",
            "actor": actor,
            "from_stage": old_stage,
            "current_stage": self.current_stage,
        }

    def get_blockers(self) -> List[str]:
        """Return list of pending approvers for the current stage."""
        required = self.required_approvers.get(self.current_stage, [])
        current = set(self.current_approvals.get(self.current_stage, []))
        return [a for a in required if a not in current]


# ---------------------------------------------------------------------------
# Workflow Manager — persistence and lookup
# ---------------------------------------------------------------------------

class WorkflowManager:
    """Manages multiple approval workflows with persistent storage."""

    def __init__(self):
        self._lock = threading.Lock()
        self._workflows: Dict[str, ApprovalWorkflow] = {}
        self._load_all()

    def _storage_path(self, workflow_id: str) -> str:
        return os.path.join(_APPROVALS_DIR, f"{workflow_id}.json")

    def _load_all(self):
        """Load all workflow files from disk."""
        if not os.path.isdir(_APPROVALS_DIR):
            return
        for fname in os.listdir(_APPROVALS_DIR):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(_APPROVALS_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                wf = ApprovalWorkflow.from_dict(data)
                self._workflows[wf.id] = wf
            except Exception as exc:
                logger.warning("Failed to load workflow %s: %s", fname, exc)
        logger.debug("Loaded %d approval workflows", len(self._workflows))

    def _save(self, wf: ApprovalWorkflow):
        """Persist a workflow to disk."""
        os.makedirs(_APPROVALS_DIR, exist_ok=True)
        path = self._storage_path(wf.id)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(wf.to_dict(), fh, indent=2)
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("Failed to save workflow %s: %s", wf.id, exc)
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def create(self, project_id: str, project_name: str = "",
               required_approvers: Optional[Dict[str, List[str]]] = None,
               deadline: float = 0.0,
               metadata: Optional[Dict] = None) -> ApprovalWorkflow:
        """Create a new approval workflow for a project."""
        if not project_id:
            raise ValueError("project_id is required")

        wf = ApprovalWorkflow(
            project_id=project_id,
            project_name=project_name or project_id,
            required_approvers=required_approvers or {},
            deadline=deadline,
            metadata=metadata or {},
        )

        with self._lock:
            self._workflows[wf.id] = wf
            self._save(wf)

        logger.info("Created approval workflow %s for project %s",
                     wf.id, project_id)
        return wf

    def get(self, workflow_id: str) -> Optional[ApprovalWorkflow]:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)

    def get_by_project(self, project_id: str) -> Optional[ApprovalWorkflow]:
        """Get the latest workflow for a project."""
        matches = [
            wf for wf in self._workflows.values()
            if wf.project_id == project_id
        ]
        if not matches:
            return None
        return max(matches, key=lambda w: w.created_at)

    def approve(self, workflow_id: str, actor: str, notes: str = "") -> dict:
        """Approve the current stage of a workflow."""
        with self._lock:
            wf = self._workflows.get(workflow_id)
            if wf is None:
                raise ValueError(f"Workflow not found: {workflow_id}")
            result = wf.approve(actor, notes)
            self._save(wf)
        return result

    def reject(self, workflow_id: str, actor: str, reason: str = "") -> dict:
        """Reject the current stage of a workflow."""
        with self._lock:
            wf = self._workflows.get(workflow_id)
            if wf is None:
                raise ValueError(f"Workflow not found: {workflow_id}")
            result = wf.reject(actor, reason)
            self._save(wf)
        return result

    def request_changes(self, workflow_id: str, actor: str,
                        notes: str = "") -> dict:
        """Request changes on the current stage."""
        with self._lock:
            wf = self._workflows.get(workflow_id)
            if wf is None:
                raise ValueError(f"Workflow not found: {workflow_id}")
            result = wf.request_changes(actor, notes)
            self._save(wf)
        return result

    def advance(self, workflow_id: str, actor: str, notes: str = "") -> dict:
        """Manually advance a workflow to the next stage."""
        with self._lock:
            wf = self._workflows.get(workflow_id)
            if wf is None:
                raise ValueError(f"Workflow not found: {workflow_id}")
            result = wf.advance(actor, notes)
            self._save(wf)
        return result

    def list_workflows(self, stage: Optional[str] = None,
                       project_id: Optional[str] = None,
                       include_overdue: bool = False) -> List[dict]:
        """List all workflows, optionally filtered."""
        results = []
        for wf in self._workflows.values():
            if stage and wf.current_stage != stage:
                continue
            if project_id and wf.project_id != project_id:
                continue
            if include_overdue and not wf.is_overdue:
                continue
            results.append(wf.to_dict())
        results.sort(key=lambda w: w.get("updated_at", 0), reverse=True)
        return results

    def dashboard(self) -> dict:
        """Return a status dashboard of all active workflows."""
        workflows = list(self._workflows.values())
        active = [w for w in workflows if w.current_stage != "final"]
        overdue = [w for w in active if w.is_overdue]

        by_stage: Dict[str, int] = {}
        for wf in active:
            by_stage[wf.current_stage] = by_stage.get(wf.current_stage, 0) + 1

        items = []
        for wf in active:
            items.append({
                "id": wf.id,
                "project_id": wf.project_id,
                "project_name": wf.project_name,
                "current_stage": wf.current_stage,
                "age_hours": wf.age_hours,
                "is_overdue": wf.is_overdue,
                "blockers": wf.get_blockers(),
            })
        items.sort(key=lambda x: x["age_hours"], reverse=True)

        return {
            "total_active": len(active),
            "total_completed": len(workflows) - len(active),
            "overdue_count": len(overdue),
            "by_stage": by_stage,
            "workflows": items,
        }

    def delete(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        with self._lock:
            wf = self._workflows.pop(workflow_id, None)
            if wf is None:
                return False
            path = self._storage_path(workflow_id)
            try:
                if os.path.isfile(path):
                    os.unlink(path)
            except OSError as exc:
                logger.warning("Failed to delete workflow file %s: %s", path, exc)
        return True


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: Optional[WorkflowManager] = None
_manager_lock = threading.Lock()


def _get_manager() -> WorkflowManager:
    """Get the singleton WorkflowManager instance."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = WorkflowManager()
    return _manager


def reset_manager():
    """Reset the singleton (for testing)."""
    global _manager
    with _manager_lock:
        _manager = None


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def create_workflow(project_id: str, **kwargs) -> dict:
    """Create a new approval workflow. Returns workflow dict."""
    mgr = _get_manager()
    wf = mgr.create(project_id, **kwargs)
    return wf.to_dict()


def get_status(workflow_id: Optional[str] = None,
               project_id: Optional[str] = None) -> dict:
    """Get workflow status by ID or project ID."""
    mgr = _get_manager()
    if workflow_id:
        wf = mgr.get(workflow_id)
    elif project_id:
        wf = mgr.get_by_project(project_id)
    else:
        return mgr.dashboard()

    if wf is None:
        raise ValueError("Workflow not found")
    return wf.to_dict()


def approve_workflow(workflow_id: str, actor: str, notes: str = "") -> dict:
    """Approve the current stage."""
    return _get_manager().approve(workflow_id, actor, notes)


def reject_workflow(workflow_id: str, actor: str, reason: str = "") -> dict:
    """Reject the current stage."""
    return _get_manager().reject(workflow_id, actor, reason)


def request_changes_workflow(workflow_id: str, actor: str,
                             notes: str = "") -> dict:
    """Request changes on the current stage."""
    return _get_manager().request_changes(workflow_id, actor, notes)


def advance_workflow(workflow_id: str, actor: str, notes: str = "") -> dict:
    """Manually advance to the next stage."""
    return _get_manager().advance(workflow_id, actor, notes)


def list_workflows(**kwargs) -> List[dict]:
    """List workflows with optional filters."""
    return _get_manager().list_workflows(**kwargs)


def get_dashboard() -> dict:
    """Get the approval dashboard."""
    return _get_manager().dashboard()
