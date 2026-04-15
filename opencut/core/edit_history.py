"""
OpenCut Edit History — Immutable Audit Log

Tracks all operations performed on a project as an immutable audit log.
Each entry records operation type, timestamp, parameters, input/output
files, duration, and user.

Supports undo marking, diff between history points, export as timeline
visualization data, and replay capability (generate API call sequence).

Storage: append-only JSONL files in ``~/.opencut/history/``, one per
project (keyed by project ID or path hash).

Statistics: most-used operations, average processing time per op type.
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
_HISTORY_DIR = os.path.join(_OPENCUT_DIR, "history")


@dataclass
class HistoryEntry:
    """A single immutable edit history entry."""

    id: str = ""
    operation_type: str = ""
    timestamp: float = 0.0
    parameters: Dict = field(default_factory=dict)
    input_file: str = ""
    output_file: str = ""
    duration_sec: float = 0.0
    user: str = ""
    undone: bool = False
    undone_at: float = 0.0
    session_id: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonl(self) -> str:
        """Serialize to a single JSONL line."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "HistoryEntry":
        known = {
            "id", "operation_type", "timestamp", "parameters", "input_file",
            "output_file", "duration_sec", "user", "undone", "undone_at",
            "session_id", "notes",
        }
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


def _project_key(project_id: str) -> str:
    """Create a safe filename key from a project ID or path."""
    # If it looks like a file path, hash it
    if os.sep in project_id or "/" in project_id:
        return hashlib.sha256(
            os.path.normpath(project_id).encode("utf-8")
        ).hexdigest()[:16]
    # Otherwise use sanitized project_id directly
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_id)
    return safe[:64] if safe else "default"


def _history_file(project_id: str) -> str:
    """Return the JSONL file path for a project's history."""
    return os.path.join(_HISTORY_DIR, f"{_project_key(project_id)}.jsonl")


# ---------------------------------------------------------------------------
# EditHistory — manages the audit log for one project
# ---------------------------------------------------------------------------

class EditHistory:
    """Manages the immutable edit history for a single project."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self._lock = threading.Lock()
        self._entries: List[HistoryEntry] = []
        self._load()

    def _load(self):
        """Load history entries from the JSONL file."""
        path = _history_file(self.project_id)
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        self._entries.append(HistoryEntry.from_dict(data))
                    except (json.JSONDecodeError, Exception) as exc:
                        logger.debug("Skipping malformed history line: %s", exc)
            logger.debug("Loaded %d history entries for %s",
                         len(self._entries), self.project_id)
        except Exception as exc:
            logger.warning("Failed to load history for %s: %s",
                           self.project_id, exc)

    def _append_to_file(self, entry: HistoryEntry):
        """Append a single entry to the JSONL file."""
        os.makedirs(_HISTORY_DIR, exist_ok=True)
        path = _history_file(self.project_id)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(entry.to_jsonl() + "\n")
        except Exception as exc:
            logger.warning("Failed to append history entry: %s", exc)

    def _rewrite_file(self):
        """Rewrite the entire JSONL file (after undo marking)."""
        os.makedirs(_HISTORY_DIR, exist_ok=True)
        path = _history_file(self.project_id)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                for entry in self._entries:
                    fh.write(entry.to_jsonl() + "\n")
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("Failed to rewrite history file: %s", exc)
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    # -- append -------------------------------------------------------------

    def add_entry(
        self,
        operation_type: str,
        parameters: Optional[Dict] = None,
        input_file: str = "",
        output_file: str = "",
        duration_sec: float = 0.0,
        user: str = "",
        session_id: str = "",
        notes: str = "",
    ) -> HistoryEntry:
        """Record a new operation in the history log."""
        if not operation_type:
            raise ValueError("operation_type is required")

        entry = HistoryEntry(
            operation_type=operation_type,
            parameters=parameters or {},
            input_file=input_file,
            output_file=output_file,
            duration_sec=duration_sec,
            user=user,
            session_id=session_id,
            notes=notes,
        )

        with self._lock:
            self._entries.append(entry)
            self._append_to_file(entry)

        logger.debug("Recorded history entry %s: %s", entry.id, operation_type)
        return entry

    # -- query --------------------------------------------------------------

    def get_entries(self, include_undone: bool = False) -> List[HistoryEntry]:
        """Return all entries, optionally excluding undone operations."""
        if include_undone:
            return list(self._entries)
        return [e for e in self._entries if not e.undone]

    def get_entry(self, entry_id: str) -> Optional[HistoryEntry]:
        """Get a single entry by ID."""
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None

    def filter_by_operation(self, operation_type: str) -> List[HistoryEntry]:
        """Return entries matching an operation type."""
        return [e for e in self._entries
                if e.operation_type == operation_type and not e.undone]

    def filter_by_user(self, user: str) -> List[HistoryEntry]:
        """Return entries by a specific user."""
        user_lower = user.lower()
        return [e for e in self._entries
                if e.user.lower() == user_lower and not e.undone]

    def filter_by_time_range(self, start: float, end: float) -> List[HistoryEntry]:
        """Return entries within a timestamp range."""
        return [e for e in self._entries
                if start <= e.timestamp <= end and not e.undone]

    def filter_by_session(self, session_id: str) -> List[HistoryEntry]:
        """Return entries for a specific session."""
        return [e for e in self._entries
                if e.session_id == session_id and not e.undone]

    # -- undo ---------------------------------------------------------------

    def mark_undone(self, entry_id: str) -> HistoryEntry:
        """Mark an entry as undone. Does not delete it (immutable log)."""
        with self._lock:
            for entry in self._entries:
                if entry.id == entry_id:
                    if entry.undone:
                        raise ValueError(f"Entry {entry_id} is already undone")
                    entry.undone = True
                    entry.undone_at = time.time()
                    self._rewrite_file()
                    return entry
        raise ValueError(f"Entry not found: {entry_id}")

    def undo_last(self) -> Optional[HistoryEntry]:
        """Mark the most recent non-undone entry as undone."""
        with self._lock:
            for entry in reversed(self._entries):
                if not entry.undone:
                    entry.undone = True
                    entry.undone_at = time.time()
                    self._rewrite_file()
                    return entry
        return None

    # -- diff ---------------------------------------------------------------

    def diff(self, index_a: int, index_b: int) -> dict:
        """Diff the history between two indices.

        Returns dict with entries added between index_a and index_b,
        plus counts of operations performed.
        """
        if index_a < 0 or index_b < 0:
            raise ValueError("Indices must be non-negative")
        if index_a > len(self._entries) or index_b > len(self._entries):
            raise ValueError("Index out of range")

        start = min(index_a, index_b)
        end = max(index_a, index_b)
        between = self._entries[start:end]

        op_counts: Dict[str, int] = {}
        for entry in between:
            op_counts[entry.operation_type] = op_counts.get(
                entry.operation_type, 0
            ) + 1

        return {
            "from_index": start,
            "to_index": end,
            "entry_count": len(between),
            "entries": [e.to_dict() for e in between],
            "operation_counts": op_counts,
        }

    # -- export -------------------------------------------------------------

    def export_json(self, include_undone: bool = False) -> str:
        """Export history as JSON string."""
        entries = self.get_entries(include_undone=include_undone)
        data = {
            "project_id": self.project_id,
            "exported_at": time.time(),
            "entry_count": len(entries),
            "entries": [e.to_dict() for e in entries],
        }
        return json.dumps(data, indent=2)

    def export_timeline(self) -> List[dict]:
        """Export as timeline visualization data (for frontend chart).

        Each item has x (timestamp), y (operation index), label, color hint.
        """
        entries = self.get_entries()
        timeline = []
        for i, entry in enumerate(entries):
            timeline.append({
                "x": entry.timestamp,
                "y": i,
                "label": entry.operation_type,
                "duration": entry.duration_sec,
                "user": entry.user,
                "undone": entry.undone,
                "id": entry.id,
            })
        return timeline

    def export_replay(self) -> List[dict]:
        """Generate an API call sequence to reproduce the edit chain.

        Returns a list of dicts, each describing an API call with
        endpoint, method, and body.
        """
        entries = self.get_entries(include_undone=False)
        calls = []
        for entry in entries:
            call = {
                "sequence": len(calls) + 1,
                "operation": entry.operation_type,
                "endpoint": f"/api/{entry.operation_type.replace('_', '-')}",
                "method": "POST",
                "body": {
                    "filepath": entry.input_file,
                    "output_path": entry.output_file,
                    **entry.parameters,
                },
                "original_id": entry.id,
                "original_timestamp": entry.timestamp,
            }
            calls.append(call)
        return calls

    # -- statistics ---------------------------------------------------------

    def statistics(self) -> dict:
        """Compute usage statistics across all entries."""
        entries = self.get_entries(include_undone=True)
        if not entries:
            return {
                "total_entries": 0,
                "total_undone": 0,
                "most_used_operations": [],
                "avg_duration_by_op": {},
                "total_duration_sec": 0,
                "unique_users": [],
                "time_span_hours": 0,
            }

        op_counts: Dict[str, int] = {}
        op_durations: Dict[str, List[float]] = {}
        users = set()
        total_duration = 0.0
        undone_count = 0

        for entry in entries:
            op = entry.operation_type
            op_counts[op] = op_counts.get(op, 0) + 1
            if op not in op_durations:
                op_durations[op] = []
            if entry.duration_sec > 0:
                op_durations[op].append(entry.duration_sec)
            total_duration += entry.duration_sec
            if entry.user:
                users.add(entry.user)
            if entry.undone:
                undone_count += 1

        most_used = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)
        avg_by_op = {}
        for op, durs in op_durations.items():
            if durs:
                avg_by_op[op] = round(sum(durs) / len(durs), 3)

        timestamps = [e.timestamp for e in entries if e.timestamp > 0]
        span_hours = 0.0
        if len(timestamps) >= 2:
            span_hours = round((max(timestamps) - min(timestamps)) / 3600, 2)

        return {
            "total_entries": len(entries),
            "total_undone": undone_count,
            "most_used_operations": [
                {"operation": op, "count": cnt} for op, cnt in most_used[:20]
            ],
            "avg_duration_by_op": avg_by_op,
            "total_duration_sec": round(total_duration, 2),
            "unique_users": sorted(users),
            "time_span_hours": span_hours,
        }

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_histories: Dict[str, EditHistory] = {}
_histories_lock = threading.Lock()


def get_history(project_id: str) -> EditHistory:
    """Get or create an EditHistory for a project."""
    with _histories_lock:
        if project_id not in _histories:
            _histories[project_id] = EditHistory(project_id)
        return _histories[project_id]


def clear_history_cache():
    """Clear the in-memory history cache (for testing)."""
    with _histories_lock:
        _histories.clear()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def record_operation(project_id: str, operation_type: str, **kwargs) -> dict:
    """Record an operation. Returns entry dict."""
    history = get_history(project_id)
    entry = history.add_entry(operation_type=operation_type, **kwargs)
    return entry.to_dict()


def get_entries(project_id: str, include_undone: bool = False) -> List[dict]:
    """Get all history entries for a project."""
    history = get_history(project_id)
    return [e.to_dict() for e in history.get_entries(include_undone=include_undone)]


def undo_entry(project_id: str, entry_id: str) -> dict:
    """Mark an entry as undone. Returns updated entry dict."""
    history = get_history(project_id)
    entry = history.mark_undone(entry_id)
    return entry.to_dict()


def export_history(project_id: str, fmt: str = "json",
                   include_undone: bool = False) -> str:
    """Export history in the given format."""
    history = get_history(project_id)
    if fmt == "timeline":
        return json.dumps(history.export_timeline(), indent=2)
    if fmt == "replay":
        return json.dumps(history.export_replay(), indent=2)
    return history.export_json(include_undone=include_undone)


def get_statistics(project_id: str) -> dict:
    """Get statistics for a project's edit history."""
    history = get_history(project_id)
    return history.statistics()
