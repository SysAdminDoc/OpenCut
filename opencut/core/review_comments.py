"""
OpenCut Frame-Accurate Review Comments

Manages review comments anchored to specific frames/timestamps in a
project.  Supports threaded discussions, annotation types (text, drawing),
status tracking (open/resolved/wontfix), and export to JSON or CSV.

Comments are persisted as JSON files in ``~/.opencut/reviews/``, one per
project (keyed by a hash of the project file path).

Frame.io-compatible JSON import is also supported so teams can migrate
existing review data.
"""

import csv
import hashlib
import io
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
_REVIEWS_DIR = os.path.join(_OPENCUT_DIR, "reviews")

# ---------------------------------------------------------------------------
# Annotation types
# ---------------------------------------------------------------------------
ANNOTATION_TYPES = frozenset({"text", "drawing_rect", "drawing_circle", "drawing_arrow"})

# ---------------------------------------------------------------------------
# Comment status values
# ---------------------------------------------------------------------------
VALID_STATUSES = frozenset({"open", "resolved", "wontfix"})


@dataclass
class ReviewComment:
    """A single review comment anchored to a frame in the timeline."""

    id: str = ""
    timestamp_sec: float = 0.0
    frame_number: int = 0
    author: str = ""
    text: str = ""
    status: str = "open"
    created_at: float = 0.0
    updated_at: float = 0.0
    parent_id: str = ""
    annotation_type: str = "text"
    annotation_data: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = time.time()
        if not self.updated_at:
            self.updated_at = self.created_at
        if self.status not in VALID_STATUSES:
            self.status = "open"
        if self.annotation_type not in ANNOTATION_TYPES:
            self.annotation_type = "text"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewComment":
        known = {
            "id", "timestamp_sec", "frame_number", "author", "text",
            "status", "created_at", "updated_at", "parent_id",
            "annotation_type", "annotation_data", "tags",
        }
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


def _project_hash(project_path: str) -> str:
    """Deterministic hash for a project path, used as filename."""
    return hashlib.sha256(os.path.normpath(project_path).encode("utf-8")).hexdigest()[:16]


def _review_file(project_path: str) -> str:
    """Return the JSON file path for a project's review data."""
    return os.path.join(_REVIEWS_DIR, f"{_project_hash(project_path)}.json")


# ---------------------------------------------------------------------------
# ReviewSession — manages comments for one project
# ---------------------------------------------------------------------------

class ReviewSession:
    """Manages review comments for a single project file."""

    def __init__(self, project_path: str):
        self.project_path = project_path
        self._lock = threading.Lock()
        self._comments: Dict[str, ReviewComment] = {}
        self._load()

    # -- persistence --------------------------------------------------------

    def _load(self):
        """Load comments from disk if the file exists."""
        path = _review_file(self.project_path)
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for entry in data.get("comments", []):
                comment = ReviewComment.from_dict(entry)
                self._comments[comment.id] = comment
            logger.debug("Loaded %d review comments for %s",
                         len(self._comments), self.project_path)
        except Exception as exc:
            logger.warning("Failed to load review file %s: %s", path, exc)

    def _save(self):
        """Persist comments to disk."""
        os.makedirs(_REVIEWS_DIR, exist_ok=True)
        path = _review_file(self.project_path)
        data = {
            "project_path": self.project_path,
            "saved_at": time.time(),
            "comments": [c.to_dict() for c in self._comments.values()],
        }
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("Failed to save review file %s: %s", path, exc)
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    # -- CRUD ---------------------------------------------------------------

    def add_comment(
        self,
        text: str,
        author: str = "anonymous",
        timestamp_sec: float = 0.0,
        frame_number: int = 0,
        parent_id: str = "",
        annotation_type: str = "text",
        annotation_data: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> ReviewComment:
        """Add a new review comment and persist."""
        if not text or not text.strip():
            raise ValueError("Comment text is required")

        if parent_id and parent_id not in self._comments:
            raise ValueError(f"Parent comment not found: {parent_id}")

        comment = ReviewComment(
            timestamp_sec=timestamp_sec,
            frame_number=frame_number,
            author=author,
            text=text.strip(),
            parent_id=parent_id,
            annotation_type=annotation_type if annotation_type in ANNOTATION_TYPES else "text",
            annotation_data=annotation_data or {},
            tags=tags or [],
        )

        with self._lock:
            self._comments[comment.id] = comment
            self._save()

        logger.info("Added review comment %s by %s at %.2fs",
                     comment.id, author, timestamp_sec)
        return comment

    def get_comment(self, comment_id: str) -> Optional[ReviewComment]:
        """Get a single comment by ID."""
        return self._comments.get(comment_id)

    def update_comment(self, comment_id: str, text: Optional[str] = None,
                       status: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> ReviewComment:
        """Update a comment's text, status, or tags."""
        with self._lock:
            comment = self._comments.get(comment_id)
            if comment is None:
                raise ValueError(f"Comment not found: {comment_id}")

            if text is not None:
                comment.text = text.strip()
            if status is not None:
                if status not in VALID_STATUSES:
                    raise ValueError(f"Invalid status: {status}. Must be one of: {sorted(VALID_STATUSES)}")
                comment.status = status
            if tags is not None:
                comment.tags = tags
            comment.updated_at = time.time()
            self._save()

        return comment

    def resolve_comment(self, comment_id: str, status: str = "resolved") -> ReviewComment:
        """Resolve (or wontfix) a comment."""
        if status not in ("resolved", "wontfix"):
            raise ValueError(f"Invalid resolve status: {status}")
        return self.update_comment(comment_id, status=status)

    def delete_comment(self, comment_id: str) -> bool:
        """Delete a comment. Returns True if found and deleted."""
        with self._lock:
            if comment_id not in self._comments:
                return False
            # Also delete child comments (replies)
            children = [c.id for c in self._comments.values()
                        if c.parent_id == comment_id]
            del self._comments[comment_id]
            for child_id in children:
                self._comments.pop(child_id, None)
            self._save()
        logger.info("Deleted review comment %s (and %d replies)",
                     comment_id, len(children))
        return True

    def list_comments(self, sort_by: str = "timestamp_sec",
                      ascending: bool = True) -> List[ReviewComment]:
        """Return all comments, sorted by the given field."""
        comments = list(self._comments.values())
        reverse = not ascending
        if sort_by == "created_at":
            comments.sort(key=lambda c: c.created_at, reverse=reverse)
        elif sort_by == "updated_at":
            comments.sort(key=lambda c: c.updated_at, reverse=reverse)
        elif sort_by == "author":
            comments.sort(key=lambda c: c.author.lower(), reverse=reverse)
        else:
            comments.sort(key=lambda c: c.timestamp_sec, reverse=reverse)
        return comments

    def filter_by_status(self, status: str) -> List[ReviewComment]:
        """Return comments matching the given status."""
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        return [c for c in self._comments.values() if c.status == status]

    def filter_by_time_range(self, start_sec: float, end_sec: float) -> List[ReviewComment]:
        """Return comments within a timestamp range (inclusive)."""
        return [
            c for c in self._comments.values()
            if start_sec <= c.timestamp_sec <= end_sec
        ]

    def filter_by_author(self, author: str) -> List[ReviewComment]:
        """Return comments by a specific author (case-insensitive)."""
        author_lower = author.lower()
        return [c for c in self._comments.values()
                if c.author.lower() == author_lower]

    def get_thread(self, comment_id: str) -> List[ReviewComment]:
        """Return a comment and all its replies, sorted by created_at."""
        if comment_id not in self._comments:
            raise ValueError(f"Comment not found: {comment_id}")
        thread = [self._comments[comment_id]]
        thread.extend(c for c in self._comments.values()
                      if c.parent_id == comment_id)
        thread.sort(key=lambda c: c.created_at)
        return thread

    # -- export / import ----------------------------------------------------

    def export_json(self) -> str:
        """Export all comments as a JSON string."""
        data = {
            "project_path": self.project_path,
            "exported_at": time.time(),
            "comments": [c.to_dict() for c in self.list_comments()],
        }
        return json.dumps(data, indent=2)

    def export_csv(self) -> str:
        """Export all comments as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "id", "timestamp_sec", "frame_number", "author", "text",
            "status", "created_at", "annotation_type", "parent_id", "tags",
        ])
        for c in self.list_comments():
            writer.writerow([
                c.id, c.timestamp_sec, c.frame_number, c.author, c.text,
                c.status, c.created_at, c.annotation_type, c.parent_id,
                ";".join(c.tags),
            ])
        return output.getvalue()

    def import_json(self, json_str: str, merge: bool = True) -> int:
        """Import comments from JSON string. Returns count of imported comments.

        If *merge* is True, existing comments with matching IDs are kept.
        If False, imported comments overwrite existing ones with the same ID.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        entries = data.get("comments", [])
        if not isinstance(entries, list):
            raise ValueError("Expected 'comments' array in JSON data")

        imported = 0
        with self._lock:
            for entry in entries:
                try:
                    comment = ReviewComment.from_dict(entry)
                except Exception as exc:
                    logger.warning("Skipping invalid comment entry: %s", exc)
                    continue

                if merge and comment.id in self._comments:
                    continue

                self._comments[comment.id] = comment
                imported += 1

            if imported > 0:
                self._save()

        logger.info("Imported %d comments from JSON for %s", imported, self.project_path)
        return imported

    def import_frameio(self, json_str: str) -> int:
        """Import comments from Frame.io-compatible JSON format.

        Frame.io format uses ``timestamp``, ``text``, ``owner.name``, etc.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        entries = data if isinstance(data, list) else data.get("comments", [])
        imported = 0

        with self._lock:
            for entry in entries:
                try:
                    author = entry.get("owner", {}).get("name", "") if isinstance(entry.get("owner"), dict) else entry.get("author", "unknown")
                    comment = ReviewComment(
                        timestamp_sec=float(entry.get("timestamp", 0)),
                        frame_number=int(entry.get("frame", 0)),
                        author=author,
                        text=entry.get("text", ""),
                        status="resolved" if entry.get("completed") else "open",
                        created_at=entry.get("inserted_at", time.time()),
                    )
                    self._comments[comment.id] = comment
                    imported += 1
                except Exception as exc:
                    logger.warning("Skipping Frame.io entry: %s", exc)

            if imported > 0:
                self._save()

        logger.info("Imported %d Frame.io comments for %s", imported, self.project_path)
        return imported

    # -- stats --------------------------------------------------------------

    def summary_stats(self) -> dict:
        """Return summary statistics for the review session."""
        comments = list(self._comments.values())
        total = len(comments)
        by_status: Dict[str, int] = {}
        by_author: Dict[str, int] = {}

        for c in comments:
            by_status[c.status] = by_status.get(c.status, 0) + 1
            by_author[c.author] = by_author.get(c.author, 0) + 1

        return {
            "total": total,
            "open": by_status.get("open", 0),
            "resolved": by_status.get("resolved", 0),
            "wontfix": by_status.get("wontfix", 0),
            "by_author": by_author,
            "by_status": by_status,
        }

    @property
    def comment_count(self) -> int:
        return len(self._comments)


# ---------------------------------------------------------------------------
# Module-level session cache
# ---------------------------------------------------------------------------

_sessions: Dict[str, ReviewSession] = {}
_sessions_lock = threading.Lock()


def get_session(project_path: str) -> ReviewSession:
    """Get or create a ReviewSession for a project path."""
    key = os.path.normpath(project_path)
    with _sessions_lock:
        if key not in _sessions:
            _sessions[key] = ReviewSession(project_path)
        return _sessions[key]


def clear_session_cache():
    """Clear the in-memory session cache (for testing)."""
    with _sessions_lock:
        _sessions.clear()


# ---------------------------------------------------------------------------
# Convenience functions (module-level)
# ---------------------------------------------------------------------------

def add_comment(project_path: str, text: str, **kwargs) -> dict:
    """Add a review comment to a project. Returns comment dict."""
    session = get_session(project_path)
    comment = session.add_comment(text=text, **kwargs)
    return comment.to_dict()


def resolve_comment(project_path: str, comment_id: str,
                    status: str = "resolved") -> dict:
    """Resolve a comment. Returns updated comment dict."""
    session = get_session(project_path)
    comment = session.resolve_comment(comment_id, status=status)
    return comment.to_dict()


def list_comments(project_path: str, status: Optional[str] = None,
                  start_sec: Optional[float] = None,
                  end_sec: Optional[float] = None,
                  author: Optional[str] = None) -> List[dict]:
    """List and optionally filter comments for a project."""
    session = get_session(project_path)

    if status:
        comments = session.filter_by_status(status)
    elif start_sec is not None and end_sec is not None:
        comments = session.filter_by_time_range(start_sec, end_sec)
    elif author:
        comments = session.filter_by_author(author)
    else:
        comments = session.list_comments()

    return [c.to_dict() for c in comments]


def delete_comment(project_path: str, comment_id: str) -> bool:
    """Delete a comment. Returns True if deleted."""
    session = get_session(project_path)
    return session.delete_comment(comment_id)


def export_comments(project_path: str, fmt: str = "json") -> str:
    """Export comments as JSON or CSV string."""
    session = get_session(project_path)
    if fmt == "csv":
        return session.export_csv()
    return session.export_json()


def import_comments(project_path: str, json_str: str,
                    source: str = "opencut", merge: bool = True) -> int:
    """Import comments from JSON string. Returns count imported."""
    session = get_session(project_path)
    if source == "frameio":
        return session.import_frameio(json_str)
    return session.import_json(json_str, merge=merge)


def get_stats(project_path: str) -> dict:
    """Get summary statistics for a project's review comments."""
    session = get_session(project_path)
    return session.summary_stats()
