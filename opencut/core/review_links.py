"""Versioned review and approval links.

Each review keeps a stable identity while render artifacts, comments, and
approval state are scoped to immutable versions. Legacy single-file records
are migrated in place after an exact pre-migration backup is written.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Callable, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

REVIEWS_DIR = os.path.join(OPENCUT_DIR, "reviews")
REVIEW_SCHEMA_VERSION = 2
_VALID_STATUSES = {"pending", "approved", "rejected", "revision_requested"}
_reviews_lock = threading.RLock()


@dataclass
class ReviewComment:
    """A timestamped comment bound to one immutable review version."""

    comment_id: str
    review_id: str
    timestamp: float
    text: str
    author: str
    created_at: float = field(default_factory=time.time)
    version_id: str = ""


@dataclass
class ReviewVersion:
    """One immutable render artifact under a stable review identity."""

    version_id: str
    number: int
    video_path: str
    label: str = ""
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    status_updated_at: Optional[float] = None
    artifact_sha256: str = ""
    size_bytes: int = 0
    managed: bool = False


@dataclass
class ReviewLink:
    """A shareable review identity with one or more render versions."""

    review_id: str
    video_path: str
    token: str
    status: str = "pending"
    title: str = ""
    project_id: str = ""
    created_at: float = field(default_factory=time.time)
    status_updated_at: Optional[float] = None
    expires_at: Optional[float] = None
    comments: List[ReviewComment] = field(default_factory=list)
    versions: List[ReviewVersion] = field(default_factory=list)
    current_version_id: str = ""
    schema_version: int = REVIEW_SCHEMA_VERSION


def _reviews_path() -> str:
    os.makedirs(REVIEWS_DIR, exist_ok=True)
    return os.path.join(REVIEWS_DIR, "reviews.json")


def _backup_path(path: Optional[str] = None) -> str:
    reviews_path = path or _reviews_path()
    stem, _extension = os.path.splitext(reviews_path)
    return f"{stem}.pre-versioning.json"


def _atomic_write_json(path: str, payload: Dict[str, dict]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, temporary_path = tempfile.mkstemp(prefix=".reviews-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def _save_reviews(reviews: Dict[str, dict]) -> None:
    with _reviews_lock:
        _atomic_write_json(_reviews_path(), reviews)


def _safe_extension(video_path: str) -> str:
    extension = os.path.splitext(video_path)[1].lower()
    if 1 < len(extension) <= 16 and extension[1:].isalnum():
        return extension
    return ".bin"


def _artifact_path(review_id: str, version_id: str, video_path: str) -> str:
    reviews_directory = os.path.dirname(os.path.abspath(_reviews_path()))
    storage_id = hashlib.sha256(review_id.encode("utf-8")).hexdigest()[:32]
    artifact_directory = os.path.join(reviews_directory, "artifacts", storage_id)
    os.makedirs(artifact_directory, exist_ok=True)
    return os.path.join(artifact_directory, f"{version_id}{_safe_extension(video_path)}")


def _snapshot_artifact(
    source_path: str,
    review_id: str,
    version_id: str,
    on_progress: Optional[Callable] = None,
) -> tuple[str, str, int]:
    """Copy an artifact into managed storage while hashing the copied bytes."""
    destination = _artifact_path(review_id, version_id, source_path)
    directory = os.path.dirname(destination)
    handle, temporary_path = tempfile.mkstemp(prefix=f".{version_id}-", suffix=".tmp", dir=directory)
    digest = hashlib.sha256()
    size_bytes = 0
    source_size = max(1, os.path.getsize(source_path))
    try:
        with open(source_path, "rb") as source, os.fdopen(handle, "wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                size_bytes += len(chunk)
                if on_progress:
                    on_progress(min(90, 10 + int(80 * size_bytes / source_size)), "Snapshotting review artifact")
            target.flush()
            os.fsync(target.fileno())
        shutil.copystat(source_path, temporary_path)
        os.replace(temporary_path, destination)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise
    return destination, digest.hexdigest(), size_bytes


def _legacy_version(review_id: str, record: dict) -> dict:
    """Build the first version without dropping any legacy review fields."""
    source_path = str(record.get("video_path") or "")
    stored_path = source_path
    artifact_sha256 = ""
    size_bytes = 0
    managed = False
    if source_path and os.path.isfile(source_path):
        try:
            stored_path, artifact_sha256, size_bytes = _snapshot_artifact(source_path, review_id, "v1")
            managed = True
        except OSError as exc:
            logger.warning("Could not snapshot legacy review %s: %s", review_id, exc)
    created_at = record.get("created_at")
    return {
        "version_id": "v1",
        "number": 1,
        "video_path": stored_path,
        "label": "Version 1",
        "status": str(record.get("status") or "pending"),
        "created_at": float(created_at if created_at is not None else time.time()),
        "status_updated_at": record.get("status_updated_at"),
        "artifact_sha256": artifact_sha256,
        "size_bytes": size_bytes,
        "managed": managed,
    }


def _migrate_reviews(reviews: Dict[str, dict]) -> tuple[Dict[str, dict], bool]:
    migrated = False
    for review_id, raw in list(reviews.items()):
        if not isinstance(raw, dict):
            continue
        record = dict(raw)
        versions = record.get("versions")
        if not isinstance(versions, list) or not versions:
            versions = [_legacy_version(review_id, record)]
            record["versions"] = versions
            migrated = True

        current_version_id = str(record.get("current_version_id") or versions[-1].get("version_id") or "v1")
        version_ids = {str(version.get("version_id") or "") for version in versions if isinstance(version, dict)}
        if current_version_id not in version_ids:
            current_version_id = str(versions[-1].get("version_id") or "v1")
            migrated = True
        if record.get("current_version_id") != current_version_id:
            record["current_version_id"] = current_version_id
            migrated = True

        comments = []
        for raw_comment in record.get("comments", []) or []:
            if not isinstance(raw_comment, dict):
                continue
            comment = dict(raw_comment)
            if not comment.get("version_id"):
                comment["version_id"] = current_version_id
                migrated = True
            comments.append(comment)
        if comments != record.get("comments", []):
            record["comments"] = comments

        current = next(
            (version for version in versions if str(version.get("version_id") or "") == current_version_id),
            versions[-1],
        )
        for key in ("video_path", "status", "status_updated_at"):
            value = current.get(key)
            if record.get(key) != value:
                record[key] = value
                migrated = True
        if record.get("schema_version") != REVIEW_SCHEMA_VERSION:
            record["schema_version"] = REVIEW_SCHEMA_VERSION
            migrated = True
        reviews[review_id] = record
    return reviews, migrated


def _load_reviews() -> Dict[str, dict]:
    with _reviews_lock:
        path = _reviews_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as stream:
                reviews = json.load(stream)
            if not isinstance(reviews, dict):
                raise ValueError("reviews root must be an object")
        except (json.JSONDecodeError, OSError, ValueError):
            logger.warning("Corrupt reviews file, starting fresh")
            return {}

        original = json.loads(json.dumps(reviews))
        reviews, migrated = _migrate_reviews(reviews)
        if migrated:
            backup = _backup_path(path)
            if not os.path.exists(backup):
                _atomic_write_json(backup, original)
            _atomic_write_json(path, reviews)
            logger.info("Migrated review data to schema %s; backup: %s", REVIEW_SCHEMA_VERSION, backup)
        return reviews


def rollback_review_migration() -> str:
    """Restore the exact pre-versioning JSON export and return its path."""
    with _reviews_lock:
        path = _reviews_path()
        backup = _backup_path(path)
        if not os.path.isfile(backup):
            raise FileNotFoundError(f"Review migration backup not found: {backup}")
        try:
            with open(backup, "r", encoding="utf-8") as stream:
                data = json.load(stream)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Review migration backup is invalid: {backup}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Review migration backup is invalid: {backup}")
        _atomic_write_json(path, data)
        logger.info("Restored pre-versioning review data from %s", backup)
        return backup


def export_review_data(output_path: str) -> str:
    """Export the current versioned review records to an explicit JSON path."""
    destination = os.path.abspath(os.path.expanduser(str(output_path or "").strip()))
    if not output_path or os.path.isdir(destination):
        raise ValueError("output_path must name a JSON file")
    with _reviews_lock:
        _atomic_write_json(destination, _load_reviews())
    return destination


def _generate_token(video_path: str) -> str:
    raw = f"{video_path}-{time.time()}-{os.getpid()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _generate_id(video_path: str) -> str:
    raw = f"review-{video_path}-{time.time()}"
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()[:16]


def _version_for(record: dict, version_id: str = "") -> dict:
    versions = record.get("versions") or []
    selected_id = str(version_id or record.get("current_version_id") or "")
    for version in versions:
        if isinstance(version, dict) and version.get("version_id") == selected_id:
            return version
    raise KeyError(f"Review version not found: {selected_id}")


def _comment_from_dict(raw: dict) -> ReviewComment:
    allowed = {item.name for item in fields(ReviewComment)}
    return ReviewComment(**{key: value for key, value in raw.items() if key in allowed})


def _version_from_dict(raw: dict) -> ReviewVersion:
    allowed = {item.name for item in fields(ReviewVersion)}
    return ReviewVersion(**{key: value for key, value in raw.items() if key in allowed})


def _link_from_record(record: dict) -> ReviewLink:
    comments = [_comment_from_dict(raw) for raw in record.get("comments", []) if isinstance(raw, dict)]
    versions = [_version_from_dict(raw) for raw in record.get("versions", []) if isinstance(raw, dict)]
    allowed = {item.name for item in fields(ReviewLink)} - {"comments", "versions"}
    link = ReviewLink(**{key: value for key, value in record.items() if key in allowed})
    link.comments = comments
    link.versions = versions
    return link


def create_review_link(
    video_path: str,
    title: str = "",
    project_id: str = "",
    expires_hours: Optional[float] = None,
    on_progress: Optional[Callable] = None,
    review_id: str = "",
    version_label: str = "",
) -> ReviewLink:
    """Create a review, or append a render version to an existing review ID."""
    if review_id:
        return add_review_version(
            review_id,
            video_path,
            label=version_label,
            title=title,
            on_progress=on_progress,
        )
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(5, "Generating review identity")
    created_at = time.time()
    review_id = _generate_id(video_path)
    token = _generate_token(video_path)
    expires_at = time.time() + (expires_hours * 3600) if expires_hours and expires_hours > 0 else None
    managed_path, digest, size_bytes = _snapshot_artifact(video_path, review_id, "v1", on_progress)
    version = ReviewVersion(
        version_id="v1",
        number=1,
        video_path=managed_path,
        label=version_label.strip() or "Version 1",
        created_at=created_at,
        artifact_sha256=digest,
        size_bytes=size_bytes,
        managed=True,
    )
    link = ReviewLink(
        review_id=review_id,
        video_path=managed_path,
        token=token,
        title=title or os.path.basename(video_path),
        project_id=str(project_id or "").strip(),
        created_at=created_at,
        expires_at=expires_at,
        versions=[version],
        current_version_id=version.version_id,
    )

    if on_progress:
        on_progress(95, "Saving review")
    with _reviews_lock:
        reviews = _load_reviews()
        reviews[review_id] = asdict(link)
        _save_reviews(reviews)
    if on_progress:
        on_progress(100, "Review link created")
    logger.info("Created review %s version %s for %s", review_id, version.version_id, video_path)
    return link


def add_review_version(
    review_id: str,
    video_path: str,
    *,
    label: str = "",
    title: str = "",
    on_progress: Optional[Callable] = None,
) -> ReviewLink:
    """Append an immutable artifact version and make it the current version."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    with _reviews_lock:
        reviews = _load_reviews()
        if review_id not in reviews:
            raise KeyError(f"Review not found: {review_id}")
        record = reviews[review_id]
        versions = record.get("versions") or []
        number = (
            max((int(version.get("number") or 0) for version in versions if isinstance(version, dict)), default=0) + 1
        )
        version_id = f"v{number}"
        managed_path, digest, size_bytes = _snapshot_artifact(video_path, review_id, version_id, on_progress)
        version = ReviewVersion(
            version_id=version_id,
            number=number,
            video_path=managed_path,
            label=label.strip() or f"Version {number}",
            artifact_sha256=digest,
            size_bytes=size_bytes,
            managed=True,
        )
        versions.append(asdict(version))
        record["versions"] = versions
        record["current_version_id"] = version_id
        record["video_path"] = managed_path
        record["status"] = version.status
        record["status_updated_at"] = None
        record["schema_version"] = REVIEW_SCHEMA_VERSION
        if title.strip():
            record["title"] = title.strip()
        _save_reviews(reviews)
        link = _link_from_record(record)
    if on_progress:
        on_progress(100, f"Review {version_id} created")
    logger.info("Created review %s version %s for %s", review_id, version_id, video_path)
    return link


def get_review_versions(review_id: str) -> List[ReviewVersion]:
    """Return every retained version in creation order."""
    reviews = _load_reviews()
    if review_id not in reviews:
        raise KeyError(f"Review not found: {review_id}")
    versions = [_version_from_dict(raw) for raw in reviews[review_id].get("versions", []) if isinstance(raw, dict)]
    return sorted(versions, key=lambda version: (version.number, version.created_at))


def add_review_comment(
    review_id: str,
    timestamp: float,
    text: str,
    author: str,
    on_progress: Optional[Callable] = None,
    version_id: str = "",
) -> ReviewComment:
    """Add a timestamped comment to one review version."""
    if not text or not text.strip():
        raise ValueError("Comment text cannot be empty")
    if timestamp < 0:
        raise ValueError("Timestamp must be non-negative")

    with _reviews_lock:
        reviews = _load_reviews()
        if review_id not in reviews:
            raise KeyError(f"Review not found: {review_id}")
        selected = _version_for(reviews[review_id], version_id)
        selected_id = str(selected["version_id"])
        comment_id = hashlib.md5(
            f"{review_id}-{selected_id}-{timestamp}-{time.time()}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]
        comment = ReviewComment(
            comment_id=comment_id,
            review_id=review_id,
            version_id=selected_id,
            timestamp=timestamp,
            text=text.strip(),
            author=author,
        )
        reviews[review_id].setdefault("comments", []).append(asdict(comment))
        _save_reviews(reviews)
    logger.info("Added comment %s to review %s version %s", comment_id, review_id, selected_id)
    return comment


def get_review_comments(
    review_id: str,
    on_progress: Optional[Callable] = None,
    version_id: str = "",
) -> List[ReviewComment]:
    """Get review comments, optionally restricted to one version."""
    reviews = _load_reviews()
    if review_id not in reviews:
        raise KeyError(f"Review not found: {review_id}")
    if version_id:
        _version_for(reviews[review_id], version_id)
    comments = [
        _comment_from_dict(raw)
        for raw in reviews[review_id].get("comments", [])
        if isinstance(raw, dict) and (not version_id or raw.get("version_id") == version_id)
    ]
    comments.sort(key=lambda comment: (comment.timestamp, comment.created_at, comment.comment_id))
    return comments


def update_review_status(
    review_id: str,
    status: str,
    on_progress: Optional[Callable] = None,
    version_id: str = "",
) -> ReviewLink:
    """Update approval state for one review version."""
    if status not in _VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {_VALID_STATUSES}")
    with _reviews_lock:
        reviews = _load_reviews()
        if review_id not in reviews:
            raise KeyError(f"Review not found: {review_id}")
        record = reviews[review_id]
        selected = _version_for(record, version_id)
        selected_id = str(selected["version_id"])
        updated_at = time.time()
        selected["status"] = status
        selected["status_updated_at"] = updated_at
        if selected_id == record.get("current_version_id"):
            record["status"] = status
            record["status_updated_at"] = updated_at
        _save_reviews(reviews)
        link = _link_from_record(record)
        link.status = status
        link.status_updated_at = updated_at
    logger.info("Updated review %s version %s status to %s", review_id, selected_id, status)
    return link


def get_review(review_id: str, token: str, version_id: str = "") -> ReviewLink:
    """Retrieve a review by stable ID and bearer token."""
    reviews = _load_reviews()
    if review_id not in reviews:
        raise KeyError(f"Review not found: {review_id}")
    record = reviews[review_id]
    if record["token"] != token:
        raise PermissionError("Invalid review token")
    if record.get("expires_at") and time.time() > record["expires_at"]:
        raise PermissionError("Review link has expired")
    link = _link_from_record(record)
    if version_id:
        selected = _version_for(record, version_id)
        link.video_path = str(selected.get("video_path") or "")
        link.status = str(selected.get("status") or "pending")
        link.status_updated_at = selected.get("status_updated_at")
        link.comments = [comment for comment in link.comments if comment.version_id == version_id]
    return link
