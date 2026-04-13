"""
Review & Approval Links.

Generate shareable review URLs with token-based access,
timestamped comments, and an approval workflow.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

REVIEWS_DIR = os.path.join(OPENCUT_DIR, "reviews")


@dataclass
class ReviewComment:
    """A timestamped comment on a review."""
    comment_id: str
    review_id: str
    timestamp: float
    text: str
    author: str
    created_at: float = field(default_factory=time.time)


@dataclass
class ReviewLink:
    """A shareable review link with access token."""
    review_id: str
    video_path: str
    token: str
    status: str = "pending"          # pending | approved | rejected | revision_requested
    title: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    comments: List[ReviewComment] = field(default_factory=list)


def _reviews_path() -> str:
    os.makedirs(REVIEWS_DIR, exist_ok=True)
    return os.path.join(REVIEWS_DIR, "reviews.json")


def _load_reviews() -> Dict[str, dict]:
    path = _reviews_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt reviews file, starting fresh")
    return {}


def _save_reviews(reviews: Dict[str, dict]) -> None:
    path = _reviews_path()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(reviews, fh, indent=2)


def _generate_token(video_path: str) -> str:
    raw = f"{video_path}-{time.time()}-{os.getpid()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _generate_id(video_path: str) -> str:
    raw = f"review-{video_path}-{time.time()}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def create_review_link(
    video_path: str,
    title: str = "",
    expires_hours: Optional[float] = None,
    on_progress: Optional[Callable] = None,
) -> ReviewLink:
    """Create a shareable review link for a video.

    Args:
        video_path: Path to the video file.
        title: Optional title for the review.
        expires_hours: Optional expiry in hours from now.
        on_progress: Optional progress callback.

    Returns:
        ReviewLink dataclass with token and review_id.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(10, "Generating review token")

    review_id = _generate_id(video_path)
    token = _generate_token(video_path)
    expires_at = None
    if expires_hours and expires_hours > 0:
        expires_at = time.time() + (expires_hours * 3600)

    link = ReviewLink(
        review_id=review_id,
        video_path=video_path,
        token=token,
        title=title or os.path.basename(video_path),
        expires_at=expires_at,
    )

    if on_progress:
        on_progress(50, "Saving review")

    reviews = _load_reviews()
    reviews[review_id] = asdict(link)
    _save_reviews(reviews)

    if on_progress:
        on_progress(100, "Review link created")

    logger.info("Created review link %s for %s", review_id, video_path)
    return link


def add_review_comment(
    review_id: str,
    timestamp: float,
    text: str,
    author: str,
    on_progress: Optional[Callable] = None,
) -> ReviewComment:
    """Add a timestamped comment to a review.

    Args:
        review_id: The review identifier.
        timestamp: Video timestamp in seconds.
        text: Comment text.
        author: Author name or identifier.

    Returns:
        The created ReviewComment.
    """
    if not text or not text.strip():
        raise ValueError("Comment text cannot be empty")
    if timestamp < 0:
        raise ValueError("Timestamp must be non-negative")

    reviews = _load_reviews()
    if review_id not in reviews:
        raise KeyError(f"Review not found: {review_id}")

    comment_id = hashlib.md5(
        f"{review_id}-{timestamp}-{time.time()}".encode()
    ).hexdigest()[:12]

    comment = ReviewComment(
        comment_id=comment_id,
        review_id=review_id,
        timestamp=timestamp,
        text=text.strip(),
        author=author,
    )

    reviews[review_id].setdefault("comments", []).append(asdict(comment))
    _save_reviews(reviews)

    logger.info("Added comment %s to review %s", comment_id, review_id)
    return comment


def get_review_comments(
    review_id: str,
    on_progress: Optional[Callable] = None,
) -> List[ReviewComment]:
    """Get all comments for a review, sorted by timestamp.

    Args:
        review_id: The review identifier.

    Returns:
        List of ReviewComment sorted by timestamp.
    """
    reviews = _load_reviews()
    if review_id not in reviews:
        raise KeyError(f"Review not found: {review_id}")

    raw_comments = reviews[review_id].get("comments", [])
    comments = [ReviewComment(**c) for c in raw_comments]
    comments.sort(key=lambda c: c.timestamp)
    return comments


def update_review_status(
    review_id: str,
    status: str,
    on_progress: Optional[Callable] = None,
) -> ReviewLink:
    """Update the approval status of a review.

    Args:
        review_id: The review identifier.
        status: New status (pending, approved, rejected, revision_requested).

    Returns:
        Updated ReviewLink.
    """
    valid_statuses = {"pending", "approved", "rejected", "revision_requested"}
    if status not in valid_statuses:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {valid_statuses}")

    reviews = _load_reviews()
    if review_id not in reviews:
        raise KeyError(f"Review not found: {review_id}")

    reviews[review_id]["status"] = status
    _save_reviews(reviews)

    data = reviews[review_id]
    comments = [ReviewComment(**c) for c in data.pop("comments", [])]
    link = ReviewLink(**data)
    link.comments = comments

    logger.info("Updated review %s status to %s", review_id, status)
    return link


def get_review(review_id: str, token: str) -> ReviewLink:
    """Retrieve a review by ID and token (access check).

    Args:
        review_id: The review identifier.
        token: Access token.

    Returns:
        ReviewLink if token matches.
    """
    reviews = _load_reviews()
    if review_id not in reviews:
        raise KeyError(f"Review not found: {review_id}")

    data = reviews[review_id]
    if data["token"] != token:
        raise PermissionError("Invalid review token")

    if data.get("expires_at") and time.time() > data["expires_at"]:
        raise PermissionError("Review link has expired")

    comments = [ReviewComment(**c) for c in data.get("comments", [])]
    raw = {k: v for k, v in data.items() if k != "comments"}
    link = ReviewLink(**raw)
    link.comments = comments
    return link
