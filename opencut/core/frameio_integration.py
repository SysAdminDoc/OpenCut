"""
Frame.io Review Integration.

Upload videos to Frame.io, receive comments, resolve them,
and sync review state two-way.
"""

import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

FRAMEIO_API_BASE = "https://api.frame.io/v2"


@dataclass
class FrameIOComment:
    """A comment from Frame.io."""
    comment_id: str
    text: str
    author: str = ""
    timestamp: Optional[float] = None   # seconds into video
    created_at: str = ""
    completed: bool = False
    replies: List[Dict] = field(default_factory=list)


@dataclass
class FrameIOUploadResult:
    """Result of uploading to Frame.io."""
    asset_id: str = ""
    name: str = ""
    upload_url: str = ""
    status: str = ""
    file_size: int = 0


def _frameio_request(
    method: str,
    path: str,
    api_key: str,
    data: Optional[dict] = None,
) -> dict:
    """Make an authenticated request to Frame.io API."""
    url = f"{FRAMEIO_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Frame.io API error ({exc.code}): {error_body[:200]}"
        )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def upload_to_frameio(
    video_path: str,
    project_id: str,
    api_key: str,
    name: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> FrameIOUploadResult:
    """Upload a video to Frame.io for review.

    Args:
        video_path: Path to the video file.
        project_id: Frame.io project ID.
        api_key: Frame.io API key/token.
        name: Display name for the asset (defaults to filename).

    Returns:
        FrameIOUploadResult with asset ID and status.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not api_key:
        raise ValueError("Frame.io API key is required")

    file_size = os.path.getsize(video_path)
    if name is None:
        name = os.path.basename(video_path)

    if on_progress:
        on_progress(10, "Creating Frame.io asset")

    # Get the root asset ID for the project
    project = _frameio_request("GET", f"/projects/{project_id}", api_key)
    root_asset_id = project.get("root_asset_id", "")
    if not root_asset_id:
        raise RuntimeError("Could not determine project root asset")

    # Create asset entry
    asset_data = {
        "name": name,
        "type": "file",
        "filetype": "video/" + os.path.splitext(video_path)[1].lstrip("."),
        "filesize": file_size,
    }
    asset = _frameio_request(
        "POST", f"/assets/{root_asset_id}/children", api_key, data=asset_data
    )

    asset_id = asset.get("id", "")
    upload_urls = asset.get("upload_urls", [])

    if on_progress:
        on_progress(30, "Uploading video data")

    # Upload file chunks
    chunk_size = 50 * 1024 * 1024  # 50MB chunks
    with open(video_path, "rb") as fh:
        for i, url in enumerate(upload_urls):
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            req = urllib.request.Request(url, data=chunk, method="PUT")
            req.add_header("Content-Type", "application/octet-stream")
            req.add_header("Content-Length", str(len(chunk)))
            urllib.request.urlopen(req, timeout=120)

            if on_progress:
                pct = 30 + int(60 * (i + 1) / max(len(upload_urls), 1))
                on_progress(pct, f"Uploading chunk {i+1}/{len(upload_urls)}")

    if on_progress:
        on_progress(100, "Upload complete")

    logger.info("Uploaded %s to Frame.io: %s", name, asset_id)
    return FrameIOUploadResult(
        asset_id=asset_id,
        name=name,
        upload_url=asset.get("url", ""),
        status="uploaded",
        file_size=file_size,
    )


def get_frameio_comments(
    asset_id: str,
    api_key: str,
    on_progress: Optional[Callable] = None,
) -> List[FrameIOComment]:
    """Get all comments on a Frame.io asset.

    Args:
        asset_id: Frame.io asset ID.
        api_key: Frame.io API key/token.

    Returns:
        List of FrameIOComment sorted by timestamp.
    """
    if not api_key:
        raise ValueError("Frame.io API key is required")

    if on_progress:
        on_progress(30, "Fetching comments")

    data = _frameio_request("GET", f"/assets/{asset_id}/comments", api_key)
    comments = []

    for c in data if isinstance(data, list) else data.get("comments", data.get("results", [])):
        timestamp = c.get("timestamp", None)
        if timestamp is not None:
            timestamp = float(timestamp)

        replies = []
        for r in c.get("replies", []):
            replies.append({
                "text": r.get("text", ""),
                "author": r.get("owner", {}).get("name", ""),
                "created_at": r.get("inserted_at", ""),
            })

        comments.append(FrameIOComment(
            comment_id=c.get("id", ""),
            text=c.get("text", ""),
            author=c.get("owner", {}).get("name", ""),
            timestamp=timestamp,
            created_at=c.get("inserted_at", ""),
            completed=c.get("completed", False),
            replies=replies,
        ))

    comments.sort(key=lambda c: c.timestamp if c.timestamp is not None else float("inf"))

    if on_progress:
        on_progress(100, f"Found {len(comments)} comments")

    return comments


def resolve_frameio_comment(
    comment_id: str,
    api_key: str,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Mark a Frame.io comment as resolved/completed.

    Args:
        comment_id: Frame.io comment ID.
        api_key: Frame.io API key/token.

    Returns:
        Dict with updated comment status.
    """
    if not api_key:
        raise ValueError("Frame.io API key is required")

    if on_progress:
        on_progress(30, "Resolving comment")

    _frameio_request(
        "PUT",
        f"/comments/{comment_id}",
        api_key,
        data={"completed": True},
    )

    if on_progress:
        on_progress(100, "Comment resolved")

    return {
        "comment_id": comment_id,
        "completed": True,
        "updated": True,
    }


def sync_frameio_comments(
    asset_id: str,
    api_key: str,
    local_comments: Optional[List[Dict]] = None,
    on_progress: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Two-way sync comments between Frame.io and local review.

    Args:
        asset_id: Frame.io asset ID.
        api_key: Frame.io API key/token.
        local_comments: List of local comment dicts to sync upstream.

    Returns:
        Dict with synced comments, new_remote count, new_local count.
    """
    if not api_key:
        raise ValueError("Frame.io API key is required")

    if on_progress:
        on_progress(20, "Fetching remote comments")

    remote_comments = get_frameio_comments(asset_id, api_key)
    local_comments = local_comments or []

    if on_progress:
        on_progress(50, "Syncing comments")

    remote_ids = {c.comment_id for c in remote_comments}
    local_ids = {c.get("comment_id", "") for c in local_comments}

    # Comments on Frame.io not yet locally
    new_remote = [c for c in remote_comments if c.comment_id not in local_ids]

    # Local comments not yet on Frame.io
    new_local = []
    for lc in local_comments:
        if lc.get("comment_id", "") not in remote_ids and lc.get("text"):
            # Post local comment to Frame.io
            try:
                post_data = {
                    "text": lc["text"],
                }
                if lc.get("timestamp") is not None:
                    post_data["timestamp"] = float(lc["timestamp"])

                result = _frameio_request(
                    "POST",
                    f"/assets/{asset_id}/comments",
                    api_key,
                    data=post_data,
                )
                new_local.append(result)
            except Exception as exc:
                logger.warning("Failed to sync local comment: %s", exc)

    if on_progress:
        on_progress(100, "Sync complete")

    return {
        "remote_comments": len(remote_comments),
        "new_from_remote": len(new_remote),
        "pushed_to_remote": len(new_local),
        "synced_comments": [
            {
                "comment_id": c.comment_id,
                "text": c.text,
                "author": c.author,
                "timestamp": c.timestamp,
                "completed": c.completed,
            }
            for c in remote_comments
        ],
    }
