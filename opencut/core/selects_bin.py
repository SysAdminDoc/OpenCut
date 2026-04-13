"""
OpenCut Selects Bin Module (14.2)

Rate clips 1-5 stars, apply custom tags, store in SQLite.
Filter/search by rating and tags. Export matching selects.
"""

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    get_video_info,
)

logger = logging.getLogger("opencut")

_DB_NAME = "selects.db"
_db_lock = threading.Lock()


def _get_db_path() -> str:
    """Return the path to the selects SQLite database."""
    from opencut.helpers import OPENCUT_DIR, _ensure_opencut_dir
    _ensure_opencut_dir()
    return os.path.join(OPENCUT_DIR, _DB_NAME)


def _init_db(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            clip_path TEXT PRIMARY KEY,
            rating    INTEGER DEFAULT 0,
            tags      TEXT DEFAULT '[]',
            notes     TEXT DEFAULT '',
            added_at  REAL DEFAULT (julianday('now')),
            duration  REAL DEFAULT 0,
            width     INTEGER DEFAULT 0,
            height    INTEGER DEFAULT 0,
            fps       REAL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_clips_rating ON clips(rating)
    """)
    conn.commit()


def _get_connection() -> sqlite3.Connection:
    """Open a connection and ensure tables exist."""
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ClipMetadata:
    """Metadata for a clip in the selects bin."""
    clip_path: str
    rating: int = 0
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0


@dataclass
class SelectsSearchResult:
    """Result of a selects search."""
    clips: List[ClipMetadata] = field(default_factory=list)
    total: int = 0
    filters_applied: Dict = field(default_factory=dict)


def rate_clip(
    clip_path: str,
    rating: int,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Rate a clip from 1-5 stars. Creates entry if it doesn't exist.

    Args:
        clip_path: Absolute path to the video clip.
        rating: Rating value (1-5). Use 0 to clear rating.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with clip_path, rating, and status.
    """
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"Clip not found: {clip_path}")
    if not isinstance(rating, int) or rating < 0 or rating > 5:
        raise ValueError("Rating must be an integer between 0 and 5")

    if on_progress:
        on_progress(10, "Updating clip rating...")

    # Probe video info for new entries
    info = get_video_info(clip_path)

    with _db_lock:
        conn = _get_connection()
        try:
            existing = conn.execute(
                "SELECT clip_path FROM clips WHERE clip_path = ?",
                (clip_path,),
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE clips SET rating = ? WHERE clip_path = ?",
                    (rating, clip_path),
                )
            else:
                conn.execute(
                    """INSERT INTO clips (clip_path, rating, duration, width, height, fps)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (clip_path, rating, info["duration"], info["width"],
                     info["height"], info["fps"]),
                )
            conn.commit()
        finally:
            conn.close()

    if on_progress:
        on_progress(100, f"Clip rated {rating} stars")

    return {"clip_path": clip_path, "rating": rating, "status": "ok"}


def tag_clip(
    clip_path: str,
    tags: List[str],
    mode: str = "set",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Apply custom tags to a clip.

    Args:
        clip_path: Absolute path to the video clip.
        tags: List of tag strings to apply.
        mode: 'set' replaces all tags, 'add' appends, 'remove' removes specified.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with clip_path, tags, and status.
    """
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"Clip not found: {clip_path}")
    if not isinstance(tags, list):
        raise ValueError("Tags must be a list of strings")

    # Normalize tags: strip whitespace, lowercase, remove empty
    clean_tags = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]

    if on_progress:
        on_progress(10, "Updating clip tags...")

    info = get_video_info(clip_path)

    with _db_lock:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT tags FROM clips WHERE clip_path = ?",
                (clip_path,),
            ).fetchone()

            if row:
                existing_tags = json.loads(row["tags"]) if row["tags"] else []
            else:
                existing_tags = []
                # Insert new entry
                conn.execute(
                    """INSERT INTO clips (clip_path, duration, width, height, fps)
                       VALUES (?, ?, ?, ?, ?)""",
                    (clip_path, info["duration"], info["width"],
                     info["height"], info["fps"]),
                )

            if mode == "add":
                final_tags = list(set(existing_tags + clean_tags))
            elif mode == "remove":
                final_tags = [t for t in existing_tags if t not in clean_tags]
            else:  # set
                final_tags = clean_tags

            final_tags.sort()

            conn.execute(
                "UPDATE clips SET tags = ? WHERE clip_path = ?",
                (json.dumps(final_tags), clip_path),
            )
            conn.commit()
        finally:
            conn.close()

    if on_progress:
        on_progress(100, f"Applied {len(final_tags)} tags")

    return {"clip_path": clip_path, "tags": final_tags, "status": "ok"}


def search_selects(
    filters: Optional[Dict] = None,
    on_progress: Optional[Callable] = None,
) -> SelectsSearchResult:
    """Search selects bin by rating and/or tags.

    Args:
        filters: Dict with optional keys:
            - min_rating (int): Minimum star rating (1-5).
            - max_rating (int): Maximum star rating (1-5).
            - tags (list): Clips must have ALL of these tags.
            - any_tags (list): Clips must have ANY of these tags.
            - search (str): Text search in clip path and notes.
            - limit (int): Max results (default 100).
            - offset (int): Pagination offset (default 0).
        on_progress: Progress callback(pct, msg).

    Returns:
        SelectsSearchResult with matching clips.
    """
    filters = filters or {}

    if on_progress:
        on_progress(10, "Searching selects bin...")

    conditions = []
    params = []

    min_rating = filters.get("min_rating")
    if min_rating is not None:
        conditions.append("rating >= ?")
        params.append(int(min_rating))

    max_rating = filters.get("max_rating")
    if max_rating is not None:
        conditions.append("rating <= ?")
        params.append(int(max_rating))

    search_text = filters.get("search")
    if search_text:
        conditions.append("(clip_path LIKE ? OR notes LIKE ?)")
        params.extend([f"%{search_text}%", f"%{search_text}%"])

    limit = min(int(filters.get("limit", 100)), 1000)
    offset = max(int(filters.get("offset", 0)), 0)

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = f"SELECT * FROM clips WHERE {where_clause} ORDER BY rating DESC, added_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    count_query = f"SELECT COUNT(*) as cnt FROM clips WHERE {where_clause}"
    count_params = params[:-2]  # exclude limit/offset

    with _db_lock:
        conn = _get_connection()
        try:
            rows = conn.execute(query, params).fetchall()
            total = conn.execute(count_query, count_params).fetchone()["cnt"]
        finally:
            conn.close()

    if on_progress:
        on_progress(50, "Filtering by tags...")

    clips = []
    required_tags = filters.get("tags", [])
    any_tags = filters.get("any_tags", [])

    for row in rows:
        row_tags = json.loads(row["tags"]) if row["tags"] else []

        # Filter by required tags (all must match)
        if required_tags:
            if not all(t.lower() in row_tags for t in required_tags):
                continue

        # Filter by any_tags (at least one must match)
        if any_tags:
            if not any(t.lower() in row_tags for t in any_tags):
                continue

        clips.append(ClipMetadata(
            clip_path=row["clip_path"],
            rating=row["rating"],
            tags=row_tags,
            notes=row["notes"] or "",
            duration=row["duration"] or 0.0,
            width=row["width"] or 0,
            height=row["height"] or 0,
            fps=row["fps"] or 0.0,
        ))

    if on_progress:
        on_progress(100, f"Found {len(clips)} clips")

    return SelectsSearchResult(
        clips=clips,
        total=total,
        filters_applied=filters,
    )


def get_clip_metadata(
    clip_path: str,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Get metadata for a clip in the selects bin.

    Args:
        clip_path: Absolute path to the video clip.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with clip metadata or None if not in bin.
    """
    if on_progress:
        on_progress(10, "Looking up clip metadata...")

    with _db_lock:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM clips WHERE clip_path = ?",
                (clip_path,),
            ).fetchone()
        finally:
            conn.close()

    if not row:
        # Probe from disk if not in DB
        if os.path.isfile(clip_path):
            info = get_video_info(clip_path)
            result = {
                "clip_path": clip_path,
                "rating": 0,
                "tags": [],
                "notes": "",
                "in_bin": False,
                "duration": info["duration"],
                "width": info["width"],
                "height": info["height"],
                "fps": info["fps"],
            }
        else:
            result = {"clip_path": clip_path, "in_bin": False, "error": "File not found"}

        if on_progress:
            on_progress(100, "Clip not in selects bin")
        return result

    tags = json.loads(row["tags"]) if row["tags"] else []

    if on_progress:
        on_progress(100, "Metadata retrieved")

    return {
        "clip_path": row["clip_path"],
        "rating": row["rating"],
        "tags": tags,
        "notes": row["notes"] or "",
        "in_bin": True,
        "duration": row["duration"] or 0.0,
        "width": row["width"] or 0,
        "height": row["height"] or 0,
        "fps": row["fps"] or 0.0,
    }


def export_selects(
    filters: Optional[Dict] = None,
    output_path_str: Optional[str] = None,
    format: str = "json",
    on_progress: Optional[Callable] = None,
) -> dict:
    """Export clips matching filters to a JSON or CSV manifest.

    Args:
        filters: Same filter dict as search_selects().
        output_path_str: Output file path. Auto-generated if None.
        format: 'json' or 'csv'.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, clip_count, and format.
    """
    if on_progress:
        on_progress(5, "Searching for clips to export...")

    result = search_selects(filters)
    clips = result.clips

    if not clips:
        raise ValueError("No clips match the given filters")

    if on_progress:
        on_progress(40, f"Exporting {len(clips)} clips...")

    if output_path_str is None:
        from opencut.helpers import OPENCUT_DIR, _ensure_opencut_dir
        _ensure_opencut_dir()
        ext = ".csv" if format == "csv" else ".json"
        output_path_str = os.path.join(OPENCUT_DIR, f"selects_export{ext}")

    if format == "csv":
        import csv
        with open(output_path_str, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["clip_path", "rating", "tags", "notes",
                             "duration", "width", "height", "fps"])
            for clip in clips:
                writer.writerow([
                    clip.clip_path, clip.rating, ";".join(clip.tags),
                    clip.notes, clip.duration, clip.width, clip.height, clip.fps,
                ])
    else:
        data = []
        for clip in clips:
            data.append({
                "clip_path": clip.clip_path,
                "rating": clip.rating,
                "tags": clip.tags,
                "notes": clip.notes,
                "duration": clip.duration,
                "width": clip.width,
                "height": clip.height,
                "fps": clip.fps,
            })
        with open(output_path_str, "w", encoding="utf-8") as f:
            json.dump({"selects": data, "count": len(data),
                        "filters": filters or {}}, f, indent=2)

    if on_progress:
        on_progress(100, "Export complete")

    return {
        "output_path": output_path_str,
        "clip_count": len(clips),
        "format": format,
    }
