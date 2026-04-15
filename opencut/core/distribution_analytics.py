"""
OpenCut Distribution Analytics

Post-publish analytics aggregation and insights.  Store per-video publish
records, track manually entered metrics (views, likes, comments, shares,
watch_time_avg, CTR), compute cross-platform aggregates, per-video
comparison, content type analysis, and export as CSV/JSON.
"""

import csv
import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DB_PATH = os.path.join(OPENCUT_DIR, "analytics.db")
_db_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PublishRecord:
    """A single video publish record."""
    id: int = 0
    video_title: str = ""
    platform: str = ""
    publish_date: str = ""
    url: str = ""
    file_path: str = ""
    resolution: str = ""
    duration_sec: float = 0.0
    file_size_mb: float = 0.0
    category: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "video_title": self.video_title,
            "platform": self.platform,
            "publish_date": self.publish_date,
            "url": self.url,
            "file_path": self.file_path,
            "resolution": self.resolution,
            "duration_sec": round(self.duration_sec, 2),
            "file_size_mb": round(self.file_size_mb, 2),
            "category": self.category,
            "tags": list(self.tags),
        }


@dataclass
class MetricsEntry:
    """Metrics for a specific publish record at a point in time."""
    record_id: int = 0
    recorded_at: str = ""
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    watch_time_avg: float = 0.0
    ctr: float = 0.0

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "recorded_at": self.recorded_at,
            "views": self.views,
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "watch_time_avg": round(self.watch_time_avg, 2),
            "ctr": round(self.ctr, 4),
        }


@dataclass
class PlatformStats:
    """Aggregated stats for a single platform."""
    platform: str = ""
    total_videos: int = 0
    total_views: int = 0
    total_likes: int = 0
    total_comments: int = 0
    total_shares: int = 0
    avg_views: float = 0.0
    avg_engagement_rate: float = 0.0
    avg_ctr: float = 0.0

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "total_videos": self.total_videos,
            "total_views": self.total_views,
            "total_likes": self.total_likes,
            "total_comments": self.total_comments,
            "total_shares": self.total_shares,
            "avg_views": round(self.avg_views, 1),
            "avg_engagement_rate": round(self.avg_engagement_rate, 4),
            "avg_ctr": round(self.avg_ctr, 4),
        }


@dataclass
class AnalyticsReport:
    """Full analytics report across all platforms."""
    per_platform_stats: List[PlatformStats] = field(default_factory=list)
    top_performing: List[dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    total_reach: int = 0
    best_platform: str = ""
    overall_engagement_rate: float = 0.0
    growth_trend: str = ""
    total_videos: int = 0
    date_range: str = ""

    def to_dict(self) -> dict:
        return {
            "per_platform_stats": [s.to_dict() for s in self.per_platform_stats],
            "top_performing": list(self.top_performing),
            "recommendations": list(self.recommendations),
            "total_reach": self.total_reach,
            "best_platform": self.best_platform,
            "overall_engagement_rate": round(self.overall_engagement_rate, 4),
            "growth_trend": self.growth_trend,
            "total_videos": self.total_videos,
            "date_range": self.date_range,
        }


# ---------------------------------------------------------------------------
# SQLite database
# ---------------------------------------------------------------------------
def _get_db_connection() -> sqlite3.Connection:
    """Get a thread-local SQLite connection with WAL mode."""
    os.makedirs(OPENCUT_DIR, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """Initialize the analytics database schema."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS publish_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_title TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    publish_date TEXT DEFAULT (date('now')),
                    url TEXT DEFAULT '',
                    file_path TEXT DEFAULT '',
                    resolution TEXT DEFAULT '',
                    duration_sec REAL DEFAULT 0,
                    file_size_mb REAL DEFAULT 0,
                    category TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id INTEGER NOT NULL,
                    recorded_at TEXT DEFAULT (datetime('now')),
                    views INTEGER DEFAULT 0,
                    likes INTEGER DEFAULT 0,
                    comments INTEGER DEFAULT 0,
                    shares INTEGER DEFAULT 0,
                    watch_time_avg REAL DEFAULT 0,
                    ctr REAL DEFAULT 0,
                    FOREIGN KEY (record_id) REFERENCES publish_records(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_pr_platform ON publish_records(platform);
                CREATE INDEX IF NOT EXISTS idx_pr_date ON publish_records(publish_date);
                CREATE INDEX IF NOT EXISTS idx_metrics_record ON metrics(record_id);
            """)
            conn.commit()
        finally:
            conn.close()


# Initialize on import
_init_db()


# ---------------------------------------------------------------------------
# Publish record CRUD
# ---------------------------------------------------------------------------
def add_publish_record(record: PublishRecord) -> int:
    """Add a publish record. Returns the new record ID."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            cur = conn.execute(
                """INSERT INTO publish_records
                   (video_title, platform, publish_date, url, file_path,
                    resolution, duration_sec, file_size_mb, category, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record.video_title, record.platform, record.publish_date,
                 record.url, record.file_path, record.resolution,
                 record.duration_sec, record.file_size_mb, record.category,
                 json.dumps(record.tags)),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_publish_record(record_id: int) -> Optional[PublishRecord]:
    """Get a publish record by ID."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            row = conn.execute(
                "SELECT * FROM publish_records WHERE id = ?", (record_id,)
            ).fetchone()
            if not row:
                return None
            return _row_to_record(row)
        finally:
            conn.close()


def list_publish_records(platform: str = "",
                         limit: int = 100,
                         offset: int = 0) -> List[PublishRecord]:
    """List publish records, optionally filtered by platform."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            if platform:
                rows = conn.execute(
                    "SELECT * FROM publish_records WHERE platform = ? "
                    "ORDER BY publish_date DESC LIMIT ? OFFSET ?",
                    (platform, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM publish_records "
                    "ORDER BY publish_date DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            return [_row_to_record(r) for r in rows]
        finally:
            conn.close()


def delete_publish_record(record_id: int) -> bool:
    """Delete a publish record and its metrics."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            conn.execute("DELETE FROM metrics WHERE record_id = ?", (record_id,))
            cur = conn.execute("DELETE FROM publish_records WHERE id = ?", (record_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def _row_to_record(row) -> PublishRecord:
    """Convert a database row to a PublishRecord."""
    tags = []
    try:
        tags = json.loads(row["tags"] or "[]")
    except (json.JSONDecodeError, TypeError):
        pass
    return PublishRecord(
        id=row["id"],
        video_title=row["video_title"],
        platform=row["platform"],
        publish_date=row["publish_date"] or "",
        url=row["url"] or "",
        file_path=row["file_path"] or "",
        resolution=row["resolution"] or "",
        duration_sec=float(row["duration_sec"] or 0),
        file_size_mb=float(row["file_size_mb"] or 0),
        category=row["category"] or "",
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Metrics CRUD
# ---------------------------------------------------------------------------
def record_metrics(entry: MetricsEntry) -> int:
    """Record metrics for a publish record. Returns the metric entry ID."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            # Verify record exists
            exists = conn.execute(
                "SELECT 1 FROM publish_records WHERE id = ?", (entry.record_id,)
            ).fetchone()
            if not exists:
                raise ValueError(f"Publish record {entry.record_id} not found")

            cur = conn.execute(
                """INSERT INTO metrics
                   (record_id, views, likes, comments, shares, watch_time_avg, ctr)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (entry.record_id, entry.views, entry.likes, entry.comments,
                 entry.shares, entry.watch_time_avg, entry.ctr),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_latest_metrics(record_id: int) -> Optional[MetricsEntry]:
    """Get the most recent metrics for a publish record."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            row = conn.execute(
                "SELECT * FROM metrics WHERE record_id = ? "
                "ORDER BY recorded_at DESC LIMIT 1",
                (record_id,),
            ).fetchone()
            if not row:
                return None
            return _row_to_metrics(row)
        finally:
            conn.close()


def get_metrics_history(record_id: int) -> List[MetricsEntry]:
    """Get all metrics entries for a publish record."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM metrics WHERE record_id = ? ORDER BY recorded_at ASC",
                (record_id,),
            ).fetchall()
            return [_row_to_metrics(r) for r in rows]
        finally:
            conn.close()


def _row_to_metrics(row) -> MetricsEntry:
    """Convert a database row to a MetricsEntry."""
    return MetricsEntry(
        record_id=row["record_id"],
        recorded_at=row["recorded_at"] or "",
        views=int(row["views"] or 0),
        likes=int(row["likes"] or 0),
        comments=int(row["comments"] or 0),
        shares=int(row["shares"] or 0),
        watch_time_avg=float(row["watch_time_avg"] or 0),
        ctr=float(row["ctr"] or 0),
    )


# ---------------------------------------------------------------------------
# Analytics aggregation
# ---------------------------------------------------------------------------
def _compute_engagement_rate(views: int, likes: int, comments: int, shares: int) -> float:
    """Compute engagement rate as (likes + comments + shares) / views."""
    if views <= 0:
        return 0.0
    return (likes + comments + shares) / views


def compute_platform_stats(platform: str = "") -> List[PlatformStats]:
    """Compute aggregated stats per platform."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            if platform:
                platforms = [platform]
            else:
                rows = conn.execute(
                    "SELECT DISTINCT platform FROM publish_records"
                ).fetchall()
                platforms = [r["platform"] for r in rows]

            stats_list = []
            for plat in platforms:
                records = conn.execute(
                    "SELECT id FROM publish_records WHERE platform = ?", (plat,)
                ).fetchall()
                record_ids = [r["id"] for r in records]
                if not record_ids:
                    continue

                placeholders = ",".join("?" * len(record_ids))
                latest_metrics = conn.execute(
                    f"""SELECT m.* FROM metrics m
                        INNER JOIN (
                            SELECT record_id, MAX(recorded_at) as max_at
                            FROM metrics WHERE record_id IN ({placeholders})
                            GROUP BY record_id
                        ) latest ON m.record_id = latest.record_id
                        AND m.recorded_at = latest.max_at""",
                    record_ids,
                ).fetchall()

                total_views = sum(int(m["views"] or 0) for m in latest_metrics)
                total_likes = sum(int(m["likes"] or 0) for m in latest_metrics)
                total_comments = sum(int(m["comments"] or 0) for m in latest_metrics)
                total_shares = sum(int(m["shares"] or 0) for m in latest_metrics)

                num_videos = len(record_ids)
                avg_views = total_views / max(num_videos, 1)
                avg_engagement = _compute_engagement_rate(
                    total_views, total_likes, total_comments, total_shares)
                avg_ctr = (sum(float(m["ctr"] or 0) for m in latest_metrics)
                           / max(len(latest_metrics), 1))

                stats_list.append(PlatformStats(
                    platform=plat,
                    total_videos=num_videos,
                    total_views=total_views,
                    total_likes=total_likes,
                    total_comments=total_comments,
                    total_shares=total_shares,
                    avg_views=avg_views,
                    avg_engagement_rate=avg_engagement,
                    avg_ctr=avg_ctr,
                ))

            return stats_list
        finally:
            conn.close()


def _compute_growth_trend(days: int = 30) -> str:
    """Compute growth trend over the last N days."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            midpoint = (datetime.now() - timedelta(days=days // 2)).strftime("%Y-%m-%d")

            first_half = conn.execute(
                "SELECT COALESCE(SUM(m.views), 0) as total FROM metrics m "
                "INNER JOIN publish_records p ON m.record_id = p.id "
                "WHERE m.recorded_at >= ? AND m.recorded_at < ?",
                (cutoff, midpoint),
            ).fetchone()

            second_half = conn.execute(
                "SELECT COALESCE(SUM(m.views), 0) as total FROM metrics m "
                "INNER JOIN publish_records p ON m.record_id = p.id "
                "WHERE m.recorded_at >= ?",
                (midpoint,),
            ).fetchone()

            first_total = int(first_half["total"]) if first_half else 0
            second_total = int(second_half["total"]) if second_half else 0

            if first_total == 0 and second_total == 0:
                return "no_data"
            if first_total == 0:
                return "growing"
            ratio = second_total / first_total
            if ratio > 1.2:
                return "growing"
            elif ratio < 0.8:
                return "declining"
            return "stable"
        finally:
            conn.close()


def _get_top_performing(limit: int = 10) -> List[dict]:
    """Get top performing videos by total engagement."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            rows = conn.execute(
                """SELECT p.id, p.video_title, p.platform,
                          COALESCE(m.views, 0) as views,
                          COALESCE(m.likes, 0) as likes,
                          COALESCE(m.comments, 0) as comments,
                          COALESCE(m.shares, 0) as shares
                   FROM publish_records p
                   LEFT JOIN (
                       SELECT record_id, views, likes, comments, shares,
                              ROW_NUMBER() OVER (PARTITION BY record_id ORDER BY recorded_at DESC) as rn
                       FROM metrics
                   ) m ON p.id = m.record_id AND m.rn = 1
                   ORDER BY (COALESCE(m.views, 0) + COALESCE(m.likes, 0) * 10 +
                             COALESCE(m.comments, 0) * 20 + COALESCE(m.shares, 0) * 30) DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

            result = []
            for r in rows:
                views = int(r["views"] or 0)
                likes = int(r["likes"] or 0)
                comments = int(r["comments"] or 0)
                shares = int(r["shares"] or 0)
                result.append({
                    "id": r["id"],
                    "video_title": r["video_title"],
                    "platform": r["platform"],
                    "views": views,
                    "likes": likes,
                    "comments": comments,
                    "shares": shares,
                    "engagement_rate": round(
                        _compute_engagement_rate(views, likes, comments, shares), 4),
                })
            return result
        finally:
            conn.close()


def _generate_recommendations(stats: List[PlatformStats]) -> List[str]:
    """Generate actionable recommendations from analytics data."""
    recs = []
    if not stats:
        recs.append("Start publishing content to build your analytics baseline.")
        return recs

    # Find best and worst platforms
    by_engagement = sorted(stats, key=lambda s: s.avg_engagement_rate, reverse=True)
    if len(by_engagement) >= 2:
        best = by_engagement[0]
        worst = by_engagement[-1]
        if best.avg_engagement_rate > worst.avg_engagement_rate * 2:
            recs.append(
                f"Focus more on {best.platform} where engagement is "
                f"{best.avg_engagement_rate:.1%} vs {worst.platform} at "
                f"{worst.avg_engagement_rate:.1%}."
            )

    by_views = sorted(stats, key=lambda s: s.avg_views, reverse=True)
    if by_views:
        top = by_views[0]
        if top.total_views > 0:
            recs.append(f"Your best reach is on {top.platform} with "
                        f"{top.avg_views:.0f} avg views per video.")

    for s in stats:
        if s.avg_ctr > 0 and s.avg_ctr < 0.02:
            recs.append(f"CTR on {s.platform} is low ({s.avg_ctr:.1%}). "
                        f"Try improving thumbnails and titles.")
        if s.total_videos < 5:
            recs.append(f"Post more on {s.platform} ({s.total_videos} videos) "
                        f"to gather meaningful data.")

    return recs[:8]


def generate_report(days: int = 30) -> AnalyticsReport:
    """Generate a comprehensive analytics report."""
    report = AnalyticsReport()

    stats = compute_platform_stats()
    report.per_platform_stats = stats
    report.total_videos = sum(s.total_videos for s in stats)
    report.total_reach = sum(s.total_views for s in stats)

    if stats:
        best = max(stats, key=lambda s: s.avg_engagement_rate)
        report.best_platform = best.platform
        total_views = sum(s.total_views for s in stats)
        total_likes = sum(s.total_likes for s in stats)
        total_comments = sum(s.total_comments for s in stats)
        total_shares = sum(s.total_shares for s in stats)
        report.overall_engagement_rate = _compute_engagement_rate(
            total_views, total_likes, total_comments, total_shares)

    report.growth_trend = _compute_growth_trend(days)
    report.top_performing = _get_top_performing()
    report.recommendations = _generate_recommendations(stats)
    report.date_range = f"Last {days} days"

    return report


# ---------------------------------------------------------------------------
# Content type analysis
# ---------------------------------------------------------------------------
def content_type_analysis() -> List[dict]:
    """Analyze which content categories get the best engagement."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            rows = conn.execute(
                """SELECT p.category, COUNT(DISTINCT p.id) as video_count,
                          COALESCE(AVG(m.views), 0) as avg_views,
                          COALESCE(AVG(m.likes), 0) as avg_likes,
                          COALESCE(AVG(m.comments), 0) as avg_comments,
                          COALESCE(AVG(m.shares), 0) as avg_shares
                   FROM publish_records p
                   LEFT JOIN (
                       SELECT record_id, views, likes, comments, shares,
                              ROW_NUMBER() OVER (PARTITION BY record_id ORDER BY recorded_at DESC) as rn
                       FROM metrics
                   ) m ON p.id = m.record_id AND m.rn = 1
                   WHERE p.category != ''
                   GROUP BY p.category
                   ORDER BY avg_views DESC"""
            ).fetchall()

            results = []
            for r in rows:
                avg_views = float(r["avg_views"] or 0)
                avg_likes = float(r["avg_likes"] or 0)
                avg_comments = float(r["avg_comments"] or 0)
                avg_shares = float(r["avg_shares"] or 0)
                results.append({
                    "category": r["category"],
                    "video_count": int(r["video_count"]),
                    "avg_views": round(avg_views, 1),
                    "avg_likes": round(avg_likes, 1),
                    "avg_comments": round(avg_comments, 1),
                    "avg_shares": round(avg_shares, 1),
                    "engagement_rate": round(
                        _compute_engagement_rate(
                            int(avg_views), int(avg_likes),
                            int(avg_comments), int(avg_shares)), 4),
                })
            return results
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_csv(output_path: str = "") -> str:
    """Export all analytics data as CSV. Returns file path."""
    if not output_path:
        output_path = os.path.join(OPENCUT_DIR, "analytics_export.csv")

    records = list_publish_records(limit=10000)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID", "Title", "Platform", "Publish Date", "URL", "Category",
            "Duration (s)", "File Size (MB)", "Views", "Likes", "Comments",
            "Shares", "Watch Time Avg", "CTR", "Engagement Rate",
        ])
        for rec in records:
            metrics = get_latest_metrics(rec.id)
            views = metrics.views if metrics else 0
            likes = metrics.likes if metrics else 0
            comments = metrics.comments if metrics else 0
            shares = metrics.shares if metrics else 0
            wta = metrics.watch_time_avg if metrics else 0
            ctr = metrics.ctr if metrics else 0
            eng_rate = _compute_engagement_rate(views, likes, comments, shares)
            writer.writerow([
                rec.id, rec.video_title, rec.platform, rec.publish_date,
                rec.url, rec.category, rec.duration_sec, rec.file_size_mb,
                views, likes, comments, shares, wta, ctr, round(eng_rate, 4),
            ])

    return output_path


def export_json(output_path: str = "") -> str:
    """Export all analytics data as JSON. Returns file path."""
    if not output_path:
        output_path = os.path.join(OPENCUT_DIR, "analytics_export.json")

    report = generate_report()
    records = list_publish_records(limit=10000)

    data = {
        "report": report.to_dict(),
        "records": [r.to_dict() for r in records],
        "content_analysis": content_type_analysis(),
        "exported_at": datetime.now().isoformat(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return output_path


def get_record_count() -> int:
    """Return total number of publish records."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM publish_records").fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()
