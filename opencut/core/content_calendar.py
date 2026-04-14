"""
OpenCut Cross-Platform Content Calendar

Take extracted clips list, apply platform posting frequency rules,
distribute across a calendar, and export as CSV or iCal (.ics).
"""

import csv
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ContentItem:
    """A single content item to schedule."""
    title: str = ""
    description: str = ""
    platform: str = ""
    clip_path: str = ""
    duration: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class ScheduledPost:
    """A content item assigned to a specific date/time."""
    date: str = ""
    time: str = ""
    platform: str = ""
    title: str = ""
    description: str = ""
    clip_path: str = ""
    status: str = "scheduled"


@dataclass
class ContentCalendarResult:
    """Result from content calendar generation."""
    scheduled_posts: List[ScheduledPost] = field(default_factory=list)
    csv_path: str = ""
    ics_path: str = ""
    total_posts: int = 0
    date_range: str = ""
    platform_breakdown: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Platform posting rules
# ---------------------------------------------------------------------------
_PLATFORM_RULES = {
    "youtube": {
        "posts_per_week": 2,
        "best_days": ["tuesday", "thursday", "saturday"],
        "best_times": ["09:00", "14:00", "17:00"],
        "min_gap_hours": 48,
    },
    "tiktok": {
        "posts_per_week": 5,
        "best_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
        "best_times": ["07:00", "12:00", "19:00", "21:00"],
        "min_gap_hours": 12,
    },
    "instagram": {
        "posts_per_week": 3,
        "best_days": ["monday", "wednesday", "friday", "saturday"],
        "best_times": ["08:00", "12:00", "17:00", "20:00"],
        "min_gap_hours": 24,
    },
    "twitter": {
        "posts_per_week": 7,
        "best_days": ["monday", "tuesday", "wednesday", "thursday", "friday",
                       "saturday", "sunday"],
        "best_times": ["08:00", "12:00", "17:00"],
        "min_gap_hours": 8,
    },
    "linkedin": {
        "posts_per_week": 2,
        "best_days": ["tuesday", "wednesday", "thursday"],
        "best_times": ["08:00", "10:00", "12:00"],
        "min_gap_hours": 48,
    },
    "facebook": {
        "posts_per_week": 3,
        "best_days": ["wednesday", "thursday", "friday"],
        "best_times": ["09:00", "13:00", "16:00"],
        "min_gap_hours": 24,
    },
}

_DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday",
              "saturday", "sunday"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _distribute_posts(
    items: List[ContentItem],
    platforms: List[str],
    start_date: datetime,
    weeks: int = 4,
) -> List[ScheduledPost]:
    """Distribute content items across the calendar based on platform rules."""
    scheduled = []
    platform_queues: Dict[str, List[ContentItem]] = {}

    # Group items by platform
    for item in items:
        p = item.platform.lower() if item.platform else platforms[0] if platforms else "youtube"
        if p not in platform_queues:
            platform_queues[p] = []
        platform_queues[p].append(item)

    # If items have no platform, distribute evenly
    unassigned = [it for it in items if not it.platform]
    if unassigned and platforms:
        per_platform = max(1, len(unassigned) // len(platforms))
        for i, item in enumerate(unassigned):
            p = platforms[i % len(platforms)]
            if p not in platform_queues:
                platform_queues[p] = []
            platform_queues[p].append(item)

    # Schedule each platform
    for platform, queue in platform_queues.items():
        rules = _PLATFORM_RULES.get(platform, _PLATFORM_RULES["youtube"])
        best_days = rules["best_days"]
        best_times = rules["best_times"]
        posts_per_week = rules["posts_per_week"]

        item_idx = 0
        for week in range(weeks):
            week_posts = 0
            for day_offset in range(7):
                current_date = start_date + timedelta(days=week * 7 + day_offset)
                day_name = _DAY_NAMES[current_date.weekday()]

                if day_name not in best_days:
                    continue
                if week_posts >= posts_per_week:
                    break
                if item_idx >= len(queue):
                    break

                time_str = best_times[week_posts % len(best_times)]
                item = queue[item_idx]

                scheduled.append(ScheduledPost(
                    date=current_date.strftime("%Y-%m-%d"),
                    time=time_str,
                    platform=platform,
                    title=item.title,
                    description=item.description,
                    clip_path=item.clip_path,
                    status="scheduled",
                ))

                item_idx += 1
                week_posts += 1

    # Sort by date/time
    scheduled.sort(key=lambda p: (p.date, p.time))
    return scheduled


def _export_csv(posts: List[ScheduledPost], output_path: str) -> str:
    """Export scheduled posts to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Date", "Time", "Platform", "Title", "Description",
            "Clip Path", "Status",
        ])
        for post in posts:
            writer.writerow([
                post.date, post.time, post.platform, post.title,
                post.description, post.clip_path, post.status,
            ])
    return output_path


def _export_ics(posts: List[ScheduledPost], output_path: str) -> str:
    """Export scheduled posts as iCal (.ics) file."""
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//OpenCut//Content Calendar//EN",
        "CALSCALE:GREGORIAN",
    ]

    for i, post in enumerate(posts):
        # Parse date and time
        try:
            dt = datetime.strptime(f"{post.date} {post.time}", "%Y-%m-%d %H:%M")
        except ValueError:
            continue

        dtstart = dt.strftime("%Y%m%dT%H%M%S")
        dtend = (dt + timedelta(minutes=30)).strftime("%Y%m%dT%H%M%S")
        uid = f"opencut-{i}-{dt.strftime('%Y%m%d%H%M%S')}@opencut.local"

        summary = f"[{post.platform.upper()}] {post.title}"
        description = post.description.replace("\n", "\\n")

        lines.extend([
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTART:{dtstart}",
            f"DTEND:{dtend}",
            f"SUMMARY:{summary}",
            f"DESCRIPTION:{description}",
            f"CATEGORIES:{post.platform.upper()}",
            f"STATUS:CONFIRMED",
            "END:VEVENT",
        ])

    lines.append("END:VCALENDAR")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\r\n".join(lines) + "\r\n")

    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_platform_rules() -> Dict[str, dict]:
    """Return platform posting frequency rules."""
    return {k: dict(v) for k, v in _PLATFORM_RULES.items()}


def generate_content_calendar(
    clips: List[dict],
    platforms: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    weeks: int = 4,
    output_format: str = "both",
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> ContentCalendarResult:
    """
    Generate a cross-platform content calendar.

    Args:
        clips: List of clip dicts with title, description, clip_path, platform, duration.
        platforms: Target platforms. Defaults to all available.
        start_date: Start date as YYYY-MM-DD. Defaults to tomorrow.
        weeks: Number of weeks to schedule.
        output_format: "csv", "ics", or "both".
        output_dir: Output directory. Defaults to current directory.
        on_progress: Progress callback(pct, msg).

    Returns:
        ContentCalendarResult with scheduled posts and export paths.
    """
    if not clips:
        raise ValueError("At least one clip is required")

    if platforms is None:
        platforms = list(_PLATFORM_RULES.keys())
    platforms = [p.lower() for p in platforms]

    weeks = max(1, min(52, weeks))

    # Parse start date
    if start_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {start_date}. Use YYYY-MM-DD.")
    else:
        start_dt = datetime.now() + timedelta(days=1)
        start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    if not output_dir:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if on_progress:
        on_progress(10, "Building content items...")

    # Build content items
    items = []
    for clip in clips:
        items.append(ContentItem(
            title=str(clip.get("title", "")),
            description=str(clip.get("description", "")),
            platform=str(clip.get("platform", "")),
            clip_path=str(clip.get("clip_path", "")),
            duration=float(clip.get("duration", 0)),
            tags=clip.get("tags", []),
        ))

    if on_progress:
        on_progress(30, "Distributing posts across calendar...")

    # Distribute
    scheduled = _distribute_posts(items, platforms, start_dt, weeks)

    if on_progress:
        on_progress(60, "Exporting calendar...")

    result = ContentCalendarResult(
        scheduled_posts=scheduled,
        total_posts=len(scheduled),
    )

    # Date range
    if scheduled:
        result.date_range = f"{scheduled[0].date} to {scheduled[-1].date}"

    # Platform breakdown
    breakdown: Dict[str, int] = {}
    for post in scheduled:
        breakdown[post.platform] = breakdown.get(post.platform, 0) + 1
    result.platform_breakdown = breakdown

    # Export CSV
    if output_format in ("csv", "both"):
        csv_path = os.path.join(output_dir, "content_calendar.csv")
        _export_csv(scheduled, csv_path)
        result.csv_path = csv_path

    # Export iCal
    if output_format in ("ics", "both"):
        ics_path = os.path.join(output_dir, "content_calendar.ics")
        _export_ics(scheduled, ics_path)
        result.ics_path = ics_path

    if on_progress:
        on_progress(100, "Content calendar generated")

    return result
