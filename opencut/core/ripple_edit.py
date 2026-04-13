"""
OpenCut Ripple Edit Automation

After cuts, find gaps in the timeline and close them by shifting
subsequent items earlier.  Supports locked tracks that are not
affected by the ripple.
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("opencut")

# Minimum gap (seconds) to consider for closing -- smaller gaps are
# likely just rounding artifacts.
MIN_GAP_THRESHOLD = 0.001


@dataclass
class TimelineGap:
    """A detected gap in the timeline."""
    index: int          # Position in the timeline (after this item)
    start: float        # Gap start time (seconds)
    end: float          # Gap end time (seconds)
    duration: float     # Gap duration (seconds)
    track: str = ""     # Track identifier, if available


@dataclass
class GapDetectionResult:
    """Result of gap detection."""
    gaps: List[TimelineGap]
    total_gap_duration: float
    timeline_duration: float
    item_count: int


@dataclass
class RippleResult:
    """Result of ripple-close operation."""
    items: List[Dict[str, Any]]
    gaps_closed: int
    total_shift: float
    locked_tracks_skipped: List[str]


def detect_gaps(
    timeline_items: List[Dict[str, Any]],
    min_gap: float = MIN_GAP_THRESHOLD,
    on_progress: Optional[Callable] = None,
) -> GapDetectionResult:
    """Detect gaps between timeline items.

    Args:
        timeline_items: List of dicts, each with:
            - 'start': Start time in seconds on the timeline.
            - 'end': End time in seconds on the timeline.
            Or alternatively:
            - 'start': Start time in seconds.
            - 'duration': Duration in seconds.
            Optionally:
            - 'track': Track identifier.
        min_gap: Minimum gap duration to report (filters noise).
        on_progress: Optional progress callback.

    Returns:
        GapDetectionResult with list of gaps found.
    """
    if on_progress:
        on_progress(5, "Analyzing timeline for gaps...")

    if not timeline_items:
        return GapDetectionResult(
            gaps=[],
            total_gap_duration=0.0,
            timeline_duration=0.0,
            item_count=0,
        )

    # Normalize items: ensure 'end' is present
    normalized = []
    for item in timeline_items:
        start = float(item.get("start", 0))
        end = item.get("end")
        if end is not None:
            end = float(end)
        else:
            duration = float(item.get("duration", 0))
            end = start + duration
        normalized.append({
            "start": start,
            "end": end,
            "track": item.get("track", ""),
            "_original": item,
        })

    # Sort by start time
    normalized.sort(key=lambda x: x["start"])

    gaps: List[TimelineGap] = []
    total_gap = 0.0

    # Check for initial gap (if first item doesn't start at 0)
    if normalized[0]["start"] > min_gap:
        gap = TimelineGap(
            index=-1,
            start=0.0,
            end=normalized[0]["start"],
            duration=normalized[0]["start"],
            track=normalized[0]["track"],
        )
        gaps.append(gap)
        total_gap += gap.duration

    # Check gaps between consecutive items
    for i in range(len(normalized) - 1):
        if on_progress and len(normalized) > 10:
            pct = 5 + int((i / (len(normalized) - 1)) * 90)
            on_progress(pct, f"Checking gap {i + 1}/{len(normalized) - 1}...")

        current_end = normalized[i]["end"]
        next_start = normalized[i + 1]["start"]

        gap_duration = next_start - current_end
        if gap_duration > min_gap:
            gap = TimelineGap(
                index=i,
                start=current_end,
                end=next_start,
                duration=gap_duration,
                track=normalized[i].get("track", ""),
            )
            gaps.append(gap)
            total_gap += gap_duration

    # Calculate total timeline duration
    if normalized:
        timeline_end = max(n["end"] for n in normalized)
    else:
        timeline_end = 0.0

    if on_progress:
        on_progress(100, f"Found {len(gaps)} gap(s)")

    logger.info(
        "Detected %d gaps (%.3fs total) in %d timeline items",
        len(gaps), total_gap, len(timeline_items),
    )
    return GapDetectionResult(
        gaps=gaps,
        total_gap_duration=total_gap,
        timeline_duration=timeline_end,
        item_count=len(timeline_items),
    )


def ripple_close_gaps(
    timeline_items: List[Dict[str, Any]],
    locked_tracks: Optional[List[str]] = None,
    min_gap: float = MIN_GAP_THRESHOLD,
    on_progress: Optional[Callable] = None,
) -> RippleResult:
    """Close gaps in the timeline by shifting items earlier.

    Items on locked tracks are not shifted.  All unlocked items after
    each gap are shifted left by the gap duration.

    Args:
        timeline_items: List of timeline item dicts (same format as
                        detect_gaps).
        locked_tracks: List of track identifiers that should not be
                       shifted.  Items on these tracks are preserved
                       in their original positions.
        min_gap: Minimum gap to close.
        on_progress: Optional progress callback.

    Returns:
        RippleResult with the modified timeline items and stats.
    """
    if on_progress:
        on_progress(5, "Preparing ripple close...")

    locked: Set[str] = set(locked_tracks or [])

    if not timeline_items:
        return RippleResult(
            items=[],
            gaps_closed=0,
            total_shift=0.0,
            locked_tracks_skipped=sorted(locked),
        )

    # Deep copy to avoid mutating input
    items = []
    for item in timeline_items:
        new_item = copy.deepcopy(item)
        start = float(new_item.get("start", 0))
        end = new_item.get("end")
        if end is not None:
            end = float(end)
        else:
            duration = float(new_item.get("duration", 0))
            end = start + duration
        new_item["start"] = start
        new_item["end"] = end
        items.append(new_item)

    # Sort by start time for gap analysis
    items.sort(key=lambda x: x["start"])

    if on_progress:
        on_progress(20, "Detecting gaps...")

    # Calculate all gaps
    gaps_found = []

    # Check initial gap
    if items[0]["start"] > min_gap:
        gaps_found.append({
            "position": 0,
            "duration": items[0]["start"],
        })

    for i in range(len(items) - 1):
        current_end = items[i]["end"]
        next_start = items[i + 1]["start"]
        gap = next_start - current_end
        if gap > min_gap:
            gaps_found.append({
                "position": i + 1,
                "duration": gap,
            })

    if not gaps_found:
        if on_progress:
            on_progress(100, "No gaps to close")
        return RippleResult(
            items=items,
            gaps_closed=0,
            total_shift=0.0,
            locked_tracks_skipped=sorted(locked),
        )

    if on_progress:
        on_progress(50, f"Closing {len(gaps_found)} gap(s)...")

    # Apply cumulative shift to each item
    # Walk through items in order; accumulate shift from gaps before each item
    cumulative_shift = 0.0
    gap_idx = 0
    gaps_closed = 0

    for i, item in enumerate(items):
        # Accumulate any gaps at or before this position
        while gap_idx < len(gaps_found) and gaps_found[gap_idx]["position"] <= i:
            cumulative_shift += gaps_found[gap_idx]["duration"]
            gaps_closed += 1
            gap_idx += 1

        # Apply shift if item is not on a locked track
        track = item.get("track", "")
        if track not in locked and cumulative_shift > 0:
            item["start"] -= cumulative_shift
            item["end"] -= cumulative_shift

    total_shift = cumulative_shift

    # Recalculate duration if present
    for item in items:
        if "duration" in item:
            item["duration"] = item["end"] - item["start"]

    if on_progress:
        on_progress(100, f"Closed {gaps_closed} gap(s), shifted by {total_shift:.3f}s")

    logger.info(
        "Ripple close: %d gaps closed, %.3fs total shift (%d locked tracks)",
        gaps_closed, total_shift, len(locked),
    )
    return RippleResult(
        items=items,
        gaps_closed=gaps_closed,
        total_shift=total_shift,
        locked_tracks_skipped=sorted(locked),
    )
