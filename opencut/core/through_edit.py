"""
OpenCut Through-Edit Cleanup

Detect adjacent cuts from the same source with continuous timecode
and offer merge suggestions.  A "through edit" is a cut point that
exists in the timeline but doesn't actually represent a discontinuity
in the source media -- the two clips are consecutive frames from the
same source and can be safely merged into one.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")

# Maximum gap (in seconds) between source_out of clip N and source_in of
# clip N+1 for them to be considered continuous.
DEFAULT_CONTINUITY_TOLERANCE = 0.05  # ~1.5 frames at 30fps


@dataclass
class ThroughEdit:
    """A detected through-edit between two adjacent cuts."""
    index_a: int
    index_b: int
    source_file: str
    gap_seconds: float
    merged_in: float
    merged_out: float


@dataclass
class ThroughEditResult:
    """Result of through-edit detection."""
    through_edits: List[ThroughEdit]
    total_cuts: int
    mergeable_count: int


def detect_through_edits(
    cut_list: List[Dict[str, Any]],
    tolerance: float = DEFAULT_CONTINUITY_TOLERANCE,
    on_progress: Optional[Callable] = None,
) -> ThroughEditResult:
    """Detect through-edits in a cut list.

    A through-edit occurs when two adjacent cuts:
    1. Reference the same source file
    2. Have continuous timecode (source_out of first ~= source_in of second)

    Args:
        cut_list: List of cut dicts, each with:
            - 'source_file': Path to source media.
            - 'source_in': Start time in seconds.
            - 'source_out': End time in seconds.
            Optionally:
            - 'track': Track identifier (through-edits only on same track).
        tolerance: Maximum gap in seconds to still be considered continuous.
        on_progress: Optional progress callback.

    Returns:
        ThroughEditResult with list of detected through-edits.
    """
    if on_progress:
        on_progress(5, "Analyzing cut list for through-edits...")

    if len(cut_list) < 2:
        return ThroughEditResult(
            through_edits=[],
            total_cuts=len(cut_list),
            mergeable_count=0,
        )

    through_edits: List[ThroughEdit] = []

    for i in range(len(cut_list) - 1):
        if on_progress and len(cut_list) > 10:
            pct = 5 + int((i / (len(cut_list) - 1)) * 90)
            on_progress(pct, f"Checking cut {i + 1}/{len(cut_list) - 1}...")

        cut_a = cut_list[i]
        cut_b = cut_list[i + 1]

        # Same source file?
        source_a = cut_a.get("source_file", "")
        source_b = cut_b.get("source_file", "")
        if not source_a or source_a != source_b:
            continue

        # Same track? (if track info is available)
        track_a = cut_a.get("track", None)
        track_b = cut_b.get("track", None)
        if track_a is not None and track_b is not None and track_a != track_b:
            continue

        # Check timecode continuity
        out_a = float(cut_a.get("source_out", 0))
        in_b = float(cut_b.get("source_in", 0))

        gap = abs(in_b - out_a)
        if gap <= tolerance:
            # This is a through-edit
            merged_in = float(cut_a.get("source_in", 0))
            merged_out = float(cut_b.get("source_out", 0))

            through_edits.append(ThroughEdit(
                index_a=i,
                index_b=i + 1,
                source_file=source_a,
                gap_seconds=gap,
                merged_in=merged_in,
                merged_out=merged_out,
            ))

    if on_progress:
        on_progress(100, f"Found {len(through_edits)} through-edit(s)")

    logger.info(
        "Detected %d through-edits in %d cuts",
        len(through_edits), len(cut_list),
    )
    return ThroughEditResult(
        through_edits=through_edits,
        total_cuts=len(cut_list),
        mergeable_count=len(through_edits),
    )


def merge_through_edits(
    cut_list: List[Dict[str, Any]],
    indices: Optional[List[Tuple[int, int]]] = None,
    tolerance: float = DEFAULT_CONTINUITY_TOLERANCE,
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Merge through-edits in a cut list.

    If indices are provided, only those specific pairs are merged.
    Otherwise, auto-detects and merges all through-edits.

    Args:
        cut_list: List of cut dicts (same format as detect_through_edits).
        indices: Optional list of (index_a, index_b) pairs to merge.
                 If None, all detected through-edits are merged.
        tolerance: Continuity tolerance in seconds.
        on_progress: Optional progress callback.

    Returns:
        New cut list with through-edits merged.
    """
    if on_progress:
        on_progress(5, "Preparing to merge through-edits...")

    if len(cut_list) < 2:
        return list(cut_list)

    # Determine which pairs to merge
    if indices is None:
        result = detect_through_edits(cut_list, tolerance)
        pairs_to_merge = {(te.index_a, te.index_b) for te in result.through_edits}
    else:
        pairs_to_merge = {(a, b) for a, b in indices}

    if not pairs_to_merge:
        if on_progress:
            on_progress(100, "No through-edits to merge")
        return list(cut_list)

    if on_progress:
        on_progress(30, f"Merging {len(pairs_to_merge)} through-edit(s)...")

    # Build merged list by walking through cuts
    # Track which indices are "consumed" by a merge
    consumed = set()
    merged_list: List[Dict[str, Any]] = []

    i = 0
    while i < len(cut_list):
        if i in consumed:
            i += 1
            continue

        # Check if this cut starts a merge chain
        current = dict(cut_list[i])
        j = i

        while (j, j + 1) in pairs_to_merge and j + 1 < len(cut_list):
            # Merge j+1 into current
            next_cut = cut_list[j + 1]
            current["source_out"] = next_cut.get("source_out", current.get("source_out"))
            current["record_out"] = next_cut.get("record_out", current.get("record_out"))
            # Preserve duration info
            src_in = float(current.get("source_in", 0))
            src_out = float(current.get("source_out", 0))
            current["duration"] = src_out - src_in
            consumed.add(j + 1)
            j += 1

        merged_list.append(current)
        i += 1

    if on_progress:
        on_progress(100, f"Merged to {len(merged_list)} cuts (from {len(cut_list)})")

    logger.info(
        "Merged through-edits: %d cuts -> %d cuts",
        len(cut_list), len(merged_list),
    )
    return merged_list
