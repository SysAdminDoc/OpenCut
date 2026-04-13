"""
OpenCut Nested Sequence Detection Module

Detect repeated clip patterns in a timeline and offer nesting —
collapsing identical sub-sequences into a single reusable reference.

The detector works on abstract timeline item dicts (each with ``clip_id``,
``in``, ``out``, ``source`` keys) and finds runs of consecutive items that
repeat elsewhere in the timeline.  When a pattern is confirmed the user
can collapse all instances to a ``nested_ref``.
"""

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


@dataclass
class PatternMatch:
    """A detected repeated pattern in the timeline."""
    pattern_id: str = ""
    items: List[Dict[str, Any]] = field(default_factory=list)
    length: int = 0
    occurrences: int = 0
    positions: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NestedSequence:
    """A collapsed nested sequence created from a pattern."""
    sequence_id: str = ""
    name: str = ""
    items: List[Dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0
    item_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item_signature(item: Dict[str, Any]) -> str:
    """Create a hashable signature for a timeline item."""
    parts = [
        str(item.get("source", "")),
        f"{item.get('in', 0):.3f}",
        f"{item.get('out', 0):.3f}",
    ]
    return "|".join(parts)


def _sequence_hash(items: List[Dict[str, Any]]) -> str:
    """Generate a stable hash for a sequence of items."""
    sigs = [_item_signature(it) for it in items]
    combined = "::".join(sigs)
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_repeated_patterns(
    timeline_items: List[Dict[str, Any]],
    min_length: int = 2,
    min_occurrences: int = 2,
    on_progress: Optional[Callable] = None,
) -> List[PatternMatch]:
    """Detect repeated clip patterns in a timeline.

    Scans all sub-sequences of length ``min_length`` up to half the
    timeline length, counting occurrences by content signature.  Only
    patterns that appear at least ``min_occurrences`` times are returned.

    Args:
        timeline_items: List of timeline item dicts (need ``source``,
            ``in``, ``out`` keys at minimum).
        min_length: Minimum pattern length (number of items).
        min_occurrences: Minimum repetition count to qualify.
        on_progress: Optional ``(pct, msg)`` callback.

    Returns:
        List of PatternMatch objects, sorted by occurrences descending.
    """
    if on_progress:
        on_progress(10, "Scanning timeline for patterns...")

    n = len(timeline_items)
    if n < min_length * min_occurrences:
        if on_progress:
            on_progress(100, "Timeline too short for patterns")
        return []

    max_pat_len = n // 2
    sig_list = [_item_signature(it) for it in timeline_items]

    # Collect all sub-sequence signatures and their start positions
    pattern_positions: Dict[str, List[int]] = {}

    for length in range(min_length, max_pat_len + 1):
        for start in range(n - length + 1):
            key = "::".join(sig_list[start:start + length])
            if key not in pattern_positions:
                pattern_positions[key] = []
            pattern_positions[key].append(start)

        if on_progress:
            pct = 10 + int(((length - min_length + 1) / max(1, max_pat_len - min_length + 1)) * 70)
            on_progress(pct, f"Scanning patterns of length {length}...")

    # Filter to patterns with enough occurrences (non-overlapping)
    results: List[PatternMatch] = []

    for key, positions in pattern_positions.items():
        if len(positions) < min_occurrences:
            continue

        # Filter overlapping positions
        length = key.count("::") + 1
        non_overlapping = []
        last_end = -1
        for pos in sorted(positions):
            if pos >= last_end:
                non_overlapping.append(pos)
                last_end = pos + length

        if len(non_overlapping) < min_occurrences:
            continue

        first_pos = non_overlapping[0]
        items_slice = timeline_items[first_pos:first_pos + length]
        pat_id = _sequence_hash(items_slice)

        results.append(PatternMatch(
            pattern_id=pat_id,
            items=items_slice,
            length=length,
            occurrences=len(non_overlapping),
            positions=non_overlapping,
        ))

    # Sort by occurrences descending, then length descending
    results.sort(key=lambda p: (-p.occurrences, -p.length))

    # Remove duplicates by pattern_id
    seen = set()
    unique = []
    for p in results:
        if p.pattern_id not in seen:
            seen.add(p.pattern_id)
            unique.append(p)

    if on_progress:
        on_progress(100, f"Found {len(unique)} repeated patterns")

    logger.info("Detected %d repeated patterns in %d timeline items", len(unique), n)
    return unique


def create_nested_sequence(
    pattern: PatternMatch,
    name: str = "",
    on_progress: Optional[Callable] = None,
) -> NestedSequence:
    """Create a nested sequence from a detected pattern.

    Args:
        pattern: A PatternMatch describing the repeating sub-sequence.
        name: Human-readable name (auto-generated if empty).
        on_progress: Optional progress callback.

    Returns:
        NestedSequence object representing the collapsed group.
    """
    if on_progress:
        on_progress(30, "Creating nested sequence...")

    if not name:
        name = f"Nested_{pattern.pattern_id[:8]}"

    total_duration = sum(
        float(it.get("out", 0)) - float(it.get("in", 0))
        for it in pattern.items
    )

    seq = NestedSequence(
        sequence_id=pattern.pattern_id,
        name=name,
        items=list(pattern.items),
        duration=total_duration,
        item_count=len(pattern.items),
    )

    if on_progress:
        on_progress(100, "Nested sequence created")

    logger.info("Created nested sequence '%s' (%d items, %.1fs)",
                name, len(pattern.items), total_duration)
    return seq


def replace_with_nested(
    timeline_items: List[Dict[str, Any]],
    pattern: PatternMatch,
    nested_ref: NestedSequence,
    on_progress: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Replace all occurrences of a pattern with a nested sequence reference.

    Walks the timeline and substitutes each occurrence of the pattern
    with a single item referencing ``nested_ref``.

    Args:
        timeline_items: Original timeline item list.
        pattern: The detected pattern to replace.
        nested_ref: The nested sequence to use as replacement.
        on_progress: Optional progress callback.

    Returns:
        New timeline item list with patterns replaced.
    """
    if on_progress:
        on_progress(10, "Replacing patterns with nested refs...")

    sig_list = [_item_signature(it) for it in timeline_items]
    pat_sigs = [_item_signature(it) for it in pattern.items]
    pat_key = "::".join(pat_sigs)
    pat_len = len(pat_sigs)

    new_timeline: List[Dict[str, Any]] = []
    i = 0
    replacements = 0

    while i < len(timeline_items):
        if i + pat_len <= len(timeline_items):
            candidate = "::".join(sig_list[i:i + pat_len])
            if candidate == pat_key:
                new_timeline.append({
                    "type": "nested_sequence",
                    "sequence_id": nested_ref.sequence_id,
                    "name": nested_ref.name,
                    "duration": nested_ref.duration,
                    "item_count": nested_ref.item_count,
                })
                i += pat_len
                replacements += 1
                continue

        new_timeline.append(timeline_items[i])
        i += 1

    if on_progress:
        on_progress(100, f"Replaced {replacements} occurrences")

    logger.info("Replaced %d pattern occurrences with nested ref '%s'",
                replacements, nested_ref.name)
    return new_timeline
