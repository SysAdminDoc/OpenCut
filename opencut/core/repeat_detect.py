"""
Repeated Take Detection

Identifies when a speaker restarts a sentence or repeats a phrase,
using word-overlap (Jaccard similarity) on WhisperX word-level segments.
Falls back to segment-level similarity if word timestamps unavailable.
"""

import logging
import string
from typing import List

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> List[str]:
    """Lowercase, strip punctuation, return non-empty tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t]


def _jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Jaccard similarity between two token lists."""
    if not tokens_a or not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_repeated_takes(
    segments: List[dict],
    threshold: float = 0.6,
    gap_tolerance: float = 2.0,
) -> dict:
    """
    Detect repeated/fumbled takes in a list of transcript segments.

    A repeated take occurs when a speaker says the same thing (or very similar)
    twice in succession — the first attempt should be removed.

    Args:
        segments: List of dicts with keys "text" (str), "start" (float),
                  "end" (float), and optionally "words" (list).
        threshold: Jaccard similarity threshold above which two segments are
                   considered repeats. Default 0.6.
        gap_tolerance: Maximum gap in seconds between end of segment i and
                       start of segment i+1 for them to be considered
                       successive takes. Default 2.0.

    Returns:
        Dict with:
            "repeats": list of repeat entries (first take, to be removed)
            "clean_ranges": list of {"start", "end"} ranges to keep
    """
    if not segments:
        return {"repeats": [], "clean_ranges": []}

    # Normalise tokens for each segment
    tokenised = []
    for seg in segments:
        tokens = _normalise(seg.get("text", ""))
        # If word-level data is present, prefer concatenating word texts
        words = seg.get("words", [])
        if words:
            word_tokens = _normalise(" ".join(w.get("word", w.get("text", "")) for w in words))
            if word_tokens:
                tokens = word_tokens
        tokenised.append(tokens)

    n = len(segments)
    repeat_indices = set()  # indices of segments marked as the first (bad) take

    for i in range(n - 1):
        j = i + 1

        # Gap check: segment i must end within gap_tolerance of segment j start
        end_i = segments[i].get("end", 0.0)
        start_j = segments[j].get("start", 0.0)
        gap = start_j - end_i
        if gap > gap_tolerance:
            continue

        sim = _jaccard(tokenised[i], tokenised[j])
        if sim >= threshold:
            # Mark segment i as the fumbled / repeated take
            repeat_indices.add(i)
            logger.debug(
                "Repeat detected: segment %d (%.2fs-%.2fs) ~ segment %d (%.2fs-%.2fs) "
                "similarity=%.3f",
                i,
                segments[i].get("start", 0.0),
                end_i,
                j,
                start_j,
                segments[j].get("end", 0.0),
                sim,
            )

    repeats = []
    for idx in sorted(repeat_indices):
        seg = segments[idx]
        # Find the paired segment (first non-repeat after idx)
        paired = idx + 1
        repeats.append({
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", ""),
            "similarity": round(
                _jaccard(tokenised[idx], tokenised[paired] if paired < n else []), 4
            ),
            "paired_with": paired,
        })

    # Build clean ranges: all segments that are NOT marked as repeats
    clean_ranges = _build_clean_ranges(segments, repeat_indices)

    return {"repeats": repeats, "clean_ranges": clean_ranges}


# ---------------------------------------------------------------------------
# Range helpers
# ---------------------------------------------------------------------------

def _build_clean_ranges(segments: List[dict], repeat_indices: set) -> List[dict]:
    """Return time ranges corresponding to segments that are not repeats."""
    clean = []
    for i, seg in enumerate(segments):
        if i not in repeat_indices:
            clean.append({
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
            })
    # Merge adjacent/overlapping clean ranges
    return _merge_ranges(clean)


def _merge_ranges(ranges: List[dict]) -> List[dict]:
    """Merge a list of {"start", "end"} dicts, sorting and collapsing overlaps."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda r: r["start"])
    merged = [dict(sorted_ranges[0])]
    for r in sorted_ranges[1:]:
        if r["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], r["end"])
        else:
            merged.append(dict(r))
    return merged


def merge_repeat_ranges(repeats: List[dict]) -> List[dict]:
    """
    Merge overlapping removal ranges from a list of repeat entries.

    Args:
        repeats: List of repeat dicts as returned by detect_repeated_takes().

    Returns:
        List of merged {"start": float, "end": float} dicts.
    """
    raw = [{"start": r["start"], "end": r["end"]} for r in repeats]
    return _merge_ranges(raw)
