"""
Repeated Take Detection

Identifies when a speaker restarts a sentence or repeats a phrase,
using word-overlap (Jaccard similarity) on WhisperX word-level segments.
Falls back to segment-level similarity if word timestamps unavailable.
"""

import logging
import string
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Best-take ranking
# ---------------------------------------------------------------------------

#: A clean read of the same line usually ends in terminal punctuation. A take
#: that trails off mid-sentence is the one the speaker abandoned.
_TERMINAL_PUNCTUATION = ".!?…\"'"

#: Weights for the keep-candidate score. Deliberately small and legible: the
#: point is a defensible preselection a human can overrule, not a model.
_SCORE_WEIGHTS = {
    "completion": 2.0,
    "filler_penalty": 0.6,
    "rate_stability": 1.5,
    "length": 0.8,
}

#: Outside this band a take is either rushed or halting relative to the
#: cluster's median, both of which read as a worse take.
_RATE_TOLERANCE = 0.35


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
    rank_takes: bool = True,
    llm_verdict: Optional[Callable[[List[dict]], Optional[int]]] = None,
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

        rank_takes: Also group the repeats into clusters and recommend which
                    take to keep. Additive; the existing keys are unchanged.
        llm_verdict: Optional callable handed a cluster's takes that may return
                     the index to keep. The heuristic runs without it.

    Returns:
        Dict with:
            "repeats": list of repeat entries (first take, to be removed)
            "clean_ranges": list of {"start", "end"} ranges to keep
            "clusters": ranked takes with a keep recommendation (when
                        rank_takes is set)
    """
    if not segments:
        return {"repeats": [], "clean_ranges": [], "clusters": []}

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

    result = {"repeats": repeats, "clean_ranges": clean_ranges}
    if rank_takes:
        result["clusters"] = rank_repeat_clusters(segments, repeat_indices, llm_verdict)
    return result


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


# ---------------------------------------------------------------------------
# Best-take ranking (F344)
# ---------------------------------------------------------------------------

def _filler_count(text: str) -> int:
    """Count filler tokens using the shared list, falling back to a small set."""
    try:
        from opencut.core.fillers import FILLER_WORDS

        singles = {variant for group in FILLER_WORDS.values() for variant in group if " " not in variant}
        phrases = [variant for group in FILLER_WORDS.values() for variant in group if " " in variant]
    except Exception:  # pragma: no cover - fillers module is always present in-tree
        singles, phrases = {"um", "uh", "er", "ah", "like"}, []

    lowered = " ".join(_normalise(text))
    count = sum(1 for token in lowered.split() if token in singles)
    for phrase in phrases:
        count += lowered.count(phrase)
    return count


def _take_signals(segment: dict) -> Dict[str, float]:
    """Observable properties of one take, before any cluster comparison."""
    text = str(segment.get("text", "") or "")
    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", 0.0) or 0.0)
    duration = max(end - start, 0.0)
    words = len(_normalise(text))
    stripped = text.strip()
    return {
        "word_count": float(words),
        "duration": round(duration, 3),
        "filler_count": float(_filler_count(text)),
        "words_per_minute": round((words / duration * 60.0) if duration > 0 else 0.0, 2),
        "completed": 1.0 if stripped and stripped[-1] in _TERMINAL_PUNCTUATION else 0.0,
    }


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _score_take(signals: Dict[str, float], median_rate: float, max_words: float) -> float:
    """Heuristic quality score. Higher is a better keep candidate."""
    score = _SCORE_WEIGHTS["completion"] * signals["completed"]
    score -= _SCORE_WEIGHTS["filler_penalty"] * signals["filler_count"]
    if median_rate > 0 and signals["words_per_minute"] > 0:
        deviation = abs(signals["words_per_minute"] - median_rate) / median_rate
        score += _SCORE_WEIGHTS["rate_stability"] * max(0.0, 1.0 - deviation / _RATE_TOLERANCE)
    if max_words > 0:
        score += _SCORE_WEIGHTS["length"] * (signals["word_count"] / max_words)
    return round(score, 4)


def _cluster_indices(repeat_indices: set, total: int) -> List[List[int]]:
    """Group each repeat and the take it was paired with into one cluster.

    A run of consecutive repeats is one cluster: the speaker fumbled the same
    line several times, and only the final attempt survives detection.
    """
    clusters: List[List[int]] = []
    for index in sorted(repeat_indices):
        if clusters and index == clusters[-1][-1] + 1:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    grouped = []
    for cluster in clusters:
        tail = cluster[-1] + 1
        if tail < total:
            cluster = [*cluster, tail]
        grouped.append(cluster)
    return grouped


def rank_repeat_clusters(
    segments: List[dict],
    repeat_indices: set,
    llm_verdict: Optional[Callable[[List[dict]], Optional[int]]] = None,
) -> List[dict]:
    """Rank the takes in each repeat cluster and recommend one to keep.

    The heuristic runs with nothing configured. *llm_verdict* is an optional
    callable handed the cluster's takes that may return the index to keep; when
    it is absent or fails, the fallback is recorded on the cluster rather than
    silently swallowed, so a reviewer can see which judgement they are reading.
    """
    clusters: List[dict] = []
    for indices in _cluster_indices(repeat_indices, len(segments)):
        takes = []
        for index in indices:
            segment = segments[index]
            takes.append({
                "index": index,
                "start": float(segment.get("start", 0.0) or 0.0),
                "end": float(segment.get("end", 0.0) or 0.0),
                "text": str(segment.get("text", "") or ""),
                "signals": _take_signals(segment),
            })

        rates = [t["signals"]["words_per_minute"] for t in takes if t["signals"]["words_per_minute"] > 0]
        median_rate = _median(rates)
        max_words = max((t["signals"]["word_count"] for t in takes), default=0.0)
        for take in takes:
            take["score"] = _score_take(take["signals"], median_rate, max_words)

        # Ties go to the later take: without evidence, the speaker's final
        # attempt is the one they meant to keep, which is also the behaviour
        # the detect-only output has always had.
        best = max(takes, key=lambda t: (t["score"], t["index"]))
        keep_index = best["index"]
        source = "heuristic"
        fallback_reason = ""

        if llm_verdict is not None:
            try:
                verdict = llm_verdict(takes)
            except Exception as exc:
                verdict = None
                fallback_reason = f"llm verdict failed: {type(exc).__name__}: {exc}"
            if isinstance(verdict, int) and verdict in indices:
                keep_index = verdict
                source = "llm"
            elif not fallback_reason:
                fallback_reason = "llm verdict was unavailable or out of range"
        else:
            fallback_reason = "no llm configured"

        clusters.append({
            "indices": list(indices),
            "keep_index": keep_index,
            "cut_indices": [i for i in indices if i != keep_index],
            "decision_source": source,
            "fallback_reason": fallback_reason,
            "takes": sorted(takes, key=lambda t: (-t["score"], t["index"])),
        })
    return clusters
