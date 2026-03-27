"""
Auto B-Roll Insertion

Analyzes a transcript to identify natural B-roll insertion points:
1. Detects pauses/gaps in dialogue where B-roll would fit
2. Extracts topic keywords from each segment for B-roll matching
3. Returns a list of insertion windows with suggested search terms

Used with the footage search system to auto-match B-roll clips
from the user's media library.

No external dependencies beyond what's already in the pipeline.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger("opencut")

# Common filler/transition words that indicate topic shifts (good B-roll cues)
_TOPIC_SHIFT_PHRASES = {
    "for example", "let me show you", "take a look", "here's what",
    "moving on", "next up", "another thing", "speaking of",
    "on the other hand", "in contrast", "meanwhile", "let's talk about",
    "the key thing is", "what's interesting is", "here's the thing",
}

# Words to exclude from keyword extraction (too common to be useful)
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "it", "they", "them", "their", "this", "that",
    "these", "those", "and", "but", "or", "not", "no", "so", "if", "then",
    "than", "too", "very", "just", "about", "also", "only", "really", "like",
    "um", "uh", "know", "think", "going", "get", "got", "thing", "things",
    "way", "one", "much", "more", "some", "any", "all", "up", "out", "in",
    "on", "at", "to", "of", "for", "with", "from", "into", "what", "when",
    "where", "how", "why", "who", "which", "there", "here",
}


@dataclass
class BRollWindow:
    """A time window where B-roll could be inserted."""
    start: float           # Start time in seconds
    end: float             # End time in seconds
    duration: float        # Duration in seconds
    reason: str            # Why this is a good B-roll point
    keywords: List[str]    # Suggested search terms for matching B-roll
    score: float = 0.0     # Priority score (higher = better insertion point)
    context: str = ""      # Surrounding transcript text for context

    @property
    def is_valid(self) -> bool:
        return self.duration >= 0.5 and self.end > self.start


@dataclass
class BRollPlan:
    """Complete B-roll insertion plan for a video."""
    windows: List[BRollWindow] = field(default_factory=list)
    total_broll_time: float = 0.0
    total_windows: int = 0
    keywords_used: List[str] = field(default_factory=list)


def analyze_broll_opportunities(
    transcript_segments: List[Dict],
    min_gap: float = 1.0,
    max_gap: float = 10.0,
    min_broll_duration: float = 1.5,
    max_results: int = 20,
) -> BRollPlan:
    """
    Analyze a transcript to find natural B-roll insertion points.

    Identifies:
    1. Dialogue gaps/pauses (natural cutaway points)
    2. Topic shifts (where visual context change helps)
    3. Descriptive language (references to things that could be shown)

    Args:
        transcript_segments: List of dicts with start, end, text keys.
        min_gap: Minimum gap between segments to consider for B-roll (seconds).
        max_gap: Maximum gap to consider (longer gaps may be intentional silence).
        min_broll_duration: Minimum B-roll clip duration to suggest.
        max_results: Maximum number of B-roll windows to return.

    Returns:
        BRollPlan with ranked insertion windows and keywords.
    """
    if not transcript_segments:
        return BRollPlan()

    # Sort segments by start time
    sorted_segs = sorted(transcript_segments, key=lambda s: float(s.get("start", 0)))

    windows = []
    all_keywords = set()

    # 1. Find gaps between segments (natural pauses)
    for i in range(len(sorted_segs) - 1):
        seg_end = float(sorted_segs[i].get("end", 0))
        next_start = float(sorted_segs[i + 1].get("start", 0))
        gap = next_start - seg_end

        if min_gap <= gap <= max_gap:
            # Extract keywords from surrounding context
            before_text = sorted_segs[i].get("text", "")
            after_text = sorted_segs[i + 1].get("text", "")
            keywords = _extract_keywords(before_text + " " + after_text)
            all_keywords.update(keywords)

            # Score: longer gaps + more keywords = better
            score = min(1.0, gap / max_gap) * 0.6 + min(1.0, len(keywords) / 5) * 0.4

            windows.append(BRollWindow(
                start=seg_end,
                end=next_start,
                duration=round(gap, 2),
                reason="dialogue_pause",
                keywords=keywords[:5],
                score=round(score, 3),
                context=f"...{before_text[-60:]} [B-ROLL] {after_text[:60]}...",
            ))

    # 2. Find topic shifts within segments
    for i, seg in enumerate(sorted_segs):
        text = seg.get("text", "").lower()
        for phrase in _TOPIC_SHIFT_PHRASES:
            if phrase in text:
                seg_start = float(seg.get("start", 0))
                seg_end = float(seg.get("end", 0))

                # Insert B-roll just before the topic shift
                broll_start = max(0, seg_start - 0.5)
                broll_end = seg_start + min_broll_duration

                # Get keywords from the NEW topic
                after_text = text[text.index(phrase) + len(phrase):]
                keywords = _extract_keywords(after_text)
                all_keywords.update(keywords)

                if keywords:
                    windows.append(BRollWindow(
                        start=round(broll_start, 2),
                        end=round(broll_end, 2),
                        duration=round(broll_end - broll_start, 2),
                        reason="topic_shift",
                        keywords=keywords[:5],
                        score=0.7,
                        context=f"Topic shift: '{phrase}' → {', '.join(keywords[:3])}",
                    ))
                break  # Only one B-roll per segment for topic shifts

    # 3. Find descriptive/visual references
    for seg in sorted_segs:
        text = seg.get("text", "")
        visual_refs = _find_visual_references(text)
        if visual_refs:
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", 0))
            seg_dur = seg_end - seg_start

            # Insert B-roll during the descriptive segment
            if seg_dur >= min_broll_duration:
                all_keywords.update(visual_refs)
                windows.append(BRollWindow(
                    start=round(seg_start, 2),
                    end=round(min(seg_start + min(seg_dur, 5.0), seg_end), 2),
                    duration=round(min(seg_dur, 5.0), 2),
                    reason="visual_reference",
                    keywords=visual_refs[:5],
                    score=0.6,
                    context=f"Visual reference: {', '.join(visual_refs[:3])}",
                ))

    # Deduplicate overlapping windows (keep higher score)
    windows = _deduplicate_windows(windows)

    # Sort by score descending, limit results
    windows.sort(key=lambda w: w.score, reverse=True)
    windows = windows[:max_results]

    total_time = sum(w.duration for w in windows)

    return BRollPlan(
        windows=windows,
        total_broll_time=round(total_time, 2),
        total_windows=len(windows),
        keywords_used=sorted(all_keywords)[:30],
    )


def _extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    """Extract meaningful keywords from text for B-roll search."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    # Filter stopwords and count frequency
    counts = {}
    for w in words:
        if w not in _STOPWORDS and len(w) >= 3:
            counts[w] = counts.get(w, 0) + 1

    # Sort by frequency, take top N
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in ranked[:max_keywords]]


def _find_visual_references(text: str) -> List[str]:
    """Find references to visual objects/scenes in text."""
    visual_patterns = [
        r'\b(camera|lens|photo|picture|image|screen|video|film)\b',
        r'\b(building|house|city|street|road|bridge|mountain|ocean|beach|forest|sky)\b',
        r'\b(car|bike|boat|plane|train|truck)\b',
        r'\b(food|cooking|kitchen|restaurant|coffee|water|drink)\b',
        r'\b(computer|phone|laptop|keyboard|monitor|desk)\b',
        r'\b(dog|cat|animal|bird|fish|horse)\b',
        r'\b(person|people|crowd|team|group|family|children)\b',
        r'\b(sunset|sunrise|night|day|rain|snow|weather|storm)\b',
        r'\b(product|package|box|tool|equipment|machine|device)\b',
    ]

    matches = set()
    text_lower = text.lower()
    for pattern in visual_patterns:
        for m in re.finditer(pattern, text_lower):
            matches.add(m.group(1))

    return list(matches)


def _deduplicate_windows(windows: List[BRollWindow]) -> List[BRollWindow]:
    """Remove overlapping windows, keeping the one with higher score."""
    if len(windows) <= 1:
        return windows

    # Sort by start time
    sorted_wins = sorted(windows, key=lambda w: w.start)
    result = [sorted_wins[0]]

    for w in sorted_wins[1:]:
        prev = result[-1]
        # Check overlap
        if w.start < prev.end:
            # Overlapping — keep the one with higher score
            if w.score > prev.score:
                result[-1] = w
            # else keep prev (already in result)
        else:
            result.append(w)

    return result
