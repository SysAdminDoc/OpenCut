"""
OpenCut AI B-Roll Suggestion Module v0.1.0

Analyze transcript for B-roll insertion cues, search a footage index,
and generate a cut list with matched clips:
- Extract topic-shift phrases and visual keywords from transcript
- Score and rank B-roll insertion opportunities
- Match cues against a footage metadata index
- Generate an edit-ready cut list with timecodes

Works with the existing footage search/index system in OpenCut.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# B-Roll cue detection data
# ---------------------------------------------------------------------------
_VISUAL_CUE_PHRASES = {
    "look at", "take a look", "here's what", "let me show",
    "for example", "such as", "like this", "check this out",
    "picture this", "imagine", "you can see", "as you see",
    "watch this", "notice how", "here we have",
}

_TOPIC_SHIFT_PHRASES = {
    "moving on", "next up", "another thing", "speaking of",
    "on the other hand", "in contrast", "meanwhile", "also",
    "furthermore", "additionally", "let's talk about",
    "switching to", "now let's", "the next thing",
}

_VISUAL_KEYWORD_PATTERNS = [
    # Nature/outdoor
    (re.compile(r"\b(mountain|ocean|river|forest|sunset|sunrise|landscape|beach|sky|cloud)\b", re.I),
     "nature"),
    # Technology
    (re.compile(r"\b(computer|phone|screen|device|app|software|code|website|server|robot)\b", re.I),
     "technology"),
    # People/action
    (re.compile(r"\b(team|meeting|crowd|audience|people|person|walking|running|driving)\b", re.I),
     "people"),
    # Food
    (re.compile(r"\b(food|cooking|kitchen|restaurant|recipe|meal|ingredient)\b", re.I),
     "food"),
    # City/urban
    (re.compile(r"\b(city|building|street|traffic|downtown|skyline|office|store)\b", re.I),
     "urban"),
    # Business
    (re.compile(r"\b(chart|graph|data|report|presentation|whiteboard|money|finance)\b", re.I),
     "business"),
]

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "i", "me", "my", "we", "our",
    "you", "your", "he", "she", "it", "they", "them", "their", "this", "that",
    "and", "but", "or", "not", "so", "if", "then", "than", "too", "very",
    "just", "about", "also", "only", "really", "like", "um", "uh", "know",
    "think", "going", "get", "got", "thing", "things", "way", "one",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BRollCue:
    """A detected B-roll insertion cue from transcript analysis."""
    start: float           # Start time in seconds
    end: float             # End time in seconds
    cue_type: str          # "visual_phrase", "topic_shift", "pause", "keyword"
    keywords: List[str] = field(default_factory=list)
    category: str = ""     # Visual category (nature, tech, etc.)
    text: str = ""         # Surrounding transcript text
    score: float = 0.0     # Priority score (higher = better)
    phrase_match: str = ""  # The matched trigger phrase


@dataclass
class BRollMatch:
    """A matched footage clip for a B-roll cue."""
    cue_index: int         # Index of the matching cue
    clip_path: str = ""    # Path to the footage file
    clip_name: str = ""    # Display name
    relevance: float = 0.0  # Match relevance score (0-1)
    in_point: float = 0.0  # Suggested in point (seconds)
    out_point: float = 0.0  # Suggested out point (seconds)
    match_keywords: List[str] = field(default_factory=list)


@dataclass
class BRollCutList:
    """Complete B-roll cut list for an edit."""
    cues: List[BRollCue] = field(default_factory=list)
    matches: List[BRollMatch] = field(default_factory=list)
    unmatched_cues: int = 0
    total_broll_duration: float = 0.0
    coverage_ratio: float = 0.0    # Fraction of cues with matches


# ---------------------------------------------------------------------------
# Cue Detection
# ---------------------------------------------------------------------------
def detect_broll_cues(
    transcript: List[Dict],
    min_gap: float = 1.0,
    min_score: float = 0.3,
) -> List[BRollCue]:
    """
    Detect B-roll insertion cues from a transcript.

    Analyzes transcript segments for:
    - Visual cue phrases ("look at this", "here's what", etc.)
    - Topic shifts ("moving on", "next up", etc.)
    - Pauses/gaps between speech segments
    - Visual keywords (nature, technology, etc.)

    Args:
        transcript: List of segment dicts with 'start', 'end', 'text' keys.
        min_gap: Minimum gap duration (seconds) to consider as a pause cue.
        min_score: Minimum score threshold for cues.

    Returns:
        List of BRollCue objects sorted by score (descending).
    """
    cues = []

    for i, seg in enumerate(transcript):
        text = seg.get("text", "").strip()
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text_lower = text.lower()

        if not text:
            continue

        # Check for visual cue phrases
        for phrase in _VISUAL_CUE_PHRASES:
            if phrase in text_lower:
                words = [
                    w for w in re.findall(r"\b\w+\b", text_lower)
                    if w not in _STOPWORDS and len(w) > 2
                ]
                cues.append(BRollCue(
                    start=start,
                    end=end,
                    cue_type="visual_phrase",
                    keywords=words[:5],
                    text=text[:100],
                    score=0.9,
                    phrase_match=phrase,
                ))
                break

        # Check for topic shifts
        for phrase in _TOPIC_SHIFT_PHRASES:
            if phrase in text_lower:
                words = [
                    w for w in re.findall(r"\b\w+\b", text_lower)
                    if w not in _STOPWORDS and len(w) > 2
                ]
                cues.append(BRollCue(
                    start=start,
                    end=end,
                    cue_type="topic_shift",
                    keywords=words[:5],
                    text=text[:100],
                    score=0.7,
                    phrase_match=phrase,
                ))
                break

        # Check for visual keywords
        for pattern, category in _VISUAL_KEYWORD_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                cues.append(BRollCue(
                    start=start,
                    end=end,
                    cue_type="keyword",
                    keywords=[m.lower() for m in matches[:5]],
                    category=category,
                    text=text[:100],
                    score=0.6,
                ))
                break

        # Check for pauses/gaps between segments
        if i + 1 < len(transcript):
            next_start = transcript[i + 1].get("start", 0.0)
            gap = next_start - end
            if gap >= min_gap:
                gap_score = min(1.0, 0.4 + gap * 0.1)
                words = [
                    w for w in re.findall(r"\b\w+\b", text_lower)
                    if w not in _STOPWORDS and len(w) > 2
                ]
                cues.append(BRollCue(
                    start=end,
                    end=next_start,
                    cue_type="pause",
                    keywords=words[:3],
                    text=text[:100],
                    score=gap_score,
                ))

    # Filter by minimum score and sort
    cues = [c for c in cues if c.score >= min_score]
    cues.sort(key=lambda c: c.score, reverse=True)

    return cues


# ---------------------------------------------------------------------------
# Footage Matching
# ---------------------------------------------------------------------------
def _match_cue_to_footage(cue: BRollCue, footage_index: List[Dict],
                          cue_idx: int) -> List[BRollMatch]:
    """Match a single cue against the footage index."""
    matches = []

    for clip in footage_index:
        clip_keywords = set(k.lower() for k in clip.get("keywords", []))
        clip_tags = set(t.lower() for t in clip.get("tags", []))
        clip_desc = clip.get("description", "").lower()
        clip.get("name", "").lower()

        all_clip_terms = clip_keywords | clip_tags
        if clip_desc:
            all_clip_terms.update(re.findall(r"\b\w+\b", clip_desc))

        # Score based on keyword overlap
        cue_keywords = set(k.lower() for k in cue.keywords)
        overlap = cue_keywords & all_clip_terms
        if not overlap and cue.category:
            # Try category match
            if cue.category in clip_tags or cue.category in clip_desc:
                overlap = {cue.category}

        if not overlap:
            continue

        relevance = len(overlap) / max(1, len(cue_keywords))

        # Bonus for category match
        if cue.category and cue.category in (clip.get("category", "") or "").lower():
            relevance = min(1.0, relevance + 0.2)

        clip_duration = clip.get("duration", 5.0)
        cue_duration = max(1.0, cue.end - cue.start)
        out_point = min(clip_duration, cue_duration + 1.0)

        matches.append(BRollMatch(
            cue_index=cue_idx,
            clip_path=clip.get("path", ""),
            clip_name=clip.get("name", ""),
            relevance=round(relevance, 3),
            in_point=0.0,
            out_point=round(out_point, 2),
            match_keywords=list(overlap),
        ))

    matches.sort(key=lambda m: m.relevance, reverse=True)
    return matches[:3]  # Top 3 matches per cue


# ---------------------------------------------------------------------------
# Cut List Generation
# ---------------------------------------------------------------------------
def generate_broll_cutlist(
    cues: List[BRollCue],
    matches: List[BRollMatch],
    max_entries: int = 20,
) -> BRollCutList:
    """
    Generate a B-roll cut list from cues and matches.

    Args:
        cues: List of detected B-roll cues.
        matches: List of footage matches.
        max_entries: Maximum cut list entries.

    Returns:
        BRollCutList with matched and unmatched cues.
    """
    # Group matches by cue index
    cue_matches = {}
    for m in matches:
        cue_matches.setdefault(m.cue_index, []).append(m)

    unmatched = sum(1 for i in range(len(cues)) if i not in cue_matches)
    total_duration = 0.0
    best_matches = []

    for i, cue in enumerate(cues[:max_entries]):
        if i in cue_matches:
            best = cue_matches[i][0]
            best_matches.append(best)
            total_duration += best.out_point - best.in_point

    coverage = (len(cues) - unmatched) / max(1, len(cues))

    return BRollCutList(
        cues=cues[:max_entries],
        matches=best_matches,
        unmatched_cues=unmatched,
        total_broll_duration=round(total_duration, 2),
        coverage_ratio=round(coverage, 3),
    )


# ---------------------------------------------------------------------------
# Main Suggest Pipeline
# ---------------------------------------------------------------------------
def suggest_broll(
    transcript: List[Dict],
    footage_index: Optional[List[Dict]] = None,
    min_gap: float = 1.0,
    max_results: int = 20,
    on_progress: Optional[Callable] = None,
) -> BRollCutList:
    """
    Analyze transcript and suggest B-roll clips from footage index.

    Full pipeline:
    1. Detect B-roll insertion cues from transcript
    2. Match cues against footage index (if provided)
    3. Generate an edit-ready cut list

    Args:
        transcript: List of segment dicts with 'start', 'end', 'text'.
        footage_index: List of footage metadata dicts with 'name', 'path',
                       'keywords', 'tags', 'duration', 'category', 'description'.
        min_gap: Minimum gap duration for pause detection.
        max_results: Maximum suggestions to return.
        on_progress: Callback(pct, msg) for progress updates.

    Returns:
        BRollCutList with cues, matches, and statistics.
    """
    if on_progress:
        on_progress(10, "Analyzing transcript for B-roll cues...")

    cues = detect_broll_cues(transcript, min_gap=min_gap)

    if on_progress:
        on_progress(40, f"Found {len(cues)} B-roll cues, matching footage...")

    matches = []
    if footage_index:
        for i, cue in enumerate(cues[:max_results]):
            cue_matches = _match_cue_to_footage(cue, footage_index, i)
            matches.extend(cue_matches)

            if on_progress and (i + 1) % 5 == 0:
                pct = 40 + int((i / max(1, min(len(cues), max_results))) * 40)
                on_progress(pct, f"Matching cue {i + 1}/{min(len(cues), max_results)}...")

    if on_progress:
        on_progress(85, "Generating cut list...")

    cut_list = generate_broll_cutlist(cues, matches, max_entries=max_results)

    if on_progress:
        on_progress(100, f"B-roll suggestion complete: {len(cut_list.matches)} matches")

    return cut_list
