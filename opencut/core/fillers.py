"""
Filler word detection and removal using Whisper word-level timestamps.

Detects common filler words (um, uh, like, so, you know, etc.) and
produces refined speech segments with those words excised.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .captions import CaptionSegment, TranscriptionResult, Word
from .silence import TimeSegment


# -------------------------------------------------------------------
# Filler word dictionary
# -------------------------------------------------------------------
# Each entry: normalised form -> set of Whisper spellings that match.
# Whisper sometimes capitalises or adds punctuation; we strip those
# before matching, but include common variants for clarity.

FILLER_WORDS: Dict[str, Set[str]] = {
    # Hesitation sounds
    "um":       {"um", "umm", "ummm"},
    "uh":       {"uh", "uhh", "uhhh"},
    "er":       {"er", "err"},
    "ah":       {"ah", "ahh"},
    "hm":       {"hm", "hmm", "hmmm"},
    "mm":       {"mm", "mmm", "mhm"},

    # Verbal fillers
    "like":     {"like"},
    "so":       {"so"},
    "basically":{"basically"},
    "actually": {"actually"},
    "literally":{"literally"},
    "honestly": {"honestly"},
    "obviously":{"obviously"},
    "right":    {"right"},
    "anyway":   {"anyway", "anyways"},
    "well":     {"well"},

    # Phrases (checked as bigrams / trigrams)
    "you know": {"you know"},
    "i mean":   {"i mean"},
    "kind of":  {"kind of", "kinda"},
    "sort of":  {"sort of", "sorta"},
}

# Split into single-word fillers and multi-word phrases
_SINGLE_FILLERS: Dict[str, Set[str]] = {}
_PHRASE_FILLERS: Dict[str, List[str]] = {}  # key -> ordered word list

for _key, _variants in FILLER_WORDS.items():
    parts = _key.split()
    if len(parts) == 1:
        _SINGLE_FILLERS[_key] = _variants
    else:
        _PHRASE_FILLERS[_key] = parts

# The "safe" fillers that are almost always filler and rarely meaningful
SAFE_FILLERS = {
    "um", "uh", "er", "ah", "hm", "mm",
}

# Context-dependent fillers that may carry meaning in some sentences
CONTEXT_FILLERS = {
    "like", "so", "right", "well", "actually", "basically",
    "literally", "honestly", "obviously", "anyway",
    "you know", "i mean", "kind of", "sort of",
}


@dataclass
class FillerHit:
    """A detected filler word instance with its location."""
    text: str           # The original text from Whisper
    filler_key: str     # Normalised filler category (e.g. "um", "you know")
    start: float        # Start time in seconds
    end: float          # End time in seconds
    confidence: float   # Whisper's confidence for this word
    safe: bool          # True if this is a "safe" (always-filler) word

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class FillerAnalysis:
    """Complete analysis of filler words in a transcription."""
    hits: List[FillerHit]
    total_filler_time: float
    filler_counts: Dict[str, int]
    total_words: int
    filler_percentage: float  # Percentage of words that are fillers

    def get_hits_for(self, filler_keys: List[str]) -> List[FillerHit]:
        """Get hits for specific filler types."""
        key_set = set(filler_keys)
        return [h for h in self.hits if h.filler_key in key_set]


def _normalise(text: str) -> str:
    """Strip punctuation and lowercase for matching."""
    return text.strip().lower().strip(".,!?;:\"'()-")


def detect_fillers(
    transcription: TranscriptionResult,
    include_context_fillers: bool = True,
    custom_words: Optional[List[str]] = None,
    min_confidence: float = 0.0,
) -> FillerAnalysis:
    """
    Detect filler words in a transcription using word-level timestamps.

    Args:
        transcription: Whisper transcription with word timestamps.
        include_context_fillers: Include context-dependent fillers (like, so, etc.)
        custom_words: Additional custom filler words to detect.
        min_confidence: Minimum Whisper confidence to consider a word.

    Returns:
        FillerAnalysis with all detected filler instances.
    """
    # Build the active filler set
    active_singles: Dict[str, str] = {}  # normalised_word -> filler_key
    for key, variants in _SINGLE_FILLERS.items():
        if key in SAFE_FILLERS or include_context_fillers:
            for v in variants:
                active_singles[v] = key

    active_phrases: Dict[str, List[str]] = {}
    if include_context_fillers:
        active_phrases = dict(_PHRASE_FILLERS)

    # Add custom words
    if custom_words:
        for cw in custom_words:
            norm = _normalise(cw)
            if norm:
                parts = norm.split()
                if len(parts) == 1:
                    active_singles[norm] = norm
                    # Also add it to the SAFE set since user explicitly asked
                    SAFE_FILLERS.add(norm)
                else:
                    active_phrases[norm] = parts

    hits: List[FillerHit] = []
    total_words = 0

    for seg in transcription.segments:
        if not seg.words:
            continue

        words = seg.words
        total_words += len(words)
        i = 0

        while i < len(words):
            w = words[i]

            if w.confidence < min_confidence:
                i += 1
                continue

            norm = _normalise(w.text)

            # Check multi-word phrases first (greedy)
            phrase_matched = False
            for pkey, pwords in active_phrases.items():
                plen = len(pwords)
                if i + plen <= len(words):
                    candidate = [_normalise(words[i + j].text) for j in range(plen)]
                    if candidate == pwords:
                        # Phrase match
                        phrase_start = words[i].start
                        phrase_end = words[i + plen - 1].end
                        phrase_text = " ".join(words[i + j].text for j in range(plen))
                        avg_conf = sum(words[i + j].confidence for j in range(plen)) / plen

                        hits.append(FillerHit(
                            text=phrase_text,
                            filler_key=pkey,
                            start=phrase_start,
                            end=phrase_end,
                            confidence=avg_conf,
                            safe=(pkey in SAFE_FILLERS),
                        ))
                        i += plen
                        phrase_matched = True
                        break

            if phrase_matched:
                continue

            # Check single-word fillers
            if norm in active_singles:
                hits.append(FillerHit(
                    text=w.text,
                    filler_key=active_singles[norm],
                    start=w.start,
                    end=w.end,
                    confidence=w.confidence,
                    safe=(active_singles[norm] in SAFE_FILLERS),
                ))

            i += 1

    # Build summary
    filler_counts: Dict[str, int] = {}
    total_filler_time = 0.0
    for h in hits:
        filler_counts[h.filler_key] = filler_counts.get(h.filler_key, 0) + 1
        total_filler_time += h.duration

    filler_word_count = len(hits)
    filler_pct = (filler_word_count / total_words * 100) if total_words > 0 else 0.0

    return FillerAnalysis(
        hits=hits,
        total_filler_time=round(total_filler_time, 3),
        filler_counts=filler_counts,
        total_words=total_words,
        filler_percentage=round(filler_pct, 1),
    )


def remove_fillers_from_segments(
    segments: List[TimeSegment],
    filler_hits: List[FillerHit],
    padding: float = 0.03,
    min_gap: float = 0.08,
) -> List[TimeSegment]:
    """
    Remove filler word time ranges from speech segments.

    Takes the existing speech segments and "punches holes" where
    filler words were detected, producing refined segments.

    Args:
        segments: Original speech segments from silence detection.
        filler_hits: Filler word hits to remove.
        padding: Extra padding (seconds) around each filler cut.
        min_gap: Minimum gap between segments; smaller gaps merge.

    Returns:
        New list of TimeSegment with filler words excised.
    """
    if not filler_hits:
        return list(segments)

    # Build sorted list of time ranges to cut
    cuts = []
    for h in sorted(filler_hits, key=lambda x: x.start):
        cut_start = max(0, h.start - padding)
        cut_end = h.end + padding
        cuts.append((cut_start, cut_end))

    # Merge overlapping cuts
    merged_cuts = [cuts[0]]
    for cs, ce in cuts[1:]:
        prev_s, prev_e = merged_cuts[-1]
        if cs <= prev_e + min_gap:
            merged_cuts[-1] = (prev_s, max(prev_e, ce))
        else:
            merged_cuts.append((cs, ce))

    # Subtract cuts from each segment
    result = []
    for seg in sorted(segments, key=lambda s: s.start):
        remaining = [(seg.start, seg.end)]

        for cut_s, cut_e in merged_cuts:
            new_remaining = []
            for rs, re in remaining:
                if cut_e <= rs or cut_s >= re:
                    # No overlap
                    new_remaining.append((rs, re))
                elif cut_s <= rs and cut_e >= re:
                    # Cut completely covers this segment piece - remove it
                    pass
                elif cut_s <= rs:
                    # Cut trims the start
                    if cut_e < re:
                        new_remaining.append((cut_e, re))
                elif cut_e >= re:
                    # Cut trims the end
                    if cut_s > rs:
                        new_remaining.append((rs, cut_s))
                else:
                    # Cut splits the segment in two
                    new_remaining.append((rs, cut_s))
                    new_remaining.append((cut_e, re))
            remaining = new_remaining

        for rs, re in remaining:
            if re - rs >= 0.05:  # Skip tiny fragments
                result.append(TimeSegment(
                    start=round(rs, 4),
                    end=round(re, 4),
                    label="speech",
                ))

    # Sort and merge segments that are very close together
    result.sort(key=lambda s: s.start)
    if len(result) > 1:
        merged = [result[0]]
        for seg in result[1:]:
            prev = merged[-1]
            if seg.start - prev.end < min_gap:
                merged[-1] = TimeSegment(
                    start=prev.start,
                    end=seg.end,
                    label="speech",
                )
            else:
                merged.append(seg)
        result = merged

    return result
