"""
Multicam Podcast Auto-Switching

Uses speaker diarization results to generate cut decisions
for multicam editing — cut to the camera assigned to whoever is speaking.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Segment merging
# ---------------------------------------------------------------------------

def merge_diarization_segments(
    segments: List[dict],
    gap_tolerance: float = 0.5,
) -> List[dict]:
    """
    Merge consecutive segments from the same speaker separated by a short gap.

    Args:
        segments: List of {"speaker": str, "start": float, "end": float}.
        gap_tolerance: Maximum gap in seconds between same-speaker segments
                       that will be merged. Default 0.5 s.

    Returns:
        New list of merged segments, same format as input.
    """
    if not segments:
        return []

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: s["start"])
    merged = [dict(sorted_segs[0])]

    for seg in sorted_segs[1:]:
        last = merged[-1]
        same_speaker = seg["speaker"] == last["speaker"]
        gap = seg["start"] - last["end"]

        if same_speaker and gap <= gap_tolerance:
            # Extend the current segment
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(dict(seg))

    return merged


# ---------------------------------------------------------------------------
# Speaker → track assignment
# ---------------------------------------------------------------------------

def auto_assign_speakers(segments: List[dict]) -> Dict[str, int]:
    """
    Assign speakers to video track indices in order of first appearance.

    Args:
        segments: List of diarization segment dicts.

    Returns:
        Dict mapping speaker label → 0-based track index.
        E.g. {"SPEAKER_00": 0, "SPEAKER_01": 1}
    """
    assignment: Dict[str, int] = {}
    next_track = 0

    for seg in sorted(segments, key=lambda s: s.get("start", 0.0)):
        speaker = seg.get("speaker", "")
        if speaker and speaker not in assignment:
            assignment[speaker] = next_track
            next_track += 1

    return assignment


# ---------------------------------------------------------------------------
# Cut generation
# ---------------------------------------------------------------------------

def generate_multicam_cuts(
    diarization_segments: List[dict],
    speaker_to_track: Optional[Dict[str, int]] = None,
    min_cut_duration: float = 1.0,
) -> dict:
    """
    Generate multicam cut decisions from speaker diarization data.

    Each cut represents a point in time where the active camera should switch
    to the one assigned to the current speaker.

    Args:
        diarization_segments: List of {"speaker": str, "start": float, "end": float}.
        speaker_to_track: Dict mapping speaker label → video track index (0-based).
                          If None, auto-assigns based on order of first appearance.
        min_cut_duration: Minimum segment duration in seconds to keep as a cut.
                          Shorter segments are dropped. Default 1.0 s.

    Returns:
        Dict with:
            "cuts": list of {"time": float, "track": int, "speaker": str, "duration": float}
            "total_cuts": int
            "speaker_to_track": the mapping used
    """
    if not diarization_segments:
        return {"cuts": [], "total_cuts": 0, "speaker_to_track": {}}

    # Auto-assign speakers if no mapping provided
    if speaker_to_track is None:
        speaker_to_track = auto_assign_speakers(diarization_segments)

    # Merge consecutive same-speaker segments
    merged = merge_diarization_segments(diarization_segments)

    # Filter out very short segments
    filtered = [
        seg for seg in merged
        if (seg.get("end", 0.0) - seg.get("start", 0.0)) >= min_cut_duration
    ]

    if not filtered:
        logger.warning("All diarization segments were shorter than min_cut_duration=%.2f s", min_cut_duration)
        return {"cuts": [], "total_cuts": 0, "speaker_to_track": speaker_to_track}

    cuts = []
    for seg in filtered:
        speaker = seg.get("speaker", "")
        track = speaker_to_track.get(speaker, 0)
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        duration = end - start

        cuts.append({
            "time": round(start, 4),
            "track": track,
            "speaker": speaker,
            "duration": round(duration, 4),
        })

    logger.info(
        "Generated %d multicam cuts from %d diarization segments",
        len(cuts), len(diarization_segments),
    )

    return {
        "cuts": cuts,
        "total_cuts": len(cuts),
        "speaker_to_track": speaker_to_track,
    }
