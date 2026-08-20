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
    sorted_segs = sorted(segments, key=lambda s: float(s.get("start", 0.0)))
    merged = [dict(sorted_segs[0])]

    for seg in sorted_segs[1:]:
        last = merged[-1]
        same_speaker = seg.get("speaker", "") == last.get("speaker", "")
        gap = float(seg.get("start", 0.0)) - float(last.get("end", 0.0))

        if same_speaker and gap <= gap_tolerance:
            # Extend the current segment
            last["end"] = max(float(last.get("end", 0.0)), float(seg.get("end", 0.0)))
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
    wide_track: Optional[int] = None,
    wide_every_n_cuts: int = 0,
    wide_every_seconds: float = 0.0,
    wide_duration: float = 2.0,
    cut_on_interruption: bool = True,
    speaker_min_duration: Optional[Dict[str, float]] = None,
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
        wide_track: Track index of a wide/establishing angle, if the shoot has
                    one. Required for any wide-shot cadence.
        wide_every_n_cuts: Insert a wide shot after this many speaker cuts.
                           0 disables. Cutting only between talking heads is the
                           complaint editors have about automated multicam; a
                           periodic wide is what makes the result watchable.
        wide_every_seconds: Insert a wide shot when this long has passed since
                            the last one. 0 disables. Combines with the count
                            rule — whichever triggers first.
        wide_duration: How long a inserted wide shot holds, in seconds.
        cut_on_interruption: When False, a speaker change that happens while
                             another speaker is still talking is not cut to.
                             Overlapping speech is where automated multicam
                             looks most obviously wrong.
        speaker_min_duration: Optional per-speaker minimum duration overriding
                              `min_cut_duration`, so a host who interjects
                              constantly can be given a longer floor than a
                              guest who speaks in full paragraphs.

    Returns:
        Dict with:
            "cuts": list of {"time": float, "track": int, "speaker": str, "duration": float}
            "total_cuts": int
            "speaker_to_track": the mapping used
            "wide_cuts": number of inserted wide shots
            "grammar": the effective rule set, so a caller can show what ran
    """
    min_cut_duration = max(0.0, float(min_cut_duration))
    wide_every_n_cuts = max(0, int(wide_every_n_cuts or 0))
    wide_every_seconds = max(0.0, float(wide_every_seconds or 0.0))
    wide_duration = max(0.0, float(wide_duration or 0.0))
    speaker_min_duration = dict(speaker_min_duration or {})
    grammar = {
        "min_cut_duration": min_cut_duration,
        "wide_track": wide_track,
        "wide_every_n_cuts": wide_every_n_cuts,
        "wide_every_seconds": wide_every_seconds,
        "wide_duration": wide_duration,
        "cut_on_interruption": bool(cut_on_interruption),
        "speaker_min_duration": speaker_min_duration,
    }

    if not diarization_segments:
        return {
            "cuts": [], "total_cuts": 0, "speaker_to_track": {},
            "wide_cuts": 0, "grammar": grammar,
        }

    # Auto-assign speakers if no mapping provided
    if speaker_to_track is None:
        speaker_to_track = auto_assign_speakers(diarization_segments)

    # Merge consecutive same-speaker segments
    merged = merge_diarization_segments(diarization_segments)

    # Filter out very short segments, honouring any per-speaker override.
    filtered = []
    for seg in merged:
        floor = speaker_min_duration.get(seg.get("speaker", ""), min_cut_duration)
        if (seg.get("end", 0.0) - seg.get("start", 0.0)) >= max(0.0, float(floor)):
            filtered.append(seg)

    if not cut_on_interruption:
        # Drop a switch that begins while the previous speaker is still going.
        # `merge_diarization_segments` keeps source order, so the previous kept
        # segment's end is the right thing to compare against.
        kept = []
        for seg in filtered:
            if kept and seg.get("start", 0.0) < kept[-1].get("end", 0.0):
                continue
            kept.append(seg)
        filtered = kept

    if not filtered:
        logger.warning("All diarization segments were shorter than min_cut_duration=%.2f s", min_cut_duration)
        return {
            "cuts": [], "total_cuts": 0, "speaker_to_track": speaker_to_track,
            "wide_cuts": 0, "grammar": grammar,
        }

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

    wide_cuts = 0
    if wide_track is not None and (wide_every_n_cuts or wide_every_seconds) and wide_duration > 0:
        cuts, wide_cuts = _insert_wide_shots(
            cuts,
            wide_track=int(wide_track),
            every_n_cuts=wide_every_n_cuts,
            every_seconds=wide_every_seconds,
            hold=wide_duration,
        )

    logger.info(
        "Generated %d multicam cuts (%d wide) from %d diarization segments",
        len(cuts), wide_cuts, len(diarization_segments),
    )

    return {
        "cuts": cuts,
        "total_cuts": len(cuts),
        "speaker_to_track": speaker_to_track,
        "wide_cuts": wide_cuts,
        "grammar": grammar,
    }


def _insert_wide_shots(
    cuts: List[dict],
    *,
    wide_track: int,
    every_n_cuts: int,
    every_seconds: float,
    hold: float,
) -> tuple:
    """Interleave wide-angle shots between speaker cuts.

    A wide is only inserted where the speaker shot it interrupts is long enough
    to still read after being split; cutting away from a one-second line and
    back again is worse than not cutting at all.
    """
    out: List[dict] = []
    inserted = 0
    since_count = 0
    last_wide_time = None

    for cut in cuts:
        out.append(cut)
        since_count += 1
        start = float(cut.get("time", 0.0))
        duration = float(cut.get("duration", 0.0))
        end = start + duration

        by_count = every_n_cuts and since_count >= every_n_cuts
        by_time = (
            every_seconds
            and (last_wide_time is None or (end - last_wide_time) >= every_seconds)
        )
        if not (by_count or by_time):
            continue
        # Only split a shot with room for the wide plus some speaker either side.
        if duration < hold * 2:
            continue

        wide_start = end - hold
        # Shorten the speaker shot so the wide has somewhere to live.
        cut["duration"] = round(max(0.0, wide_start - start), 4)
        out.append({
            "time": round(wide_start, 4),
            "track": wide_track,
            "speaker": "",
            "duration": round(hold, 4),
            "shot": "wide",
        })
        inserted += 1
        since_count = 0
        last_wide_time = wide_start

    return out, inserted
