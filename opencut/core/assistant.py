"""Sequence Assistant — "what should I edit next?" heuristic engine.

v1.10.0, feature E. Given a Premiere sequence snapshot (the dict that
ExtendScript ``ocGetSequenceInfo`` already returns), emit ranked
suggestions with concrete, one-click actions. The goal isn't to be
smart — it's to notice the *obvious* next edit the user hasn't gotten
to yet (3 minutes of dead air on track 2, no captions, LUFS all over
the place).

Each suggestion:

* ``id``           — stable id for dismiss state
* ``title``        — one-line label
* ``why``          — short explanation of the signal
* ``confidence``   — 0..1 internal score for ranking
* ``action``       — ``{endpoint, payload}`` the panel can dispatch
* ``preview_data`` — optional small payload for preview rendering
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("opencut")


def _num(v, default=0.0) -> float:
    try:
        f = float(v)
        if f != f:  # NaN
            return default
        return f
    except (TypeError, ValueError):
        return default


def _pick_clip_path(seq_info: Dict[str, Any]) -> str:
    """Find the most representative clip path in the sequence snapshot.

    Prefers the longest clip on V1 (the 'A-roll'); falls back to the
    first clip on any track.
    """
    best_path = ""
    best_duration = -1.0
    tracks = seq_info.get("tracks") or []
    for track in tracks:
        tname = track.get("name", "")
        clips = track.get("clips") or []
        for clip in clips:
            path = (clip.get("source_path") or clip.get("path")
                    or clip.get("filepath") or "")
            if not path:
                continue
            dur = _num(clip.get("duration"), 0.0)
            is_v1 = "V1" in tname or tname.startswith("Video 1")
            # Prefer V1 clips, then longest
            score = dur + (10000 if is_v1 else 0)
            if score > best_duration:
                best_duration = score
                best_path = path
    return best_path


def _count_gaps(seq_info: Dict[str, Any], min_sec: float = 0.8) -> int:
    """Count inter-clip gaps on audio tracks > *min_sec* seconds.

    Signal for "run silence removal on this sequence".
    """
    gaps = 0
    for track in seq_info.get("tracks") or []:
        if track.get("media_type") and track["media_type"].lower() != "audio":
            continue
        clips = track.get("clips") or []
        if len(clips) < 2:
            continue
        # Sort by start, accepting either seconds or ticks-as-string
        def _s(c): return _num(c.get("start", c.get("in", 0)))
        sorted_clips = sorted(clips, key=_s)
        for i in range(1, len(sorted_clips)):
            prev_end = _num(sorted_clips[i - 1].get("end",
                            _s(sorted_clips[i - 1]) +
                            _num(sorted_clips[i - 1].get("duration"))))
            next_start = _s(sorted_clips[i])
            if next_start - prev_end > min_sec:
                gaps += 1
    return gaps


def _has_captions_track(seq_info: Dict[str, Any]) -> bool:
    """Heuristic: does any track's name look like a caption track?"""
    for track in seq_info.get("tracks") or []:
        name = (track.get("name") or "").lower()
        if "caption" in name or "subtitle" in name or "cc" in name:
            clips = track.get("clips") or []
            if clips:
                return True
    return False


def _audio_track_count(seq_info: Dict[str, Any]) -> int:
    count = 0
    for track in seq_info.get("tracks") or []:
        mt = (track.get("media_type") or "").lower()
        if mt == "audio" and (track.get("clips") or []):
            count += 1
    return count


def _total_duration(seq_info: Dict[str, Any]) -> float:
    dur = _num(seq_info.get("duration"))
    if dur > 0:
        return dur
    max_end = 0.0
    for track in seq_info.get("tracks") or []:
        for clip in track.get("clips") or []:
            end = _num(clip.get("end"))
            if end > max_end:
                max_end = end
    return max_end


def analyze_sequence(seq_info: Dict[str, Any],
                     dismissed_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Analyze *seq_info* and return ranked suggestions.

    *dismissed_ids* lets the panel persistently hide suggestions the
    user has already said no to in this session.
    """
    dismissed = set(dismissed_ids or [])
    out: List[Dict[str, Any]] = []

    if not seq_info or seq_info.get("error"):
        return out

    clip_path = _pick_clip_path(seq_info)
    duration = _total_duration(seq_info)
    tracks = seq_info.get("tracks") or []

    # 1. Dead air — silence removal
    gaps = _count_gaps(seq_info, min_sec=0.8)
    if gaps >= 3 and "silence-dead-air" not in dismissed and clip_path:
        out.append({
            "id": "silence-dead-air",
            "title": f"Remove {gaps} dead-air gaps",
            "why": f"Found {gaps} silent gaps >800ms across your audio tracks. "
                   "Running Silence Removal keeps the takes you want and cuts the pauses.",
            "confidence": min(0.4 + (gaps * 0.05), 0.95),
            "action": {
                "endpoint": "/silence",
                "payload": {"filepath": clip_path},
            },
        })

    # 2. No captions track present -> suggest Generate Captions
    if not _has_captions_track(seq_info) and duration > 30 and clip_path \
            and "generate-captions" not in dismissed:
        out.append({
            "id": "generate-captions",
            "title": "Generate captions for this sequence",
            "why": f"No caption or subtitle track detected across {len(tracks)} tracks. "
                   "Running Transcription produces an SRT you can burn in or add as a native Premiere caption track.",
            "confidence": 0.7,
            "action": {
                "endpoint": "/transcript",
                "payload": {"filepath": clip_path, "model": "base",
                            "export_format": "srt"},
            },
        })

    # 3. Multiple audio tracks active -> suggest loudness match
    audio_tracks = _audio_track_count(seq_info)
    if audio_tracks >= 2 and clip_path and "loudness-match" not in dismissed:
        out.append({
            "id": "loudness-match",
            "title": f"Match loudness across {audio_tracks} audio tracks",
            "why": "Multiple audio tracks with content means mic levels probably "
                   "differ between speakers/clips. Loudness Match normalizes them to a common LUFS target.",
            "confidence": 0.55 + 0.05 * min(audio_tracks - 2, 3),
            "action": {
                "endpoint": "/audio/normalize",
                "payload": {"filepath": clip_path, "target_lufs": -14},
            },
        })

    # 4. Long sequence + no chapters file -> suggest chapter generation
    if duration > 600 and clip_path and "generate-chapters" not in dismissed:
        out.append({
            "id": "generate-chapters",
            "title": "Generate YouTube chapters",
            "why": f"Sequence is {int(duration/60)} min long. Viewers expect "
                   "chapter markers on content over 10 minutes — OpenCut can generate them from the transcript.",
            "confidence": 0.5 + min(0.2, duration / 7200.0),
            "action": {
                "endpoint": "/captions/chapters",
                "payload": {"filepath": clip_path},
            },
        })

    # 5. Very long sequence -> suggest the full interview polish
    if duration > 900 and clip_path and _has_captions_track(seq_info) is False \
            and "interview-polish" not in dismissed and gaps >= 5:
        # Only pitch the big pipeline when it'll clearly pay off.
        out.append({
            "id": "interview-polish",
            "title": "Run the full Interview Polish pipeline",
            "why": f"{int(duration/60)} min of footage, {gaps} dead-air gaps, "
                   "and no captions yet. One click runs silence removal + filler cut + "
                   "repeated-take detection + captions + chapters together.",
            "confidence": 0.75,
            "action": {
                "endpoint": "/interview-polish",
                "payload": {"filepath": clip_path},
            },
        })

    # Sort by confidence descending; cap at 5 to avoid a wall of suggestions
    out.sort(key=lambda s: s["confidence"], reverse=True)
    return out[:5]
