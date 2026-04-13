"""
OpenCut SDH / HoH Subtitle Formatting

Formats SRT subtitle files with SDH (Subtitles for the Deaf and
Hard of Hearing) conventions:

- Speaker labels from diarization data: ``[SPEAKER 1]: text``
- Sound effect descriptions: ``[music]``, ``[applause]``, ``[laughter]``
- Uppercase speaker names, bracketed non-speech cues

No additional dependencies required -- pure Python SRT parsing.
"""

import logging
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")

# Common sound-effect cue patterns found in transcripts
_SOUND_CUES = [
    (re.compile(r"\b(music playing|♪|♫|🎵)\b", re.IGNORECASE), "[music]"),
    (re.compile(r"\b(applause|clapping)\b", re.IGNORECASE), "[applause]"),
    (re.compile(r"\b(laughter|laughing|laughs)\b", re.IGNORECASE), "[laughter]"),
    (re.compile(r"\b(silence|quiet)\b", re.IGNORECASE), "[silence]"),
    (re.compile(r"\b(doorbell|door bell|door knock|knocking)\b", re.IGNORECASE), "[knocking]"),
    (re.compile(r"\b(phone ringing|phone rings|ringtone)\b", re.IGNORECASE), "[phone ringing]"),
    (re.compile(r"\b(gunshot|gun shot|explosion)\b", re.IGNORECASE), "[explosion]"),
    (re.compile(r"\b(crying|sobbing|sobs)\b", re.IGNORECASE), "[crying]"),
    (re.compile(r"\b(cheering|cheers)\b", re.IGNORECASE), "[cheering]"),
    (re.compile(r"\b(sighing|sighs)\b", re.IGNORECASE), "[sighs]"),
]


# -----------------------------------------------------------------------
# SRT parsing / writing helpers
# -----------------------------------------------------------------------

def _parse_srt(srt_text: str) -> List[dict]:
    """Parse SRT text into a list of subtitle entries.

    Each entry: {"index": int, "start": str, "end": str, "text": str}
    """
    entries = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        timing = lines[1].strip()
        match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,.:]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.:]\d{3})",
            timing,
        )
        if not match:
            continue
        text = "\n".join(lines[2:])
        entries.append({
            "index": index,
            "start": match.group(1),
            "end": match.group(2),
            "text": text,
        })
    return entries


def _write_srt(entries: List[dict]) -> str:
    """Convert list of subtitle entries back to SRT text."""
    parts = []
    for i, e in enumerate(entries, 1):
        parts.append(f"{i}\n{e['start']} --> {e['end']}\n{e['text']}\n")
    return "\n".join(parts)


def _srt_ts_to_seconds(ts: str) -> float:
    """Convert SRT timestamp to seconds. Handles both , and . as ms separator."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


# -----------------------------------------------------------------------
# Sound cue detection
# -----------------------------------------------------------------------

def _detect_sound_cues(text: str) -> str:
    """Replace transcript sound markers with SDH-formatted cues."""
    # Skip if already has SDH-style brackets
    if re.match(r"^\[.+\]$", text.strip()):
        return text

    for pattern, replacement in _SOUND_CUES:
        if pattern.search(text):
            # If the entire line is a sound cue, replace it
            cleaned = pattern.sub("", text).strip()
            if not cleaned or len(cleaned) < 3:
                return replacement
            # Otherwise prepend the cue
            text = f"{replacement} {text}"
            break
    return text


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def format_sdh(
    srt_path: str,
    diarization_data: Optional[List[dict]] = None,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Format an SRT file with SDH conventions.

    Args:
        srt_path: Path to input SRT file.
        diarization_data: Optional list of dicts with "start", "end",
                          "speaker" keys for speaker label assignment.
        output_path: Explicit output path; auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with "output_path" and "entries_formatted".
    """
    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    if on_progress:
        on_progress(5, "Reading subtitle file...")

    with open(srt_path, "r", encoding="utf-8-sig") as f:
        srt_text = f.read()

    entries = _parse_srt(srt_text)
    if not entries:
        raise ValueError("No subtitle entries found in SRT file")

    # Build speaker lookup from diarization data
    speaker_map: List[Tuple[float, float, str]] = []
    if diarization_data:
        for d in diarization_data:
            speaker_map.append((
                float(d.get("start", 0)),
                float(d.get("end", 0)),
                str(d.get("speaker", "SPEAKER")),
            ))
        speaker_map.sort(key=lambda x: x[0])

    if on_progress:
        on_progress(20, "Formatting SDH subtitles...")

    formatted_count = 0
    total = len(entries)

    for idx, entry in enumerate(entries):
        text = entry["text"]
        modified = False

        # Add speaker label if diarization data available
        if speaker_map:
            entry_start = _srt_ts_to_seconds(entry["start"])
            speaker = _find_speaker(entry_start, speaker_map)
            if speaker:
                label = f"[{speaker.upper()}]:"
                # Don't add if already has a speaker label
                if not re.match(r"^\[.+\]:", text):
                    text = f"{label} {text}"
                    modified = True

        # Detect and format sound effect cues
        new_text = _detect_sound_cues(text)
        if new_text != text:
            text = new_text
            modified = True

        entry["text"] = text
        if modified:
            formatted_count += 1

        if on_progress and total > 0 and idx % max(1, total // 10) == 0:
            pct = 20 + int((idx / total) * 70)
            on_progress(pct, f"Processing entry {idx + 1}/{total}...")

    # Write output
    if output_path is None:
        base = os.path.splitext(srt_path)[0]
        output_path = f"{base}_sdh.srt"

    srt_output = _write_srt(entries)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_output)

    if on_progress:
        on_progress(100, "SDH formatting complete")

    return {
        "output_path": output_path,
        "entries_formatted": formatted_count,
        "total_entries": total,
    }


def _find_speaker(
    timestamp: float,
    speaker_map: List[Tuple[float, float, str]],
) -> Optional[str]:
    """Find the speaker active at the given timestamp."""
    for start, end, speaker in speaker_map:
        if start <= timestamp <= end:
            return speaker
    # If no exact match, find the closest preceding speaker
    best = None
    for start, end, speaker in speaker_map:
        if start <= timestamp:
            best = speaker
        else:
            break
    return best


def add_speaker_labels(
    srt_path: str,
    speakers: Dict[str, str],
    output_path: Optional[str] = None,
) -> dict:
    """
    Add speaker labels to an SRT file.

    Args:
        srt_path: Path to input SRT file.
        speakers: Dict mapping time ranges to speaker names.
                  Keys are "start-end" strings (e.g. "0.0-5.5"),
                  values are speaker names.
        output_path: Explicit output path; auto-generated if None.

    Returns:
        dict with "output_path" and "labels_added".
    """
    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    with open(srt_path, "r", encoding="utf-8-sig") as f:
        srt_text = f.read()

    entries = _parse_srt(srt_text)
    if not entries:
        raise ValueError("No subtitle entries found in SRT file")

    # Parse speaker dict into lookup list
    speaker_ranges: List[Tuple[float, float, str]] = []
    for time_range, speaker_name in speakers.items():
        parts = str(time_range).split("-")
        if len(parts) == 2:
            try:
                s = float(parts[0])
                e = float(parts[1])
                speaker_ranges.append((s, e, speaker_name))
            except ValueError:
                continue
    speaker_ranges.sort(key=lambda x: x[0])

    labels_added = 0
    for entry in entries:
        entry_start = _srt_ts_to_seconds(entry["start"])
        speaker = _find_speaker(entry_start, speaker_ranges)
        if speaker and not re.match(r"^\[.+\]:", entry["text"]):
            entry["text"] = f"[{speaker.upper()}]: {entry['text']}"
            labels_added += 1

    if output_path is None:
        base = os.path.splitext(srt_path)[0]
        output_path = f"{base}_speakers.srt"

    srt_output = _write_srt(entries)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_output)

    return {
        "output_path": output_path,
        "labels_added": labels_added,
        "total_entries": len(entries),
    }
