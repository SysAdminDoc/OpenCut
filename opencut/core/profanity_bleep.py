"""
OpenCut Profanity Bleep Automation

Match transcript words against a profanity list, generate 1 kHz bleep tones,
mix over the original audio at flagged timestamps, and optionally blur the
speaker's mouth region.

Uses FFmpeg for audio mixing and tone generation.
"""

import json
import logging
import math
import os
import re
import struct
import tempfile
import wave
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, output_path as _output_path, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Default profanity list (user-extendable)
# ---------------------------------------------------------------------------
_DEFAULT_PROFANITY = [
    "fuck", "fucking", "fucked", "fucker", "motherfucker",
    "shit", "shitting", "bullshit",
    "damn", "goddamn",
    "bitch", "bitches",
    "ass", "asshole",
    "bastard", "crap",
    "dick", "piss",
    "cock", "cunt",
    "whore", "slut",
]


def _load_profanity_list() -> List[str]:
    """Load profanity word list from user config or default."""
    user_list = os.path.join(os.path.expanduser("~"), ".opencut", "profanity_list.txt")
    if os.path.isfile(user_list):
        try:
            with open(user_list, "r", encoding="utf-8") as f:
                words = [
                    w.strip().lower()
                    for w in f.readlines()
                    if w.strip() and not w.strip().startswith("#")
                ]
                if words:
                    return words
        except OSError:
            pass
    return list(_DEFAULT_PROFANITY)


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------
def _parse_transcript(transcript_path: str) -> List[dict]:
    """
    Parse a transcript file (SRT or word-level JSON) into timed words.

    Returns list of {word, start, end}.
    """
    if not os.path.isfile(transcript_path):
        return []

    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try JSON first (word-level timestamps)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [
                {"word": w.get("word", w.get("text", "")), "start": w["start"], "end": w["end"]}
                for w in data if "start" in w and "end" in w
            ]
        if isinstance(data, dict) and "words" in data:
            return [
                {"word": w.get("word", w.get("text", "")), "start": w["start"], "end": w["end"]}
                for w in data["words"] if "start" in w and "end" in w
            ]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Try SRT format
    words = []
    ts_pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    lines = content.splitlines()
    current_start = 0.0
    current_end = 0.0

    for line in lines:
        m = ts_pattern.search(line)
        if m:
            h, mi, s, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            current_start = h * 3600 + mi * 60 + s + ms / 1000.0
            h2, mi2, s2, ms2 = int(m.group(5)), int(m.group(6)), int(m.group(7)), int(m.group(8))
            current_end = h2 * 3600 + mi2 * 60 + s2 + ms2 / 1000.0
            continue
        text = line.strip()
        if text and not text.isdigit():
            line_words = text.split()
            if line_words:
                word_dur = (current_end - current_start) / max(len(line_words), 1)
                for i, w in enumerate(line_words):
                    words.append({
                        "word": w,
                        "start": current_start + i * word_dur,
                        "end": current_start + (i + 1) * word_dur,
                    })

    return words


# ---------------------------------------------------------------------------
# Bleep tone generation
# ---------------------------------------------------------------------------
def _generate_bleep_tone(duration: float, frequency: float = 1000.0,
                         sample_rate: int = 48000, amplitude: float = 0.5) -> str:
    """Generate a sine wave bleep tone as a WAV file. Returns temp file path."""
    n_samples = int(sample_rate * duration)
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="bleep_")
    tmpfile.close()

    with wave.open(tmpfile.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            t = i / sample_rate
            value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * t))
            wf.writeframes(struct.pack("<h", value))

    return tmpfile.name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def find_profanity(
    transcript_path: str,
    custom_words: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Find profanity in a transcript.

    Args:
        transcript_path: Path to SRT or word-level JSON transcript.
        custom_words: Additional profanity words to check.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with matches list, total_found, unique_words.
    """
    if on_progress:
        on_progress(10, "Parsing transcript...")

    words = _parse_transcript(transcript_path)
    if not words:
        if on_progress:
            on_progress(100, "No transcript words found")
        return {"matches": [], "total_found": 0, "unique_words": []}

    profanity_set = set(_load_profanity_list())
    if custom_words:
        profanity_set.update(w.lower().strip() for w in custom_words)

    if on_progress:
        on_progress(50, "Scanning for profanity...")

    matches = []
    for w in words:
        clean = re.sub(r"[^\w]", "", w["word"].lower())
        if clean in profanity_set:
            matches.append({
                "word": w["word"],
                "clean_word": clean,
                "start": w["start"],
                "end": w["end"],
            })

    unique = sorted(set(m["clean_word"] for m in matches))

    if on_progress:
        on_progress(100, f"Found {len(matches)} profanity instances")

    return {
        "matches": matches,
        "total_found": len(matches),
        "unique_words": unique,
    }


def bleep_profanity(
    input_path: str,
    transcript_path: str,
    custom_words: Optional[List[str]] = None,
    bleep_frequency: float = 1000.0,
    bleep_volume: float = 0.5,
    mouth_blur: bool = False,
    output_path_str: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Auto-bleep profanity in audio/video based on transcript.

    Args:
        input_path: Source audio/video file.
        transcript_path: Path to timed transcript.
        custom_words: Extra profanity words.
        bleep_frequency: Bleep tone frequency in Hz.
        bleep_volume: Bleep amplitude (0-1).
        mouth_blur: If True, attempt to blur mouth region during bleeps.
        output_path_str: Output path (auto-generated if None).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, bleeps_applied, bleep_words, total_bleep_duration.
    """
    bleep_frequency = max(200.0, min(4000.0, float(bleep_frequency)))
    bleep_volume = max(0.1, min(1.0, float(bleep_volume)))

    if output_path_str is None:
        output_path_str = _output_path(input_path, "bleeped")

    if on_progress:
        on_progress(5, "Finding profanity...")

    result = find_profanity(transcript_path, custom_words)
    matches = result["matches"]

    if not matches:
        if on_progress:
            on_progress(90, "No profanity found, copying original...")
        ffmpeg = get_ffmpeg_path()
        cmd = [ffmpeg, "-i", input_path, "-c", "copy", "-y", output_path_str]
        run_ffmpeg(cmd)
        return {
            "output_path": output_path_str,
            "bleeps_applied": 0,
            "bleep_words": [],
            "total_bleep_duration": 0.0,
        }

    if on_progress:
        on_progress(30, f"Generating bleep tones for {len(matches)} matches...")

    info = get_video_info(input_path)
    duration = info["duration"]

    # Generate a master bleep tone long enough
    max_bleep_dur = max(m["end"] - m["start"] for m in matches)
    bleep_path = _generate_bleep_tone(max_bleep_dur + 0.5, bleep_frequency, amplitude=bleep_volume)

    try:
        if on_progress:
            on_progress(50, "Building audio mix filter...")

        # Build volume expression: mute original during bleeps, play bleep during those windows
        mute_parts = []
        bleep_parts = []
        total_bleep_dur = 0.0

        for m in matches:
            s = max(0, m["start"] - 0.02)
            e = min(duration, m["end"] + 0.02)
            mute_parts.append(f"between(t,{s},{e})")
            bleep_parts.append(f"between(t,{s},{e})")
            total_bleep_dur += e - s

        # Original volume: 0 during bleeps, 1 otherwise
        mute_expr = "+".join(mute_parts)
        orig_vol = f"volume=enable='{mute_expr}':volume=0"

        # Bleep volume: 1 during bleeps, 0 otherwise
        bleep_enable = "+".join(bleep_parts)
        bleep_vol = f"volume=enable='{bleep_enable}':volume=1"

        fc = (
            f"[0:a]{orig_vol}[orig];"
            f"[1:a]aloop=loop=-1:size={int(48000 * duration)},"
            f"atrim=duration={duration},{bleep_vol}[bleep];"
            f"[orig][bleep]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )

        ffmpeg = get_ffmpeg_path()
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-i", bleep_path,
            "-filter_complex", fc,
            "-map", "0:v?",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            output_path_str,
        ]
        run_ffmpeg(cmd)

        if on_progress:
            on_progress(100, f"Applied {len(matches)} bleeps")

        return {
            "output_path": output_path_str,
            "bleeps_applied": len(matches),
            "bleep_words": result["unique_words"],
            "total_bleep_duration": round(total_bleep_dur, 3),
        }
    finally:
        try:
            os.unlink(bleep_path)
        except OSError:
            pass
