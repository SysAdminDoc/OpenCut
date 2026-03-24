"""
YouTube Chapter Generation

Uses LLM to analyze transcript and identify topic boundaries,
then generates chapter timestamps ready to paste into YouTube descriptions.
"""

import json
import logging
import re
from typing import List, Optional

from .llm import LLMConfig, query_llm

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Time string helpers
# ---------------------------------------------------------------------------

def _time_str_to_seconds(time_str: str) -> Optional[float]:
    """
    Convert a time string (M:SS or H:MM:SS) to seconds.

    Returns None if the string cannot be parsed.
    """
    time_str = time_str.strip()
    parts = time_str.split(":")
    try:
        if len(parts) == 2:
            # M:SS
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            # H:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        pass
    return None


def _seconds_to_time_str(seconds: float) -> str:
    """Convert seconds to M:SS or H:MM:SS string."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------

def _build_condensed_transcript(segments: List[dict], max_chars: int = 8000) -> str:
    """
    Build a condensed transcript string with timestamps.

    Format: "[M:SS] text\n[M:SS] text\n..."
    Truncates to max_chars to stay within LLM context limits.
    """
    lines = []
    for seg in segments:
        start = seg.get("start", 0.0)
        text = seg.get("text", "").strip()
        if text:
            ts = _seconds_to_time_str(start)
            lines.append(f"[{ts}] {text}")
    full = "\n".join(lines)
    if len(full) > max_chars:
        full = full[:max_chars] + "\n... (transcript truncated)"
    return full


# ---------------------------------------------------------------------------
# LLM-based chapter generation
# ---------------------------------------------------------------------------

def _parse_llm_chapters(llm_text: str) -> List[dict]:
    """
    Extract a JSON array from LLM response text.

    Returns list of {"time": "M:SS", "title": str} dicts, or empty list.
    """
    # Try to find a JSON array anywhere in the response
    match = re.search(r"\[.*?\]", llm_text, re.DOTALL)
    if not match:
        logger.warning("No JSON array found in LLM chapter response")
        return []
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM chapters JSON: %s", exc)
        return []

    chapters = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        time_val = item.get("time") or item.get("timestamp") or item.get("start") or ""
        title = item.get("title") or item.get("chapter") or item.get("label") or ""
        if time_val and title:
            chapters.append({"time": str(time_val).strip(), "title": str(title).strip()})
    return chapters


_SYSTEM_PROMPT = (
    "You are a professional YouTube video editor. "
    "Your task is to analyze transcripts and identify meaningful topic changes "
    "to generate YouTube chapter timestamps."
)

_CHAPTER_PROMPT_TEMPLATE = """\
Analyze this video transcript and identify the main topic changes.
Generate YouTube chapter timestamps. Return ONLY a valid JSON array.

Rules:
- Output format: [{{"time": "M:SS", "title": "Chapter Title"}}, ...]
- Use M:SS format for times under 1 hour, H:MM:SS for longer
- First chapter MUST be at "0:00" with title "Intro"
- Maximum {max_chapters} chapters
- Chapter titles should be concise (3-6 words)
- Only add a chapter when the topic genuinely changes

Transcript:
{transcript}

Return only the JSON array, no other text.
"""


def _generate_chapters_llm(
    segments: List[dict],
    llm_config: LLMConfig,
    max_chapters: int,
    min_chapter_duration: float,
) -> List[dict]:
    """Use LLM to generate chapters. Returns list of chapter dicts."""
    transcript = _build_condensed_transcript(segments)
    if not transcript:
        return []

    prompt = _CHAPTER_PROMPT_TEMPLATE.format(
        max_chapters=max_chapters,
        transcript=transcript,
    )

    logger.info("Querying LLM for chapter generation (%s/%s)", llm_config.provider, llm_config.model)
    response = query_llm(prompt, config=llm_config, system_prompt=_SYSTEM_PROMPT)

    raw_chapters = _parse_llm_chapters(response.text)
    if not raw_chapters:
        return []

    # Convert time strings to seconds and validate
    valid = []
    for ch in raw_chapters:
        secs = _time_str_to_seconds(ch["time"])
        if secs is None:
            continue
        valid.append({"time_str": ch["time"], "seconds": secs, "title": ch["title"]})

    # Sort by time
    valid.sort(key=lambda c: c["seconds"])
    return valid


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _generate_chapters_heuristic(
    segments: List[dict],
    max_chapters: int,
    min_chapter_duration: float,
) -> List[dict]:
    """
    Fallback chapter generation without LLM.

    Splits at long pauses (>3 s gap) between segments, or every ~200 words.
    """
    if not segments:
        return []

    chapters = [{"time_str": "0:00", "seconds": 0.0, "title": "Intro"}]

    word_count = 0
    WORDS_PER_CHAPTER = 200
    PAUSE_THRESHOLD = 3.0  # seconds

    for i, seg in enumerate(segments[1:], start=1):
        prev_end = segments[i - 1].get("end", 0.0)
        curr_start = seg.get("start", 0.0)
        gap = curr_start - prev_end
        word_count += len(seg.get("text", "").split())

        is_pause_boundary = gap >= PAUSE_THRESHOLD
        is_word_boundary = word_count >= WORDS_PER_CHAPTER

        if is_pause_boundary or is_word_boundary:
            if curr_start - chapters[-1]["seconds"] >= min_chapter_duration:
                ts = _seconds_to_time_str(curr_start)
                chapter_num = len(chapters)
                chapters.append({
                    "time_str": ts,
                    "seconds": curr_start,
                    "title": f"Part {chapter_num}",
                })
                word_count = 0

        if len(chapters) >= max_chapters:
            break

    return chapters


# ---------------------------------------------------------------------------
# Filter and format
# ---------------------------------------------------------------------------

def _filter_chapters(chapters: List[dict], min_chapter_duration: float) -> List[dict]:
    """Remove chapters that are too close together."""
    if not chapters:
        return []
    filtered = [chapters[0]]
    for ch in chapters[1:]:
        if ch["seconds"] - filtered[-1]["seconds"] >= min_chapter_duration:
            filtered.append(ch)
    return filtered


def _build_description_block(chapters: List[dict]) -> str:
    """Build the YouTube description block string."""
    lines = [f"{ch['time_str']} {ch['title']}" for ch in chapters]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_chapters(
    segments: List[dict],
    llm_config: Optional[LLMConfig] = None,
    max_chapters: int = 15,
    min_chapter_duration: float = 30.0,
) -> dict:
    """
    Generate YouTube chapters from transcript segments.

    Attempts LLM-based topic detection first, falls back to heuristic
    (long-pause / word-count splitting) if LLM is unavailable or fails.

    Args:
        segments: List of transcript segment dicts with "text", "start", "end".
        llm_config: LLMConfig to use. If None, defaults to Ollama llama3.
        max_chapters: Maximum number of chapters to generate.
        min_chapter_duration: Minimum seconds between chapters.

    Returns:
        Dict with:
            "chapters": list of {"time_str", "seconds", "title"}
            "description_block": ready-to-paste YouTube description string
            "source": "llm" or "heuristic"
    """
    chapters = []
    source = "heuristic"

    if llm_config is None:
        llm_config = LLMConfig()

    # Attempt LLM generation
    try:
        llm_chapters = _generate_chapters_llm(
            segments, llm_config, max_chapters, min_chapter_duration
        )
        if llm_chapters:
            chapters = llm_chapters
            source = "llm"
            logger.info("LLM generated %d chapters", len(chapters))
    except Exception as exc:
        logger.warning("LLM chapter generation failed, using heuristic: %s", exc)

    # Heuristic fallback
    if not chapters:
        logger.info("Using heuristic chapter generation")
        chapters = _generate_chapters_heuristic(segments, max_chapters, min_chapter_duration)

    # Ensure first chapter is at 0:00
    if not chapters or chapters[0]["seconds"] != 0.0:
        chapters.insert(0, {"time_str": "0:00", "seconds": 0.0, "title": "Intro"})

    # Filter too-close chapters and cap count
    chapters = _filter_chapters(chapters, min_chapter_duration)
    chapters = chapters[:max_chapters]

    return {
        "chapters": chapters,
        "description_block": _build_description_block(chapters),
        "source": source,
    }
