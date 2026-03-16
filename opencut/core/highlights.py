"""
OpenCut Highlight Extraction & Video Summarization

LLM-powered analysis of transcripts to identify viral/interesting clips
and generate text summaries.

Depends on: opencut.core.llm (LLM abstraction)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Highlight:
    """A single identified highlight/clip."""
    start: float
    end: float
    score: float = 0.0       # Relevance score 0-1
    reason: str = ""          # Why this is interesting
    title: str = ""           # Suggested short title

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class HighlightResult:
    """Results from highlight extraction."""
    highlights: List[Highlight] = field(default_factory=list)
    total_found: int = 0
    llm_provider: str = ""
    llm_model: str = ""


@dataclass
class Summary:
    """Video summary output."""
    text: str = ""
    bullet_points: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    word_count: int = 0


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------
def _format_transcript_for_llm(segments: List[Dict], max_chars: int = 12000) -> str:
    """
    Format transcript segments into a timestamped text block for LLM input.

    Args:
        segments: List of dicts with keys: start, end, text.
        max_chars: Truncate to this many characters to fit context window.

    Returns:
        Formatted transcript string.
    """
    lines = []
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", start)
        text = seg.get("text", "").strip()
        if text:
            s_m, s_s = int(start // 60), int(start % 60)
            e_m, e_s = int(end // 60), int(end % 60)
            lines.append(f"[{s_m:02d}:{s_s:02d} - {e_m:02d}:{e_s:02d}] {text}")

    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n[...transcript truncated...]"
    return result


# ---------------------------------------------------------------------------
# Highlight extraction
# ---------------------------------------------------------------------------
_HIGHLIGHT_SYSTEM_PROMPT = """You are a professional video editor analyzing a transcript to find the most interesting, viral, or engaging moments. You specialize in identifying clips that would perform well on social media (TikTok, YouTube Shorts, Instagram Reels).

Return your analysis as a JSON array where each object has:
- "start": start time in seconds (number)
- "end": end time in seconds (number)
- "score": relevance score 0.0-1.0 (number)
- "reason": brief explanation of why this clip is interesting (string)
- "title": short catchy title for the clip (string)

Focus on:
- Surprising or counterintuitive statements
- Funny moments or jokes with good punchlines
- Emotional peaks (excitement, passion, revelation)
- Quotable one-liners or sound bites
- Key insights or "aha" moments
- Story climaxes or dramatic reveals

IMPORTANT: Return ONLY the JSON array, no other text."""


def _parse_highlights_json(text: str) -> List[Highlight]:
    """Parse LLM response into Highlight objects. Handles various formats."""
    text = text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    try:
        data = json.loads(text)
        if isinstance(data, list):
            highlights = []
            for item in data:
                highlights.append(Highlight(
                    start=float(item.get("start", 0)),
                    end=float(item.get("end", 0)),
                    score=float(item.get("score", 0.5)),
                    reason=str(item.get("reason", "")),
                    title=str(item.get("title", "")),
                ))
            return highlights
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: find JSON array in response
    match = re.search(r"\[[\s\S]*?\]", text)
    if match:
        try:
            data = json.loads(match.group(0))
            highlights = []
            for item in data:
                if isinstance(item, dict):
                    highlights.append(Highlight(
                        start=float(item.get("start", 0)),
                        end=float(item.get("end", 0)),
                        score=float(item.get("score", 0.5)),
                        reason=str(item.get("reason", "")),
                        title=str(item.get("title", "")),
                    ))
            return highlights
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Fallback: regex for timestamp patterns
    highlights = []
    pattern = r"(\d+(?:\.\d+)?)\s*[-\u2013to]+\s*(\d+(?:\.\d+)?)"
    for m in re.finditer(pattern, text):
        try:
            start = float(m.group(1))
            end = float(m.group(2))
            if end > start:
                highlights.append(Highlight(start=start, end=end, score=0.5))
        except ValueError:
            continue

    return highlights


def extract_highlights(
    transcript_segments: List[Dict],
    max_highlights: int = 5,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    llm_config=None,
    on_progress: Optional[Callable] = None,
) -> HighlightResult:
    """
    Extract highlight clips from a transcript using LLM analysis.

    Args:
        transcript_segments: List of transcript segment dicts (start, end, text).
        max_highlights: Maximum number of highlights to return.
        min_duration: Minimum clip duration in seconds.
        max_duration: Maximum clip duration in seconds.
        llm_config: LLMConfig for the query. Uses defaults if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        HighlightResult with ranked highlights.
    """
    from opencut.core.llm import query_llm, LLMConfig

    if llm_config is None:
        llm_config = LLMConfig()

    if not transcript_segments:
        return HighlightResult()

    if on_progress:
        on_progress(10, "Formatting transcript for analysis...")

    formatted = _format_transcript_for_llm(transcript_segments)

    prompt = (
        f"Analyze this transcript and find the {max_highlights} most interesting, "
        f"viral, or engaging moments. Each clip should be between "
        f"{min_duration:.0f}-{max_duration:.0f} seconds long.\n\n"
        f"TRANSCRIPT:\n{formatted}"
    )

    if on_progress:
        on_progress(20, "Querying LLM for highlight analysis...")

    response = query_llm(
        prompt=prompt,
        config=llm_config,
        system_prompt=_HIGHLIGHT_SYSTEM_PROMPT,
        on_progress=on_progress,
    )

    if on_progress:
        on_progress(80, "Parsing highlights...")

    if response.text.startswith("LLM error:"):
        logger.error("LLM query failed: %s", response.text)
        return HighlightResult(llm_provider=response.provider, llm_model=response.model)

    highlights = _parse_highlights_json(response.text)

    # Filter by duration constraints
    filtered = []
    for h in highlights:
        if h.duration < min_duration:
            h.end = h.start + min_duration
        if h.duration > max_duration:
            h.end = h.start + max_duration
        if h.end > h.start:
            filtered.append(h)

    # Sort by score descending, take top N
    filtered.sort(key=lambda h: h.score, reverse=True)
    filtered = filtered[:max_highlights]

    if on_progress:
        on_progress(100, f"Found {len(filtered)} highlights")

    return HighlightResult(
        highlights=filtered,
        total_found=len(filtered),
        llm_provider=response.provider,
        llm_model=response.model,
    )


# ---------------------------------------------------------------------------
# Video Summarization
# ---------------------------------------------------------------------------
_SUMMARY_SYSTEM_PROMPT = """You are a professional content analyst. Summarize the given video transcript clearly and concisely. Identify the main topics, key points, and important takeaways.

Return your summary as a JSON object with:
- "summary": A paragraph summary of the content (string)
- "bullet_points": Array of key points as strings
- "topics": Array of main topic/keyword strings

IMPORTANT: Return ONLY the JSON object, no other text."""


def summarize_video(
    transcript_segments: List[Dict],
    style: str = "bullets",
    llm_config=None,
    on_progress: Optional[Callable] = None,
) -> Summary:
    """
    Generate a text summary of video content from its transcript.

    Args:
        transcript_segments: List of transcript segment dicts (start, end, text).
        style: Summary style - "bullets", "paragraph", or "detailed".
        llm_config: LLMConfig for the query. Uses defaults if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        Summary with text, bullet points, and topics.
    """
    from opencut.core.llm import query_llm, LLMConfig

    if llm_config is None:
        llm_config = LLMConfig()

    if not transcript_segments:
        return Summary(text="No transcript provided.")

    if on_progress:
        on_progress(10, "Preparing transcript for summarization...")

    formatted = _format_transcript_for_llm(transcript_segments)

    style_instructions = {
        "bullets": "Provide a concise bullet-point summary with 5-10 key points.",
        "paragraph": "Provide a 2-3 paragraph narrative summary.",
        "detailed": "Provide a detailed summary with sections for overview, key points, takeaways, and notable quotes.",
    }

    style_instruction = style_instructions.get(style, style_instructions["bullets"])

    prompt = (
        f"{style_instruction}\n\n"
        f"TRANSCRIPT:\n{formatted}"
    )

    if on_progress:
        on_progress(20, "Querying LLM for summary...")

    response = query_llm(
        prompt=prompt,
        config=llm_config,
        system_prompt=_SUMMARY_SYSTEM_PROMPT,
        on_progress=on_progress,
    )

    if on_progress:
        on_progress(80, "Parsing summary...")

    if response.text.startswith("LLM error:"):
        return Summary(text=response.text)

    text = response.text.strip()

    # Remove markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    try:
        data = json.loads(text)
        summary = Summary(
            text=str(data.get("summary", "")),
            bullet_points=data.get("bullet_points", []),
            topics=data.get("topics", []),
            word_count=len(str(data.get("summary", "")).split()),
        )
    except (json.JSONDecodeError, TypeError):
        summary = Summary(
            text=response.text,
            word_count=len(response.text.split()),
        )

    if on_progress:
        on_progress(100, "Summary complete")

    return summary
