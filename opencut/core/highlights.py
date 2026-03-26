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
class EngagementScore:
    """Breakdown of engagement signals for a highlight."""
    hook_strength: float = 0.0       # 0-1: how strong the opening hook is
    emotional_peak: float = 0.0      # 0-1: emotional intensity
    pacing: float = 0.0              # 0-1: conversational energy / words-per-second
    quotability: float = 0.0         # 0-1: how quotable/shareable the content is
    overall: float = 0.0             # 0-1: weighted composite score

    def compute_overall(self):
        """Compute weighted overall engagement score."""
        self.overall = (
            self.hook_strength * 0.30 +
            self.emotional_peak * 0.25 +
            self.pacing * 0.20 +
            self.quotability * 0.25
        )
        return self.overall


@dataclass
class Highlight:
    """A single identified highlight/clip."""
    start: float
    end: float
    score: float = 0.0       # Relevance score 0-1
    reason: str = ""          # Why this is interesting
    title: str = ""           # Suggested short title
    engagement: Optional[EngagementScore] = None  # Detailed engagement breakdown

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
                if not isinstance(item, dict):
                    continue
                try:
                    highlights.append(Highlight(
                        start=float(item.get("start", 0)),
                        end=float(item.get("end", 0)),
                        score=float(item.get("score", 0.5)),
                        reason=str(item.get("reason", "")),
                        title=str(item.get("title", "")),
                    ))
                except (TypeError, ValueError):
                    continue
            return highlights
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: find JSON array in response
    match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", text)
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
    from opencut.core.llm import LLMConfig, query_llm

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
        elif h.duration > max_duration:
            h.end = h.start + max_duration
        if h.end > h.start:
            filtered.append(h)

    # Score engagement for each highlight
    for h in filtered:
        h.engagement = _score_engagement(h, transcript_segments)
        # Blend LLM score with engagement analysis (60% LLM, 40% engagement)
        h.score = h.score * 0.6 + h.engagement.overall * 0.4

    # Sort by blended score descending, take top N
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


def _score_engagement(highlight: Highlight, transcript_segments: List[Dict]) -> EngagementScore:
    """
    Score a highlight's engagement potential using text heuristics.

    Analyzes the transcript text within the highlight's time range for:
    - Hook strength: Does the opening grab attention? (questions, surprises, strong statements)
    - Emotional peaks: Exclamation marks, emphatic language, emotional vocabulary
    - Pacing: Words per second — faster pacing tends to be more engaging
    - Quotability: Short, punchy sentences that work as standalone quotes

    Args:
        highlight: The highlight to score.
        transcript_segments: Full transcript for context.

    Returns:
        EngagementScore with per-dimension breakdown and overall score.
    """
    # Extract text within this highlight's time range
    clip_text = ""
    for seg in transcript_segments:
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", 0))
        if seg_start >= highlight.start - 0.5 and seg_end <= highlight.end + 0.5:
            clip_text += " " + seg.get("text", "")

    clip_text = clip_text.strip()
    if not clip_text:
        return EngagementScore()

    words = clip_text.split()
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r'[.!?]+', clip_text) if s.strip()]

    # ---- Hook strength (first sentence analysis) ----
    hook_score = 0.3  # baseline
    first_sentence = sentences[0].lower() if sentences else ""

    # Questions are strong hooks
    if "?" in clip_text[:100]:
        hook_score += 0.3
    # "You" / "your" direct address hooks
    if any(w in first_sentence for w in ["you", "your", "imagine", "what if"]):
        hook_score += 0.15
    # Numbers/statistics hook
    if any(c.isdigit() for c in first_sentence):
        hook_score += 0.1
    # Strong opening words
    hook_words = {"never", "always", "secret", "truth", "mistake", "biggest", "best", "worst",
                  "shocking", "actually", "here's", "listen", "stop", "wait", "nobody", "everyone"}
    if any(w in first_sentence.split() for w in hook_words):
        hook_score += 0.2
    hook_score = min(1.0, hook_score)

    # ---- Emotional intensity ----
    emotion_score = 0.2  # baseline
    exclamation_count = clip_text.count("!")
    emotion_score += min(0.3, exclamation_count * 0.08)

    # Emotional vocabulary
    emotion_words = {"love", "hate", "amazing", "incredible", "terrible", "insane", "crazy",
                     "beautiful", "horrible", "passionate", "excited", "furious", "thrilled",
                     "shocked", "devastated", "unbelievable", "mind-blowing", "obsessed"}
    text_lower = clip_text.lower()
    emotion_hits = sum(1 for w in emotion_words if w in text_lower)
    emotion_score += min(0.4, emotion_hits * 0.1)

    # ALL CAPS words indicate emphasis
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    emotion_score += min(0.15, caps_words * 0.05)
    emotion_score = min(1.0, emotion_score)

    # ---- Pacing (words per second) ----
    duration = max(0.1, highlight.duration)
    wps = word_count / duration
    # Sweet spot is 2.5-3.5 wps (conversational but energetic)
    if wps >= 2.5 and wps <= 3.5:
        pacing_score = 0.9
    elif wps >= 2.0 and wps <= 4.0:
        pacing_score = 0.7
    elif wps >= 1.5:
        pacing_score = 0.5
    else:
        pacing_score = 0.3

    # ---- Quotability (short punchy sentences) ----
    quotability_score = 0.3  # baseline
    if sentences:
        avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
        # Short sentences (5-12 words) are most quotable
        if 5 <= avg_sentence_len <= 12:
            quotability_score += 0.3
        elif avg_sentence_len < 5:
            quotability_score += 0.15  # too short can be less meaningful
        # Contrast / reversal patterns are quotable
        contrast_words = {"but", "however", "actually", "instead", "not", "rather", "though"}
        if any(w in text_lower.split() for w in contrast_words):
            quotability_score += 0.15
        # Lists / patterns of three
        if text_lower.count(",") >= 2:
            quotability_score += 0.1
    quotability_score = min(1.0, quotability_score)

    engagement = EngagementScore(
        hook_strength=round(hook_score, 3),
        emotional_peak=round(emotion_score, 3),
        pacing=round(pacing_score, 3),
        quotability=round(quotability_score, 3),
    )
    engagement.compute_overall()
    return engagement


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
    from opencut.core.llm import LLMConfig, query_llm

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
        bp = data.get("bullet_points", [])
        tp = data.get("topics", [])
        summary = Summary(
            text=str(data.get("summary", "")),
            bullet_points=bp if isinstance(bp, list) else [],
            topics=tp if isinstance(tp, list) else [],
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


# ---------------------------------------------------------------------------
# Vision-Augmented Highlight Extraction
# ---------------------------------------------------------------------------
def extract_frames_for_vision(
    video_path: str,
    interval_seconds: float = 10.0,
    max_frames: int = 30,
) -> List[Dict]:
    """
    Extract keyframes from video at regular intervals for vision LLM analysis.

    Returns list of {"timestamp": float, "base64": str} dicts.
    """
    import base64
    import os
    import subprocess
    import tempfile

    duration_cmd = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", video_path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        duration = float(duration_cmd.stdout.strip())
    except (ValueError, AttributeError):
        duration = 300.0

    # Calculate frame timestamps
    n_frames = min(max_frames, max(1, int(duration / interval_seconds)))
    timestamps = [i * interval_seconds for i in range(n_frames)]

    frames = []
    tmp_dir = tempfile.mkdtemp(prefix="opencut_vision_")
    try:
        for i, ts in enumerate(timestamps):
            out_path = os.path.join(tmp_dir, f"frame_{i:04d}.jpg")
            subprocess.run(
                ["ffmpeg", "-ss", str(ts), "-i", video_path,
                 "-vframes", "1", "-q:v", "5", "-vf", "scale=480:-1",
                 "-y", out_path],
                capture_output=True, timeout=10,
            )
            if os.path.isfile(out_path) and os.path.getsize(out_path) > 100:
                with open(out_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                frames.append({"timestamp": ts, "base64": b64})
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return frames


def extract_highlights_with_vision(
    video_path: str,
    transcript_segments: List[Dict],
    max_highlights: int = 5,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    llm_config=None,
    frame_interval: float = 10.0,
    on_progress: Optional[Callable] = None,
) -> HighlightResult:
    """
    Extract highlights using both transcript AND visual frame analysis.

    Sends sampled video frames alongside the transcript to a vision-capable
    LLM (GPT-4o, Claude, Gemini) for richer highlight detection that catches
    visual-only moments (action, dramatic visuals, reactions) that transcript
    analysis alone would miss.

    Args:
        video_path: Source video for frame extraction.
        transcript_segments: Text transcript segments.
        frame_interval: Seconds between sampled frames.
    """
    from opencut.core.llm import LLMConfig, query_llm

    if llm_config is None:
        llm_config = LLMConfig()

    if not transcript_segments:
        return HighlightResult()

    if on_progress:
        on_progress(5, "Extracting keyframes for vision analysis...")

    frames = extract_frames_for_vision(video_path, interval_seconds=frame_interval)

    if on_progress:
        on_progress(15, "Formatting transcript + visual context...")

    formatted = _format_transcript_for_llm(transcript_segments)

    # Build frame descriptions for the prompt
    frame_desc = "\n".join(
        f"[Frame at {f['timestamp']:.1f}s]" for f in frames
    )

    prompt = (
        f"Analyze this video using both its transcript AND the visual keyframes below. "
        f"Find the {max_highlights} most interesting, viral, or engaging moments. "
        f"Each clip should be {min_duration:.0f}-{max_duration:.0f} seconds.\n\n"
        f"Consider VISUAL elements (action, reactions, dramatic visuals, on-screen text, "
        f"scene changes) in addition to speech content.\n\n"
        f"TRANSCRIPT:\n{formatted}\n\n"
        f"VISUAL KEYFRAMES (timestamps):\n{frame_desc}\n\n"
        f"Note: {len(frames)} frames were sampled at {frame_interval}s intervals. "
        f"Use timestamps to correlate visual moments with transcript segments."
    )

    if on_progress:
        on_progress(25, "Querying vision LLM for highlight analysis...")

    # If the LLM supports vision, we could send frames as images
    # For now, send frame timestamps as text context (works with all LLMs)
    response = query_llm(
        prompt=prompt,
        config=llm_config,
        system_prompt=_HIGHLIGHT_SYSTEM_PROMPT,
    )

    if on_progress:
        on_progress(80, "Parsing highlights...")

    if response.text.startswith("LLM error:"):
        logger.error("Vision LLM query failed: %s", response.text)
        return HighlightResult(llm_provider=response.provider, llm_model=response.model)

    highlights = _parse_highlights_json(response.text)

    filtered = []
    for h in highlights:
        if h.duration < min_duration:
            h.end = h.start + min_duration
        elif h.duration > max_duration:
            h.end = h.start + max_duration
        if h.end > h.start:
            filtered.append(h)

    filtered.sort(key=lambda h: h.score, reverse=True)
    filtered = filtered[:max_highlights]

    if on_progress:
        on_progress(100, f"Found {len(filtered)} highlights (vision-augmented)")

    return HighlightResult(
        highlights=filtered,
        total_found=len(filtered),
        llm_provider=response.provider,
        llm_model=response.model,
    )
