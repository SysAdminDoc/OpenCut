"""
OpenCut AI Show Notes & Transcript Summary

Generate structured show notes from transcript text:
- Summary (2-3 sentences)
- Key topics with timestamps
- Notable quotes
- Chapter markers
- Resources/URLs mentioned

Uses the configured LLM provider (opencut.core.llm) with a fallback
to basic keyword extraction when no LLM is available.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class ShowNotes:
    """Structured show notes extracted from a transcript."""
    summary: str = ""
    key_topics: List[Dict] = field(default_factory=list)
    quotes: List[str] = field(default_factory=list)
    chapter_markers: List[Dict] = field(default_factory=list)
    resources_mentioned: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM-based extraction
# ---------------------------------------------------------------------------
_SHOW_NOTES_SYSTEM_PROMPT = """You are a professional podcast editor. Extract structured show notes from the transcript provided.

Return your response in the following exact format (use these exact headings):

## Summary
2-3 sentence summary of the content.

## Key Topics
- [HH:MM:SS] Topic description
- [HH:MM:SS] Topic description

## Notable Quotes
- "Exact quote from the transcript"
- "Another notable quote"

## Chapter Markers
- [HH:MM:SS] Chapter title
- [HH:MM:SS] Chapter title

## Resources Mentioned
- Resource or URL mentioned
- Another resource

If timestamps are not available in the transcript, omit the [HH:MM:SS] prefix.
If a section has no content, write "None" under that heading."""


def _parse_llm_response(text: str) -> ShowNotes:
    """Parse the LLM response into a ShowNotes dataclass."""
    notes = ShowNotes()

    sections = re.split(r"##\s+", text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.split("\n")
        heading = lines[0].strip().lower()
        body_lines = [ln.strip() for ln in lines[1:] if ln.strip() and ln.strip().lower() != "none"]

        if "summary" in heading:
            notes.summary = " ".join(body_lines)

        elif "key topic" in heading:
            for line in body_lines:
                line = line.lstrip("- ").strip()
                ts_match = re.match(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.*)", line)
                if ts_match:
                    notes.key_topics.append({
                        "timestamp": ts_match.group(1),
                        "topic": ts_match.group(2),
                    })
                else:
                    notes.key_topics.append({"timestamp": "", "topic": line})

        elif "quote" in heading:
            for line in body_lines:
                line = line.lstrip("- ").strip()
                # Remove surrounding quotes if present
                line = line.strip('"').strip("'")
                if line:
                    notes.quotes.append(line)

        elif "chapter" in heading:
            for line in body_lines:
                line = line.lstrip("- ").strip()
                ts_match = re.match(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.*)", line)
                if ts_match:
                    notes.chapter_markers.append({
                        "timestamp": ts_match.group(1),
                        "title": ts_match.group(2),
                    })
                else:
                    notes.chapter_markers.append({"timestamp": "", "title": line})

        elif "resource" in heading:
            for line in body_lines:
                line = line.lstrip("- ").strip()
                if line:
                    notes.resources_mentioned.append(line)

    return notes


# ---------------------------------------------------------------------------
# Fallback: basic keyword extraction
# ---------------------------------------------------------------------------
def _fallback_show_notes(transcript_text: str) -> ShowNotes:
    """
    Generate basic show notes without an LLM.

    Uses simple heuristics: first/last sentences for summary,
    keyword frequency for topics, URL regex for resources.
    """
    notes = ShowNotes()

    # Split into sentences
    sentences = re.split(r"[.!?]+", transcript_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    # Summary: first and last non-trivial sentences
    if sentences:
        summary_parts = [sentences[0]]
        if len(sentences) > 1:
            summary_parts.append(sentences[-1])
        notes.summary = ". ".join(summary_parts) + "."

    # Key topics: find frequently occurring multi-word phrases
    words = re.findall(r"\b[a-zA-Z]{4,}\b", transcript_text.lower())
    word_freq = {}
    for w in words:
        # Skip common stop words
        if w in _STOP_WORDS:
            continue
        word_freq[w] = word_freq.get(w, 0) + 1

    # Top keywords as topics
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:8]
    for word, count in top_words:
        notes.key_topics.append({
            "timestamp": "",
            "topic": f"{word.capitalize()} (mentioned {count} times)",
        })

    # Resources: find URLs
    urls = re.findall(r"https?://[^\s<>\"']+", transcript_text)
    notes.resources_mentioned = list(set(urls))[:10]

    # Quotes: longest sentences (likely most substantive)
    if sentences:
        by_length = sorted(sentences, key=len, reverse=True)
        for s in by_length[:3]:
            if len(s) > 30:
                notes.quotes.append(s.strip())

    return notes


_STOP_WORDS = frozenset({
    "that", "this", "with", "from", "have", "been", "were", "they",
    "their", "about", "would", "could", "should", "which", "there",
    "when", "what", "will", "just", "like", "know", "than", "them",
    "then", "also", "more", "some", "very", "into", "over", "such",
    "your", "said", "each", "make", "does", "made", "after", "being",
    "here", "through", "where", "most", "these", "think", "because",
    "thing", "things", "going", "really", "actually", "something",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_show_notes(
    transcript_text: str,
    format: str = "markdown",
    on_progress: Optional[Callable] = None,
) -> ShowNotes:
    """
    Generate structured show notes from transcript text.

    Uses the configured LLM provider if available, otherwise falls back
    to basic keyword extraction.

    Args:
        transcript_text: Full transcript text to analyze.
        format: Ignored for generation (used by export_show_notes).
        on_progress: Progress callback(pct, msg).

    Returns:
        ShowNotes dataclass with summary, key_topics, quotes,
        chapter_markers, resources_mentioned.
    """
    if not transcript_text or not transcript_text.strip():
        return ShowNotes(summary="No transcript text provided.")

    if on_progress:
        on_progress(10, "Analyzing transcript...")

    # Try LLM-based extraction
    try:
        from opencut.core.llm import check_llm_reachable, query_llm

        if on_progress:
            on_progress(15, "Checking LLM availability...")

        status = check_llm_reachable()
        if status.get("available"):
            if on_progress:
                on_progress(20, f"Using LLM ({status.get('provider', 'unknown')}) for show notes...")

            # Truncate very long transcripts to fit in context
            max_chars = 15000
            truncated = transcript_text[:max_chars]
            if len(transcript_text) > max_chars:
                truncated += "\n\n[Transcript truncated...]"

            prompt = f"Extract show notes from this transcript:\n\n{truncated}"

            response = query_llm(
                prompt=prompt,
                system_prompt=_SHOW_NOTES_SYSTEM_PROMPT,
                on_progress=None,
            )

            if response.text and "error" not in response.text.lower()[:20]:
                if on_progress:
                    on_progress(80, "Parsing LLM response...")
                notes = _parse_llm_response(response.text)
                if notes.summary:
                    if on_progress:
                        on_progress(100, "Show notes generated via LLM")
                    logger.info("Generated show notes via LLM (%s/%s)",
                                response.provider, response.model)
                    return notes

            logger.warning("LLM response was empty or invalid, falling back to keyword extraction")
        else:
            logger.info("LLM not reachable, using fallback keyword extraction")

    except ImportError:
        logger.info("LLM module not available, using fallback keyword extraction")
    except Exception as exc:
        logger.warning("LLM show notes failed (%s), using fallback", exc)

    # Fallback: basic extraction
    if on_progress:
        on_progress(50, "Using keyword extraction fallback...")

    notes = _fallback_show_notes(transcript_text)

    if on_progress:
        on_progress(100, "Show notes generated (keyword extraction)")

    logger.info("Generated show notes via keyword extraction fallback")
    return notes


def export_show_notes(
    show_notes: ShowNotes,
    format: str = "markdown",
) -> str:
    """
    Export ShowNotes to a formatted string.

    Args:
        show_notes: ShowNotes dataclass to export.
        format: Output format - "markdown", "html", or "text".

    Returns:
        Formatted string.
    """
    format = format.lower().strip()
    if format not in ("markdown", "html", "text"):
        format = "markdown"

    if format == "markdown":
        return _export_markdown(show_notes)
    elif format == "html":
        return _export_html(show_notes)
    else:
        return _export_text(show_notes)


def _export_markdown(notes: ShowNotes) -> str:
    """Export show notes as Markdown."""
    parts = []

    parts.append("# Show Notes\n")

    if notes.summary:
        parts.append("## Summary\n")
        parts.append(notes.summary + "\n")

    if notes.key_topics:
        parts.append("\n## Key Topics\n")
        for topic in notes.key_topics:
            ts = f"[{topic['timestamp']}] " if topic.get("timestamp") else ""
            parts.append(f"- {ts}{topic.get('topic', '')}")

    if notes.chapter_markers:
        parts.append("\n\n## Chapters\n")
        for ch in notes.chapter_markers:
            ts = f"[{ch['timestamp']}] " if ch.get("timestamp") else ""
            parts.append(f"- {ts}{ch.get('title', '')}")

    if notes.quotes:
        parts.append("\n\n## Notable Quotes\n")
        for q in notes.quotes:
            parts.append(f'- "{q}"')

    if notes.resources_mentioned:
        parts.append("\n\n## Resources\n")
        for r in notes.resources_mentioned:
            parts.append(f"- {r}")

    return "\n".join(parts) + "\n"


def _export_html(notes: ShowNotes) -> str:
    """Export show notes as HTML."""
    parts = ["<div class='show-notes'>"]

    parts.append("<h1>Show Notes</h1>")

    if notes.summary:
        parts.append(f"<h2>Summary</h2><p>{_html_escape(notes.summary)}</p>")

    if notes.key_topics:
        parts.append("<h2>Key Topics</h2><ul>")
        for topic in notes.key_topics:
            ts = f"<span class='timestamp'>[{_html_escape(topic.get('timestamp', ''))}]</span> " if topic.get("timestamp") else ""
            parts.append(f"<li>{ts}{_html_escape(topic.get('topic', ''))}</li>")
        parts.append("</ul>")

    if notes.chapter_markers:
        parts.append("<h2>Chapters</h2><ul>")
        for ch in notes.chapter_markers:
            ts = f"<span class='timestamp'>[{_html_escape(ch.get('timestamp', ''))}]</span> " if ch.get("timestamp") else ""
            parts.append(f"<li>{ts}{_html_escape(ch.get('title', ''))}</li>")
        parts.append("</ul>")

    if notes.quotes:
        parts.append("<h2>Notable Quotes</h2><blockquote>")
        for q in notes.quotes:
            parts.append(f"<p>&ldquo;{_html_escape(q)}&rdquo;</p>")
        parts.append("</blockquote>")

    if notes.resources_mentioned:
        parts.append("<h2>Resources</h2><ul>")
        for r in notes.resources_mentioned:
            escaped = _html_escape(r)
            if r.startswith("http"):
                parts.append(f"<li><a href='{escaped}'>{escaped}</a></li>")
            else:
                parts.append(f"<li>{escaped}</li>")
        parts.append("</ul>")

    parts.append("</div>")
    return "\n".join(parts)


def _export_text(notes: ShowNotes) -> str:
    """Export show notes as plain text."""
    parts = []

    parts.append("SHOW NOTES")
    parts.append("=" * 40)

    if notes.summary:
        parts.append("\nSUMMARY")
        parts.append(notes.summary)

    if notes.key_topics:
        parts.append("\nKEY TOPICS")
        for topic in notes.key_topics:
            ts = f"[{topic['timestamp']}] " if topic.get("timestamp") else ""
            parts.append(f"  * {ts}{topic.get('topic', '')}")

    if notes.chapter_markers:
        parts.append("\nCHAPTERS")
        for ch in notes.chapter_markers:
            ts = f"[{ch['timestamp']}] " if ch.get("timestamp") else ""
            parts.append(f"  * {ts}{ch.get('title', '')}")

    if notes.quotes:
        parts.append("\nNOTABLE QUOTES")
        for q in notes.quotes:
            parts.append(f'  "{q}"')

    if notes.resources_mentioned:
        parts.append("\nRESOURCES")
        for r in notes.resources_mentioned:
            parts.append(f"  * {r}")

    return "\n".join(parts) + "\n"


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
