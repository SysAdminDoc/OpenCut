"""
OpenCut AI Video Summarization

Generates text summaries from transcripts (LLM-powered with keyword fallback)
and visual summaries by selecting and concatenating top scene moments.

Uses FFmpeg for clip extraction and concatenation.
"""

import logging
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# Common stop words to exclude from keyword extraction
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "am", "in", "on",
    "at", "to", "for", "of", "with", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because", "but",
    "and", "or", "if", "while", "about", "up", "its", "it", "this", "that",
    "these", "those", "he", "she", "they", "we", "you", "i", "me", "my",
    "your", "his", "her", "our", "their", "what", "which", "who", "whom",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TextSummaryResult:
    """Text summary output."""
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    key_sentences: List[str] = field(default_factory=list)
    word_count: int = 0
    method: str = "keyword"  # "llm" or "keyword"


@dataclass
class VisualSummaryResult:
    """Visual summary output."""
    output_path: str = ""
    selected_scenes: List[Dict] = field(default_factory=list)
    total_duration: float = 0.0
    summary_duration: float = 0.0
    scene_count: int = 0


# ---------------------------------------------------------------------------
# Text analysis helpers
# ---------------------------------------------------------------------------
def _extract_keywords(text: str, top_n: int = 15) -> List[str]:
    """Extract top keywords from text using frequency analysis."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in _STOP_WORDS]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Handle common sentence-ending patterns
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def _score_sentence(sentence: str, keywords: List[str]) -> float:
    """Score a sentence by keyword relevance, position weighting, and length."""
    words = set(re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower()))
    keyword_overlap = len(words.intersection(set(keywords)))
    # Prefer medium-length sentences (not too short, not too long)
    word_count = len(sentence.split())
    length_score = 1.0 - abs(word_count - 15) / 30.0
    length_score = max(0.1, min(1.0, length_score))
    return keyword_overlap * length_score


def _build_keyword_summary(text: str, max_sentences: int = 5) -> TextSummaryResult:
    """Build a summary using keyword-based extractive summarization."""
    keywords = _extract_keywords(text, top_n=15)
    sentences = _split_sentences(text)

    if not sentences:
        return TextSummaryResult(
            summary=text[:500] if text else "No transcript content available.",
            keywords=keywords,
            key_sentences=[],
            word_count=len(text.split()),
            method="keyword",
        )

    # Score and rank sentences
    scored = [(s, _score_sentence(s, keywords)) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Select top sentences, but maintain original order
    top_sentences = scored[:max_sentences]
    # Re-order by original position
    ordered = sorted(top_sentences, key=lambda x: sentences.index(x[0]))
    key_sentences = [s[0] for s in ordered]

    summary = " ".join(key_sentences)

    return TextSummaryResult(
        summary=summary,
        keywords=keywords,
        key_sentences=key_sentences,
        word_count=len(text.split()),
        method="keyword",
    )


# ---------------------------------------------------------------------------
# Public API: text_summary
# ---------------------------------------------------------------------------
def text_summary(
    transcript: str,
    max_sentences: int = 5,
    llm_config: Optional[dict] = None,
    on_progress: Optional[Callable] = None,
) -> TextSummaryResult:
    """
    Generate a text summary from a transcript.

    Uses LLM when available and configured, otherwise falls back to
    keyword-based extractive summarization.

    Args:
        transcript: Full transcript text.
        max_sentences: Maximum sentences in summary.
        llm_config: Optional dict with LLM provider settings.
        on_progress: Progress callback(pct, msg).

    Returns:
        TextSummaryResult with summary text and metadata.
    """
    if not transcript or not transcript.strip():
        return TextSummaryResult(
            summary="No transcript provided for summarization.",
            method="keyword",
        )

    transcript = transcript.strip()

    if on_progress:
        on_progress(10, "Analyzing transcript...")

    # Try LLM-based summary first
    if llm_config:
        try:
            from opencut.core.llm import LLMConfig, query_llm

            if on_progress:
                on_progress(20, "Generating LLM summary...")

            config = LLMConfig()
            if "provider" in llm_config:
                config.provider = llm_config["provider"]
            if "model" in llm_config:
                config.model = llm_config["model"]
            if "api_key" in llm_config:
                config.api_key = llm_config["api_key"]
            if "base_url" in llm_config:
                config.base_url = llm_config["base_url"]

            # Truncate very long transcripts for LLM context window
            truncated = transcript[:8000] if len(transcript) > 8000 else transcript
            prompt = (
                f"Summarize this video transcript in {max_sentences} concise sentences. "
                f"Focus on the main topics, key points, and conclusions.\n\n"
                f"Transcript:\n{truncated}"
            )
            system_prompt = (
                "You are a video content summarizer. Generate clear, concise summaries "
                "that capture the essential content of video transcripts."
            )

            response = query_llm(prompt, config=config, system_prompt=system_prompt)
            if response.text and not response.text.startswith("LLM error:"):
                keywords = _extract_keywords(transcript, top_n=15)
                if on_progress:
                    on_progress(100, "LLM summary complete")
                return TextSummaryResult(
                    summary=response.text.strip(),
                    keywords=keywords,
                    key_sentences=_split_sentences(response.text.strip())[:max_sentences],
                    word_count=len(transcript.split()),
                    method="llm",
                )
        except (ImportError, Exception) as exc:
            logger.debug("LLM summary failed, falling back to keywords: %s", exc)

    # Keyword fallback
    if on_progress:
        on_progress(50, "Running keyword-based summarization...")

    result = _build_keyword_summary(transcript, max_sentences=max_sentences)

    if on_progress:
        on_progress(100, "Summary complete")

    return result


# ---------------------------------------------------------------------------
# Visual summary helpers
# ---------------------------------------------------------------------------
def _extract_clip(video_path: str, start: float, end: float, clip_path: str) -> bool:
    """Extract a clip segment from the video."""
    duration = end - start
    if duration <= 0:
        return False
    cmd = (
        FFmpegCmd()
        .input(video_path, ss=str(start), t=str(duration))
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(clip_path)
        .build()
    )
    try:
        run_ffmpeg(cmd, timeout=120)
        return os.path.isfile(clip_path) and os.path.getsize(clip_path) > 0
    except RuntimeError as exc:
        logger.debug("Clip extraction failed (%.2f-%.2f): %s", start, end, exc)
        return False


def _concatenate_clips(clip_paths: List[str], output_file: str) -> bool:
    """Concatenate clip files into a single video using FFmpeg concat demuxer."""
    if not clip_paths:
        return False

    tmp_list = None
    try:
        # Write concat file list
        fd, tmp_list = tempfile.mkstemp(suffix=".txt", prefix="opencut_concat_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for path in clip_paths:
                # Escape single quotes for ffmpeg concat format
                escaped = path.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        cmd = (
            FFmpegCmd()
            .pre_input("f", "concat")
            .pre_input("safe", "0")
            .input(tmp_list)
            .copy_streams()
            .faststart()
            .output(output_file)
            .build()
        )
        run_ffmpeg(cmd, timeout=300)
        return os.path.isfile(output_file) and os.path.getsize(output_file) > 0
    except RuntimeError as exc:
        logger.debug("Clip concatenation failed: %s", exc)
        return False
    finally:
        if tmp_list and os.path.isfile(tmp_list):
            try:
                os.unlink(tmp_list)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Public API: visual_summary
# ---------------------------------------------------------------------------
def visual_summary(
    video_path: str,
    scenes: List[Dict],
    top_n: int = 5,
    clip_duration: float = 3.0,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> VisualSummaryResult:
    """
    Create a visual summary by selecting top-scoring scenes and concatenating clips.

    Each scene dict should have: {"timestamp": float, "score": float}
    Optionally: {"end": float} for explicit end time.

    Args:
        video_path: Path to source video.
        scenes: List of scene dicts with timestamp and score.
        top_n: Number of top scenes to include.
        clip_duration: Default clip duration per scene (seconds).
        output_dir: Output directory. Uses source dir if empty.
        on_progress: Progress callback(pct, msg).

    Returns:
        VisualSummaryResult with output path and metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not scenes:
        raise ValueError("No scenes provided for visual summary")

    video_info = get_video_info(video_path)
    total_duration = video_info.get("duration", 0)

    if on_progress:
        on_progress(5, "Selecting top scenes...")

    # Sort scenes by score descending and select top_n
    scored_scenes = sorted(scenes, key=lambda s: s.get("score", 0), reverse=True)
    selected = scored_scenes[:top_n]

    # Re-sort selected scenes by timestamp for chronological order
    selected.sort(key=lambda s: s.get("timestamp", 0))

    if on_progress:
        on_progress(15, f"Extracting {len(selected)} clips...")

    tmp_dir = tempfile.mkdtemp(prefix="opencut_vizsummary_")
    clip_paths: List[str] = []
    selected_info: List[Dict] = []

    try:
        for i, scene in enumerate(selected):
            ts = scene.get("timestamp", 0)
            end = scene.get("end", ts + clip_duration)
            # Clamp to video duration
            if total_duration > 0:
                ts = min(ts, max(0, total_duration - 0.5))
                end = min(end, total_duration)

            clip_path = os.path.join(tmp_dir, f"clip_{i:04d}.mp4")
            if on_progress:
                pct = 15 + int(60 * (i + 1) / len(selected))
                on_progress(pct, f"Extracting clip {i + 1}/{len(selected)} at {ts:.1f}s...")

            if _extract_clip(video_path, ts, end, clip_path):
                clip_paths.append(clip_path)
                selected_info.append({
                    "timestamp": round(ts, 3),
                    "end": round(end, 3),
                    "duration": round(end - ts, 3),
                    "score": round(scene.get("score", 0), 3),
                })

        if not clip_paths:
            raise RuntimeError("Failed to extract any clips for visual summary")

        if on_progress:
            on_progress(80, "Concatenating clips...")

        # Build output path
        out_dir = output_dir or os.path.dirname(os.path.abspath(video_path))
        out_file = output_path(video_path, "summary", out_dir)

        success = _concatenate_clips(clip_paths, out_file)
        if not success:
            raise RuntimeError("Failed to concatenate summary clips")

        # Calculate actual summary duration
        summary_info = get_video_info(out_file)
        summary_dur = summary_info.get("duration", 0)

        if on_progress:
            on_progress(100, f"Visual summary created ({len(selected_info)} scenes)")

        return VisualSummaryResult(
            output_path=out_file,
            selected_scenes=selected_info,
            total_duration=total_duration,
            summary_duration=summary_dur,
            scene_count=len(selected_info),
        )

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
