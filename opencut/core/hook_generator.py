"""
OpenCut AI Hook Generator

Generate and insert compelling opening hooks in the first 3 seconds.
Supports multiple hook types: teaser, question, statistic, quote, and auto.

Uses LLM when available (opencut.core.llm), else heuristic extraction.
Inspired by Opus Clip's hook generation feature.
"""

import logging
import math
import os
import re
import subprocess as _sp
import tempfile
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

from opencut.helpers import (
    get_ffmpeg_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

HOOK_TYPES = ("auto", "question", "statistic", "teaser", "quote")
_HOOK_DURATION = 3.0  # seconds


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class HookResult:
    """Result of hook generation."""
    hook_text: str = ""
    hook_type: str = "auto"
    insertion_method: str = "caption_overlay"
    preview_text: str = ""
    teaser_start: float = 0.0
    teaser_end: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Heuristic text extraction helpers
# ---------------------------------------------------------------------------
def _extract_sentences(transcript: str) -> List[str]:
    """Split transcript into sentences."""
    cleaned = re.sub(
        r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}",
        "", transcript,
    )
    cleaned = re.sub(r"^\d+\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = " ".join(cleaned.split())

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _find_question(sentences: List[str]) -> Optional[str]:
    """Find the most compelling question in the transcript."""
    questions = [s for s in sentences if "?" in s]
    if not questions:
        return None
    # Prefer shorter, punchier questions
    scored = sorted(questions, key=lambda q: (abs(len(q) - 50), -len(q)))
    return scored[0]


def _find_statistic(sentences: List[str]) -> Optional[str]:
    """Find a sentence containing a compelling statistic."""
    stat_patterns = [
        r"\d+\s*%",
        r"\$[\d,.]+",
        r"\d+\s*(million|billion|thousand|x|times)",
        r"(doubled|tripled|quadrupled|half|twice)",
        r"\d+\s*(out of|in every)",
    ]
    for sent in sentences:
        for pat in stat_patterns:
            if re.search(pat, sent, re.IGNORECASE):
                return sent
    return None


def _find_quote(sentences: List[str]) -> Optional[str]:
    """Find a powerful/emotional quote from the transcript."""
    # Look for strong emotional or impactful language
    power_words = {
        "never", "always", "secret", "truth", "amazing", "incredible",
        "important", "critical", "shocking", "unbelievable", "powerful",
        "dangerous", "revolutionary", "transformed", "changed", "discovered",
        "revealed", "proven", "essential", "ultimate",
    }
    scored = []
    for sent in sentences:
        lower = sent.lower()
        hits = sum(1 for w in power_words if w in lower)
        if hits > 0:
            scored.append((hits, sent))
    scored.sort(reverse=True)
    return scored[0][1] if scored else None


def _find_highest_energy_segment(
    video_path: str, duration: float, segment_len: float = 2.0
) -> tuple:
    """Find the 2-second segment with the highest audio energy."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-nostats",
        "-i", video_path,
        "-af", "astats=metadata=1:reset=44100,"
               "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level",
        "-f", "null", "-",
    ]
    try:
        result = _sp.run(cmd, capture_output=True, text=True, timeout=120)
    except _sp.TimeoutExpired:
        return (duration * 0.3, duration * 0.3 + segment_len)

    rms_values = re.findall(
        r"lavfi\.astats\.Overall\.RMS_level=([-\d.]+)", result.stderr
    )
    if not rms_values:
        return (duration * 0.3, duration * 0.3 + segment_len)

    levels = []
    for v in rms_values:
        try:
            val = float(v)
            levels.append(val if not math.isinf(val) else -60.0)
        except ValueError:
            levels.append(-60.0)

    if len(levels) < 3:
        return (duration * 0.3, duration * 0.3 + segment_len)

    # Skip first 3s (hook area) and find the loudest window after that
    seg_samples = max(1, int(segment_len))
    skip_samples = max(0, int(_HOOK_DURATION))
    best_start = skip_samples
    best_energy = -999.0

    for i in range(skip_samples, len(levels) - seg_samples + 1):
        window = levels[i: i + seg_samples]
        avg = sum(window) / len(window)
        if avg > best_energy:
            best_energy = avg
            best_start = i

    start_time = float(best_start)
    end_time = min(start_time + segment_len, duration)
    return (start_time, end_time)


# ---------------------------------------------------------------------------
# LLM-assisted hook generation
# ---------------------------------------------------------------------------
def _try_llm_hook(
    transcript: str, hook_type: str, on_progress: Optional[Callable] = None
) -> Optional[str]:
    """Attempt to generate a hook using an LLM. Returns None if unavailable."""
    try:
        from opencut.core.llm import query_llm
    except ImportError:
        return None

    type_instructions = {
        "question": "Generate a compelling question that hooks viewers.",
        "statistic": "Extract or craft a surprising statistic from the content.",
        "quote": "Extract the most powerful quote or statement.",
        "auto": "Generate the most compelling hook of any type.",
    }
    instruction = type_instructions.get(hook_type, type_instructions["auto"])

    prompt = (
        f"Given this video transcript, {instruction}\n"
        f"Keep it under 15 words. Make it punchy and attention-grabbing.\n\n"
        f"Transcript:\n{transcript[:2000]}\n\n"
        f"Hook text (just the hook, nothing else):"
    )

    try:
        response = query_llm(prompt, on_progress=on_progress)
        if hasattr(response, "text"):
            text = response.text.strip().strip('"').strip("'")
        else:
            text = str(response).strip().strip('"').strip("'")
        return text if len(text) > 5 else None
    except Exception as exc:
        logger.debug("LLM hook generation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def generate_hook(
    video_path: str,
    transcript: Optional[str] = None,
    hook_type: str = "auto",
    on_progress: Optional[Callable] = None,
) -> HookResult:
    """
    Generate a compelling hook for a video.

    Args:
        video_path: Path to the video file.
        transcript: Optional transcript text.
        hook_type: One of ``"auto"``, ``"question"``, ``"statistic"``,
                   ``"teaser"``, ``"quote"``.
        on_progress: Progress callback ``(pct, msg)``.

    Returns:
        :class:`HookResult` with hook text and insertion metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if hook_type not in HOOK_TYPES:
        raise ValueError(
            f"Invalid hook_type: {hook_type}. Must be one of {HOOK_TYPES}"
        )

    if on_progress:
        on_progress(5, "Analyzing video for hook generation...")

    info = get_video_info(video_path)
    duration = info.get("duration", 0)

    result = HookResult(hook_type=hook_type)

    # Teaser type: find best moment and prepend clip
    if hook_type == "teaser" or (hook_type == "auto" and not transcript):
        if on_progress:
            on_progress(20, "Finding highest-energy segment...")
        start, end = _find_highest_energy_segment(video_path, duration)
        result.hook_text = "Preview of what's coming..."
        result.insertion_method = "prepend_clip"
        result.teaser_start = round(start, 2)
        result.teaser_end = round(end, 2)
        result.preview_text = f"[Teaser clip from {start:.1f}s to {end:.1f}s]"
        if on_progress:
            on_progress(100, "Teaser hook generated")
        return result

    # Text-based hooks require a transcript
    sentences: List[str] = []
    if transcript:
        sentences = _extract_sentences(transcript)

    # Try LLM first
    if transcript and hook_type != "teaser":
        if on_progress:
            on_progress(20, "Attempting LLM-based hook generation...")
        llm_text = _try_llm_hook(transcript, hook_type, on_progress)
        if llm_text:
            result.hook_text = llm_text
            result.insertion_method = "caption_overlay"
            result.preview_text = llm_text
            if on_progress:
                on_progress(100, "Hook generated via LLM")
            return result

    # Heuristic fallback
    if on_progress:
        on_progress(40, "Generating hook from transcript heuristics...")

    hook_text = None

    if hook_type == "question" or (hook_type == "auto" and not hook_text):
        hook_text = _find_question(sentences)
        if hook_text:
            result.hook_type = "question"

    if hook_type == "statistic" or (hook_type == "auto" and not hook_text):
        found = _find_statistic(sentences)
        if found:
            hook_text = found
            result.hook_type = "statistic"

    if hook_type == "quote" or (hook_type == "auto" and not hook_text):
        found = _find_quote(sentences)
        if found:
            hook_text = found
            result.hook_type = "quote"

    # Final fallback: use first sentence
    if not hook_text and sentences:
        hook_text = sentences[0]
        result.hook_type = "quote"

    if not hook_text:
        # No transcript and no teaser: fall back to teaser
        if on_progress:
            on_progress(60, "No transcript -- falling back to teaser hook...")
        start, end = _find_highest_energy_segment(video_path, duration)
        result.hook_text = "Preview of what's coming..."
        result.hook_type = "teaser"
        result.insertion_method = "prepend_clip"
        result.teaser_start = round(start, 2)
        result.teaser_end = round(end, 2)
        result.preview_text = f"[Teaser clip from {start:.1f}s to {end:.1f}s]"
    else:
        # Truncate to ~15 words max for hook
        words = hook_text.split()
        if len(words) > 15:
            hook_text = " ".join(words[:15]) + "..."
        result.hook_text = hook_text
        result.insertion_method = "caption_overlay"
        result.preview_text = hook_text

    if on_progress:
        on_progress(100, "Hook generated")

    return result


def apply_hook(
    video_path: str,
    hook_result: HookResult,
    output: Optional[str] = None,
    tts_voice: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply a generated hook to a video.

    Args:
        video_path: Input video path.
        hook_result: :class:`HookResult` from :func:`generate_hook`.
        output: Optional output path.
        tts_voice: Reserved for future TTS integration.
        on_progress: Progress callback ``(pct, msg)``.

    Returns:
        Path to the output video with hook applied.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    out = output or output_path(video_path, "hooked")

    if on_progress:
        on_progress(10, "Applying hook...")

    info = get_video_info(video_path)
    w = info.get("width", 1920)
    h = info.get("height", 1080)

    if hook_result.insertion_method == "prepend_clip":
        # Extract the teaser segment and prepend it
        if on_progress:
            on_progress(20, "Extracting teaser clip...")

        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as tmp:
            teaser_path = tmp.name

        try:
            # Extract teaser segment
            start = hook_result.teaser_start
            dur = hook_result.teaser_end - hook_result.teaser_start
            extract_cmd = [
                get_ffmpeg_path(), "-y",
                "-ss", str(start), "-t", str(dur),
                "-i", video_path,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                teaser_path,
            ]
            run_ffmpeg(extract_cmd)

            if on_progress:
                on_progress(50, "Concatenating teaser with main video...")

            # Create concat list
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as lst:
                lst_path = lst.name
                lst.write(f"file '{teaser_path}'\n")
                lst.write(f"file '{os.path.abspath(video_path)}'\n")

            try:
                concat_cmd = [
                    get_ffmpeg_path(), "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", lst_path,
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-c:a", "aac", "-b:a", "192k",
                    out,
                ]
                run_ffmpeg(concat_cmd)
            finally:
                try:
                    os.unlink(lst_path)
                except OSError:
                    pass
        finally:
            try:
                os.unlink(teaser_path)
            except OSError:
                pass

    else:
        # Caption overlay on first 3 seconds
        if on_progress:
            on_progress(30, "Rendering caption overlay hook...")

        escaped_text = (
            hook_result.hook_text
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace(":", "\\:")
            .replace("%", "%%")
        )
        y_pos = int(h * 0.45)
        font_size = max(24, min(60, int(w / 25)))

        drawtext = (
            f"drawtext=text='{escaped_text}'"
            f":fontsize={font_size}"
            f":fontcolor=#FFFFFF"
            f":shadowcolor=#000000"
            f":shadowx=3:shadowy=3"
            f":x=(w-text_w)/2"
            f":y={y_pos}"
            f":enable='between(t,0,{_HOOK_DURATION})'"
        )

        cmd = [
            get_ffmpeg_path(), "-y",
            "-i", video_path,
            "-vf", drawtext,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            out,
        ]
        run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Hook applied")

    return out
