"""
OpenCut Isochronous Translation

Translates dialogue segments to fit within the same duration as the original.

Pipeline:
1. Measure original segment duration
2. Translate text to target language
3. Estimate TTS duration of translated text
4. If too long/short (>10% deviation), use LLM to rephrase
5. Iterate until within +-10% of original duration

Ensures lip-sync-friendly dubbing by constraining translated text length.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.security import safe_float

logger = logging.getLogger("opencut")

# Average speaking rates by language (words per minute)
SPEAKING_RATES_WPM = {
    "en": 150, "es": 160, "fr": 170, "de": 130, "it": 160,
    "pt": 155, "ru": 130, "zh": 160, "ja": 200, "ko": 160,
    "ar": 140, "hi": 150, "tr": 145, "vi": 155, "th": 150,
    "pl": 140, "nl": 145, "sv": 140, "da": 140, "no": 140,
    "fi": 130, "cs": 135, "hu": 130, "ro": 150, "el": 145,
}

# Average characters per word by language
CHARS_PER_WORD = {
    "en": 5.0, "es": 5.5, "fr": 5.2, "de": 6.5, "it": 5.5,
    "pt": 5.5, "ru": 6.0, "zh": 1.5, "ja": 1.5, "ko": 2.0,
    "ar": 5.0, "hi": 4.5, "tr": 6.0, "vi": 4.0, "th": 5.0,
}

# Tolerance for duration matching
DEFAULT_TOLERANCE = 0.10  # +-10%
MAX_ITERATIONS = 5


@dataclass
class IsochronousSegment:
    """A single isochronously translated segment."""
    index: int = 0
    original_text: str = ""
    translated_text: str = ""
    original_duration: float = 0.0
    estimated_tts_duration: float = 0.0
    duration_ratio: float = 1.0
    iterations_needed: int = 0
    within_tolerance: bool = False


@dataclass
class IsochronousResult:
    """Result of isochronous translation."""
    source_language: str = ""
    target_language: str = ""
    total_segments: int = 0
    segments_within_tolerance: int = 0
    tolerance: float = 0.10
    segments: List[Dict] = field(default_factory=list)


def _estimate_tts_duration(text: str, language: str) -> float:
    """
    Estimate the TTS speaking duration for a given text in seconds.

    Uses language-specific speaking rates and character/word counts.
    """
    if not text.strip():
        return 0.0

    # Get speaking rate for language
    wpm = SPEAKING_RATES_WPM.get(language, 150)
    cpw = CHARS_PER_WORD.get(language, 5.0)

    # Estimate word count
    if language in ("zh", "ja"):
        # CJK: character-based estimation
        char_count = len(text.replace(" ", ""))
        word_count = char_count / cpw
    elif language == "ko":
        # Korean: syllable blocks
        word_count = len(text.split())
    else:
        word_count = len(text.split())

    # Duration = words / words_per_minute * 60
    duration = (word_count / wpm) * 60.0

    # Add small pause overhead per sentence
    sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
    duration += sentence_count * 0.3

    return max(0.5, duration)


def _translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """Translate text using LLM backend."""
    try:
        from opencut.core.ai_dubbing import SUPPORTED_LANGUAGES
        from opencut.core.llm import llm_chat

        src_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
        tgt_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)

        prompt = (
            f"Translate the following from {src_name} to {tgt_name}. "
            f"Return only the translated text.\n\n{text}"
        )
        result = llm_chat(prompt)
        return result.strip() if isinstance(result, str) else text
    except Exception as exc:
        logger.debug("LLM translation failed: %s", exc)
        return f"[{target_lang}] {text}"


def _rephrase_for_duration(
    text: str,
    target_duration: float,
    current_duration: float,
    language: str,
) -> str:
    """
    Use LLM to rephrase text to better fit target duration.

    If current is too long, asks for a more concise version.
    If too short, asks for a slightly expanded version.
    """
    try:
        from opencut.core.llm import llm_chat

        ratio = current_duration / target_duration if target_duration > 0 else 1.0

        if ratio > 1.0:
            instruction = (
                f"Rephrase the following text to be approximately {int((ratio - 1) * 100)}% "
                f"shorter while preserving the exact meaning. Use fewer words and simpler "
                f"phrasing. Return only the rephrased text in the same language.\n\n{text}"
            )
        else:
            instruction = (
                f"Rephrase the following text to be approximately {int((1 - ratio) * 100)}% "
                f"longer while preserving the exact meaning. Add natural filler words or "
                f"slightly expand descriptions. Return only the rephrased text in the same "
                f"language.\n\n{text}"
            )

        result = llm_chat(instruction)
        return result.strip() if isinstance(result, str) else text
    except Exception as exc:
        logger.debug("LLM rephrase failed: %s", exc)
        return text


def translate_isochronous(
    segments: List[Dict],
    source_language: str,
    target_language: str,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = MAX_ITERATIONS,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Perform isochronous translation on dialogue segments.

    Each segment's translated text is iteratively adjusted to fit
    within +-tolerance of the original speaking duration.

    Args:
        segments: List of dicts with 'text', 'start', 'end' fields.
        source_language: Source language code (e.g. "en").
        target_language: Target language code (e.g. "es").
        tolerance: Acceptable duration deviation (0.10 = 10%).
        max_iterations: Maximum rephrase iterations per segment.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with translated segments and fit statistics.
    """
    if not segments:
        raise ValueError("No segments provided")

    if tolerance <= 0 or tolerance > 0.5:
        tolerance = DEFAULT_TOLERANCE

    if on_progress:
        on_progress(5, f"Translating {len(segments)} segments...")

    results = []
    within_count = 0
    total = len(segments)

    for idx, seg in enumerate(segments):
        original_text = seg.get("text", "").strip()
        start = safe_float(seg.get("start", 0), 0.0)
        end = safe_float(seg.get("end", 0), 0.0)
        original_duration = max(0.5, end - start)

        if not original_text or original_text.startswith("["):
            results.append({
                "index": idx + 1,
                "original_text": original_text,
                "translated_text": original_text,
                "original_duration": round(original_duration, 3),
                "estimated_tts_duration": round(original_duration, 3),
                "duration_ratio": 1.0,
                "iterations_needed": 0,
                "within_tolerance": True,
            })
            within_count += 1
            continue

        # Initial translation
        translated = _translate_text(original_text, source_language, target_language)
        est_duration = _estimate_tts_duration(translated, target_language)

        iterations = 0
        for iteration in range(max_iterations):
            ratio = est_duration / original_duration if original_duration > 0 else 1.0
            deviation = abs(ratio - 1.0)

            if deviation <= tolerance:
                break  # Within tolerance

            iterations = iteration + 1
            translated = _rephrase_for_duration(
                translated, original_duration, est_duration, target_language
            )
            est_duration = _estimate_tts_duration(translated, target_language)

        final_ratio = est_duration / original_duration if original_duration > 0 else 1.0
        is_within = abs(final_ratio - 1.0) <= tolerance

        if is_within:
            within_count += 1

        results.append({
            "index": idx + 1,
            "original_text": original_text,
            "translated_text": translated,
            "original_duration": round(original_duration, 3),
            "estimated_tts_duration": round(est_duration, 3),
            "duration_ratio": round(final_ratio, 3),
            "iterations_needed": iterations,
            "within_tolerance": is_within,
        })

        if on_progress:
            pct = 5 + int(90 * (idx + 1) / total)
            status = "OK" if is_within else "OVER"
            on_progress(pct, f"Segment {idx + 1}/{total}: {status}")

    if on_progress:
        on_progress(100, f"Isochronous translation complete: "
                         f"{within_count}/{total} within tolerance")

    return {
        "source_language": source_language,
        "target_language": target_language,
        "total_segments": total,
        "segments_within_tolerance": within_count,
        "tolerance": tolerance,
        "tolerance_percent": f"+-{int(tolerance * 100)}%",
        "segments": results,
    }


def estimate_segment_duration(text: str, language: str) -> dict:
    """
    Estimate TTS duration for a single text segment.

    Utility endpoint for testing duration estimation.
    """
    duration = _estimate_tts_duration(text, language)
    word_count = len(text.split())
    wpm = SPEAKING_RATES_WPM.get(language, 150)

    return {
        "text": text,
        "language": language,
        "estimated_duration_seconds": round(duration, 3),
        "word_count": word_count,
        "speaking_rate_wpm": wpm,
    }
