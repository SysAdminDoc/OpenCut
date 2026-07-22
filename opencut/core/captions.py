"""
Caption/subtitle generation using OpenAI Whisper.

Supports multiple Whisper backends: openai-whisper, faster-whisper, whisperx.
Falls back gracefully if Whisper is not installed.
"""

import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..utils.config import CaptionConfig
from . import transcript_cache
from .asr_provenance import (
    ASRProvenance,
    apply_language_decision,
    build_provenance,
    model_identity,
    normalize_engine,
    provenance_from_dict,
    provenance_to_dict,
    whisperx_alignment_identity,
)
from .audio import extract_audio_wav

logger = logging.getLogger(__name__)


REVIEW_ASR_CONFIDENCE_THRESHOLD = 0.70
REVIEW_BOUNDARY_CONFIDENCE_THRESHOLD = 0.65
REVIEW_LANGUAGE_CONFIDENCE_THRESHOLD = 0.80
HUMAN_REVIEW_LANGUAGE_CODES = frozenset({
    "ar",
    "arb",
    "arz",
    "ary",
    "arq",
    "ars",
    "apc",
    "acm",
    "hi",
    "hin",
})
_LANGUAGE_ALIASES = {
    "arabic": "ar",
    "hindi": "hi",
    "modern standard arabic": "ar",
}


def _clamp_confidence(value: Any, default: float = 1.0) -> float:
    """Coerce a backend confidence value into the 0..1 range."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    if math.isnan(score) or math.isinf(score):
        score = default
    return max(0.0, min(1.0, score))


def _clamp_optional_confidence(value: Any) -> Optional[float]:
    """Return a bounded confidence value while preserving unknown/None."""
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score) or math.isinf(score):
        return None
    return max(0.0, min(1.0, score))


@dataclass
class Word:
    """A single word with timestamp."""
    text: str
    start: float
    end: float
    confidence: float = 1.0
    boundary_confidence: Optional[float] = None

    def __post_init__(self) -> None:
        self.confidence = _clamp_confidence(self.confidence)
        self.boundary_confidence = _clamp_optional_confidence(self.boundary_confidence)


def normalize_language_code(language: Optional[str]) -> str:
    """Return a stable lower-case language code/name for review rules."""
    if not language:
        return ""
    value = str(language).strip().lower().replace("_", "-")
    if not value:
        return ""
    value = _LANGUAGE_ALIASES.get(value, value)
    if "-" in value:
        value = value.split("-", 1)[0]
    return _LANGUAGE_ALIASES.get(value, value)


def is_human_review_language(language: Optional[str]) -> bool:
    """Return True for languages that should receive manual ASR review."""
    code = normalize_language_code(language)
    return code in HUMAN_REVIEW_LANGUAGE_CODES


def segment_confidence_from_words(words: List[Word], fallback: float = 1.0) -> float:
    """Average word confidence values for a segment."""
    scores = [
        _clamp_confidence(getattr(w, "confidence", fallback), fallback)
        for w in (words or [])
    ]
    if not scores:
        return _clamp_confidence(fallback)
    return round(sum(scores) / len(scores), 4)


def _confidence_from_backend_metadata(
    words: List[Word],
    *,
    avg_logprob: Optional[float] = None,
    no_speech_prob: Optional[float] = None,
    fallback: float = 1.0,
) -> float:
    """Derive a segment confidence from word or backend-level signals."""
    if words:
        return segment_confidence_from_words(words, fallback=fallback)

    candidates: List[float] = []
    try:
        if avg_logprob is not None:
            candidates.append(_clamp_confidence(math.exp(float(avg_logprob)), fallback))
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        if no_speech_prob is not None:
            candidates.append(_clamp_confidence(1.0 - float(no_speech_prob), fallback))
    except (TypeError, ValueError):
        pass
    if candidates:
        return round(min(candidates), 4)
    return _clamp_confidence(fallback)


def caption_review_reasons(
    *,
    language: Optional[str],
    language_confidence: float = 1.0,
    confidence: float = 1.0,
    boundary_confidence: Optional[float] = None,
) -> List[str]:
    """Return stable machine-readable reasons a segment needs review."""
    reasons: List[str] = []
    if is_human_review_language(language):
        reasons.append("language_requires_human_review")
    if _clamp_confidence(language_confidence) < REVIEW_LANGUAGE_CONFIDENCE_THRESHOLD:
        reasons.append("low_language_confidence")
    if _clamp_confidence(confidence) < REVIEW_ASR_CONFIDENCE_THRESHOLD:
        reasons.append("low_asr_confidence")
    boundary_score = _clamp_optional_confidence(boundary_confidence)
    if boundary_score is not None and boundary_score < REVIEW_BOUNDARY_CONFIDENCE_THRESHOLD:
        reasons.append("low_boundary_confidence")
    return reasons


def _dedupe_reasons(reasons: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for reason in reasons or []:
        key = str(reason).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


@dataclass
class CaptionSegment:
    """A caption segment (typically one sentence or phrase)."""
    text: str
    start: float
    end: float
    words: List[Word] = field(default_factory=list)
    speaker: Optional[str] = None
    language: Optional[str] = None
    language_confidence: float = 1.0
    confidence: float = 1.0
    boundary_confidence: Optional[float] = None
    human_review_recommended: bool = False
    review_reasons: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.words and self.confidence == 1.0:
            self.confidence = segment_confidence_from_words(self.words)
        else:
            self.confidence = _clamp_confidence(self.confidence)
        self.language_confidence = _clamp_confidence(self.language_confidence)
        if self.boundary_confidence is None and self.words:
            boundary_scores = [
                score
                for score in (
                    _clamp_optional_confidence(getattr(word, "boundary_confidence", None))
                    for word in self.words
                )
                if score is not None
            ]
            if boundary_scores:
                self.boundary_confidence = round(
                    sum(boundary_scores) / len(boundary_scores),
                    4,
                )
        else:
            self.boundary_confidence = _clamp_optional_confidence(
                self.boundary_confidence
            )
        computed = caption_review_reasons(
            language=self.language,
            language_confidence=self.language_confidence,
            confidence=self.confidence,
            boundary_confidence=self.boundary_confidence,
        )
        self.review_reasons = _dedupe_reasons([*self.review_reasons, *computed])
        self.human_review_recommended = bool(
            self.human_review_recommended or self.review_reasons
        )

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """Complete transcription output."""
    segments: List[CaptionSegment]
    language: str = "en"
    duration: float = 0.0
    language_confidence: float = 1.0
    cache_hit: bool = False
    cache_key: Optional[str] = None
    cache_path: Optional[str] = None
    provenance: ASRProvenance = field(default_factory=ASRProvenance)

    def __post_init__(self) -> None:
        self.language_confidence = _clamp_confidence(self.language_confidence)
        if not isinstance(self.provenance, ASRProvenance):
            self.provenance = provenance_from_dict(self.provenance)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)

    @property
    def word_count(self) -> int:
        return sum(len(s.words) for s in self.segments)

    @property
    def human_review_recommended(self) -> bool:
        return any(getattr(s, "human_review_recommended", False) for s in self.segments)

    @property
    def review_segment_count(self) -> int:
        return sum(1 for s in self.segments if getattr(s, "human_review_recommended", False))


def caption_segment_to_dict(
    seg: CaptionSegment,
    *,
    include_words: bool = True,
    precision: Optional[int] = None,
) -> Dict:
    """Serialize a caption segment with review metadata."""

    def _number(value: Any) -> float:
        try:
            val = float(value)
        except (TypeError, ValueError):
            val = 0.0
        return round(val, precision) if precision is not None else val

    language = getattr(seg, "language", None)
    language_confidence = _clamp_confidence(getattr(seg, "language_confidence", 1.0))
    confidence = _clamp_confidence(getattr(seg, "confidence", 1.0))
    boundary_confidence = _clamp_optional_confidence(
        getattr(seg, "boundary_confidence", None)
    )
    review_reasons = _dedupe_reasons([
        *list(getattr(seg, "review_reasons", []) or []),
        *caption_review_reasons(
            language=language,
            language_confidence=language_confidence,
            confidence=confidence,
            boundary_confidence=boundary_confidence,
        ),
    ])

    payload = {
        "text": str(getattr(seg, "text", "")),
        "start": _number(getattr(seg, "start", 0.0)),
        "end": _number(getattr(seg, "end", 0.0)),
        "speaker": getattr(seg, "speaker", None),
        "language": language,
        "language_confidence": language_confidence,
        "confidence": confidence,
        "boundary_confidence": boundary_confidence,
        "human_review_recommended": bool(getattr(seg, "human_review_recommended", False) or review_reasons),
        "review_reasons": review_reasons,
    }
    if include_words:
        payload["words"] = [
            {
                "text": str(getattr(w, "text", getattr(w, "word", ""))).strip(),
                "start": _number(getattr(w, "start", 0.0)),
                "end": _number(getattr(w, "end", 0.0)),
                "confidence": _clamp_confidence(getattr(w, "confidence", 1.0)),
                "boundary_confidence": _clamp_optional_confidence(
                    getattr(w, "boundary_confidence", None)
                ),
            }
            for w in (getattr(seg, "words", None) or [])
        ]
    return payload


def transcription_result_to_dict(result: TranscriptionResult) -> Dict[str, Any]:
    """Serialize a transcription result for cache storage or JSON export."""
    return {
        "language": getattr(result, "language", "en"),
        "duration": float(getattr(result, "duration", 0.0) or 0.0),
        "language_confidence": _clamp_confidence(
            getattr(result, "language_confidence", 1.0)
        ),
        "provenance": provenance_to_dict(getattr(result, "provenance", None)),
        "segments": [
            caption_segment_to_dict(seg, include_words=True)
            for seg in (getattr(result, "segments", []) or [])
        ],
    }


def _word_from_dict(data: Dict[str, Any]) -> Word:
    return Word(
        text=str(data.get("text", data.get("word", ""))),
        start=data.get("start", 0.0),
        end=data.get("end", 0.0),
        confidence=data.get("confidence", 1.0),
        boundary_confidence=data.get("boundary_confidence"),
    )


def _segment_from_dict(data: Dict[str, Any]) -> CaptionSegment:
    words = [
        _word_from_dict(word_data)
        for word_data in (data.get("words") or [])
        if isinstance(word_data, dict)
    ]
    return CaptionSegment(
        text=str(data.get("text", "")),
        start=data.get("start", 0.0),
        end=data.get("end", 0.0),
        words=words,
        speaker=data.get("speaker"),
        language=data.get("language"),
        language_confidence=data.get("language_confidence", 1.0),
        confidence=data.get("confidence", 1.0),
        boundary_confidence=data.get("boundary_confidence"),
        human_review_recommended=bool(data.get("human_review_recommended", False)),
        review_reasons=[
            str(reason)
            for reason in (data.get("review_reasons") or [])
            if isinstance(reason, (str, int, float))
        ],
    )


def transcription_result_from_dict(
    payload: Dict[str, Any],
    *,
    cache_hit: bool = False,
    cache_key: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> TranscriptionResult:
    """Rehydrate a cached transcription payload."""
    segments = [
        _segment_from_dict(seg_data)
        for seg_data in (payload.get("segments") or [])
        if isinstance(seg_data, dict)
    ]
    return TranscriptionResult(
        segments=segments,
        language=str(payload.get("language") or "en"),
        duration=payload.get("duration", 0.0),
        language_confidence=payload.get("language_confidence", 1.0),
        cache_hit=cache_hit,
        cache_key=cache_key,
        cache_path=cache_path,
        provenance=provenance_from_dict(payload.get("provenance")),
    )


def check_whisper_available() -> Tuple[bool, str]:
    """
    Check which Whisper backend is available.

    Returns:
        Tuple of (available: bool, backend_name: str).
        backend_name is one of: "whisperx", "faster-whisper", "openai-whisper", "none"
    """
    # Try WhisperX first (best word-level timestamps)
    try:
        import whisperx  # noqa: F401
        return True, "whisperx"
    except ImportError:
        pass

    # Try faster-whisper (fastest inference)
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        return True, "faster-whisper"
    except ImportError:
        pass

    # Try openai-whisper (reference implementation)
    try:
        import whisper  # noqa: F401
        return True, "openai-whisper"
    except ImportError:
        pass

    return False, "none"


def _whisper_backend_available(backend: str) -> bool:
    try:
        if backend == "whisperx":
            import whisperx  # noqa: F401
        elif backend == "faster-whisper":
            from faster_whisper import WhisperModel  # noqa: F401
        elif backend == "openai-whisper":
            import whisper  # noqa: F401
        else:
            return False
    except ImportError:
        return False
    return True


def resolve_whisper_backend(override: Optional[str] = None) -> Tuple[str, str]:
    """Resolve an optional strict engine override and its fallback reason."""
    requested = normalize_engine(override)
    if requested and requested != "auto":
        if requested not in {"whisperx", "faster-whisper", "openai-whisper"}:
            raise ValueError(f"Unsupported ASR engine override: {override}")
        if not _whisper_backend_available(requested):
            raise RuntimeError(
                f"Requested ASR engine '{requested}' is not installed; "
                "choose auto or install that engine"
            )
        return requested, ""
    available, backend = check_whisper_available()
    if not available:
        raise RuntimeError(
            "No Whisper backend installed. Install faster-whisper, "
            "openai-whisper, or whisperx."
        )
    return backend, ""


def _backend_provenance(
    engine: str,
    config: CaptionConfig,
    language: str,
    *,
    device: str,
    compute_type: str,
    fallback_reason: str = "",
) -> ASRProvenance:
    provenance = build_provenance(
        engine=engine,
        requested_engine=getattr(config, "engine", None),
        model=config.model,
        model_revision=getattr(config, "model_revision", None),
        requested_language=config.language,
        word_timestamps=config.word_timestamps,
        translate=config.translate,
        diarize=config.diarize,
        min_speakers=config.min_speakers,
        max_speakers=config.max_speakers,
        fallback_reason=fallback_reason,
    )
    provenance.device = device
    provenance.compute_type = compute_type
    apply_language_decision(provenance, language)
    return provenance


def plan_transcription_engine(language=None, override=None):
    """Route a caption/transcription request to the best engine for *language*.

    Delegates to :func:`opencut.core.asr_router.route_asr`: NVIDIA Parakeet for
    its supported European languages (Canary via explicit override) and Whisper
    ``large-v3-turbo`` for the long tail. The decision is honest about stub
    engines, so today it resolves to Whisper until the NeMo adapters/models are
    installed locally. Returns an ``ASRRoute``.
    """
    from opencut.core.asr_router import route_asr

    return route_asr(language=language, override=override)


def remap_captions_to_segments(
    captions: "TranscriptionResult",
    speech_segments: list,
) -> "TranscriptionResult":
    """
    Remap caption timestamps from the original file timeline to a condensed
    timeline where only speech segments are kept (silences removed).

    After silence removal, the kept speech segments are concatenated:
      Original:   [seg0: 0.5-3.0] [gap] [seg1: 5.0-8.0] [gap] [seg2: 10.0-12.0]
      Condensed:  [seg0: 0.0-2.5]       [seg1: 2.5-5.5]       [seg2: 5.5-7.5]

    Each caption's start/end (and word-level timestamps) are remapped so they
    align with the condensed video.  Captions that fall entirely within a
    silence gap are dropped.

    Args:
        captions: Original TranscriptionResult from Whisper.
        speech_segments: List of speech TimeSegment objects (with .start, .end).

    Returns:
        New TranscriptionResult with remapped timestamps.
    """
    if not speech_segments or not captions.segments:
        return captions

    # Sort segments by start time
    sorted_segs = sorted(speech_segments, key=lambda s: s.start)

    # Build cumulative offset table:
    # condensed_starts[i] = start time of segment i in the condensed timeline
    condensed_starts = []
    running = 0.0
    for seg in sorted_segs:
        condensed_starts.append(running)
        running += (seg.end - seg.start)
    total_condensed = running

    def _map_time(t: float) -> float:
        """Map an original-timeline timestamp to the condensed timeline."""
        for i, seg in enumerate(sorted_segs):
            if t < seg.start:
                # Before this segment — snap to its condensed start
                # (this time was in a silence gap or before all speech)
                return condensed_starts[i]
            if t <= seg.end:
                # Inside this segment
                return condensed_starts[i] + (t - seg.start)
        # After all segments — clamp to the end
        return total_condensed

    def _overlaps_speech(start: float, end: float) -> bool:
        """Check if a time range overlaps with any speech segment."""
        for seg in sorted_segs:
            if start < seg.end and end > seg.start:
                return True
        return False

    # Remap each caption segment
    new_segments = []
    for cap in captions.segments:
        # Skip captions that don't overlap with any speech segment
        if not _overlaps_speech(cap.start, cap.end):
            continue

        new_start = _map_time(cap.start)
        new_end = _map_time(cap.end)

        # Ensure minimum duration
        if new_end - new_start < 0.05:
            new_end = new_start + 0.05

        # Remap word-level timestamps
        new_words = []
        for w in cap.words:
            w_start = _map_time(w.start)
            w_end = _map_time(w.end)
            if w_end - w_start < 0.01:
                w_end = w_start + 0.03
            new_words.append(Word(
                text=w.text,
                start=round(w_start, 4),
                end=round(w_end, 4),
                confidence=w.confidence,
                boundary_confidence=getattr(w, "boundary_confidence", None),
            ))

        new_segments.append(CaptionSegment(
            text=cap.text,
            start=round(new_start, 4),
            end=round(new_end, 4),
            words=new_words,
            speaker=cap.speaker,
            language=getattr(cap, "language", None),
            language_confidence=getattr(cap, "language_confidence", 1.0),
            confidence=getattr(cap, "confidence", 1.0),
            boundary_confidence=getattr(cap, "boundary_confidence", None),
            human_review_recommended=getattr(cap, "human_review_recommended", False),
            review_reasons=list(getattr(cap, "review_reasons", []) or []),
        ))

    return TranscriptionResult(
        segments=new_segments,
        language=captions.language,
        duration=total_condensed,
        language_confidence=getattr(captions, "language_confidence", 1.0),
        provenance=getattr(captions, "provenance", ASRProvenance()),
    )


def transcribe(
    filepath: str,
    config: Optional[CaptionConfig] = None,
    timeout: Optional[float] = None,
    *,
    use_cache: bool = True,
) -> TranscriptionResult:
    """
    Transcribe audio/video to text with timestamps.

    Automatically selects the best available Whisper backend.

    Args:
        filepath: Path to the media file.
        config: Caption configuration. Uses defaults if None.
        timeout: Maximum time in seconds for transcription. None = no timeout.
        use_cache: Use the persistent content-addressed transcript cache.

    Returns:
        TranscriptionResult with segments and word-level timestamps.

    Raises:
        RuntimeError: If no Whisper backend is installed.
        TimeoutError: If transcription takes longer than timeout.
    """
    if config is None:
        config = CaptionConfig()

    backend, fallback_reason = resolve_whisper_backend(
        getattr(config, "engine", None)
    )
    provenance = build_provenance(
        engine=backend,
        requested_engine=getattr(config, "engine", None),
        model=config.model,
        model_revision=getattr(config, "model_revision", None),
        requested_language=config.language,
        word_timestamps=config.word_timestamps,
        translate=config.translate,
        diarize=config.diarize,
        min_speakers=config.min_speakers,
        max_speakers=config.max_speakers,
        fallback_reason=fallback_reason,
    )

    cache_key: Optional[str] = None
    cache_metadata: Optional[Dict[str, Any]] = None
    if use_cache and transcript_cache.cache_enabled():
        try:
            cache_key, cache_metadata = transcript_cache.build_cache_key(
                filepath,
                backend=backend,
                config=config,
                extra={"asr_provenance": provenance.cache_identity()},
            )
            cached = transcript_cache.load_transcript(cache_key)
            if not cached:
                legacy_key, _legacy_metadata = transcript_cache.build_legacy_cache_key(
                    filepath,
                    backend=backend,
                    config=config,
                )
                legacy = transcript_cache.load_transcript(
                    legacy_key,
                    count_miss=False,
                )
                if legacy:
                    migrated = transcription_result_from_dict(
                        legacy["result"],
                        cache_hit=True,
                        cache_key=cache_key,
                        cache_path=transcript_cache.cache_entry_path(cache_key),
                    )
                    if migrated.provenance.engine == "legacy-unknown":
                        migrated.provenance = provenance
                        migrated.provenance.fallback_reason = (
                            "migrated from transcript cache schema 1"
                        )
                        apply_language_decision(migrated.provenance, migrated.language)
                    transcript_cache.store_transcript(
                        cache_key,
                        cache_metadata,
                        transcription_result_to_dict(migrated),
                    )
                    return migrated
            if cached:
                logger.info("Transcript cache hit for %s", filepath)
                return transcription_result_from_dict(
                    cached["result"],
                    cache_hit=True,
                    cache_key=cache_key,
                    cache_path=transcript_cache.cache_entry_path(cache_key),
                )
            logger.debug("Transcript cache miss for %s", filepath)
        except OSError as exc:
            logger.debug("Transcript cache key build failed for %s: %s", filepath, exc)
            cache_key = None
            cache_metadata = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Transcript cache read failed for %s: %s", filepath, exc)
            cache_key = None
            cache_metadata = None

    # Extract audio to WAV for consistent input
    logger.debug(f"Extracting audio from {filepath}")
    wav_path = extract_audio_wav(filepath, sample_rate=16000)
    logger.debug(f"Audio extracted to {wav_path}")

    def _do_transcribe():
        if backend == "whisperx":
            return _transcribe_whisperx(wav_path, config)
        elif backend == "faster-whisper":
            return _transcribe_faster_whisper(wav_path, config)
        else:
            return _transcribe_openai_whisper(wav_path, config)

    try:
        if timeout:
            # Use concurrent.futures for timeout support (works on Windows).
            # NOTE: do not use the executor as a context manager — its __exit__
            # calls shutdown(wait=True), which blocks until the worker finishes
            # and turns the timeout into a no-op (the caller stays pinned for
            # the full transcription). Manage it manually and shut down without
            # waiting on timeout so control returns immediately.
            import concurrent.futures
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_do_transcribe)
            try:
                result = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                executor.shutdown(wait=False, cancel_futures=True)
                logger.error(f"Transcription timed out after {timeout}s")
                raise TimeoutError(f"Transcription timed out after {timeout} seconds. "
                                 "Try using a smaller model or enabling CPU mode in Settings.")
            executor.shutdown(wait=False)
        else:
            result = _do_transcribe()

        result.provenance = provenance_from_dict(
            getattr(result, "provenance", provenance.to_dict())
        )
        if result.provenance.engine == "legacy-unknown":
            result.provenance = provenance
        apply_language_decision(result.provenance, result.language)
        result.cache_hit = False
        result.cache_key = cache_key
        if cache_key and cache_metadata:
            try:
                result.cache_path = transcript_cache.store_transcript(
                    cache_key,
                    cache_metadata,
                    transcription_result_to_dict(result),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Transcript cache write failed for %s: %s", filepath, exc)
        return result
    finally:
        if os.path.exists(wav_path) and wav_path.startswith(tempfile.gettempdir()):
            try:
                os.unlink(wav_path)
            except OSError:
                from opencut.helpers import _schedule_temp_cleanup
                _schedule_temp_cleanup(wav_path)


def transcribe_audio(
    filepath: str,
    model: str = "base",
    language: Optional[str] = None,
    timeout: Optional[float] = None,
) -> List[Dict]:
    """Convenience wrapper around :func:`transcribe` returning plain segment dicts.

    The CLI commands ``chapters`` / ``repeat-detect`` / ``search index``
    and other consumers want a flat list of ``{start, end, text}`` dicts
    rather than a :class:`TranscriptionResult` object. This wrapper
    builds a minimal :class:`CaptionConfig` from the keyword arguments
    most callers actually care about (model, language) and projects the
    result down to the segment list.

    Raises the same errors as :func:`transcribe` (RuntimeError when no
    Whisper backend is installed, TimeoutError on overrun).
    """
    config = CaptionConfig(model=model, language=language)
    result = transcribe(filepath, config=config, timeout=timeout)
    segments = getattr(result, "segments", None)
    if segments is None and isinstance(result, dict):
        segments = result.get("segments", [])
    if segments is None:
        return []
    out: List[Dict] = []
    for seg in segments:
        if isinstance(seg, dict):
            out.append(seg)
            continue
        # Dataclass / namespace-style object
        try:
            out.append(caption_segment_to_dict(seg, include_words=True))
        except (TypeError, ValueError):
            continue
    return out


def _transcribe_openai_whisper(wav_path: str, config: CaptionConfig) -> TranscriptionResult:
    """Transcribe using the openai-whisper package."""
    import os

    import whisper

    # Use bundled model path if available
    download_root = os.environ.get("WHISPER_MODELS_DIR", None)
    if download_root and os.path.isdir(download_root):
        model = whisper.load_model(config.model, download_root=download_root)
    else:
        model = whisper.load_model(config.model)

    task = "translate" if config.translate else "transcribe"
    result = model.transcribe(
        wav_path,
        language=config.language,
        task=task,
        word_timestamps=config.word_timestamps,
        verbose=False,
    )

    language = result.get("language", config.language or "en")
    language_confidence = _clamp_confidence(result.get("language_probability", 1.0))

    segments = []
    for seg in result.get("segments", []):
        words = []
        if config.word_timestamps and "words" in seg:
            for w in seg["words"]:
                words.append(Word(
                    text=w.get("word", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    confidence=w.get("probability", 1.0),
                ))
        segment_confidence = _confidence_from_backend_metadata(
            words,
            avg_logprob=seg.get("avg_logprob"),
            no_speech_prob=seg.get("no_speech_prob"),
        )

        segments.append(CaptionSegment(
            text=seg.get("text", "").strip(),
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            words=words,
            language=language,
            language_confidence=language_confidence,
            confidence=segment_confidence,
        ))

    return TranscriptionResult(
        segments=segments,
        language=language,
        language_confidence=language_confidence,
        provenance=_backend_provenance(
            "openai-whisper",
            config,
            language,
            device="cpu-or-cuda",
            compute_type="backend-default",
        ),
    )


def _clear_model_cache(model_name: str):
    """Clear cached files for a specific whisper model to force re-download."""
    import shutil

    # Resolve HF cache dir (respects HF_HOME / HUGGINGFACE_HUB_CACHE env vars)
    hf_home = os.environ.get("HF_HOME", os.environ.get(
        "HUGGINGFACE_HUB_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    ))
    # HF_HOME points to ~/.cache/huggingface; actual hub cache is inside /hub
    if os.path.isdir(os.path.join(hf_home, "hub")):
        hf_home = os.path.join(hf_home, "hub")

    if not os.path.isdir(hf_home):
        return

    # Match model directories like "models--Systran--faster-whisper-base"
    safe_name = model_name.replace(".", "-")
    try:
        items = os.listdir(hf_home)
    except OSError:
        return
    for item in items:
        if "whisper" in item.lower() and safe_name in item.lower():
            target = os.path.join(hf_home, item)
            logger.warning(f"Clearing corrupt model cache: {target}")
            shutil.rmtree(target, ignore_errors=True)

    # Also nuke any orphaned .lock files for this model
    try:
        items = os.listdir(hf_home)
    except OSError:
        return
    for item in items:
        if item.endswith(".lock") and "whisper" in item.lower() and safe_name in item.lower():
            try:
                os.unlink(os.path.join(hf_home, item))
            except Exception:
                pass


def _download_model(model_name: str, requested_revision: Optional[str] = None):
    """Force-download the same immutable faster-whisper revision used at runtime."""
    repo_id, revision = model_identity(
        "faster-whisper",
        model_name,
        requested_revision,
    )
    try:
        from huggingface_hub import snapshot_download
        logger.info(f"Downloading model '{repo_id}' from HuggingFace Hub...")
        snapshot_download(repo_id, revision=revision, force_download=True)
        logger.info(f"Model '{repo_id}' downloaded successfully.")
    except ImportError:
        # huggingface_hub not available — faster-whisper will download on load
        logger.debug("huggingface_hub not available, relying on faster-whisper auto-download")
    except Exception as e:
        logger.warning(f"Model download via huggingface_hub failed: {e}")


def _transcribe_faster_whisper(wav_path: str, config: CaptionConfig) -> TranscriptionResult:
    """Transcribe using faster-whisper (CTranslate2 backend)."""
    from faster_whisper import WhisperModel

    _model_id, resolved_revision = model_identity(
        "faster-whisper",
        config.model,
        getattr(config, "model_revision", None),
    )
    model_revision = None if os.path.isdir(config.model) else resolved_revision
    runtime_fallback_reason = ""

    # Check for forced CPU mode from settings
    force_cpu = False
    try:
        settings_file = os.path.join(os.path.expanduser("~"), ".opencut", "whisper_settings.json")
        if os.path.exists(settings_file):
            import json as _json
            with open(settings_file, "r", encoding="utf-8") as f:
                settings = _json.load(f)
                force_cpu = settings.get("cpu_mode", False)
    except Exception:
        pass

    # Determine device and compute type
    if force_cpu:
        logger.debug("CPU mode enabled via settings - skipping GPU")
        device = "cpu"
        compute_type = "int8"
    else:
        # Try to detect best device - GPU with fallback to CPU
        device = "cuda"
        compute_type = "auto"

        # Check if CUDA is actually available
        try:
            import torch
            if not torch.cuda.is_available():
                logger.debug("CUDA not available, using CPU")
                device = "cpu"
                compute_type = "int8"
        except ImportError:
            # No torch, let faster-whisper decide but be ready to fall back
            pass

    # Try to load model with auto-repair on corrupt cache
    model = None
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            logger.debug(f"Loading Whisper model '{config.model}' on {device} (attempt {attempt + 1}/{max_attempts})")
            model = WhisperModel(
                config.model,
                device=device,
                compute_type=compute_type,
                revision=model_revision,
            )
            break
        except RuntimeError as e:
            err_str = str(e).lower()
            # CUDA library errors - fall back to CPU
            if "cuda" in err_str or "cublas" in err_str or "cudnn" in err_str:
                logger.warning(f"CUDA error, falling back to CPU: {e}")
                runtime_fallback_reason = "CUDA model load failed; retried on CPU"
                device = "cpu"
                compute_type = "int8"
                model = WhisperModel(
                    config.model,
                    device=device,
                    compute_type=compute_type,
                    revision=model_revision,
                )
                break
            # Corrupt model cache - clear, re-download, and retry
            elif ("unable to open file" in err_str or "model.bin" in err_str
                  or "corrupt" in err_str or "invalid" in err_str
                  or "no such file" in err_str or "not found" in err_str):
                if attempt < max_attempts - 1:
                    logger.warning(f"Model cache appears corrupt (attempt {attempt + 1}), purging and re-downloading: {e}")
                    _clear_model_cache(config.model)
                    _download_model(
                        config.model,
                        getattr(config, "model_revision", None),
                    )
                    continue
                else:
                    raise RuntimeError(
                        f"Model '{config.model}' failed to load after {max_attempts} attempts. "
                        f"Please check your internet connection and disk space, then try again. Error: {e}"
                    )
            else:
                raise
        except Exception as e:
            err_str = str(e).lower()
            if ("unable to open file" in err_str or "model.bin" in err_str
                or "corrupt" in err_str or "no such file" in err_str
                or "not found" in err_str):
                if attempt < max_attempts - 1:
                    logger.warning(f"Model cache error (attempt {attempt + 1}), purging and re-downloading: {e}")
                    _clear_model_cache(config.model)
                    _download_model(
                        config.model,
                        getattr(config, "model_revision", None),
                    )
                    continue
                else:
                    raise
            else:
                raise

    task = "translate" if config.translate else "transcribe"

    logger.info(f"Starting transcription of {wav_path}")

    try:
        # Also wrap transcription in case CUDA fails mid-process
        try:
            result_segments, info = model.transcribe(
                wav_path,
                language=config.language,
                task=task,
                word_timestamps=config.word_timestamps,
                vad_filter=True,
            )
            # Consume the generator to catch any CUDA errors during processing
            result_segments = list(result_segments)
        except RuntimeError as e:
            if "cuda" in str(e).lower() or "cublas" in str(e).lower() or "cudnn" in str(e).lower():
                logger.warning(f"CUDA error during transcription, retrying with CPU: {e}")
                runtime_fallback_reason = "CUDA inference failed; retried on CPU"
                device = "cpu"
                compute_type = "int8"
                model = WhisperModel(
                    config.model,
                    device=device,
                    compute_type=compute_type,
                    revision=model_revision,
                )
                result_segments, info = model.transcribe(
                    wav_path,
                    language=config.language,
                    task=task,
                    word_timestamps=config.word_timestamps,
                    vad_filter=True,
                )
                result_segments = list(result_segments)
            else:
                raise

        logger.info(f"Transcription complete: {len(result_segments)} segments")

        language = info.language or config.language or "en"
        language_confidence = _clamp_confidence(getattr(info, "language_probability", 1.0))

        segments = []
        for seg in result_segments:
            words = []
            if config.word_timestamps and seg.words:
                for w in seg.words:
                    words.append(Word(
                        text=w.word,
                        start=w.start,
                        end=w.end,
                        confidence=w.probability,
                    ))
            segment_confidence = _confidence_from_backend_metadata(
                words,
                avg_logprob=getattr(seg, "avg_logprob", None),
                no_speech_prob=getattr(seg, "no_speech_prob", None),
            )

            segments.append(CaptionSegment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                words=words,
                language=language,
                language_confidence=language_confidence,
                confidence=segment_confidence,
            ))

        return TranscriptionResult(
            segments=segments,
            language=language,
            language_confidence=language_confidence,
            provenance=_backend_provenance(
                "faster-whisper",
                config,
                language,
                device=device,
                compute_type=compute_type,
                fallback_reason=runtime_fallback_reason,
            ),
        )
    finally:
        try:
            del model
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _transcribe_whisperx(wav_path: str, config: CaptionConfig) -> TranscriptionResult:
    """Transcribe using WhisperX (best word-level alignment).

    If ``config.diarize`` is True and a Hugging Face token is available
    (via ``config.hf_token`` or the ``HF_TOKEN`` env var), WhisperX's
    ``DiarizationPipeline`` tags each caption segment with a
    ``speaker`` label (SPEAKER_00, SPEAKER_01, …) using pyannote 3.x.
    Falls back silently to untagged segments if the token is missing or
    pyannote models can't be loaded.
    """
    import os

    import torch
    import whisperx
    from faster_whisper.utils import download_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Resolve named checkpoints through the immutable revision recorded in the
    # provenance contract, then pass the snapshot path to WhisperX.
    model_source = config.model
    if not os.path.isdir(config.model):
        _model_id, resolved_revision = model_identity(
            "whisperx",
            config.model,
            getattr(config, "model_revision", None),
        )
        model_source = download_model(config.model, revision=resolved_revision)

    # Load model and transcribe
    model = whisperx.load_model(model_source, device, compute_type=compute_type)
    align_model = None
    diarize_pipeline = None
    try:
        audio = whisperx.load_audio(wav_path)
        result = model.transcribe(audio, batch_size=16, language=config.language)

        # Align for word-level timestamps
        if config.word_timestamps:
            align_model, metadata = whisperx.load_align_model(
                language_code=result.get("language", "en"),
                device=device,
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                device,
            )

        # Optional diarisation — gated on token + flag
        if getattr(config, "diarize", False):
            hf_token = getattr(config, "hf_token", None) or os.environ.get("HF_TOKEN")
            if hf_token:
                try:
                    # DiarizationPipeline lives under different names in
                    # different WhisperX versions — try both.
                    Pipeline = (
                        getattr(whisperx, "DiarizationPipeline", None)
                        or getattr(whisperx, "diarize", None)
                    )
                    if Pipeline is None:
                        # Fall back to pyannote directly
                        from pyannote.audio import Pipeline as PyaPipeline  # noqa: N811
                        diarize_pipeline = PyaPipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=hf_token,
                        )
                        # pyannote returns a pyannote.core.Annotation
                        diar_result = diarize_pipeline(
                            {"uri": "audio", "audio": wav_path},
                            min_speakers=config.min_speakers,
                            max_speakers=config.max_speakers,
                        )
                        # Transform Annotation → WhisperX-compatible segments
                        import pandas as _pd  # WhisperX uses a DataFrame
                        rows = []
                        for turn, _, speaker in diar_result.itertracks(yield_label=True):
                            rows.append({
                                "start": float(turn.start),
                                "end": float(turn.end),
                                "speaker": speaker,
                            })
                        if rows:
                            diar_segments = _pd.DataFrame(rows)
                            result = whisperx.assign_word_speakers(diar_segments, result)
                    else:
                        # Newer WhisperX: keep DiarizationPipeline path
                        diarize_pipeline = Pipeline(
                            use_auth_token=hf_token,
                            device=device,
                        )
                        diar_segments = diarize_pipeline(
                            audio,
                            min_speakers=config.min_speakers,
                            max_speakers=config.max_speakers,
                        )
                        result = whisperx.assign_word_speakers(diar_segments, result)
                except Exception as exc:  # noqa: BLE001
                    # Never kill transcription over a diarisation failure —
                    # log and fall through with plain segments.
                    logger.warning("WhisperX diarisation failed: %s", exc)
    finally:
        try:
            del model
        except Exception:
            pass
        if align_model is not None:
            try:
                del align_model
            except Exception:
                pass
        if diarize_pipeline is not None:
            try:
                del diarize_pipeline
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    language = result.get("language", config.language or "en")
    language_confidence = _clamp_confidence(result.get("language_probability", 1.0))

    segments = []
    for seg in result.get("segments", []):
        words = []
        if config.word_timestamps:
            for w in seg.get("words", []):
                words.append(Word(
                    text=w.get("word", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    confidence=w.get("probability", 1.0),
                    boundary_confidence=w.get("score"),
                ))
        segment_confidence = _confidence_from_backend_metadata(
            words,
            avg_logprob=seg.get("avg_logprob"),
            no_speech_prob=seg.get("no_speech_prob"),
        )

        segments.append(CaptionSegment(
            text=seg.get("text", "").strip(),
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            words=words,
            speaker=seg.get("speaker") if isinstance(seg.get("speaker"), str) else None,
            language=language,
            language_confidence=language_confidence,
            confidence=segment_confidence,
        ))

    provenance = _backend_provenance(
        "whisperx",
        config,
        language,
        device=device,
        compute_type=compute_type,
    )
    if config.word_timestamps:
        align_model_id, align_revision = whisperx_alignment_identity(language)
        provenance.alignment_model_id = align_model_id
        provenance.alignment_model_revision = align_revision

    return TranscriptionResult(
        segments=segments,
        language=language,
        language_confidence=language_confidence,
        provenance=provenance,
    )
