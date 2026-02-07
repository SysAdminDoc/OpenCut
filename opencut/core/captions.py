"""
Caption/subtitle generation using OpenAI Whisper.

Supports multiple Whisper backends: openai-whisper, faster-whisper, whisperx.
Falls back gracefully if Whisper is not installed.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..utils.config import CaptionConfig
from .audio import extract_audio_wav


@dataclass
class Word:
    """A single word with timestamp."""
    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class CaptionSegment:
    """A caption segment (typically one sentence or phrase)."""
    text: str
    start: float
    end: float
    words: List[Word] = field(default_factory=list)
    speaker: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """Complete transcription output."""
    segments: List[CaptionSegment]
    language: str = "en"
    duration: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)

    @property
    def word_count(self) -> int:
        return sum(len(s.words) for s in self.segments)


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
            ))

        new_segments.append(CaptionSegment(
            text=cap.text,
            start=round(new_start, 4),
            end=round(new_end, 4),
            words=new_words,
            speaker=cap.speaker,
        ))

    return TranscriptionResult(
        segments=new_segments,
        language=captions.language,
        duration=total_condensed,
    )


def transcribe(
    filepath: str,
    config: Optional[CaptionConfig] = None,
) -> TranscriptionResult:
    """
    Transcribe audio/video to text with timestamps.

    Automatically selects the best available Whisper backend.

    Args:
        filepath: Path to the media file.
        config: Caption configuration. Uses defaults if None.

    Returns:
        TranscriptionResult with segments and word-level timestamps.

    Raises:
        RuntimeError: If no Whisper backend is installed.
    """
    if config is None:
        config = CaptionConfig()

    available, backend = check_whisper_available()

    if not available:
        raise RuntimeError(
            "No Whisper backend installed. Install one of:\n"
            "  pip install openai-whisper        # Reference implementation\n"
            "  pip install faster-whisper         # Fastest (recommended)\n"
            "  pip install whisperx               # Best word timestamps\n"
        )

    # Extract audio to WAV for consistent input
    wav_path = extract_audio_wav(filepath, sample_rate=16000)

    try:
        if backend == "whisperx":
            return _transcribe_whisperx(wav_path, config)
        elif backend == "faster-whisper":
            return _transcribe_faster_whisper(wav_path, config)
        else:
            return _transcribe_openai_whisper(wav_path, config)
    finally:
        # Cleanup temp wav
        if os.path.exists(wav_path) and wav_path.startswith(tempfile.gettempdir()):
            os.unlink(wav_path)


def _transcribe_openai_whisper(wav_path: str, config: CaptionConfig) -> TranscriptionResult:
    """Transcribe using the openai-whisper package."""
    import whisper

    model = whisper.load_model(config.model)

    task = "translate" if config.translate else "transcribe"
    result = model.transcribe(
        wav_path,
        language=config.language,
        task=task,
        word_timestamps=config.word_timestamps,
        verbose=False,
    )

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

        segments.append(CaptionSegment(
            text=seg.get("text", "").strip(),
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            words=words,
        ))

    return TranscriptionResult(
        segments=segments,
        language=result.get("language", config.language or "en"),
    )


def _transcribe_faster_whisper(wav_path: str, config: CaptionConfig) -> TranscriptionResult:
    """Transcribe using faster-whisper (CTranslate2 backend)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(config.model, compute_type="auto")

    task = "translate" if config.translate else "transcribe"
    result_segments, info = model.transcribe(
        wav_path,
        language=config.language,
        task=task,
        word_timestamps=config.word_timestamps,
        vad_filter=True,
    )

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

        segments.append(CaptionSegment(
            text=seg.text.strip(),
            start=seg.start,
            end=seg.end,
            words=words,
        ))

    return TranscriptionResult(
        segments=segments,
        language=info.language or config.language or "en",
    )


def _transcribe_whisperx(wav_path: str, config: CaptionConfig) -> TranscriptionResult:
    """Transcribe using WhisperX (best word-level alignment)."""
    import whisperx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Load model and transcribe
    model = whisperx.load_model(config.model, device, compute_type=compute_type)
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

    segments = []
    for seg in result.get("segments", []):
        words = []
        if config.word_timestamps:
            for w in seg.get("words", []):
                words.append(Word(
                    text=w.get("word", ""),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    confidence=w.get("score", 1.0),
                ))

        segments.append(CaptionSegment(
            text=seg.get("text", "").strip(),
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            words=words,
        ))

    return TranscriptionResult(
        segments=segments,
        language=result.get("language", config.language or "en"),
    )
