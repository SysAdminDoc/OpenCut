"""
Caption/subtitle generation using OpenAI Whisper.

Supports multiple Whisper backends: openai-whisper, faster-whisper, whisperx.
Falls back gracefully if Whisper is not installed.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..utils.config import CaptionConfig
from .audio import extract_audio_wav

logger = logging.getLogger(__name__)


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
    timeout: Optional[float] = None,
) -> TranscriptionResult:
    """
    Transcribe audio/video to text with timestamps.

    Automatically selects the best available Whisper backend.

    Args:
        filepath: Path to the media file.
        config: Caption configuration. Uses defaults if None.
        timeout: Maximum time in seconds for transcription. None = no timeout.

    Returns:
        TranscriptionResult with segments and word-level timestamps.

    Raises:
        RuntimeError: If no Whisper backend is installed.
        TimeoutError: If transcription takes longer than timeout.
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
    logger.info(f"Extracting audio from {filepath}")
    wav_path = extract_audio_wav(filepath, sample_rate=16000)
    logger.info(f"Audio extracted to {wav_path}")

    def _do_transcribe():
        if backend == "whisperx":
            return _transcribe_whisperx(wav_path, config)
        elif backend == "faster-whisper":
            return _transcribe_faster_whisper(wav_path, config)
        else:
            return _transcribe_openai_whisper(wav_path, config)

    try:
        if timeout:
            # Use concurrent.futures for timeout support (works on Windows)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_transcribe)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Transcription timed out after {timeout}s")
                    raise TimeoutError(f"Transcription timed out after {timeout} seconds. "
                                     "Try using a smaller model or enabling CPU mode in Settings.")
        else:
            return _do_transcribe()
    finally:
        # Cleanup temp wav
        if os.path.exists(wav_path) and wav_path.startswith(tempfile.gettempdir()):
            try:
                os.unlink(wav_path)
            except Exception:
                pass


def _transcribe_openai_whisper(wav_path: str, config: CaptionConfig) -> TranscriptionResult:
    """Transcribe using the openai-whisper package."""
    import whisper
    import os

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


def _clear_model_cache(model_name: str):
    """Clear cached files for a specific whisper model to force re-download."""
    import shutil
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if not os.path.isdir(cache_dir):
        return
    # Match model directories like "models--Systran--faster-whisper-base"
    for item in os.listdir(cache_dir):
        if "whisper" in item.lower() and model_name.replace(".", "-") in item.lower():
            target = os.path.join(cache_dir, item)
            logger.warning(f"Clearing corrupt model cache: {target}")
            shutil.rmtree(target, ignore_errors=True)


def _transcribe_faster_whisper(wav_path: str, config: CaptionConfig) -> TranscriptionResult:
    """Transcribe using faster-whisper (CTranslate2 backend)."""
    from faster_whisper import WhisperModel
    
    # Check for forced CPU mode from settings
    force_cpu = False
    try:
        settings_file = os.path.join(os.path.expanduser("~"), ".opencut", "whisper_settings.json")
        if os.path.exists(settings_file):
            import json as _json
            with open(settings_file, "r") as f:
                settings = _json.load(f)
                force_cpu = settings.get("cpu_mode", False)
    except Exception:
        pass
    
    # Determine device and compute type
    if force_cpu:
        logger.info("CPU mode enabled via settings - skipping GPU")
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
                logger.info("CUDA not available, using CPU")
                device = "cpu"
                compute_type = "int8"
        except ImportError:
            # No torch, let faster-whisper decide but be ready to fall back
            pass
    
    # Try to load model with auto-repair on corrupt cache
    model = None
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            logger.info(f"Loading Whisper model '{config.model}' on {device} (attempt {attempt + 1})")
            model = WhisperModel(config.model, device=device, compute_type=compute_type)
            break
        except RuntimeError as e:
            err_str = str(e).lower()
            # CUDA library errors - fall back to CPU
            if "cuda" in err_str or "cublas" in err_str or "cudnn" in err_str:
                logger.warning(f"CUDA error, falling back to CPU: {e}")
                device = "cpu"
                compute_type = "int8"
                model = WhisperModel(config.model, device=device, compute_type=compute_type)
                break
            # Corrupt model cache - clear and retry
            elif ("unable to open file" in err_str or "model.bin" in err_str
                  or "corrupt" in err_str or "invalid" in err_str):
                if attempt < max_attempts - 1:
                    logger.warning(f"Model cache appears corrupt, clearing and re-downloading: {e}")
                    _clear_model_cache(config.model)
                    continue
                else:
                    raise RuntimeError(
                        f"Model '{config.model}' failed to load after re-download. "
                        f"Try: Settings > Clear Cache > Reinstall Whisper. Error: {e}"
                    )
            else:
                raise
        except Exception as e:
            err_str = str(e).lower()
            if ("unable to open file" in err_str or "model.bin" in err_str
                or "corrupt" in err_str or "no such file" in err_str):
                if attempt < max_attempts - 1:
                    logger.warning(f"Model cache error, clearing and re-downloading: {e}")
                    _clear_model_cache(config.model)
                    continue
                else:
                    raise
            else:
                raise
    
    task = "translate" if config.translate else "transcribe"
    
    logger.info(f"Starting transcription of {wav_path}")
    
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
            model = WhisperModel(config.model, device="cpu", compute_type="int8")
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
