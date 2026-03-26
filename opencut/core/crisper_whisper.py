"""
CrisperWhisper — Verbatim filler word detection via modified Whisper.

CrisperWhisper transcribes speech *verbatim*, explicitly marking filler words
as [UH] and [UM] with precise word-level timestamps. This is far more accurate
than standard Whisper + text matching, as standard Whisper actively omits fillers.

Requires: pip install transformers torch

Model: nyrahealth/CrisperWhisper (HuggingFace)
Paper: https://github.com/nyrahealth/CrisperWhisper
"""

import logging
from typing import List, Optional, Callable

logger = logging.getLogger("opencut")

# Filler tokens that CrisperWhisper marks in output
FILLER_TOKENS = {"[UH]", "[UM]", "[UHM]"}

# Common text-based filler words for fallback detection
TEXT_FILLERS = {
    "um", "uh", "uhm", "uh-huh", "mm-hmm", "hmm",
    "like", "you know", "i mean", "basically", "actually",
    "literally", "right", "so", "well", "er", "ah",
}


def detect_fillers_crisper(
    filepath: str,
    language: Optional[str] = None,
    custom_words: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Detect filler words using CrisperWhisper (verbatim ASR with filler marking).

    Args:
        filepath: Path to audio/video file.
        language: Language code or None for auto-detect.
        custom_words: Additional words to treat as fillers.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with keys: fillers (list of {word, start, end}), count, total_filler_time,
        cuts (list of {start, end} for removal), transcript (full text).

    Raises:
        ImportError: If transformers/torch not installed.
    """
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError:
        raise ImportError(
            "CrisperWhisper requires transformers and torch. "
            "Install with: pip install transformers torch"
        )

    if on_progress:
        on_progress(5, "Loading CrisperWhisper model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "nyrahealth/CrisperWhisper"

    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)

        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        logger.warning("CrisperWhisper model not available: %s. Falling back to faster-whisper.", e)
        return _fallback_filler_detection(filepath, custom_words, on_progress)

    if on_progress:
        on_progress(20, "Transcribing with CrisperWhisper (verbatim mode)...")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps="word",
    )

    generate_kwargs = {}
    if language:
        generate_kwargs["language"] = language

    result = pipe(filepath, generate_kwargs=generate_kwargs)

    if on_progress:
        on_progress(70, "Analyzing filler words...")

    # Build filler detection set
    filler_set = set(FILLER_TOKENS)
    if custom_words:
        for w in custom_words:
            filler_set.add(w.strip().lower())

    # Extract fillers from word-level timestamps
    fillers = []
    chunks = result.get("chunks", [])

    for chunk in chunks:
        word = chunk.get("text", "").strip()
        timestamps = chunk.get("timestamp", (None, None))

        if timestamps[0] is None or timestamps[1] is None:
            continue

        start = float(timestamps[0])
        end = float(timestamps[1])

        # Check if this is a filler token ([UH], [UM]) or a text-match filler
        word_clean = word.strip(".,!?;:\"'()[]{}").lower()
        is_filler = (
            word in FILLER_TOKENS
            or word_clean in FILLER_TOKENS
            or word_clean in filler_set
            or word_clean in TEXT_FILLERS
        )

        if is_filler:
            fillers.append({
                "word": word,
                "start": round(start, 3),
                "end": round(end, 3),
            })

    if on_progress:
        on_progress(90, f"Found {len(fillers)} filler(s)...")

    # Build cuts (regions to remove)
    total_filler_time = sum(f["end"] - f["start"] for f in fillers)
    cuts = [{"start": f["start"], "end": f["end"]} for f in fillers]

    return {
        "fillers": fillers,
        "count": len(fillers),
        "total_filler_time": round(total_filler_time, 3),
        "cuts": cuts,
        "transcript": result.get("text", ""),
        "method": "crisper_whisper",
    }


def _fallback_filler_detection(
    filepath: str,
    custom_words: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Fallback filler detection using faster-whisper + text matching.

    Used when CrisperWhisper model is not available.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "Neither CrisperWhisper nor faster-whisper is available. "
            "Install with: pip install faster-whisper"
        )

    if on_progress:
        on_progress(10, "Falling back to faster-whisper for filler detection...")

    model = WhisperModel("base", device="auto", compute_type="auto")

    segments, info = model.transcribe(
        filepath,
        word_timestamps=True,
        vad_filter=True,
    )

    if on_progress:
        on_progress(50, "Scanning for filler words...")

    filler_set = set(TEXT_FILLERS)
    if custom_words:
        for w in custom_words:
            filler_set.add(w.strip().lower())

    fillers = []
    for segment in segments:
        if not segment.words:
            continue
        for word in segment.words:
            word_clean = word.word.strip(".,!?;:\"'()[]{}").strip().lower()
            if word_clean in filler_set:
                fillers.append({
                    "word": word.word.strip(),
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                })

    total_filler_time = sum(f["end"] - f["start"] for f in fillers)
    cuts = [{"start": f["start"], "end": f["end"]} for f in fillers]

    return {
        "fillers": fillers,
        "count": len(fillers),
        "total_filler_time": round(total_filler_time, 3),
        "cuts": cuts,
        "transcript": "",
        "method": "faster_whisper_fallback",
    }
