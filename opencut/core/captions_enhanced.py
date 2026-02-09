"""
OpenCut Enhanced Captions Module v0.7.1

Enhanced caption features:
- WhisperX word-level phoneme alignment (wav2vec2)
- Speaker diarization labels
- NLLB translation (200+ languages via CTranslate2)
- ASS karaoke subtitle export with smooth fill/wipe
- pysubs2 subtitle format conversion (SRT/ASS/VTT/MicroDVD)

WhisperX extends faster-whisper with precise word boundaries.
NLLB runs through CTranslate2 (already a dep via faster-whisper).
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


def _ensure_package(pkg_name: str, pip_name: str = None, on_progress: Callable = None):
    try:
        __import__(pkg_name)
        return True
    except ImportError:
        pip_name = pip_name or pkg_name
        if on_progress:
            on_progress(5, f"Installing {pip_name}...")
        logger.info(f"Installing: {pip_name}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name,
             "--break-system-packages", "-q"],
            capture_output=True, timeout=600,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")
            raise RuntimeError(f"Failed to install {pip_name}: {err[-300:]}")
        return True


# ---------------------------------------------------------------------------
# Availability Checks
# ---------------------------------------------------------------------------
def check_whisperx_available() -> bool:
    try:
        import whisperx  # noqa: F401
        return True
    except ImportError:
        return False


def check_nllb_available() -> bool:
    try:
        import ctranslate2  # noqa: F401
        return True
    except ImportError:
        return False


def check_pysubs2_available() -> bool:
    try:
        import pysubs2  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# WhisperX Word-Level Alignment
# ---------------------------------------------------------------------------
def whisperx_transcribe(
    audio_path: str,
    model_size: str = "base",
    language: str = "",
    diarize: bool = False,
    hf_token: str = "",
    device: str = "auto",
    on_progress: Optional[Callable] = None,
) -> Dict:
    """
    Transcribe audio with WhisperX for precise word-level timestamps.

    Uses wav2vec2 phoneme alignment on top of faster-whisper for
    60-70x realtime speed with per-word start/end timestamps.

    Args:
        audio_path: Path to audio/video file.
        model_size: Whisper model (tiny, base, small, medium, large-v3).
        language: Language code (auto-detect if empty).
        diarize: Enable speaker diarization (requires HuggingFace token).
        hf_token: HuggingFace API token for diarization model.
        device: "auto", "cuda", or "cpu".

    Returns:
        Dict with segments, each containing words with precise timestamps.
    """
    _ensure_package("whisperx", "whisperx", on_progress)
    import whisperx
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    if on_progress:
        on_progress(10, f"Loading WhisperX model ({model_size})...")

    # Step 1: Transcribe with batched inference
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)

    if on_progress:
        on_progress(25, "Transcribing audio...")

    result = model.transcribe(audio, batch_size=16, language=language or None)
    detected_lang = result.get("language", language or "en")

    if on_progress:
        on_progress(50, "Aligning word timestamps (wav2vec2)...")

    # Step 2: Word-level alignment
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_lang, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, align_metadata,
        audio, device, return_char_alignments=False
    )

    # Step 3: Optional speaker diarization
    if diarize and hf_token:
        if on_progress:
            on_progress(70, "Running speaker diarization...")
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device=device
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            logger.warning(f"Diarization failed: {e}")

    if on_progress:
        on_progress(90, "Formatting results...")

    # Format output
    segments = []
    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []):
            words.append({
                "text": w.get("word", ""),
                "start": round(w.get("start", 0), 3),
                "end": round(w.get("end", 0), 3),
                "score": round(w.get("score", 1.0), 3),
            })
        segments.append({
            "text": seg.get("text", "").strip(),
            "start": round(seg.get("start", 0), 3),
            "end": round(seg.get("end", 0), 3),
            "words": words,
            "speaker": seg.get("speaker", None),
        })

    # Free GPU memory
    try:
        del model, align_model
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    if on_progress:
        on_progress(100, "WhisperX transcription complete")

    return {
        "segments": segments,
        "language": detected_lang,
        "word_count": sum(len(s["words"]) for s in segments),
    }


# ---------------------------------------------------------------------------
# NLLB Translation
# ---------------------------------------------------------------------------
# Common language codes for NLLB
NLLB_LANGUAGES = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "ru": "rus_Cyrl",
    "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang", "ar": "arb_Arab",
    "hi": "hin_Deva", "tr": "tur_Latn", "pl": "pol_Latn", "sv": "swe_Latn",
    "da": "dan_Latn", "no": "nob_Latn", "fi": "fin_Latn", "el": "ell_Grek",
    "cs": "ces_Latn", "ro": "ron_Latn", "hu": "hun_Latn", "uk": "ukr_Cyrl",
    "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn", "ms": "zsm_Latn",
    "he": "heb_Hebr", "fa": "pes_Arab", "bg": "bul_Cyrl", "hr": "hrv_Latn",
    "sk": "slk_Latn", "lt": "lit_Latn", "lv": "lvs_Latn", "et": "est_Latn",
    "sl": "slv_Latn", "mk": "mkd_Cyrl", "bn": "ben_Beng", "ta": "tam_Taml",
    "te": "tel_Telu", "ml": "mal_Mlym", "ur": "urd_Arab", "sw": "swh_Latn",
}

TRANSLATION_LANGUAGES = [
    {"code": "en", "label": "English"},
    {"code": "es", "label": "Spanish"},
    {"code": "fr", "label": "French"},
    {"code": "de", "label": "German"},
    {"code": "it", "label": "Italian"},
    {"code": "pt", "label": "Portuguese"},
    {"code": "nl", "label": "Dutch"},
    {"code": "ru", "label": "Russian"},
    {"code": "zh", "label": "Chinese (Simplified)"},
    {"code": "ja", "label": "Japanese"},
    {"code": "ko", "label": "Korean"},
    {"code": "ar", "label": "Arabic"},
    {"code": "hi", "label": "Hindi"},
    {"code": "tr", "label": "Turkish"},
    {"code": "pl", "label": "Polish"},
    {"code": "sv", "label": "Swedish"},
    {"code": "th", "label": "Thai"},
    {"code": "vi", "label": "Vietnamese"},
    {"code": "id", "label": "Indonesian"},
]


def translate_text(
    text: str,
    source_lang: str = "en",
    target_lang: str = "es",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Translate text using NLLB via CTranslate2.

    CTranslate2 is already a dependency of faster-whisper.
    Uses the distilled 600M NLLB model for good quality with fast inference.
    """
    _ensure_package("ctranslate2", "ctranslate2", on_progress)
    _ensure_package("sentencepiece", "sentencepiece", on_progress)
    import ctranslate2
    import sentencepiece

    model_dir = os.path.join(os.path.expanduser("~"), ".opencut", "models", "nllb-200-distilled-600M")

    if not os.path.isdir(model_dir):
        if on_progress:
            on_progress(10, "Downloading NLLB translation model (~1.2GB)...")
        _ensure_package("huggingface_hub", "huggingface-hub", on_progress)
        from huggingface_hub import snapshot_download
        snapshot_download(
            "JustFrederik/nllb-200-distilled-600M-ct2-float16",
            local_dir=model_dir,
        )

    src_code = NLLB_LANGUAGES.get(source_lang, f"{source_lang}_Latn")
    tgt_code = NLLB_LANGUAGES.get(target_lang, f"{target_lang}_Latn")

    if on_progress:
        on_progress(30, "Loading translation model...")

    translator = ctranslate2.Translator(model_dir, device="auto")
    sp_path = os.path.join(model_dir, "sentencepiece.bpe.model")

    if not os.path.isfile(sp_path):
        # Try alternate name
        sp_path = os.path.join(model_dir, "source.spm")
    if not os.path.isfile(sp_path):
        raise FileNotFoundError(f"SentencePiece model not found in {model_dir}")

    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(sp_path)

    if on_progress:
        on_progress(50, "Translating...")

    # Tokenize and translate
    source_tokens = sp.Encode(text, out_type=str)
    source_tokens = [src_code] + source_tokens

    results = translator.translate_batch(
        [source_tokens],
        target_prefix=[[tgt_code]],
        max_batch_size=1,
        beam_size=4,
    )

    translated_tokens = results[0].hypotheses[0]
    # Remove language tag from output
    if translated_tokens and translated_tokens[0] == tgt_code:
        translated_tokens = translated_tokens[1:]

    translated = sp.Decode(translated_tokens)

    if on_progress:
        on_progress(100, "Translation complete")

    del translator
    return translated


def translate_segments(
    segments: List[Dict],
    source_lang: str = "en",
    target_lang: str = "es",
    on_progress: Optional[Callable] = None,
) -> List[Dict]:
    """
    Translate a list of caption segments, preserving timestamps.
    """
    _ensure_package("ctranslate2", "ctranslate2", on_progress)
    _ensure_package("sentencepiece", "sentencepiece", on_progress)
    import ctranslate2
    import sentencepiece

    model_dir = os.path.join(os.path.expanduser("~"), ".opencut", "models", "nllb-200-distilled-600M")

    if not os.path.isdir(model_dir):
        if on_progress:
            on_progress(10, "Downloading NLLB model (~1.2GB)...")
        _ensure_package("huggingface_hub", "huggingface-hub", on_progress)
        from huggingface_hub import snapshot_download
        snapshot_download(
            "JustFrederik/nllb-200-distilled-600M-ct2-float16",
            local_dir=model_dir,
        )

    src_code = NLLB_LANGUAGES.get(source_lang, f"{source_lang}_Latn")
    tgt_code = NLLB_LANGUAGES.get(target_lang, f"{target_lang}_Latn")

    if on_progress:
        on_progress(20, "Loading translation model...")

    translator = ctranslate2.Translator(model_dir, device="auto")
    sp_path = os.path.join(model_dir, "sentencepiece.bpe.model")
    if not os.path.isfile(sp_path):
        sp_path = os.path.join(model_dir, "source.spm")

    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(sp_path)

    translated_segments = []
    total = len(segments)
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            translated_segments.append(seg.copy())
            continue

        source_tokens = [src_code] + sp.Encode(text, out_type=str)
        results = translator.translate_batch(
            [source_tokens],
            target_prefix=[[tgt_code]],
            beam_size=4,
        )
        out_tokens = results[0].hypotheses[0]
        if out_tokens and out_tokens[0] == tgt_code:
            out_tokens = out_tokens[1:]
        translated_text = sp.Decode(out_tokens)

        new_seg = seg.copy()
        new_seg["text"] = translated_text
        new_seg["original_text"] = text
        translated_segments.append(new_seg)

        if on_progress and i % max(1, total // 10) == 0:
            pct = 20 + int((i / total) * 75)
            on_progress(pct, f"Translating {i+1}/{total}...")

    del translator
    if on_progress:
        on_progress(100, "Translation complete")

    return translated_segments


# ---------------------------------------------------------------------------
# ASS Karaoke Export
# ---------------------------------------------------------------------------
def segments_to_ass_karaoke(
    segments: List[Dict],
    output_path: str,
    style: str = "Default",
    font_name: str = "Arial",
    font_size: int = 48,
    primary_color: str = "&H00FFFFFF",
    highlight_color: str = "&H0000FFFF",
    outline_color: str = "&H00000000",
    outline_width: int = 3,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Export segments with word-level timestamps to ASS with karaoke tags.

    Uses \\kf (smooth fill) for word-by-word highlight animation.
    """
    _ensure_package("pysubs2", "pysubs2", on_progress)
    import pysubs2

    subs = pysubs2.SSAFile()

    # Configure style
    s = pysubs2.SSAStyle()
    s.fontname = font_name
    s.fontsize = font_size
    s.primarycolor = pysubs2.Color(*_parse_ass_color(primary_color))
    s.secondarycolor = pysubs2.Color(*_parse_ass_color(highlight_color))
    s.outlinecolor = pysubs2.Color(*_parse_ass_color(outline_color))
    s.outline = outline_width
    s.shadow = 1
    s.alignment = 2  # Bottom center
    s.marginv = 40
    subs.styles[style] = s

    for seg in segments:
        words = seg.get("words", [])
        if not words:
            # No word-level timing, use plain text
            evt = pysubs2.SSAEvent(
                start=int(seg["start"] * 1000),
                end=int(seg["end"] * 1000),
                text=seg["text"],
                style=style,
            )
            subs.events.append(evt)
            continue

        # Build karaoke line with \kf tags
        parts = []
        for w in words:
            duration_cs = max(1, int((w["end"] - w["start"]) * 100))
            word_text = w["text"].strip()
            if word_text:
                parts.append(f"{{\\kf{duration_cs}}}{word_text} ")

        if parts:
            evt = pysubs2.SSAEvent(
                start=int(words[0]["start"] * 1000),
                end=int(words[-1]["end"] * 1000),
                text="".join(parts).strip(),
                style=style,
            )
            subs.events.append(evt)

    subs.save(output_path)
    if on_progress:
        on_progress(100, "ASS karaoke exported")
    return output_path


def _parse_ass_color(color_str: str) -> Tuple:
    """Parse ASS color string (&HAABBGGRR) to (R,G,B,A) tuple."""
    try:
        color_str = color_str.replace("&H", "").replace("&h", "")
        val = int(color_str, 16)
        a = (val >> 24) & 0xFF
        b = (val >> 16) & 0xFF
        g = (val >> 8) & 0xFF
        r = val & 0xFF
        return (r, g, b, a)
    except Exception:
        return (255, 255, 255, 0)


# ---------------------------------------------------------------------------
# Subtitle Format Conversion
# ---------------------------------------------------------------------------
def convert_subtitle_format(
    input_path: str,
    output_format: str = "srt",
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Convert between subtitle formats using pysubs2.

    Supported formats: srt, ass, ssa, vtt, microdvd, json, txt
    """
    _ensure_package("pysubs2", "pysubs2", on_progress)
    import pysubs2

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        ext_map = {
            "srt": ".srt", "ass": ".ass", "ssa": ".ssa",
            "vtt": ".vtt", "microdvd": ".sub", "json": ".json", "txt": ".txt",
        }
        output_path = base + ext_map.get(output_format, f".{output_format}")

    subs = pysubs2.load(input_path)
    subs.save(output_path, format_=output_format)

    if on_progress:
        on_progress(100, f"Converted to {output_format.upper()}")
    return output_path
