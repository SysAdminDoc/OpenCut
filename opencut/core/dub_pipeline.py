"""
OpenCut Dubbing Pipeline v1.30.0

Full local dubbing: STT → translate → TTS → composite → optional lipsync.
Delegates to opencut.core.auto_dub_pipeline which provides the complete
multi-stage pipeline with voice cloning, music preservation, and fallbacks.
"""
from __future__ import annotations

from typing import Callable, Optional

from opencut.core.auto_dub_pipeline import (
    LANGUAGE_NAMES,
    SUPPORTED_LANGUAGES,
    DubConfig,
    DubResult,
    auto_dub,
)

INSTALL_HINT = "pip install faster-whisper ctranslate2 edge-tts  # STT + translate + TTS"


def check_dub_pipeline_available() -> bool:
    """Returns True — pipeline uses edge-tts which is always available; Whisper is optional."""
    return True


def dub(
    video_path: str,
    target_language: str,
    voice: Optional[str] = None,
    output: Optional[str] = None,
    whisper_model: str = "base",
    voice_clone: bool = True,
    lip_sync: bool = False,
    preserve_music: bool = True,
    tts_engine: str = "edge",
    on_progress: Optional[Callable] = None,
) -> DubResult:
    """
    Dub a video into a target language.

    Args:
        video_path: Path to the source video.
        target_language: ISO 639-1 language code (e.g. "es", "fr", "de").
        voice: Unused — voice is chosen automatically per language via edge-tts.
        output: Output path override. Auto-generated if None.
        whisper_model: Whisper model size for transcription (tiny/base/small/medium/large).
        voice_clone: Attempt to clone the original speaker's voice characteristics.
        lip_sync: Apply lip sync to the dubbed video (requires Wav2Lip/LatentSync).
        preserve_music: Mix background music at reduced volume under dubbed audio.
        tts_engine: TTS backend — "edge" (default), "kokoro", or "chatterbox".
        on_progress: Optional callback(pct: int, msg: str).

    Returns:
        DubResult with output_path, segment details, and pipeline status.
    """
    config = DubConfig(
        target_language=target_language,
        whisper_model=whisper_model,
        voice_clone=voice_clone,
        lip_sync=lip_sync,
        preserve_music=preserve_music,
        tts_engine=tts_engine,
        output_dir=output or "",
    )
    return auto_dub(video_path, target_language=target_language,
                    config=config, on_progress=on_progress)


__all__ = [
    "check_dub_pipeline_available",
    "INSTALL_HINT",
    "SUPPORTED_LANGUAGES",
    "LANGUAGE_NAMES",
    "DubResult",
    "dub",
]
