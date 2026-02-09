"""
OpenCut Voice Generation Module v0.8.0

Text-to-speech for voiceovers, narration, and dubbing:
- edge-tts: Microsoft Azure TTS (free, zero API key, 400+ voices, 75+ languages)
- Kokoro-82M: Offline local TTS (350MB model, high quality, English/multilingual)

edge-tts runs online via Microsoft's free endpoint - no auth needed.
Kokoro runs fully offline on CPU at near-realtime speed.
"""

import asyncio
import logging
import os
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Optional

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
# Availability
# ---------------------------------------------------------------------------
def check_edge_tts_available() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


def check_kokoro_available() -> bool:
    try:
        import kokoro  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Edge-TTS Voices (curated popular voices)
# ---------------------------------------------------------------------------
EDGE_VOICES = [
    # English
    {"id": "en-US-AriaNeural", "label": "Aria (US Female)", "lang": "en", "gender": "female"},
    {"id": "en-US-GuyNeural", "label": "Guy (US Male)", "lang": "en", "gender": "male"},
    {"id": "en-US-JennyNeural", "label": "Jenny (US Female)", "lang": "en", "gender": "female"},
    {"id": "en-US-ChristopherNeural", "label": "Christopher (US Male)", "lang": "en", "gender": "male"},
    {"id": "en-US-EricNeural", "label": "Eric (US Male)", "lang": "en", "gender": "male"},
    {"id": "en-US-MichelleNeural", "label": "Michelle (US Female)", "lang": "en", "gender": "female"},
    {"id": "en-US-RogerNeural", "label": "Roger (US Male, Narration)", "lang": "en", "gender": "male"},
    {"id": "en-US-SteffanNeural", "label": "Steffan (US Male)", "lang": "en", "gender": "male"},
    {"id": "en-GB-SoniaNeural", "label": "Sonia (UK Female)", "lang": "en", "gender": "female"},
    {"id": "en-GB-RyanNeural", "label": "Ryan (UK Male)", "lang": "en", "gender": "male"},
    {"id": "en-AU-NatashaNeural", "label": "Natasha (AU Female)", "lang": "en", "gender": "female"},
    {"id": "en-AU-WilliamNeural", "label": "William (AU Male)", "lang": "en", "gender": "male"},
    # Spanish
    {"id": "es-ES-ElviraNeural", "label": "Elvira (Spain Female)", "lang": "es", "gender": "female"},
    {"id": "es-MX-DaliaNeural", "label": "Dalia (Mexico Female)", "lang": "es", "gender": "female"},
    {"id": "es-MX-JorgeNeural", "label": "Jorge (Mexico Male)", "lang": "es", "gender": "male"},
    # French
    {"id": "fr-FR-DeniseNeural", "label": "Denise (France Female)", "lang": "fr", "gender": "female"},
    {"id": "fr-FR-HenriNeural", "label": "Henri (France Male)", "lang": "fr", "gender": "male"},
    # German
    {"id": "de-DE-KatjaNeural", "label": "Katja (Germany Female)", "lang": "de", "gender": "female"},
    {"id": "de-DE-ConradNeural", "label": "Conrad (Germany Male)", "lang": "de", "gender": "male"},
    # Japanese
    {"id": "ja-JP-NanamiNeural", "label": "Nanami (Japan Female)", "lang": "ja", "gender": "female"},
    {"id": "ja-JP-KeitaNeural", "label": "Keita (Japan Male)", "lang": "ja", "gender": "male"},
    # Chinese
    {"id": "zh-CN-XiaoxiaoNeural", "label": "Xiaoxiao (China Female)", "lang": "zh", "gender": "female"},
    {"id": "zh-CN-YunxiNeural", "label": "Yunxi (China Male)", "lang": "zh", "gender": "male"},
    # Korean
    {"id": "ko-KR-SunHiNeural", "label": "Sun-Hi (Korea Female)", "lang": "ko", "gender": "female"},
    {"id": "ko-KR-InJoonNeural", "label": "InJoon (Korea Male)", "lang": "ko", "gender": "male"},
    # Portuguese
    {"id": "pt-BR-FranciscaNeural", "label": "Francisca (Brazil Female)", "lang": "pt", "gender": "female"},
    {"id": "pt-BR-AntonioNeural", "label": "Antonio (Brazil Male)", "lang": "pt", "gender": "male"},
    # Italian
    {"id": "it-IT-ElsaNeural", "label": "Elsa (Italy Female)", "lang": "it", "gender": "female"},
    {"id": "it-IT-DiegoNeural", "label": "Diego (Italy Male)", "lang": "it", "gender": "male"},
    # Hindi
    {"id": "hi-IN-SwaraNeural", "label": "Swara (India Female)", "lang": "hi", "gender": "female"},
    {"id": "hi-IN-MadhurNeural", "label": "Madhur (India Male)", "lang": "hi", "gender": "male"},
    # Arabic
    {"id": "ar-SA-ZariyahNeural", "label": "Zariyah (Saudi Female)", "lang": "ar", "gender": "female"},
    {"id": "ar-SA-HamedNeural", "label": "Hamed (Saudi Male)", "lang": "ar", "gender": "male"},
    # Russian
    {"id": "ru-RU-SvetlanaNeural", "label": "Svetlana (Russia Female)", "lang": "ru", "gender": "female"},
    {"id": "ru-RU-DmitryNeural", "label": "Dmitry (Russia Male)", "lang": "ru", "gender": "male"},
    # Turkish
    {"id": "tr-TR-EmelNeural", "label": "Emel (Turkey Female)", "lang": "tr", "gender": "female"},
    # Dutch
    {"id": "nl-NL-ColetteNeural", "label": "Colette (Netherlands Female)", "lang": "nl", "gender": "female"},
    # Polish
    {"id": "pl-PL-AgnieszkaNeural", "label": "Agnieszka (Poland Female)", "lang": "pl", "gender": "female"},
    # Swedish
    {"id": "sv-SE-SofieNeural", "label": "Sofie (Sweden Female)", "lang": "sv", "gender": "female"},
    # Thai
    {"id": "th-TH-PremwadeeNeural", "label": "Premwadee (Thai Female)", "lang": "th", "gender": "female"},
    # Vietnamese
    {"id": "vi-VN-HoaiMyNeural", "label": "HoaiMy (Vietnam Female)", "lang": "vi", "gender": "female"},
]


def get_voice_list(engine: str = "edge") -> List[Dict]:
    """Return available voices for a TTS engine."""
    if engine == "edge":
        return EDGE_VOICES
    return []


# ---------------------------------------------------------------------------
# Edge-TTS (Online, Zero Auth)
# ---------------------------------------------------------------------------
def edge_tts_generate(
    text: str,
    voice: str = "en-US-AriaNeural",
    output_path: Optional[str] = None,
    output_dir: str = "",
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Generate speech audio using Microsoft Edge TTS.

    Free, no API key needed, 400+ voices, 75+ languages.
    Requires internet connection.

    Args:
        text: Text to speak.
        voice: Voice ID from EDGE_VOICES.
        rate: Speed adjustment (e.g. "+20%", "-10%").
        pitch: Pitch adjustment (e.g. "+5Hz", "-3Hz").
        volume: Volume adjustment (e.g. "+10%", "-5%").
    """
    _ensure_package("edge_tts", "edge-tts", on_progress)
    import edge_tts

    if output_path is None:
        safe_voice = voice.replace("-", "_")[:20]
        directory = output_dir or tempfile.gettempdir()
        output_path = os.path.join(directory, f"tts_{safe_voice}.mp3")

    if on_progress:
        on_progress(10, f"Generating speech ({voice})...")

    async def _generate():
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
        await communicate.save(output_path)

    # Run async in sync context
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _generate())
                future.result(timeout=120)
        else:
            loop.run_until_complete(_generate())
    except RuntimeError:
        asyncio.run(_generate())

    if on_progress:
        on_progress(100, "Speech generated!")

    return output_path


# ---------------------------------------------------------------------------
# Edge-TTS with subtitle generation (word-level timing)
# ---------------------------------------------------------------------------
def edge_tts_with_subtitles(
    text: str,
    voice: str = "en-US-AriaNeural",
    output_dir: str = "",
    rate: str = "+0%",
    pitch: str = "+0Hz",
    on_progress: Optional[Callable] = None,
) -> Dict:
    """
    Generate speech + word-level subtitle file (VTT).

    Returns dict with audio_path and subtitle_path.
    """
    _ensure_package("edge_tts", "edge-tts", on_progress)
    import edge_tts

    directory = output_dir or tempfile.gettempdir()
    safe_voice = voice.replace("-", "_")[:20]
    audio_path = os.path.join(directory, f"tts_{safe_voice}.mp3")
    sub_path = os.path.join(directory, f"tts_{safe_voice}.vtt")

    if on_progress:
        on_progress(10, f"Generating speech + subtitles ({voice})...")

    async def _generate():
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        subs = edge_tts.SubMaker()
        with open(audio_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    subs.create_sub(
                        (chunk["offset"], chunk["duration"]),
                        chunk["text"],
                    )
        with open(sub_path, "w", encoding="utf-8") as f:
            f.write(subs.generate_subs())

    try:
        asyncio.run(_generate())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_generate())
        loop.close()

    if on_progress:
        on_progress(100, "Speech + subtitles generated!")

    return {
        "audio_path": audio_path,
        "subtitle_path": sub_path,
    }


# ---------------------------------------------------------------------------
# Kokoro Offline TTS (350MB model)
# ---------------------------------------------------------------------------
def kokoro_generate(
    text: str,
    voice: str = "af_heart",
    output_path: Optional[str] = None,
    output_dir: str = "",
    speed: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Generate speech using Kokoro-82M offline TTS.

    Runs locally on CPU, 350MB model, near-realtime speed.
    High quality English voices. Model auto-downloads on first use.

    Args:
        voice: Kokoro voice preset (af_heart, af_bella, am_adam, am_michael, etc.)
        speed: Playback speed multiplier.
    """
    _ensure_package("kokoro", "kokoro>=0.3", on_progress)
    import kokoro

    if output_path is None:
        directory = output_dir or tempfile.gettempdir()
        output_path = os.path.join(directory, f"kokoro_{voice}.wav")

    if on_progress:
        on_progress(10, f"Loading Kokoro model ({voice})...")

    pipeline = kokoro.KPipeline(lang_code="a")

    if on_progress:
        on_progress(40, "Synthesizing speech...")

    # Generate audio
    generator = pipeline(text, voice=voice, speed=speed)
    samples = None
    for i, (gs, ps, audio) in enumerate(generator):
        if samples is None:
            samples = audio
        else:
            import numpy as np
            samples = np.concatenate([samples, audio])

    if samples is None:
        raise RuntimeError("Kokoro produced no audio output")

    if on_progress:
        on_progress(80, "Saving audio...")

    # Save as WAV
    import soundfile as sf
    sf.write(output_path, samples, 24000)

    if on_progress:
        on_progress(100, "Kokoro speech generated!")

    return output_path


# ---------------------------------------------------------------------------
# Voice Cloning Stub (future: OpenVoice / F5-TTS)
# ---------------------------------------------------------------------------
KOKORO_VOICES = [
    {"id": "af_heart", "label": "Heart (Female, Warm)", "lang": "en", "gender": "female"},
    {"id": "af_bella", "label": "Bella (Female, Clear)", "lang": "en", "gender": "female"},
    {"id": "af_sarah", "label": "Sarah (Female, Soft)", "lang": "en", "gender": "female"},
    {"id": "af_nicole", "label": "Nicole (Female, Professional)", "lang": "en", "gender": "female"},
    {"id": "am_adam", "label": "Adam (Male, Deep)", "lang": "en", "gender": "male"},
    {"id": "am_michael", "label": "Michael (Male, Warm)", "lang": "en", "gender": "male"},
    {"id": "bf_emma", "label": "Emma (British Female)", "lang": "en", "gender": "female"},
    {"id": "bm_george", "label": "George (British Male)", "lang": "en", "gender": "male"},
]
