"""
OpenCut Voice Lab - Local TTS, Voice Cloning & Voice Design

Powered by Qwen3-TTS (Alibaba Cloud / Qwen Team).
All processing runs locally - no cloud API, no per-character costs.

Models (downloaded on first use to %LOCALAPPDATA%/OpenCut/models/):
  - Qwen3-TTS-12Hz-0.6B-CustomVoice  (~1.2 GB)  - Preset speakers
  - Qwen3-TTS-12Hz-1.7B-CustomVoice  (~3.4 GB)  - Preset speakers (better)
  - Qwen3-TTS-12Hz-0.6B-Base         (~1.2 GB)  - Voice cloning
  - Qwen3-TTS-12Hz-1.7B-Base         (~3.4 GB)  - Voice cloning (better)
  - Qwen3-TTS-12Hz-1.7B-VoiceDesign  (~3.4 GB)  - Voice design from description
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _get_models_dir() -> Path:
    """Return the OpenCut models directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
    else:
        base = Path(os.path.expanduser("~"))
    d = base / "OpenCut" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_voices_dir() -> Path:
    """Return the voice profiles directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
    else:
        base = Path(os.path.expanduser("~"))
    d = base / "OpenCut" / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_cache_dir() -> Path:
    """Return the cache directory for temporary audio files."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
    else:
        base = Path(os.path.expanduser("~"))
    d = base / "OpenCut" / "cache" / "voice"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class VoiceProfile:
    """A saved voice profile for cloning."""
    id: str
    name: str
    language: str = "English"
    ref_audio_path: str = ""
    ref_text: str = ""
    description: str = ""
    created_at: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TTSSpeaker:
    """A built-in TTS speaker."""
    name: str
    language: str
    gender: str
    description: str
    model_variant: str = "CustomVoice"


@dataclass
class GenerationResult:
    """Result of a TTS generation."""
    output_path: str
    duration_seconds: float = 0.0
    sample_rate: int = 24000
    model_used: str = ""
    speaker: str = ""
    text: str = ""


# ---------------------------------------------------------------------------
# Built-in Speakers (Qwen3-TTS CustomVoice presets)
# ---------------------------------------------------------------------------
BUILTIN_SPEAKERS: List[TTSSpeaker] = [
    TTSSpeaker("Vivian", "English", "Female",
               "Warm, natural female voice with clear articulation"),
    TTSSpeaker("Ryan", "English", "Male",
               "Professional male voice, great for narration"),
    TTSSpeaker("Chelsie", "English", "Female",
               "Friendly, upbeat female voice with expressive delivery"),
    TTSSpeaker("Aidan", "English", "Male",
               "Deep, authoritative male voice for presentations"),
    TTSSpeaker("Serena", "Chinese", "Female",
               "Clear Mandarin female voice with neutral accent"),
    TTSSpeaker("Lucas", "Chinese", "Male",
               "Warm Mandarin male voice for conversational use"),
    TTSSpeaker("Yuki", "Japanese", "Female",
               "Natural Japanese female voice with gentle tone"),
    TTSSpeaker("Sophie", "French", "Female",
               "Elegant French female voice with Parisian accent"),
    TTSSpeaker("Elena", "Spanish", "Female",
               "Energetic Spanish female voice for dynamic content"),
]

SUPPORTED_LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
]

EMOTION_PRESETS = [
    {"name": "neutral", "label": "Neutral", "instruct": ""},
    {"name": "calm", "label": "Calm", "instruct": "Speak in a calm, soothing tone."},
    {"name": "cheerful", "label": "Cheerful", "instruct": "Speak in a cheerful, upbeat manner."},
    {"name": "dramatic", "label": "Dramatic", "instruct": "Speak with dramatic emphasis and emotion."},
    {"name": "serious", "label": "Serious", "instruct": "Speak in a serious, measured tone."},
    {"name": "whispering", "label": "Whispering", "instruct": "Speak in a soft whisper."},
    {"name": "excited", "label": "Excited", "instruct": "Speak with excitement and energy."},
    {"name": "sad", "label": "Sad", "instruct": "Speak with a melancholic, sad tone."},
]


# ---------------------------------------------------------------------------
# Model Management
# ---------------------------------------------------------------------------
# Lazy-loaded model cache (singleton per model variant)
_loaded_models: Dict[str, object] = {}


def _get_model_name(task: str, quality: str = "auto") -> str:
    """
    Return the HuggingFace model name for a given task.

    Tasks: 'tts', 'clone', 'design'
    Quality: 'auto', '0.6b', '1.7b'
    """
    if quality == "auto":
        quality = _auto_select_quality()

    size = "0.6B" if quality == "0.6b" else "1.7B"

    if task == "tts":
        return f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice"
    elif task == "clone":
        return f"Qwen/Qwen3-TTS-12Hz-{size}-Base"
    elif task == "design":
        # Voice design only available in 1.7B
        return "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    else:
        return f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice"


def _auto_select_quality() -> str:
    """Auto-select model quality based on available GPU VRAM."""
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            if vram_gb >= 8:
                return "1.7b"
    except ImportError:
        pass
    return "0.6b"


def check_qwen_tts_available() -> Tuple[bool, str]:
    """Check if qwen-tts package is installed."""
    try:
        import qwen_tts  # noqa: F401
        return True, "qwen-tts is installed"
    except ImportError:
        return False, "qwen-tts is not installed"


def install_qwen_tts(progress_callback=None) -> bool:
    """
    Install qwen-tts and its dependencies.
    This is a heavyweight install (torch + transformers + qwen-tts).
    """
    steps = [
        {
            "label": "Installing PyTorch",
            "cmd": [sys.executable, "-m", "pip", "install",
                    "torch", "torchaudio", "--quiet"],
        },
        {
            "label": "Installing qwen-tts",
            "cmd": [sys.executable, "-m", "pip", "install",
                    "-U", "qwen-tts", "--quiet"],
        },
        {
            "label": "Installing soundfile",
            "cmd": [sys.executable, "-m", "pip", "install",
                    "soundfile", "--quiet"],
        },
    ]

    for i, step in enumerate(steps):
        if progress_callback:
            pct = int((i / len(steps)) * 80) + 10
            progress_callback(pct, step["label"])
        logger.info(f"Voice Lab install: {step['label']}")

        try:
            result = subprocess.run(
                step["cmd"], capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                logger.error(f"Install failed: {result.stderr[-500:]}")
                return False
        except Exception as e:
            logger.error(f"Install error: {e}")
            return False

    return True


def _load_model(model_name: str, progress_callback=None):
    """
    Load a Qwen3-TTS model (lazy, cached after first load).
    Downloads model weights on first use.
    """
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    import torch
    from qwen_tts import Qwen3TTSModel

    if progress_callback:
        progress_callback(20, f"Loading model: {model_name.split('/')[-1]}...")

    logger.info(f"Loading Qwen3-TTS model: {model_name}")

    # Determine device and dtype
    device_map = "cpu"
    dtype = torch.float32
    attn_impl = "eager"

    if torch.cuda.is_available():
        device_map = "cuda:0"
        dtype = torch.bfloat16
        # Try flash attention
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"  # PyTorch native attention

    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    _loaded_models[model_name] = model
    logger.info(f"Model loaded: {model_name} (device={device_map}, attn={attn_impl})")

    if progress_callback:
        progress_callback(40, "Model loaded")

    return model


def unload_models():
    """Unload all cached models to free memory."""
    global _loaded_models
    for name in list(_loaded_models.keys()):
        del _loaded_models[name]
    _loaded_models = {}

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    logger.info("All voice models unloaded")


# ---------------------------------------------------------------------------
# Text-to-Speech (Preset Voices)
# ---------------------------------------------------------------------------
def generate_tts(
    text: str,
    speaker: str = "Vivian",
    language: str = "English",
    emotion: str = "neutral",
    speed: float = 1.0,
    quality: str = "auto",
    output_dir: str = "",
    progress_callback=None,
) -> GenerationResult:
    """
    Generate speech from text using a preset speaker voice.

    Args:
        text: The text to synthesize.
        speaker: Name of built-in speaker (Vivian, Ryan, Chelsie, etc.)
        language: Target language.
        emotion: Emotion preset name.
        speed: Playback speed multiplier (0.5 - 2.0).
        quality: Model quality ('auto', '0.6b', '1.7b').
        output_dir: Output directory for the generated audio.
        progress_callback: Optional (progress_pct, message) callback.

    Returns:
        GenerationResult with output path and metadata.
    """
    import soundfile as sf

    model_name = _get_model_name("tts", quality)
    model = _load_model(model_name, progress_callback)

    if progress_callback:
        progress_callback(50, f"Generating speech with {speaker}...")

    # Build kwargs
    kwargs = {
        "text": text,
        "speaker": speaker,
    }

    # Add language if supported
    if language:
        kwargs["language"] = language

    # Add emotion/instruction if not neutral
    emotion_instruct = ""
    for ep in EMOTION_PRESETS:
        if ep["name"] == emotion and ep["instruct"]:
            emotion_instruct = ep["instruct"]
            break
    if emotion_instruct:
        kwargs["instruct"] = emotion_instruct

    # Generate
    wavs, sr = model.generate_custom_voice(**kwargs)

    if progress_callback:
        progress_callback(80, "Post-processing audio...")

    # Apply speed adjustment if not 1.0
    audio_data = wavs[0]
    if abs(speed - 1.0) > 0.05:
        audio_data = _adjust_speed(audio_data, sr, speed)

    # Save output
    if not output_dir:
        output_dir = str(_get_cache_dir())
    os.makedirs(output_dir, exist_ok=True)

    timestamp = int(time.time())
    safe_speaker = speaker.replace(" ", "_").lower()
    output_path = os.path.join(output_dir, f"tts_{safe_speaker}_{timestamp}.wav")

    sf.write(output_path, audio_data, sr)

    duration = len(audio_data) / sr

    if progress_callback:
        progress_callback(100, "Speech generated!")

    logger.info(f"TTS generated: {output_path} ({duration:.1f}s, {speaker})")

    return GenerationResult(
        output_path=output_path,
        duration_seconds=round(duration, 2),
        sample_rate=sr,
        model_used=model_name.split("/")[-1],
        speaker=speaker,
        text=text,
    )


# ---------------------------------------------------------------------------
# Voice Cloning
# ---------------------------------------------------------------------------
def generate_voice_clone(
    text: str,
    ref_audio: str,
    ref_text: str = "",
    language: str = "English",
    speed: float = 1.0,
    quality: str = "auto",
    output_dir: str = "",
    progress_callback=None,
) -> GenerationResult:
    """
    Generate speech by cloning a voice from a reference audio sample.

    Args:
        text: The text to synthesize in the cloned voice.
        ref_audio: Path to reference audio file (3-15 seconds ideal).
        ref_text: Transcript of the reference audio (optional but improves quality).
        language: Target language.
        speed: Playback speed multiplier (0.5 - 2.0).
        quality: Model quality ('auto', '0.6b', '1.7b').
        output_dir: Output directory.
        progress_callback: Optional callback.

    Returns:
        GenerationResult with output path.
    """
    import soundfile as sf

    if not os.path.isfile(ref_audio):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    model_name = _get_model_name("clone", quality)
    model = _load_model(model_name, progress_callback)

    if progress_callback:
        progress_callback(50, "Cloning voice and generating speech...")

    # If no ref_text provided, try to transcribe it
    if not ref_text:
        ref_text = _transcribe_reference(ref_audio)

    kwargs = {
        "text": text,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
    }
    if language:
        kwargs["language"] = language

    wavs, sr = model.generate_voice_clone(**kwargs)

    if progress_callback:
        progress_callback(80, "Post-processing...")

    audio_data = wavs[0]
    if abs(speed - 1.0) > 0.05:
        audio_data = _adjust_speed(audio_data, sr, speed)

    if not output_dir:
        output_dir = str(_get_cache_dir())
    os.makedirs(output_dir, exist_ok=True)

    timestamp = int(time.time())
    output_path = os.path.join(output_dir, f"clone_{timestamp}.wav")
    sf.write(output_path, audio_data, sr)

    duration = len(audio_data) / sr

    if progress_callback:
        progress_callback(100, "Voice clone generated!")

    logger.info(f"Voice clone generated: {output_path} ({duration:.1f}s)")

    return GenerationResult(
        output_path=output_path,
        duration_seconds=round(duration, 2),
        sample_rate=sr,
        model_used=model_name.split("/")[-1],
        speaker="(cloned)",
        text=text,
    )


# ---------------------------------------------------------------------------
# Voice Design (from description)
# ---------------------------------------------------------------------------
def generate_voice_design(
    text: str,
    voice_description: str,
    language: str = "English",
    speed: float = 1.0,
    output_dir: str = "",
    progress_callback=None,
) -> GenerationResult:
    """
    Generate speech using a voice designed from a natural language description.

    Args:
        text: The text to synthesize.
        voice_description: Natural language description of the desired voice.
                          e.g. "A warm female voice with a slight British accent"
        language: Target language.
        speed: Playback speed.
        output_dir: Output directory.
        progress_callback: Optional callback.

    Returns:
        GenerationResult with output path.
    """
    import soundfile as sf

    # Voice design requires the 1.7B VoiceDesign model
    model_name = _get_model_name("design")
    model = _load_model(model_name, progress_callback)

    if progress_callback:
        progress_callback(50, "Designing voice and generating speech...")

    wavs, sr = model.generate_voice_design(
        text=text,
        instruct=voice_description,
    )

    if progress_callback:
        progress_callback(80, "Post-processing...")

    audio_data = wavs[0]
    if abs(speed - 1.0) > 0.05:
        audio_data = _adjust_speed(audio_data, sr, speed)

    if not output_dir:
        output_dir = str(_get_cache_dir())
    os.makedirs(output_dir, exist_ok=True)

    timestamp = int(time.time())
    output_path = os.path.join(output_dir, f"design_{timestamp}.wav")
    sf.write(output_path, audio_data, sr)

    duration = len(audio_data) / sr

    if progress_callback:
        progress_callback(100, "Voice design generated!")

    logger.info(f"Voice design generated: {output_path} ({duration:.1f}s)")

    return GenerationResult(
        output_path=output_path,
        duration_seconds=round(duration, 2),
        sample_rate=sr,
        model_used=model_name.split("/")[-1],
        speaker="(designed)",
        text=text,
    )


# ---------------------------------------------------------------------------
# Voice Profiles (save/load/delete)
# ---------------------------------------------------------------------------
def save_voice_profile(
    name: str,
    ref_audio_path: str,
    ref_text: str = "",
    language: str = "English",
    description: str = "",
) -> VoiceProfile:
    """
    Save a voice profile for reuse in voice cloning.

    Copies the reference audio to the voices directory and saves metadata.
    """
    voices_dir = _get_voices_dir()
    profile_id = str(uuid.uuid4())[:8]

    # Copy reference audio to voices dir
    ext = os.path.splitext(ref_audio_path)[1] or ".wav"
    stored_audio = str(voices_dir / f"{profile_id}_ref{ext}")
    shutil.copy2(ref_audio_path, stored_audio)

    # Get audio duration
    duration = _get_audio_duration(stored_audio)

    # If no ref_text, try to transcribe
    if not ref_text:
        ref_text = _transcribe_reference(stored_audio)

    profile = VoiceProfile(
        id=profile_id,
        name=name,
        language=language,
        ref_audio_path=stored_audio,
        ref_text=ref_text,
        description=description,
        created_at=time.time(),
        duration_seconds=round(duration, 1),
    )

    # Save metadata
    meta_path = str(voices_dir / f"{profile_id}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)

    logger.info(f"Voice profile saved: {name} ({profile_id}, {duration:.1f}s)")
    return profile


def list_voice_profiles() -> List[VoiceProfile]:
    """List all saved voice profiles."""
    voices_dir = _get_voices_dir()
    profiles = []

    for meta_file in sorted(voices_dir.glob("*_meta.json")):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            profiles.append(VoiceProfile(**data))
        except Exception as e:
            logger.warning(f"Could not load profile {meta_file}: {e}")

    return profiles


def get_voice_profile(profile_id: str) -> Optional[VoiceProfile]:
    """Get a specific voice profile by ID."""
    voices_dir = _get_voices_dir()
    meta_path = voices_dir / f"{profile_id}_meta.json"

    if not meta_path.exists():
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return VoiceProfile(**data)
    except Exception:
        return None


def delete_voice_profile(profile_id: str) -> bool:
    """Delete a voice profile and its reference audio."""
    voices_dir = _get_voices_dir()
    meta_path = voices_dir / f"{profile_id}_meta.json"

    if not meta_path.exists():
        return False

    try:
        # Load to get audio path
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Delete audio file
        audio_path = data.get("ref_audio_path", "")
        if audio_path and os.path.isfile(audio_path):
            os.remove(audio_path)

        # Delete metadata
        os.remove(str(meta_path))

        logger.info(f"Voice profile deleted: {profile_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting profile {profile_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Voice Replace (re-synthesize specific words in audio)
# ---------------------------------------------------------------------------
def voice_replace(
    original_audio: str,
    original_text: str,
    replacement_text: str,
    language: str = "English",
    quality: str = "auto",
    output_dir: str = "",
    progress_callback=None,
) -> GenerationResult:
    """
    Replace spoken words in audio by:
    1. Cloning the speaker's voice from the original audio
    2. Generating the replacement text in that voice
    3. Splicing the replacement into the original audio

    This is a simplified approach: it generates the full replacement text
    in the cloned voice. For precise word-level replacement, more
    sophisticated alignment would be needed.

    Args:
        original_audio: Path to original audio file.
        original_text: The original transcript (what was said).
        replacement_text: The new text to generate.
        language: Language of the text.
        quality: Model quality.
        output_dir: Output directory.
        progress_callback: Optional callback.

    Returns:
        GenerationResult with the replacement audio path.
    """
    if not os.path.isfile(original_audio):
        raise FileNotFoundError(f"Original audio not found: {original_audio}")

    if progress_callback:
        progress_callback(10, "Analyzing original voice...")

    # Use the original audio as reference for cloning
    result = generate_voice_clone(
        text=replacement_text,
        ref_audio=original_audio,
        ref_text=original_text,
        language=language,
        quality=quality,
        output_dir=output_dir,
        progress_callback=_shift_callback(progress_callback, 10, 90),
    )

    if progress_callback:
        progress_callback(100, "Voice replacement complete!")

    logger.info(f"Voice replace: '{original_text[:30]}...' -> '{replacement_text[:30]}...'")

    return result


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def _adjust_speed(audio_data, sample_rate: int, speed: float):
    """Adjust audio speed using FFmpeg atempo filter via temp files."""
    import tempfile
    import soundfile as sf

    # Clamp speed to valid range
    speed = max(0.5, min(2.0, speed))

    # Write original to temp
    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_in.close()
    tmp_out.close()

    try:
        sf.write(tmp_in.name, audio_data, sample_rate)

        # atempo filter only supports 0.5 to 2.0
        # Chain filters for extreme values
        filters = []
        remaining = speed
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.4f}")

        filter_chain = ",".join(filters)

        cmd = [
            "ffmpeg", "-y", "-i", tmp_in.name,
            "-filter:a", filter_chain,
            "-ar", str(sample_rate),
            tmp_out.name,
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)

        adjusted, _ = sf.read(tmp_out.name)
        return adjusted
    finally:
        try:
            os.unlink(tmp_in.name)
        except OSError:
            pass
        try:
            os.unlink(tmp_out.name)
        except OSError:
            pass


def _get_audio_duration(filepath: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                filepath,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _transcribe_reference(audio_path: str) -> str:
    """
    Attempt to transcribe reference audio using Whisper.
    Falls back to empty string if Whisper is not available.
    """
    try:
        try:
            from .captions import transcribe as _transcribe_fn
        except ImportError:
            from opencut.core.captions import transcribe as _transcribe_fn

        result = _transcribe_fn(audio_path, model_size="base")
        if result and hasattr(result, "segments"):
            text = " ".join(seg.text.strip() for seg in result.segments)
            return text.strip()
    except Exception as e:
        logger.debug(f"Could not auto-transcribe reference audio: {e}")

    return ""


def _shift_callback(callback, start_pct: int, end_pct: int):
    """Create a progress callback that maps 0-100 to start_pct-end_pct."""
    if not callback:
        return None

    def shifted(pct, msg):
        mapped = start_pct + int((pct / 100.0) * (end_pct - start_pct))
        callback(mapped, msg)

    return shifted


# ---------------------------------------------------------------------------
# Speaker listing helpers
# ---------------------------------------------------------------------------
def get_speakers() -> List[Dict]:
    """
    Return list of all available speakers:
    built-in presets + saved voice profiles.
    """
    speakers = []

    # Built-in speakers
    for sp in BUILTIN_SPEAKERS:
        speakers.append({
            "name": sp.name,
            "type": "builtin",
            "language": sp.language,
            "gender": sp.gender,
            "description": sp.description,
        })

    # Saved voice profiles
    for profile in list_voice_profiles():
        speakers.append({
            "name": profile.name,
            "type": "profile",
            "id": profile.id,
            "language": profile.language,
            "description": profile.description,
            "duration_seconds": profile.duration_seconds,
        })

    return speakers


def get_emotion_presets() -> List[Dict]:
    """Return the list of emotion presets."""
    return EMOTION_PRESETS


def get_supported_languages() -> List[str]:
    """Return supported languages."""
    return SUPPORTED_LANGUAGES
