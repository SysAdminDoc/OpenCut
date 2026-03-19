"""
OpenCut AI Music Generation Module v0.9.0

AI-powered music and audio generation:
- AudioCraft/MusicGen: Text-to-music (Meta, 300M-3.3B params)
- Melody conditioning: Generate music matching a reference melody
- Continuation: Extend existing audio with AI-generated content

Heavy GPU module: MusicGen small=300M (1.2GB), medium=1.5B (6GB), large=3.3B (13GB).
Falls back to the SFX/tone generators in music_gen.py for CPU-only systems.
"""

import logging
import os
import tempfile
from typing import Callable, Dict, Optional

from opencut.helpers import ensure_package

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def check_audiocraft_available() -> bool:
    try:
        import audiocraft  # noqa: F401
        return True
    except ImportError:
        return False


def check_torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# MusicGen Text-to-Music
# ---------------------------------------------------------------------------
def generate_music(
    prompt: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    duration: float = 10.0,
    model_size: str = "small",
    temperature: float = 1.0,
    top_k: int = 250,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Generate music from a text prompt using MusicGen.

    Args:
        prompt: Text description of desired music (e.g. "upbeat electronic dance music
                with driving bassline and synth arpeggios").
        duration: Length in seconds (max ~30s for small, ~60s for medium).
        model_size: "small" (300M, 1.2GB), "medium" (1.5B, 6GB), "large" (3.3B, 13GB).
        temperature: Sampling temperature (higher = more creative).
        top_k: Top-k sampling parameter.
    """
    if not ensure_package("audiocraft", "audiocraft", on_progress):
        raise RuntimeError(
            "AudioCraft not installed. Run: pip install audiocraft\n"
            "Requires PyTorch with CUDA support."
        )

    if on_progress:
        on_progress(5, f"Loading MusicGen ({model_size})...")

    import soundfile as sf
    import torch
    from audiocraft.models import MusicGen

    model_name = f"facebook/musicgen-{model_size}"
    model = MusicGen.get_pretrained(model_name)

    model.set_generation_params(
        duration=min(duration, 30.0 if model_size == "small" else 60.0),
        temperature=temperature,
        top_k=top_k,
    )

    if output_path is None:
        directory = output_dir or tempfile.gettempdir()
        import re
        safe_prompt = re.sub(r'[^\w\-]', '_', prompt[:30]).strip('_')
        output_path = os.path.join(directory, f"musicgen_{safe_prompt}.wav")

    if on_progress:
        on_progress(20, f"Generating music: '{prompt[:50]}'...")

    # Generate
    with torch.inference_mode():
        wav = model.generate([prompt])

    # wav shape: [batch, channels, samples] at 32000Hz
    audio_data = wav[0].cpu().numpy()
    if audio_data.ndim == 2:
        audio_data = audio_data.T  # [samples, channels]
    elif audio_data.ndim == 1:
        pass  # mono

    if on_progress:
        on_progress(85, "Saving audio...")

    sf.write(output_path, audio_data, 32000)

    if on_progress:
        on_progress(100, "Music generated!")
    return output_path


# ---------------------------------------------------------------------------
# MusicGen with Melody Conditioning
# ---------------------------------------------------------------------------
def generate_music_with_melody(
    prompt: str,
    melody_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    duration: float = 10.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Generate music matching a reference melody.

    The melody audio is used as conditioning - MusicGen will generate
    new audio that follows the same melodic contour.
    """
    if not ensure_package("audiocraft", "audiocraft", on_progress):
        raise RuntimeError("AudioCraft not installed")

    if not os.path.isfile(melody_path):
        raise FileNotFoundError(f"Melody file not found: {melody_path}")

    if on_progress:
        on_progress(5, "Loading MusicGen melody model...")

    import soundfile as sf
    import torch
    from audiocraft.data.audio import audio_read
    from audiocraft.models import MusicGen

    model = MusicGen.get_pretrained("facebook/musicgen-melody")
    model.set_generation_params(duration=min(duration, 30.0))

    if output_path is None:
        directory = output_dir or tempfile.gettempdir()
        output_path = os.path.join(directory, "musicgen_melody.wav")

    if on_progress:
        on_progress(15, "Loading melody reference...")

    melody_wav, sr = audio_read(melody_path)
    # Ensure 2D: [channels, samples]
    if melody_wav.ndim == 1:
        melody_wav = melody_wav.unsqueeze(0)

    if on_progress:
        on_progress(25, "Generating with melody conditioning...")

    with torch.inference_mode():
        wav = model.generate_with_chroma([prompt], melody_wav.expand(1, -1, -1), sr)

    audio_data = wav[0].cpu().numpy()
    if audio_data.ndim == 2:
        audio_data = audio_data.T

    sf.write(output_path, audio_data, 32000)

    if on_progress:
        on_progress(100, "Melody-conditioned music generated!")
    return output_path


# ---------------------------------------------------------------------------
# Audio Continuation
# ---------------------------------------------------------------------------
def continue_audio(
    audio_path: str,
    prompt: str = "",
    output_path: Optional[str] = None,
    output_dir: str = "",
    duration: float = 10.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Continue/extend existing audio with AI-generated content.
    """
    if not ensure_package("audiocraft", "audiocraft", on_progress):
        raise RuntimeError("AudioCraft not installed")

    if on_progress:
        on_progress(5, "Loading MusicGen for continuation...")

    import soundfile as sf
    import torch
    from audiocraft.data.audio import audio_read
    from audiocraft.models import MusicGen

    model = MusicGen.get_pretrained("facebook/musicgen-small")
    model.set_generation_params(duration=min(duration, 30.0))

    if output_path is None:
        directory = output_dir or tempfile.gettempdir()
        output_path = os.path.join(directory, "musicgen_continued.wav")

    audio_wav, sr = audio_read(audio_path)
    if audio_wav.ndim == 1:
        audio_wav = audio_wav.unsqueeze(0)

    if on_progress:
        on_progress(20, "Generating continuation...")

    with torch.inference_mode():
        if prompt:
            wav = model.generate_continuation(audio_wav.expand(1, -1, -1), sr,
                                               descriptions=[prompt])
        else:
            wav = model.generate_continuation(audio_wav.expand(1, -1, -1), sr)

    audio_data = wav[0].cpu().numpy()
    if audio_data.ndim == 2:
        audio_data = audio_data.T

    sf.write(output_path, audio_data, 32000)

    if on_progress:
        on_progress(100, "Audio continued!")
    return output_path


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------
MUSICGEN_MODELS = [
    {"name": "small", "label": "Small (300M)", "size": "1.2GB", "max_duration": 30},
    {"name": "medium", "label": "Medium (1.5B)", "size": "6GB", "max_duration": 60},
    {"name": "large", "label": "Large (3.3B)", "size": "13GB", "max_duration": 60},
    {"name": "melody", "label": "Melody (1.5B)", "size": "6GB", "max_duration": 30},
]


def check_ace_step_available() -> bool:
    try:
        import ace_step  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# ACE-Step Music Generation (full songs with vocals + lyrics)
# ---------------------------------------------------------------------------
def generate_music_ace_step(
    prompt: str,
    lyrics: str = "",
    output_path: Optional[str] = None,
    output_dir: str = "",
    duration: float = 30.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Generate music (with optional vocals + lyrics) using ACE-Step 1.5.

    Superior to MusicGen: full songs with vocals+lyrics, 10x faster,
    4x less VRAM (<4GB), 1000+ styles, 19 languages. Apache 2.0.

    Args:
        prompt: Style/genre description (e.g. "upbeat pop, female vocalist, catchy melody").
        lyrics: Optional lyrics for vocal generation. Empty = instrumental.
        duration: Song length in seconds (10-600).
    """
    if not ensure_package("ace_step", "ace-step", on_progress):
        raise RuntimeError("ACE-Step not installed. Run: pip install ace-step")

    if on_progress:
        on_progress(5, "Loading ACE-Step model...")

    import torch
    from ace_step import ACEStep

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ACEStep.from_pretrained(device=device)

    if output_path is None:
        directory = output_dir or tempfile.gettempdir()
        import re
        safe_prompt = re.sub(r'[^\w\-]', '_', prompt[:30]).strip('_')
        output_path = os.path.join(directory, f"ace_step_{safe_prompt}.wav")

    duration = max(10.0, min(600.0, duration))

    if on_progress:
        on_progress(20, f"Generating music: '{prompt[:50]}'...")

    with torch.inference_mode():
        result = model.generate(
            prompt=prompt,
            lyrics=lyrics or None,
            duration=duration,
        )

    if on_progress:
        on_progress(80, "Saving audio...")

    # Save output
    import soundfile as sf
    audio_data = result["audio"]
    if hasattr(audio_data, "cpu"):
        audio_data = audio_data.cpu().numpy()
    if audio_data.ndim == 2:
        audio_data = audio_data.T
    sf.write(output_path, audio_data, result.get("sample_rate", 44100))

    # Free GPU
    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if on_progress:
        on_progress(100, "Music generated with ACE-Step!")
    return output_path


def get_music_ai_capabilities() -> Dict:
    return {
        "audiocraft": check_audiocraft_available(),
        "ace_step": check_ace_step_available(),
        "cuda": check_torch_cuda(),
        "models": MUSICGEN_MODELS,
    }
