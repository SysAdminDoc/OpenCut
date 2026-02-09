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
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


def _ensure_package(pkg, pip_name=None, on_progress=None):
    try:
        __import__(pkg)
        return True
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", pip_name or pkg,
                            "--break-system-packages", "-q"], capture_output=True, timeout=600)
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False


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
    if not _ensure_package("audiocraft", "audiocraft", on_progress):
        raise RuntimeError(
            "AudioCraft not installed. Run: pip install audiocraft\n"
            "Requires PyTorch with CUDA support."
        )

    if on_progress:
        on_progress(5, f"Loading MusicGen ({model_size})...")

    from audiocraft.models import MusicGen
    import torch
    import soundfile as sf

    model_name = f"facebook/musicgen-{model_size}"
    model = MusicGen.get_pretrained(model_name)

    model.set_generation_params(
        duration=min(duration, 30.0 if model_size == "small" else 60.0),
        temperature=temperature,
        top_k=top_k,
    )

    if output_path is None:
        directory = output_dir or tempfile.gettempdir()
        safe_prompt = prompt[:30].replace(" ", "_").replace("/", "")
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
    if not _ensure_package("audiocraft", "audiocraft", on_progress):
        raise RuntimeError("AudioCraft not installed")

    if not os.path.isfile(melody_path):
        raise FileNotFoundError(f"Melody file not found: {melody_path}")

    if on_progress:
        on_progress(5, "Loading MusicGen melody model...")

    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_read
    import torch
    import soundfile as sf

    model = MusicGen.get_pretrained("facebook/musicgen-melody")
    model.set_generation_params(duration=min(duration, 30.0))

    if output_path is None:
        directory = output_dir or tempfile.gettempdir()
        output_path = os.path.join(directory, f"musicgen_melody.wav")

    if on_progress:
        on_progress(15, "Loading melody reference...")

    melody_wav, sr = audio_read(melody_path)
    # Ensure 2D: [channels, samples]
    if melody_wav.ndim == 1:
        melody_wav = melody_wav.unsqueeze(0)

    if on_progress:
        on_progress(25, f"Generating with melody conditioning...")

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
    if not _ensure_package("audiocraft", "audiocraft", on_progress):
        raise RuntimeError("AudioCraft not installed")

    if on_progress:
        on_progress(5, "Loading MusicGen for continuation...")

    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_read
    import torch
    import soundfile as sf

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


def get_music_ai_capabilities() -> Dict:
    return {
        "audiocraft": check_audiocraft_available(),
        "cuda": check_torch_cuda(),
        "models": MUSICGEN_MODELS,
    }
