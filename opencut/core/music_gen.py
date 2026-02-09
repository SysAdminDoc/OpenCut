"""
OpenCut Music & SFX Generation Module v0.8.0

Audio generation for video production:
- Tone generator: sine, square, sawtooth, white noise via FFmpeg
- SFX synthesizer: swoosh, impact, rise, drop, click, beep via FFmpeg
- Silence/padding generator
- AudioCraft/MusicGen hook for AI music generation (1.2GB+ models)

Tone and SFX generation are FFmpeg-only, zero dependencies.
AudioCraft requires GPU and large model downloads.
"""

import logging
import os
import subprocess
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("opencut")


def _run_ffmpeg(cmd: List[str], timeout: int = 120) -> str:
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode(errors='replace')[-500:]}")
    return result.stderr.decode(errors="replace")


# ---------------------------------------------------------------------------
# Tone Generator (FFmpeg, zero deps)
# ---------------------------------------------------------------------------
WAVEFORMS = ["sine", "square", "sawtooth", "triangle"]

def generate_tone(
    output_path: Optional[str] = None,
    output_dir: str = "",
    frequency: float = 440.0,
    duration: float = 3.0,
    waveform: str = "sine",
    volume: float = 0.5,
    fade_in: float = 0.1,
    fade_out: float = 0.5,
    sample_rate: int = 48000,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Generate a pure tone using FFmpeg's aevalsrc.

    Args:
        frequency: Tone frequency in Hz (20-20000).
        duration: Length in seconds.
        waveform: sine, square, sawtooth, triangle.
        volume: 0.0-1.0.
        fade_in/fade_out: Fade duration in seconds.
    """
    if output_path is None:
        directory = output_dir or os.path.join(os.path.expanduser("~"), "opencut_audio")
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, f"tone_{waveform}_{int(frequency)}hz.wav")

    if on_progress:
        on_progress(10, f"Generating {waveform} tone at {frequency}Hz...")

    # Build lavfi expression
    if waveform == "sine":
        expr = f"sin(2*PI*{frequency}*t)*{volume}"
    elif waveform == "square":
        expr = f"(2*(sin(2*PI*{frequency}*t)>0)-1)*{volume}"
    elif waveform == "sawtooth":
        expr = f"(2*({frequency}*t-floor({frequency}*t))-1)*{volume}"
    elif waveform == "triangle":
        expr = f"(4*abs({frequency}*t-floor({frequency}*t+0.5))-1)*{volume}"
    else:
        expr = f"sin(2*PI*{frequency}*t)*{volume}"

    af_parts = []
    if fade_in > 0:
        af_parts.append(f"afade=t=in:st=0:d={fade_in}")
    if fade_out > 0:
        af_parts.append(f"afade=t=out:st={max(0, duration - fade_out)}:d={fade_out}")

    af_filter = "," + ",".join(af_parts) if af_parts else ""

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i",
        f"aevalsrc={expr}:s={sample_rate}:d={duration}",
        "-af", f"aresample={sample_rate}{af_filter}",
        "-c:a", "pcm_s16le",
        output_path,
    ]
    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Tone generated!")
    return output_path


# ---------------------------------------------------------------------------
# SFX Synthesizer (FFmpeg, zero deps)
# ---------------------------------------------------------------------------
SFX_PRESETS = {
    "swoosh": {
        "label": "Swoosh",
        "description": "Quick whoosh/swoosh transition sound",
    },
    "impact": {
        "label": "Impact",
        "description": "Heavy bass impact hit",
    },
    "rise": {
        "label": "Rise / Riser",
        "description": "Tension-building ascending tone",
    },
    "drop": {
        "label": "Drop",
        "description": "Bass drop / descending hit",
    },
    "click": {
        "label": "Click",
        "description": "Clean UI click sound",
    },
    "beep": {
        "label": "Beep",
        "description": "Alert/notification beep",
    },
    "white_noise": {
        "label": "White Noise",
        "description": "Ambient white noise / static",
    },
    "countdown_tick": {
        "label": "Countdown Tick",
        "description": "Clock tick / countdown sound",
    },
}


def generate_sfx(
    preset: str = "swoosh",
    output_path: Optional[str] = None,
    output_dir: str = "",
    duration: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Generate a synthesized sound effect using FFmpeg lavfi.
    """
    if output_path is None:
        directory = output_dir or os.path.join(os.path.expanduser("~"), "opencut_audio")
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, f"sfx_{preset}.wav")

    if on_progress:
        on_progress(10, f"Generating SFX: {preset}...")

    sr = 48000

    if preset == "swoosh":
        # Frequency sweep 2000->200 Hz with fade
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            f"aevalsrc=sin(2*PI*(2000-1800*t/{duration})*t)*0.4*(1-t/{duration}):s={sr}:d={duration}",
            "-af", f"afade=t=in:d=0.02,afade=t=out:st={max(0.05, duration-0.15)}:d=0.15",
            output_path,
        ]
    elif preset == "impact":
        # Low frequency burst with fast decay
        d = min(duration, 1.5)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            f"aevalsrc=sin(2*PI*60*t)*0.8*exp(-5*t)+sin(2*PI*120*t)*0.4*exp(-8*t):s={sr}:d={d}",
            "-af", "afade=t=in:d=0.005,afade=t=out:st=0.3:d=0.3,alimiter=limit=0.9",
            output_path,
        ]
    elif preset == "rise":
        # Ascending frequency sweep
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            f"aevalsrc=sin(2*PI*(200+3000*t*t/{duration}/{duration})*t)*0.4*(t/{duration}):s={sr}:d={duration}",
            "-af", f"afade=t=in:d=0.1,afade=t=out:st={max(0.1, duration-0.05)}:d=0.05",
            output_path,
        ]
    elif preset == "drop":
        # Descending sweep with bass
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            f"aevalsrc=sin(2*PI*(1000*exp(-3*t/{duration}))*t)*0.6*exp(-2*t/{duration}):s={sr}:d={duration}",
            "-af", "afade=t=in:d=0.005,alimiter=limit=0.9",
            output_path,
        ]
    elif preset == "click":
        d = 0.08
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            f"aevalsrc=sin(2*PI*3500*t)*0.7*exp(-80*t):s={sr}:d={d}",
            output_path,
        ]
    elif preset == "beep":
        d = min(duration, 0.5)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            f"aevalsrc=sin(2*PI*1000*t)*0.5:s={sr}:d={d}",
            "-af", f"afade=t=in:d=0.01,afade=t=out:st={max(0.01, d-0.05)}:d=0.05",
            output_path,
        ]
    elif preset == "white_noise":
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            f"anoisesrc=d={duration}:c=white:r={sr}:a=0.3",
            "-af", f"afade=t=in:d=0.2,afade=t=out:st={max(0.2, duration-0.5)}:d=0.5",
            output_path,
        ]
    elif preset == "countdown_tick":
        d = 0.1
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            f"aevalsrc=(sin(2*PI*2000*t)+sin(2*PI*4000*t))*0.3*exp(-40*t):s={sr}:d={d}",
            output_path,
        ]
    else:
        raise ValueError(f"Unknown SFX preset: {preset}")

    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, f"SFX '{preset}' generated!")
    return output_path


# ---------------------------------------------------------------------------
# Silence / Padding Generator
# ---------------------------------------------------------------------------
def generate_silence(
    duration: float = 1.0,
    output_path: Optional[str] = None,
    output_dir: str = "",
    sample_rate: int = 48000,
    on_progress: Optional[Callable] = None,
) -> str:
    """Generate a silent audio file."""
    if output_path is None:
        directory = output_dir or os.path.join(os.path.expanduser("~"), "opencut_audio")
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, f"silence_{duration}s.wav")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i",
        f"anullsrc=r={sample_rate}:cl=stereo:d={duration}",
        "-c:a", "pcm_s16le",
        output_path,
    ]
    _run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Silence generated!")
    return output_path


# ---------------------------------------------------------------------------
# Concatenate audio files
# ---------------------------------------------------------------------------
def concatenate_audio(
    input_paths: List[str],
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """Concatenate multiple audio files into one."""
    import tempfile

    if not input_paths:
        raise ValueError("No input files provided")

    if output_path is None:
        directory = output_dir or os.path.dirname(input_paths[0])
        output_path = os.path.join(directory, "concatenated_audio.wav")

    if on_progress:
        on_progress(10, f"Concatenating {len(input_paths)} audio files...")

    # Create concat list file
    list_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    try:
        for p in input_paths:
            list_file.write(f"file '{p}'\n")
        list_file.close()

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", list_file.name,
            "-c:a", "pcm_s16le",
            output_path,
        ]
        _run_ffmpeg(cmd)
    finally:
        os.unlink(list_file.name)

    if on_progress:
        on_progress(100, "Audio concatenated!")
    return output_path


# ---------------------------------------------------------------------------
# Available generators
# ---------------------------------------------------------------------------
def get_audio_generators() -> Dict:
    """Return available audio generation capabilities."""
    return {
        "tts": {
            "edge_tts": check_edge_tts_available(),
            "kokoro": check_kokoro_available(),
        },
        "sfx_presets": [
            {"name": k, "label": v["label"], "description": v["description"]}
            for k, v in SFX_PRESETS.items()
        ],
        "tones": WAVEFORMS,
    }
