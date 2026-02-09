"""
OpenCut Audio Pro Module v0.7.0

Professional audio processing:
- Pedalboard studio effects (reverb, EQ, compressor, limiter, chorus, delay, etc.)
- DeepFilterNet AI noise reduction (real-time 48kHz speech enhancement)
- VST3 plugin hosting via Pedalboard

Pedalboard (by Spotify) is 300x faster than pySoX and provides JUCE-based
studio-quality audio processing. DeepFilterNet provides broadcast-grade
noise suppression with <10MB model weight.
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------
def _ensure_package(pkg_name: str, pip_name: str = None, on_progress: Callable = None):
    """Import a package, installing it if missing."""
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


def _run_ffmpeg(cmd: List[str], timeout: int = 1800) -> str:
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode(errors='replace')[-500:]}")
    return result.stderr.decode(errors="replace")


# ---------------------------------------------------------------------------
# Pedalboard availability
# ---------------------------------------------------------------------------
def check_pedalboard_available() -> bool:
    try:
        import pedalboard  # noqa: F401
        return True
    except ImportError:
        return False


def check_deepfilter_available() -> bool:
    try:
        import df  # noqa: F401
        return True
    except ImportError:
        # Also check CLI
        try:
            result = subprocess.run(
                ["deep-filter", "--version"],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Pedalboard Effects Definitions
# ---------------------------------------------------------------------------
PEDALBOARD_EFFECTS = {
    # === Dynamics ===
    "compressor": {
        "label": "Compressor",
        "description": "Dynamic range compression",
        "category": "dynamics",
        "params": {
            "threshold_db": {"label": "Threshold (dB)", "min": -60, "max": 0, "default": -20, "step": 1},
            "ratio": {"label": "Ratio", "min": 1, "max": 20, "default": 4, "step": 0.5},
            "attack_ms": {"label": "Attack (ms)", "min": 0.1, "max": 100, "default": 1, "step": 0.1},
            "release_ms": {"label": "Release (ms)", "min": 10, "max": 1000, "default": 100, "step": 10},
        },
    },
    "limiter": {
        "label": "Limiter",
        "description": "Brick-wall limiter for broadcast loudness",
        "category": "dynamics",
        "params": {
            "threshold_db": {"label": "Threshold (dB)", "min": -30, "max": 0, "default": -1, "step": 0.5},
            "release_ms": {"label": "Release (ms)", "min": 10, "max": 500, "default": 100, "step": 10},
        },
    },
    "noise_gate": {
        "label": "Noise Gate",
        "description": "Gate to remove quiet noise between speech",
        "category": "dynamics",
        "params": {
            "threshold_db": {"label": "Threshold (dB)", "min": -80, "max": -10, "default": -40, "step": 1},
            "attack_ms": {"label": "Attack (ms)", "min": 0.1, "max": 50, "default": 1, "step": 0.1},
            "release_ms": {"label": "Release (ms)", "min": 10, "max": 500, "default": 100, "step": 10},
        },
    },
    # === EQ ===
    "low_shelf": {
        "label": "Low Shelf EQ",
        "description": "Boost or cut low frequencies",
        "category": "eq",
        "params": {
            "cutoff_hz": {"label": "Frequency (Hz)", "min": 20, "max": 500, "default": 200, "step": 10},
            "gain_db": {"label": "Gain (dB)", "min": -20, "max": 20, "default": 3, "step": 0.5},
        },
    },
    "high_shelf": {
        "label": "High Shelf EQ",
        "description": "Boost or cut high frequencies",
        "category": "eq",
        "params": {
            "cutoff_hz": {"label": "Frequency (Hz)", "min": 2000, "max": 16000, "default": 8000, "step": 500},
            "gain_db": {"label": "Gain (dB)", "min": -20, "max": 20, "default": 3, "step": 0.5},
        },
    },
    "peak_eq": {
        "label": "Parametric EQ",
        "description": "Precise frequency band adjustment",
        "category": "eq",
        "params": {
            "cutoff_hz": {"label": "Frequency (Hz)", "min": 20, "max": 16000, "default": 1000, "step": 10},
            "gain_db": {"label": "Gain (dB)", "min": -20, "max": 20, "default": 3, "step": 0.5},
            "q": {"label": "Q (Bandwidth)", "min": 0.1, "max": 10, "default": 1.0, "step": 0.1},
        },
    },
    "highpass": {
        "label": "High-Pass Filter",
        "description": "Remove low frequencies (rumble removal)",
        "category": "eq",
        "params": {
            "cutoff_hz": {"label": "Cutoff (Hz)", "min": 20, "max": 500, "default": 80, "step": 5},
        },
    },
    "lowpass": {
        "label": "Low-Pass Filter",
        "description": "Remove high frequencies",
        "category": "eq",
        "params": {
            "cutoff_hz": {"label": "Cutoff (Hz)", "min": 500, "max": 20000, "default": 12000, "step": 100},
        },
    },
    # === Spatial / Time ===
    "reverb": {
        "label": "Studio Reverb",
        "description": "High-quality algorithmic reverb",
        "category": "spatial",
        "params": {
            "room_size": {"label": "Room Size", "min": 0, "max": 1, "default": 0.5, "step": 0.05},
            "damping": {"label": "Damping", "min": 0, "max": 1, "default": 0.5, "step": 0.05},
            "wet_level": {"label": "Wet Level", "min": 0, "max": 1, "default": 0.33, "step": 0.05},
            "dry_level": {"label": "Dry Level", "min": 0, "max": 1, "default": 0.7, "step": 0.05},
            "width": {"label": "Width", "min": 0, "max": 1, "default": 1.0, "step": 0.05},
        },
    },
    "delay": {
        "label": "Delay / Echo",
        "description": "Configurable delay with feedback",
        "category": "spatial",
        "params": {
            "delay_seconds": {"label": "Delay (s)", "min": 0.01, "max": 2.0, "default": 0.3, "step": 0.01},
            "feedback": {"label": "Feedback", "min": 0, "max": 0.95, "default": 0.3, "step": 0.05},
            "mix": {"label": "Mix", "min": 0, "max": 1, "default": 0.3, "step": 0.05},
        },
    },
    "chorus": {
        "label": "Chorus",
        "description": "Chorus modulation effect",
        "category": "spatial",
        "params": {
            "rate_hz": {"label": "Rate (Hz)", "min": 0.1, "max": 10, "default": 1.0, "step": 0.1},
            "depth": {"label": "Depth", "min": 0, "max": 1, "default": 0.25, "step": 0.05},
            "mix": {"label": "Mix", "min": 0, "max": 1, "default": 0.5, "step": 0.05},
            "feedback": {"label": "Feedback", "min": 0, "max": 0.95, "default": 0, "step": 0.05},
        },
    },
    "phaser": {
        "label": "Phaser",
        "description": "Phase modulation effect",
        "category": "spatial",
        "params": {
            "rate_hz": {"label": "Rate (Hz)", "min": 0.1, "max": 10, "default": 1.0, "step": 0.1},
            "depth": {"label": "Depth", "min": 0, "max": 1, "default": 0.5, "step": 0.05},
            "mix": {"label": "Mix", "min": 0, "max": 1, "default": 0.5, "step": 0.05},
            "feedback": {"label": "Feedback", "min": 0, "max": 0.95, "default": 0, "step": 0.05},
        },
    },
    # === Distortion / Character ===
    "distortion": {
        "label": "Distortion",
        "description": "Overdrive / saturation",
        "category": "character",
        "params": {
            "drive_db": {"label": "Drive (dB)", "min": 0, "max": 40, "default": 15, "step": 1},
        },
    },
    "bitcrush": {
        "label": "Bitcrusher",
        "description": "Lo-fi bit depth reduction",
        "category": "character",
        "params": {
            "bit_depth": {"label": "Bit Depth", "min": 2, "max": 16, "default": 8, "step": 1},
        },
    },
    # === Pitch ===
    "pitch_shift": {
        "label": "Pitch Shift",
        "description": "Shift pitch without changing tempo",
        "category": "pitch",
        "params": {
            "semitones": {"label": "Semitones", "min": -12, "max": 12, "default": 0, "step": 1},
        },
    },
    # === Presets (combined chains) ===
    "podcast_master": {
        "label": "Podcast Master",
        "description": "Complete podcast chain: HPF + comp + limiter + normalize",
        "category": "preset",
        "params": {},
    },
    "vocal_enhance": {
        "label": "Vocal Enhance",
        "description": "De-ess + presence boost + light compression",
        "category": "preset",
        "params": {},
    },
    "cinematic_voice": {
        "label": "Cinematic Voice",
        "description": "Deep, warm voiceover tone",
        "category": "preset",
        "params": {},
    },
}


def get_pedalboard_effects() -> List[Dict]:
    """Return pedalboard effects metadata for UI."""
    return [
        {
            "name": k,
            "label": v["label"],
            "description": v["description"],
            "category": v["category"],
            "params": v.get("params", {}),
        }
        for k, v in PEDALBOARD_EFFECTS.items()
    ]


# ---------------------------------------------------------------------------
# Pedalboard Effect Application
# ---------------------------------------------------------------------------
def apply_pedalboard_effect(
    input_path: str,
    effect_name: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    params: Optional[Dict] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply a Pedalboard effect to audio.

    For video files, extracts audio, processes, then remuxes.
    """
    _ensure_package("pedalboard", "pedalboard", on_progress)
    import pedalboard
    from pedalboard.io import AudioFile

    if effect_name not in PEDALBOARD_EFFECTS:
        raise ValueError(f"Unknown effect: {effect_name}")

    params = params or {}
    if output_path is None:
        output_path = os.path.join(
            output_dir or os.path.dirname(input_path),
            f"{os.path.splitext(os.path.basename(input_path))[0]}_{effect_name}{os.path.splitext(input_path)[1]}",
        )

    if on_progress:
        on_progress(10, f"Applying {PEDALBOARD_EFFECTS[effect_name]['label']}...")

    # Check if input is video - extract audio first
    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
    audio_input = input_path
    temp_audio = None
    temp_output = None

    try:
        if is_video:
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            _run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-i", input_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100",
                temp_audio,
            ])
            audio_input = temp_audio
            temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        else:
            temp_output = output_path
            if ext not in (".wav", ".flac", ".aiff"):
                # Convert to WAV for processing
                temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                _run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-y", "-i", input_path,
                    "-acodec", "pcm_s16le", "-ar", "44100",
                    temp_audio,
                ])
                audio_input = temp_audio
                temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        # Build the pedalboard
        board = _build_pedalboard(effect_name, params, pedalboard)

        if on_progress:
            on_progress(30, "Processing audio...")

        # Stream-process the audio
        with AudioFile(audio_input) as infile:
            with AudioFile(
                temp_output,
                "w",
                samplerate=infile.samplerate,
                num_channels=infile.num_channels,
            ) as outfile:
                chunk_size = 8192
                total_frames = infile.frames
                processed = 0
                while infile.tell() < total_frames:
                    audio = infile.read(chunk_size)
                    effected = board(audio, infile.samplerate, reset=False)
                    outfile.write(effected)
                    processed += chunk_size
                    if on_progress and processed % (chunk_size * 50) == 0:
                        pct = 30 + int((processed / max(1, total_frames)) * 60)
                        on_progress(min(90, pct), "Processing audio...")

        if on_progress:
            on_progress(92, "Encoding output...")

        # Remux if video
        if is_video:
            _run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-i", input_path, "-i", temp_output,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                output_path,
            ])
        elif ext not in (".wav", ".flac", ".aiff"):
            # Re-encode to original format
            _run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-i", temp_output,
                output_path,
            ])

        if on_progress:
            on_progress(100, f"{PEDALBOARD_EFFECTS[effect_name]['label']} applied")
        return output_path

    finally:
        for f in [temp_audio, temp_output if is_video else None]:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except OSError:
                    pass


def _build_pedalboard(effect_name: str, params: Dict, pb_module):
    """Build a pedalboard from effect name and params."""
    pb = pb_module

    if effect_name == "compressor":
        return pb.Pedalboard([
            pb.Compressor(
                threshold_db=params.get("threshold_db", -20),
                ratio=params.get("ratio", 4),
                attack_ms=params.get("attack_ms", 1),
                release_ms=params.get("release_ms", 100),
            ),
        ])
    elif effect_name == "limiter":
        return pb.Pedalboard([
            pb.Limiter(
                threshold_db=params.get("threshold_db", -1),
                release_ms=params.get("release_ms", 100),
            ),
        ])
    elif effect_name == "noise_gate":
        return pb.Pedalboard([
            pb.NoiseGate(
                threshold_db=params.get("threshold_db", -40),
                attack_ms=params.get("attack_ms", 1),
                release_ms=params.get("release_ms", 100),
            ),
        ])
    elif effect_name == "low_shelf":
        return pb.Pedalboard([
            pb.LowShelfFilter(
                cutoff_frequency_hz=params.get("cutoff_hz", 200),
                gain_db=params.get("gain_db", 3),
            ),
        ])
    elif effect_name == "high_shelf":
        return pb.Pedalboard([
            pb.HighShelfFilter(
                cutoff_frequency_hz=params.get("cutoff_hz", 8000),
                gain_db=params.get("gain_db", 3),
            ),
        ])
    elif effect_name == "peak_eq":
        return pb.Pedalboard([
            pb.PeakFilter(
                cutoff_frequency_hz=params.get("cutoff_hz", 1000),
                gain_db=params.get("gain_db", 3),
                q=params.get("q", 1.0),
            ),
        ])
    elif effect_name == "highpass":
        return pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=params.get("cutoff_hz", 80)),
        ])
    elif effect_name == "lowpass":
        return pb.Pedalboard([
            pb.LowpassFilter(cutoff_frequency_hz=params.get("cutoff_hz", 12000)),
        ])
    elif effect_name == "reverb":
        return pb.Pedalboard([
            pb.Reverb(
                room_size=params.get("room_size", 0.5),
                damping=params.get("damping", 0.5),
                wet_level=params.get("wet_level", 0.33),
                dry_level=params.get("dry_level", 0.7),
                width=params.get("width", 1.0),
            ),
        ])
    elif effect_name == "delay":
        return pb.Pedalboard([
            pb.Delay(
                delay_seconds=params.get("delay_seconds", 0.3),
                feedback=params.get("feedback", 0.3),
                mix=params.get("mix", 0.3),
            ),
        ])
    elif effect_name == "chorus":
        return pb.Pedalboard([
            pb.Chorus(
                rate_hz=params.get("rate_hz", 1.0),
                depth=params.get("depth", 0.25),
                mix=params.get("mix", 0.5),
                feedback=params.get("feedback", 0),
            ),
        ])
    elif effect_name == "phaser":
        return pb.Pedalboard([
            pb.Phaser(
                rate_hz=params.get("rate_hz", 1.0),
                depth=params.get("depth", 0.5),
                mix=params.get("mix", 0.5),
                feedback=params.get("feedback", 0),
            ),
        ])
    elif effect_name == "distortion":
        return pb.Pedalboard([
            pb.Distortion(drive_db=params.get("drive_db", 15)),
        ])
    elif effect_name == "bitcrush":
        return pb.Pedalboard([
            pb.Bitcrush(bit_depth=params.get("bit_depth", 8)),
        ])
    elif effect_name == "pitch_shift":
        return pb.Pedalboard([
            pb.PitchShift(semitones=params.get("semitones", 0)),
        ])
    # === Preset chains ===
    elif effect_name == "podcast_master":
        return pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=80),
            pb.Compressor(threshold_db=-18, ratio=3, attack_ms=5, release_ms=100),
            pb.LowShelfFilter(cutoff_frequency_hz=200, gain_db=1.5),
            pb.HighShelfFilter(cutoff_frequency_hz=8000, gain_db=2),
            pb.Limiter(threshold_db=-1, release_ms=50),
            pb.Gain(gain_db=2),
        ])
    elif effect_name == "vocal_enhance":
        return pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=100),
            pb.PeakFilter(cutoff_frequency_hz=3000, gain_db=3, q=1.5),
            pb.PeakFilter(cutoff_frequency_hz=6000, gain_db=-2, q=2),
            pb.Compressor(threshold_db=-20, ratio=2.5, attack_ms=5, release_ms=80),
            pb.Gain(gain_db=1),
        ])
    elif effect_name == "cinematic_voice":
        return pb.Pedalboard([
            pb.HighpassFilter(cutoff_frequency_hz=60),
            pb.LowShelfFilter(cutoff_frequency_hz=250, gain_db=3),
            pb.PeakFilter(cutoff_frequency_hz=2500, gain_db=-1.5, q=1),
            pb.Compressor(threshold_db=-15, ratio=4, attack_ms=2, release_ms=120),
            pb.Reverb(room_size=0.2, damping=0.7, wet_level=0.08, dry_level=0.92),
            pb.Limiter(threshold_db=-0.5),
        ])
    else:
        raise ValueError(f"No pedalboard chain for effect: {effect_name}")


# ---------------------------------------------------------------------------
# DeepFilterNet AI Noise Reduction
# ---------------------------------------------------------------------------
def deepfilter_denoise(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    AI noise reduction using DeepFilterNet.

    Broadcast-grade speech enhancement that runs real-time on CPU.
    Model size: <10MB. Supports 48kHz full-band audio.
    """
    _ensure_package("df", "deepfilternet", on_progress)

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1]
        directory = output_dir or os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_ai_denoised{ext}")

    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

    if on_progress:
        on_progress(10, "Loading DeepFilterNet model...")

    temp_audio_in = None
    temp_audio_out = None

    try:
        # Extract audio if video
        temp_audio_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        _run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
            temp_audio_in,
        ])

        if on_progress:
            on_progress(25, "Running AI noise reduction...")

        from df.enhance import enhance, init_df, load_audio, save_audio

        model, df_state, _ = init_df()

        audio, _ = load_audio(temp_audio_in, sr=df_state.sr())

        if on_progress:
            on_progress(50, "Enhancing audio...")

        enhanced = enhance(model, df_state, audio)

        temp_audio_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        save_audio(temp_audio_out, enhanced, sr=df_state.sr())

        if on_progress:
            on_progress(85, "Encoding output...")

        if is_video:
            _run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-i", input_path, "-i", temp_audio_out,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                output_path,
            ])
        else:
            _run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-i", temp_audio_out,
                output_path,
            ])

        if on_progress:
            on_progress(100, "AI noise reduction complete")
        return output_path

    finally:
        for f in [temp_audio_in, temp_audio_out]:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except OSError:
                    pass
        # Free model memory
        try:
            del model, df_state
        except Exception:
            pass
