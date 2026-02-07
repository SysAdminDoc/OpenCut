"""
OpenCut Audio Suite v0.7.0

Provides audio processing features:
- Noise reduction (FFmpeg afftdn / highpass+lowpass / DeepFilterNet3)
- Loudness normalization (FFmpeg loudnorm, EBU R128)
- Beat detection (FFmpeg + energy analysis)
- Audio ducking (VAD-based volume automation, mix with music)
- Voice isolation (FFmpeg bandpass / Demucs v4 stem separation)
- Audio effects (reverb, echo, pitch shift, etc.)
- EQ presets (voice enhance, bass boost, podcast, broadcast, etc.)
- Audio crossfade utilities

FFmpeg features work out-of-box. AI features (DeepFilterNet, Demucs)
download models on first use to %LOCALAPPDATA%/OpenCut/models/
"""

import json
import logging
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Model Directory
# ---------------------------------------------------------------------------
def _get_models_dir() -> Path:
    """Get the OpenCut models directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")))
    models_dir = base / "OpenCut" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class LoudnessInfo:
    """EBU R128 loudness measurement."""
    input_i: float = -24.0      # Integrated loudness (LUFS)
    input_tp: float = -1.0      # True peak (dBTP)
    input_lra: float = 7.0      # Loudness range (LU)
    input_thresh: float = -34.0  # Threshold
    target_offset: float = 0.0   # Offset applied


@dataclass
class BeatInfo:
    """Beat detection results."""
    bpm: float = 120.0
    beat_times: List[float] = field(default_factory=list)
    downbeat_times: List[float] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class AudioEffect:
    """An audio effect to apply."""
    name: str
    params: Dict = field(default_factory=dict)


@dataclass
class StemSeparationResult:
    """Result from Demucs stem separation."""
    vocals_path: Optional[str] = None
    drums_path: Optional[str] = None
    bass_path: Optional[str] = None
    other_path: Optional[str] = None
    model_name: str = "htdemucs"


@dataclass
class DuckingKeyframe:
    """A single keyframe for audio ducking."""
    time: float
    volume_db: float


@dataclass
class CrossfadeResult:
    """Result from audio crossfade operation."""
    output_path: str
    duration_ms: float
    fade_type: str


# ---------------------------------------------------------------------------
# Noise Reduction
# ---------------------------------------------------------------------------
def _check_deepfilter_available() -> Tuple[bool, str]:
    """Check if DeepFilterNet is installed."""
    try:
        import df
        return True, "deepfilternet"
    except ImportError:
        return False, "not_installed"


def _install_deepfilter() -> bool:
    """Attempt to install DeepFilterNet."""
    try:
        logger.info("Installing DeepFilterNet3...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "deepfilternet", "--quiet"],
            capture_output=True, timeout=300
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"DeepFilterNet install failed: {e}")
        return False


def denoise_audio(
    input_path: str,
    output_path: Optional[str] = None,
    method: str = "afftdn",
    noise_floor: float = -30.0,
    strength: float = 0.7,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Remove background noise from audio/video.

    Args:
        input_path: Source media file.
        output_path: Output path (auto-generated if None).
        method: "afftdn" (adaptive FFT), "highpass" (simple filter), 
                "gate" (noise gate), "deepfilter" (AI-based DeepFilterNet3).
        noise_floor: Noise floor in dB for gate method.
        strength: Reduction strength 0.0-1.0 (maps to FFmpeg params).
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to denoised output file.
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_denoised{ext}"

    if on_progress:
        on_progress(10, "Analyzing noise profile...")

    # DeepFilterNet3 path (AI-based, high quality)
    if method == "deepfilter":
        return _denoise_deepfilter(input_path, output_path, strength, on_progress)

    # Build filter chain based on method
    if method == "afftdn":
        # Adaptive FFT denoiser - best quality
        # nr = noise reduction amount (0-97 dB), nt = noise type
        nr_amount = int(strength * 40 + 5)  # 5-45 dB reduction
        af_filter = f"afftdn=nr={nr_amount}:nf={noise_floor}:tn=1"

    elif method == "highpass":
        # Simple highpass + lowpass to remove rumble and hiss
        hp_freq = int(60 + strength * 120)   # 60-180 Hz highpass
        lp_freq = int(16000 - strength * 4000)  # 12000-16000 Hz lowpass
        af_filter = f"highpass=f={hp_freq},lowpass=f={lp_freq}"

    elif method == "gate":
        # Noise gate - silence audio below threshold
        gate_thresh = noise_floor + (strength * 10)  # Adjust gate threshold
        af_filter = f"agate=threshold={gate_thresh}dB:ratio=10:attack=5:release=50"

    else:
        raise ValueError(f"Unknown denoise method: {method}")

    if on_progress:
        on_progress(30, f"Applying noise reduction ({method})...")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-af", af_filter,
        "-c:v", "copy",  # Pass video through unchanged
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Noise reduction failed: {result.stderr.decode()}")

    if on_progress:
        on_progress(100, "Noise reduction complete")

    return output_path


def _denoise_deepfilter(
    input_path: str,
    output_path: str,
    strength: float = 0.7,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply DeepFilterNet3 AI-based noise reduction.
    
    Downloads model on first use (~40MB).
    """
    available, _ = _check_deepfilter_available()
    if not available:
        if on_progress:
            on_progress(15, "Installing DeepFilterNet3 (first-time setup)...")
        if not _install_deepfilter():
            raise RuntimeError(
                "DeepFilterNet3 not installed. Install with: pip install deepfilternet"
            )
    
    if on_progress:
        on_progress(20, "Loading DeepFilterNet3 model...")
    
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        from df.utils import download_file
    except ImportError as e:
        raise RuntimeError(f"DeepFilterNet import failed: {e}")
    
    # Extract audio to temp WAV for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_wav = os.path.join(tmpdir, "input.wav")
        temp_out = os.path.join(tmpdir, "output.wav")
        
        if on_progress:
            on_progress(30, "Extracting audio...")
        
        # Extract audio as 48kHz WAV (DeepFilter optimal sample rate)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            "-vn", "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
            temp_wav,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Audio extraction failed: {result.stderr.decode()}")
        
        if on_progress:
            on_progress(50, "Applying DeepFilterNet3 AI denoise...")
        
        # Initialize model and enhance
        model, df_state, _ = init_df()
        audio, _ = load_audio(temp_wav, sr=df_state.sr())
        
        # Strength maps to atten_lim: lower = more aggressive
        atten_lim = int(100 - (strength * 60))  # 40-100 dB attenuation limit
        enhanced = enhance(model, df_state, audio, atten_lim_db=atten_lim)
        
        save_audio(temp_out, enhanced, df_state.sr())
        
        if on_progress:
            on_progress(80, "Merging enhanced audio...")
        
        # Determine if source has video
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", input_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        has_video = probe_result.stdout.strip() not in ("", "0", "N/A")
        
        if has_video:
            # Merge enhanced audio back with video
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-i", input_path, "-i", temp_out,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                output_path,
            ]
        else:
            # Audio only - just copy the enhanced audio
            ext = os.path.splitext(output_path)[1].lower()
            if ext in (".mp3",):
                cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                       "-y", "-i", temp_out, "-c:a", "libmp3lame", "-b:a", "192k", output_path]
            elif ext in (".aac", ".m4a"):
                cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                       "-y", "-i", temp_out, "-c:a", "aac", "-b:a", "192k", output_path]
            else:
                cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                       "-y", "-i", temp_out, "-c:a", "pcm_s16le", output_path]
        
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"Audio merge failed: {result.stderr.decode()}")
    
    if on_progress:
        on_progress(100, "DeepFilterNet3 denoise complete")
    
    return output_path


# ---------------------------------------------------------------------------
# Voice Isolation (Bandpass Emphasis / Demucs v4)
# ---------------------------------------------------------------------------
def _check_demucs_available() -> Tuple[bool, str]:
    """Check if Demucs is installed."""
    try:
        import demucs
        return True, "demucs"
    except ImportError:
        return False, "not_installed"


def _install_demucs() -> bool:
    """Attempt to install Demucs."""
    try:
        logger.info("Installing Demucs v4...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "demucs", "--quiet"],
            capture_output=True, timeout=600
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"Demucs install failed: {e}")
        return False


def isolate_voice(
    input_path: str,
    output_path: Optional[str] = None,
    method: str = "bandpass",
    low_freq: int = 200,
    high_freq: int = 4000,
    boost_db: float = 6.0,
    reduce_db: float = -12.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Isolate vocals using FFmpeg bandpass filtering.

    For proper AI-based stem separation, use isolate_stems() with Demucs.

    Args:
        input_path: Source media file.
        output_path: Output path.
        method: "bandpass" (FFmpeg EQ) or "demucs" (AI stem separation).
        low_freq: Low cutoff for voice band (Hz).
        high_freq: High cutoff for voice band (Hz).
        boost_db: Amount to boost voice frequencies.
        reduce_db: Amount to reduce non-voice frequencies.

    Returns:
        Path to output file (for demucs, returns vocals stem path).
    """
    if method == "demucs":
        result = isolate_stems(input_path, on_progress=on_progress)
        return result.vocals_path or input_path
    
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_voice{ext}"

    if on_progress:
        on_progress(20, "Isolating voice frequencies...")

    # Use equalizer bands to emphasize voice range
    af_filter = (
        f"equalizer=f=100:t=h:w=200:g={reduce_db},"
        f"equalizer=f={low_freq}:t=h:w=100:g={boost_db / 2},"
        f"equalizer=f=1000:t=h:w=800:g={boost_db},"
        f"equalizer=f=3000:t=h:w=1000:g={boost_db / 2},"
        f"equalizer=f={high_freq}:t=h:w=2000:g={reduce_db},"
        f"equalizer=f=10000:t=h:w=4000:g={reduce_db}"
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-af", af_filter,
        "-c:v", "copy",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Voice isolation failed: {result.stderr.decode()}")

    if on_progress:
        on_progress(100, "Voice isolation complete")

    return output_path


def isolate_stems(
    input_path: str,
    output_dir: Optional[str] = None,
    model: str = "htdemucs",
    stems: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> StemSeparationResult:
    """
    Separate audio into stems using Demucs v4.
    
    Downloads the htdemucs model (~80MB) on first use.
    
    Args:
        input_path: Source media file.
        output_dir: Directory for output stems (auto-created if None).
        model: Demucs model name ("htdemucs", "htdemucs_ft", "mdx_extra").
        stems: List of stems to export. None = all (vocals, drums, bass, other).
        on_progress: Progress callback(pct, msg).
    
    Returns:
        StemSeparationResult with paths to each stem.
    """
    available, _ = _check_demucs_available()
    if not available:
        if on_progress:
            on_progress(5, "Installing Demucs v4 (first-time setup)...")
        if not _install_demucs():
            raise RuntimeError(
                "Demucs not installed. Install with: pip install demucs"
            )
    
    if on_progress:
        on_progress(10, f"Loading {model} model...")
    
    try:
        import torch
        import torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError as e:
        raise RuntimeError(f"Demucs import failed: {e}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_path), "stems")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demucs_model = get_model(model)
    demucs_model.to(device)
    demucs_model.eval()
    
    if on_progress:
        on_progress(20, "Loading audio...")
    
    # Extract audio to temp WAV
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_wav = os.path.join(tmpdir, "input.wav")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            "-vn", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
            temp_wav,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Audio extraction failed: {result.stderr.decode()}")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(temp_wav)
        
        # Ensure stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        
        # Add batch dimension
        waveform = waveform.unsqueeze(0).to(device)
        
        if on_progress:
            on_progress(40, "Separating stems (this may take a while)...")
        
        # Apply model
        with torch.no_grad():
            sources = apply_model(demucs_model, waveform, device=device)
        
        # sources shape: (batch, sources, channels, samples)
        # Model sources order: drums, bass, other, vocals
        source_names = demucs_model.sources  # ['drums', 'bass', 'other', 'vocals']
        
        if on_progress:
            on_progress(80, "Saving stems...")
        
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        result = StemSeparationResult(model_name=model)
        
        stems_to_export = stems or source_names
        
        for idx, name in enumerate(source_names):
            if name not in stems_to_export:
                continue
            
            stem_audio = sources[0, idx].cpu()
            stem_path = os.path.join(output_dir, f"{base_name}_{name}.wav")
            torchaudio.save(stem_path, stem_audio, sample_rate)
            
            if name == "vocals":
                result.vocals_path = stem_path
            elif name == "drums":
                result.drums_path = stem_path
            elif name == "bass":
                result.bass_path = stem_path
            elif name == "other":
                result.other_path = stem_path
    
    if on_progress:
        on_progress(100, f"Stem separation complete: {len(stems_to_export)} stems")
    
    return result


# ---------------------------------------------------------------------------
# Loudness Normalization (EBU R128)
# ---------------------------------------------------------------------------
LOUDNESS_PRESETS = {
    "youtube":    {"i": -14.0, "tp": -1.0, "lra": 11.0},
    "podcast":    {"i": -16.0, "tp": -1.5, "lra": 8.0},
    "broadcast":  {"i": -23.0, "tp": -2.0, "lra": 7.0},
    "streaming":  {"i": -14.0, "tp": -1.0, "lra": 11.0},
    "tiktok":     {"i": -14.0, "tp": -1.0, "lra": 11.0},
    "spotify":    {"i": -14.0, "tp": -1.0, "lra": 9.0},
    "apple_music": {"i": -16.0, "tp": -1.0, "lra": 10.0},
    "cinema":     {"i": -24.0, "tp": -2.0, "lra": 20.0},
}


def measure_loudness(input_path: str) -> LoudnessInfo:
    """
    Measure audio loudness using FFmpeg loudnorm (EBU R128).

    Returns:
        LoudnessInfo with integrated loudness, true peak, and range.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info",
        "-i", input_path,
        "-af", "loudnorm=print_format=json",
        "-f", "null", "-",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    # loudnorm prints JSON to stderr
    output = result.stderr

    # Parse the JSON block from loudnorm output
    info = LoudnessInfo()
    try:
        # Find the JSON block
        json_start = output.rfind("{")
        json_end = output.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(output[json_start:json_end])
            info.input_i = float(data.get("input_i", -24.0))
            info.input_tp = float(data.get("input_tp", -1.0))
            info.input_lra = float(data.get("input_lra", 7.0))
            info.input_thresh = float(data.get("input_thresh", -34.0))
            info.target_offset = float(data.get("target_offset", 0.0))
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Could not parse loudnorm output: {e}")

    return info


def normalize_loudness(
    input_path: str,
    output_path: Optional[str] = None,
    preset: str = "youtube",
    target_lufs: Optional[float] = None,
    target_tp: Optional[float] = None,
    target_lra: Optional[float] = None,
    two_pass: bool = True,
    on_progress: Optional[Callable] = None,
) -> Tuple[str, LoudnessInfo]:
    """
    Normalize audio loudness to a target (EBU R128).

    Args:
        input_path: Source media file.
        output_path: Output path.
        preset: Loudness preset name (youtube, podcast, broadcast, etc.).
        target_lufs: Override target integrated loudness (LUFS).
        target_tp: Override target true peak (dBTP).
        target_lra: Override target loudness range (LU).
        two_pass: Use two-pass normalization for better quality.

    Returns:
        Tuple of (output_path, LoudnessInfo with measurements).
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_normalized{ext}"

    # Get targets from preset or overrides
    targets = LOUDNESS_PRESETS.get(preset, LOUDNESS_PRESETS["youtube"]).copy()
    if target_lufs is not None:
        targets["i"] = target_lufs
    if target_tp is not None:
        targets["tp"] = target_tp
    if target_lra is not None:
        targets["lra"] = target_lra

    if two_pass:
        # Pass 1: Measure
        if on_progress:
            on_progress(10, "Pass 1/2: Measuring loudness...")
        info = measure_loudness(input_path)

        if on_progress:
            on_progress(50, f"Pass 2/2: Normalizing ({info.input_i:.1f} -> {targets['i']:.1f} LUFS)...")

        # Pass 2: Apply with measured values for precision
        af_filter = (
            f"loudnorm="
            f"I={targets['i']}:TP={targets['tp']}:LRA={targets['lra']}:"
            f"measured_I={info.input_i}:measured_TP={info.input_tp}:"
            f"measured_LRA={info.input_lra}:measured_thresh={info.input_thresh}:"
            f"offset={info.target_offset}:linear=true:print_format=json"
        )
    else:
        if on_progress:
            on_progress(20, f"Normalizing to {targets['i']:.1f} LUFS...")
        info = LoudnessInfo()
        af_filter = (
            f"loudnorm="
            f"I={targets['i']}:TP={targets['tp']}:LRA={targets['lra']}:"
            f"print_format=json"
        )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info",
        "-y", "-i", input_path,
        "-af", af_filter,
        "-c:v", "copy",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Loudness normalization failed: {result.stderr}")

    if on_progress:
        on_progress(100, "Loudness normalization complete")

    return output_path, info


# ---------------------------------------------------------------------------
# Beat Detection (FFmpeg energy-based)
# ---------------------------------------------------------------------------
def detect_beats(
    input_path: str,
    sensitivity: float = 0.5,
    min_bpm: int = 60,
    max_bpm: int = 200,
    on_progress: Optional[Callable] = None,
) -> BeatInfo:
    """
    Detect beats using audio energy onset detection.

    Uses FFmpeg to extract audio, then analyzes energy transients
    to find beat positions. Works well for music and rhythmic content.

    Args:
        input_path: Source media file.
        sensitivity: Beat detection sensitivity (0.0=few, 1.0=many).
        min_bpm: Minimum expected BPM.
        max_bpm: Maximum expected BPM.

    Returns:
        BeatInfo with BPM, beat times, and confidence.
    """
    if on_progress:
        on_progress(10, "Extracting audio for beat analysis...")

    # Extract audio as raw PCM
    sample_rate = 44100
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        "-f", "s16le", "-",
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr.decode()}")

    pcm_data = result.stdout
    num_samples = len(pcm_data) // 2
    if num_samples < sample_rate:
        return BeatInfo()

    if on_progress:
        on_progress(30, "Analyzing energy transients...")

    samples = struct.unpack(f"<{num_samples}h", pcm_data)
    max_val = 32768.0

    # Compute short-time energy with onset detection
    window_ms = 20  # 20ms windows
    hop_ms = 10     # 10ms hop
    window_samples = int(sample_rate * window_ms / 1000)
    hop_samples = int(sample_rate * hop_ms / 1000)

    energies = []
    pos = 0
    while pos + window_samples <= num_samples:
        window = samples[pos:pos + window_samples]
        rms = math.sqrt(sum(s * s for s in window) / len(window)) / max_val
        energies.append(rms)
        pos += hop_samples

    if not energies:
        return BeatInfo()

    if on_progress:
        on_progress(50, "Detecting onsets...")

    # Onset detection: spectral flux approximation via energy derivative
    # Compute difference function (positive changes = onsets)
    onset_strength = []
    for i in range(1, len(energies)):
        diff = energies[i] - energies[i - 1]
        onset_strength.append(max(0.0, diff))

    if not onset_strength:
        return BeatInfo()

    # Adaptive threshold for onset picking
    max_onset = max(onset_strength) if onset_strength else 1.0
    if max_onset <= 0:
        return BeatInfo()

    # Sensitivity maps to threshold: high sensitivity = lower threshold
    threshold = max_onset * (0.8 - sensitivity * 0.6)  # 0.2 to 0.8 of max

    # Find peaks above threshold with minimum spacing
    min_spacing_ms = int(60000 / max_bpm)  # Min time between beats
    min_spacing_frames = max(1, min_spacing_ms // hop_ms)

    beat_frames = []
    last_beat = -min_spacing_frames
    for i, val in enumerate(onset_strength):
        if val > threshold and (i - last_beat) >= min_spacing_frames:
            # Check if local peak (within 3 frames)
            start = max(0, i - 3)
            end = min(len(onset_strength), i + 4)
            if val >= max(onset_strength[start:end]):
                beat_frames.append(i)
                last_beat = i

    if on_progress:
        on_progress(70, "Estimating tempo...")

    # Convert frame indices to times
    beat_times = [f * hop_ms / 1000.0 for f in beat_frames]

    # Estimate BPM from inter-beat intervals
    bpm = 120.0
    confidence = 0.0
    if len(beat_times) >= 4:
        intervals = [beat_times[i + 1] - beat_times[i] for i in range(len(beat_times) - 1)]
        # Filter reasonable intervals
        reasonable = [iv for iv in intervals if (60 / max_bpm) <= iv <= (60 / min_bpm)]
        if reasonable:
            # Use median interval for robustness
            reasonable.sort()
            median_interval = reasonable[len(reasonable) // 2]
            bpm = round(60.0 / median_interval, 1)

            # Confidence based on interval consistency
            mean_interval = sum(reasonable) / len(reasonable)
            variance = sum((iv - mean_interval) ** 2 for iv in reasonable) / len(reasonable)
            std = variance ** 0.5
            # Lower std = higher confidence
            confidence = max(0.0, min(1.0, 1.0 - (std / mean_interval)))

    # Identify downbeats (every 4th beat, assuming 4/4 time)
    downbeat_times = beat_times[::4] if beat_times else []

    if on_progress:
        on_progress(100, f"Beat detection complete: {bpm:.0f} BPM, {len(beat_times)} beats")

    return BeatInfo(
        bpm=bpm,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Audio Ducking
# ---------------------------------------------------------------------------
def generate_ducking_keyframes(
    input_path: str,
    music_volume: float = -12.0,
    duck_volume: float = -24.0,
    attack_ms: float = 200.0,
    release_ms: float = 500.0,
    speech_threshold_db: float = -30.0,
    on_progress: Optional[Callable] = None,
) -> List[Dict]:
    """
    Generate volume keyframes for audio ducking.

    Analyzes speech activity and creates keyframes that reduce
    background music volume during speech.

    Args:
        input_path: Source media file.
        music_volume: Normal music volume (dB).
        duck_volume: Ducked music volume during speech (dB).
        attack_ms: Time to duck down (ms).
        release_ms: Time to come back up (ms).
        speech_threshold_db: RMS threshold to detect speech.

    Returns:
        List of keyframe dicts: {"time": float, "volume_db": float}
    """
    if on_progress:
        on_progress(10, "Analyzing speech activity...")

    sample_rate = 16000
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        "-f", "s16le", "-",
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr.decode()}")

    pcm_data = result.stdout
    num_samples = len(pcm_data) // 2
    if num_samples < sample_rate:
        return []

    samples = struct.unpack(f"<{num_samples}h", pcm_data)
    max_val = 32768.0

    # Analyze energy in 50ms windows
    window_size = int(sample_rate * 0.05)
    hop_size = int(sample_rate * 0.025)

    if on_progress:
        on_progress(40, "Detecting speech regions...")

    # Convert threshold from dB to linear
    threshold_linear = 10 ** (speech_threshold_db / 20.0)

    speech_regions = []
    in_speech = False
    speech_start = 0.0
    pos = 0

    while pos + window_size <= num_samples:
        window = samples[pos:pos + window_size]
        rms = math.sqrt(sum(s * s for s in window) / len(window)) / max_val
        time_sec = pos / sample_rate

        if rms > threshold_linear:
            if not in_speech:
                speech_start = time_sec
                in_speech = True
        else:
            if in_speech:
                speech_regions.append({"start": speech_start, "end": time_sec})
                in_speech = False

        pos += hop_size

    # Close final region
    if in_speech:
        speech_regions.append({"start": speech_start, "end": num_samples / sample_rate})

    if on_progress:
        on_progress(70, f"Generating ducking keyframes ({len(speech_regions)} speech regions)...")

    # Merge close speech regions (within release time)
    release_sec = release_ms / 1000.0
    merged = []
    for region in speech_regions:
        if merged and (region["start"] - merged[-1]["end"]) < release_sec:
            merged[-1]["end"] = region["end"]
        else:
            merged.append(region.copy())

    # Generate keyframes
    attack_sec = attack_ms / 1000.0
    keyframes = [{"time": 0.0, "volume_db": music_volume}]

    for region in merged:
        # Duck down before speech
        duck_start = max(0.0, region["start"] - attack_sec)
        keyframes.append({"time": duck_start, "volume_db": music_volume})
        keyframes.append({"time": region["start"], "volume_db": duck_volume})
        # Come back up after speech
        keyframes.append({"time": region["end"], "volume_db": duck_volume})
        release_end = region["end"] + release_sec
        keyframes.append({"time": release_end, "volume_db": music_volume})

    if on_progress:
        on_progress(100, f"Audio ducking complete: {len(keyframes)} keyframes")

    return keyframes


# ---------------------------------------------------------------------------
# Audio Effects
# ---------------------------------------------------------------------------
AUDIO_EFFECTS = {
    "reverb": {
        "label": "Reverb",
        "description": "Add room reverb",
        "filter": "aecho=0.8:0.88:60:0.4",
    },
    "echo": {
        "label": "Echo",
        "description": "Add echo effect",
        "filter": "aecho=0.8:0.9:500:0.3",
    },
    "bass_boost": {
        "label": "Bass Boost",
        "description": "Enhance low frequencies",
        "filter": "bass=g=10:f=110:w=0.6",
    },
    "treble_boost": {
        "label": "Treble Boost",
        "description": "Enhance high frequencies",
        "filter": "treble=g=8:f=4000:w=0.6",
    },
    "telephone": {
        "label": "Telephone",
        "description": "Lo-fi telephone sound",
        "filter": "highpass=f=300,lowpass=f=3400,volume=1.5",
    },
    "radio": {
        "label": "Radio",
        "description": "Vintage radio effect",
        "filter": "highpass=f=200,lowpass=f=5000,acompressor=threshold=-20dB:ratio=4",
    },
    "deep_voice": {
        "label": "Deep Voice",
        "description": "Lower pitch for deeper voice",
        "filter": "asetrate=44100*0.85,aresample=44100",
    },
    "high_voice": {
        "label": "High Voice",
        "description": "Higher pitch for lighter voice",
        "filter": "asetrate=44100*1.2,aresample=44100",
    },
    "robot": {
        "label": "Robot",
        "description": "Robotic voice effect",
        "filter": "afftfilt=real='hypot(re,im)*cos(0)':imag='hypot(re,im)*sin(0)'",
    },
    "underwater": {
        "label": "Underwater",
        "description": "Muffled underwater sound",
        "filter": "lowpass=f=500,aecho=0.8:0.9:40:0.5",
    },
    "stadium": {
        "label": "Stadium",
        "description": "Large venue reverb",
        "filter": "aecho=0.8:0.88:120:0.5,aecho=0.8:0.88:240:0.3",
    },
    "vinyl": {
        "label": "Vinyl",
        "description": "Vinyl record warmth",
        "filter": "highpass=f=40,lowpass=f=12000,bass=g=3:f=200,treble=g=-2:f=8000",
    },
}


def apply_audio_effect(
    input_path: str,
    effect_name: str,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply a named audio effect to a media file.

    Args:
        input_path: Source media file.
        effect_name: Effect name from AUDIO_EFFECTS.
        output_path: Output path.

    Returns:
        Path to output file.
    """
    if effect_name not in AUDIO_EFFECTS:
        available = ", ".join(AUDIO_EFFECTS.keys())
        raise ValueError(f"Unknown effect '{effect_name}'. Available: {available}")

    effect = AUDIO_EFFECTS[effect_name]

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_{effect_name}{ext}"

    if on_progress:
        on_progress(20, f"Applying {effect['label']} effect...")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-af", effect["filter"],
        "-c:v", "copy",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Audio effect failed: {result.stderr.decode()}")

    if on_progress:
        on_progress(100, f"{effect['label']} applied")

    return output_path


def get_available_effects() -> List[Dict]:
    """Return list of available audio effects with metadata."""
    return [
        {"name": k, "label": v["label"], "description": v["description"]}
        for k, v in AUDIO_EFFECTS.items()
    ]


# ---------------------------------------------------------------------------
# EQ Presets
# ---------------------------------------------------------------------------
EQ_PRESETS = {
    "voice_enhance": {
        "label": "Voice Enhance",
        "description": "Clarity and presence for speech",
        "filter": (
            "highpass=f=80,"
            "equalizer=f=200:t=q:w=1:g=2,"
            "equalizer=f=2500:t=q:w=2:g=4,"
            "equalizer=f=5000:t=q:w=2:g=2,"
            "acompressor=threshold=-20dB:ratio=3:attack=5:release=50"
        ),
    },
    "bass_boost": {
        "label": "Bass Boost",
        "description": "Enhanced low-end frequencies",
        "filter": (
            "equalizer=f=60:t=q:w=1:g=6,"
            "equalizer=f=100:t=q:w=1:g=4,"
            "equalizer=f=200:t=q:w=1:g=2"
        ),
    },
    "treble_boost": {
        "label": "Treble Boost",
        "description": "Enhanced high-end frequencies",
        "filter": (
            "equalizer=f=4000:t=q:w=2:g=3,"
            "equalizer=f=8000:t=q:w=2:g=4,"
            "equalizer=f=12000:t=q:w=2:g=3"
        ),
    },
    "podcast_voice": {
        "label": "Podcast Voice",
        "description": "Warm, clear voice for podcasts",
        "filter": (
            "highpass=f=80,"
            "equalizer=f=120:t=q:w=1:g=3,"
            "equalizer=f=250:t=q:w=1:g=-2,"
            "equalizer=f=2000:t=q:w=2:g=3,"
            "equalizer=f=4000:t=q:w=2:g=2,"
            "acompressor=threshold=-18dB:ratio=4:attack=10:release=100"
        ),
    },
    "broadcast": {
        "label": "Broadcast",
        "description": "Professional broadcast EQ",
        "filter": (
            "highpass=f=60,"
            "lowpass=f=15000,"
            "equalizer=f=100:t=q:w=1:g=2,"
            "equalizer=f=3000:t=q:w=2:g=3,"
            "equalizer=f=8000:t=q:w=2:g=1,"
            "acompressor=threshold=-16dB:ratio=4:attack=5:release=50,"
            "alimiter=limit=0.95"
        ),
    },
    "telephone": {
        "label": "Telephone",
        "description": "Lo-fi telephone/radio effect",
        "filter": (
            "highpass=f=300,"
            "lowpass=f=3400,"
            "equalizer=f=1000:t=q:w=1:g=3"
        ),
    },
    "warmth": {
        "label": "Warmth",
        "description": "Add analog warmth to audio",
        "filter": (
            "equalizer=f=100:t=q:w=1:g=3,"
            "equalizer=f=250:t=q:w=1:g=2,"
            "equalizer=f=8000:t=q:w=2:g=-2,"
            "equalizer=f=12000:t=q:w=2:g=-3"
        ),
    },
    "presence": {
        "label": "Presence",
        "description": "Boost vocal presence and clarity",
        "filter": (
            "equalizer=f=2000:t=q:w=2:g=3,"
            "equalizer=f=4000:t=q:w=2:g=4,"
            "equalizer=f=6000:t=q:w=2:g=2"
        ),
    },
}


def apply_eq_preset(
    input_path: str,
    preset: str,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply an EQ preset to audio/video.

    Args:
        input_path: Source media file.
        preset: Preset name from EQ_PRESETS.
        output_path: Output path.
        on_progress: Progress callback.

    Returns:
        Path to output file.
    """
    if preset not in EQ_PRESETS:
        available = ", ".join(EQ_PRESETS.keys())
        raise ValueError(f"Unknown EQ preset '{preset}'. Available: {available}")

    eq = EQ_PRESETS[preset]

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_{preset}{ext}"

    if on_progress:
        on_progress(20, f"Applying {eq['label']} EQ...")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", input_path,
        "-af", eq["filter"],
        "-c:v", "copy",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"EQ preset failed: {result.stderr.decode()}")

    if on_progress:
        on_progress(100, f"{eq['label']} EQ applied")

    return output_path


def get_available_eq_presets() -> List[Dict]:
    """Return list of available EQ presets with metadata."""
    return [
        {"name": k, "label": v["label"], "description": v["description"]}
        for k, v in EQ_PRESETS.items()
    ]


# ---------------------------------------------------------------------------
# Audio Ducking Mix
# ---------------------------------------------------------------------------
def duck_audio(
    speech_path: str,
    music_path: str,
    output_path: Optional[str] = None,
    duck_level: float = -12.0,
    attack_ms: float = 200.0,
    release_ms: float = 500.0,
    speech_threshold_db: float = -30.0,
    on_progress: Optional[Callable] = None,
) -> Tuple[str, List[Dict]]:
    """
    Mix speech and music with automatic ducking.

    Analyzes speech to detect when someone is talking, then
    reduces music volume during those segments.

    Args:
        speech_path: Audio/video file with speech (voice track).
        music_path: Music/background audio file.
        output_path: Output path for mixed audio.
        duck_level: How much to reduce music (dB, negative).
        attack_ms: Time to duck down when speech starts.
        release_ms: Time to bring music back up after speech ends.
        speech_threshold_db: RMS threshold for speech detection.
        on_progress: Progress callback.

    Returns:
        Tuple of (output_path, keyframes list).
    """
    if output_path is None:
        base, ext = os.path.splitext(speech_path)
        output_path = f"{base}_ducked{ext}"

    if on_progress:
        on_progress(5, "Analyzing speech for ducking...")

    # Generate ducking keyframes from speech analysis
    keyframes = generate_ducking_keyframes(
        speech_path,
        music_volume=0.0,  # Normal level
        duck_volume=duck_level,
        attack_ms=attack_ms,
        release_ms=release_ms,
        speech_threshold_db=speech_threshold_db,
        on_progress=lambda p, m: on_progress(5 + int(p * 0.4), m) if on_progress else None,
    )

    if on_progress:
        on_progress(50, "Building ducking automation...")

    # Build FFmpeg volume filter expression from keyframes
    if len(keyframes) < 2:
        # No ducking needed - just mix at equal levels
        vol_expr = "volume=1"
    else:
        # Convert keyframes to FFmpeg volume expression
        # Using step-based volume changes
        vol_parts = []
        for i, kf in enumerate(keyframes):
            time = kf["time"]
            vol_linear = 10 ** (kf["volume_db"] / 20.0)
            if i == 0:
                vol_parts.append(f"if(lt(t,{keyframes[1]['time']:.3f}),{vol_linear:.4f}")
            elif i == len(keyframes) - 1:
                vol_parts.append(f",{vol_linear:.4f})")
            else:
                next_time = keyframes[i + 1]["time"]
                vol_parts.append(f",if(lt(t,{next_time:.3f}),{vol_linear:.4f}")
        
        # Close all if statements
        vol_expr = "volume='" + "".join(vol_parts) + "'"

    if on_progress:
        on_progress(60, "Mixing speech and music...")

    # Mix with ducking applied to music track
    # [0] = speech, [1] = music (with volume automation)
    filter_complex = (
        f"[1:a]{vol_expr}[music_ducked];"
        f"[0:a][music_ducked]amix=inputs=2:duration=first:dropout_transition=2[out]"
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", speech_path,
        "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        # Fallback to simpler mix if complex filter fails
        if on_progress:
            on_progress(65, "Using simplified ducking mix...")
        
        # Simple constant duck level
        vol_linear = 10 ** (duck_level / 20.0)
        filter_simple = (
            f"[1:a]volume={vol_linear:.4f}[music];"
            f"[0:a][music]amix=inputs=2:duration=first[out]"
        )
        
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y",
            "-i", speech_path,
            "-i", music_path,
            "-filter_complex", filter_simple,
            "-map", "[out]",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"Audio ducking mix failed: {result.stderr.decode()}")

    if on_progress:
        on_progress(100, f"Audio ducking complete: {len(keyframes)} keyframes")

    return output_path, keyframes


# ---------------------------------------------------------------------------
# Audio Crossfade
# ---------------------------------------------------------------------------
def crossfade_audio(
    input_path1: str,
    input_path2: str,
    output_path: Optional[str] = None,
    duration_ms: float = 1000.0,
    fade_type: str = "linear",
    on_progress: Optional[Callable] = None,
) -> CrossfadeResult:
    """
    Create a smooth crossfade between two audio files.

    Args:
        input_path1: First audio file (will fade out at end).
        input_path2: Second audio file (will fade in at start).
        output_path: Output path for combined audio.
        duration_ms: Crossfade duration in milliseconds.
        fade_type: Fade curve type ("linear", "exponential", "logarithmic").
        on_progress: Progress callback.

    Returns:
        CrossfadeResult with output path and settings.
    """
    if output_path is None:
        base1 = os.path.splitext(os.path.basename(input_path1))[0]
        base2 = os.path.splitext(os.path.basename(input_path2))[0]
        ext = os.path.splitext(input_path1)[1]
        output_dir = os.path.dirname(input_path1)
        output_path = os.path.join(output_dir, f"{base1}_x_{base2}{ext}")

    if on_progress:
        on_progress(20, "Preparing crossfade...")

    duration_sec = duration_ms / 1000.0

    # Map fade type to FFmpeg curve
    curve_map = {
        "linear": "tri",
        "exponential": "exp",
        "logarithmic": "log",
        "quarter": "qsin",
        "half": "hsin",
    }
    curve = curve_map.get(fade_type, "tri")

    if on_progress:
        on_progress(40, f"Applying {fade_type} crossfade ({duration_ms:.0f}ms)...")

    # Use acrossfade filter
    filter_complex = (
        f"[0:a][1:a]acrossfade=d={duration_sec}:c1={curve}:c2={curve}[out]"
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", input_path1,
        "-i", input_path2,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Audio crossfade failed: {result.stderr.decode()}")

    if on_progress:
        on_progress(100, "Crossfade complete")

    return CrossfadeResult(
        output_path=output_path,
        duration_ms=duration_ms,
        fade_type=fade_type,
    )


def get_crossfade_types() -> List[Dict]:
    """Return available crossfade curve types."""
    return [
        {"name": "linear", "label": "Linear", "description": "Constant rate fade"},
        {"name": "exponential", "label": "Exponential", "description": "Accelerating fade"},
        {"name": "logarithmic", "label": "Logarithmic", "description": "Decelerating fade"},
        {"name": "quarter", "label": "Quarter Sine", "description": "Smooth quarter-wave"},
        {"name": "half", "label": "Half Sine", "description": "Smooth half-wave"},
    ]
