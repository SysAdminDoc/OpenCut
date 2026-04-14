"""
OpenCut Enhanced Speech Module

AI-powered speech restoration: denoise, clarity enhance, bandwidth extension.
Adobe ships "Enhance Speech" natively; this is OpenCut's answer.

Primary backend: Resemble Enhance (pip: resemble-enhance)
Fallback: DeepFilterNet denoise + FFmpeg aresample for bandwidth extension
"""

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from opencut.helpers import (
    ensure_package,
    get_ffmpeg_path,
    get_ffprobe_path,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class AudioStats:
    """Audio statistics for before/after comparison."""
    sample_rate: int = 0
    snr_estimate: float = 0.0


@dataclass
class EnhanceSpeechResult:
    """Result of speech enhancement."""
    output_path: str = ""
    mode: str = ""
    original_stats: Dict = field(default_factory=dict)
    enhanced_stats: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audio probing helpers
# ---------------------------------------------------------------------------
def _probe_audio_stats(filepath: str) -> AudioStats:
    """Probe audio sample rate and estimate SNR."""
    import json as _json

    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe, "-v", "quiet", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,codec_name",
        "-of", "json", filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            data = _json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams:
                sr = int(streams[0].get("sample_rate", 0))
                return AudioStats(sample_rate=sr, snr_estimate=_estimate_snr(filepath))
    except Exception as exc:
        logger.debug("Audio probe failed for %s: %s", filepath, exc)
    return AudioStats(sample_rate=0, snr_estimate=0.0)


def _estimate_snr(filepath: str) -> float:
    """Rough SNR estimate using FFmpeg volumedetect."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-i", filepath,
        "-af", "volumedetect", "-vn", "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stderr = result.stderr
        import re
        mean_match = re.search(r"mean_volume:\s*(-?[\d.]+)", stderr)
        max_match = re.search(r"max_volume:\s*(-?[\d.]+)", stderr)
        if mean_match and max_match:
            mean_vol = float(mean_match.group(1))
            max_vol = float(max_match.group(1))
            return round(max_vol - mean_vol, 2)
    except Exception as exc:
        logger.debug("SNR estimation failed: %s", exc)
    return 0.0


def _extract_audio_wav(input_path: str, output_wav: str) -> None:
    """Extract audio as mono WAV for processing."""
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-y",
        "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "44100", "-ac", "1",
        output_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        stderr = result.stderr[-500:] if result.stderr else "unknown error"
        raise RuntimeError(f"Audio extraction failed: {stderr}")


def _normalize_lufs(input_path: str, output_path: str, target_lufs: float = -16.0) -> None:
    """Normalize audio to target LUFS using FFmpeg loudnorm."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", input_path,
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-ar", "48000",
        "-acodec", "pcm_s16le",
        output_path,
    ]
    run_ffmpeg(cmd)


def _bandwidth_extend_ffmpeg(input_path: str, output_path: str, target_sr: int = 48000) -> None:
    """Extend bandwidth to target sample rate using FFmpeg high-quality resampler."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", input_path,
        "-af", f"aresample={target_sr}:resampler=soxr:precision=28",
        "-acodec", "pcm_s16le",
        output_path,
    ]
    run_ffmpeg(cmd)


# ---------------------------------------------------------------------------
# Resemble Enhance backend
# ---------------------------------------------------------------------------
def _enhance_resemble(audio_path: str, output_wav: str, denoise: bool, enhance: bool,
                      on_progress: Optional[Callable] = None) -> int:
    """Run Resemble Enhance pipeline. Returns output sample rate."""
    try:
        import torch
        import torchaudio
    except ImportError as err:
        raise RuntimeError(
            "torch and torchaudio required. Install: pip install torch torchaudio"
        ) from err

    try:
        from resemble_enhance.enhancer.inference import denoise as _denoise_fn
        from resemble_enhance.enhancer.inference import enhance as _enhance_fn
    except ImportError as err:
        raise RuntimeError(
            "resemble-enhance required. Install: pip install resemble-enhance"
        ) from err

    audio, sr = torchaudio.load(audio_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if audio.shape[0] > 1:
            mono = audio.mean(dim=0)
        else:
            mono = audio.squeeze(0)

        if denoise:
            if on_progress:
                on_progress(35, "Denoising speech (Resemble)...")
            mono, sr = _denoise_fn(mono, sr, device)

        if enhance:
            if on_progress:
                on_progress(60, "Enhancing clarity (Resemble)...")
            mono, sr = _enhance_fn(mono, sr, device, nfe=64, solver="midpoint")

        if mono.dim() == 1:
            mono = mono.unsqueeze(0)

        torchaudio.save(output_wav, mono, sr)
        return sr
    finally:
        del audio, mono
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# DeepFilterNet fallback
# ---------------------------------------------------------------------------
def _denoise_deepfilternet(input_path: str, output_path: str,
                           on_progress: Optional[Callable] = None) -> None:
    """Denoise using DeepFilterNet CLI."""
    if not ensure_package("df", pip_name="deepfilternet"):
        raise RuntimeError("DeepFilterNet not available. Install: pip install deepfilternet")

    if on_progress:
        on_progress(30, "Denoising with DeepFilterNet...")

    import shutil
    df_bin = shutil.which("deepFilter")
    if not df_bin:
        raise RuntimeError("deepFilter CLI not found in PATH after install")

    out_dir = os.path.dirname(output_path)
    cmd = [df_bin, input_path, "-o", out_dir]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"DeepFilterNet failed: {result.stderr[-300:]}")

    # DeepFilterNet writes to output dir with same basename
    expected = os.path.join(out_dir, os.path.basename(input_path))
    if os.path.isfile(expected) and expected != output_path:
        os.replace(expected, output_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
_VIDEO_EXTS = frozenset({".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"})


def enhance_speech(
    audio_path: str,
    output_path: Optional[str] = None,
    mode: str = "full",
    on_progress: Optional[Callable] = None,
) -> EnhanceSpeechResult:
    """
    AI Enhanced Speech -- restore badly recorded dialogue.

    Args:
        audio_path: Path to audio or video file.
        output_path: Explicit output path. Auto-generated if None.
        mode: "denoise_only", "enhance", or "full" (denoise + enhance + bandwidth extension).
        on_progress: Progress callback(pct, msg).

    Returns:
        EnhanceSpeechResult with output path and stats.
    """
    valid_modes = ("denoise_only", "enhance", "full")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Input file not found: {audio_path}")

    if on_progress:
        on_progress(5, "Probing audio...")

    # Determine if we need to extract audio from video
    ext = os.path.splitext(audio_path)[1].lower()
    is_video = ext in _VIDEO_EXTS

    # Probe original stats
    original_stats = _probe_audio_stats(audio_path)

    if on_progress:
        on_progress(10, f"Original sample rate: {original_stats.sample_rate} Hz")

    # Build output path
    if output_path is None:
        suffix = f"enhanced_{mode}"
        output_path = _output_path(audio_path, suffix, "")
        # Force .wav extension for audio output
        base, _ = os.path.splitext(output_path)
        output_path = base + ".wav"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    tmp_files = []
    try:
        # Extract audio from video if needed
        work_path = audio_path
        if is_video:
            if on_progress:
                on_progress(12, "Extracting audio from video...")
            fd, work_path = tempfile.mkstemp(suffix=".wav", prefix="opencut_espeech_")
            os.close(fd)
            tmp_files.append(work_path)
            _extract_audio_wav(audio_path, work_path)

        # Determine denoise/enhance flags from mode
        do_denoise = mode in ("denoise_only", "full")
        do_enhance = mode in ("enhance", "full")
        do_bw_extend = mode == "full" and original_stats.sample_rate > 0 and original_stats.sample_rate < 16000

        # Try Resemble Enhance first
        resemble_available = ensure_package("resemble_enhance", pip_name="resemble-enhance")

        fd_out, tmp_enhanced = tempfile.mkstemp(suffix=".wav", prefix="opencut_espeech_enh_")
        os.close(fd_out)
        tmp_files.append(tmp_enhanced)

        if resemble_available:
            if on_progress:
                on_progress(20, "Using Resemble Enhance backend...")
            _enhance_resemble(work_path, tmp_enhanced, denoise=do_denoise,
                              enhance=do_enhance, on_progress=on_progress)
        else:
            # Fallback: DeepFilterNet for denoise + FFmpeg for bandwidth
            if on_progress:
                on_progress(20, "Resemble unavailable, using fallback pipeline...")

            if do_denoise:
                _denoise_deepfilternet(work_path, tmp_enhanced, on_progress=on_progress)
                work_after_denoise = tmp_enhanced
            else:
                work_after_denoise = work_path

            if do_enhance or do_bw_extend:
                if on_progress:
                    on_progress(55, "Bandwidth extension via FFmpeg...")
                fd2, tmp_bw = tempfile.mkstemp(suffix=".wav", prefix="opencut_espeech_bw_")
                os.close(fd2)
                tmp_files.append(tmp_bw)
                _bandwidth_extend_ffmpeg(work_after_denoise, tmp_bw, target_sr=48000)
                # Use bandwidth-extended version as the enhanced output
                if os.path.isfile(tmp_bw):
                    os.replace(tmp_bw, tmp_enhanced)
                    if tmp_bw in tmp_files:
                        tmp_files.remove(tmp_bw)
            elif not do_denoise:
                # Nothing to do -- copy source
                import shutil
                shutil.copy2(work_path, tmp_enhanced)

        # Bandwidth extension if needed (for Resemble path too)
        if do_bw_extend and resemble_available:
            if on_progress:
                on_progress(75, "Applying bandwidth extension to 48kHz...")
            fd3, tmp_bw2 = tempfile.mkstemp(suffix=".wav", prefix="opencut_espeech_bw2_")
            os.close(fd3)
            tmp_files.append(tmp_bw2)
            _bandwidth_extend_ffmpeg(tmp_enhanced, tmp_bw2, target_sr=48000)
            os.replace(tmp_bw2, tmp_enhanced)
            if tmp_bw2 in tmp_files:
                tmp_files.remove(tmp_bw2)

        # Normalize to -16 LUFS
        if on_progress:
            on_progress(85, "Normalizing to -16 LUFS...")
        _normalize_lufs(tmp_enhanced, output_path, target_lufs=-16.0)

        # Probe enhanced stats
        enhanced_stats = _probe_audio_stats(output_path)

        if on_progress:
            on_progress(100, f"Speech enhanced ({mode})")

        return EnhanceSpeechResult(
            output_path=output_path,
            mode=mode,
            original_stats={
                "sample_rate": original_stats.sample_rate,
                "snr_estimate": original_stats.snr_estimate,
            },
            enhanced_stats={
                "sample_rate": enhanced_stats.sample_rate,
                "snr_estimate": enhanced_stats.snr_estimate,
            },
        )

    finally:
        for tmp in tmp_files:
            try:
                if os.path.isfile(tmp):
                    os.unlink(tmp)
            except OSError:
                pass
