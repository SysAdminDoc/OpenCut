"""
OpenCut AV1 Encoding Module v1.0.0

Provides AV1 encoding with automatic encoder selection:
  - Hardware: av1_nvenc (NVIDIA), av1_qsv (Intel)
  - Software: libsvtav1 (fastest), libaom-av1 (reference/slow)

Quality presets map to encoder-specific speed/quality tradeoffs:
  - speed:    SVT preset 10, fast encode for previews
  - balanced: SVT preset 8, good quality/speed tradeoff (default)
  - quality:  SVT preset 4, slower but higher quality
"""

import logging
import os
import subprocess as _sp
import time
from typing import Callable, Dict, Optional

from opencut.helpers import get_ffmpeg_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Encoder definitions
# ---------------------------------------------------------------------------

AV1_ENCODERS = {
    "libsvtav1": {
        "label": "SVT-AV1 (Software)",
        "type": "software",
        "priority": 3,
        "description": "Fast, production-quality AV1 encoder",
    },
    "libaom-av1": {
        "label": "libaom (Reference, Software)",
        "type": "software",
        "priority": 4,
        "description": "Reference AV1 encoder — slow but highest quality",
    },
    "av1_nvenc": {
        "label": "NVIDIA NVENC AV1 (Hardware)",
        "type": "hardware",
        "priority": 1,
        "description": "Hardware-accelerated AV1 via NVIDIA GPU",
    },
    "av1_qsv": {
        "label": "Intel QSV AV1 (Hardware)",
        "type": "hardware",
        "priority": 2,
        "description": "Hardware-accelerated AV1 via Intel Quick Sync",
    },
}

# Quality presets per encoder
_QUALITY_PRESETS = {
    "libsvtav1": {
        "speed":    {"preset": "10", "crf": "35"},
        "balanced": {"preset": "8",  "crf": "28"},
        "quality":  {"preset": "4",  "crf": "22"},
    },
    "libaom-av1": {
        "speed":    {"cpu-used": "8", "crf": "35"},
        "balanced": {"cpu-used": "6", "crf": "28"},
        "quality":  {"cpu-used": "3", "crf": "22"},
    },
    "av1_nvenc": {
        "speed":    {"preset": "p1", "cq": "35"},
        "balanced": {"preset": "p4", "cq": "28"},
        "quality":  {"preset": "p7", "cq": "22"},
    },
    "av1_qsv": {
        "speed":    {"preset": "veryfast", "global_quality": "35"},
        "balanced": {"preset": "medium",   "global_quality": "28"},
        "quality":  {"preset": "veryslow", "global_quality": "22"},
    },
}

# ---------------------------------------------------------------------------
# Encoder detection
# ---------------------------------------------------------------------------

_detected_encoders: Optional[Dict[str, dict]] = None


def detect_av1_encoders() -> Dict[str, dict]:
    """Detect available AV1 encoders from FFmpeg.

    Returns a dict keyed by encoder name with availability and label info.
    Results are cached after first call.
    """
    global _detected_encoders
    if _detected_encoders is not None:
        return _detected_encoders

    ffmpeg = get_ffmpeg_path()
    available = {}

    try:
        result = _sp.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        stdout = result.stdout if result.returncode == 0 else ""
    except (OSError, _sp.TimeoutExpired):
        stdout = ""

    for enc_name, enc_info in AV1_ENCODERS.items():
        found = enc_name in stdout
        available[enc_name] = {
            "name": enc_name,
            "label": enc_info["label"],
            "type": enc_info["type"],
            "available": found,
            "description": enc_info["description"],
        }

    _detected_encoders = available
    return available


def _pick_best_encoder(preferred: str = "auto") -> Optional[str]:
    """Select the best available AV1 encoder.

    If *preferred* is ``"auto"``, pick by priority:
    HW (nvenc > qsv) then SW (libsvtav1 > libaom-av1).
    Otherwise, try the requested encoder.
    """
    available = detect_av1_encoders()

    if preferred != "auto":
        enc = available.get(preferred)
        if enc and enc["available"]:
            return preferred
        return None

    # Sort by priority and pick first available
    ranked = sorted(
        AV1_ENCODERS.items(),
        key=lambda kv: kv[1]["priority"],
    )
    for name, _ in ranked:
        if available.get(name, {}).get("available"):
            return name
    return None


# ---------------------------------------------------------------------------
# Export function
# ---------------------------------------------------------------------------

def export_av1(
    input_path: str,
    encoder: str = "auto",
    quality: str = "balanced",
    crf: int = 28,
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Export a video file using AV1 encoding.

    Args:
        input_path: Path to the source video file.
        encoder: Encoder name or ``"auto"`` for automatic selection.
            Valid: ``libsvtav1``, ``libaom-av1``, ``av1_nvenc``,
            ``av1_qsv``, ``auto``.
        quality: Quality preset — ``speed``, ``balanced``, ``quality``.
        crf: Constant Rate Factor override (0-63).
        output_path_override: Explicit output path.  If *None*,
            auto-generated from *input_path*.
        on_progress: Optional callback ``(percent, message)``.

    Returns:
        Dict with ``output_path``, ``encoder_used``, ``quality``,
        ``crf``, ``encode_time_seconds``, ``file_size_mb``,
        ``hw_accelerated``.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        RuntimeError: If no AV1 encoder is available.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    quality = quality.lower().strip()
    if quality not in ("speed", "balanced", "quality"):
        quality = "balanced"

    crf = max(0, min(63, int(crf)))

    if on_progress:
        on_progress(5, "Detecting AV1 encoders...")

    chosen = _pick_best_encoder(encoder)
    if chosen is None:
        raise RuntimeError(
            "No AV1 encoder available. Install FFmpeg with libsvtav1 "
            "or libaom-av1, or use a GPU with AV1 hardware encoding."
        )

    hw_accelerated = AV1_ENCODERS[chosen]["type"] == "hardware"

    if on_progress:
        label = AV1_ENCODERS[chosen]["label"]
        on_progress(10, f"Encoding AV1 with {label}...")

    # Determine output path
    if output_path_override:
        out = output_path_override
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = os.path.dirname(input_path)
        out = os.path.join(directory, f"{base}_av1_{quality}.mp4")

    # Build FFmpeg command
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-c:v", chosen,
    ]

    # Apply quality preset parameters
    presets = _QUALITY_PRESETS.get(chosen, {}).get(quality, {})

    if chosen == "libsvtav1":
        cmd += ["-preset", presets.get("preset", "8")]
        cmd += ["-crf", str(crf)]
        cmd += ["-pix_fmt", "yuv420p10le"]
    elif chosen == "libaom-av1":
        cmd += ["-cpu-used", presets.get("cpu-used", "6")]
        cmd += ["-crf", str(crf)]
        cmd += ["-b:v", "0"]
        cmd += ["-pix_fmt", "yuv420p10le"]
        cmd += ["-row-mt", "1"]
        cmd += ["-tiles", "2x2"]
    elif chosen == "av1_nvenc":
        cmd += ["-preset", presets.get("preset", "p4")]
        cmd += ["-cq", presets.get("cq", str(crf))]
        cmd += ["-pix_fmt", "yuv420p"]
    elif chosen == "av1_qsv":
        cmd += ["-preset", presets.get("preset", "medium")]
        cmd += ["-global_quality", presets.get("global_quality", str(crf))]
        cmd += ["-pix_fmt", "yuv420p"]

    # Audio
    cmd += ["-c:a", "libopus", "-b:a", "128k"]

    # Faststart for MP4
    cmd += ["-movflags", "+faststart"]

    cmd.append(out)

    start_time = time.time()
    run_ffmpeg(cmd, timeout=14400)  # AV1 can be slow
    encode_time = time.time() - start_time

    file_size_bytes = os.path.getsize(out) if os.path.exists(out) else 0
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    if on_progress:
        on_progress(100, f"AV1 export complete ({chosen})")

    return {
        "output_path": out,
        "encoder_used": chosen,
        "quality": quality,
        "crf": crf,
        "encode_time_seconds": round(encode_time, 2),
        "file_size_mb": file_size_mb,
        "hw_accelerated": hw_accelerated,
    }
