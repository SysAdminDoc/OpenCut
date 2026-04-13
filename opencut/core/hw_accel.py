"""
OpenCut Hardware-Accelerated Encoding Module v1.0.0

Detects available GPU encoders (NVENC, QSV, AMF, VideoToolbox),
verifies they actually work via synthetic test encodes, and provides
hardware-accelerated encoding with automatic fallback to software.

Priority: nvenc > qsv > amf > videotoolbox > software
"""

import logging
import os
import subprocess as _sp
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import FFmpegCmd, get_ffmpeg_path, output_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HWEncoder:
    """A single hardware encoder detected on the system."""
    name: str               # e.g. "h264_nvenc"
    codec: str              # e.g. "h264", "hevc", "av1"
    hw_type: str            # "nvenc", "qsv", "amf", "videotoolbox"
    supports_h264: bool = False
    supports_hevc: bool = False
    supports_av1: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HWEncoderInfo:
    """Result of hardware encoder detection."""
    available_encoders: List[HWEncoder] = field(default_factory=list)
    preferred_h264: Optional[str] = None    # encoder name e.g. "h264_nvenc"
    preferred_hevc: Optional[str] = None
    preferred_av1: Optional[str] = None
    gpu_name: Optional[str] = None
    detection_time: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "available_encoders": [e.to_dict() for e in self.available_encoders],
            "preferred_h264": self.preferred_h264,
            "preferred_hevc": self.preferred_hevc,
            "preferred_av1": self.preferred_av1,
            "gpu_name": self.gpu_name,
            "detection_time": round(self.detection_time, 2),
            "has_hw_accel": len(self.available_encoders) > 0,
        }


# ---------------------------------------------------------------------------
# Known encoder definitions
# ---------------------------------------------------------------------------

# Encoder name -> (codec, hw_type)
_KNOWN_HW_ENCODERS = {
    "h264_nvenc":         ("h264",  "nvenc"),
    "hevc_nvenc":         ("hevc",  "nvenc"),
    "av1_nvenc":          ("av1",   "nvenc"),
    "h264_qsv":           ("h264",  "qsv"),
    "hevc_qsv":           ("hevc",  "qsv"),
    "av1_qsv":            ("av1",   "qsv"),
    "h264_amf":           ("h264",  "amf"),
    "hevc_amf":           ("hevc",  "amf"),
    "av1_amf":            ("av1",   "amf"),
    "h264_videotoolbox":  ("h264",  "videotoolbox"),
    "hevc_videotoolbox":  ("hevc",  "videotoolbox"),
}

# Priority order for hardware types (lower index = higher priority)
_HW_PRIORITY = ["nvenc", "qsv", "amf", "videotoolbox"]

# Quality presets per hardware type
_QUALITY_PRESETS = {
    "nvenc": {
        "speed":    ["-preset", "p1", "-rc", "vbr", "-cq", "28", "-b:v", "0"],
        "balanced": ["-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"],
        "quality":  ["-preset", "p7", "-rc", "vbr", "-cq", "18", "-b:v", "0"],
    },
    "qsv": {
        "speed":    ["-preset", "veryfast", "-global_quality", "28"],
        "balanced": ["-preset", "medium", "-global_quality", "23"],
        "quality":  ["-preset", "veryslow", "-global_quality", "18"],
    },
    "amf": {
        "speed":    ["-quality", "speed", "-rc", "vbr_latency", "-qp_i", "28", "-qp_p", "28"],
        "balanced": ["-quality", "balanced", "-rc", "vbr_latency", "-qp_i", "23", "-qp_p", "23"],
        "quality":  ["-quality", "quality", "-rc", "vbr_latency", "-qp_i", "18", "-qp_p", "18"],
    },
    "videotoolbox": {
        "speed":    ["-q:v", "65", "-allow_sw", "1"],
        "balanced": ["-q:v", "50", "-allow_sw", "1"],
        "quality":  ["-q:v", "35", "-allow_sw", "1"],
    },
}

# Software fallback presets (libx264/libx265/libsvtav1)
_SOFTWARE_PRESETS = {
    "h264": {
        "speed":    ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"],
        "balanced": ["-c:v", "libx264", "-preset", "medium", "-crf", "23"],
        "quality":  ["-c:v", "libx264", "-preset", "slow", "-crf", "18"],
    },
    "hevc": {
        "speed":    ["-c:v", "libx265", "-preset", "ultrafast", "-crf", "28"],
        "balanced": ["-c:v", "libx265", "-preset", "medium", "-crf", "23"],
        "quality":  ["-c:v", "libx265", "-preset", "slow", "-crf", "18"],
    },
    "av1": {
        "speed":    ["-c:v", "libsvtav1", "-preset", "10", "-crf", "35"],
        "balanced": ["-c:v", "libsvtav1", "-preset", "6", "-crf", "28"],
        "quality":  ["-c:v", "libsvtav1", "-preset", "3", "-crf", "22"],
    },
}

# HW encoding presets for the preset system
HW_ENCODING_PRESETS = {
    "h264_hw_fast": {
        "label": "H.264 HW Accelerated (Fast)",
        "description": "GPU-accelerated H.264 encoding optimized for speed",
        "category": "hw_accel",
        "width": 1920, "height": 1080,
        "codec": "h264", "quality": "speed", "hw_type": "auto",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
        "hw_accel": True,
    },
    "h264_hw_quality": {
        "label": "H.264 HW Accelerated (Quality)",
        "description": "GPU-accelerated H.264 encoding optimized for quality",
        "category": "hw_accel",
        "width": 1920, "height": 1080,
        "codec": "h264", "quality": "quality", "hw_type": "auto",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
        "hw_accel": True,
    },
    "hevc_hw_fast": {
        "label": "HEVC HW Accelerated (Fast)",
        "description": "GPU-accelerated H.265/HEVC encoding optimized for speed",
        "category": "hw_accel",
        "width": 1920, "height": 1080,
        "codec": "hevc", "quality": "speed", "hw_type": "auto",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
        "hw_accel": True,
    },
    "hevc_hw_quality": {
        "label": "HEVC HW Accelerated (Quality)",
        "description": "GPU-accelerated H.265/HEVC encoding optimized for quality",
        "category": "hw_accel",
        "width": 1920, "height": 1080,
        "codec": "hevc", "quality": "quality", "hw_type": "auto",
        "audio_codec": "aac", "audio_bitrate": "192k",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
        "hw_accel": True,
    },
    "av1_hw": {
        "label": "AV1 HW Accelerated",
        "description": "GPU-accelerated AV1 encoding (NVENC/QSV/AMF)",
        "category": "hw_accel",
        "width": 1920, "height": 1080,
        "codec": "av1", "quality": "balanced", "hw_type": "auto",
        "audio_codec": "libopus", "audio_bitrate": "128k",
        "pix_fmt": "yuv420p",
        "ext": ".mp4",
        "hw_accel": True,
    },
}


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()
_cached_info: Optional[HWEncoderInfo] = None


# ---------------------------------------------------------------------------
# GPU Name Detection
# ---------------------------------------------------------------------------

def _detect_gpu_name() -> Optional[str]:
    """Try to detect the GPU name using platform-specific tools."""
    # Try nvidia-smi first (NVIDIA GPUs)
    try:
        result = _sp.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0].strip()
    except (FileNotFoundError, _sp.TimeoutExpired, OSError):
        pass

    # Try Intel GPU detection via vainfo (Linux) or system_profiler (macOS)
    import platform
    if platform.system() == "Darwin":
        try:
            result = _sp.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Chipset Model:" in line:
                        return line.split(":", 1)[1].strip()
        except (FileNotFoundError, _sp.TimeoutExpired, OSError):
            pass
    elif platform.system() == "Windows":
        try:
            result = _sp.run(
                ["wmic", "path", "win32_videocontroller", "get", "name"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                lines = [ln.strip() for ln in result.stdout.strip().split("\n") if ln.strip() and ln.strip() != "Name"]
                if lines:
                    return lines[0]
        except (FileNotFoundError, _sp.TimeoutExpired, OSError):
            pass

    return None


# ---------------------------------------------------------------------------
# Encoder Detection
# ---------------------------------------------------------------------------

def _parse_encoders_output(output: str) -> List[str]:
    """Parse `ffmpeg -encoders` output and return names of known HW encoders found."""
    found = []
    for line in output.split("\n"):
        line = line.strip()
        # Lines look like: " V..... h264_nvenc  NVIDIA NVENC H.264 encoder (codec h264)"
        # The encoder name is the second token after the flags
        parts = line.split()
        if len(parts) >= 2:
            # The flags field is 6 chars like "V....." or "V.S..."
            # The encoder name follows
            candidate = parts[1] if len(parts[0]) == 6 and parts[0][0] == "V" else None
            if candidate is None and len(parts) > 2:
                # Sometimes the format puts flags and name differently
                for p in parts:
                    if p in _KNOWN_HW_ENCODERS:
                        candidate = p
                        break
            if candidate and candidate in _KNOWN_HW_ENCODERS:
                if candidate not in found:
                    found.append(candidate)
    return found


def _test_encoder(encoder_name: str) -> bool:
    """
    Test if an encoder actually works by running a tiny synthetic encode.

    Some systems report encoders in ffmpeg -encoders that fail at runtime
    (e.g. NVENC without a GPU, QSV without proper driver setup).
    """
    ffmpeg = get_ffmpeg_path()
    tmp_out = None
    try:
        tmp_fd, tmp_out = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)

        # Generate 1 frame of black video and encode it
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.04:r=25",
            "-c:v", encoder_name,
            "-frames:v", "1",
        ]

        # Add minimal encoder-specific options to avoid errors
        codec, hw_type = _KNOWN_HW_ENCODERS[encoder_name]
        if hw_type == "nvenc":
            cmd.extend(["-preset", "p1"])
        elif hw_type == "qsv":
            cmd.extend(["-preset", "veryfast"])
        elif hw_type == "amf":
            cmd.extend(["-quality", "speed"])

        cmd.append(tmp_out)

        result = _sp.run(cmd, capture_output=True, timeout=15)
        return result.returncode == 0
    except (OSError, _sp.TimeoutExpired, Exception) as e:
        logger.debug("Encoder test failed for %s: %s", encoder_name, e)
        return False
    finally:
        if tmp_out and os.path.exists(tmp_out):
            try:
                os.unlink(tmp_out)
            except OSError:
                pass


def detect_hw_encoders(force_refresh: bool = False) -> HWEncoderInfo:
    """
    Detect available hardware encoders on this system.

    Runs `ffmpeg -encoders`, filters for known HW encoders, then verifies
    each one actually works with a synthetic test encode. Results are
    cached until explicitly refreshed.

    Args:
        force_refresh: If True, bypass cache and re-detect.

    Returns:
        HWEncoderInfo with all detected and verified encoders.
    """
    global _cached_info

    with _cache_lock:
        if _cached_info is not None and not force_refresh:
            return _cached_info

    start = time.time()
    info = HWEncoderInfo(timestamp=time.time())

    # Detect GPU name (non-blocking, best effort)
    info.gpu_name = _detect_gpu_name()

    # Query FFmpeg for available encoders
    ffmpeg = get_ffmpeg_path()
    try:
        result = _sp.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg -encoders failed (rc=%d)", result.returncode)
            info.detection_time = time.time() - start
            with _cache_lock:
                _cached_info = info
            return info

        reported_encoders = _parse_encoders_output(result.stdout)
    except (OSError, _sp.TimeoutExpired) as e:
        logger.warning("Failed to query ffmpeg encoders: %s", e)
        info.detection_time = time.time() - start
        with _cache_lock:
            _cached_info = info
        return info

    # Verify each reported encoder actually works
    verified_encoders: List[HWEncoder] = []
    for enc_name in reported_encoders:
        codec, hw_type = _KNOWN_HW_ENCODERS[enc_name]
        logger.debug("Testing encoder %s ...", enc_name)
        if _test_encoder(enc_name):
            encoder = HWEncoder(
                name=enc_name,
                codec=codec,
                hw_type=hw_type,
                supports_h264=(codec == "h264"),
                supports_hevc=(codec == "hevc"),
                supports_av1=(codec == "av1"),
            )
            verified_encoders.append(encoder)
            logger.info("HW encoder verified: %s (%s)", enc_name, hw_type)
        else:
            logger.debug("Encoder %s reported but test encode failed, skipping", enc_name)

    info.available_encoders = verified_encoders

    # Determine preferred encoder for each codec, respecting priority
    info.preferred_h264 = _pick_preferred(verified_encoders, "h264")
    info.preferred_hevc = _pick_preferred(verified_encoders, "hevc")
    info.preferred_av1 = _pick_preferred(verified_encoders, "av1")

    info.detection_time = time.time() - start
    logger.info(
        "HW encoder detection complete in %.1fs: %d encoders found "
        "(h264=%s, hevc=%s, av1=%s, gpu=%s)",
        info.detection_time, len(verified_encoders),
        info.preferred_h264 or "software",
        info.preferred_hevc or "software",
        info.preferred_av1 or "software",
        info.gpu_name or "unknown",
    )

    with _cache_lock:
        _cached_info = info
    return info


def _pick_preferred(encoders: List[HWEncoder], codec: str) -> Optional[str]:
    """Pick the best encoder for a codec based on hardware priority."""
    candidates = [e for e in encoders if e.codec == codec]
    if not candidates:
        return None

    # Sort by hardware priority
    def _priority_key(enc: HWEncoder) -> int:
        try:
            return _HW_PRIORITY.index(enc.hw_type)
        except ValueError:
            return 999

    candidates.sort(key=_priority_key)
    return candidates[0].name


# ---------------------------------------------------------------------------
# Get Encoding Arguments
# ---------------------------------------------------------------------------

def get_hw_encode_args(
    codec: str = "h264",
    quality: str = "balanced",
    hw_type: str = "auto",
) -> List[str]:
    """
    Get FFmpeg arguments for hardware-accelerated encoding.

    Args:
        codec: Target codec - "h264", "hevc", or "av1".
        quality: Quality preset - "speed", "balanced", or "quality".
        hw_type: Hardware type - "auto", "nvenc", "qsv", "amf",
                 "videotoolbox", or "software".

    Returns:
        List of FFmpeg arguments (e.g. ["-c:v", "h264_nvenc", "-preset", "p4", ...]).
        Falls back to software encoding if no HW encoder is available.
    """
    if codec not in ("h264", "hevc", "av1"):
        codec = "h264"
    if quality not in ("speed", "balanced", "quality"):
        quality = "balanced"

    # Software fallback
    if hw_type == "software":
        return list(_SOFTWARE_PRESETS[codec][quality])

    # Detect available encoders
    info = detect_hw_encoders()

    # Pick the encoder
    encoder_name = None
    if hw_type == "auto":
        # Use the preferred encoder for this codec
        if codec == "h264":
            encoder_name = info.preferred_h264
        elif codec == "hevc":
            encoder_name = info.preferred_hevc
        elif codec == "av1":
            encoder_name = info.preferred_av1
    else:
        # User requested a specific HW type
        if hw_type in _HW_PRIORITY:
            target_name = f"{codec}_{hw_type}"
            for enc in info.available_encoders:
                if enc.name == target_name:
                    encoder_name = target_name
                    break

    # No HW encoder available -- fall back to software
    if encoder_name is None:
        logger.info(
            "No HW encoder available for codec=%s hw_type=%s, falling back to software",
            codec, hw_type,
        )
        return list(_SOFTWARE_PRESETS[codec][quality])

    # Build the argument list
    _, actual_hw_type = _KNOWN_HW_ENCODERS[encoder_name]
    args = ["-c:v", encoder_name]

    # Add quality preset args for this HW type
    preset_args = _QUALITY_PRESETS.get(actual_hw_type, {}).get(quality)
    if preset_args:
        args.extend(preset_args)

    return args


# ---------------------------------------------------------------------------
# Hardware Encode Function
# ---------------------------------------------------------------------------

def hw_encode(
    input_path: str,
    output_path_override: Optional[str] = None,
    codec: str = "h264",
    quality: str = "balanced",
    hw_type: str = "auto",
    extra_args: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Encode a video using hardware acceleration with automatic fallback.

    Args:
        input_path: Path to input video file.
        output_path_override: Explicit output path. If None, auto-generated
            from input_path with a suffix.
        codec: Target codec - "h264", "hevc", or "av1".
        quality: Quality preset - "speed", "balanced", or "quality".
        hw_type: Hardware type - "auto", "nvenc", "qsv", "amf",
                 "videotoolbox", or "software".
        extra_args: Additional FFmpeg arguments to append.
        on_progress: Callback(percent, message) for progress updates.

    Returns:
        dict with keys: output_path, encoder_used, encode_time_seconds,
        hw_accelerated, file_size_mb.
    """
    if codec not in ("h264", "hevc", "av1"):
        codec = "h264"
    if quality not in ("speed", "balanced", "quality"):
        quality = "balanced"

    # Determine output path
    if output_path_override:
        out = output_path_override
    else:
        suffix = f"hw_{codec}_{quality}"
        out = output_path(input_path, suffix)

    if on_progress:
        on_progress(5, "Detecting hardware encoders...")

    # Get HW encoding args (includes fallback logic)
    encode_args = get_hw_encode_args(codec=codec, quality=quality, hw_type=hw_type)

    # Determine if we got HW or software encoding
    encoder_used = "unknown"
    hw_accelerated = False
    for i, arg in enumerate(encode_args):
        if arg == "-c:v" and i + 1 < len(encode_args):
            encoder_used = encode_args[i + 1]
            hw_accelerated = encoder_used in _KNOWN_HW_ENCODERS
            break

    if on_progress:
        accel_label = "hardware" if hw_accelerated else "software"
        on_progress(10, f"Encoding with {encoder_used} ({accel_label})...")

    # Build FFmpeg command
    (
        FFmpegCmd()
        .input(input_path)
        .output(out)
    )

    # We'll build manually since we need custom codec args
    ffmpeg = get_ffmpeg_path()
    cmd_list = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
    ]

    # Video encoding args
    cmd_list.extend(encode_args)

    # Pixel format
    cmd_list.extend(["-pix_fmt", "yuv420p"])

    # Audio pass-through or re-encode
    cmd_list.extend(["-c:a", "aac", "-b:a", "192k"])

    # Faststart for MP4
    cmd_list.extend(["-movflags", "+faststart"])

    # Extra args from caller
    if extra_args:
        cmd_list.extend(extra_args)

    cmd_list.append(out)

    # Encode with timing
    if on_progress:
        on_progress(15, "Encoding in progress...")

    start_time = time.time()

    try:
        run_ffmpeg(cmd_list, timeout=7200)
    except RuntimeError as e:
        # If HW encoding failed, try software fallback
        if hw_accelerated:
            logger.warning(
                "HW encoder %s failed, falling back to software: %s",
                encoder_used, e,
            )
            if on_progress:
                on_progress(15, "HW encoder failed, falling back to software...")

            sw_args = _SOFTWARE_PRESETS[codec][quality]
            cmd_list_sw = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", input_path,
            ]
            cmd_list_sw.extend(sw_args)
            cmd_list_sw.extend(["-pix_fmt", "yuv420p"])
            cmd_list_sw.extend(["-c:a", "aac", "-b:a", "192k"])
            cmd_list_sw.extend(["-movflags", "+faststart"])
            if extra_args:
                cmd_list_sw.extend(extra_args)
            cmd_list_sw.append(out)

            encoder_used = sw_args[1]  # codec name after -c:v
            hw_accelerated = False

            run_ffmpeg(cmd_list_sw, timeout=7200)
        else:
            raise

    encode_time = time.time() - start_time

    # Get output file size
    file_size_bytes = os.path.getsize(out) if os.path.exists(out) else 0
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    if on_progress:
        on_progress(100, f"Encode complete ({encoder_used}, {encode_time:.1f}s)")

    return {
        "output_path": out,
        "encoder_used": encoder_used,
        "encode_time_seconds": round(encode_time, 2),
        "hw_accelerated": hw_accelerated,
        "file_size_mb": file_size_mb,
        "codec": codec,
        "quality": quality,
    }


# ---------------------------------------------------------------------------
# Utility: get available HW presets (for route)
# ---------------------------------------------------------------------------

def get_hw_presets() -> List[dict]:
    """Return the list of HW encoding presets with availability info."""
    info = detect_hw_encoders()
    presets = []
    for name, preset in HW_ENCODING_PRESETS.items():
        codec = preset["codec"]
        available = False
        encoder = None
        if codec == "h264" and info.preferred_h264:
            available = True
            encoder = info.preferred_h264
        elif codec == "hevc" and info.preferred_hevc:
            available = True
            encoder = info.preferred_hevc
        elif codec == "av1" and info.preferred_av1:
            available = True
            encoder = info.preferred_av1

        presets.append({
            "name": name,
            "label": preset["label"],
            "description": preset["description"],
            "category": preset["category"],
            "codec": codec,
            "quality": preset.get("quality", "balanced"),
            "hw_available": available,
            "encoder": encoder,
            "fallback": f"lib{'x264' if codec == 'h264' else 'x265' if codec == 'hevc' else 'svtav1'}",
        })
    return presets
