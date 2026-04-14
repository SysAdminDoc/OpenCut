"""
OpenCut VR/360 Video Stabilization Module

Stabilize 360/VR video using gyroscope metadata or visual feature tracking:
- Parse gyro metadata from GoPro GPMF / Insta360 streams
- Apply inverse rotation in equirectangular space via FFmpeg v360 filter
- Fallback: visual feature tracking for metadata-free footage

All via FFmpeg + optional OpenCV -- no ML dependencies.
"""

import json
import logging
import math
import os
import struct
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from opencut.helpers import (
    FFmpegCmd,
    get_ffprobe_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class GyroSample:
    """A single gyroscope reading."""
    timestamp: float  # seconds
    yaw: float        # degrees
    pitch: float      # degrees
    roll: float       # degrees


@dataclass
class VRStabilizeResult:
    """Result of VR stabilization."""
    output_path: str = ""
    method: str = "visual"           # "gyro_gpmf", "gyro_insta360", "visual"
    gyro_samples: int = 0
    smoothing: int = 10
    frames_processed: int = 0
    camera_model: str = ""


# ---------------------------------------------------------------------------
# Gyro metadata parsing
# ---------------------------------------------------------------------------
GPMF_GYRO_FOURCC = b"GYRO"
GPMF_ACCL_FOURCC = b"ACCL"


def _parse_gpmf_gyro(video_path: str) -> List[GyroSample]:
    """
    Extract gyroscope data from GoPro GPMF metadata stream.

    Uses ffprobe to find the GPMF data stream and extracts
    rotation samples at the embedded sample rate.
    """
    import subprocess

    ffprobe = get_ffprobe_path()
    # Find GPMF data stream
    cmd = [
        ffprobe, "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_entries",
        "stream=index,codec_tag_string,codec_type",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []
        probe = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return []

    # Look for GPMF or data stream
    gpmf_index = None
    for stream in probe.get("streams", []):
        tag = stream.get("codec_tag_string", "").lower()
        codec_type = stream.get("codec_type", "")
        if "gpmf" in tag or (codec_type == "data" and "gpmd" in tag):
            gpmf_index = stream.get("index")
            break

    if gpmf_index is None:
        return []

    # Extract raw data stream
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False, prefix="oc_gpmf_")
    tmp_path = tmp.name
    tmp.close()

    try:
        extract_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-map", f"0:{gpmf_index}",
            "-c", "copy", "-f", "rawvideo",
            tmp_path,
        ]
        run_ffmpeg(extract_cmd, timeout=60)

        if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
            return []

        return _decode_gpmf_gyro_data(tmp_path)
    except Exception:
        logger.debug("Failed to extract GPMF gyro data", exc_info=True)
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _decode_gpmf_gyro_data(raw_path: str) -> List[GyroSample]:
    """Decode GPMF binary data to extract gyro samples."""
    samples = []
    try:
        with open(raw_path, "rb") as f:
            data = f.read()
    except OSError:
        return []

    # GPMF uses a KLV-like structure: 4-byte key, 1-byte type,
    # 1-byte struct_size, 2-byte repeat count, then payload
    offset = 0
    timestamp = 0.0
    sample_rate = 200.0  # typical GoPro gyro rate

    while offset + 8 <= len(data):
        key = data[offset:offset + 4]
        if len(key) < 4:
            break

        try:
            type_byte = data[offset + 4]
            struct_size = data[offset + 5]
            repeat = struct.unpack(">H", data[offset + 6:offset + 8])[0]
        except (struct.error, IndexError):
            break

        payload_size = struct_size * repeat
        # Align to 4 bytes
        aligned_size = (payload_size + 3) & ~3
        payload_start = offset + 8
        payload_end = payload_start + payload_size

        if key == GPMF_GYRO_FOURCC and struct_size >= 6 and type_byte == ord("s"):
            # Each sample: 3 x int16 (yaw_rate, pitch_rate, roll_rate) in deg/s
            for i in range(repeat):
                sample_offset = payload_start + i * struct_size
                if sample_offset + 6 <= len(data):
                    try:
                        yaw_rate, pitch_rate, roll_rate = struct.unpack(
                            ">hhh", data[sample_offset:sample_offset + 6]
                        )
                        # Integrate angular rates
                        dt = 1.0 / sample_rate
                        timestamp += dt
                        samples.append(GyroSample(
                            timestamp=timestamp,
                            yaw=yaw_rate / 32768.0 * 2000.0 * dt,
                            pitch=pitch_rate / 32768.0 * 2000.0 * dt,
                            roll=roll_rate / 32768.0 * 2000.0 * dt,
                        ))
                    except struct.error:
                        continue

        offset = payload_start + aligned_size

    return samples


def _parse_insta360_gyro(video_path: str) -> List[GyroSample]:
    """
    Extract gyroscope data from Insta360 metadata.

    Insta360 cameras store gyro data in a trailer at the end of MP4 files
    with a specific magic marker.
    """
    samples = []
    INSTA360_MAGIC = b"8db42d694ccc418790edff439fe026bf"

    try:
        file_size = os.path.getsize(video_path)
        # Insta360 trailer is typically in the last 32MB
        read_size = min(file_size, 32 * 1024 * 1024)

        with open(video_path, "rb") as f:
            f.seek(max(0, file_size - read_size))
            trailer = f.read()
    except OSError:
        return []

    # Look for Insta360 gyro marker
    magic_hex = INSTA360_MAGIC
    magic_pos = trailer.find(magic_hex)
    if magic_pos < 0:
        # Try binary form
        try:
            magic_bytes = bytes.fromhex(magic_hex.decode("ascii"))
            magic_pos = trailer.find(magic_bytes)
        except (ValueError, UnicodeDecodeError):
            return []

    if magic_pos < 0:
        return []

    # Parse gyro records after magic marker
    # Format: each record is 6 floats (timestamp, gyro_x, gyro_y, gyro_z, accel_x, accel_y)
    data_start = magic_pos + len(magic_hex)
    record_size = 24  # 6 x 4-byte floats
    timestamp = 0.0
    dt = 1.0 / 200.0  # typical 200Hz

    offset = data_start
    while offset + record_size <= len(trailer):
        try:
            values = struct.unpack("<6f", trailer[offset:offset + record_size])
            gyro_x, gyro_y, gyro_z = values[1], values[2], values[3]

            # Convert angular velocity (rad/s) to degrees
            timestamp += dt
            samples.append(GyroSample(
                timestamp=timestamp,
                yaw=math.degrees(gyro_z) * dt,
                pitch=math.degrees(gyro_x) * dt,
                roll=math.degrees(gyro_y) * dt,
            ))
        except struct.error:
            break

        offset += record_size
        if len(samples) >= 100000:  # safety limit
            break

    return samples


def detect_gyro_source(video_path: str) -> Tuple[str, List[GyroSample]]:
    """
    Detect and extract gyroscope data from a video file.

    Tries GoPro GPMF first, then Insta360 format.

    Returns:
        Tuple of (source_type, samples) where source_type is
        "gpmf", "insta360", or "none".
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Try GoPro GPMF
    samples = _parse_gpmf_gyro(video_path)
    if samples:
        return "gpmf", samples

    # Try Insta360
    samples = _parse_insta360_gyro(video_path)
    if samples:
        return "insta360", samples

    return "none", []


# ---------------------------------------------------------------------------
# Gyro smoothing and inverse rotation
# ---------------------------------------------------------------------------
def _smooth_gyro(
    samples: List[GyroSample],
    smoothing: int = 10,
) -> List[GyroSample]:
    """
    Apply a moving average filter to gyro samples for smooth stabilization.

    Args:
        samples: Raw gyroscope samples.
        smoothing: Window size for the moving average.

    Returns:
        Smoothed gyro samples.
    """
    if not samples or smoothing < 2:
        return samples

    n = len(samples)
    half = smoothing // 2
    smoothed = []

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        count = hi - lo
        avg_yaw = sum(s.yaw for s in samples[lo:hi]) / count
        avg_pitch = sum(s.pitch for s in samples[lo:hi]) / count
        avg_roll = sum(s.roll for s in samples[lo:hi]) / count

        smoothed.append(GyroSample(
            timestamp=samples[i].timestamp,
            yaw=avg_yaw,
            pitch=avg_pitch,
            roll=avg_roll,
        ))

    return smoothed


def _gyro_to_v360_script(
    samples: List[GyroSample],
    fps: float,
    duration: float,
    smoothing: int = 10,
) -> str:
    """
    Convert gyro samples into an FFmpeg v360 filter expression with
    per-frame yaw/pitch/roll corrections.

    Returns a complex filtergraph string.
    """
    smoothed = _smooth_gyro(samples, smoothing)
    if not smoothed:
        return "v360=e:e"

    # Accumulate rotation offsets
    cumulative_yaw = 0.0
    cumulative_pitch = 0.0
    cumulative_roll = 0.0
    frame_corrections = []

    total_frames = int(duration * fps)
    sample_idx = 0

    for frame_num in range(total_frames):
        frame_time = frame_num / fps

        # Find the nearest gyro sample
        while sample_idx < len(smoothed) - 1 and smoothed[sample_idx].timestamp < frame_time:
            sample_idx += 1

        if sample_idx < len(smoothed):
            s = smoothed[sample_idx]
            cumulative_yaw += s.yaw
            cumulative_pitch += s.pitch
            cumulative_roll += s.roll

        # Inverse rotation to stabilize
        frame_corrections.append((
            -cumulative_yaw,
            -cumulative_pitch,
            -cumulative_roll,
        ))

    if not frame_corrections:
        return "v360=e:e"

    # Use average correction as a static rotation (FFmpeg v360 doesn't
    # support per-frame dynamic yaw/pitch natively, so we apply the
    # average correction and use vidstab for residual)
    avg_yaw = sum(c[0] for c in frame_corrections) / len(frame_corrections)
    avg_pitch = sum(c[1] for c in frame_corrections) / len(frame_corrections)
    avg_roll = sum(c[2] for c in frame_corrections) / len(frame_corrections)

    return (
        f"v360=e:e:yaw={avg_yaw:.4f}:pitch={avg_pitch:.4f}:roll={avg_roll:.4f}"
    )


# ---------------------------------------------------------------------------
# Visual feature-tracking fallback
# ---------------------------------------------------------------------------
def _visual_stabilize_360(
    video_path: str,
    output_path_str: str,
    smoothing: int = 10,
    on_progress: Optional[Callable] = None,
) -> VRStabilizeResult:
    """
    Stabilize 360 video using FFmpeg vidstab (visual feature tracking).

    Fallback when no gyroscope metadata is available.
    """
    transforms_file = os.path.join(
        tempfile.gettempdir(), f"oc_vrstab_{os.getpid()}.trf"
    )

    if on_progress:
        on_progress(15, "Pass 1: Analysing 360 video motion (visual tracking)...")

    # Pass 1: Detect motion
    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", f"vidstabdetect=result={transforms_file}:shakiness=8:accuracy=15",
        "-f", "null", os.devnull,
    ], timeout=7200)

    if on_progress:
        on_progress(55, "Pass 2: Applying 360 stabilisation (visual tracking)...")

    # Pass 2: Apply stabilisation
    vf = (
        f"vidstabtransform=input={transforms_file}"
        f":smoothing={smoothing}:interpol=bicubic:optzoom=0"
    )

    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        output_path_str,
    ], timeout=7200)

    try:
        os.unlink(transforms_file)
    except OSError:
        pass

    info = get_video_info(video_path)
    frames = int(info.get("duration", 0) * info.get("fps", 30))

    return VRStabilizeResult(
        output_path=output_path_str,
        method="visual",
        gyro_samples=0,
        smoothing=smoothing,
        frames_processed=frames,
    )


# ---------------------------------------------------------------------------
# Main stabilization entry point
# ---------------------------------------------------------------------------
def stabilize_vr(
    video_path: str,
    output_path_override: Optional[str] = None,
    output_dir: str = "",
    smoothing: int = 10,
    force_visual: bool = False,
    on_progress: Optional[Callable] = None,
) -> VRStabilizeResult:
    """
    Stabilize a 360/VR video using gyro metadata or visual feature tracking.

    Attempts to parse gyroscope data from GoPro GPMF or Insta360 metadata.
    If no gyro data is found (or force_visual=True), falls back to
    FFmpeg vidstab-based visual feature tracking.

    Args:
        video_path: Path to input 360 video.
        output_path_override: Output file path. Auto-generated if None.
        output_dir: Output directory.
        smoothing: Stabilization smoothing strength (1-30).
        force_visual: If True, skip gyro detection and use visual tracking.
        on_progress: Progress callback(pct, msg).

    Returns:
        VRStabilizeResult with output path and metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    smoothing = max(1, min(30, int(smoothing)))
    out = output_path_override or output_path(video_path, "vr_stabilized", output_dir)

    if on_progress:
        on_progress(5, "Detecting gyroscope metadata...")

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    duration = info.get("duration", 0.0)

    gyro_source = "none"
    gyro_samples: List[GyroSample] = []

    if not force_visual:
        gyro_source, gyro_samples = detect_gyro_source(video_path)

    if gyro_samples and gyro_source != "none":
        if on_progress:
            on_progress(10, f"Found {len(gyro_samples)} gyro samples ({gyro_source})")

        # Generate v360 correction filter
        v360_expr = _gyro_to_v360_script(gyro_samples, fps, duration, smoothing)

        if on_progress:
            on_progress(20, "Applying gyro-based 360 stabilization...")

        # Apply v360 rotation correction + vidstab for residual motion
        transforms_file = os.path.join(
            tempfile.gettempdir(), f"oc_vrstab_gyro_{os.getpid()}.trf"
        )

        # Pass 1: detect residual motion after v360 correction
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-vf", f"{v360_expr},vidstabdetect=result={transforms_file}:shakiness=4",
            "-f", "null", os.devnull,
        ], timeout=7200)

        if on_progress:
            on_progress(60, "Applying residual stabilization...")

        # Pass 2: apply both corrections
        vf = (
            f"{v360_expr},"
            f"vidstabtransform=input={transforms_file}"
            f":smoothing={smoothing}:interpol=bicubic:optzoom=0"
        )

        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            out,
        ], timeout=7200)

        try:
            os.unlink(transforms_file)
        except OSError:
            pass

        if on_progress:
            on_progress(100, "VR stabilization complete (gyro + visual)!")

        method_name = f"gyro_{gyro_source}"
        frames = int(duration * fps)

        return VRStabilizeResult(
            output_path=out,
            method=method_name,
            gyro_samples=len(gyro_samples),
            smoothing=smoothing,
            frames_processed=frames,
            camera_model=gyro_source,
        )
    else:
        if on_progress:
            on_progress(10, "No gyro metadata found, using visual feature tracking...")

        result = _visual_stabilize_360(video_path, out, smoothing, on_progress)

        if on_progress:
            on_progress(100, "VR stabilization complete (visual tracking)!")

        return result
