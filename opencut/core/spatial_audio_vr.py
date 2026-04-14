"""
OpenCut Spatial Audio for VR Module

Map speaker positions to spatial audio and convert to first-order
ambisonics for VR/360 video:
- Face detection to locate speaker positions in equirectangular space
- Map screen positions to azimuth/elevation
- Convert mono/stereo audio to first-order ambisonics (4-channel ACN/SN3D)
- Mux ambisonic audio with equirectangular video

Uses FFmpeg for audio processing and muxing.
"""

import json
import logging
import math
import os
import subprocess
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
class SpeakerPosition:
    """A speaker position in equirectangular space."""
    label: str = ""
    azimuth: float = 0.0     # degrees, -180 to 180 (0 = front)
    elevation: float = 0.0   # degrees, -90 to 90 (0 = horizon)
    distance: float = 1.0    # normalized distance (0.0 to 1.0)
    audio_track: int = 0     # which audio track/channel to spatialize


@dataclass
class SpatialAudioResult:
    """Result of spatial audio processing."""
    output_path: str = ""
    speakers: int = 0
    channels: int = 4          # first-order ambisonics = 4 channels
    format: str = "ambix"      # ACN/SN3D (AmbiX)
    sample_rate: int = 48000
    muxed: bool = False


# ---------------------------------------------------------------------------
# Position mapping
# ---------------------------------------------------------------------------
def pixel_to_spherical(
    x: float, y: float,
    frame_width: int, frame_height: int,
) -> Tuple[float, float]:
    """
    Convert pixel coordinates in equirectangular frame to spherical angles.

    Args:
        x: Horizontal pixel position (0 to frame_width).
        y: Vertical pixel position (0 to frame_height).
        frame_width: Width of equirectangular frame.
        frame_height: Height of equirectangular frame.

    Returns:
        (azimuth, elevation) in degrees.
        Azimuth: -180 to 180 (0 = center/front).
        Elevation: -90 to 90 (0 = horizon).
    """
    azimuth = (x / max(1, frame_width)) * 360.0 - 180.0
    elevation = 90.0 - (y / max(1, frame_height)) * 180.0
    return azimuth, elevation


def spherical_to_pixel(
    azimuth: float, elevation: float,
    frame_width: int, frame_height: int,
) -> Tuple[int, int]:
    """
    Convert spherical angles to pixel coordinates in equirectangular frame.
    """
    x = int(((azimuth + 180.0) / 360.0) * frame_width)
    y = int(((90.0 - elevation) / 180.0) * frame_height)
    return max(0, min(frame_width - 1, x)), max(0, min(frame_height - 1, y))


# ---------------------------------------------------------------------------
# Ambisonics encoding
# ---------------------------------------------------------------------------
def _encode_ambisonics_coefficients(
    azimuth_deg: float,
    elevation_deg: float,
    distance: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Compute first-order ambisonics (FOA) encoding coefficients
    for a source at the given spherical position.

    Uses ACN channel ordering with SN3D normalization (AmbiX format):
    - Channel 0 (W): omnidirectional
    - Channel 1 (Y): left-right (sin azimuth * cos elevation)
    - Channel 2 (Z): up-down (sin elevation)
    - Channel 3 (X): front-back (cos azimuth * cos elevation)

    Args:
        azimuth_deg: Horizontal angle in degrees (-180 to 180).
        elevation_deg: Vertical angle in degrees (-90 to 90).
        distance: Source distance (0-1, affects gain).

    Returns:
        (W, Y, Z, X) coefficients.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    # Distance attenuation (inverse distance, clamped)
    gain = 1.0 / max(0.1, distance) if distance > 0 else 1.0
    gain = min(gain, 10.0)

    # SN3D normalization (AmbiX)
    w = 1.0 * gain                           # omnidirectional
    y = math.sin(az) * math.cos(el) * gain   # left-right
    z = math.sin(el) * gain                  # up-down
    x = math.cos(az) * math.cos(el) * gain   # front-back

    return w, y, z, x


def _build_ambisonics_filter(
    speakers: List[SpeakerPosition],
    input_channels: int,
) -> str:
    """
    Build an FFmpeg filter_complex expression that encodes
    mono/stereo sources to first-order ambisonics.

    For each speaker, applies pan filter to create 4-channel
    ambisonics from the corresponding input channel.
    """
    if not speakers:
        return ""

    filter_parts = []

    for i, speaker in enumerate(speakers):
        w, y, z, x = _encode_ambisonics_coefficients(
            speaker.azimuth, speaker.elevation, speaker.distance
        )

        ch_idx = min(speaker.audio_track, max(0, input_channels - 1))

        # Extract channel, then pan to 4-channel ambisonics
        filter_parts.append(
            f"[0:a]pan=mono|c0=c{ch_idx}[spk{i}]"
        )
        filter_parts.append(
            f"[spk{i}]pan=4c|"
            f"c0={w:.6f}*c0|"
            f"c1={y:.6f}*c0|"
            f"c2={z:.6f}*c0|"
            f"c3={x:.6f}*c0"
            f"[amb{i}]"
        )

    # Mix all ambisonic streams together
    if len(speakers) == 1:
        filter_parts.append(f"[amb0]acopy[ambiout]")
    else:
        mix_inputs = "".join(f"[amb{i}]" for i in range(len(speakers)))
        filter_parts.append(
            f"{mix_inputs}amix=inputs={len(speakers)}:normalize=0[ambiout]"
        )

    return ";".join(filter_parts)


# ---------------------------------------------------------------------------
# Face detection for speaker localization
# ---------------------------------------------------------------------------
def detect_speaker_positions(
    video_path: str,
    num_samples: int = 5,
    on_progress: Optional[Callable] = None,
) -> List[SpeakerPosition]:
    """
    Detect likely speaker positions in equirectangular video by
    analyzing face-like regions using FFmpeg's cropdetect and
    scene analysis.

    Falls back to default front-center position if detection fails.

    Args:
        video_path: Input equirectangular video.
        num_samples: Number of frames to sample.
        on_progress: Progress callback.

    Returns:
        List of SpeakerPosition objects.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = get_video_info(video_path)
    w = info.get("width", 3840)
    h = info.get("height", 1920)
    duration = info.get("duration", 0.0)

    if duration <= 0:
        return [SpeakerPosition(label="center", azimuth=0.0, elevation=0.0)]

    tmp_dir = tempfile.mkdtemp(prefix="oc_speaker_detect_")
    positions = []

    try:
        # Sample frames and analyze brightness/contrast hotspots
        # as proxy for face regions
        for i in range(num_samples):
            t = duration * (i + 1) / (num_samples + 1)
            frame_path = os.path.join(tmp_dir, f"frame_{i}.png")

            try:
                run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(t),
                    "-i", video_path,
                    "-frames:v", "1",
                    "-vf", "scale=960:480",
                    "-q:v", "5",
                    frame_path,
                ], timeout=15)
            except Exception:
                continue

            if on_progress:
                on_progress(
                    10 + int(30 * (i + 1) / num_samples),
                    f"Analyzing frame {i + 1}/{num_samples}",
                )

        # Divide frame into grid and detect high-activity regions
        grid_cols = 6
        grid_rows = 3
        cell_w = 960 // grid_cols
        cell_h = 480 // grid_rows

        # Check each grid cell for face-like characteristics using
        # the first sample frame
        first_frame = os.path.join(tmp_dir, "frame_0.png")
        if os.path.isfile(first_frame):
            for row in range(grid_rows):
                for col in range(grid_cols):
                    cx = col * cell_w
                    cy = row * cell_h

                    try:
                        result = subprocess.run([
                            "ffmpeg", "-hide_banner", "-loglevel", "error",
                            "-i", first_frame,
                            "-vf", (
                                f"crop={cell_w}:{cell_h}:{cx}:{cy},"
                                "signalstats"
                            ),
                            "-f", "null", "-",
                        ], capture_output=True, text=True, timeout=10)

                        # Look for regions with skin-tone-like brightness
                        for line in result.stderr.split("\n"):
                            if "YAVG" in line:
                                parts = line.split("YAVG:")
                                if len(parts) > 1:
                                    try:
                                        avg = float(parts[1].strip().split()[0])
                                        # Skin tones typically 100-200 in Y
                                        if 80 < avg < 200:
                                            azimuth = (col + 0.5) / grid_cols * 360.0 - 180.0
                                            elevation = 90.0 - (row + 0.5) / grid_rows * 180.0
                                            # Only consider near-horizon (likely face height)
                                            if abs(elevation) < 45:
                                                positions.append(SpeakerPosition(
                                                    label=f"speaker_{len(positions) + 1}",
                                                    azimuth=azimuth,
                                                    elevation=elevation,
                                                    distance=0.8,
                                                    audio_track=0,
                                                ))
                                    except (ValueError, IndexError):
                                        pass
                                break
                    except (subprocess.TimeoutExpired, OSError):
                        continue

    finally:
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    # Deduplicate close positions
    filtered = []
    for pos in positions:
        is_dup = False
        for existing in filtered:
            if (abs(existing.azimuth - pos.azimuth) < 40 and
                    abs(existing.elevation - pos.elevation) < 20):
                is_dup = True
                break
        if not is_dup:
            filtered.append(pos)

    if not filtered:
        # Default: center front
        filtered = [SpeakerPosition(label="center", azimuth=0.0, elevation=0.0)]

    return filtered[:6]


# ---------------------------------------------------------------------------
# Main spatial audio pipeline
# ---------------------------------------------------------------------------
def spatialize_audio(
    video_path: str,
    speakers: Optional[List[Dict]] = None,
    auto_detect: bool = True,
    output_path_override: Optional[str] = None,
    output_dir: str = "",
    sample_rate: int = 48000,
    on_progress: Optional[Callable] = None,
) -> SpatialAudioResult:
    """
    Convert audio to first-order ambisonics and mux with 360 video.

    Detects or accepts speaker positions, encodes audio to 4-channel
    ACN/SN3D (AmbiX) ambisonics, and muxes with the equirectangular
    video stream.

    Args:
        video_path: Input equirectangular video with audio.
        speakers: Optional list of speaker position dicts with keys:
            label, azimuth, elevation, distance, audio_track.
            If None and auto_detect=True, positions are auto-detected.
        auto_detect: Whether to auto-detect speaker positions.
        output_path_override: Output file path. Auto-generated if None.
        output_dir: Output directory.
        sample_rate: Output sample rate (default 48000).
        on_progress: Progress callback(pct, msg).

    Returns:
        SpatialAudioResult with output path and metadata.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    out = output_path_override or output_path(video_path, "spatial_audio", output_dir)

    if on_progress:
        on_progress(5, "Analyzing audio and video...")

    info = get_video_info(video_path)

    # Get audio channel count
    ffprobe = get_ffprobe_path()
    try:
        probe_result = subprocess.run([
            ffprobe, "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "a:0",
            video_path,
        ], capture_output=True, text=True, timeout=30)

        probe_data = json.loads(probe_result.stdout)
        audio_streams = probe_data.get("streams", [])
        input_channels = int(audio_streams[0].get("channels", 2)) if audio_streams else 2
    except (json.JSONDecodeError, IndexError, subprocess.TimeoutExpired, OSError):
        input_channels = 2

    # Resolve speaker positions
    speaker_positions: List[SpeakerPosition] = []

    if speakers:
        for sp in speakers:
            speaker_positions.append(SpeakerPosition(
                label=sp.get("label", f"speaker_{len(speaker_positions) + 1}"),
                azimuth=float(sp.get("azimuth", 0.0)),
                elevation=float(sp.get("elevation", 0.0)),
                distance=float(sp.get("distance", 1.0)),
                audio_track=int(sp.get("audio_track", 0)),
            ))
    elif auto_detect:
        if on_progress:
            on_progress(10, "Auto-detecting speaker positions...")
        speaker_positions = detect_speaker_positions(video_path, on_progress=on_progress)
    else:
        # Default center position
        speaker_positions = [
            SpeakerPosition(label="center", azimuth=0.0, elevation=0.0)
        ]

    if on_progress:
        on_progress(50, f"Encoding {len(speaker_positions)} speakers to ambisonics...")

    # Build ambisonics encoding filter
    filter_complex = _build_ambisonics_filter(speaker_positions, input_channels)

    if not filter_complex:
        # Fallback: simple channel layout conversion
        filter_complex = "[0:a]pan=4c|c0=c0|c1=c0|c2=c0|c3=c0[ambiout]"

    # Encode ambisonics audio and mux with video
    if on_progress:
        on_progress(60, "Muxing ambisonic audio with video...")

    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "[ambiout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "256k",
        "-ar", str(sample_rate),
        "-ac", "4",
        "-metadata:s:a:0", "handler_name=SpatialAudio",
        out,
    ], timeout=7200)

    if on_progress:
        on_progress(100, "Spatial audio encoding complete!")

    return SpatialAudioResult(
        output_path=out,
        speakers=len(speaker_positions),
        channels=4,
        format="ambix",
        sample_rate=sample_rate,
        muxed=True,
    )
