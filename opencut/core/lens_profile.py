"""
OpenCut Lens Profile Auto-Detection Module

Automatically reads camera/lens EXIF data from video files, looks up
the corresponding lens profile in the database, and auto-applies
corrections. Provides comprehensive camera info display.

Uses ffprobe for metadata extraction and delegates to lens_correction
for the actual correction pipeline.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    get_ffprobe_path,
    get_video_info,
    output_path,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class LensInfo:
    """Comprehensive lens and camera information."""
    camera_make: str = ""
    camera_model: str = ""
    lens_model: str = ""
    focal_length_mm: float = 0.0
    focal_length_35mm: float = 0.0
    aperture: str = ""
    iso: int = 0
    shutter_speed: str = ""
    resolution: str = ""
    frame_rate: float = 0.0
    codec: str = ""
    color_space: str = ""
    bit_depth: int = 0
    creation_date: str = ""
    gps_location: str = ""
    software: str = ""
    profile_match: str = ""       # matched profile ID or ""
    profile_confidence: float = 0.0


@dataclass
class AutoCorrectionResult:
    """Result of automatic lens profile correction."""
    output_path: str = ""
    lens_info: Optional[LensInfo] = None
    profile_id: str = ""
    corrections_applied: List[str] = field(default_factory=list)
    k1: float = 0.0
    k2: float = 0.0


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------
def _extract_full_metadata(video_path: str) -> Dict[str, str]:
    """
    Extract all available metadata from a video file using ffprobe.

    Returns a flat dict of tag_name -> value.
    """
    ffprobe = get_ffprobe_path()

    try:
        result = subprocess.run([
            ffprobe, "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", "-show_entries",
            "format_tags:stream_tags",
            video_path,
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)
    except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError):
        return {}

    metadata = {}

    # Collect from format tags
    fmt_tags = data.get("format", {}).get("tags", {})
    for k, v in fmt_tags.items():
        metadata[k.lower().strip()] = str(v).strip()

    # Collect from stream tags
    for stream in data.get("streams", []):
        for k, v in stream.get("tags", {}).items():
            key = k.lower().strip()
            if key not in metadata:
                metadata[key] = str(v).strip()
        # Also grab codec info
        if "codec_name" not in metadata and "codec_name" in stream:
            metadata["codec_name"] = stream["codec_name"]
        if "pix_fmt" not in metadata and "pix_fmt" in stream:
            metadata["pix_fmt"] = stream.get("pix_fmt", "")
        if "bits_per_raw_sample" not in metadata and "bits_per_raw_sample" in stream:
            metadata["bits_per_raw_sample"] = stream.get("bits_per_raw_sample", "")

    return metadata


def get_lens_info(video_path: str) -> LensInfo:
    """
    Extract comprehensive lens and camera information from a video file.

    Reads EXIF data, QuickTime metadata, and container tags to
    provide detailed camera and shooting information.

    Args:
        video_path: Path to video file.

    Returns:
        LensInfo with all available camera/lens details.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    metadata = _extract_full_metadata(video_path)
    info = get_video_info(video_path)

    lens_info = LensInfo()

    # Camera make
    for key in ["com.apple.quicktime.make", "make", "manufacturer"]:
        val = metadata.get(key, "")
        if val:
            lens_info.camera_make = val
            break

    # Camera model
    for key in ["com.apple.quicktime.model", "model", "camera model",
                "camera", "com.android.model"]:
        val = metadata.get(key, "")
        if val:
            lens_info.camera_model = val
            break

    # Lens model
    for key in ["lens", "lens_model", "com.apple.quicktime.lens_model",
                "lens model"]:
        val = metadata.get(key, "")
        if val:
            lens_info.lens_model = val
            break

    # Focal length
    for key in ["focal_length", "com.apple.quicktime.focal_length",
                "focallength"]:
        val = metadata.get(key, "")
        if val:
            try:
                lens_info.focal_length_mm = float(val.split("/")[0])
            except (ValueError, IndexError):
                pass
            break

    # 35mm equivalent focal length
    for key in ["focal_length_in_35mm_film", "focallengthin35mmfilm"]:
        val = metadata.get(key, "")
        if val:
            try:
                lens_info.focal_length_35mm = float(val)
            except ValueError:
                pass
            break

    # Aperture
    for key in ["fnumber", "f_number", "aperture"]:
        val = metadata.get(key, "")
        if val:
            lens_info.aperture = f"f/{val}" if not val.startswith("f") else val
            break

    # ISO
    for key in ["iso", "photographicsensitivity",
                "com.apple.quicktime.camera.iso"]:
        val = metadata.get(key, "")
        if val:
            try:
                lens_info.iso = int(float(val))
            except (ValueError, OverflowError):
                pass
            break

    # Shutter speed
    for key in ["exposuretime", "exposure_time", "shutter_speed"]:
        val = metadata.get(key, "")
        if val:
            lens_info.shutter_speed = val
            break

    # Resolution
    w = info.get("width", 0)
    h = info.get("height", 0)
    if w and h:
        lens_info.resolution = f"{w}x{h}"

    # Frame rate
    lens_info.frame_rate = info.get("fps", 0.0)

    # Codec
    lens_info.codec = metadata.get("codec_name", info.get("codec", ""))

    # Color space
    for key in ["color_space", "colorspace", "color_primaries"]:
        val = metadata.get(key, "")
        if val:
            lens_info.color_space = val
            break

    # Bit depth
    bd = metadata.get("bits_per_raw_sample", "")
    if bd:
        try:
            lens_info.bit_depth = int(bd)
        except ValueError:
            pass

    # Creation date
    for key in ["creation_time", "date", "com.apple.quicktime.creationdate"]:
        val = metadata.get(key, "")
        if val:
            lens_info.creation_date = val
            break

    # GPS
    for key in ["location", "com.apple.quicktime.location.iso6709",
                "gps", "gps_location"]:
        val = metadata.get(key, "")
        if val:
            lens_info.gps_location = val
            break

    # Software
    for key in ["software", "encoder", "com.apple.quicktime.software",
                "handler_name"]:
        val = metadata.get(key, "")
        if val:
            lens_info.software = val
            break

    # Try to match against camera profiles
    from opencut.core.lens_correction import auto_detect_camera
    try:
        detection = auto_detect_camera(video_path)
        if detection.profile:
            lens_info.profile_match = detection.camera_model
            lens_info.profile_confidence = detection.confidence
    except Exception:
        pass

    return lens_info


# ---------------------------------------------------------------------------
# Auto-correction pipeline
# ---------------------------------------------------------------------------
def auto_correct_lens(
    video_path: str,
    output_path_override: Optional[str] = None,
    output_dir: str = "",
    include_ca: bool = False,
    on_progress: Optional[Callable] = None,
) -> AutoCorrectionResult:
    """
    Automatically detect camera, look up lens profile, and apply corrections.

    Reads EXIF/metadata from the video, matches against the camera profile
    database, and applies the appropriate lens distortion correction.
    Optionally also corrects chromatic aberration.

    Args:
        video_path: Path to input video.
        output_path_override: Output file path. Auto-generated if None.
        output_dir: Output directory.
        include_ca: Also correct chromatic aberration.
        on_progress: Progress callback(pct, msg).

    Returns:
        AutoCorrectionResult with output path and correction details.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    out = output_path_override or output_path(video_path, "auto_corrected", output_dir)

    if on_progress:
        on_progress(5, "Reading camera metadata...")

    # Get full lens info
    lens_info = get_lens_info(video_path)

    # Auto-detect camera and get profile
    from opencut.core.lens_correction import (
        auto_detect_camera,
        correct_lens_distortion,
    )

    if on_progress:
        on_progress(10, "Detecting camera profile...")

    detection = auto_detect_camera(video_path)

    corrections_applied = []
    current_input = video_path
    profile_id = ""
    k1, k2 = 0.0, 0.0

    if detection.profile:
        profile_id = detection.camera_model
        k1 = detection.suggested_k1
        k2 = detection.suggested_k2

        if on_progress:
            on_progress(20, f"Applying lens correction for {detection.camera_model}...")

        # Apply lens distortion correction
        intermediate_out = out if not include_ca else output_path(
            video_path, "lens_tmp", output_dir
        )

        result = correct_lens_distortion(
            input_path=current_input,
            output_path_override=intermediate_out,
            k1=k1,
            k2=k2,
            on_progress=lambda p, m: on_progress(20 + int(p * 0.4), m) if on_progress else None,
        )

        current_input = result["output_path"]
        corrections_applied.append(f"lens_distortion(k1={k1}, k2={k2})")

        lens_info.profile_match = detection.camera_model
        lens_info.profile_confidence = detection.confidence
    else:
        logger.info("No camera profile found for auto-correction")

    # Optional CA correction
    if include_ca:
        from opencut.core.chromatic_aberration import correct_chromatic_aberration

        if on_progress:
            on_progress(65, "Checking for chromatic aberration...")

        ca_result = correct_chromatic_aberration(
            video_path=current_input,
            auto_detect=True,
            output_path_override=out,
            on_progress=lambda p, m: on_progress(65 + int(p * 0.3), m) if on_progress else None,
        )

        if ca_result.severity != "none":
            corrections_applied.append(
                f"chromatic_aberration(R:{ca_result.red_shift_x},{ca_result.red_shift_y} "
                f"B:{ca_result.blue_shift_x},{ca_result.blue_shift_y})"
            )

        # Clean up intermediate file if we used lens correction too
        if detection.profile and current_input != video_path:
            try:
                os.unlink(current_input)
            except OSError:
                pass

    if not corrections_applied:
        # No corrections to apply, just copy the file
        if on_progress:
            on_progress(50, "No corrections needed, copying file...")

        from opencut.helpers import run_ffmpeg
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-c", "copy",
            out,
        ], timeout=600)

    if on_progress:
        on_progress(100, f"Auto-correction complete ({len(corrections_applied)} corrections)")

    return AutoCorrectionResult(
        output_path=out,
        lens_info=lens_info,
        profile_id=profile_id,
        corrections_applied=corrections_applied,
        k1=k1,
        k2=k2,
    )
