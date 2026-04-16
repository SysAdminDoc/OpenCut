"""
OpenCut Lens Distortion Correction

Corrects barrel/pincushion lens distortion and levels horizons
using FFmpeg's lenscorrection and rotate filters.

Also includes:
- Camera profile database for common cameras
- Auto-detection of camera model from video metadata
- Lens profile auto-application

Uses FFmpeg only — no additional dependencies required.
"""

import json
import logging
import math
import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffprobe_path,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Lens correction presets
# ---------------------------------------------------------------------------
LENS_PRESETS: Dict[str, dict] = {
    "gopro_wide": {"k1": -0.3, "k2": 0.0, "name": "GoPro Wide"},
    "gopro_linear": {"k1": -0.1, "k2": 0.0, "name": "GoPro Linear"},
    "dji_mini": {"k1": -0.2, "k2": 0.0, "name": "DJI Mini"},
    "dji_mavic": {"k1": -0.15, "k2": 0.0, "name": "DJI Mavic"},
    "iphone_wide": {"k1": -0.05, "k2": 0.0, "name": "iPhone Wide"},
    "fisheye_moderate": {"k1": -0.5, "k2": 0.1, "name": "Moderate Fisheye"},
    "fisheye_strong": {"k1": -0.7, "k2": 0.2, "name": "Strong Fisheye"},
}


# ---------------------------------------------------------------------------
# Lens distortion correction
# ---------------------------------------------------------------------------
def correct_lens_distortion(
    input_path: str,
    output_path_override: str = None,
    k1: float = None,
    k2: float = None,
    preset: str = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Correct barrel or pincushion lens distortion.

    Uses FFmpeg's lenscorrection filter with radial distortion
    coefficients k1 and k2.  A preset name can be given instead
    of explicit coefficients.

    Args:
        input_path: Source video file.
        output_path_override: Optional output path. Auto-generated if None.
        k1: Primary radial distortion coefficient (negative = barrel,
            positive = pincushion).
        k2: Secondary radial distortion coefficient.
        preset: Name of a camera lens preset (overrides k1/k2).
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path of the corrected file.

    Raises:
        ValueError: If neither preset nor k1 is specified, or preset
            is unknown.
    """
    # Resolve coefficients
    if preset:
        preset_lower = preset.lower().strip()
        if preset_lower not in LENS_PRESETS:
            raise ValueError(
                f"Unknown lens preset '{preset}'. "
                f"Available: {', '.join(sorted(LENS_PRESETS))}"
            )
        p = LENS_PRESETS[preset_lower]
        k1 = p["k1"]
        k2 = p["k2"]
    else:
        if k1 is None:
            raise ValueError("Either preset or k1 must be specified")
        if k2 is None:
            k2 = 0.0

    if on_progress:
        label = f"preset '{preset}'" if preset else f"k1={k1}, k2={k2}"
        on_progress(10, f"Correcting lens distortion ({label})...")

    vf = f"lenscorrection=k1={k1}:k2={k2}"

    out = output_path_override or output_path(input_path, "lens_corrected")

    if on_progress:
        on_progress(20, "Applying lenscorrection filter...")

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(out)
        .build()
    )

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Lens correction complete")

    return {"output_path": out}


# ---------------------------------------------------------------------------
# Horizon levelling
# ---------------------------------------------------------------------------
def level_horizon(
    input_path: str,
    output_path_override: str = None,
    angle: float = 0.0,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Level a tilted horizon by rotating the video.

    Uses FFmpeg's rotate filter.  Positive angle rotates clockwise,
    negative rotates counter-clockwise (degrees).

    Args:
        input_path: Source video file.
        output_path_override: Optional output path.
        angle: Rotation angle in degrees.  Positive = clockwise.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path.
    """
    if on_progress:
        on_progress(10, f"Levelling horizon ({angle:+.1f} deg)...")

    # FFmpeg rotate filter takes radians and uses bilinear fill
    angle_rad = angle * (math.pi / 180.0)

    # Get video dimensions so we can set the output size to avoid black bars
    info = get_video_info(input_path)
    w = info["width"]
    h = info["height"]

    # The rotate filter expression:
    #   rotate=angle:ow=rotw(angle):oh=roth(angle):fillcolor=black
    # We keep original dimensions and let the edges fill with black
    vf = (
        f"rotate={angle_rad}:"
        f"ow={w}:oh={h}:"
        f"fillcolor=black"
    )

    out = output_path_override or output_path(input_path, "levelled")

    if on_progress:
        on_progress(20, "Applying rotation filter...")

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(out)
        .build()
    )

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Horizon levelling complete")

    return {"output_path": out}


# ---------------------------------------------------------------------------
# List presets
# ---------------------------------------------------------------------------
def list_lens_presets() -> List[dict]:
    """
    Return available lens correction presets.

    Returns:
        List of dicts, each with id, name, k1, k2.
    """
    presets = []
    for preset_id, data in sorted(LENS_PRESETS.items()):
        presets.append({
            "id": preset_id,
            "name": data["name"],
            "k1": data["k1"],
            "k2": data["k2"],
        })
    return presets


# ---------------------------------------------------------------------------
# Data types for enhanced lens features
# ---------------------------------------------------------------------------
@dataclass
class CameraProfile:
    """A camera lens profile with distortion coefficients."""
    camera_model: str = ""
    lens_model: str = ""
    k1: float = 0.0
    k2: float = 0.0
    focal_length_mm: float = 0.0
    sensor_width_mm: float = 0.0
    fov_degrees: float = 0.0
    category: str = ""  # "action_cam", "drone", "phone", "cinema", "dslr"


@dataclass
class CameraDetectionResult:
    """Result of automatic camera detection."""
    detected: bool = False
    camera_model: str = ""
    lens_model: str = ""
    profile: Optional[CameraProfile] = None
    metadata: Dict = field(default_factory=dict)
    confidence: float = 0.0
    suggested_k1: float = 0.0
    suggested_k2: float = 0.0


# ---------------------------------------------------------------------------
# 52.1 + 52.4: Camera Profile Database
# ---------------------------------------------------------------------------
CAMERA_PROFILES: Dict[str, CameraProfile] = {
    # GoPro cameras
    "gopro_hero12_wide": CameraProfile(
        camera_model="GoPro HERO12", lens_model="Wide",
        k1=-0.30, k2=0.05, focal_length_mm=3.0,
        sensor_width_mm=6.17, fov_degrees=156, category="action_cam",
    ),
    "gopro_hero12_linear": CameraProfile(
        camera_model="GoPro HERO12", lens_model="Linear",
        k1=-0.10, k2=0.0, focal_length_mm=3.0,
        sensor_width_mm=6.17, fov_degrees=87, category="action_cam",
    ),
    "gopro_hero11_wide": CameraProfile(
        camera_model="GoPro HERO11", lens_model="Wide",
        k1=-0.30, k2=0.05, focal_length_mm=3.0,
        sensor_width_mm=6.17, fov_degrees=156, category="action_cam",
    ),
    "gopro_hero11_linear": CameraProfile(
        camera_model="GoPro HERO11", lens_model="Linear",
        k1=-0.10, k2=0.0, focal_length_mm=3.0,
        sensor_width_mm=6.17, fov_degrees=87, category="action_cam",
    ),
    "gopro_hero10_wide": CameraProfile(
        camera_model="GoPro HERO10", lens_model="Wide",
        k1=-0.28, k2=0.04, focal_length_mm=3.0,
        sensor_width_mm=6.17, fov_degrees=150, category="action_cam",
    ),
    "gopro_hero10_linear": CameraProfile(
        camera_model="GoPro HERO10", lens_model="Linear",
        k1=-0.10, k2=0.0, focal_length_mm=3.0,
        sensor_width_mm=6.17, fov_degrees=87, category="action_cam",
    ),
    "gopro_hero9_wide": CameraProfile(
        camera_model="GoPro HERO9", lens_model="Wide",
        k1=-0.28, k2=0.04, focal_length_mm=3.0,
        sensor_width_mm=6.17, fov_degrees=150, category="action_cam",
    ),
    # DJI drones
    "dji_mini4_pro": CameraProfile(
        camera_model="DJI Mini 4 Pro", lens_model="Standard",
        k1=-0.18, k2=0.02, focal_length_mm=6.7,
        sensor_width_mm=9.6, fov_degrees=82, category="drone",
    ),
    "dji_mini3_pro": CameraProfile(
        camera_model="DJI Mini 3 Pro", lens_model="Standard",
        k1=-0.15, k2=0.01, focal_length_mm=6.7,
        sensor_width_mm=9.6, fov_degrees=82, category="drone",
    ),
    "dji_air3": CameraProfile(
        camera_model="DJI Air 3", lens_model="Wide",
        k1=-0.12, k2=0.01, focal_length_mm=6.7,
        sensor_width_mm=9.6, fov_degrees=82, category="drone",
    ),
    "dji_mavic3": CameraProfile(
        camera_model="DJI Mavic 3", lens_model="Hasselblad",
        k1=-0.08, k2=0.0, focal_length_mm=12.3,
        sensor_width_mm=17.3, fov_degrees=84, category="drone",
    ),
    "dji_mavic_air2": CameraProfile(
        camera_model="DJI Mavic Air 2", lens_model="Standard",
        k1=-0.15, k2=0.01, focal_length_mm=4.49,
        sensor_width_mm=6.4, fov_degrees=84, category="drone",
    ),
    "dji_action4": CameraProfile(
        camera_model="DJI Action 4", lens_model="Wide",
        k1=-0.25, k2=0.03, focal_length_mm=2.77,
        sensor_width_mm=6.17, fov_degrees=155, category="action_cam",
    ),
    # iPhone models
    "iphone_15_pro_wide": CameraProfile(
        camera_model="iPhone 15 Pro", lens_model="Wide (24mm)",
        k1=-0.04, k2=0.0, focal_length_mm=6.765,
        sensor_width_mm=9.8, fov_degrees=73, category="phone",
    ),
    "iphone_15_pro_ultrawide": CameraProfile(
        camera_model="iPhone 15 Pro", lens_model="Ultra Wide (13mm)",
        k1=-0.15, k2=0.02, focal_length_mm=2.22,
        sensor_width_mm=4.8, fov_degrees=120, category="phone",
    ),
    "iphone_14_pro_wide": CameraProfile(
        camera_model="iPhone 14 Pro", lens_model="Wide (24mm)",
        k1=-0.04, k2=0.0, focal_length_mm=6.86,
        sensor_width_mm=9.8, fov_degrees=73, category="phone",
    ),
    "iphone_14_pro_ultrawide": CameraProfile(
        camera_model="iPhone 14 Pro", lens_model="Ultra Wide (13mm)",
        k1=-0.15, k2=0.02, focal_length_mm=2.22,
        sensor_width_mm=4.8, fov_degrees=120, category="phone",
    ),
    "iphone_13_wide": CameraProfile(
        camera_model="iPhone 13", lens_model="Wide (26mm)",
        k1=-0.05, k2=0.0, focal_length_mm=5.1,
        sensor_width_mm=5.7, fov_degrees=68, category="phone",
    ),
    # Samsung
    "samsung_s24_ultra_wide": CameraProfile(
        camera_model="Samsung S24 Ultra", lens_model="Wide",
        k1=-0.04, k2=0.0, focal_length_mm=6.3,
        sensor_width_mm=8.0, fov_degrees=72, category="phone",
    ),
    # Insta360
    "insta360_one_rs": CameraProfile(
        camera_model="Insta360 ONE RS", lens_model="4K Boost",
        k1=-0.35, k2=0.08, focal_length_mm=3.0,
        sensor_width_mm=6.17, fov_degrees=170, category="action_cam",
    ),
    "insta360_x3": CameraProfile(
        camera_model="Insta360 X3", lens_model="Standard",
        k1=-0.40, k2=0.10, focal_length_mm=1.9,
        sensor_width_mm=4.8, fov_degrees=178, category="action_cam",
    ),
    # Sony
    "sony_a7iv_24mm": CameraProfile(
        camera_model="Sony A7 IV", lens_model="24mm f/1.4",
        k1=-0.02, k2=0.0, focal_length_mm=24.0,
        sensor_width_mm=35.9, fov_degrees=84, category="dslr",
    ),
    "sony_zv_e10": CameraProfile(
        camera_model="Sony ZV-E10", lens_model="16-50mm Kit",
        k1=-0.06, k2=0.01, focal_length_mm=16.0,
        sensor_width_mm=23.5, fov_degrees=83, category="dslr",
    ),
    # Canon
    "canon_r6_24mm": CameraProfile(
        camera_model="Canon EOS R6", lens_model="24-105mm Kit",
        k1=-0.02, k2=0.0, focal_length_mm=24.0,
        sensor_width_mm=36.0, fov_degrees=84, category="dslr",
    ),
}

# Mapping from common ffprobe metadata strings to profile keys
_CAMERA_MODEL_MAP: Dict[str, str] = {
    "hero12 black": "gopro_hero12_wide",
    "hero11 black": "gopro_hero11_wide",
    "hero10 black": "gopro_hero10_wide",
    "hero9 black": "gopro_hero9_wide",
    "gopro hero12": "gopro_hero12_wide",
    "gopro hero11": "gopro_hero11_wide",
    "gopro hero10": "gopro_hero10_wide",
    "gopro hero9": "gopro_hero9_wide",
    "dji mini 4 pro": "dji_mini4_pro",
    "dji mini 3 pro": "dji_mini3_pro",
    "dji air 3": "dji_air3",
    "dji mavic 3": "dji_mavic3",
    "dji mavic air 2": "dji_mavic_air2",
    "dji action 4": "dji_action4",
    "fc3582": "dji_mini3_pro",       # DJI model code
    "fc3411": "dji_mavic3",          # DJI model code
    "fc7303": "dji_mini4_pro",       # DJI model code
    "iphone 15 pro": "iphone_15_pro_wide",
    "iphone 15 pro max": "iphone_15_pro_wide",
    "iphone 14 pro": "iphone_14_pro_wide",
    "iphone 14 pro max": "iphone_14_pro_wide",
    "iphone 13": "iphone_13_wide",
    "iphone 13 pro": "iphone_13_wide",
    "samsung sm-s928": "samsung_s24_ultra_wide",
    "insta360 one rs": "insta360_one_rs",
    "insta360 x3": "insta360_x3",
    "ilce-7m4": "sony_a7iv_24mm",    # Sony A7 IV
    "zv-e10": "sony_zv_e10",
    "canon eos r6": "canon_r6_24mm",
}


# ---------------------------------------------------------------------------
# 52.1: Camera auto-detection from metadata
# ---------------------------------------------------------------------------
def auto_detect_camera(video_path: str) -> CameraDetectionResult:
    """
    Auto-detect camera model from video metadata using ffprobe.

    Reads EXIF, QuickTime, and container-level metadata to identify
    the camera and match it against the profile database.

    Args:
        video_path: Path to video file.

    Returns:
        CameraDetectionResult with detected camera info and suggested
        correction parameters.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    ffprobe = get_ffprobe_path()

    # Extract comprehensive metadata
    try:
        result = subprocess.run([
            ffprobe, "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams",
            video_path,
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return CameraDetectionResult(detected=False)

        probe_data = json.loads(result.stdout)
    except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError):
        return CameraDetectionResult(detected=False)

    # Collect metadata from all sources
    metadata: Dict[str, str] = {}

    # Format-level tags
    fmt_tags = probe_data.get("format", {}).get("tags", {})
    for key, value in fmt_tags.items():
        metadata[key.lower()] = str(value)

    # Stream-level tags
    for stream in probe_data.get("streams", []):
        for key, value in stream.get("tags", {}).items():
            metadata[key.lower()] = str(value)

    # Look for camera model in various tag fields
    camera_model = ""
    lens_model = ""

    camera_fields = [
        "com.apple.quicktime.model",
        "model",
        "camera model",
        "camera",
        "com.android.model",
        "make",
        "com.apple.quicktime.make",
    ]

    for field_name in camera_fields:
        val = metadata.get(field_name, "").strip()
        if val and not camera_model:
            camera_model = val

    lens_fields = [
        "lens",
        "lens_model",
        "com.apple.quicktime.lens_model",
    ]

    for field_name in lens_fields:
        val = metadata.get(field_name, "").strip()
        if val and not lens_model:
            lens_model = val

    # Also check handler_name for GoPro signature
    handler = metadata.get("handler_name", "")
    if "gopro" in handler.lower() and not camera_model:
        camera_model = "GoPro"

    # Try to match against profile database
    matched_profile = None
    confidence = 0.0

    if camera_model:
        model_lower = camera_model.lower().strip()
        # Direct match
        for pattern, profile_key in _CAMERA_MODEL_MAP.items():
            if pattern in model_lower or model_lower in pattern:
                matched_profile = CAMERA_PROFILES.get(profile_key)
                confidence = 0.9 if pattern == model_lower else 0.7
                break

        # Fuzzy match on key words
        if not matched_profile:
            for profile_key, profile in CAMERA_PROFILES.items():
                pm = profile.camera_model.lower()
                if any(word in model_lower for word in pm.split() if len(word) > 2):
                    matched_profile = profile
                    confidence = 0.5
                    break

    return CameraDetectionResult(
        detected=camera_model != "",
        camera_model=camera_model,
        lens_model=lens_model,
        profile=matched_profile,
        metadata=metadata,
        confidence=confidence,
        suggested_k1=matched_profile.k1 if matched_profile else 0.0,
        suggested_k2=matched_profile.k2 if matched_profile else 0.0,
    )


def list_camera_profiles(category: Optional[str] = None) -> List[Dict]:
    """
    List all available camera profiles.

    Args:
        category: Optional filter by category
            ("action_cam", "drone", "phone", "cinema", "dslr").

    Returns:
        List of profile dicts.
    """
    profiles = []
    for profile_id, profile in sorted(CAMERA_PROFILES.items()):
        if category and profile.category != category:
            continue
        profiles.append({
            "id": profile_id,
            "camera_model": profile.camera_model,
            "lens_model": profile.lens_model,
            "k1": profile.k1,
            "k2": profile.k2,
            "focal_length_mm": profile.focal_length_mm,
            "sensor_width_mm": profile.sensor_width_mm,
            "fov_degrees": profile.fov_degrees,
            "category": profile.category,
        })
    return profiles


def correct_with_profile(
    input_path: str,
    profile_id: Optional[str] = None,
    auto_detect: bool = True,
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Apply lens correction using a camera profile.

    If profile_id is given, uses that profile. If auto_detect is True
    and no profile_id is given, attempts to detect the camera and
    apply the matching profile.

    Args:
        input_path: Source video file.
        profile_id: Specific camera profile ID.
        auto_detect: Try to auto-detect camera if no profile_id.
        output_path_override: Optional output path.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with output_path, camera info, and correction parameters.
    """
    profile = None
    detection = None

    if profile_id:
        if profile_id not in CAMERA_PROFILES:
            raise ValueError(
                f"Unknown profile '{profile_id}'. "
                f"Use list_camera_profiles() to see available profiles."
            )
        profile = CAMERA_PROFILES[profile_id]
    elif auto_detect:
        if on_progress:
            on_progress(5, "Auto-detecting camera model...")
        detection = auto_detect_camera(input_path)
        if detection.profile:
            profile = detection.profile
        else:
            raise ValueError(
                "Could not auto-detect camera. "
                "Please specify a profile_id or k1/k2 values."
            )

    if not profile:
        raise ValueError("No profile specified and auto-detect disabled")

    if on_progress:
        on_progress(10, f"Applying correction for {profile.camera_model}...")

    result = correct_lens_distortion(
        input_path=input_path,
        output_path_override=output_path_override,
        k1=profile.k1,
        k2=profile.k2,
        on_progress=on_progress,
    )

    return {
        "output_path": result["output_path"],
        "camera_model": profile.camera_model,
        "lens_model": profile.lens_model,
        "k1": profile.k1,
        "k2": profile.k2,
        "auto_detected": detection is not None and detection.detected,
        "confidence": detection.confidence if detection else 1.0,
    }
