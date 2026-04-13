"""
OpenCut LOG / Camera Profile Pipeline

Auto-detect camera LOG profiles from FFmpeg probe metadata, map to known
profiles (S-Log3, C-Log3, V-Log, ARRI LogC, BRAW), and apply correct
Input Display Transforms via generated LUTs.  Supports LUT stacking
(technical LUT + creative LUT).

Uses FFmpeg lut3d filter for all LUT application.
"""

import json
import logging
import os
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_ffprobe_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

LUTS_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "luts")


# ---------------------------------------------------------------------------
# Known LOG profiles
# ---------------------------------------------------------------------------
@dataclass
class LogProfile:
    """Descriptor for a camera LOG profile."""
    name: str
    camera: str
    color_trc: List[str] = field(default_factory=list)
    color_primaries: List[str] = field(default_factory=list)
    transfer_characteristics: List[str] = field(default_factory=list)
    description: str = ""


KNOWN_PROFILES: Dict[str, LogProfile] = {
    "slog3": LogProfile(
        name="S-Log3",
        camera="Sony",
        color_trc=["arib-std-b67", "bt2020-10", "smpte2084"],
        color_primaries=["bt2020"],
        transfer_characteristics=["arib-std-b67", "smpte2084"],
        description="Sony S-Log3/S-Gamut3.Cine for cinema cameras (FX6, FX9, Venice)",
    ),
    "clog3": LogProfile(
        name="C-Log3",
        camera="Canon",
        color_trc=["bt709", "arib-std-b67"],
        color_primaries=["bt709", "bt2020"],
        transfer_characteristics=["bt709"],
        description="Canon C-Log3 for Cinema EOS (C70, C300 III, R5C)",
    ),
    "vlog": LogProfile(
        name="V-Log",
        camera="Panasonic",
        color_trc=["arib-std-b67", "bt2020-10"],
        color_primaries=["bt2020"],
        transfer_characteristics=["arib-std-b67"],
        description="Panasonic V-Log/V-Gamut for S-series and Varicam",
    ),
    "logc": LogProfile(
        name="ARRI LogC",
        camera="ARRI",
        color_trc=["bt709", "arib-std-b67"],
        color_primaries=["bt709", "bt2020"],
        transfer_characteristics=["bt709"],
        description="ARRI LogC3/LogC4 for Alexa series",
    ),
    "braw": LogProfile(
        name="Blackmagic Film",
        camera="Blackmagic",
        color_trc=["bt709", "arib-std-b67"],
        color_primaries=["bt709", "bt2020"],
        transfer_characteristics=["bt709"],
        description="Blackmagic Film Gen5 for BMPCC and URSA cameras",
    ),
    "nlog": LogProfile(
        name="N-Log",
        camera="Nikon",
        color_trc=["bt709"],
        color_primaries=["bt709"],
        transfer_characteristics=["bt709"],
        description="Nikon N-Log for Z series mirrorless",
    ),
    "flog": LogProfile(
        name="F-Log",
        camera="Fujifilm",
        color_trc=["bt709"],
        color_primaries=["bt709"],
        transfer_characteristics=["bt709"],
        description="Fujifilm F-Log for X-T and X-H series",
    ),
    "dlog": LogProfile(
        name="D-Log M",
        camera="DJI",
        color_trc=["bt709"],
        color_primaries=["bt709"],
        transfer_characteristics=["bt709"],
        description="DJI D-Log M for Mavic 3 / Inspire 3",
    ),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class LogDetectResult:
    """Result from LOG profile detection."""
    detected_profile: str = ""
    profile_name: str = ""
    camera: str = ""
    confidence: float = 0.0
    color_trc: str = ""
    color_primaries: str = ""
    color_transfer: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class IDTApplyResult:
    """Result from applying an Input Display Transform."""
    output_path: str = ""
    profile: str = ""
    lut_path: str = ""
    method: str = "generated_lut"


@dataclass
class LUTStackResult:
    """Result from stacking multiple LUTs."""
    output_path: str = ""
    luts_applied: List[str] = field(default_factory=list)
    method: str = "sequential"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_log_profile(
    video_path: str,
    on_progress: Optional[Callable] = None,
) -> LogDetectResult:
    """
    Auto-detect camera LOG profile from FFmpeg probe metadata.

    Examines color_primaries, color_trc, and transfer_characteristics
    from the video stream, then matches against known camera profiles.
    Also checks container-level metadata tags for camera model info.

    Args:
        video_path: Path to video file.
        on_progress: Progress callback(pct, msg).

    Returns:
        LogDetectResult with detected profile name and confidence.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if on_progress:
        on_progress(10, "Probing video metadata...")

    # Probe color metadata
    cmd = [
        get_ffprobe_path(), "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=color_transfer,color_primaries,color_space,color_range,"
        "bits_per_raw_sample,pix_fmt,codec_name",
        "-show_entries", "format_tags",
        "-of", "json",
        video_path,
    ]
    result = _sp.run(cmd, capture_output=True, timeout=30)

    if result.returncode != 0:
        logger.warning("ffprobe failed for LOG detection on %s", video_path)
        return LogDetectResult()

    try:
        data = json.loads(result.stdout.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return LogDetectResult()

    if on_progress:
        on_progress(40, "Analyzing color metadata...")

    streams = data.get("streams", [])
    stream = streams[0] if streams else {}
    format_tags = data.get("format", {}).get("tags", {})

    color_trc = stream.get("color_transfer", "").lower()
    color_primaries = stream.get("color_primaries", "").lower()
    color_space = stream.get("color_space", "").lower()
    codec_name = stream.get("codec_name", "").lower()
    pix_fmt = stream.get("pix_fmt", "").lower()
    bits = stream.get("bits_per_raw_sample", "")

    # Collect all metadata as lowercase for camera model detection
    all_tags = {k.lower(): str(v).lower() for k, v in format_tags.items()}
    camera_model = all_tags.get("model", all_tags.get("com.apple.quicktime.model", ""))

    if on_progress:
        on_progress(60, "Matching against known profiles...")

    # Scoring: match metadata against known profiles
    best_profile = ""
    best_score = 0.0
    best_name = ""
    best_camera = ""

    for pid, profile in KNOWN_PROFILES.items():
        score = 0.0

        # Camera model string matching
        cam_lower = profile.camera.lower()
        if cam_lower in camera_model:
            score += 3.0
        # Check all format tags for camera brand mention
        for tag_val in all_tags.values():
            if cam_lower in tag_val:
                score += 1.0
                break

        # Color TRC matching
        if color_trc and color_trc in [t.lower() for t in profile.color_trc]:
            score += 2.0

        # Color primaries matching
        if color_primaries and color_primaries in [p.lower() for p in profile.color_primaries]:
            score += 2.0

        # Codec-specific hints
        if pid == "braw" and ("braw" in codec_name or "blackmagic" in codec_name):
            score += 4.0
        if pid == "logc" and "arri" in codec_name:
            score += 4.0

        # Bit depth hint (LOG footage is typically 10+ bit)
        if bits and int(bits) >= 10:
            score += 0.5

        # Wide gamut pixel format hint
        if "10le" in pix_fmt or "10be" in pix_fmt or "12le" in pix_fmt:
            score += 0.5

        if score > best_score:
            best_score = score
            best_profile = pid
            best_name = profile.name
            best_camera = profile.camera

    # Confidence mapping: score 0-10 -> confidence 0-1
    confidence = min(1.0, best_score / 8.0) if best_score > 0 else 0.0

    # Minimum threshold: if score is too low, report as undetected
    if best_score < 1.5:
        best_profile = ""
        best_name = "Unknown"
        best_camera = "Unknown"
        confidence = 0.0

    if on_progress:
        on_progress(100, f"Detected: {best_name} (confidence {confidence:.0%})")

    return LogDetectResult(
        detected_profile=best_profile,
        profile_name=best_name,
        camera=best_camera,
        confidence=round(confidence, 3),
        color_trc=color_trc,
        color_primaries=color_primaries,
        color_transfer=stream.get("color_transfer", ""),
        metadata={
            "color_space": color_space,
            "codec": codec_name,
            "pix_fmt": pix_fmt,
            "bits_per_raw_sample": bits,
            "camera_model": camera_model,
        },
    )


# ---------------------------------------------------------------------------
# IDT LUT Generation
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _slog3_to_linear(x: float) -> float:
    """Sony S-Log3 to linear light."""
    if x >= 0.1711:
        return 10.0 ** ((x - 0.4105) / 0.2556) * 0.18 - 0.01
    else:
        return (x - 0.0929) / 5.0 * 0.18 - 0.01


def _clog3_to_linear(x: float) -> float:
    """Canon C-Log3 to linear (approximate)."""
    if x >= 0.097:
        return (10.0 ** ((x - 0.4135) / 0.2471) - 1.0) / 14.98
    else:
        return -(10.0 ** ((0.4135 - x) / 0.2471) - 1.0) / 14.98


def _vlog_to_linear(x: float) -> float:
    """Panasonic V-Log to linear."""
    if x >= 0.181:
        return 10.0 ** ((x - 0.598206) / 0.241514) - 0.00873
    else:
        return (x - 0.125) / 5.6


def _logc_to_linear(x: float) -> float:
    """ARRI LogC3 to linear (EI 800)."""
    # LogC3 parameters for EI 800
    a = 5.555556
    b = 0.052272
    c = 0.247190
    d = 0.385537
    e = 5.367655
    f = 0.092809
    cut = 0.010591
    if x > e * cut + f:
        return (10.0 ** ((x - d) / c) - b) / a
    else:
        return (x - f) / e


def _braw_to_linear(x: float) -> float:
    """Blackmagic Film Gen5 to linear (approximate)."""
    # Similar shape to LogC
    if x >= 0.09:
        return (10.0 ** ((x - 0.3855) / 0.2471) - 0.052) / 5.556
    else:
        return (x - 0.0929) / 5.367


def _generic_log_to_linear(x: float) -> float:
    """Generic LOG to linear for unknown profiles."""
    if x >= 0.1:
        return (10.0 ** ((x - 0.385) / 0.25) - 0.05) / 5.5
    else:
        return x / 5.0


def _linear_to_rec709(x: float) -> float:
    """Linear to Rec.709 gamma (display transform)."""
    if x <= 0.0:
        return 0.0
    if x < 0.018:
        return x * 4.5
    return 1.099 * (x ** 0.45) - 0.099


_IDT_TRANSFORMS = {
    "slog3": _slog3_to_linear,
    "clog3": _clog3_to_linear,
    "vlog": _vlog_to_linear,
    "logc": _logc_to_linear,
    "braw": _braw_to_linear,
    "nlog": _generic_log_to_linear,
    "flog": _generic_log_to_linear,
    "dlog": _generic_log_to_linear,
}


def _generate_idt_lut(profile: str, size: int = 33) -> str:
    """Generate a .cube IDT LUT for a given LOG profile.

    The LUT converts from LOG-encoded values to Rec.709 display space
    (LOG -> linear -> Rec.709 gamma).
    """
    os.makedirs(LUTS_DIR, exist_ok=True)
    lut_path = os.path.join(LUTS_DIR, f"idt_{profile}.cube")

    log_to_linear = _IDT_TRANSFORMS.get(profile, _generic_log_to_linear)

    with open(lut_path, "w", encoding="utf-8") as f:
        f.write(f"TITLE \"IDT {profile} to Rec.709\"\n")
        f.write(f"LUT_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n\n")

        for b_i in range(size):
            for g_i in range(size):
                for r_i in range(size):
                    r = r_i / (size - 1)
                    g = g_i / (size - 1)
                    b = b_i / (size - 1)

                    # LOG -> linear
                    r_lin = max(0.0, log_to_linear(r))
                    g_lin = max(0.0, log_to_linear(g))
                    b_lin = max(0.0, log_to_linear(b))

                    # Linear -> Rec.709
                    r_out = _clamp(_linear_to_rec709(r_lin))
                    g_out = _clamp(_linear_to_rec709(g_lin))
                    b_out = _clamp(_linear_to_rec709(b_lin))

                    f.write(f"{r_out:.6f} {g_out:.6f} {b_out:.6f}\n")

    return lut_path


# ---------------------------------------------------------------------------
# Apply IDT
# ---------------------------------------------------------------------------
def apply_idt(
    video_path: str,
    profile: str,
    output_path_str: Optional[str] = None,
    output_dir: str = "",
    lut_size: int = 33,
    on_progress: Optional[Callable] = None,
) -> IDTApplyResult:
    """
    Apply an Input Display Transform for a LOG profile.

    Generates a profile-specific .cube LUT (LOG -> Rec.709) and applies
    it via FFmpeg's lut3d filter.

    Args:
        video_path: Input LOG-encoded video.
        profile: Profile ID (e.g. "slog3", "clog3", "vlog", "logc", "braw").
        output_path_str: Output path. Auto-generated if None.
        output_dir: Directory for output.
        lut_size: LUT cube size (17, 33, or 65).
        on_progress: Progress callback(pct, msg).

    Returns:
        IDTApplyResult with output path and LUT info.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    profile = profile.lower().strip()
    if profile not in _IDT_TRANSFORMS:
        raise ValueError(
            f"Unknown profile: {profile}. "
            f"Supported: {', '.join(_IDT_TRANSFORMS.keys())}"
        )

    lut_size = max(17, min(65, lut_size))

    if on_progress:
        on_progress(5, f"Generating IDT LUT for {profile}...")

    lut_path = _generate_idt_lut(profile, size=lut_size)

    if on_progress:
        on_progress(20, "Applying IDT LUT to video...")

    if output_path_str is None:
        directory = output_dir or os.path.dirname(video_path)
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path_str = os.path.join(directory, f"{base}_idt_{profile}.mp4")

    cmd = (FFmpegCmd()
           .input(video_path)
           .video_filter(f"lut3d='{lut_path}'")
           .video_codec("libx264", crf=18, preset="medium")
           .audio_codec("copy")
           .faststart()
           .output(output_path_str)
           .build())
    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(100, f"IDT applied: {profile}")

    return IDTApplyResult(
        output_path=output_path_str,
        profile=profile,
        lut_path=lut_path,
        method="generated_lut",
    )


# ---------------------------------------------------------------------------
# LUT Stacking
# ---------------------------------------------------------------------------
def stack_luts(
    video_path: str,
    lut_paths: List[str],
    output_path_str: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> LUTStackResult:
    """
    Apply multiple LUTs sequentially (e.g. technical IDT + creative grade).

    Each LUT is applied in order using chained lut3d filters.

    Args:
        video_path: Input video file.
        lut_paths: Ordered list of .cube LUT file paths.
        output_path_str: Output path. Auto-generated if None.
        output_dir: Directory for output.
        on_progress: Progress callback(pct, msg).

    Returns:
        LUTStackResult with output path and applied LUTs list.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not lut_paths:
        raise ValueError("No LUT paths provided")

    # Validate all LUT files exist
    valid_luts = []
    for lp in lut_paths:
        lp = lp.strip()
        if not os.path.isfile(lp):
            raise FileNotFoundError(f"LUT file not found: {lp}")
        if not lp.lower().endswith('.cube'):
            raise ValueError(f"Only .cube LUT files are supported: {lp}")
        valid_luts.append(lp)

    if on_progress:
        on_progress(5, f"Stacking {len(valid_luts)} LUTs...")

    if output_path_str is None:
        directory = output_dir or os.path.dirname(video_path)
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path_str = os.path.join(directory, f"{base}_graded.mp4")

    # Build chained lut3d filter string
    # e.g. "lut3d='path1.cube',lut3d='path2.cube'"
    filter_parts = [f"lut3d='{lp}'" for lp in valid_luts]
    vf = ",".join(filter_parts)

    if on_progress:
        on_progress(20, "Applying LUT stack...")

    cmd = (FFmpegCmd()
           .input(video_path)
           .video_filter(vf)
           .video_codec("libx264", crf=18, preset="medium")
           .audio_codec("copy")
           .faststart()
           .output(output_path_str)
           .build())
    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(100, f"Applied {len(valid_luts)} LUTs")

    return LUTStackResult(
        output_path=output_path_str,
        luts_applied=valid_luts,
        method="sequential",
    )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def list_supported_profiles() -> List[Dict]:
    """Return a list of all supported LOG profiles with metadata."""
    profiles = []
    for pid, profile in KNOWN_PROFILES.items():
        profiles.append({
            "id": pid,
            "name": profile.name,
            "camera": profile.camera,
            "description": profile.description,
            "has_idt": pid in _IDT_TRANSFORMS,
        })
    return profiles
