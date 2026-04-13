"""
OpenCut HSL Qualifier / Secondary Color Correction (13.3)

Isolate a color range via Hue/Saturation/Luminance qualification and
apply corrections only to the selected region:
- HSL range qualification with soft edges
- Matte preview (black/white mask of qualified region)
- Secondary correction (adjust only the qualified color)

Uses FFmpeg colorhold, huerotate, and overlay filters.
"""

import logging
import os
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional, Tuple

from opencut.helpers import FFmpegCmd, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class HSLRange:
    """Defines a Hue/Saturation/Luminance range for qualification.

    Hue: 0-360 degrees. Saturation and Luminance: 0.0-1.0.
    Softness controls the edge feathering of the qualification.
    """
    hue_center: float = 120.0
    hue_width: float = 30.0
    sat_min: float = 0.2
    sat_max: float = 1.0
    lum_min: float = 0.1
    lum_max: float = 0.9
    softness: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SecondaryCorrection:
    """Color correction parameters applied to the qualified region."""
    hue_shift: float = 0.0
    saturation: float = 1.0
    brightness: float = 0.0
    contrast: float = 1.0
    temperature: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HSLQualifyResult:
    """Result of an HSL qualification operation."""
    output_path: str = ""
    hsl_range: Optional[Dict] = None
    correction: Optional[Dict] = None
    matte_preview: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _parse_hsl_range(hue_range=None, sat_range=None, lum_range=None,
                     hsl_range=None) -> HSLRange:
    """Parse HSL range from various input formats."""
    if isinstance(hsl_range, HSLRange):
        return hsl_range
    if isinstance(hsl_range, dict):
        return HSLRange(
            hue_center=float(hsl_range.get("hue_center", 120)),
            hue_width=float(hsl_range.get("hue_width", 30)),
            sat_min=float(hsl_range.get("sat_min", 0.2)),
            sat_max=float(hsl_range.get("sat_max", 1.0)),
            lum_min=float(hsl_range.get("lum_min", 0.1)),
            lum_max=float(hsl_range.get("lum_max", 0.9)),
            softness=float(hsl_range.get("softness", 0.1)),
        )
    # Build from individual range tuples
    hr = hue_range or (120, 30)
    sr = sat_range or (0.2, 1.0)
    lr = lum_range or (0.1, 0.9)
    return HSLRange(
        hue_center=float(hr[0]),
        hue_width=float(hr[1]) if len(hr) > 1 else 30.0,
        sat_min=float(sr[0]),
        sat_max=float(sr[1]) if len(sr) > 1 else 1.0,
        lum_min=float(lr[0]),
        lum_max=float(lr[1]) if len(lr) > 1 else 0.9,
    )


def _parse_correction(correction=None) -> SecondaryCorrection:
    """Parse correction from dict or dataclass."""
    if isinstance(correction, SecondaryCorrection):
        return correction
    if isinstance(correction, dict):
        return SecondaryCorrection(
            hue_shift=float(correction.get("hue_shift", 0)),
            saturation=float(correction.get("saturation", 1)),
            brightness=float(correction.get("brightness", 0)),
            contrast=float(correction.get("contrast", 1)),
            temperature=float(correction.get("temperature", 0)),
        )
    return SecondaryCorrection()


def _build_colorhold_filter(hsl: HSLRange) -> str:
    """Build FFmpeg colorhold filter to isolate a hue range."""
    # colorhold keeps only the specified hue, desaturates the rest
    # similarity controls the hue width (0-1, mapped from degrees)
    similarity = max(0.01, min(1.0, hsl.hue_width / 180.0))
    # Convert hue center to a hex color approximation for colorhold
    # Use hue angle to compute an approximate RGB for the hold color
    import colorsys
    h = (hsl.hue_center % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(h, 0.5, 1.0)
    hex_color = f"{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    blend = max(0.0, min(1.0, hsl.softness))
    return f"colorhold=color=0x{hex_color}:similarity={similarity:.3f}:blend={blend:.3f}"


def _build_secondary_filter(hsl: HSLRange, corr: SecondaryCorrection) -> str:
    """Build filter chain for secondary color correction.

    Strategy: Use colorbalance + hue shift constrained via colorhold mask.
    We use a filter_complex that:
    1. Creates a matte from the qualified color range
    2. Applies corrections to the original
    3. Blends corrected and original using the matte
    """
    import colorsys
    h = (hsl.hue_center % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(h, 0.5, 1.0)
    hex_color = f"{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    similarity = max(0.01, min(1.0, hsl.hue_width / 180.0))
    blend = max(0.0, min(1.0, hsl.softness))

    # Build the correction filter chain
    corr_filters = []
    if corr.hue_shift != 0.0:
        corr_filters.append(f"hue=h={corr.hue_shift}")
    if corr.saturation != 1.0 or corr.brightness != 0.0 or corr.contrast != 1.0:
        eq_parts = []
        if corr.saturation != 1.0:
            eq_parts.append(f"saturation={max(0.0, min(3.0, corr.saturation))}")
        if corr.brightness != 0.0:
            eq_parts.append(f"brightness={max(-1.0, min(1.0, corr.brightness))}")
        if corr.contrast != 1.0:
            eq_parts.append(f"contrast={max(0.0, min(3.0, corr.contrast))}")
        corr_filters.append("eq=" + ":".join(eq_parts))
    if corr.temperature != 0.0:
        temp = max(-1.0, min(1.0, corr.temperature))
        rshift = temp * 0.1
        bshift = -temp * 0.1
        corr_filters.append(
            f"colorbalance=rs={rshift}:bs={bshift}:rm={rshift*0.5}:bm={bshift*0.5}"
        )

    correction_chain = ",".join(corr_filters) if corr_filters else "null"

    # filter_complex: split input, create matte + corrected, blend
    fc = (
        f"[0:v]split=2[orig][corr];"
        f"[corr]{correction_chain}[corrected];"
        f"[orig]colorhold=color=0x{hex_color}:similarity={similarity:.3f}"
        f":blend={blend:.3f},format=gbrp,extractplanes=g[matte];"
        f"[corrected][orig][matte]maskedmerge[out]"
    )
    return fc


# ---------------------------------------------------------------------------
# HSL Qualification (full video)
# ---------------------------------------------------------------------------
def qualify_hsl(
    video_path: str,
    hue_range: Optional[Tuple] = None,
    sat_range: Optional[Tuple] = None,
    lum_range: Optional[Tuple] = None,
    output_path: Optional[str] = None,
    output_dir: str = "",
    hsl_range: Optional[HSLRange] = None,
    on_progress: Optional[Callable] = None,
) -> HSLQualifyResult:
    """
    Isolate a color range via HSL qualification and desaturate the rest.

    Args:
        video_path: Source video file.
        hue_range: (center, width) in degrees.
        sat_range: (min, max) in 0.0-1.0.
        lum_range: (min, max) in 0.0-1.0.
        output_path: Explicit output file path.
        hsl_range: HSLRange dataclass (overrides individual ranges).
        on_progress: Callback(percent, message).

    Returns:
        HSLQualifyResult with output path.
    """
    hsl = _parse_hsl_range(hue_range, sat_range, lum_range, hsl_range)

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_hsl_qualified.mp4")

    if on_progress:
        on_progress(10, "Applying HSL qualification...")

    vf = _build_colorhold_filter(hsl)

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "HSL qualification applied.")

    return HSLQualifyResult(
        output_path=output_path,
        hsl_range=hsl.to_dict(),
    )


# ---------------------------------------------------------------------------
# Preview Matte (single frame mask)
# ---------------------------------------------------------------------------
def preview_matte(
    video_path: str,
    timestamp: float = 0.0,
    hsl_range: Optional[HSLRange] = None,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> HSLQualifyResult:
    """
    Generate a black/white matte preview showing the qualified region.

    Args:
        video_path: Source video file.
        timestamp: Frame time in seconds.
        hsl_range: HSLRange defining the qualification.
        output_path: Explicit output file path.
        on_progress: Callback(percent, message).

    Returns:
        HSLQualifyResult with matte image path.
    """
    hsl = _parse_hsl_range(hsl_range=hsl_range)

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_matte_preview.png")

    if on_progress:
        on_progress(10, "Generating matte preview...")

    import colorsys
    h = (hsl.hue_center % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(h, 0.5, 1.0)
    hex_color = f"{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    similarity = max(0.01, min(1.0, hsl.hue_width / 180.0))

    # Generate a grayscale matte: white = qualified, black = not
    vf = (
        f"colorhold=color=0x{hex_color}:similarity={similarity:.3f}:blend=0,"
        f"hue=s=0,negate,curves=all='0/0 0.5/1 1/1'"
    )

    cmd = (
        FFmpegCmd()
        .pre_input("-ss", str(timestamp))
        .input(video_path)
        .video_filter(vf)
        .frames(1)
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Matte preview generated.")

    return HSLQualifyResult(
        output_path=output_path,
        hsl_range=hsl.to_dict(),
        matte_preview=True,
    )


# ---------------------------------------------------------------------------
# Apply Secondary Correction
# ---------------------------------------------------------------------------
def apply_secondary_correction(
    video_path: str,
    qualification: Optional[HSLRange] = None,
    correction: Optional[SecondaryCorrection] = None,
    output_path: Optional[str] = None,
    output_dir: str = "",
    on_progress: Optional[Callable] = None,
) -> HSLQualifyResult:
    """
    Apply color correction only to the HSL-qualified region.

    Args:
        video_path: Source video file.
        qualification: HSLRange defining which colors to affect.
        correction: SecondaryCorrection with adjustments.
        output_path: Explicit output file path.
        on_progress: Callback(percent, message).

    Returns:
        HSLQualifyResult with corrected video path.
    """
    hsl = _parse_hsl_range(hsl_range=qualification)
    corr = _parse_correction(correction)

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_secondary_corrected.mp4")

    if on_progress:
        on_progress(10, "Applying secondary correction...")

    fc = _build_secondary_filter(hsl, corr)

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .filter_complex(fc, maps=["[out]"])
        .video_codec("libx264", crf=18, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "Secondary correction applied.")

    return HSLQualifyResult(
        output_path=output_path,
        hsl_range=hsl.to_dict(),
        correction=corr.to_dict(),
    )
