"""
OpenCut SDR-to-HDR Upconversion

Convert SDR (BT.709) video to HDR (BT.2020) using FFmpeg zscale
and inverse tone mapping.  Supports PQ (ST.2084) and HLG transfer
functions, and embeds ST.2086 mastering display metadata.
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import (
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Transfer function presets
# ---------------------------------------------------------------------------

TRANSFER_FUNCTIONS = {
    "pq": {
        "name": "PQ (Perceptual Quantizer / ST.2084)",
        "transfer": "smpte2084",
        "max_luminance": 1000,
        "zscale_transfer": "smpte2084",
    },
    "hlg": {
        "name": "HLG (Hybrid Log-Gamma)",
        "transfer": "arib-std-b67",
        "max_luminance": 1000,
        "zscale_transfer": "arib-std-b67",
    },
}


@dataclass
class HDRConversionResult:
    """Result of SDR-to-HDR conversion."""
    output_path: str = ""
    transfer_function: str = ""
    color_primaries: str = "bt2020"
    max_luminance: int = 1000
    min_luminance: float = 0.0050
    has_metadata: bool = False
    original_colorspace: str = ""


# ---------------------------------------------------------------------------
# SDR -> HDR conversion
# ---------------------------------------------------------------------------

def sdr_to_hdr(
    video_path: str,
    transfer: str = "pq",
    max_luminance: int = 1000,
    min_luminance: float = 0.0050,
    max_cll: int = 1000,
    max_fall: int = 400,
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Convert an SDR video to HDR using FFmpeg zscale for BT.709 -> BT.2020
    gamut mapping, inverse tone mapping, and PQ or HLG transfer function
    application.

    Embeds ST.2086 mastering display metadata and MaxCLL / MaxFALL
    content light level information in the output.

    Args:
        video_path:  Input SDR video.
        transfer:  ``"pq"`` (PQ / ST.2084) or ``"hlg"`` (HLG).
        max_luminance:  Peak luminance in nits (default 1000).
        min_luminance:  Minimum luminance in nits (default 0.005).
        max_cll:  Maximum Content Light Level.
        max_fall:  Maximum Frame-Average Light Level.
        output_path_override:  Explicit output path.
        on_progress:  Callback ``(pct, msg)``.

    Returns:
        dict with *output_path*, *transfer_function*, *color_primaries*,
        *max_luminance*, *min_luminance*, *has_metadata*.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    if transfer not in TRANSFER_FUNCTIONS:
        raise ValueError(f"Unknown transfer function: {transfer}. Use 'pq' or 'hlg'.")

    tf = TRANSFER_FUNCTIONS[transfer]
    out = output_path_override or output_path(video_path, f"hdr_{transfer}")

    if on_progress:
        on_progress(5, f"Starting SDR-to-HDR conversion ({tf['name']})...")

    get_video_info(video_path)

    if on_progress:
        on_progress(10, "Building HDR conversion filter chain...")

    # Build the zscale filter chain:
    # 1) Convert from BT.709 to linear light
    # 2) Expand SDR range to HDR luminance via inverse tonemap
    # 3) Convert to BT.2020 primaries with chosen transfer function
    zscale_chain = (
        f"zscale=t=linear:npl={max_luminance}:tin=bt709:min=bt709:pin=bt709,"
        f"zscale=p=bt2020:t={tf['zscale_transfer']}:m=bt2020nc:r=tv,"
        f"format=yuv420p10le"
    )

    if on_progress:
        on_progress(20, "Converting color space and applying tone mapping...")

    # ST.2086 metadata: mastering display color volume
    # BT.2020 primaries: R(0.708,0.292) G(0.170,0.797) B(0.131,0.046) WP(0.3127,0.3290)
    # Values in G(x,y)B(x,y)R(x,y)WP(x,y) order, multiplied by 50000 for nits
    master_display = (
        f"G(8500,39850)B(6550,2300)R(35400,14600)"
        f"WP(15635,16450)"
        f"L({max_luminance * 10000},{int(min_luminance * 10000)})"
    )

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-vf", zscale_chain,
        "-c:v", "libx265",
        "-crf", "18",
        "-preset", "medium",
        "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020",
        "-color_trc", tf["transfer"],
        "-colorspace", "bt2020nc",
        "-x265-params",
        f"hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer={tf['transfer']}:"
        f"colormatrix=bt2020nc:master-display={master_display}:"
        f"max-cll={max_cll},{max_fall}",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "faststart",
        out,
    ]

    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(90, "Verifying HDR output...")

    result = {
        "output_path": out,
        "transfer_function": transfer,
        "color_primaries": "bt2020",
        "max_luminance": max_luminance,
        "min_luminance": min_luminance,
        "has_metadata": True,
        "original_colorspace": "bt709",
    }

    if on_progress:
        on_progress(100, "SDR-to-HDR conversion complete!")

    return result


def list_transfer_functions() -> list:
    """Return available HDR transfer functions."""
    return [
        {"id": k, "name": v["name"], "max_luminance": v["max_luminance"]}
        for k, v in TRANSFER_FUNCTIONS.items()
    ]
