"""
OpenCut ACES Color Pipeline (43.1)

Academy Color Encoding System (ACES) workflow:
- IDT (Input Device Transform) per camera
- ACEScg processing space
- ODT (Output Device Transform) per delivery target

Uses FFmpeg LUT3D and colorspace conversion filters.
"""

import logging
import os
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

from opencut.helpers import FFmpegCmd, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Constants: Available IDTs and ODTs
# ---------------------------------------------------------------------------
AVAILABLE_IDTS = {
    "srgb": {"name": "sRGB / Rec.709", "matrix": "bt709", "trc": "iec61966-2-1"},
    "rec709": {"name": "Rec.709", "matrix": "bt709", "trc": "bt709"},
    "rec2020": {"name": "Rec.2020", "matrix": "bt2020nc", "trc": "bt2020-10"},
    "dcip3": {"name": "DCI-P3", "matrix": "bt709", "trc": "smpte2084"},
    "slog3": {"name": "Sony S-Log3 / S-Gamut3", "matrix": "bt2020nc", "trc": "arib-std-b67"},
    "logc": {"name": "ARRI LogC / AWG", "matrix": "bt2020nc", "trc": "arib-std-b67"},
    "vlog": {"name": "Panasonic V-Log / V-Gamut", "matrix": "bt2020nc", "trc": "arib-std-b67"},
    "clog": {"name": "Canon C-Log", "matrix": "bt709", "trc": "bt709"},
    "bmdfilm": {"name": "Blackmagic Film", "matrix": "bt709", "trc": "bt709"},
    "redlog3g10": {"name": "RED Log3G10 / REDWideGamutRGB", "matrix": "bt2020nc", "trc": "arib-std-b67"},
    "acescg": {"name": "ACEScg (linear)", "matrix": "bt2020nc", "trc": "linear"},
}

AVAILABLE_ODTS = {
    "srgb": {"name": "sRGB Display", "matrix": "bt709", "trc": "iec61966-2-1"},
    "rec709": {"name": "Rec.709 (SDR TV)", "matrix": "bt709", "trc": "bt709"},
    "rec2020_pq": {"name": "Rec.2020 PQ (HDR10)", "matrix": "bt2020nc", "trc": "smpte2084"},
    "rec2020_hlg": {"name": "Rec.2020 HLG (HDR)", "matrix": "bt2020nc", "trc": "arib-std-b67"},
    "dcip3_d65": {"name": "DCI-P3 D65", "matrix": "bt709", "trc": "smpte2084"},
    "dcip3_theater": {"name": "DCI-P3 Theater", "matrix": "bt709", "trc": "smpte2084"},
    "acescg": {"name": "ACEScg (linear)", "matrix": "bt2020nc", "trc": "linear"},
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ACESConfig:
    """ACES pipeline configuration."""
    idt: str = "srgb"
    odt: str = "rec709"
    exposure: float = 0.0
    white_balance: float = 6500.0
    look: str = ""
    tonemap: str = "hable"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ACESResult:
    """Result of an ACES pipeline operation."""
    output_path: str = ""
    config: Optional[Dict] = None
    idt_name: str = ""
    odt_name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_idt_info(idt_key: str) -> dict:
    """Resolve IDT key to its parameters."""
    return AVAILABLE_IDTS.get(idt_key, AVAILABLE_IDTS["srgb"])


def _get_odt_info(odt_key: str) -> dict:
    """Resolve ODT key to its parameters."""
    return AVAILABLE_ODTS.get(odt_key, AVAILABLE_ODTS["rec709"])


def _build_aces_filter(config: ACESConfig) -> str:
    """Build FFmpeg filter chain for ACES pipeline.

    Pipeline: IDT -> ACEScg (linear) -> [look/exposure] -> ODT
    """
    idt = _get_idt_info(config.idt)
    odt = _get_odt_info(config.odt)

    filters = []

    # Step 1: IDT - Convert from camera space to linear ACEScg working space
    # Use zscale for colorspace conversion
    filters.append(
        f"zscale=transfer=linear:matrix=bt2020nc:primaries=bt2020"
        f":transferin={idt['trc']}:matrixin={idt['matrix']}"
    )

    # Step 2: Exposure adjustment in linear space
    if config.exposure != 0.0:
        # Exposure in stops: multiply by 2^exposure
        import math
        mult = math.pow(2.0, max(-5.0, min(5.0, config.exposure)))
        filters.append(f"colorlevels=rimin=0:rimax={1.0/mult}:gimin=0:gimax={1.0/mult}:bimin=0:bimax={1.0/mult}")

    # Step 3: White balance adjustment
    if config.white_balance != 6500.0:
        # Shift from 6500K reference, warm (lower) adds red, cool (higher) adds blue
        temp_offset = (config.white_balance - 6500.0) / 6500.0
        r_adj = max(-0.3, min(0.3, temp_offset * 0.15))
        b_adj = max(-0.3, min(0.3, -temp_offset * 0.15))
        filters.append(f"colorbalance=rs={r_adj}:bs={b_adj}")

    # Step 4: Tonemap from linear to display
    if config.tonemap and odt["trc"] != "linear":
        tonemap = config.tonemap if config.tonemap in (
            "none", "clip", "linear", "gamma", "reinhard", "hable",
            "mobius"
        ) else "hable"
        filters.append(f"tonemap={tonemap}:desat=2")

    # Step 5: ODT - Convert from working space to output space
    filters.append(
        f"zscale=transfer={odt['trc']}:matrix={odt['matrix']}"
        f":primaries=bt709"
    )

    # Ensure compatible pixel format
    filters.append("format=yuv420p")

    return ",".join(filters)


# ---------------------------------------------------------------------------
# Detect Camera IDT
# ---------------------------------------------------------------------------
def detect_camera_idt(
    video_path: str,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Auto-detect the appropriate IDT for a video file based on metadata.

    Analyzes color space, transfer characteristics, and camera metadata
    to recommend the best IDT.

    Args:
        video_path: Source video file.
        on_progress: Callback(percent, message).

    Returns:
        IDT key string (e.g., 'slog3', 'rec709').
    """
    if on_progress:
        on_progress(10, "Detecting camera IDT...")

    info = get_video_info(video_path)
    # Try to detect from stream metadata
    color_space = info.get("color_space", "").lower()
    color_trc = info.get("color_transfer", "").lower()

    # Match known camera log curves
    if "slog" in color_trc or "s-log" in color_trc:
        detected = "slog3"
    elif "logc" in color_trc or "arri" in color_trc:
        detected = "logc"
    elif "vlog" in color_trc or "v-log" in color_trc:
        detected = "vlog"
    elif "clog" in color_trc or "c-log" in color_trc:
        detected = "clog"
    elif "redlog" in color_trc:
        detected = "redlog3g10"
    elif "bmdfilm" in color_trc or "blackmagic" in color_trc:
        detected = "bmdfilm"
    elif "bt2020" in color_space or "rec2020" in color_space:
        detected = "rec2020"
    elif "p3" in color_space or "dci" in color_space:
        detected = "dcip3"
    elif "bt709" in color_space or "rec709" in color_trc:
        detected = "rec709"
    else:
        detected = "srgb"

    if on_progress:
        name = AVAILABLE_IDTS.get(detected, {}).get("name", detected)
        on_progress(100, f"Detected IDT: {name}")

    return detected


# ---------------------------------------------------------------------------
# Apply ACES Pipeline
# ---------------------------------------------------------------------------
def apply_aces_pipeline(
    video_path: str,
    idt: str = "srgb",
    odt: str = "rec709",
    output_path: Optional[str] = None,
    output_dir: str = "",
    config: Optional[ACESConfig] = None,
    on_progress: Optional[Callable] = None,
) -> ACESResult:
    """
    Apply the full ACES color pipeline to a video.

    Args:
        video_path: Source video file.
        idt: Input Device Transform key.
        odt: Output Device Transform key.
        output_path: Explicit output file path.
        config: ACESConfig dataclass (overrides idt/odt).
        on_progress: Callback(percent, message).

    Returns:
        ACESResult with output path and config.
    """
    if isinstance(config, ACESConfig):
        cfg = config
    elif isinstance(config, dict):
        cfg = ACESConfig(
            idt=config.get("idt", idt),
            odt=config.get("odt", odt),
            exposure=float(config.get("exposure", 0)),
            white_balance=float(config.get("white_balance", 6500)),
            look=str(config.get("look", "")),
            tonemap=str(config.get("tonemap", "hable")),
        )
    else:
        cfg = ACESConfig(idt=idt, odt=odt)

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_aces.mp4")

    if on_progress:
        idt_name = _get_idt_info(cfg.idt).get("name", cfg.idt)
        odt_name = _get_odt_info(cfg.odt).get("name", cfg.odt)
        on_progress(10, f"ACES pipeline: {idt_name} -> {odt_name}...")

    vf = _build_aces_filter(cfg)

    cmd = (
        FFmpegCmd()
        .input(video_path)
        .video_filter(vf)
        .video_codec("libx264", crf=16, preset="medium")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(output_path)
        .build()
    )
    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100, "ACES pipeline applied.")

    return ACESResult(
        output_path=output_path,
        config=cfg.to_dict(),
        idt_name=_get_idt_info(cfg.idt).get("name", cfg.idt),
        odt_name=_get_odt_info(cfg.odt).get("name", cfg.odt),
    )


# ---------------------------------------------------------------------------
# List Available IDTs / ODTs
# ---------------------------------------------------------------------------
def list_available_idts() -> List[Dict]:
    """Return list of available Input Device Transforms."""
    return [{"key": k, **v} for k, v in AVAILABLE_IDTS.items()]


def list_available_odts() -> List[Dict]:
    """Return list of available Output Device Transforms."""
    return [{"key": k, **v} for k, v in AVAILABLE_ODTS.items()]
