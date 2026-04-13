"""
OpenCut Film Stock Emulation Module v1.0.0

Emulate classic film stock looks via FFmpeg:
- Per-stock color grading (colorbalance + eq + curves)
- Film grain overlay (noise filter)
- Halation glow (highlight extraction + blur + screen blend)
- Presets: Kodak Portra 400, Kodak Ektar 100, Fuji Pro 400H,
  Kodak Vision3 500T, Fuji Velvia 50, Ilford HP5 (B&W)

All FFmpeg-based, zero external dependencies.
"""

import logging
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Film Stock Definitions
# ---------------------------------------------------------------------------

FILM_STOCKS: Dict[str, dict] = {
    "kodak_portra_400": {
        "label": "Kodak Portra 400",
        "description": "Warm tones, beautiful skin rendering, low contrast. Portrait classic.",
        "colorbalance": {"rs": 0.08, "gs": -0.02, "bs": -0.05,
                         "rm": 0.05, "gm": 0.0, "bm": -0.03,
                         "rh": 0.03, "gh": 0.01, "bh": -0.02},
        "eq": {"saturation": 0.9, "contrast": 0.95, "brightness": 0.02},
        "curves": "0/0 0.25/0.28 0.5/0.52 0.75/0.74 1/0.98",
        "grain_intensity": 0.3,
        "is_bw": False,
    },
    "kodak_ektar_100": {
        "label": "Kodak Ektar 100",
        "description": "Ultra-vivid colors, fine grain, punchy contrast. Landscape favorite.",
        "colorbalance": {"rs": 0.05, "gs": 0.0, "bs": -0.08,
                         "rm": 0.06, "gm": -0.01, "bm": -0.06,
                         "rh": 0.02, "gh": -0.02, "bh": -0.04},
        "eq": {"saturation": 1.35, "contrast": 1.15, "brightness": 0.0},
        "curves": "0/0 0.15/0.1 0.5/0.52 0.85/0.9 1/1",
        "grain_intensity": 0.15,
        "is_bw": False,
    },
    "fuji_pro_400h": {
        "label": "Fuji Pro 400H",
        "description": "Cool pastels, soft greens, gentle contrast. Wedding favorite.",
        "colorbalance": {"rs": -0.03, "gs": 0.04, "bs": 0.06,
                         "rm": -0.02, "gm": 0.03, "bm": 0.04,
                         "rh": -0.01, "gh": 0.02, "bh": 0.03},
        "eq": {"saturation": 0.85, "contrast": 0.92, "brightness": 0.03},
        "curves": "0/0.02 0.25/0.28 0.5/0.53 0.75/0.76 1/0.97",
        "grain_intensity": 0.25,
        "is_bw": False,
    },
    "kodak_vision3_500t": {
        "label": "Kodak Vision3 500T",
        "description": "Cinematic tungsten stock, teal shadows, warm highlights. Movie look.",
        "colorbalance": {"rs": -0.06, "gs": -0.02, "bs": 0.1,
                         "rm": -0.02, "gm": 0.0, "bm": 0.04,
                         "rh": 0.08, "gh": 0.03, "bh": -0.02},
        "eq": {"saturation": 0.95, "contrast": 1.05, "brightness": -0.02},
        "curves": "0/0 0.2/0.18 0.5/0.5 0.8/0.84 1/0.96",
        "grain_intensity": 0.4,
        "is_bw": False,
    },
    "fuji_velvia_50": {
        "label": "Fuji Velvia 50",
        "description": "Extreme saturation, deep blacks, vivid reds/greens. Slide film.",
        "colorbalance": {"rs": 0.06, "gs": -0.02, "bs": -0.04,
                         "rm": 0.04, "gm": 0.02, "bm": -0.03,
                         "rh": 0.02, "gh": -0.01, "bh": -0.02},
        "eq": {"saturation": 1.5, "contrast": 1.2, "brightness": -0.02},
        "curves": "0/0 0.1/0.05 0.5/0.52 0.9/0.95 1/1",
        "grain_intensity": 0.1,
        "is_bw": False,
    },
    "ilford_hp5": {
        "label": "Ilford HP5 Plus 400",
        "description": "Classic B&W, rich mid-tones, strong grain. Photojournalism staple.",
        "colorbalance": {"rs": 0.0, "gs": 0.0, "bs": 0.0,
                         "rm": 0.0, "gm": 0.0, "bm": 0.0,
                         "rh": 0.0, "gh": 0.0, "bh": 0.0},
        "eq": {"saturation": 0.0, "contrast": 1.2, "brightness": 0.0},
        "curves": "0/0 0.15/0.1 0.5/0.55 0.85/0.92 1/1",
        "grain_intensity": 0.55,
        "is_bw": True,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_film_stocks() -> List[dict]:
    """Return list of available film stock presets with metadata."""
    result = []
    for name, info in FILM_STOCKS.items():
        result.append({
            "name": name,
            "label": info["label"],
            "description": info["description"],
            "is_bw": info["is_bw"],
            "grain_intensity": info["grain_intensity"],
        })
    return result


def apply_film_stock(
    input_path: str,
    stock: str = "kodak_portra_400",
    grain_amount: float = 0.5,
    halation: float = 0.3,
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Apply film stock emulation to a video.

    Args:
        input_path: Path to source video.
        stock: Film stock preset name (see FILM_STOCKS).
        grain_amount: Grain intensity multiplier (0.0-1.0).
        halation: Halation glow strength (0.0-1.0). 0 disables.
        output_path_override: Custom output path. Auto-generated if None.
        on_progress: Callback(percent, message).

    Returns:
        dict with output_path, stock, grain_amount, halation.
    """
    if stock not in FILM_STOCKS:
        raise ValueError(f"Unknown film stock '{stock}'. Available: {', '.join(FILM_STOCKS.keys())}")

    grain_amount = max(0.0, min(1.0, grain_amount))
    halation = max(0.0, min(1.0, halation))

    preset = FILM_STOCKS[stock]
    out = output_path_override or output_path(input_path, f"film_{stock}")

    if on_progress:
        on_progress(5, f"Applying {preset['label']} film stock...")

    info = get_video_info(input_path)

    # Build filter_complex for halation, or simple -vf for non-halation
    use_halation = halation > 0.05 and not preset["is_bw"]

    # --- Color grading filters ---
    color_filters = _build_color_filters(preset, grain_amount)

    if use_halation:
        if on_progress:
            on_progress(20, "Building halation glow...")
        fc = _build_halation_filter_complex(preset, color_filters, halation, info)
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-filter_complex", fc,
            "-map", "[final]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "copy",
            out,
        ]
    else:
        if on_progress:
            on_progress(20, "Applying color grading...")
        vf = ",".join(color_filters) if color_filters else "null"
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "copy",
            out,
        ]

    if on_progress:
        on_progress(40, "Encoding...")

    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(100, f"{preset['label']} applied!")

    return {
        "output_path": out,
        "stock": stock,
        "stock_label": preset["label"],
        "grain_amount": grain_amount,
        "halation": halation,
    }


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _build_color_filters(preset: dict, grain_amount: float) -> list:
    """Build the chain of FFmpeg video filters for color + grain."""
    filters = []

    # Desaturate first for B&W stocks
    if preset["is_bw"]:
        filters.append("format=gray")
        filters.append("format=yuv420p")

    # EQ filter (brightness, contrast, saturation)
    eq = preset["eq"]
    eq_parts = []
    if abs(eq.get("brightness", 0)) > 0.005:
        eq_parts.append(f"brightness={eq['brightness']}")
    if abs(eq.get("contrast", 1.0) - 1.0) > 0.005:
        eq_parts.append(f"contrast={eq['contrast']}")
    if abs(eq.get("saturation", 1.0) - 1.0) > 0.005:
        eq_parts.append(f"saturation={eq['saturation']}")
    if eq_parts:
        filters.append(f"eq={':'.join(eq_parts)}")

    # Colorbalance
    cb = preset["colorbalance"]
    cb_parts = []
    for k, v in cb.items():
        if abs(v) > 0.005:
            cb_parts.append(f"{k}={v}")
    if cb_parts:
        filters.append(f"colorbalance={':'.join(cb_parts)}")

    # Curves (tonal response)
    curves_str = preset.get("curves", "")
    if curves_str:
        filters.append(f"curves=master='{curves_str}'")

    # Film grain via noise filter
    effective_grain = preset["grain_intensity"] * grain_amount
    if effective_grain > 0.02:
        # noise filter: alls = strength for all planes, allf = temporal flag
        strength = int(effective_grain * 40)  # 0-20 range
        strength = max(1, min(25, strength))
        filters.append(f"noise=alls={strength}:allf=t")

    return filters


def _build_halation_filter_complex(
    preset: dict, color_filters: list, halation: float, info: dict,
) -> str:
    """Build filter_complex string with halation (highlight glow) overlay."""
    color_chain = ",".join(color_filters) if color_filters else "null"
    # Blur radius for halation proportional to resolution
    blur_radius = max(10, int(min(info["width"], info["height"]) * 0.02 * halation))
    opacity = min(0.6, halation * 0.5)

    fc = (
        f"[0:v]{color_chain}[graded];"
        f"[graded]split[main][hl];"
        f"[hl]colorlevels=rimin=0.7:gimin=0.7:bimin=0.7,"
        f"gblur=sigma={blur_radius}[glow];"
        f"[main][glow]blend=all_mode=screen:all_opacity={opacity:.2f}[final]"
    )
    return fc
