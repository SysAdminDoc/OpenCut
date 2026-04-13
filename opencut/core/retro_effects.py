"""
OpenCut Retro / Analog Effects Pack

Authentic vintage video effects using FFmpeg filters:
- VHS tape simulation (noise, chroma shift, scanlines, date stamp)
- Super 8mm film (grain, vignette, warm tint, gate weave)
- Film damage (scratches, color fade, grain overlay)
- Old TV (static noise, interlace, low resolution)
- Tilt-Shift miniature effect
- Light leaks and lens flares

Uses FFmpeg only -- no additional dependencies required.
"""

import logging
import os
from typing import Callable, Optional

from opencut.helpers import (
    FFmpegCmd,
    get_video_info,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")

VALID_RETRO_EFFECTS = ("vhs", "super8", "film_damage", "old_tv")
VALID_LIGHT_LEAK_STYLES = ("warm_amber", "rainbow_prism", "cool_blue", "golden_hour", "film_edge")


# ---------------------------------------------------------------------------
# Retro / Analog Effect
# ---------------------------------------------------------------------------
def apply_retro_effect(
    input_path: str,
    effect: str = "vhs",
    intensity: float = 0.7,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Apply a retro / analog video effect.

    Args:
        input_path: Source video file.
        effect: One of "vhs", "super8", "film_damage", "old_tv".
        intensity: Effect strength 0.0-1.0 (default 0.7).
        output_path: Explicit output path.  Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with ``output_path``.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if effect not in VALID_RETRO_EFFECTS:
        raise ValueError(f"Invalid effect '{effect}'. Must be one of: {VALID_RETRO_EFFECTS}")

    intensity = max(0.0, min(1.0, float(intensity)))

    if output_path is None:
        output_path = _output_path(input_path, f"retro_{effect}")

    info = get_video_info(input_path)
    w, h = info["width"], info["height"]

    if on_progress:
        on_progress(10, f"Applying {effect} effect...")

    vf = _build_retro_filter(effect, intensity, w, h)

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .video_filter(vf)
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("copy")
        .faststart()
        .output(output_path)
        .build()
    )

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, f"{effect.upper()} effect applied")

    return {"output_path": output_path}


def _build_retro_filter(effect: str, intensity: float, w: int, h: int) -> str:
    """Build the FFmpeg -vf filter chain for the given retro effect."""
    # Scale intensity parameters
    noise_amount = int(20 * intensity)
    chroma_shift = int(4 * intensity)
    desat = 1.0 - (0.4 * intensity)  # partial desaturation
    grain_amount = int(15 * intensity)
    blur_amount = max(1, int(2 * intensity))

    if effect == "vhs":
        # VHS: noise + chromatic aberration + scanlines + date stamp + desaturation
        parts = []
        # Chroma shift (horizontal offset of color channels)
        if chroma_shift > 0:
            parts.append(f"chromashift=cbh={chroma_shift}:crh=-{chroma_shift}")
        # Noise
        parts.append(f"noise=alls={noise_amount}:allf=t")
        # Desaturation via eq
        parts.append(f"eq=saturation={desat:.2f}:contrast=1.1")
        # Scanlines via geq (darken every other line)
        scanline_strength = int(30 * intensity)
        parts.append(
            f"geq=lum='lum(X,Y)-{scanline_strength}*(mod(Y,2))':cb='cb(X,Y)':cr='cr(X,Y)'"
        )
        # VHS date stamp
        parts.append(
            f"drawtext=text='REC  %{{pts\\:hms}}':fontsize={max(16, h // 30)}:"
            f"fontcolor=white@0.8:x=w-tw-20:y=h-th-20:font=monospace"
        )
        return ",".join(parts)

    elif effect == "super8":
        # Super 8: grain + vignette + warm tint + gate weave + saturation boost
        parts = []
        # Warm color tint via colorbalance
        warm = 0.15 * intensity
        parts.append(f"colorbalance=rs={warm:.2f}:gs=-{warm / 2:.2f}:bs=-{warm:.2f}")
        # Film grain
        parts.append(f"noise=alls={grain_amount}:allf=t")
        # Vignette
        parts.append("vignette=PI/4")
        # Slight blur for soft-focus look
        if blur_amount > 0:
            parts.append(f"boxblur={blur_amount}:{blur_amount}:1")
        # Gate weave (subtle random translate)
        weave = max(1, int(3 * intensity))
        parts.append(
            f"geq=lum='lum(X+{weave}*sin(N*0.5),Y+{weave}*cos(N*0.7))':"
            f"cb='cb(X+{weave}*sin(N*0.5),Y+{weave}*cos(N*0.7))':"
            f"cr='cr(X+{weave}*sin(N*0.5),Y+{weave}*cos(N*0.7))'"
        )
        return ",".join(parts)

    elif effect == "film_damage":
        # Film damage: grain + scratches + color fade + vignette
        parts = []
        # Color fade (reduce saturation, slight sepia shift)
        fade_sat = max(0.3, 1.0 - 0.6 * intensity)
        parts.append(f"eq=saturation={fade_sat:.2f}:gamma=1.1")
        # Warm fade via colorbalance
        tint = 0.1 * intensity
        parts.append(f"colorbalance=rs={tint:.2f}:gs={tint / 2:.2f}:bs=-{tint:.2f}")
        # Grain noise
        parts.append(f"noise=alls={grain_amount}:allf=t")
        # Vignette (heavy for damaged look)
        parts.append("vignette=PI/3")
        # Simulated scratches via thin vertical drawbox lines
        scratch_alpha = intensity * 0.4
        for offset in [0.2, 0.45, 0.7]:
            x_pos = int(w * offset)
            parts.append(
                f"drawbox=x={x_pos}:y=0:w=1:h=ih:color=white@{scratch_alpha:.2f}:t=fill"
            )
        return ",".join(parts)

    elif effect == "old_tv":
        # Old TV: static noise + interlace + low resolution + slight horizontal hold
        parts = []
        # Low resolution (downscale then upscale for pixelated look)
        low_w = max(320, w // 3)
        low_h = max(240, h // 3)
        parts.append(f"scale={low_w}:{low_h}:flags=neighbor,scale={w}:{h}:flags=neighbor")
        # Heavy noise (TV static)
        tv_noise = int(40 * intensity)
        parts.append(f"noise=alls={tv_noise}:allf=t")
        # Interlace simulation via tinterlace
        parts.append("tinterlace=mode=interleave_top")
        # Desaturation + low contrast
        parts.append(f"eq=saturation={max(0.2, 1.0 - 0.7 * intensity):.2f}:contrast=0.9")
        # Scanlines
        scanline = int(40 * intensity)
        parts.append(
            f"geq=lum='lum(X,Y)-{scanline}*(mod(Y,2))':cb='cb(X,Y)':cr='cr(X,Y)'"
        )
        return ",".join(parts)

    raise ValueError(f"Unknown effect: {effect}")


# ---------------------------------------------------------------------------
# Tilt-Shift Miniature Effect
# ---------------------------------------------------------------------------
def apply_tilt_shift(
    input_path: str,
    focus_y: float = 0.5,
    focus_width: float = 0.3,
    blur_amount: int = 10,
    saturation: float = 1.5,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Apply a tilt-shift miniature effect.

    Creates a selective-focus look where a horizontal band stays sharp
    and the top/bottom are progressively blurred, with boosted saturation
    to enhance the miniature illusion.

    Args:
        input_path: Source video file.
        focus_y: Vertical position of the focus band (0.0=top, 1.0=bottom).
        focus_width: Width of the sharp band as fraction of frame height.
        blur_amount: Blur strength for out-of-focus areas (1-30).
        saturation: Color saturation multiplier (1.0=normal, 1.5=boosted).
        output_path: Explicit output path.  Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with ``output_path``.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    focus_y = max(0.0, min(1.0, float(focus_y)))
    focus_width = max(0.05, min(1.0, float(focus_width)))
    blur_amount = max(1, min(30, int(blur_amount)))
    saturation = max(0.5, min(3.0, float(saturation)))

    if output_path is None:
        output_path = _output_path(input_path, "tiltshift")

    info = get_video_info(input_path)
    h = info["height"]

    if on_progress:
        on_progress(10, "Applying tilt-shift effect...")

    # Calculate focus band boundaries in pixels
    band_half = int(h * focus_width / 2)
    center_y = int(h * focus_y)
    top = max(0, center_y - band_half)
    bottom = min(h, center_y + band_half)

    # Use split + crop + boxblur + overlay approach via filter_complex.
    # Sharp center band, blurred top/bottom regions composited together.
    # Gradient mask via geq creates smooth transition at band edges.
    #
    # Approach: blur entire frame, then use geq to blend sharp vs blurred
    # based on Y position (gradient mask).
    #
    # The geq computes a blend factor: 1.0 inside the band, fading to 0.0
    # outside.  We use the alpha plane as a mask.
    transition = max(1, int(h * focus_width * 0.3))  # soft edge

    # Build filter: split into sharp and blurred, blend with gradient mask
    fc = (
        f"[0:v]split=2[sharp][blur];"
        f"[blur]boxblur={blur_amount}:{blur_amount}:1[blurred];"
        f"[sharp]format=yuva420p,"
        f"geq="
        f"lum='lum(X,Y)':"
        f"cb='cb(X,Y)':"
        f"cr='cr(X,Y)':"
        f"a='if(between(Y,{top},{bottom}),255,"
        f"if(lt(Y,{top}),clip(255*(1-({top}-Y)/{max(1, transition)}),0,255),"
        f"clip(255*(1-(Y-{bottom})/{max(1, transition)}),0,255)))'"
        f"[masked];"
        f"[blurred][masked]overlay=0:0:format=auto[tilt];"
        f"[tilt]eq=saturation={saturation:.2f}[outv]"
    )

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .filter_complex(fc, maps=["[outv]", "0:a?"])
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(output_path)
        .build()
    )

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, "Tilt-shift effect applied")

    return {"output_path": output_path}


# ---------------------------------------------------------------------------
# Light Leaks & Lens Flares
# ---------------------------------------------------------------------------
def apply_light_leak(
    input_path: str,
    style: str = "warm_amber",
    intensity: float = 0.5,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Apply a procedural light leak / lens flare overlay.

    Generates a synthetic color light source via FFmpeg, blurs it, and
    blends it over the input video for an organic film look.

    Args:
        input_path: Source video file.
        style: Light leak style -- "warm_amber", "rainbow_prism", "cool_blue",
               "golden_hour", or "film_edge".
        intensity: Blend intensity 0.0-1.0 (default 0.5).
        output_path: Explicit output path.  Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        dict with ``output_path``.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if style not in VALID_LIGHT_LEAK_STYLES:
        raise ValueError(f"Invalid style '{style}'. Must be one of: {VALID_LIGHT_LEAK_STYLES}")

    intensity = max(0.0, min(1.0, float(intensity)))

    if output_path is None:
        output_path = _output_path(input_path, f"lightleak_{style}")

    info = get_video_info(input_path)
    w, h = info["width"], info["height"]
    fps = info["fps"]
    duration = info["duration"]

    if on_progress:
        on_progress(10, f"Generating {style} light leak...")

    # Define color and position for each style
    color_hex, blur_radius, position = _light_leak_params(style, w, h, intensity)

    # Generate a synthetic light leak:
    # 1. color source at specified color
    # 2. gaussian blur for soft glow
    # 3. overlay with screen/additive blend onto original
    opacity = intensity * 0.7  # keep it subtle

    # Build filter_complex: generate color overlay, blur, blend
    px, py = position
    box_w = max(1, int(w * 0.4))
    box_h = max(1, int(h * 0.6))
    blur_str = f"{blur_radius}:{blur_radius}"

    fc = (
        f"color=c={color_hex}:s={w}x{h}:r={fps:.2f}:d={duration:.2f},"
        f"drawbox=x={px}:y={py}:w={box_w}:h={box_h}:color={color_hex}@{opacity:.2f}:t=fill,"
        f"boxblur={blur_str}:1"
        f"[leak];"
        f"[0:v][leak]blend=all_mode=screen:all_opacity={opacity:.2f}[outv]"
    )

    cmd = (
        FFmpegCmd()
        .input(input_path)
        .filter_complex(fc, maps=["[outv]", "0:a?"])
        .video_codec("libx264", crf=18, preset="fast")
        .audio_codec("aac", bitrate="192k")
        .faststart()
        .output(output_path)
        .build()
    )

    run_ffmpeg(cmd, timeout=3600)

    if on_progress:
        on_progress(100, f"{style} light leak applied")

    return {"output_path": output_path}


def _light_leak_params(style: str, w: int, h: int, intensity: float):
    """Return (color_hex, blur_radius, (x, y)) for a light leak style."""
    blur_base = max(20, int(min(w, h) * 0.15))
    blur = int(blur_base * (0.5 + intensity * 0.5))

    if style == "warm_amber":
        return "0xFF8C00", blur, (int(w * 0.6), int(h * 0.2))
    elif style == "rainbow_prism":
        return "0xFF69B4", blur, (int(w * 0.3), int(h * 0.3))
    elif style == "cool_blue":
        return "0x4169E1", blur, (int(w * 0.1), int(h * 0.4))
    elif style == "golden_hour":
        return "0xFFD700", blur, (int(w * 0.5), int(h * 0.1))
    elif style == "film_edge":
        return "0xFF4500", blur, (0, int(h * 0.3))
    else:
        return "0xFF8C00", blur, (int(w * 0.5), int(h * 0.3))
