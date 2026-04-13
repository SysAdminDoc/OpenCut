"""
OpenCut Glitch Effects Engine v1.0.0

Creative video glitch effects via FFmpeg:
- RGB split (rgbashift)
- Scan displacement (scroll)
- Block corruption (geq with noise)
- Noise burst (random segment heavy noise)
- Chromatic aberration (channel-specific blur/shift)
- Timeline-based glitch sequences

All FFmpeg-based, zero external dependencies.
"""

import logging
import os
import tempfile
from typing import Callable, List, Optional

from opencut.helpers import get_ffmpeg_path, get_video_info, output_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Glitch Effect Registry
# ---------------------------------------------------------------------------

GLITCH_EFFECTS = {
    "rgb_split": {
        "label": "RGB Split",
        "description": "Offset RGB channels for a chromatic split look.",
    },
    "scan_displacement": {
        "label": "Scan Displacement",
        "description": "Horizontal scan-line displacement for a VHS/digital glitch look.",
    },
    "block_corruption": {
        "label": "Block Corruption",
        "description": "Random block offset artifacts simulating data corruption.",
    },
    "noise_burst": {
        "label": "Noise Burst",
        "description": "Heavy static noise applied across the frame.",
    },
    "chromatic_aberration": {
        "label": "Chromatic Aberration",
        "description": "Lens-like color fringing with per-channel blur and shift.",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_glitch(
    input_path: str,
    effect: str = "rgb_split",
    intensity: float = 0.5,
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Apply a glitch effect to a video.

    Args:
        input_path: Path to source video.
        effect: Effect name (rgb_split, scan_displacement, block_corruption,
                noise_burst, chromatic_aberration).
        intensity: Effect strength 0.0-1.0.
        output_path_override: Custom output path. Auto-generated if None.
        on_progress: Callback(percent, message).

    Returns:
        dict with output_path, effect, intensity.
    """
    if effect not in GLITCH_EFFECTS:
        raise ValueError(
            f"Unknown glitch effect '{effect}'. "
            f"Available: {', '.join(GLITCH_EFFECTS.keys())}"
        )

    intensity = max(0.0, min(1.0, intensity))
    out = output_path_override or output_path(input_path, f"glitch_{effect}")
    info = get_video_info(input_path)

    if on_progress:
        on_progress(10, f"Applying {GLITCH_EFFECTS[effect]['label']}...")

    vf, fc = _build_glitch_filter(effect, intensity, info)

    cmd = [get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
           "-i", input_path]

    if fc:
        cmd += ["-filter_complex", fc, "-map", "[out]", "-map", "0:a?"]
    else:
        cmd += ["-vf", vf]

    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "copy", out]

    if on_progress:
        on_progress(30, "Encoding...")

    run_ffmpeg(cmd, timeout=7200)

    if on_progress:
        on_progress(100, f"{GLITCH_EFFECTS[effect]['label']} applied!")

    return {
        "output_path": out,
        "effect": effect,
        "effect_label": GLITCH_EFFECTS[effect]["label"],
        "intensity": intensity,
    }


def apply_glitch_sequence(
    input_path: str,
    effects_timeline: List[dict],
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Apply a sequence of glitch effects at specific time ranges.

    Args:
        input_path: Path to source video.
        effects_timeline: List of {effect, start_time, end_time, intensity}.
        output_path_override: Custom output path.
        on_progress: Callback(percent, message).

    Returns:
        dict with output_path, effects_applied count.
    """
    if not effects_timeline:
        raise ValueError("effects_timeline cannot be empty")

    for i, entry in enumerate(effects_timeline):
        if "effect" not in entry:
            raise ValueError(f"Entry {i} missing 'effect' key")
        if entry["effect"] not in GLITCH_EFFECTS:
            raise ValueError(f"Entry {i}: unknown effect '{entry['effect']}'")
        if "start_time" not in entry or "end_time" not in entry:
            raise ValueError(f"Entry {i} missing 'start_time' or 'end_time'")

    out = output_path_override or output_path(input_path, "glitch_seq")
    info = get_video_info(input_path)
    total = len(effects_timeline)
    tmp_dir = tempfile.mkdtemp(prefix="opencut_glitch_")

    try:
        # Strategy: apply effects one at a time using enable expressions
        # Build a single filter chain with time-based enable for each effect
        current_input = input_path

        for idx, entry in enumerate(effects_timeline):
            if on_progress:
                pct = 5 + int((idx / total) * 85)
                on_progress(pct, f"Applying effect {idx+1}/{total}...")

            effect = entry["effect"]
            start = float(entry["start_time"])
            end = float(entry["end_time"])
            eff_intensity = max(0.0, min(1.0, float(entry.get("intensity", 0.5))))

            seg_out = os.path.join(tmp_dir, f"pass_{idx:03d}.mp4")
            vf, fc = _build_glitch_filter(effect, eff_intensity, info)

            # Wrap filter with enable expression for time range
            if fc:
                # filter_complex doesn't support enable easily; apply to whole
                # and use trim + concat approach
                _apply_timed_effect_complex(
                    current_input, seg_out, fc, start, end, info["duration"],
                    tmp_dir, idx,
                )
            else:
                # Use enable= on the vf filter
                enabled_vf = f"{vf}:enable='between(t,{start},{end})'"
                cmd = [
                    get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
                    "-i", current_input,
                    "-vf", enabled_vf,
                    "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-c:a", "copy",
                    seg_out,
                ]
                run_ffmpeg(cmd, timeout=7200)

            current_input = seg_out

        # Copy final result to output
        if current_input != out:
            import shutil
            shutil.copy2(current_input, out)

        if on_progress:
            on_progress(100, f"Glitch sequence applied ({total} effects)!")

        return {
            "output_path": out,
            "effects_applied": total,
        }

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Internal: Build Filters Per Effect
# ---------------------------------------------------------------------------

def _build_glitch_filter(
    effect: str, intensity: float, info: dict,
) -> tuple:
    """Return (vf_string, filter_complex_string). One will be empty."""
    info.get("width", 1920)
    info.get("height", 1080)

    if effect == "rgb_split":
        # rgbashift: shift R and B channels horizontally
        offset = max(2, int(intensity * 30))
        vf = f"rgbashift=rh={offset}:bh={-offset}:rv={int(offset*0.3)}:bv={int(-offset*0.3)}"
        return vf, ""

    elif effect == "scan_displacement":
        # scroll filter for horizontal displacement
        h_speed = intensity * 0.08
        v_speed = intensity * 0.02
        vf = f"scroll=h={h_speed:.4f}:v={v_speed:.4f}"
        return vf, ""

    elif effect == "block_corruption":
        # geq with random block offsets simulating data corruption
        max(8, int(32 - intensity * 24))
        shift = max(2, int(intensity * 60))
        vf = (
            f"geq="
            f"lum='if(gt(random(1),{1.0 - intensity * 0.3:.2f}),"
            f"lum(X+{shift},Y),lum(X,Y))':"
            f"cb='cb(X,Y)':cr='cr(X,Y)'"
        )
        return vf, ""

    elif effect == "noise_burst":
        # Heavy noise filter
        strength = max(10, int(intensity * 80))
        vf = f"noise=alls={strength}:allf=t"
        return vf, ""

    elif effect == "chromatic_aberration":
        # Channel-specific blur and shift for lens-like aberration
        blur_amount = max(1, int(intensity * 5))
        shift = max(2, int(intensity * 15))
        fc = (
            f"[0:v]split=3[r][g][b];"
            f"[r]lutrgb=g=0:b=0,gblur=sigma={blur_amount},"
            f"rgbashift=rh={shift}:rv={int(shift*0.5)}[rc];"
            f"[g]lutrgb=r=0:b=0[gc];"
            f"[b]lutrgb=r=0:g=0,gblur=sigma={blur_amount},"
            f"rgbashift=bh={-shift}:bv={int(-shift*0.5)}[bc];"
            f"[rc][gc]blend=all_mode=addition[rg];"
            f"[rg][bc]blend=all_mode=addition[out]"
        )
        return "", fc

    # Fallback
    return "null", ""


def _apply_timed_effect_complex(
    input_path: str, output_path: str, fc: str,
    start: float, end: float, duration: float,
    tmp_dir: str, idx: int,
) -> None:
    """Apply a filter_complex effect only within a time range using trim/concat."""
    # Split into 3 parts: before, during (with effect), after
    parts = []

    # Before segment
    if start > 0.1:
        before_path = os.path.join(tmp_dir, f"before_{idx}.mp4")
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_path, "-t", str(start),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-c:a", "copy",
            before_path,
        ]
        run_ffmpeg(cmd, timeout=7200)
        parts.append(before_path)

    # Effect segment
    effect_path = os.path.join(tmp_dir, f"effect_{idx}.mp4")
    # Replace [0:v] with [0:v] in fc since we already feed the trimmed input
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-t", str(end - start),
        "-i", input_path,
        "-filter_complex", fc, "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        effect_path,
    ]
    run_ffmpeg(cmd, timeout=7200)
    parts.append(effect_path)

    # After segment
    if end < duration - 0.1:
        after_path = os.path.join(tmp_dir, f"after_{idx}.mp4")
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(end), "-i", input_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-c:a", "copy",
            after_path,
        ]
        run_ffmpeg(cmd, timeout=7200)
        parts.append(after_path)

    # Concatenate parts
    list_file = os.path.join(tmp_dir, f"concat_{idx}.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p}'\n")

    run_ffmpeg([
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-c", "copy", output_path,
    ], timeout=7200)
