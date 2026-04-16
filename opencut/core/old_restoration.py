"""
OpenCut Old Footage Restoration Pipeline

Chain-based restoration for vintage / damaged footage:
  stabilize -> deinterlace -> denoise -> upscale -> color restore
  -> frame interpolation

Presets: "VHS", "8mm Film", "Early Digital".
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import (
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

RESTORATION_PRESETS: Dict[str, Dict] = {
    "VHS": {
        "description": "VHS tape restoration — heavy denoise, deinterlace, color boost",
        "stabilize": True,
        "stabilize_smoothing": 20,
        "deinterlace": True,
        "deinterlace_method": "bwdif",
        "denoise_strength": "heavy",
        "denoise_filter": "hqdn3d=8:6:12:9",
        "upscale": True,
        "upscale_factor": 2,
        "color_restore": True,
        "color_filters": "eq=saturation=1.4:contrast=1.2:brightness=0.05,curves=vintage",
        "frame_interpolate": False,
        "target_fps": None,
    },
    "8mm Film": {
        "description": "8mm / Super8 film restoration — stabilize, grain reduce, warm color",
        "stabilize": True,
        "stabilize_smoothing": 30,
        "deinterlace": False,
        "deinterlace_method": "yadif",
        "denoise_strength": "medium",
        "denoise_filter": "hqdn3d=4:3:6:4.5",
        "upscale": True,
        "upscale_factor": 2,
        "color_restore": True,
        "color_filters": "eq=saturation=1.2:contrast=1.15:brightness=0.03,colorbalance=rs=0.05:gs=-0.02:bs=-0.05",
        "frame_interpolate": True,
        "target_fps": 24,
    },
    "Early Digital": {
        "description": "Early digital camera restoration — light denoise, sharpen, color fix",
        "stabilize": False,
        "stabilize_smoothing": 10,
        "deinterlace": True,
        "deinterlace_method": "yadif",
        "denoise_strength": "light",
        "denoise_filter": "hqdn3d=2:1.5:3:2.25",
        "upscale": True,
        "upscale_factor": 2,
        "color_restore": True,
        "color_filters": "eq=saturation=1.1:contrast=1.1,unsharp=3:3:0.5",
        "frame_interpolate": False,
        "target_fps": None,
    },
}


@dataclass
class RestorationResult:
    """Result of the full restoration pipeline."""
    output_path: str = ""
    preset: str = ""
    stages_applied: List[str] = field(default_factory=list)
    original_resolution: str = ""
    output_resolution: str = ""
    original_fps: float = 0.0
    output_fps: float = 0.0
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _stage_stabilize(input_path: str, out_path: str, smoothing: int = 20) -> str:
    """Two-pass vidstab stabilization."""
    transforms_file = out_path + ".trf"
    # Pass 1: detect
    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", f"vidstabdetect=shakiness=8:accuracy=15:result={transforms_file}",
        "-f", "null", "-",
    ], timeout=3600)
    # Pass 2: apply
    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", f"vidstabtransform=input={transforms_file}:smoothing={smoothing}:crop=black,unsharp=5:5:0.8:3:3:0.4",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ], timeout=3600)
    # Cleanup transforms file
    try:
        os.unlink(transforms_file)
    except OSError:
        pass
    return out_path


def _stage_deinterlace(input_path: str, out_path: str,
                       method: str = "bwdif") -> str:
    """Deinterlace with chosen method."""
    vf = f"{method}=mode=send_frame:parity=0:deint=all" if method != "yadif" \
        else "yadif=mode=0:parity=0:deint=0"
    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ], timeout=3600)
    return out_path


def _stage_denoise(input_path: str, out_path: str,
                   denoise_filter: str = "hqdn3d=4:3:6:4.5") -> str:
    """Temporal / spatial denoise."""
    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", denoise_filter,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ], timeout=3600)
    return out_path


def _stage_upscale(input_path: str, out_path: str, factor: int = 2) -> str:
    """Upscale using Lanczos resampling."""
    vf = f"scale=iw*{factor}:ih*{factor}:flags=lanczos"
    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ], timeout=7200)
    return out_path


def _stage_color_restore(input_path: str, out_path: str,
                         color_filters: str = "") -> str:
    """Apply color correction filters."""
    if not color_filters:
        color_filters = "eq=saturation=1.2:contrast=1.1"
    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", color_filters,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ], timeout=3600)
    return out_path


def _stage_frame_interpolate(input_path: str, out_path: str,
                             target_fps: float = 24) -> str:
    """Frame rate interpolation via minterpolate."""
    vf = f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ], timeout=7200)
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def list_presets() -> List[Dict]:
    """Return available restoration presets with descriptions."""
    return [
        {"name": name, "description": cfg["description"]}
        for name, cfg in RESTORATION_PRESETS.items()
    ]


def restore_old_footage(
    video_path: str,
    preset: str = "VHS",
    output_path_override: Optional[str] = None,
    custom_stages: Optional[Dict] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Run the full old-footage restoration pipeline.

    Chains stabilize -> deinterlace -> denoise -> upscale -> color restore
    -> frame interpolation, with each stage controlled by the selected
    preset or custom overrides.

    Args:
        video_path:  Path to input video.
        preset:  ``"VHS"``, ``"8mm Film"``, or ``"Early Digital"``.
        output_path_override:  Explicit output path.
        custom_stages:  Dict of stage overrides (keys match preset dict).
        on_progress:  Callback ``(pct, msg)``.

    Returns:
        dict with *output_path*, *preset*, *stages_applied*,
        *original_resolution*, *output_resolution*, *original_fps*,
        *output_fps*, *duration*.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    if preset not in RESTORATION_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(RESTORATION_PRESETS.keys())}")

    cfg = dict(RESTORATION_PRESETS[preset])
    if custom_stages:
        cfg.update(custom_stages)

    info = get_video_info(video_path)
    orig_w = info.get("width", 0)
    orig_h = info.get("height", 0)
    orig_fps = info.get("fps", 0)

    result = RestorationResult(
        preset=preset,
        original_resolution=f"{orig_w}x{orig_h}",
        original_fps=orig_fps,
        duration=info.get("duration", 0),
    )

    out = output_path_override or output_path(video_path, f"restored_{preset.lower().replace(' ', '_')}")
    current = video_path
    tmp_files = []

    stages = [
        ("stabilize", cfg.get("stabilize", False)),
        ("deinterlace", cfg.get("deinterlace", False)),
        ("denoise", True),
        ("upscale", cfg.get("upscale", False)),
        ("color_restore", cfg.get("color_restore", False)),
        ("frame_interpolate", cfg.get("frame_interpolate", False)),
    ]

    total_stages = sum(1 for _, enabled in stages if enabled)
    stage_idx = 0

    for stage_name, enabled in stages:
        if not enabled:
            continue

        stage_idx += 1
        base_pct = int((stage_idx - 1) / total_stages * 90) + 5
        if on_progress:
            on_progress(base_pct, f"Stage {stage_idx}/{total_stages}: {stage_name}...")

        _ntf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_out = _ntf.name
        _ntf.close()
        tmp_files.append(tmp_out)

        try:
            if stage_name == "stabilize":
                _stage_stabilize(current, tmp_out, cfg.get("stabilize_smoothing", 20))
            elif stage_name == "deinterlace":
                _stage_deinterlace(current, tmp_out, cfg.get("deinterlace_method", "bwdif"))
            elif stage_name == "denoise":
                _stage_denoise(current, tmp_out, cfg.get("denoise_filter", "hqdn3d=4:3:6:4.5"))
            elif stage_name == "upscale":
                _stage_upscale(current, tmp_out, cfg.get("upscale_factor", 2))
            elif stage_name == "color_restore":
                _stage_color_restore(current, tmp_out, cfg.get("color_filters", ""))
            elif stage_name == "frame_interpolate":
                target_fps = cfg.get("target_fps") or 24
                _stage_frame_interpolate(current, tmp_out, target_fps)

            result.stages_applied.append(stage_name)
            current = tmp_out
        except RuntimeError as e:
            logger.warning("Restoration stage '%s' failed: %s", stage_name, e)
            # Remove failed temp file; continue with current input
            try:
                os.unlink(tmp_out)
            except OSError:
                pass
            tmp_files.remove(tmp_out)

    # Final output
    if on_progress:
        on_progress(95, "Finalizing output...")

    if current != video_path:
        # Copy last stage output to final destination
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", current,
            "-c", "copy",
            "-movflags", "faststart",
            out,
        ], timeout=3600)
    else:
        # No stages succeeded — copy original
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path,
            "-c", "copy",
            out,
        ], timeout=3600)

    # Cleanup temp files
    for tf in tmp_files:
        try:
            os.unlink(tf)
        except OSError:
            pass

    # Get output info
    out_info = get_video_info(out)
    result.output_path = out
    result.output_resolution = f"{out_info.get('width', 0)}x{out_info.get('height', 0)}"
    result.output_fps = out_info.get("fps", orig_fps)

    if on_progress:
        on_progress(100, "Restoration complete!")

    return {
        "output_path": result.output_path,
        "preset": result.preset,
        "stages_applied": result.stages_applied,
        "original_resolution": result.original_resolution,
        "output_resolution": result.output_resolution,
        "original_fps": result.original_fps,
        "output_fps": result.output_fps,
        "duration": result.duration,
    }
