"""
OpenCut One-Click Enhance Pipeline

Single-button enhancement: analyze -> denoise audio -> stabilize ->
denoise video -> auto color -> upscale.  CapCut's most popular feature.

Each step runs only if analysis detects the corresponding issue.
"""

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from opencut.helpers import (
    get_ffmpeg_path,
    get_video_info,
    run_ffmpeg,
)
from opencut.helpers import (
    output_path as _output_path,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class EnhanceResult:
    """Result of one-click enhance pipeline."""
    output_path: str = ""
    steps_applied: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
_PRESETS = {
    "fast": {
        "skip_upscale": True,
        "denoise_strength": "light",
        "stabilize": True,
        "color_correct": True,
        "audio_denoise": True,
        "crf": 20,
        "preset": "fast",
    },
    "balanced": {
        "skip_upscale": False,
        "denoise_strength": "medium",
        "stabilize": True,
        "color_correct": True,
        "audio_denoise": True,
        "crf": 18,
        "preset": "medium",
    },
    "quality": {
        "skip_upscale": False,
        "denoise_strength": "strong",
        "stabilize": True,
        "color_correct": True,
        "audio_denoise": True,
        "crf": 16,
        "preset": "slow",
    },
}


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def _analyze_clip(video_path: str, on_progress: Optional[Callable] = None) -> dict:
    """Detect issues in the clip to decide which steps to apply."""
    issues = {
        "is_noisy_audio": False,
        "is_shaky": False,
        "is_grainy": False,
        "needs_color_fix": False,
        "needs_upscale": False,
        "avg_brightness": 128.0,
        "noise_level": 0.0,
    }

    info = get_video_info(video_path)
    width = info.get("width", 1920)
    height = info.get("height", 1080)

    # Check resolution -- needs upscale if < 1080p
    if height < 1080 and width < 1920:
        issues["needs_upscale"] = True

    # Analyze noise and brightness via FFmpeg signalstats
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-hide_banner", "-i", video_path,
        "-vf", "signalstats=stat=tout+vrep+brng,metadata=mode=print",
        "-t", "5",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stderr = result.stderr

        import re
        # Parse YAVG (average luminance)
        yavg_vals = re.findall(r"YAVG=(\d+\.?\d*)", stderr)
        if yavg_vals:
            avg_y = sum(float(v) for v in yavg_vals) / len(yavg_vals)
            issues["avg_brightness"] = round(avg_y, 1)
            # Poor exposure suggests color correction needed
            if avg_y < 80 or avg_y > 200:
                issues["needs_color_fix"] = True

        # Parse TOUT (temporal outliers -- high = noise or grain)
        tout_vals = re.findall(r"TOUT=(\d+\.?\d*)", stderr)
        if tout_vals:
            avg_tout = sum(float(v) for v in tout_vals) / len(tout_vals)
            issues["noise_level"] = round(avg_tout, 4)
            if avg_tout > 0.02:
                issues["is_grainy"] = True

        # Parse VREP (vertical line repetitions -- low = possible shake)
        vrep_vals = re.findall(r"VREP=(\d+\.?\d*)", stderr)
        if vrep_vals:
            avg_vrep = sum(float(v) for v in vrep_vals) / len(vrep_vals)
            if avg_vrep < 0.01:
                issues["is_shaky"] = True

    except Exception as exc:
        logger.debug("Clip analysis failed: %s", exc)
        # Conservative defaults: assume issues exist
        issues["is_grainy"] = True
        issues["needs_color_fix"] = True

    # Audio noise check via volumedetect
    cmd_audio = [
        ffmpeg, "-hide_banner", "-i", video_path,
        "-af", "volumedetect", "-vn", "-t", "10", "-f", "null", "-",
    ]
    try:
        result_audio = subprocess.run(cmd_audio, capture_output=True, text=True, timeout=30)
        stderr_audio = result_audio.stderr
        import re as _re
        mean_match = _re.search(r"mean_volume:\s*(-?[\d.]+)", stderr_audio)
        if mean_match:
            mean_vol = float(mean_match.group(1))
            # Very quiet audio with high dynamic range suggests noise
            if mean_vol < -35:
                issues["is_noisy_audio"] = True
    except Exception:
        pass

    return issues


def _stabilize_video(input_path: str, output_path: str,
                     on_progress: Optional[Callable] = None) -> None:
    """Stabilize video using FFmpeg vidstab (two-pass)."""
    fd, transforms_file = tempfile.mkstemp(suffix=".trf", prefix="opencut_stab_")
    os.close(fd)

    try:
        if on_progress:
            on_progress(0, "Stabilize pass 1: detecting motion...")

        # Pass 1: detect
        cmd1 = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", input_path,
            "-vf", f"vidstabdetect=shakiness=5:accuracy=15:result={transforms_file}",
            "-f", "null", "-",
        ]
        run_ffmpeg(cmd1)

        if on_progress:
            on_progress(50, "Stabilize pass 2: applying transforms...")

        # Pass 2: apply
        cmd2 = [
            get_ffmpeg_path(), "-hide_banner", "-y",
            "-i", input_path,
            "-vf", f"vidstabtransform=input={transforms_file}:smoothing=10:interpol=bicubic,unsharp=5:5:0.8:3:3:0.4",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            output_path,
        ]
        run_ffmpeg(cmd2)
    finally:
        try:
            os.unlink(transforms_file)
        except OSError:
            pass


def _denoise_video(input_path: str, output_path: str, strength: str = "medium") -> None:
    """Denoise video using FFmpeg hqdn3d filter."""
    strength_map = {
        "light": "2:1.5:3:2.5",
        "medium": "4:3:6:4.5",
        "strong": "6:4.5:9:6.5",
    }
    params = strength_map.get(strength, strength_map["medium"])

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", input_path,
        "-vf", f"hqdn3d={params}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path,
    ]
    run_ffmpeg(cmd)


def _auto_color(input_path: str, output_path: str, avg_brightness: float = 128.0) -> None:
    """Auto color correct using FFmpeg eq filter."""
    # Adjust brightness and contrast based on analysis
    brightness_adj = 0.0
    contrast_adj = 1.0
    saturation_adj = 1.1

    if avg_brightness < 80:
        brightness_adj = min(0.15, (80 - avg_brightness) / 255.0)
        contrast_adj = 1.1
    elif avg_brightness > 200:
        brightness_adj = max(-0.15, (200 - avg_brightness) / 255.0)

    eq_filter = (
        f"eq=brightness={brightness_adj:.3f}"
        f":contrast={contrast_adj:.2f}"
        f":saturation={saturation_adj:.2f}"
    )

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", input_path,
        "-vf", eq_filter,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path,
    ]
    run_ffmpeg(cmd)


def _upscale_video(input_path: str, output_path: str, target_height: int = 1080) -> None:
    """Upscale video to target height using FFmpeg lanczos."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", input_path,
        "-vf", f"scale=-2:{target_height}:flags=lanczos",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path,
    ]
    run_ffmpeg(cmd)


def _denoise_audio(input_path: str, output_path: str) -> None:
    """Denoise audio track using FFmpeg afftdn filter."""
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", input_path,
        "-af", "afftdn=nf=-25:nr=12:nt=w",
        "-c:v", "copy",
        output_path,
    ]
    run_ffmpeg(cmd)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def one_click_enhance(
    video_path: str,
    output_path: Optional[str] = None,
    preset: str = "balanced",
    on_progress: Optional[Callable] = None,
) -> EnhanceResult:
    """
    One-Click Enhance Pipeline -- single button for full video enhancement.

    Pipeline: analyze -> audio denoise -> stabilize -> video denoise ->
    auto color -> upscale. Each step runs only if analysis detects the issue.

    Args:
        video_path: Path to input video file.
        output_path: Explicit output path. Auto-generated if None.
        preset: "fast", "balanced", or "quality".
        on_progress: Progress callback(pct, msg).

    Returns:
        EnhanceResult with output path, steps applied, and duration.
    """
    valid_presets = ("fast", "balanced", "quality")
    if preset not in valid_presets:
        raise ValueError(f"Invalid preset '{preset}'. Must be one of: {valid_presets}")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input file not found: {video_path}")

    start_time = time.time()
    cfg = _PRESETS[preset]
    steps_applied = []

    if on_progress:
        on_progress(5, "Analyzing clip...")

    # Step 1: Analyze
    issues = _analyze_clip(video_path, on_progress=on_progress)
    if on_progress:
        on_progress(15, f"Analysis complete: {sum(1 for v in issues.values() if v is True)} issues detected")

    # Build output path
    if output_path is None:
        output_path = _output_path(video_path, f"enhanced_{preset}", "")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    tmp_files = []
    current_path = video_path
    step_pct = 20  # Start progress at 20%
    step_increment = 15  # Each step gets ~15% of progress bar

    try:
        # Step 2: Audio denoise (if noisy)
        if cfg["audio_denoise"] and issues["is_noisy_audio"]:
            if on_progress:
                on_progress(step_pct, "Denoising audio...")
            fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="opencut_oce_audio_")
            os.close(fd)
            tmp_files.append(tmp)
            _denoise_audio(current_path, tmp)
            current_path = tmp
            steps_applied.append("audio_denoise")
            step_pct += step_increment

        # Step 3: Stabilize (if shaky)
        if cfg["stabilize"] and issues["is_shaky"]:
            if on_progress:
                on_progress(step_pct, "Stabilizing video...")
            fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="opencut_oce_stab_")
            os.close(fd)
            tmp_files.append(tmp)

            def _stab_progress(pct, msg=""):
                scaled = step_pct + int(pct / 100.0 * step_increment)
                if on_progress:
                    on_progress(scaled, msg)

            _stabilize_video(current_path, tmp, on_progress=_stab_progress)
            current_path = tmp
            steps_applied.append("stabilize")
            step_pct += step_increment

        # Step 4: Video denoise (if grainy)
        if issues["is_grainy"]:
            if on_progress:
                on_progress(step_pct, "Denoising video...")
            fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="opencut_oce_denoise_")
            os.close(fd)
            tmp_files.append(tmp)
            _denoise_video(current_path, tmp, strength=cfg["denoise_strength"])
            current_path = tmp
            steps_applied.append("video_denoise")
            step_pct += step_increment

        # Step 5: Auto color (if needed)
        if cfg["color_correct"] and issues["needs_color_fix"]:
            if on_progress:
                on_progress(step_pct, "Auto color correction...")
            fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="opencut_oce_color_")
            os.close(fd)
            tmp_files.append(tmp)
            _auto_color(current_path, tmp, avg_brightness=issues["avg_brightness"])
            current_path = tmp
            steps_applied.append("color_correct")
            step_pct += step_increment

        # Step 6: Upscale (if < 1080p and not skipped)
        if not cfg["skip_upscale"] and issues["needs_upscale"]:
            if on_progress:
                on_progress(step_pct, "Upscaling to 1080p...")
            fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="opencut_oce_upscale_")
            os.close(fd)
            tmp_files.append(tmp)
            _upscale_video(current_path, tmp, target_height=1080)
            current_path = tmp
            steps_applied.append("upscale")

        # Final output
        if on_progress:
            on_progress(90, "Finalizing output...")

        if current_path == video_path:
            # No steps applied -- just copy with faststart
            cmd = [
                get_ffmpeg_path(), "-hide_banner", "-y",
                "-i", video_path,
                "-c", "copy",
                "-movflags", "+faststart",
                output_path,
            ]
            run_ffmpeg(cmd)
            steps_applied.append("passthrough")
        elif current_path != output_path:
            # Re-encode final output with correct settings
            cmd = [
                get_ffmpeg_path(), "-hide_banner", "-y",
                "-i", current_path,
                "-c:v", "libx264", "-crf", str(cfg["crf"]),
                "-preset", cfg["preset"],
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                output_path,
            ]
            run_ffmpeg(cmd)

        duration = round(time.time() - start_time, 2)

        if on_progress:
            on_progress(100, f"Enhanced ({len(steps_applied)} steps, {duration:.1f}s)")

        return EnhanceResult(
            output_path=output_path,
            steps_applied=steps_applied,
            duration_seconds=duration,
        )

    finally:
        for tmp in tmp_files:
            try:
                if os.path.isfile(tmp):
                    os.unlink(tmp)
            except OSError:
                pass
