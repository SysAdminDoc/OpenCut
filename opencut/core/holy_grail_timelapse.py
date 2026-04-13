"""
OpenCut Holy Grail Timelapse Module v0.9.0

Day-to-night (or night-to-day) timelapse exposure ramping:
- Analyse exposure ramp across image sequence
- Apply per-frame exposure compensation
- White balance shift correction
- Combined deflicker for smooth transitions

Requires: pip install opencv-python-headless numpy
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from opencut.helpers import ensure_package, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class HolyGrailConfig:
    """Configuration for holy grail timelapse processing."""
    target_brightness: float = 0.45
    smoothing_window: int = 15
    wb_correction: bool = True
    deflicker: bool = True
    deflicker_strength: float = 0.8
    output_fps: float = 24.0
    output_codec: str = "libx264"
    output_crf: int = 18


# ---------------------------------------------------------------------------
# Exposure Ramp Analysis
# ---------------------------------------------------------------------------
def analyze_exposure_ramp(
    image_paths: List[str],
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Analyse the exposure ramp across an image sequence.

    Measures per-frame brightness, detects exposure steps (from
    camera bracketing or manual adjustments), and identifies
    the overall ramp direction.

    Args:
        image_paths: List of image file paths in chronological order.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with brightness_values, exposure_steps, ramp_direction,
        wb_shifts, recommended_adjustments.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("Failed to install opencv-python-headless")
    if not ensure_package("numpy", "numpy"):
        raise RuntimeError("Failed to install numpy")
    import cv2
    import numpy as np

    if not image_paths:
        raise ValueError("No image paths provided")

    # Validate all paths exist
    for p in image_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Image not found: {p}")

    if on_progress:
        on_progress(5, f"Analysing {len(image_paths)} frames...")

    brightness_values = []
    wb_shifts = []  # (R/G ratio, B/G ratio) for each frame

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            logger.warning("Could not read image: %s", path)
            brightness_values.append(0.0)
            wb_shifts.append((1.0, 1.0))
            continue

        # Measure brightness (mean of V channel in HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = float(np.mean(hsv[:, :, 2]) / 255.0)
        brightness_values.append(round(brightness, 4))

        # Measure white balance shift (R/G and B/G ratios)
        b_mean = float(np.mean(img[:, :, 0]))
        g_mean = float(np.mean(img[:, :, 1])) + 1e-8
        r_mean = float(np.mean(img[:, :, 2]))
        wb_shifts.append((
            round(r_mean / g_mean, 4),
            round(b_mean / g_mean, 4),
        ))

        if on_progress and (i + 1) % 10 == 0:
            pct = 5 + int((i / len(image_paths)) * 85)
            on_progress(pct, f"Analysing frame {i + 1}/{len(image_paths)}...")

    # Detect exposure steps (jumps in brightness)
    exposure_steps = []
    for i in range(1, len(brightness_values)):
        diff = brightness_values[i] - brightness_values[i - 1]
        if abs(diff) > 0.05:  # Significant brightness jump
            exposure_steps.append({
                "frame": i,
                "change": round(diff, 4),
                "direction": "brighter" if diff > 0 else "darker",
            })

    # Overall ramp direction
    if len(brightness_values) >= 2:
        first_quarter = np.mean(brightness_values[:max(1, len(brightness_values) // 4)])
        last_quarter = np.mean(brightness_values[-(max(1, len(brightness_values) // 4)):])
        if last_quarter > first_quarter + 0.05:
            ramp_direction = "brightening"
        elif last_quarter < first_quarter - 0.05:
            ramp_direction = "darkening"
        else:
            ramp_direction = "stable"
    else:
        ramp_direction = "unknown"

    # Recommended adjustments: target a smooth ramp to target_brightness
    target = 0.45
    recommended = []
    for b in brightness_values:
        if b > 0:
            adj = target / b
        else:
            adj = 1.0
        recommended.append(round(adj, 4))

    if on_progress:
        on_progress(100, "Exposure analysis complete!")

    return {
        "brightness_values": brightness_values,
        "exposure_steps": exposure_steps,
        "ramp_direction": ramp_direction,
        "wb_shifts": wb_shifts,
        "recommended_adjustments": recommended,
        "frame_count": len(image_paths),
    }


# ---------------------------------------------------------------------------
# Per-Frame Exposure Compensation
# ---------------------------------------------------------------------------
def apply_exposure_compensation(
    image_path: str,
    adjustment: float = 1.0,
    wb_adjustment: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply exposure and white balance compensation to a single image.

    Args:
        image_path: Input image path.
        adjustment: Exposure multiplier (1.0 = no change, 2.0 = double).
        wb_adjustment: Optional (R_scale, B_scale) white balance correction.
        output_path: Output path. Auto-generated if None.
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to compensated image.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("Failed to install opencv-python-headless")
    if not ensure_package("numpy", "numpy"):
        raise RuntimeError("Failed to install numpy")
    import cv2
    import numpy as np

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_compensated{ext}"

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    img_f = img.astype(np.float32)

    # Apply exposure adjustment
    img_f *= adjustment

    # Apply white balance correction
    if wb_adjustment:
        r_scale, b_scale = wb_adjustment
        img_f[:, :, 2] *= r_scale  # Red channel
        img_f[:, :, 0] *= b_scale  # Blue channel

    img_f = np.clip(img_f, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, img_f)

    if on_progress:
        on_progress(100, "Exposure compensation applied!")

    return output_path


# ---------------------------------------------------------------------------
# Full Holy Grail Processing
# ---------------------------------------------------------------------------
def process_holy_grail(
    image_paths: List[str],
    output_path: Optional[str] = None,
    output_dir: str = "",
    config: Optional[dict] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Process a holy grail timelapse image sequence.

    Full pipeline: analyse exposure ramp, apply per-frame compensation,
    optional WB correction and deflicker, then encode to video.

    Args:
        image_paths: List of image file paths in chronological order.
        output_path: Output video path. Auto-generated if None.
        output_dir: Output directory.
        config: Dict with HolyGrailConfig fields, or None for defaults.
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path, frame_count, ramp_direction,
        exposure_range.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("Failed to install opencv-python-headless")
    if not ensure_package("numpy", "numpy"):
        raise RuntimeError("Failed to install numpy")
    import cv2
    import numpy as np

    if not image_paths:
        raise ValueError("No image paths provided")
    if len(image_paths) < 3:
        raise ValueError("Need at least 3 images for timelapse processing")

    cfg = HolyGrailConfig()
    if config:
        for k, v in config.items():
            if hasattr(cfg, k):
                setattr(cfg, k, type(getattr(cfg, k))(v))

    if output_path is None:
        base = os.path.splitext(os.path.basename(image_paths[0]))[0]
        directory = output_dir or os.path.dirname(image_paths[0])
        output_path = os.path.join(directory, f"{base}_holygrail.mp4")

    if on_progress:
        on_progress(2, "Starting holy grail timelapse processing...")

    # Step 1: Analyse exposure ramp
    def _analysis_progress(pct, msg=""):
        if on_progress:
            on_progress(2 + int(pct * 0.2), msg)

    analysis = analyze_exposure_ramp(image_paths, on_progress=_analysis_progress)
    brightness = analysis["brightness_values"]
    wb_data = analysis["wb_shifts"]

    if on_progress:
        on_progress(25, "Computing smooth exposure curve...")

    # Step 2: Compute smooth target brightness curve
    adjustments = analysis["recommended_adjustments"]

    # Smooth the adjustment curve
    if cfg.smoothing_window > 1 and len(adjustments) > cfg.smoothing_window:
        kernel = np.ones(cfg.smoothing_window) / cfg.smoothing_window
        padded = np.pad(adjustments, (cfg.smoothing_window // 2, cfg.smoothing_window // 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")[:len(adjustments)]
        adjustments = smoothed.tolist()

    # Step 3: Process frames
    if on_progress:
        on_progress(30, "Processing frames...")

    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="oc_holygrail_")

    processed_paths = []
    try:
        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                logger.warning("Skipping unreadable image: %s", path)
                continue

            img_f = img.astype(np.float32)

            # Apply exposure compensation
            adj = adjustments[i] if i < len(adjustments) else 1.0
            img_f *= adj

            # Apply WB correction
            if cfg.wb_correction and i < len(wb_data):
                # Normalise WB to first frame
                r_ratio, b_ratio = wb_data[i]
                r_ref, b_ref = wb_data[0]
                if r_ratio > 0 and b_ratio > 0:
                    r_corr = r_ref / r_ratio
                    b_corr = b_ref / b_ratio
                    img_f[:, :, 2] *= r_corr  # Red
                    img_f[:, :, 0] *= b_corr  # Blue

            img_f = np.clip(img_f, 0, 255).astype(np.uint8)

            # Deflicker: blend with neighbours
            if cfg.deflicker and i > 0 and len(processed_paths) > 0:
                prev_img = cv2.imread(processed_paths[-1])
                if prev_img is not None:
                    alpha = cfg.deflicker_strength * 0.3
                    img_f = cv2.addWeighted(
                        img_f, 1.0 - alpha,
                        prev_img, alpha, 0,
                    )

            out_frame = os.path.join(tmp_dir, f"frame_{i:06d}.png")
            cv2.imwrite(out_frame, img_f)
            processed_paths.append(out_frame)

            if on_progress and (i + 1) % 10 == 0:
                pct = 30 + int((i / len(image_paths)) * 55)
                on_progress(pct, f"Processing frame {i + 1}/{len(image_paths)}...")

        if not processed_paths:
            raise RuntimeError("No frames could be processed")

        if on_progress:
            on_progress(88, "Encoding timelapse video...")

        # Step 4: Encode to video
        frame_pattern = os.path.join(tmp_dir, "frame_%06d.png")
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(cfg.output_fps),
            "-i", frame_pattern,
            "-c:v", cfg.output_codec, "-crf", str(cfg.output_crf),
            "-preset", "medium", "-pix_fmt", "yuv420p",
            output_path,
        ], timeout=7200)

    finally:
        # Clean up temp frames
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    if on_progress:
        on_progress(100, "Holy grail timelapse complete!")

    return {
        "output_path": output_path,
        "frame_count": len(processed_paths),
        "ramp_direction": analysis["ramp_direction"],
        "exposure_range": (
            round(min(brightness) if brightness else 0, 4),
            round(max(brightness) if brightness else 0, 4),
        ),
    }
