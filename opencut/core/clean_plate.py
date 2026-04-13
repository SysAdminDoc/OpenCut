"""
OpenCut Clean Plate Generation Module v0.9.0

Generate a static clean background frame from video:
- Temporal median composite across sampled frames
- Gap detection and inpainting for remaining artifacts
- Useful for background subtraction, VFX paint-outs, plate work

Requires: pip install opencv-python-headless numpy
"""

import logging
import os
from typing import Callable, Optional

from opencut.helpers import ensure_package, get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Median Composite
# ---------------------------------------------------------------------------
def median_composite(frames: list) -> object:
    """
    Compute pixel-wise median across a list of frames.

    The median naturally removes transient objects (people, cars) that
    appear in fewer than half the sampled frames, revealing the static
    background.

    Args:
        frames: List of BGR numpy arrays, all same shape.

    Returns:
        Median composite as BGR uint8 numpy array.
    """
    if not ensure_package("numpy", "numpy"):
        raise RuntimeError("Failed to install numpy")
    import numpy as np

    if not frames:
        raise ValueError("No frames provided for median composite")

    # Validate all frames have same shape
    shape = frames[0].shape
    for i, f in enumerate(frames):
        if f.shape != shape:
            raise ValueError(
                f"Frame {i} shape {f.shape} != expected {shape}"
            )

    # Stack and compute median
    stack = np.stack(frames, axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)

    return median


# ---------------------------------------------------------------------------
# Inpaint Gaps
# ---------------------------------------------------------------------------
def inpaint_gaps(
    image,
    mask,
    method: str = "telea",
    radius: int = 5,
) -> object:
    """
    Inpaint masked regions of an image.

    Fills gaps left by the median composite where transient objects
    were present in too many frames.

    Args:
        image: BGR input image (H, W, 3) uint8.
        mask: Binary mask (H, W) uint8 where 255 = region to inpaint.
        method: "telea" (fast marching) or "ns" (Navier-Stokes).
        radius: Inpainting neighbourhood radius in pixels.

    Returns:
        Inpainted image as BGR uint8 numpy array.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("Failed to install opencv-python-headless")
    import cv2

    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    if mask is None or mask.size == 0:
        raise ValueError("Mask is empty")

    # Ensure mask is single channel uint8
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != "uint8":
        import numpy as np
        mask = mask.astype(np.uint8)

    method_flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    result = cv2.inpaint(image, mask, radius, method_flag)

    return result


# ---------------------------------------------------------------------------
# Full Clean Plate Generation
# ---------------------------------------------------------------------------
def generate_clean_plate(
    video_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    num_samples: int = 30,
    sample_interval: float = 0.0,
    inpaint: bool = True,
    inpaint_method: str = "telea",
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Generate a clean background plate from a video.

    Samples frames across the video duration, computes a pixel-wise
    median to remove transient objects, and optionally inpaints any
    remaining artifacts.

    Args:
        video_path: Path to input video.
        output_path: Path for output image. Auto-generated if None.
        output_dir: Output directory.
        num_samples: Number of frames to sample (more = cleaner but slower).
        sample_interval: Seconds between samples. 0 = auto-distribute.
        inpaint: Whether to run inpainting on detected gaps.
        inpaint_method: "telea" or "ns".
        on_progress: Progress callback(pct, msg).

    Returns:
        Dict with output_path, width, height, num_samples.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("Failed to install opencv-python-headless")
    if not ensure_package("numpy", "numpy"):
        raise RuntimeError("Failed to install numpy")
    import cv2
    import numpy as np

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_clean_plate.png")

    if on_progress:
        on_progress(5, "Sampling frames for clean plate...")

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    info.get("duration", 0.0)
    w = info.get("width", 0)
    h = info.get("height", 0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_samples = max(3, min(num_samples, total_frames))

    # Calculate sample positions
    if sample_interval > 0 and fps > 0:
        frame_indices = []
        frame_step = int(sample_interval * fps)
        idx = 0
        while idx < total_frames and len(frame_indices) < num_samples:
            frame_indices.append(idx)
            idx += frame_step
    else:
        # Evenly distribute across video
        frame_indices = [
            int(i * (total_frames - 1) / max(1, num_samples - 1))
            for i in range(num_samples)
        ]

    # Sample frames
    sampled_frames = []
    for i, fidx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)

        if on_progress and (i + 1) % 5 == 0:
            pct = 5 + int((i / len(frame_indices)) * 50)
            on_progress(pct, f"Sampling frame {i + 1}/{len(frame_indices)}...")

    cap.release()

    if len(sampled_frames) < 3:
        raise RuntimeError(
            f"Only {len(sampled_frames)} frames sampled -- need at least 3"
        )

    if on_progress:
        on_progress(60, f"Computing median composite from {len(sampled_frames)} frames...")

    # Compute median
    clean_plate = median_composite(sampled_frames)

    # Optional inpainting for remaining artifacts
    if inpaint:
        if on_progress:
            on_progress(80, "Detecting and inpainting gaps...")

        # Detect potential artifact regions: areas with high variance
        # across samples might still have remnants
        stack = np.stack(sampled_frames, axis=0).astype(np.float32)
        variance = np.var(stack, axis=0).mean(axis=2)

        # High-variance regions likely had moving objects
        threshold = np.percentile(variance, 95)
        artifact_mask = (variance > threshold).astype(np.uint8) * 255

        # Dilate mask to cover edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        artifact_mask = cv2.dilate(artifact_mask, kernel, iterations=2)

        if np.count_nonzero(artifact_mask) > 0:
            clean_plate = inpaint_gaps(
                clean_plate, artifact_mask,
                method=inpaint_method,
            )

    if on_progress:
        on_progress(95, "Saving clean plate...")

    cv2.imwrite(output_path, clean_plate)

    if on_progress:
        on_progress(100, "Clean plate generated!")

    return {
        "output_path": output_path,
        "width": w,
        "height": h,
        "num_samples": len(sampled_frames),
    }
