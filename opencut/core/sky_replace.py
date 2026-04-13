"""
OpenCut AI Sky Replacement Module

Detect sky regions via brightness/color segmentation, replace sky pixels
with a source image or video, and adjust foreground lighting to match.

Uses OpenCV for frame-level processing and FFmpeg for final encoding.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_video_info,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class SkyMaskResult:
    """Result from sky region detection."""
    mask_path: str = ""
    sky_fraction: float = 0.0
    horizon_y: int = 0
    confidence: float = 0.0
    width: int = 0
    height: int = 0


@dataclass
class SkyReplaceResult:
    """Result from full sky replacement pipeline."""
    output_path: str = ""
    frames_processed: int = 0
    avg_sky_fraction: float = 0.0
    foreground_adjusted: bool = False
    method: str = "brightness"


# ---------------------------------------------------------------------------
# Sky Detection
# ---------------------------------------------------------------------------
def detect_sky_mask(
    frame_path: str,
    method: str = "brightness",
    threshold: float = 0.55,
    blue_weight: float = 0.3,
    on_progress: Optional[Callable] = None,
) -> SkyMaskResult:
    """
    Detect sky region in a single frame using brightness/color segmentation.

    The detector combines luminance thresholding with blue-channel weighting
    and spatial priors (sky tends to be in the upper portion of the frame).
    A connected-component analysis selects the largest bright region
    touching the top edge as the sky mask.

    Args:
        frame_path: Path to an image file (JPEG/PNG).
        method: Detection method -- "brightness" (default) or "color".
        threshold: Brightness threshold 0-1 (higher = stricter).
        blue_weight: Extra weight for blue-channel contribution (0-1).
        on_progress: Optional progress callback(pct, msg).

    Returns:
        SkyMaskResult with mask path and sky statistics.
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required for sky detection")
    import cv2
    import numpy as np

    if not os.path.isfile(frame_path):
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    if on_progress:
        on_progress(10, "Loading frame...")

    img = cv2.imread(frame_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {frame_path}")

    h, w = img.shape[:2]

    if on_progress:
        on_progress(20, "Computing sky probability map...")

    # Convert to float 0-1
    img_f = img.astype(np.float32) / 255.0
    b_ch, g_ch, r_ch = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    if method == "color":
        # Color-based: favour blue-ish, bright pixels
        # Sky tends to be high-blue, high-brightness, low-saturation
        brightness = (r_ch + g_ch + b_ch) / 3.0
        blue_dominance = b_ch - np.maximum(r_ch, g_ch)
        sky_prob = brightness * (1.0 - blue_weight) + np.clip(blue_dominance + 0.5, 0, 1) * blue_weight
    else:
        # Brightness-based with blue weighting
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        sky_prob = gray * (1.0 - blue_weight) + b_ch * blue_weight

    if on_progress:
        on_progress(40, "Applying spatial prior...")

    # Spatial prior: gradient from top (1.0) to bottom (0.0)
    # This strongly favours the upper part of the image
    y_prior = np.linspace(1.0, 0.0, h, dtype=np.float32).reshape(-1, 1)
    y_prior = np.broadcast_to(y_prior, (h, w))
    sky_prob = sky_prob * 0.6 + y_prior * 0.4

    # Threshold
    thresh_val = int(threshold * 255)
    sky_u8 = (sky_prob * 255).astype(np.uint8)
    _, binary = cv2.threshold(sky_u8, thresh_val, 255, cv2.THRESH_BINARY)

    if on_progress:
        on_progress(60, "Finding connected sky region...")

    # Connected-component analysis: keep only components touching the top row
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    sky_mask = np.zeros((h, w), dtype=np.uint8)
    for label_id in range(1, num_labels):
        # Check if this component touches the top 5% of the image
        comp_mask = (labels == label_id)
        top_overlap = comp_mask[:max(1, h // 20), :].any()
        if top_overlap:
            sky_mask[comp_mask] = 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
    sky_mask = cv2.GaussianBlur(sky_mask, (21, 21), 0)

    if on_progress:
        on_progress(80, "Computing sky statistics...")

    sky_fraction = float(np.count_nonzero(sky_mask > 127)) / (h * w)

    # Find approximate horizon line (lowest row with >50% sky)
    horizon_y = 0
    for row_idx in range(h):
        row_sky = np.count_nonzero(sky_mask[row_idx, :] > 127)
        if row_sky > w * 0.5:
            horizon_y = row_idx

    # Confidence based on how clean the mask boundary is
    edge = cv2.Canny(sky_mask, 50, 150)
    edge_ratio = float(np.count_nonzero(edge)) / max(1, w)
    confidence = max(0.0, min(1.0, 1.0 - edge_ratio / 10.0))

    # Save mask
    mask_dir = tempfile.gettempdir()
    mask_path = os.path.join(mask_dir, f"sky_mask_{os.getpid()}.png")
    cv2.imwrite(mask_path, sky_mask)

    if on_progress:
        on_progress(100, f"Sky detected: {sky_fraction:.1%} of frame")

    return SkyMaskResult(
        mask_path=mask_path,
        sky_fraction=round(sky_fraction, 4),
        horizon_y=horizon_y,
        confidence=round(confidence, 3),
        width=w,
        height=h,
    )


# ---------------------------------------------------------------------------
# Foreground Lighting Adjustment
# ---------------------------------------------------------------------------
def adjust_foreground_lighting(
    frame,  # numpy array BGR
    sky_brightness: float,
    target_brightness: float = 0.5,
    strength: float = 0.6,
) -> object:
    """
    Adjust foreground color temperature/brightness to match a new sky.

    Shifts the non-sky portion of the frame so that its tonal balance is
    consistent with the replacement sky's overall brightness.

    Args:
        frame: BGR numpy array (uint8).
        sky_brightness: Mean brightness of the replacement sky (0-1).
        target_brightness: Neutral brightness reference (0-1).
        strength: Adjustment strength 0-1.

    Returns:
        Adjusted BGR frame (uint8).
    """
    import numpy as np

    if strength <= 0:
        return frame

    # Compute brightness adjustment factor
    if target_brightness > 0:
        ratio = sky_brightness / target_brightness
    else:
        ratio = 1.0

    # Mild warm/cool shift based on sky brightness
    # Bright sky -> slightly warm foreground, dark sky -> slightly cool
    warm_shift = (ratio - 1.0) * strength * 15  # small pixel-value offset

    frame.astype(np.float32)

    # Apply brightness scaling to luminance channel (Lab space)
    import cv2
    lab = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)

    # Scale L channel towards sky brightness
    brightness_factor = 1.0 + (ratio - 1.0) * strength * 0.3
    brightness_factor = max(0.5, min(1.5, brightness_factor))
    lab[:, :, 0] = np.clip(lab[:, :, 0] * brightness_factor, 0, 255)

    # Warm/cool shift on a/b channels
    lab[:, :, 1] = np.clip(lab[:, :, 1] + warm_shift * 0.3, 0, 255)  # a: green-red
    lab[:, :, 2] = np.clip(lab[:, :, 2] + warm_shift * 0.5, 0, 255)  # b: blue-yellow

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
    return result


# ---------------------------------------------------------------------------
# Full Sky Replacement Pipeline
# ---------------------------------------------------------------------------
def replace_sky(
    video_path: str,
    sky_source: str,
    output_path_str: Optional[str] = None,
    output_dir: str = "",
    method: str = "brightness",
    threshold: float = 0.55,
    blue_weight: float = 0.3,
    feather: int = 15,
    adjust_lighting: bool = True,
    lighting_strength: float = 0.6,
    on_progress: Optional[Callable] = None,
) -> SkyReplaceResult:
    """
    Replace sky in a video with a sky source image or video.

    Pipeline:
      1. For each frame, detect sky mask via brightness/color segmentation.
      2. Read corresponding frame from sky source (looping if needed).
      3. Optionally adjust foreground lighting to match sky brightness.
      4. Composite: sky in masked region, original foreground elsewhere.
      5. Encode final video with audio from original.

    Args:
        video_path: Input video file.
        sky_source: Replacement sky (image or video file).
        output_path_str: Output path. Auto-generated if None.
        output_dir: Directory for output (used if output_path_str is None).
        method: Sky detection method ("brightness" or "color").
        threshold: Sky detection threshold 0-1.
        blue_weight: Blue-channel weight for detection.
        feather: Edge feathering radius in pixels.
        adjust_lighting: Whether to adjust foreground lighting.
        lighting_strength: Foreground lighting adjustment strength.
        on_progress: Progress callback(pct, msg).

    Returns:
        SkyReplaceResult with output path and processing statistics.
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required for sky replacement")
    import cv2
    import numpy as np

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(sky_source):
        raise FileNotFoundError(f"Sky source not found: {sky_source}")

    if on_progress:
        on_progress(2, "Initializing sky replacement...")

    # Output path
    if output_path_str is None:
        directory = output_dir or os.path.dirname(video_path)
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path_str = os.path.join(directory, f"{base}_sky_replaced.mp4")

    info = get_video_info(video_path)
    w, h, fps = info["width"], info["height"], info["fps"]

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Open sky source
    sky_is_video = sky_source.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))
    sky_cap = cv2.VideoCapture(sky_source) if sky_is_video else None
    sky_img = None if sky_is_video else cv2.imread(sky_source)
    if not sky_is_video and sky_img is None:
        cap.release()
        raise FileNotFoundError(f"Could not read sky image: {sky_source}")
    if sky_img is not None:
        sky_img = cv2.resize(sky_img, (w, h))

    # Temp output for raw frames
    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        if sky_cap:
            sky_cap.release()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise RuntimeError("Failed to create video writer")

    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_idx = 0
    sky_fractions: List[float] = []

    # Precompute spatial prior
    y_prior = np.linspace(1.0, 0.0, h, dtype=np.float32).reshape(-1, 1)
    y_prior = np.broadcast_to(y_prior, (h, w)).copy()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Sky mask for this frame ---
            img_f = frame.astype(np.float32) / 255.0
            b_ch = img_f[:, :, 0]
            g_ch = img_f[:, :, 1]
            r_ch = img_f[:, :, 2]

            if method == "color":
                brightness = (r_ch + g_ch + b_ch) / 3.0
                blue_dom = b_ch - np.maximum(r_ch, g_ch)
                sky_prob = brightness * (1.0 - blue_weight) + np.clip(blue_dom + 0.5, 0, 1) * blue_weight
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                sky_prob = gray * (1.0 - blue_weight) + b_ch * blue_weight

            sky_prob = sky_prob * 0.6 + y_prior * 0.4
            sky_u8 = (sky_prob * 255).astype(np.uint8)
            thresh_val = int(threshold * 255)
            _, binary = cv2.threshold(sky_u8, thresh_val, 255, cv2.THRESH_BINARY)

            # Connected components: keep regions touching top
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            sky_mask = np.zeros((h, w), dtype=np.uint8)
            for lid in range(1, num_labels):
                comp_mask = (labels == lid)
                if comp_mask[:max(1, h // 20), :].any():
                    sky_mask[comp_mask] = 255

            # Feathering
            if feather > 0:
                ksize = feather * 2 + 1
                sky_mask = cv2.GaussianBlur(sky_mask, (ksize, ksize), 0)

            sky_frac = float(np.count_nonzero(sky_mask > 127)) / (h * w)
            sky_fractions.append(sky_frac)

            # --- Get sky frame ---
            if sky_cap is not None:
                ret_sky, sky_frame = sky_cap.read()
                if not ret_sky:
                    sky_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_sky, sky_frame = sky_cap.read()
                if sky_frame is None:
                    sky_frame = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    sky_frame = cv2.resize(sky_frame, (w, h))
            else:
                sky_frame = sky_img.copy()

            # --- Foreground lighting adjustment ---
            fg_frame = frame.copy()
            if adjust_lighting and sky_frac > 0.05:
                sky_brightness = float(np.mean(sky_frame)) / 255.0
                fg_frame = adjust_foreground_lighting(
                    fg_frame, sky_brightness, strength=lighting_strength,
                )

            # --- Composite ---
            alpha = sky_mask.astype(np.float32) / 255.0
            alpha_3 = np.stack([alpha, alpha, alpha], axis=-1)
            result = (sky_frame.astype(np.float32) * alpha_3 +
                      fg_frame.astype(np.float32) * (1.0 - alpha_3))
            writer.write(result.astype(np.uint8))

            frame_idx += 1
            if on_progress and frame_idx % 15 == 0:
                pct = 5 + int((frame_idx / total_frames) * 85)
                on_progress(min(90, pct), f"Replacing sky: frame {frame_idx}/{total_frames}")

    finally:
        cap.release()
        if sky_cap:
            sky_cap.release()
        writer.release()

    if on_progress:
        on_progress(92, "Encoding final video with audio...")

    # Mux with audio from original
    try:
        cmd = (FFmpegCmd()
               .input(tmp_path)
               .input(video_path)
               .map("0:v", "1:a?")
               .video_codec("libx264", crf=18, preset="medium")
               .audio_codec("aac", bitrate="192k")
               .option("shortest")
               .faststart()
               .output(output_path_str)
               .build())
        run_ffmpeg(cmd, timeout=7200)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    avg_sky = sum(sky_fractions) / max(1, len(sky_fractions))

    if on_progress:
        on_progress(100, "Sky replacement complete!")

    return SkyReplaceResult(
        output_path=output_path_str,
        frames_processed=frame_idx,
        avg_sky_fraction=round(avg_sky, 4),
        foreground_adjusted=adjust_lighting,
        method=method,
    )
