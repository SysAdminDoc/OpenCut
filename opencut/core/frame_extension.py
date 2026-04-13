"""
OpenCut AI Frame Extension / Outpainting Module v0.1.0

Extend video frames beyond their original boundaries:
- Spatial extension: change aspect ratio (e.g., 4:3 to 16:9) by
  generating content for the extended borders
- Temporal extension: add extra frames at start/end using
  motion extrapolation
- Edge-aware inpainting with reflection/mirror fallback

Falls back to blur-fill + reflection when AI models unavailable.
Uses frame-by-frame processing with FFmpeg reassembly.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from opencut.helpers import ensure_package, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Configuration / Result
# ---------------------------------------------------------------------------
@dataclass
class FrameExtensionConfig:
    """Configuration for frame extension."""
    fill_method: str = "reflect"    # "reflect", "blur", "replicate", "inpaint"
    blur_strength: int = 51         # Gaussian blur kernel size for "blur" fill
    inpaint_radius: int = 5         # Inpaint radius for "inpaint" fill
    temporal_method: str = "hold"   # "hold" (freeze), "reverse", "flow"
    flow_iterations: int = 3        # Optical flow iterations for temporal ext.


@dataclass
class SpatialExtensionResult:
    """Result of spatial frame extension."""
    output_path: str = ""
    original_aspect: str = ""
    target_aspect: str = ""
    original_size: Tuple[int, int] = (0, 0)
    output_size: Tuple[int, int] = (0, 0)
    frames_processed: int = 0
    fill_method: str = ""


@dataclass
class TemporalExtensionResult:
    """Result of temporal frame extension."""
    output_path: str = ""
    extra_seconds: float = 0.0
    frames_added: int = 0
    method: str = ""
    total_duration: float = 0.0


# ---------------------------------------------------------------------------
# Edge Region Detection
# ---------------------------------------------------------------------------
def detect_edge_regions(frame) -> Dict:
    """
    Analyze edge regions of a frame for extension suitability.

    Returns dict with:
        - edge_complexity: dict with 'top', 'bottom', 'left', 'right' scores (0-1)
        - dominant_colors: dict with 'top', 'bottom', 'left', 'right' RGB tuples
        - recommended_fill: str ("reflect", "blur", "inpaint")
    """
    import cv2

    h, w = frame.shape[:2]
    border = max(10, min(h, w) // 20)  # 5% border strip

    regions = {
        "top": frame[:border, :, :],
        "bottom": frame[-border:, :, :],
        "left": frame[:, :border, :],
        "right": frame[:, -border:, :],
    }

    complexity = {}
    colors = {}

    for name, region in regions.items():
        # Edge complexity: variance of Laplacian
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var = float(lap.var())
        # Normalize to 0-1 range (empirical: 0-5000 maps to 0-1)
        complexity[name] = min(1.0, var / 5000.0)

        # Dominant color
        avg_color = region.mean(axis=(0, 1)).astype(int).tolist()
        colors[name] = tuple(avg_color)

    # Recommend fill method based on complexity
    avg_complexity = sum(complexity.values()) / 4
    if avg_complexity < 0.1:
        recommended = "reflect"  # Simple edges, reflection works well
    elif avg_complexity < 0.3:
        recommended = "blur"  # Moderate complexity, blur fill
    else:
        recommended = "inpaint"  # Complex edges, need inpainting

    return {
        "edge_complexity": complexity,
        "dominant_colors": colors,
        "recommended_fill": recommended,
    }


def _parse_aspect_ratio(aspect: str) -> Tuple[int, int]:
    """Parse aspect ratio string like '16:9' to (16, 9)."""
    parts = aspect.replace("/", ":").split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid aspect ratio: {aspect}")
    return (int(parts[0]), int(parts[1]))


def _compute_target_size(
    orig_w: int, orig_h: int, target_w_ratio: int, target_h_ratio: int,
) -> Tuple[int, int]:
    """Compute target pixel dimensions preserving original content area."""
    orig_ratio = orig_w / orig_h
    target_ratio = target_w_ratio / target_h_ratio

    if target_ratio > orig_ratio:
        # Wider: extend horizontally
        new_w = int(orig_h * target_ratio)
        new_w = new_w + (new_w % 2)  # Ensure even
        new_h = orig_h
    else:
        # Taller: extend vertically
        new_w = orig_w
        new_h = int(orig_w / target_ratio)
        new_h = new_h + (new_h % 2)

    return (new_w, new_h)


def _extend_frame(frame, target_w: int, target_h: int,
                  config: FrameExtensionConfig):
    """Extend a single frame to target dimensions."""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]

    if target_w == w and target_h == h:
        return frame

    # Create output canvas
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Center the original frame
    x_offset = (target_w - w) // 2
    y_offset = (target_h - h) // 2
    result[y_offset:y_offset + h, x_offset:x_offset + w] = frame

    method = config.fill_method

    if method == "reflect":
        # Mirror/reflect edges into extended regions
        # Top
        if y_offset > 0:
            strip = frame[:y_offset, :, :]
            if strip.shape[0] > 0:
                flipped = cv2.flip(strip, 0)
                rh = min(y_offset, flipped.shape[0])
                result[y_offset - rh:y_offset, x_offset:x_offset + w] = flipped[:rh]

        # Bottom
        bottom_offset = y_offset + h
        if bottom_offset < target_h:
            gap = target_h - bottom_offset
            strip = frame[-gap:, :, :]
            if strip.shape[0] > 0:
                flipped = cv2.flip(strip, 0)
                rh = min(gap, flipped.shape[0])
                result[bottom_offset:bottom_offset + rh, x_offset:x_offset + w] = flipped[:rh]

        # Left
        if x_offset > 0:
            strip = frame[:, :x_offset, :]
            if strip.shape[1] > 0:
                flipped = cv2.flip(strip, 1)
                rw = min(x_offset, flipped.shape[1])
                result[y_offset:y_offset + h, x_offset - rw:x_offset] = flipped[:, :rw]

        # Right
        right_offset = x_offset + w
        if right_offset < target_w:
            gap = target_w - right_offset
            strip = frame[:, -gap:, :]
            if strip.shape[1] > 0:
                flipped = cv2.flip(strip, 1)
                rw = min(gap, flipped.shape[1])
                result[y_offset:y_offset + h, right_offset:right_offset + rw] = flipped[:, :rw]

        # Fill corners by blending
        if x_offset > 0 and y_offset > 0:
            # Apply Gaussian blur to smooth edges
            mask = np.zeros((target_h, target_w), dtype=np.float32)
            mask[y_offset:y_offset + h, x_offset:x_offset + w] = 1.0
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            mask_3d = mask[:, :, np.newaxis]

            blurred = cv2.GaussianBlur(result, (31, 31), 0)
            result = (result * mask_3d + blurred * (1 - mask_3d)).astype(np.uint8)
            # Restore original center
            result[y_offset:y_offset + h, x_offset:x_offset + w] = frame

    elif method == "blur":
        # Fill with blurred version of original
        scaled = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        ksize = config.blur_strength
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(scaled, (ksize, ksize), 0)
        result = blurred.copy()
        result[y_offset:y_offset + h, x_offset:x_offset + w] = frame

    elif method == "replicate":
        # Replicate border pixels
        result = cv2.copyMakeBorder(
            frame,
            y_offset, target_h - h - y_offset,
            x_offset, target_w - w - x_offset,
            cv2.BORDER_REPLICATE,
        )

    elif method == "inpaint":
        # Use OpenCV inpainting for extended regions
        mask = np.ones((target_h, target_w), dtype=np.uint8) * 255
        mask[y_offset:y_offset + h, x_offset:x_offset + w] = 0

        # First fill with blur, then inpaint for better results
        scaled = cv2.resize(frame, (target_w, target_h))
        ksize = max(15, config.blur_strength // 2)
        if ksize % 2 == 0:
            ksize += 1
        base = cv2.GaussianBlur(scaled, (ksize, ksize), 0)
        base[y_offset:y_offset + h, x_offset:x_offset + w] = frame

        result = cv2.inpaint(base, mask, config.inpaint_radius, cv2.INPAINT_TELEA)

    return result


# ---------------------------------------------------------------------------
# Spatial Extension
# ---------------------------------------------------------------------------
def extend_frame_spatial(
    video_path: str,
    target_aspect: str = "16:9",
    output_path: Optional[str] = None,
    output_dir: str = "",
    config: Optional[FrameExtensionConfig] = None,
    on_progress: Optional[Callable] = None,
) -> SpatialExtensionResult:
    """
    Extend video frames to a new aspect ratio by generating border content.

    For example, convert 4:3 footage to 16:9 by extending the sides
    with reflected/blurred/inpainted content.

    Args:
        video_path: Path to input video.
        target_aspect: Target aspect ratio (e.g., "16:9", "21:9").
        output_path: Optional explicit output path.
        output_dir: Output directory (defaults to input dir).
        config: FrameExtensionConfig with fill parameters.
        on_progress: Callback(pct, msg) for progress updates.

    Returns:
        SpatialExtensionResult with output path and dimensions.
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required")

    import cv2

    if config is None:
        config = FrameExtensionConfig()

    target_w_ratio, target_h_ratio = _parse_aspect_ratio(target_aspect)
    result = SpatialExtensionResult(fill_method=config.fill_method)

    info = get_video_info(video_path)
    orig_w = info.get("width", 1920)
    orig_h = info.get("height", 1080)
    result.original_size = (orig_w, orig_h)
    result.original_aspect = f"{orig_w}:{orig_h}"
    result.target_aspect = target_aspect

    target_w, target_h = _compute_target_size(orig_w, orig_h, target_w_ratio, target_h_ratio)
    result.output_size = (target_w, target_h)

    if target_w == orig_w and target_h == orig_h:
        logger.info("Video already matches target aspect ratio")
        if output_path is None:
            output_path = video_path
        result.output_path = output_path
        return result

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        aspect_tag = target_aspect.replace(":", "x")
        output_path = os.path.join(directory, f"{base}_{aspect_tag}.mp4")

    if on_progress:
        on_progress(5, f"Extending {orig_w}x{orig_h} to {target_w}x{target_h}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    _ntf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video = _ntf.name
    _ntf.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (target_w, target_h))
    if not writer.isOpened():
        cap.release()
        try:
            os.unlink(tmp_video)
        except OSError:
            pass
        raise RuntimeError(f"Cannot create video writer for: {tmp_video}")

    if on_progress:
        on_progress(10, "Processing frames...")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            extended = _extend_frame(frame, target_w, target_h, config)
            writer.write(extended)
            frame_idx += 1

            if on_progress and frame_idx % 10 == 0:
                pct = 10 + int((frame_idx / total) * 80)
                on_progress(pct, f"Extending frame {frame_idx}/{total}...")
    finally:
        cap.release()
        writer.release()

    result.frames_processed = frame_idx

    if on_progress:
        on_progress(92, "Encoding with audio...")

    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_video, "-i", video_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", output_path,
        ], timeout=7200)
    finally:
        try:
            os.unlink(tmp_video)
        except OSError:
            pass

    result.output_path = output_path

    if on_progress:
        on_progress(100, "Frame extension complete!")

    return result


# ---------------------------------------------------------------------------
# Temporal Extension
# ---------------------------------------------------------------------------
def extend_frame_temporal(
    video_path: str,
    extra_seconds: float = 2.0,
    position: str = "end",
    output_path: Optional[str] = None,
    output_dir: str = "",
    config: Optional[FrameExtensionConfig] = None,
    on_progress: Optional[Callable] = None,
) -> TemporalExtensionResult:
    """
    Extend video duration by generating extra frames at start or end.

    Methods:
    - "hold": Freeze the first/last frame
    - "reverse": Play frames in reverse (boomerang effect)
    - "flow": Extrapolate motion using optical flow

    Args:
        video_path: Path to input video.
        extra_seconds: Seconds to add.
        position: "start", "end", or "both".
        output_path: Optional explicit output path.
        output_dir: Output directory.
        config: FrameExtensionConfig with temporal parameters.
        on_progress: Callback(pct, msg) for progress.

    Returns:
        TemporalExtensionResult with output path and metadata.
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required")

    import cv2
    import numpy as np

    if config is None:
        config = FrameExtensionConfig()

    result = TemporalExtensionResult(extra_seconds=extra_seconds, method=config.temporal_method)

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_extended.mp4")

    info = get_video_info(video_path)
    fps = info.get("fps", 30)
    orig_duration = info.get("duration", 0)
    width = info.get("width", 1920)
    height = info.get("height", 1080)

    extra_frames = int(extra_seconds * fps)
    if position == "both":
        extra_frames_start = extra_frames // 2
        extra_frames_end = extra_frames - extra_frames_start
    elif position == "start":
        extra_frames_start = extra_frames
        extra_frames_end = 0
    else:
        extra_frames_start = 0
        extra_frames_end = extra_frames

    if on_progress:
        on_progress(5, f"Adding {extra_seconds}s ({extra_frames} frames)...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError("No frames read from video")

    if on_progress:
        on_progress(20, "Generating extension frames...")

    method = config.temporal_method

    # Generate extension frames
    start_frames = []
    end_frames = []

    if method == "hold":
        start_frames = [frames[0].copy()] * extra_frames_start
        end_frames = [frames[-1].copy()] * extra_frames_end
    elif method == "reverse":
        if extra_frames_start > 0:
            src = frames[:extra_frames_start]
            start_frames = list(reversed(src))
            while len(start_frames) < extra_frames_start:
                start_frames = start_frames + list(reversed(start_frames))
            start_frames = start_frames[:extra_frames_start]
        if extra_frames_end > 0:
            src = frames[-extra_frames_end:]
            end_frames = list(reversed(src))
            while len(end_frames) < extra_frames_end:
                end_frames = end_frames + list(reversed(end_frames))
            end_frames = end_frames[:extra_frames_end]
    elif method == "flow":
        # Optical flow extrapolation
        if extra_frames_end > 0 and len(frames) >= 2:
            gray_a = cv2.cvtColor(frames[-2], cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray_a, gray_b, None, 0.5, 3, 15, 3, 7, 1.5, 0)

            gy, gx = np.mgrid[0:height, 0:width].astype(np.float32)
            last_frame = frames[-1].copy()

            for i in range(extra_frames_end):
                scale = (i + 1) * 0.5
                map_x = gx + flow[:, :, 0] * scale
                map_y = gy + flow[:, :, 1] * scale
                extrapolated = cv2.remap(last_frame, map_x, map_y,
                                         cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT)
                # Blend with frozen frame to reduce drift
                alpha = min(1.0, (i + 1) / (extra_frames_end * 0.7))
                blended = cv2.addWeighted(extrapolated, 1 - alpha * 0.5,
                                          last_frame, alpha * 0.5, 0)
                end_frames.append(blended)

        if extra_frames_start > 0 and len(frames) >= 2:
            gray_a = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray_a, gray_b, None, 0.5, 3, 15, 3, 7, 1.5, 0)

            gy, gx = np.mgrid[0:height, 0:width].astype(np.float32)
            first_frame = frames[0].copy()

            for i in range(extra_frames_start):
                scale = (extra_frames_start - i) * 0.5
                map_x = gx + flow[:, :, 0] * scale
                map_y = gy + flow[:, :, 1] * scale
                extrapolated = cv2.remap(first_frame, map_x, map_y,
                                         cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT)
                alpha = min(1.0, (extra_frames_start - i) / (extra_frames_start * 0.7))
                blended = cv2.addWeighted(extrapolated, 1 - alpha * 0.5,
                                          first_frame, alpha * 0.5, 0)
                start_frames.append(blended)
    else:
        # Default to hold
        start_frames = [frames[0].copy()] * extra_frames_start
        end_frames = [frames[-1].copy()] * extra_frames_end

    result.frames_added = len(start_frames) + len(end_frames)

    if on_progress:
        on_progress(60, "Writing extended video...")

    _ntf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video = _ntf.name
    _ntf.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        try:
            os.unlink(tmp_video)
        except OSError:
            pass
        raise RuntimeError(f"Cannot create video writer for: {tmp_video}")

    try:
        for f in start_frames:
            writer.write(f)
        for f in frames:
            writer.write(f)
        for f in end_frames:
            writer.write(f)
    finally:
        writer.release()

    if on_progress:
        on_progress(85, "Encoding with audio...")

    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_video, "-i", video_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", output_path,
        ], timeout=7200)
    finally:
        try:
            os.unlink(tmp_video)
        except OSError:
            pass

    result.output_path = output_path
    result.total_duration = orig_duration + extra_seconds

    if on_progress:
        on_progress(100, "Temporal extension complete!")

    return result
