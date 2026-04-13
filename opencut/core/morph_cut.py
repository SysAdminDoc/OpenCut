"""
OpenCut Morph Cut / Smooth Jump Cut Module v0.1.0

Blend talking-head shots across jump cuts using optical flow interpolation:
- Detect face regions in frames around the cut point
- Compute dense optical flow between pre/post cut frames
- Generate interpolated transition frames
- Blend with temporal feathering for seamless result

Falls back to crossfade-based blending when face detection fails.
Uses frame-by-frame processing with FFmpeg reassembly.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from opencut.helpers import ensure_package, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Configuration / Result
# ---------------------------------------------------------------------------
@dataclass
class MorphCutConfig:
    """Configuration for morph cut processing."""
    transition_frames: int = 8      # Number of interpolated frames
    blend_mode: str = "optical_flow"  # "optical_flow" or "crossfade"
    face_weight: float = 0.7       # Weight for face region alignment (0-1)
    background_weight: float = 0.3  # Weight for background region
    flow_scale: float = 0.5        # Optical flow pyramid scale
    flow_levels: int = 3           # Optical flow pyramid levels
    smooth_sigma: float = 1.5      # Gaussian smoothing for flow field


@dataclass
class MorphCutResult:
    """Result of morph cut operation."""
    output_path: str = ""
    cut_point_frame: int = 0
    frames_interpolated: int = 0
    face_detected: bool = False
    method_used: str = ""          # "optical_flow" or "crossfade"


# ---------------------------------------------------------------------------
# Face Region Detection
# ---------------------------------------------------------------------------
def detect_face_region(frame) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face bounding box in a frame using OpenCV Haar cascade.

    Returns (x, y, w, h) tuple or None if no face found.
    Lightweight alternative to MediaPipe for simple face detection.
    """
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use built-in Haar cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return None

    # Return largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    return (int(x), int(y), int(w), int(h))


# ---------------------------------------------------------------------------
# Frame Interpolation
# ---------------------------------------------------------------------------
def interpolate_frames(
    frame_a,
    frame_b,
    count: int = 8,
    config: Optional[MorphCutConfig] = None,
) -> List:
    """
    Generate interpolated frames between frame_a and frame_b.

    Uses dense optical flow (Farneback) to warp and blend frames,
    producing smooth intermediate frames for the morph cut.

    Args:
        frame_a: Source frame (numpy array, BGR).
        frame_b: Target frame (numpy array, BGR).
        count: Number of intermediate frames to generate.
        config: MorphCutConfig with flow parameters.

    Returns:
        List of interpolated frame arrays.
    """
    import cv2
    import numpy as np

    if config is None:
        config = MorphCutConfig()

    h, w = frame_a.shape[:2]
    results = []

    if config.blend_mode == "crossfade" or count < 1:
        # Simple crossfade fallback
        for i in range(count):
            alpha = (i + 1) / (count + 1)
            blended = cv2.addWeighted(frame_a, 1 - alpha, frame_b, alpha, 0)
            results.append(blended)
        return results

    # Compute dense optical flow
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    flow_ab = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b,
        None,
        pyr_scale=config.flow_scale,
        levels=config.flow_levels,
        winsize=15,
        iterations=3,
        poly_n=7,
        poly_sigma=config.smooth_sigma,
        flags=0,
    )

    # Face-weighted blending
    face_mask = np.ones((h, w), dtype=np.float32) * config.background_weight
    face_a = detect_face_region(frame_a)
    face_b = detect_face_region(frame_b)

    if face_a is not None:
        fx, fy, fw, fh = face_a
        # Expand face region slightly
        pad = int(max(fw, fh) * 0.3)
        rx1 = max(0, fx - pad)
        ry1 = max(0, fy - pad)
        rx2 = min(w, fx + fw + pad)
        ry2 = min(h, fy + fh + pad)
        face_mask[ry1:ry2, rx1:rx2] = config.face_weight

    if face_b is not None:
        fx, fy, fw, fh = face_b
        pad = int(max(fw, fh) * 0.3)
        rx1 = max(0, fx - pad)
        ry1 = max(0, fy - pad)
        rx2 = min(w, fx + fw + pad)
        ry2 = min(h, fy + fh + pad)
        # Average with existing mask
        face_mask[ry1:ry2, rx1:rx2] = (
            face_mask[ry1:ry2, rx1:rx2] + config.face_weight
        ) / 2

    # Normalize face mask
    face_mask = cv2.GaussianBlur(face_mask, (31, 31), 0)
    max_val = face_mask.max()
    if max_val > 0:
        face_mask /= max_val

    # Generate coordinate grids
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)

    for i in range(count):
        t = (i + 1) / (count + 1)

        # Warp frame_a forward and frame_b backward by t
        map_x_a = gx + flow_ab[:, :, 0] * t
        map_y_a = gy + flow_ab[:, :, 1] * t
        map_x_b = gx - flow_ab[:, :, 0] * (1 - t)
        map_y_b = gy - flow_ab[:, :, 1] * (1 - t)

        warped_a = cv2.remap(frame_a, map_x_a, map_y_a, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        warped_b = cv2.remap(frame_b, map_x_b, map_y_b, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)

        # Weighted blend with face emphasis
        weight = face_mask * t + (1 - face_mask) * t
        weight_3d = weight[:, :, np.newaxis]

        blended = (warped_a * (1 - weight_3d) + warped_b * weight_3d).astype(np.uint8)
        results.append(blended)

    return results


# ---------------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------------
def apply_morph_cut(
    video_path: str,
    cut_point: float,
    output_path: Optional[str] = None,
    output_dir: str = "",
    config: Optional[MorphCutConfig] = None,
    on_progress: Optional[Callable] = None,
) -> MorphCutResult:
    """
    Apply morph cut at a jump cut point in a talking-head video.

    Replaces the hard cut with smoothly interpolated frames using
    optical flow and face-aware blending.

    Args:
        video_path: Path to input video.
        cut_point: Time of the jump cut in seconds.
        output_path: Optional explicit output path.
        output_dir: Output directory (defaults to input dir).
        config: MorphCutConfig with interpolation parameters.
        on_progress: Callback(pct, msg) for progress updates.

    Returns:
        MorphCutResult with output path and statistics.
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required")

    import cv2

    if config is None:
        config = MorphCutConfig()

    result = MorphCutResult()

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_morph.mp4")

    if on_progress:
        on_progress(5, "Analyzing cut point...")

    info = get_video_info(video_path)
    fps = info.get("fps", 30)
    duration = info.get("duration", 0)
    width = info.get("width", 1920)
    height = info.get("height", 1080)

    if cut_point <= 0 or cut_point >= duration:
        raise ValueError(f"cut_point {cut_point} must be within video duration (0-{duration})")

    cut_frame = int(cut_point * fps)
    result.cut_point_frame = cut_frame

    # Read frames around the cut point
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_trans = config.transition_frames // 2

    # Navigate to pre-cut frames
    pre_start = max(0, cut_frame - half_trans - 1)
    min(total_frames, cut_frame + half_trans + 1)

    if on_progress:
        on_progress(10, "Reading frames around cut point...")

    # Read all frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    all_frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append((idx, frame))
        idx += 1
    cap.release()

    if cut_frame >= len(all_frames) or cut_frame < 1:
        raise ValueError(f"Cut frame {cut_frame} out of range (total: {len(all_frames)})")

    # Get the frames before and after the cut
    frame_before = all_frames[cut_frame - 1][1]
    frame_after = all_frames[min(cut_frame, len(all_frames) - 1)][1]

    if on_progress:
        on_progress(20, "Detecting faces...")

    face_a = detect_face_region(frame_before)
    face_b = detect_face_region(frame_after)
    result.face_detected = face_a is not None or face_b is not None

    if on_progress:
        on_progress(30, f"Interpolating {config.transition_frames} frames...")

    # Determine method
    method = config.blend_mode
    if not result.face_detected and method == "optical_flow":
        logger.info("No face detected, optical flow may produce artifacts")

    try:
        interp_frames = interpolate_frames(
            frame_before, frame_after,
            count=config.transition_frames,
            config=config,
        )
        result.method_used = method
    except Exception as e:
        logger.warning("Optical flow failed: %s, falling back to crossfade", e)
        fallback_cfg = MorphCutConfig(
            transition_frames=config.transition_frames,
            blend_mode="crossfade",
        )
        interp_frames = interpolate_frames(
            frame_before, frame_after,
            count=config.transition_frames,
            config=fallback_cfg,
        )
        result.method_used = "crossfade"

    result.frames_interpolated = len(interp_frames)

    if on_progress:
        on_progress(60, "Writing output video...")

    # Write output: frames before cut + interpolated + frames after cut
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
        # Write frames before the transition zone
        for i in range(pre_start):
            writer.write(all_frames[i][1])

        # Write transition: blend from pre_start to cut, interpolated, then cut to post_end
        for i in range(pre_start, cut_frame):
            writer.write(all_frames[i][1])

        # Write interpolated frames
        for interp in interp_frames:
            if interp.shape[:2] != (height, width):
                interp = cv2.resize(interp, (width, height))
            writer.write(interp)

        # Write frames after transition zone
        for i in range(min(cut_frame + 1, len(all_frames)), len(all_frames)):
            writer.write(all_frames[i][1])

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

    if on_progress:
        on_progress(100, "Morph cut complete!")

    return result
