"""
OpenCut Green-Screen-Free Background Replacement Module

AI matting without green screen, then composite a new background.
Descript and CapCut both offer this feature natively.

Methods (in priority order):
    1. SAM2 (Segment Anything Model 2) - highest quality segmentation
    2. rembg (U2-Net) - fast, good quality background removal
    3. MediaPipe selfie segmentation - lightweight, real-time capable

Background types: image, video, solid color, blur, transparent/alpha.

Functions:
    replace_background    - Full video background replacement pipeline
    replace_background_frame - Single-frame preview of background replacement
"""

import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Union

from opencut.helpers import (
    FFmpegCmd,
    ensure_package,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_METHODS = ("auto", "rembg", "mediapipe", "sam2")
VALID_BG_KEYWORDS = ("blur", "none")

# Temporal smoothing window for matte stabilization
_TEMPORAL_WINDOW = 5
# Default Gaussian blur radius for "blur" background mode
_DEFAULT_BLUR_RADIUS = 21


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BGReplaceResult:
    """Result of a background replacement operation."""
    output_path: str = ""
    method_used: str = ""
    frames_processed: int = 0
    avg_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers: method detection
# ---------------------------------------------------------------------------
def _detect_available_method() -> str:
    """Probe for the best available segmentation method.

    Returns one of: "sam2", "rembg", "mediapipe".
    """
    try:
        import importlib
        importlib.import_module("sam2")
        return "sam2"
    except ImportError:
        pass

    try:
        import importlib
        importlib.import_module("rembg")
        return "rembg"
    except ImportError:
        pass

    try:
        import importlib
        importlib.import_module("mediapipe")
        return "mediapipe"
    except ImportError:
        pass

    # Default to rembg (will be installed via ensure_package)
    return "rembg"


def _parse_hex_color(color_str: str) -> tuple:
    """Parse a hex color string to (B, G, R) tuple for OpenCV.

    Accepts: "#RRGGBB", "RRGGBB", "#RGB", "RGB".
    """
    c = color_str.lstrip("#")
    if len(c) == 3:
        c = c[0] * 2 + c[1] * 2 + c[2] * 2
    if len(c) != 6:
        raise ValueError(f"Invalid hex color: {color_str}")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (b, g, r)  # BGR for OpenCV


def _is_hex_color(value: str) -> bool:
    """Check if a string looks like a hex color."""
    if not isinstance(value, str):
        return False
    c = value.lstrip("#")
    return len(c) in (3, 6) and all(ch in "0123456789abcdefABCDEF" for ch in c)


# ---------------------------------------------------------------------------
# Internal helpers: segmentation backends
# ---------------------------------------------------------------------------
def _segment_rembg(frame, session=None):
    """Generate alpha matte using rembg (U2-Net).

    Args:
        frame: BGR numpy array (OpenCV format).
        session: Optional pre-loaded rembg session for batch efficiency.

    Returns:
        (alpha_matte, confidence) - matte is HxW uint8, confidence is 0.0-1.0.
    """
    import numpy as np
    from rembg import new_session, remove

    if session is None:
        session = new_session("u2net")

    # rembg expects PIL or raw bytes; convert BGR -> RGB -> PIL
    from PIL import Image
    rgb = frame[:, :, ::-1]
    pil_img = Image.fromarray(rgb)

    # Get RGBA output
    result = remove(pil_img, session=session, only_mask=True)
    mask_arr = np.array(result)

    # Ensure single channel
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]

    confidence = float(np.mean(mask_arr > 127)) if mask_arr.size > 0 else 0.0
    return mask_arr, confidence


def _segment_mediapipe(frame, segmenter=None):
    """Generate alpha matte using MediaPipe selfie segmentation.

    Args:
        frame: BGR numpy array (OpenCV format).
        segmenter: Optional pre-loaded MediaPipe segmenter.

    Returns:
        (alpha_matte, confidence) - matte is HxW uint8, confidence is 0.0-1.0.
    """
    import numpy as np

    if segmenter is None:
        import mediapipe as mp
        segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1,
        )

    rgb = frame[:, :, ::-1]
    results = segmenter.process(rgb)
    mask = results.segmentation_mask  # float32 [0, 1]
    alpha = (mask * 255).astype(np.uint8)
    confidence = float(np.mean(mask)) if mask is not None else 0.0
    return alpha, confidence


def _segment_sam2(frame, predictor=None):
    """Generate alpha matte using SAM2 (Segment Anything Model 2).

    Uses automatic mask generation targeting the largest foreground segment.

    Args:
        frame: BGR numpy array (OpenCV format).
        predictor: Optional pre-loaded SAM2 predictor.

    Returns:
        (alpha_matte, confidence) - matte is HxW uint8, confidence is 0.0-1.0.
    """
    import numpy as np

    if predictor is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam_model = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt")
        predictor = SAM2ImagePredictor(sam_model)

    rgb = frame[:, :, ::-1]
    predictor.set_image(rgb)

    # Use center point as foreground prompt (works well for subject-centric video)
    h, w = frame.shape[:2]
    center_point = np.array([[w // 2, h // 2]])
    center_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=center_point,
        point_labels=center_label,
        multimask_output=True,
    )

    # Pick highest-confidence mask
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx]
    alpha = (mask.astype(np.uint8) * 255)
    confidence = float(scores[best_idx])

    return alpha, confidence


def _get_segmenter(method: str):
    """Initialize and return a reusable segmenter session for batch processing."""
    if method == "rembg":
        from rembg import new_session
        return new_session("u2net")
    elif method == "mediapipe":
        import mediapipe as mp
        return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    elif method == "sam2":
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam_model = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt")
        return SAM2ImagePredictor(sam_model)
    return None


def _segment_frame(frame, method: str, session=None):
    """Route to the appropriate segmentation backend."""
    if method == "rembg":
        return _segment_rembg(frame, session)
    elif method == "mediapipe":
        return _segment_mediapipe(frame, session)
    elif method == "sam2":
        return _segment_sam2(frame, session)
    raise ValueError(f"Unknown segmentation method: {method}")


# ---------------------------------------------------------------------------
# Internal helpers: compositing
# ---------------------------------------------------------------------------
def _refine_matte(alpha, edge_blur: int = 2):
    """Refine alpha matte with morphological operations and edge feathering.

    Pipeline: slight erode -> Gaussian blur on edges -> smooth result.
    """
    import cv2

    # Erode slightly to pull edges inward (reduces halo)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined = cv2.erode(alpha, kernel, iterations=1)

    # Gaussian blur for feathered edges
    if edge_blur > 0:
        blur_size = edge_blur * 2 + 1  # Must be odd
        refined = cv2.GaussianBlur(refined, (blur_size, blur_size), 0)

    return refined


def _temporal_smooth_matte(matte_buffer: list, current_matte):
    """Apply temporal smoothing to prevent matte flickering.

    Averages the current matte with recent frames from the buffer.
    """
    import numpy as np

    matte_buffer.append(current_matte.astype(np.float32))
    if len(matte_buffer) > _TEMPORAL_WINDOW:
        matte_buffer.pop(0)

    if len(matte_buffer) == 1:
        return current_matte

    stacked = np.stack(matte_buffer, axis=0)
    smoothed = np.mean(stacked, axis=0).astype(np.uint8)
    return smoothed


def _composite_frame(fg_frame, alpha, bg_frame):
    """Composite foreground over background using alpha matte.

    Args:
        fg_frame: BGR foreground frame.
        alpha: Single-channel alpha matte (0-255).
        bg_frame: BGR background frame (same size as fg_frame).

    Returns:
        BGR composited frame.
    """
    import cv2
    import numpy as np

    # Normalize alpha to [0, 1]
    a = alpha.astype(np.float32) / 255.0
    if a.ndim == 2:
        a = a[:, :, np.newaxis]

    fg = fg_frame.astype(np.float32)
    bg = bg_frame.astype(np.float32)

    # Resize bg if needed
    h, w = fg.shape[:2]
    if bg.shape[:2] != (h, w):
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    composite = fg * a + bg * (1.0 - a)
    return composite.astype(np.uint8)


def _prepare_bg_frame(
    bg_source,
    frame_idx: int,
    frame_shape: tuple,
    original_frame=None,
    bg_cap=None,
):
    """Prepare a background frame based on background type.

    Args:
        bg_source: Background specification (path, color, "blur", "none").
        frame_idx: Current frame index (for video backgrounds).
        frame_shape: (height, width, channels) of the target frame.
        original_frame: Original video frame (needed for "blur" mode).
        bg_cap: OpenCV VideoCapture for video backgrounds.

    Returns:
        BGR numpy array of the background frame.
    """
    import cv2
    import numpy as np

    h, w = frame_shape[:2]

    if bg_source == "blur":
        if original_frame is not None:
            return cv2.GaussianBlur(original_frame, (_DEFAULT_BLUR_RADIUS, _DEFAULT_BLUR_RADIUS), 0)
        return np.zeros((h, w, 3), dtype=np.uint8)

    if bg_source == "none":
        # Transparent -- return black (alpha will be used as actual transparency)
        return np.zeros((h, w, 3), dtype=np.uint8)

    if isinstance(bg_source, tuple) and len(bg_source) == 3:
        # Solid color (B, G, R)
        bg = np.full((h, w, 3), bg_source, dtype=np.uint8)
        return bg

    if bg_cap is not None:
        # Video background -- read next frame, loop if needed
        ret, bg_frame = bg_cap.read()
        if not ret:
            bg_cap.set(1, 0)  # cv2.CAP_PROP_POS_FRAMES = 1
            ret, bg_frame = bg_cap.read()
        if ret and bg_frame is not None:
            return cv2.resize(bg_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.zeros((h, w, 3), dtype=np.uint8)

    # Image background (static)
    if isinstance(bg_source, str) and os.path.isfile(bg_source):
        bg = cv2.imread(bg_source)
        if bg is not None:
            return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)

    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def replace_background_frame(
    frame_data,
    background: Union[str, None] = "blur",
    method: str = "auto",
    edge_blur: int = 2,
) -> dict:
    """Replace background on a single frame (for preview).

    Args:
        frame_data: BGR numpy array of the input frame.
        background: Background spec -- image path, hex color, "blur", or "none".
        method: Segmentation method: "auto", "rembg", "mediapipe", "sam2".
        edge_blur: Feather radius for edge refinement (pixels).

    Returns:
        Dict with "frame" (BGR numpy array), "alpha" (mask), "confidence" (float).
    """
    import numpy as np

    method = method.lower().strip()
    if method == "auto":
        method = _detect_available_method()

    # Ensure the segmentation dependency is available
    _ensure_method_deps(method)

    alpha, confidence = _segment_frame(frame_data, method)
    alpha = _refine_matte(alpha, edge_blur)

    # Prepare background
    if _is_hex_color(str(background)):
        bg_color = _parse_hex_color(str(background))
        bg_frame = _prepare_bg_frame(bg_color, 0, frame_data.shape)
    elif background in VALID_BG_KEYWORDS:
        bg_frame = _prepare_bg_frame(
            background, 0, frame_data.shape, original_frame=frame_data,
        )
    elif isinstance(background, str) and os.path.isfile(background):
        bg_frame = _prepare_bg_frame(background, 0, frame_data.shape)
    else:
        bg_frame = _prepare_bg_frame("blur", 0, frame_data.shape, original_frame=frame_data)

    # Handle "none" (transparent output)
    if background == "none":
        # Return BGRA with alpha channel
        result_frame = np.dstack([frame_data, alpha])
    else:
        result_frame = _composite_frame(frame_data, alpha, bg_frame)

    return {
        "frame": result_frame,
        "alpha": alpha,
        "confidence": confidence,
    }


def _ensure_method_deps(method: str):
    """Ensure the dependencies for the given segmentation method are installed."""
    if method == "rembg":
        if not ensure_package("rembg", "rembg"):
            raise RuntimeError("rembg is required but could not be installed. Run: pip install rembg")
        if not ensure_package("PIL", "Pillow"):
            raise RuntimeError("Pillow is required. Run: pip install Pillow")
    elif method == "mediapipe":
        if not ensure_package("mediapipe", "mediapipe"):
            raise RuntimeError("mediapipe is required. Run: pip install mediapipe")
    elif method == "sam2":
        if not ensure_package("sam2", "segment-anything-2"):
            raise RuntimeError("SAM2 is required. Run: pip install segment-anything-2")

    # All methods need OpenCV and numpy
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("OpenCV is required. Run: pip install opencv-python-headless")
    if not ensure_package("numpy", "numpy"):
        raise RuntimeError("numpy is required. Run: pip install numpy")


def replace_background(
    video_path: str,
    background: Union[str, None] = "blur",
    output: Optional[str] = None,
    method: str = "auto",
    edge_blur: int = 2,
    on_progress: Optional[Callable] = None,
) -> BGReplaceResult:
    """Replace video background without a green screen.

    Pipeline:
    1. Extract frames from input video
    2. Generate alpha matte per frame using selected AI model
    3. Composite foreground over new background
    4. Apply edge refinement (feathering/blur)
    5. Temporal smoothing of matte to prevent flickering
    6. Encode output video

    Args:
        video_path: Path to the input video file.
        background: One of:
            - Image path (jpg/png) for static background
            - Video path (mp4/mov) for video background (synced frame-for-frame)
            - Hex color string ("#FF0000") for solid color
            - "blur" to blur the original background
            - "none" for transparent/alpha output (RGBA)
        output: Output video path. Auto-generated if None.
        method: Segmentation method: "auto", "rembg", "mediapipe", "sam2".
        edge_blur: Feather radius for edge refinement in pixels.
        on_progress: Optional progress callback(pct, msg).

    Returns:
        BGReplaceResult with output path and metadata.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If method is invalid.
        RuntimeError: If required dependencies cannot be installed.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    method = method.lower().strip()
    if method not in VALID_METHODS:
        raise ValueError(f"Invalid method '{method}'. Must be one of: {VALID_METHODS}")

    if method == "auto":
        method = _detect_available_method()

    if on_progress:
        on_progress(5, f"Initializing background replacement ({method})...")

    _ensure_method_deps(method)

    import cv2
    import numpy as np

    info = get_video_info(video_path)
    fps = info.get("fps", 30.0)
    width = info.get("width", 1920)
    height = info.get("height", 1080)

    if output is None:
        suffix = "bgreplace"
        if background == "none":
            # Use .mov for alpha channel support
            output = output_path(video_path, suffix).replace(".mp4", ".mov")
        else:
            output = output_path(video_path, suffix)

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = int(info.get("duration", 0) * fps)

    # Determine background type and open video bg if needed
    bg_cap = None
    bg_is_video = False
    bg_source = background

    if isinstance(background, str) and not _is_hex_color(background) and background not in VALID_BG_KEYWORDS:
        if os.path.isfile(background):
            # Detect if it's a video or image
            ext = os.path.splitext(background)[1].lower()
            if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                bg_cap = cv2.VideoCapture(background)
                bg_is_video = True
                bg_source = background  # Will use bg_cap
            else:
                bg_source = background  # Image path

    if _is_hex_color(str(background)):
        bg_source = _parse_hex_color(str(background))

    # Set up output writer
    tmp_dir = tempfile.mkdtemp(prefix="opencut_bgreplace_")
    fd, tmp_output = tempfile.mkstemp(suffix=".mp4", prefix="bgreplace_", dir=tmp_dir)
    os.close(fd)

    if background == "none":
        # RGBA output -- write PNG frames, encode later with alpha
        fourcc = cv2.VideoWriter_fourcc(*"png ")
        frame_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        writer = None
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_output, fourcc, fps, (width, height))

    # Initialize segmenter for batch efficiency
    try:
        session = _get_segmenter(method)
    except Exception as e:
        logger.warning("Could not pre-load segmenter: %s", e)
        session = None

    matte_buffer = []
    confidences = []
    frame_idx = 0

    if on_progress:
        on_progress(10, f"Processing {total_frames} frames...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Segment
            alpha, conf = _segment_frame(frame, method, session)
            confidences.append(conf)

            # Refine matte
            alpha = _refine_matte(alpha, edge_blur)

            # Temporal smoothing
            alpha = _temporal_smooth_matte(matte_buffer, alpha)

            # Prepare background frame
            bg_frame = _prepare_bg_frame(
                bg_source if not bg_is_video else "video",
                frame_idx, frame.shape,
                original_frame=frame,
                bg_cap=bg_cap if bg_is_video else None,
            )

            # Handle video background specifically
            if bg_is_video and bg_cap is not None:
                bg_frame = _prepare_bg_frame(
                    bg_source, frame_idx, frame.shape, bg_cap=bg_cap,
                )

            if background == "none":
                # Save BGRA frame as PNG
                rgba = np.dstack([frame, alpha])
                path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(path, rgba)
            else:
                composite = _composite_frame(frame, alpha, bg_frame)
                writer.write(composite)

            frame_idx += 1

            if on_progress and frame_idx % max(1, total_frames // 20) == 0:
                pct = 10 + int(75 * frame_idx / max(total_frames, 1))
                on_progress(min(pct, 85), f"Frame {frame_idx}/{total_frames}")

    finally:
        cap.release()
        if bg_cap is not None:
            bg_cap.release()
        if writer is not None:
            writer.release()

    if on_progress:
        on_progress(88, "Encoding final output...")

    # Encode with FFmpeg for proper codec/container
    if background == "none":
        # Encode RGBA frames to ProRes 4444 (supports alpha)
        frame_pattern = os.path.join(frame_dir, "frame_%06d.png")
        encode_cmd = (
            FFmpegCmd()
            .input(frame_pattern, framerate=str(fps))
            .input(video_path)
            .option("map", "0:v")
            .option("map", "1:a?")
            .video_codec("prores_ks")
            .option("profile:v", "4")
            .option("pix_fmt", "yuva444p10le")
            .audio_codec("aac", bitrate="192k")
            .output(output)
            .build()
        )
    else:
        # Re-encode with proper codec and copy audio from original
        encode_cmd = (
            FFmpegCmd()
            .input(tmp_output)
            .input(video_path)
            .option("map", "0:v")
            .option("map", "1:a?")
            .video_codec("libx264", crf=18, preset="fast")
            .audio_codec("aac", bitrate="192k")
            .output(output)
            .build()
        )

    run_ffmpeg(encode_cmd)

    # Cleanup
    import shutil as _shutil
    try:
        _shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    avg_confidence = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )

    if on_progress:
        on_progress(100, "Background replacement complete")

    return BGReplaceResult(
        output_path=output,
        method_used=method,
        frames_processed=frame_idx,
        avg_confidence=round(avg_confidence, 4),
    )
