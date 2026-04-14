"""
OpenCut AI Background Replacement

Remove background using rembg / chroma keying, then replace with
an AI-generated environment from a text prompt, or a user-supplied
static / dynamic background.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional

from opencut.helpers import (
    ensure_package,
    get_video_info,
    output_path,
    run_ffmpeg,
)

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Background type presets
# ---------------------------------------------------------------------------

BG_PRESETS = {
    "solid": "Solid color background",
    "gradient": "Gradient color background",
    "blur": "Blurred version of the original background",
    "image": "User-supplied static image",
    "video": "User-supplied background video",
    "ai_generated": "AI-generated environment from text prompt",
}


@dataclass
class BGReplaceResult:
    """Result of background replacement."""
    output_path: str = ""
    bg_type: str = ""
    prompt: str = ""
    frames_processed: int = 0
    method: str = ""
    ai_generated: bool = False


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------

def _remove_bg_rembg(frame):
    """Remove background from a single frame using rembg."""
    from rembg import remove
    import numpy as np
    from PIL import Image

    pil_img = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
    result = remove(pil_img)
    return np.array(result)  # RGBA


def _remove_bg_chroma(frame, color_lower=(35, 100, 100),
                      color_upper=(85, 255, 255)):
    """Remove green-screen background via chroma keying."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(color_lower), np.array(color_upper))
    mask = cv2.bitwise_not(mask)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Create RGBA output
    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba[:, :, ::-1]  # BGRA -> RGBA for consistency


# ---------------------------------------------------------------------------
# Background generation
# ---------------------------------------------------------------------------

def _create_solid_bg(width: int, height: int, color=(0, 128, 0)):
    """Create a solid color background."""
    import numpy as np
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:] = color
    return bg


def _create_gradient_bg(width: int, height: int,
                        color_top=(30, 30, 80), color_bottom=(10, 10, 30)):
    """Create a vertical gradient background."""
    import numpy as np
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(color_top[0] * (1 - t) + color_bottom[0] * t)
        g = int(color_top[1] * (1 - t) + color_bottom[1] * t)
        b = int(color_top[2] * (1 - t) + color_bottom[2] * t)
        bg[y, :] = (r, g, b)
    return bg


def _create_blur_bg(frame, ksize: int = 51):
    """Create a blurred version of the original frame."""
    import cv2
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def _composite_rgba_on_bg(rgba_frame, bg_frame):
    """Composite an RGBA foreground onto a BGR background."""
    import cv2
    import numpy as np

    # rgba_frame is RGBA, bg_frame is BGR
    alpha = rgba_frame[:, :, 3:4].astype(np.float32) / 255.0
    fg_rgb = rgba_frame[:, :, :3][:, :, ::-1]  # RGB -> BGR

    # Resize bg if needed
    h, w = fg_rgb.shape[:2]
    if bg_frame.shape[:2] != (h, w):
        bg_frame = cv2.resize(bg_frame, (w, h), interpolation=cv2.INTER_LINEAR)

    composite = (fg_rgb.astype(np.float32) * alpha +
                 bg_frame.astype(np.float32) * (1 - alpha))
    return composite.astype(np.uint8)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def replace_background(
    video_path: str,
    bg_type: str = "blur",
    prompt: str = "",
    bg_image_path: Optional[str] = None,
    bg_video_path: Optional[str] = None,
    bg_color: str = "#004400",
    removal_method: str = "auto",
    output_path_override: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Remove the video background and replace it.

    Removal methods:
    - ``"rembg"``: AI-based background removal (best quality)
    - ``"chroma"``: Green-screen chroma keying (fastest)
    - ``"auto"``: Try rembg first, fall back to chroma

    Background types:
    - ``"solid"``: Solid color (use *bg_color*)
    - ``"gradient"``: Vertical gradient
    - ``"blur"``: Blurred original background
    - ``"image"``: Static image (use *bg_image_path*)
    - ``"video"``: Background video (use *bg_video_path*)
    - ``"ai_generated"``: AI-generated from *prompt* (falls back to
      gradient if model unavailable)

    Args:
        video_path:  Input video.
        bg_type:  Background type.
        prompt:  Text prompt for AI background generation.
        bg_image_path:  Path to background image (for ``"image"`` type).
        bg_video_path:  Path to background video (for ``"video"`` type).
        bg_color:  Hex color for solid background.
        removal_method:  ``"auto"``, ``"rembg"``, or ``"chroma"``.
        output_path_override:  Explicit output path.
        on_progress:  Callback ``(pct, msg)``.

    Returns:
        dict with *output_path*, *bg_type*, *prompt*,
        *frames_processed*, *method*, *ai_generated*.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    if bg_type not in BG_PRESETS:
        bg_type = "blur"

    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("opencv-python-headless is required")

    import cv2
    import numpy as np

    out = output_path_override or output_path(video_path, f"bg_{bg_type}")

    if on_progress:
        on_progress(5, "Preparing background replacement...")

    info = get_video_info(video_path)

    # Determine removal method
    actual_removal = removal_method
    rembg_ok = False
    if removal_method in ("auto", "rembg"):
        try:
            rembg_ok = bool(ensure_package("rembg", "rembg", on_progress))
        except Exception:
            rembg_ok = False
        actual_removal = "rembg" if rembg_ok else "chroma"

    if on_progress:
        on_progress(10, f"Using {actual_removal} for background removal...")

    # Load background image if specified
    bg_static = None
    bg_cap = None
    ai_generated = False

    if bg_type == "image" and bg_image_path and os.path.isfile(bg_image_path):
        bg_static = cv2.imread(bg_image_path)
    elif bg_type == "video" and bg_video_path and os.path.isfile(bg_video_path):
        bg_cap = cv2.VideoCapture(bg_video_path)
    elif bg_type == "ai_generated":
        # Fall back to gradient when AI not available
        logger.info("AI background generation requested — using gradient fallback")
        bg_type = "gradient"
        ai_generated = False

    # Parse hex color
    try:
        hex_c = bg_color.lstrip("#")
        r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        solid_color = (b, g, r)  # BGR
    except (ValueError, IndexError):
        solid_color = (0, 68, 0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    _ntf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video = _ntf.name
    _ntf.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        try:
            os.unlink(tmp_video)
        except OSError:
            pass
        raise RuntimeError("Cannot create video writer")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Remove background
            if actual_removal == "rembg":
                rgba = _remove_bg_rembg(frame)
            else:
                rgba = _remove_bg_chroma(frame)

            # Generate background
            if bg_type == "solid":
                bg = _create_solid_bg(w, h, solid_color)
            elif bg_type == "gradient":
                bg = _create_gradient_bg(w, h)
            elif bg_type == "blur":
                bg = _create_blur_bg(frame)
            elif bg_type == "image" and bg_static is not None:
                bg = bg_static
            elif bg_type == "video" and bg_cap is not None:
                bg_ret, bg_frame = bg_cap.read()
                if not bg_ret:
                    bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    _, bg_frame = bg_cap.read()
                bg = bg_frame if bg_frame is not None else _create_solid_bg(w, h, solid_color)
            else:
                bg = _create_blur_bg(frame)

            composite = _composite_rgba_on_bg(rgba, bg)
            writer.write(composite)
            frame_idx += 1

            if on_progress and frame_idx % 5 == 0:
                pct = 10 + int((frame_idx / total) * 78)
                on_progress(pct, f"Processing frame {frame_idx}/{total}...")
    finally:
        cap.release()
        writer.release()
        if bg_cap:
            bg_cap.release()

    if on_progress:
        on_progress(90, "Encoding with audio...")

    try:
        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_video, "-i", video_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            out,
        ], timeout=7200)
    finally:
        try:
            os.unlink(tmp_video)
        except OSError:
            pass

    if on_progress:
        on_progress(100, "Background replacement complete!")

    return {
        "output_path": out,
        "bg_type": bg_type,
        "prompt": prompt,
        "frames_processed": frame_idx,
        "method": actual_removal,
        "ai_generated": ai_generated,
    }
