"""
OpenCut Style Transfer Module v0.7.1

Neural style transfer applied to video:
- Pre-trained fast style transfer models (Candy, Mosaic, Pointilism, etc.)
- OpenCV DNN backend (no PyTorch required, runs on any system)
- Adjustable style intensity via blending
- Frame-by-frame processing with FFmpeg reassembly

Uses the fast neural style transfer approach (Johnson et al.) with
pre-trained .t7 models from the OpenCV DNN samples.
"""

import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

from opencut.helpers import ensure_package, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")

MODELS_DIR = os.path.join(os.path.expanduser("~"), ".opencut", "models", "style_transfer")
os.makedirs(MODELS_DIR, exist_ok=True)

# Pre-trained fast style transfer models
# These are Torch .t7 models compatible with OpenCV's DNN module
STYLE_MODELS = {
    "candy": {
        "label": "Candy",
        "description": "Bright, colorful candy-like style",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/candy.t7",
        "filename": "candy.t7",
    },
    "mosaic": {
        "label": "Mosaic",
        "description": "Stained glass mosaic pattern",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/mosaic.t7",
        "filename": "mosaic.t7",
    },
    "rain_princess": {
        "label": "Rain Princess",
        "description": "Painterly impressionist style",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/rain_princess.t7",
        "filename": "rain_princess.t7",
    },
    "udnie": {
        "label": "Udnie",
        "description": "Abstract cubist painting style",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/udnie.t7",
        "filename": "udnie.t7",
    },
    "pointilism": {
        "label": "Pointilism",
        "description": "Dot-based pointilist painting",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/feathers.t7",
        "filename": "pointilism.t7",
    },
    "la_muse": {
        "label": "La Muse",
        "description": "Bold abstract expressionist colors",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/la_muse.t7",
        "filename": "la_muse.t7",
    },
    "the_scream": {
        "label": "The Scream",
        "description": "Edvard Munch's The Scream style",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/the_scream.t7",
        "filename": "the_scream.t7",
    },
    "starry_night": {
        "label": "Starry Night",
        "description": "Van Gogh's Starry Night swirls",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/starry_night.t7",
        "filename": "starry_night.t7",
    },
}

def _download_model(style_name: str, on_progress: Optional[Callable] = None) -> str:
    """Download style model if not cached."""
    if style_name not in STYLE_MODELS:
        raise ValueError(f"Unknown style: {style_name}")

    info = STYLE_MODELS[style_name]
    model_path = os.path.join(MODELS_DIR, info["filename"])

    if os.path.isfile(model_path):
        return model_path

    if on_progress:
        on_progress(5, f"Downloading {info['label']} style model...")

    logger.info(f"Downloading style model: {info['url']}")
    tmp_path = model_path + ".tmp"
    try:
        # Use a custom opener with per-request timeout instead of
        # process-global socket.setdefaulttimeout (not thread-safe)
        req = urllib.request.Request(info["url"])
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
        os.replace(tmp_path, model_path)
    except Exception:
        # Remove partial download
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return model_path


# ---------------------------------------------------------------------------
# Style Transfer
# ---------------------------------------------------------------------------
def get_available_styles() -> List[Dict]:
    """Return available style transfer models."""
    return [
        {
            "name": k,
            "label": v["label"],
            "description": v["description"],
            "downloaded": os.path.isfile(os.path.join(MODELS_DIR, v["filename"])),
        }
        for k, v in STYLE_MODELS.items()
    ]


def style_transfer_video(
    input_path: str,
    style_name: str = "candy",
    output_path: Optional[str] = None,
    output_dir: str = "",
    intensity: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply neural style transfer to video.

    Args:
        style_name: Name from STYLE_MODELS (candy, mosaic, etc.).
        intensity: Style blend intensity (0.0-1.0). 1.0 = full style.
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("Failed to install opencv-python-headless. Install manually: pip install opencv-python-headless")
    import cv2
    import numpy as np

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1] or ".mp4"
        directory = output_dir or os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_style_{style_name}{ext}")

    # Download model if needed
    model_path = _download_model(style_name, on_progress)

    if on_progress:
        on_progress(10, f"Loading {STYLE_MODELS[style_name]['label']} model...")

    # Load neural style model via OpenCV DNN
    net = cv2.dnn.readNetFromTorch(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    info = get_video_info(input_path)
    tmp_dir = tempfile.mkdtemp(prefix="opencut_style_")
    frames_in = os.path.join(tmp_dir, "in")
    frames_out = os.path.join(tmp_dir, "out")
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    try:
        if on_progress:
            on_progress(12, "Extracting frames...")

        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            os.path.join(frames_in, "frame_%06d.png"),
        ])

        frame_files = sorted(Path(frames_in).glob("frame_*.png"))
        total = len(frame_files)
        if total == 0:
            raise RuntimeError("No frames extracted")

        if on_progress:
            on_progress(15, f"Styling {total} frames...")

        for i, fp in enumerate(frame_files):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Create blob and run through network
            blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),
                                         (103.939, 116.779, 123.680), swapRB=False, crop=False)
            net.setInput(blob)
            styled = net.forward()

            # Post-process output
            styled = styled.reshape(3, styled.shape[2], styled.shape[3])
            styled[0] += 103.939
            styled[1] += 116.779
            styled[2] += 123.680
            styled = styled.transpose(1, 2, 0)
            styled = np.clip(styled, 0, 255).astype(np.uint8)

            # Blend with original based on intensity
            if intensity < 1.0:
                styled = cv2.addWeighted(frame, 1.0 - intensity, styled, intensity, 0)

            cv2.imwrite(os.path.join(frames_out, fp.name), styled)

            if on_progress and i % max(1, total // 20) == 0:
                pct = 15 + int((i / total) * 75)
                on_progress(pct, f"Styling frame {i+1}/{total}...")

        if on_progress:
            on_progress(92, "Encoding output video...")

        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y",
            "-framerate", str(info["fps"]),
            "-i", os.path.join(frames_out, "frame_%06d.png"),
            "-i", input_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
            output_path,
        ])

        if on_progress:
            on_progress(100, f"Style transfer complete ({STYLE_MODELS[style_name]['label']})")
        return output_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Arbitrary Style Transfer (any reference image)
# ---------------------------------------------------------------------------
def arbitrary_style_transfer(
    input_path: str,
    style_image_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    intensity: float = 1.0,
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply arbitrary style transfer using any reference image via AdaIN
    (Adaptive Instance Normalization). No pre-trained style models needed —
    the style is extracted from the reference image at inference time.

    Args:
        style_image_path: Path to any image to use as style reference.
        intensity: Blend intensity (0.0 = original, 1.0 = full style).
    """
    if not ensure_package("cv2", "opencv-python-headless", on_progress):
        raise RuntimeError("Failed to install opencv-python-headless")

    import cv2
    import numpy as np

    if not os.path.isfile(style_image_path):
        raise FileNotFoundError(f"Style image not found: {style_image_path}")

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1] or ".mp4"
        directory = output_dir or os.path.dirname(input_path)
        style_label = os.path.splitext(os.path.basename(style_image_path))[0][:20]
        output_path = os.path.join(directory, f"{base}_style_{style_label}{ext}")

    if on_progress:
        on_progress(5, "Loading style reference...")

    # Load and resize style image
    style_img = cv2.imread(style_image_path)
    if style_img is None:
        raise RuntimeError(f"Cannot read style image: {style_image_path}")

    def _adain_transfer(content_frame, style_ref):
        """Apply Adaptive Instance Normalization color/style transfer."""
        # Convert to float LAB for perceptual color transfer
        content_lab = cv2.cvtColor(content_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        style_lab = cv2.cvtColor(style_ref, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Per-channel AdaIN: normalize content, apply style statistics
        result = np.empty_like(content_lab)
        for ch in range(3):
            c_mean, c_std = content_lab[:, :, ch].mean(), content_lab[:, :, ch].std() + 1e-6
            s_mean, s_std = style_lab[:, :, ch].mean(), style_lab[:, :, ch].std() + 1e-6
            result[:, :, ch] = (content_lab[:, :, ch] - c_mean) / c_std * s_std + s_mean

        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    info = get_video_info(input_path)
    tmp_dir = tempfile.mkdtemp(prefix="opencut_arb_style_")
    frames_in = os.path.join(tmp_dir, "in")
    frames_out = os.path.join(tmp_dir, "out")
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    try:
        if on_progress:
            on_progress(10, "Extracting frames...")

        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", input_path,
            os.path.join(frames_in, "frame_%06d.png"),
        ])

        frame_files = sorted(Path(frames_in).glob("frame_*.png"))
        total = len(frame_files)
        if total == 0:
            raise RuntimeError("No frames extracted")

        # Resize style image to match first frame dimensions for consistent stats
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is not None:
            style_resized = cv2.resize(style_img, (first_frame.shape[1], first_frame.shape[0]))
        else:
            style_resized = style_img

        if on_progress:
            on_progress(15, f"Applying style to {total} frames...")

        for i, fp in enumerate(frame_files):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue

            styled = _adain_transfer(frame, style_resized)

            if intensity < 1.0:
                styled = cv2.addWeighted(frame, 1.0 - intensity, styled, intensity, 0)

            cv2.imwrite(os.path.join(frames_out, fp.name), styled)

            if on_progress and i % max(1, total // 20) == 0:
                pct = 15 + int((i / total) * 75)
                on_progress(pct, f"Styling frame {i + 1}/{total}...")

        if on_progress:
            on_progress(92, "Encoding output video...")

        run_ffmpeg([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y",
            "-framerate", str(info["fps"]),
            "-i", os.path.join(frames_out, "frame_%06d.png"),
            "-i", input_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
            output_path,
        ])

        if on_progress:
            on_progress(100, "Arbitrary style transfer complete!")
        return output_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
