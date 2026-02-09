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
import subprocess
import sys
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

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


def _ensure_package(pkg_name: str, pip_name: str = None, on_progress: Callable = None):
    try:
        __import__(pkg_name)
    except ImportError:
        pip_name = pip_name or pkg_name
        if on_progress:
            on_progress(5, f"Installing {pip_name}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name,
             "--break-system-packages", "-q"],
            capture_output=True, timeout=600,
        )


def _run_ffmpeg(cmd: List[str], timeout: int = 3600) -> str:
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr.decode(errors='replace')[-500:]}")
    return result.stderr.decode(errors="replace")


def _get_video_info(filepath: str) -> Dict:
    import json as _json
    cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    try:
        data = _json.loads(result.stdout.decode())
        stream = data["streams"][0]
        fps_parts = stream.get("r_frame_rate", "30/1").split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
        return {
            "width": int(stream.get("width", 1920)),
            "height": int(stream.get("height", 1080)),
            "fps": fps,
        }
    except Exception:
        return {"width": 1920, "height": 1080, "fps": 30.0}


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
    urllib.request.urlretrieve(info["url"], model_path)
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
    _ensure_package("cv2", "opencv-python-headless", on_progress)
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

    info = _get_video_info(input_path)
    tmp_dir = tempfile.mkdtemp(prefix="opencut_style_")
    frames_in = os.path.join(tmp_dir, "in")
    frames_out = os.path.join(tmp_dir, "out")
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)

    try:
        if on_progress:
            on_progress(12, "Extracting frames...")

        _run_ffmpeg([
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

        _run_ffmpeg([
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
