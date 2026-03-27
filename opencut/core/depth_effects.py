"""
Video Depth Effects — Depth Anything V2 + Video Depth Anything

Uses monocular depth estimation to enable:
- Depth-of-field (bokeh) simulation on flat footage
- Parallax zoom / 3D Ken Burns effect
- Depth map export for compositing

Requires: pip install torch torchvision transformers
Models: LiheYoung/depth-anything-v2-small (or base/large)

References:
  - Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
  - Video Depth Anything: https://github.com/DepthAnything/Video-Depth-Anything
"""

import logging
import os
import subprocess
from typing import Callable, Optional

logger = logging.getLogger("opencut")


def check_depth_available() -> bool:
    """Check if Depth Anything is available (needs torch + transformers)."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def estimate_depth_map(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    model_size: str = "small",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Generate a depth map video from input footage using Depth Anything V2.

    Each frame is analyzed for monocular depth and rendered as a grayscale
    depth map (white = near, black = far).

    Args:
        input_path: Path to input video.
        output_path: Explicit output path. Auto-generated if None.
        output_dir: Output directory.
        model_size: "small" (fastest), "base", or "large" (best quality).
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to the depth map video.
    """
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError:
        raise ImportError("Depth effects require torch, numpy, and opencv. Install: pip install torch opencv-python-headless")

    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError("Depth effects require transformers. Install: pip install transformers")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = output_dir or os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_depth.mp4")

    if on_progress:
        on_progress(5, f"Loading Depth Anything V2 ({model_size})...")

    model_id = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    depth_pipe = pipeline(
        "depth-estimation",
        model=model_id,
        device=device,
    )

    try:
        if on_progress:
            on_progress(15, "Opening video...")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tmp_path = output_path + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height), isColor=False)

        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Failed to initialize video writer")

        from PIL import Image

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                result = depth_pipe(pil_img)
                depth_map = result["depth"]

                depth_np = np.array(depth_map.resize((width, height)))

                if depth_np.max() > depth_np.min():
                    depth_norm = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
                else:
                    depth_norm = np.zeros((height, width), dtype=np.uint8)

                writer.write(depth_norm)
                frame_idx += 1

                if on_progress and frame_idx % max(1, total_frames // 20) == 0:
                    pct = 15 + int((frame_idx / max(1, total_frames)) * 75)
                    on_progress(min(92, pct), f"Estimating depth {frame_idx}/{total_frames}...")

        finally:
            cap.release()
            writer.release()

        if on_progress:
            on_progress(93, "Encoding depth map...")

        _ffmpeg_encode(tmp_path, input_path, output_path, fps, has_audio=False)
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        if on_progress:
            on_progress(100, "Depth map generated!")
        return output_path
    finally:
        del depth_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def apply_bokeh_effect(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    focus_point: float = 0.5,
    blur_strength: int = 25,
    model_size: str = "small",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply depth-of-field (bokeh) simulation using depth estimation.

    Blurs areas of the frame based on estimated depth, keeping the
    focus plane sharp. Simulates a shallow depth-of-field look on
    footage shot with a deep DOF (phones, webcams, action cameras).

    Args:
        input_path: Path to input video.
        output_path: Explicit output path.
        output_dir: Output directory.
        focus_point: Depth value (0-1) to keep in focus. 0.5 = mid-distance.
        blur_strength: Maximum blur kernel size for out-of-focus areas.
        model_size: Depth model size ("small", "base", "large").
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to the output video with bokeh effect.
    """
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError:
        raise ImportError("Depth effects require torch, numpy, and opencv.")

    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError("Depth effects require transformers.")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = output_dir or os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_bokeh.mp4")

    if on_progress:
        on_progress(5, f"Loading Depth Anything V2 ({model_size})...")

    model_id = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_pipe = pipeline("depth-estimation", model=model_id, device=device)

    try:
        if on_progress:
            on_progress(15, "Processing video with bokeh effect...")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tmp_path = output_path + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Failed to initialize video writer")

        from PIL import Image

        blur_strength = max(3, blur_strength | 1)

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                result = depth_pipe(pil_img)
                depth_np = np.array(result["depth"].resize((width, height)), dtype=np.float32)

                d_min, d_max = depth_np.min(), depth_np.max()
                if d_max > d_min:
                    depth_norm = (depth_np - d_min) / (d_max - d_min)
                else:
                    depth_norm = np.full((height, width), 0.5, dtype=np.float32)

                blur_mask = np.abs(depth_norm - focus_point)
                blur_mask = np.clip(blur_mask * 2.0, 0.0, 1.0)

                blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)

                blur_mask_3 = np.stack([blur_mask] * 3, axis=-1)
                composite = (frame.astype(np.float32) * (1 - blur_mask_3) +
                             blurred.astype(np.float32) * blur_mask_3).astype(np.uint8)

                writer.write(composite)
                frame_idx += 1

                if on_progress and frame_idx % max(1, total_frames // 20) == 0:
                    pct = 15 + int((frame_idx / max(1, total_frames)) * 75)
                    on_progress(min(92, pct), f"Applying bokeh {frame_idx}/{total_frames}...")

        finally:
            cap.release()
            writer.release()

        if on_progress:
            on_progress(93, "Encoding output...")

        _ffmpeg_encode(tmp_path, input_path, output_path, fps)
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        if on_progress:
            on_progress(100, "Bokeh effect applied!")
        return output_path
    finally:
        del depth_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def apply_parallax_zoom(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "",
    zoom_amount: float = 1.15,
    duration: float = 0.0,
    model_size: str = "small",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Apply 3D parallax zoom (Ken Burns) effect using depth estimation.

    Near objects move more than far objects, creating a 3D-like push-in
    effect from a single 2D image/video.

    Args:
        input_path: Path to input video.
        output_path: Explicit output path.
        output_dir: Output directory.
        zoom_amount: Maximum zoom factor (1.0-2.0). 1.15 = subtle, 1.5 = dramatic.
        duration: If > 0, only apply to first N seconds. 0 = entire video.
        model_size: Depth model size.
        on_progress: Progress callback.

    Returns:
        Path to output video with parallax effect.
    """
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError:
        raise ImportError("Depth effects require torch, numpy, and opencv.")

    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError("Depth effects require transformers.")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        directory = output_dir or os.path.dirname(input_path)
        output_path = os.path.join(directory, f"{base}_parallax.mp4")

    zoom_amount = max(1.01, min(2.0, zoom_amount))

    if on_progress:
        on_progress(5, f"Loading Depth Anything V2 ({model_size})...")

    model_id = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_pipe = pipeline("depth-estimation", model=model_id, device=device)

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0

        if duration > 0:
            effect_frames = int(min(duration, total_duration) * fps)
        else:
            effect_frames = total_frames

        if on_progress:
            on_progress(15, "Applying 3D parallax zoom...")

        tmp_path = output_path + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Failed to initialize video writer")

        from PIL import Image

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx < effect_frames:
                    t = frame_idx / max(1, effect_frames - 1)
                    current_zoom = 1.0 + (zoom_amount - 1.0) * t

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    result = depth_pipe(pil_img)
                    depth_np = np.array(result["depth"].resize((width, height)), dtype=np.float32)

                    d_min, d_max = depth_np.min(), depth_np.max()
                    if d_max > d_min:
                        depth_norm = (depth_np - d_min) / (d_max - d_min)
                    else:
                        depth_norm = np.full((height, width), 0.5, dtype=np.float32)

                    cx, cy = width / 2.0, height / 2.0
                    y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
                    displacement_scale = depth_norm * (current_zoom - 1.0)
                    dx = (cx - x_coords) * displacement_scale
                    dy = (cy - y_coords) * displacement_scale
                    map_x = (x_coords + dx).astype(np.float32)
                    map_y = (y_coords + dy).astype(np.float32)

                    parallax_frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    writer.write(parallax_frame)
                else:
                    writer.write(frame)

                frame_idx += 1

                if on_progress and frame_idx % max(1, total_frames // 20) == 0:
                    pct = 15 + int((frame_idx / max(1, total_frames)) * 75)
                    on_progress(min(92, pct), f"Parallax {frame_idx}/{total_frames}...")

        finally:
            cap.release()
            writer.release()

        if on_progress:
            on_progress(93, "Encoding output...")

        _ffmpeg_encode(tmp_path, input_path, output_path, fps)
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        if on_progress:
            on_progress(100, "Parallax zoom applied!")
        return output_path
    finally:
        del depth_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _ffmpeg_encode(tmp_video: str, original: str, output: str, fps: float, has_audio: bool = True):
    """Re-encode with proper codec and copy audio from original."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", tmp_video,
    ]
    if has_audio:
        cmd += ["-i", original, "-map", "0:v", "-map", "1:a?"]
    else:
        cmd += ["-map", "0:v"]

    cmd += [
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
    ]
    if has_audio:
        cmd += ["-c:a", "copy"]
    cmd += ["-shortest", output]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        stderr = result.stderr[-300:] if result.stderr else "unknown"
        raise RuntimeError(f"FFmpeg encoding failed: {stderr}")
