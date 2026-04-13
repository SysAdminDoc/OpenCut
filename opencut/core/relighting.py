"""
OpenCut AI Relighting Module v0.9.0

Change lighting direction and intensity using depth-estimated normal maps:
- Depth map to surface normal estimation
- Directional / point light simulation via Lambertian shading
- Per-frame relighting with configurable position, intensity, color
- Ambient + diffuse + specular lighting model

Requires: pip install opencv-python-headless numpy
Optional: pip install torch transformers (for depth estimation)
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from opencut.helpers import ensure_package, get_video_info, run_ffmpeg

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class LightConfig:
    """Configuration for a virtual light source."""
    position: Tuple[float, float, float] = (0.0, 0.5, 1.0)
    intensity: float = 1.0
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ambient: float = 0.2
    specular: float = 0.3
    specular_power: float = 16.0


# ---------------------------------------------------------------------------
# Normal Estimation from Depth Map
# ---------------------------------------------------------------------------
def estimate_normals(depth_map) -> object:
    """
    Estimate surface normals from a grayscale depth map.

    Uses Sobel gradients to compute per-pixel normal vectors from depth.

    Args:
        depth_map: 2D numpy array (H, W) with depth values (0-255 or float).

    Returns:
        3D numpy array (H, W, 3) of unit normal vectors (x, y, z).
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("Failed to install opencv-python-headless")
    if not ensure_package("numpy", "numpy"):
        raise RuntimeError("Failed to install numpy")
    import cv2
    import numpy as np

    if depth_map.ndim != 2:
        raise ValueError("depth_map must be a 2D array")

    depth_f = depth_map.astype(np.float32)
    if depth_f.max() > 1.0:
        depth_f = depth_f / 255.0

    # Sobel gradients for surface orientation
    dzdx = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)

    h, w = depth_f.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[:, :, 0] = -dzdx
    normals[:, :, 1] = -dzdy
    normals[:, :, 2] = 1.0

    # Normalize to unit vectors
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normals = normals / norms

    return normals


# ---------------------------------------------------------------------------
# Apply Lighting to a Single Frame
# ---------------------------------------------------------------------------
def apply_lighting(
    frame,
    normals,
    light_pos: Tuple[float, float, float] = (0.0, 0.5, 1.0),
    intensity: float = 1.0,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ambient: float = 0.2,
    specular: float = 0.3,
    specular_power: float = 16.0,
) -> object:
    """
    Apply Lambertian + Phong lighting to a frame using normal map.

    Args:
        frame: BGR image (H, W, 3) uint8.
        normals: Surface normals (H, W, 3) float32.
        light_pos: Light direction vector (x, y, z).
        intensity: Light intensity multiplier (0-3).
        color: Light color as (R, G, B) floats 0-1.
        ambient: Ambient light level (0-1).
        specular: Specular highlight strength (0-1).
        specular_power: Specular shininess exponent.

    Returns:
        Relit BGR frame as uint8 numpy array.
    """
    import numpy as np

    h, w = frame.shape[:2]
    frame_f = frame.astype(np.float32) / 255.0

    # Normalize light direction
    light_dir = np.array(light_pos, dtype=np.float32)
    light_norm = np.linalg.norm(light_dir)
    if light_norm > 1e-8:
        light_dir = light_dir / light_norm

    # Diffuse (Lambertian) component
    ndotl = np.sum(normals * light_dir, axis=2)
    ndotl = np.clip(ndotl, 0.0, 1.0)

    # Specular (Phong) component
    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    reflect = 2.0 * ndotl[:, :, np.newaxis] * normals - light_dir
    rdotv = np.sum(reflect * view_dir, axis=2)
    rdotv = np.clip(rdotv, 0.0, 1.0)
    spec = specular * np.power(rdotv, specular_power)

    # Combine: ambient + diffuse + specular
    light_color = np.array([color[2], color[1], color[0]], dtype=np.float32)  # RGB -> BGR
    shading = ambient + intensity * ndotl + spec
    shading = np.clip(shading, 0.0, 3.0)
    shading_3 = shading[:, :, np.newaxis] * light_color

    result = frame_f * shading_3
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# Full Video Relighting
# ---------------------------------------------------------------------------
def relight_video(
    video_path: str,
    light_config: Optional[dict] = None,
    output_path: Optional[str] = None,
    output_dir: str = "",
    depth_source: str = "auto",
    on_progress: Optional[Callable] = None,
) -> str:
    """
    Relight an entire video with a virtual light source.

    Estimates depth per frame, converts to normals, then applies
    directional lighting with configurable parameters.

    Args:
        video_path: Input video path.
        light_config: Dict with LightConfig fields, or None for defaults.
        output_path: Output video path. Auto-generated if None.
        output_dir: Output directory.
        depth_source: "auto" (use depth estimation), or path to depth map video.
        on_progress: Progress callback(pct, msg).

    Returns:
        Path to relit output video.
    """
    if not ensure_package("cv2", "opencv-python-headless"):
        raise RuntimeError("Failed to install opencv-python-headless")
    if not ensure_package("numpy", "numpy"):
        raise RuntimeError("Failed to install numpy")
    import cv2
    import numpy as np

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cfg = LightConfig()
    if light_config:
        if "position" in light_config:
            cfg.position = tuple(light_config["position"])
        if "intensity" in light_config:
            cfg.intensity = float(light_config["intensity"])
        if "color" in light_config:
            cfg.color = tuple(light_config["color"])
        if "ambient" in light_config:
            cfg.ambient = float(light_config["ambient"])
        if "specular" in light_config:
            cfg.specular = float(light_config["specular"])
        if "specular_power" in light_config:
            cfg.specular_power = float(light_config["specular_power"])

    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        directory = output_dir or os.path.dirname(video_path)
        output_path = os.path.join(directory, f"{base}_relit.mp4")

    if on_progress:
        on_progress(5, "Setting up relighting pipeline...")

    info = get_video_info(video_path)
    w, h, fps = info["width"], info["height"], info["fps"]

    # Check for depth estimation capability
    use_model = False
    depth_pipe = None
    if depth_source == "auto":
        try:
            import torch
            from transformers import pipeline as hf_pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            depth_pipe = hf_pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=device,
            )
            use_model = True
            if on_progress:
                on_progress(10, "Depth model loaded...")
        except ImportError:
            logger.info("Depth model not available, using Sobel-based depth estimation")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    depth_cap = None
    if depth_source != "auto" and os.path.isfile(depth_source):
        depth_cap = cv2.VideoCapture(depth_source)

    _ntf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video = _ntf.name
    _ntf.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get depth map for this frame
            if depth_cap is not None:
                ret_d, depth_frame = depth_cap.read()
                if ret_d:
                    depth_gray = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY) if depth_frame.ndim == 3 else depth_frame
                else:
                    depth_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif use_model and depth_pipe is not None:
                from PIL import Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                result = depth_pipe(pil_img)
                depth_arr = np.array(result["depth"].resize((w, h)), dtype=np.float32)
                if depth_arr.max() > depth_arr.min():
                    depth_gray = ((depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min()) * 255).astype(np.uint8)
                else:
                    depth_gray = np.full((h, w), 128, dtype=np.uint8)
            else:
                # Fallback: use luminance as crude depth proxy
                depth_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            normals = estimate_normals(depth_gray)
            relit = apply_lighting(
                frame, normals,
                light_pos=cfg.position,
                intensity=cfg.intensity,
                color=cfg.color,
                ambient=cfg.ambient,
                specular=cfg.specular,
                specular_power=cfg.specular_power,
            )
            writer.write(relit)

            frame_idx += 1
            if on_progress and frame_idx % 15 == 0:
                pct = 10 + int((frame_idx / total_frames) * 80)
                on_progress(min(pct, 92), f"Relighting frame {frame_idx}/{total_frames}...")

    finally:
        cap.release()
        if depth_cap is not None:
            depth_cap.release()
        writer.release()
        if use_model:
            del depth_pipe
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    if on_progress:
        on_progress(93, "Encoding relit video...")

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

    if on_progress:
        on_progress(100, "Relighting complete!")
    return output_path
